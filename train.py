"""
RF-DETR: detection transformer with ResNet-50 backbone for COCO object detection.
Single-GPU, 5-minute time budget.

Usage: uv run train.py
"""

import gc
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from scipy.optimize import linear_sum_assignment

from prepare import TIME_BUDGET, NUM_CLASSES, MAX_QUERIES, DATA_DIR, make_dataloader, evaluate_l1

# ---------------------------------------------------------------------------
# Hyperparameters (modify these)
# ---------------------------------------------------------------------------

# Model
D_MODEL     = 256       # transformer hidden dimension
N_HEAD      = 8         # attention heads
N_ENC       = 3         # number of encoder layers
N_DEC       = 3         # number of decoder layers
NUM_QUERIES = MAX_QUERIES   # object queries per image
PRETRAINED  = True      # use ImageNet-pretrained ResNet-50 backbone

# Training
BATCH_SIZE      = 4     # images per step
LR_BACKBONE     = 1e-5  # lower LR for pretrained backbone
LR_TRANSFORMER  = 1e-4  # LR for transformer and heads
WEIGHT_DECAY    = 1e-4
WARMUP_RATIO    = 0.05  # fraction of time budget used for linear warmup
LAMBDA_L1       = 5.0   # weight for L1 bounding-box loss
LAMBDA_CE       = 1.0   # weight for classification loss
BG_WEIGHT       = 0.1   # relative weight assigned to the background class

# ---------------------------------------------------------------------------
# RF-DETR model
# ---------------------------------------------------------------------------

class BackboneResNet50(nn.Module):
    """ResNet-50 backbone; returns the layer-4 feature map (stride 32)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        resnet  = tvm.resnet50(weights=weights)
        self.body = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)     # (B, 2048, H/32, W/32)


class SinCos2DPositionEmbedding(nn.Module):
    """2-D sinusoidal position embedding (row + column), channel-first."""

    def __init__(self, d_model: int, temperature: float = 10_000.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.d_model     = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  pos: (1, d_model, H, W)"""
        _, _, H, W = x.shape
        device = x.device
        d2     = self.d_model // 2       # half for rows, half for cols
        dim_t  = torch.arange(d2 // 2, device=device, dtype=torch.float32)
        dim_t  = self.temperature ** (2 * dim_t / d2)

        row_pos = torch.arange(H, device=device, dtype=torch.float32)[:, None] / dim_t  # (H, d2//2)
        col_pos = torch.arange(W, device=device, dtype=torch.float32)[:, None] / dim_t  # (W, d2//2)

        row_pe  = torch.stack([row_pos.sin(), row_pos.cos()], dim=-1).flatten(1)  # (H, d2)
        col_pe  = torch.stack([col_pos.sin(), col_pos.cos()], dim=-1).flatten(1)  # (W, d2)

        pe = torch.cat(
            [row_pe[:, None, :].expand(H, W, d2),
             col_pe[None, :, :].expand(H, W, d2)],
            dim=-1,
        )  # (H, W, d_model)

        return pe.permute(2, 0, 1).unsqueeze(0)   # (1, d_model, H, W)


class RFDETR(nn.Module):
    """
    RF-DETR: ResNet-50 backbone with a DETR-style transformer encoder-decoder.

    forward(images) → (pred_boxes, pred_logits)
      pred_boxes:   (B, Q, 4)        normalised [cx, cy, w, h]  ∈ [0, 1]
      pred_logits:  (B, Q, C+1)      foreground classes + background
    """

    def __init__(
        self,
        num_classes:    int   = NUM_CLASSES,
        num_queries:    int   = NUM_QUERIES,
        d_model:        int   = D_MODEL,
        nhead:          int   = N_HEAD,
        num_enc_layers: int   = N_ENC,
        num_dec_layers: int   = N_DEC,
        pretrained:     bool  = PRETRAINED,
    ):
        super().__init__()
        self.num_queries = num_queries

        self.backbone   = BackboneResNet50(pretrained=pretrained)
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d_model, kernel_size=1)
        self.pos_embed  = SinCos2DPositionEmbedding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        feat = self.backbone(x)                            # (B, 2048, H, W)
        feat = self.input_proj(feat)                       # (B, d_model, H, W)
        pos  = self.pos_embed(feat)                        # (1, d_model, H, W)

        seq  = (feat + pos).flatten(2).permute(0, 2, 1)   # (B, H*W, d_model)
        mem  = self.encoder(seq)                           # (B, H*W, d_model)

        q    = self.query_embed.weight[None].expand(B, -1, -1)   # (B, Q, d_model)
        hs   = self.decoder(q, mem)                               # (B, Q, d_model)

        pred_logits = self.class_head(hs)          # (B, Q, C+1)
        pred_boxes  = self.bbox_head(hs).sigmoid() # (B, Q, 4)

        return pred_boxes, pred_logits


# ---------------------------------------------------------------------------
# Loss: Hungarian matching + L1 + cross-entropy
# ---------------------------------------------------------------------------

def hungarian_match(pred_boxes, pred_logits, gt_boxes_list, gt_labels_list):
    """
    Batch Hungarian matching (runs on CPU via scipy).
    Returns a list of (pred_idx, gt_idx) LongTensor pairs, one per image.
    """
    indices = []
    for i, (gt_b, gt_l) in enumerate(zip(gt_boxes_list, gt_labels_list)):
        num_gt = gt_b.shape[0]
        if num_gt == 0:
            empty = torch.zeros(0, dtype=torch.long)
            indices.append((empty, empty))
            continue

        with torch.no_grad():
            pb = pred_boxes[i].detach().float()    # (Q, 4)
            pl = pred_logits[i].detach().float()   # (Q, C+1)
            gb = gt_b.to(pb.device).float()        # (M, 4)
            gl = gt_l.to(pl.device)                # (M,)

            cost_l1    = (pb[:, None] - gb[None]).abs().sum(-1)   # (Q, M)
            cost_class = -(pl.softmax(-1)[:, gl])                 # (Q, M)
            cost       = LAMBDA_L1 * cost_l1 + LAMBDA_CE * cost_class

        ri, ci = linear_sum_assignment(cost.cpu().numpy())
        indices.append((
            torch.tensor(ri, dtype=torch.long),
            torch.tensor(ci, dtype=torch.long),
        ))

    return indices


def compute_loss(pred_boxes, pred_logits, gt_boxes_list, gt_labels_list, indices):
    """DETR set-prediction loss (classification + L1 on matched pairs)."""
    B, Q, _ = pred_boxes.shape
    device  = pred_boxes.device
    nc      = pred_logits.shape[-1] - 1   # number of foreground classes

    # Default all queries to background; overwrite matched ones
    tgt_cls = torch.full((B, Q), nc, dtype=torch.long, device=device)
    for i, (ri, ci) in enumerate(indices):
        if ri.numel():
            tgt_cls[i, ri] = gt_labels_list[i][ci].to(device)

    bg_weight      = torch.ones(nc + 1, device=device)
    bg_weight[-1]  = BG_WEIGHT
    loss_ce = F.cross_entropy(
        pred_logits.reshape(-1, nc + 1),
        tgt_cls.reshape(-1),
        weight=bg_weight,
    )

    # L1 loss on matched pairs only
    l1_terms, num_matched = [], 0
    for i, (ri, ci) in enumerate(indices):
        if ri.numel():
            l1_terms.append(
                F.l1_loss(
                    pred_boxes[i][ri.to(device)],
                    gt_boxes_list[i][ci].to(device),
                    reduction="sum",
                )
            )
            num_matched += ri.numel()

    if l1_terms:
        loss_l1 = sum(l1_terms) / max(num_matched, 1)
    else:
        loss_l1 = pred_boxes.sum() * 0.0   # zero gradient path

    return LAMBDA_CE * loss_ce + LAMBDA_L1 * loss_l1


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

model = RFDETR().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.1f}M")

# Separate backbone (lower LR, already pretrained) from the rest
backbone_ids    = {id(p) for p in model.backbone.parameters()}
backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE,    "initial_lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_TRANSFORMER, "initial_lr": LR_TRANSFORMER},
    ],
    weight_decay=WEIGHT_DECAY,
)

# Use "val" split for training since prepare.py downloads val2017 by default.
# Run `python prepare.py --train` first to use the full train2017 split.
_train_split = "train" if os.path.isdir(os.path.join(DATA_DIR, "train2017")) else "val"
train_loader = make_dataloader(_train_split, BATCH_SIZE, num_workers=4)
train_iter   = iter(train_loader)

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

print(f"Time budget:       {TIME_BUDGET}s")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Training split:    {_train_split}  ({len(train_loader.dataset)} images)")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress: float) -> float:
    """Linear warmup then cosine decay over the training time budget."""
    if progress < WARMUP_RATIO:
        return (progress / WARMUP_RATIO) if WARMUP_RATIO > 0 else 1.0
    cos_progress = (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-8)
    return 0.5 * (1.0 + math.cos(math.pi * cos_progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training  = time.time()
total_training_time = 0.0
smooth_loss = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    try:
        images, boxes_list, labels_list = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        images, boxes_list, labels_list = next(train_iter)

    images = images.to(device)

    with autocast_ctx:
        pred_boxes, pred_logits = model(images)

    indices = hungarian_match(pred_boxes, pred_logits, boxes_list, labels_list)
    loss    = compute_loss(pred_boxes, pred_logits, boxes_list, labels_list, indices)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    loss_val = loss.detach().item()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Update LR
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * lrm

    # EMA loss for display
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
    debiased    = smooth_loss / (1 - ema ** (step + 1))

    remaining = max(0.0, TIME_BUDGET - total_training_time)
    print(
        f"\rstep {step:05d} ({100 * progress:.1f}%) | "
        f"loss: {debiased:.4f} | lrm: {lrm:.3f} | "
        f"dt: {dt * 1000:.0f}ms | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    # GC management: freeze after first step to avoid random stalls
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()   # newline after the \r log

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
with autocast_ctx:
    val_l1 = evaluate_l1(model, batch_size=BATCH_SIZE, device=str(device))

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_l1:           {val_l1:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
