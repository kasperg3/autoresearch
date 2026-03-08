"""
RF-DETR training script for COCO object detection. Single-GPU, single-file.
Based on the DETR architecture with a ResNet-50 backbone.

Usage: uv run train.py
"""

import os
import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from scipy.optimize import linear_sum_assignment

from prepare import IMG_SIZE, TIME_BUDGET, NUM_CLASSES, NUM_QUERIES, make_dataloader, evaluate_l1

# ---------------------------------------------------------------------------
# RF-DETR Model
# ---------------------------------------------------------------------------

@dataclass
class RFDETRConfig:
    num_classes: int = NUM_CLASSES
    num_queries: int = NUM_QUERIES
    d_model: int = 256
    n_head: int = 8
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    dim_ffn: int = 2048
    dropout: float = 0.1


class PositionEmbeddingSine(nn.Module):
    """Fixed 2-D sinusoidal position encoding for grid feature maps."""

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, H, 1).expand(B, H, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=x.device).view(1, 1, W).expand(B, H, W)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * (2 * math.pi)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * (2 * math.pi)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(3)
        return torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (B, d_model, H, W)


class RFDETR(nn.Module):
    """
    RF-DETR: Detection Transformer with ResNet-50 backbone and multi-scale features.
    Outputs per-query class logits and bounding boxes (cx, cy, w, h) in [0, 1].
    """

    def __init__(self, config: RFDETRConfig, pretrained_backbone: bool = True):
        super().__init__()
        d = config.d_model

        # --- Backbone (ResNet-50) ---
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        # Use C3, C4, C5 feature maps (no global avg pool / fc)
        self.bb_stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1,
        )
        self.bb_layer2 = backbone.layer2   # C3: 512 ch
        self.bb_layer3 = backbone.layer3   # C4: 1024 ch
        self.bb_layer4 = backbone.layer4   # C5: 2048 ch

        # --- Multi-scale input projections → d_model ---
        self.proj_c3 = nn.Conv2d(512, d, kernel_size=1)
        self.proj_c4 = nn.Conv2d(1024, d, kernel_size=1)
        self.proj_c5 = nn.Conv2d(2048, d, kernel_size=1)

        # --- 2D positional encoding ---
        self.pos_enc = PositionEmbeddingSine(d // 2)

        # --- Transformer encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_head,
            dim_feedforward=config.dim_ffn,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_enc_layers)

        # --- Transformer decoder ---
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.n_head,
            dim_feedforward=config.dim_ffn,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.n_dec_layers)

        # --- Learnable object queries ---
        self.query_embed = nn.Embedding(config.num_queries, d)

        # --- Prediction heads ---
        # class head: num_classes + 1 (last = no-object)
        self.class_head = nn.Linear(d, config.num_classes + 1)
        # box head: 3-layer MLP → sigmoid → (cx, cy, w, h) ∈ [0, 1]
        self.box_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.proj_c3, self.proj_c4, self.proj_c5]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.box_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            dict with 'pred_logits' (B, Q, num_classes+1)
                  and 'pred_boxes'  (B, Q, 4)  in [0,1] cx/cy/w/h
        """
        # Backbone
        x = self.bb_stem(images)
        c3 = self.bb_layer2(x)    # (B, 512,  H/8,  W/8)
        c4 = self.bb_layer3(c3)   # (B, 1024, H/16, W/16)
        c5 = self.bb_layer4(c4)   # (B, 2048, H/32, W/32)

        # Project to d_model and add position encoding, then flatten to tokens
        tokens = []
        for feat, proj in [(c3, self.proj_c3), (c4, self.proj_c4), (c5, self.proj_c5)]:
            f = proj(feat)                                 # (B, d, h, w)
            f = f + self.pos_enc(f)                        # add pos enc
            tokens.append(f.flatten(2).permute(0, 2, 1))  # (B, h*w, d)
        src = torch.cat(tokens, dim=1)                     # (B, N_tok, d)

        # Transformer encoder
        memory = self.encoder(src)                         # (B, N_tok, d)

        # Transformer decoder with learnable object queries
        B = images.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d)
        hs = self.decoder(queries, memory)                 # (B, Q, d)

        # Predictions
        pred_logits = self.class_head(hs)                  # (B, Q, C+1)
        pred_boxes = self.box_head(hs).sigmoid()           # (B, Q, 4)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    """(cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """Pairwise GIoU between two sets of boxes (xyxy), returns (N, M) matrix."""
    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = (a1[:, None] + a2[None, :] - inter).clamp(min=1e-6)
    iou = inter / union

    ex1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    ey1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    ex2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    ey2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc = ((ex2 - ex1) * (ey2 - ey1)).clamp(min=1e-6)

    return iou - (enc - union) / enc

# ---------------------------------------------------------------------------
# Hungarian matching + loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match(pred_logits, pred_boxes, tgt_boxes, tgt_labels, n):
    """
    Compute optimal bipartite assignment for one image.

    Args:
        pred_logits: (Q, C+1)
        pred_boxes:  (Q, 4)
        tgt_boxes:   (n, 4)   ground-truth boxes
        tgt_labels:  (n,)     ground-truth class indices
        n:           actual number of ground-truth objects
    Returns:
        (pred_indices, tgt_indices) — matched index tensors on same device
    """
    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pred_logits.device)
        return empty, empty

    prob = pred_logits.softmax(-1)            # (Q, C+1)
    cls_cost = -prob[:, tgt_labels]           # (Q, n)
    l1_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)             # (Q, n)
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(tgt_boxes),
    )                                         # (Q, n)

    cost = 2.0 * cls_cost + 5.0 * l1_cost + 2.0 * giou_cost
    pred_idx, tgt_idx = linear_sum_assignment(cost.cpu().float().numpy())
    dev = pred_logits.device
    return (torch.tensor(pred_idx, dtype=torch.long, device=dev),
            torch.tensor(tgt_idx, dtype=torch.long, device=dev))


def compute_loss(outputs, boxes, labels, counts):
    """
    Compute total detection loss for a batch.

    Returns:
        loss (scalar), l1_per_box (float for logging)
    """
    pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
    pred_boxes = outputs["pred_boxes"]    # (B, Q, 4)
    B, Q, _ = pred_logits.shape
    no_obj = NUM_CLASSES  # index of the "no-object" class

    total_cls = pred_logits.new_tensor(0.0)
    total_l1 = pred_logits.new_tensor(0.0)
    total_giou = pred_logits.new_tensor(0.0)
    num_matched = 0

    # Down-weight the no-object class in cross-entropy (DETR default)
    cls_weight = torch.ones(NUM_CLASSES + 1, device=pred_logits.device)
    cls_weight[no_obj] = 0.1

    for i in range(B):
        n = counts[i].item()
        tgt_boxes_i = boxes[i, :n]    # (n, 4)
        tgt_labels_i = labels[i, :n]  # (n,)

        pred_idx, tgt_idx = hungarian_match(
            pred_logits[i], pred_boxes[i], tgt_boxes_i, tgt_labels_i, n
        )

        # Classification target: no-object for all, override matched
        tgt_cls = torch.full((Q,), no_obj, dtype=torch.long, device=pred_logits.device)
        if len(pred_idx) > 0:
            tgt_cls[pred_idx] = tgt_labels_i[tgt_idx]

        total_cls = total_cls + F.cross_entropy(pred_logits[i], tgt_cls, weight=cls_weight)

        if len(pred_idx) > 0:
            mp = pred_boxes[i][pred_idx]       # matched predicted boxes
            mt = tgt_boxes_i[tgt_idx]          # matched target boxes
            total_l1 = total_l1 + F.l1_loss(mp, mt, reduction="sum")
            total_giou = total_giou + (
                1 - generalized_box_iou(
                    box_cxcywh_to_xyxy(mp),
                    box_cxcywh_to_xyxy(mt),
                ).diag()
            ).sum()
            num_matched += len(pred_idx)

    denom = max(num_matched, 1)
    loss = total_cls / B + 5.0 * total_l1 / denom + 2.0 * total_giou / denom
    l1_per_box = total_l1.item() / denom
    return loss, l1_per_box

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — this is what the agent modifies)
# ---------------------------------------------------------------------------

# Model
D_MODEL = 256           # transformer hidden dimension
N_ENC_LAYERS = 6        # transformer encoder depth
N_DEC_LAYERS = 6        # transformer decoder depth
N_HEAD = 8              # attention heads
DIM_FFN = 2048          # feedforward dimension
NUM_OBJ_QUERIES = NUM_QUERIES  # number of object queries

# Optimization
BATCH_SIZE = 4          # images per gradient step (reduce if OOM)
LEARNING_RATE = 1e-4    # base learning rate (AdamW)
BACKBONE_LR_SCALE = 0.1 # backbone LR relative to transformer LR
WEIGHT_DECAY = 1e-4     # AdamW weight decay
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.40   # fraction of time budget for cosine cool-down
FINAL_LR_FRAC = 0.01    # final LR as fraction of peak LR
CLIP_GRAD_NORM = 0.1    # gradient clipping max norm

# Backbone
PRETRAINED_BACKBONE = True  # use ImageNet-pretrained ResNet-50 weights

# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

config = RFDETRConfig(
    num_classes=NUM_CLASSES,
    num_queries=NUM_OBJ_QUERIES,
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_enc_layers=N_ENC_LAYERS,
    n_dec_layers=N_DEC_LAYERS,
    dim_ffn=DIM_FFN,
)
print(f"Model config: {asdict(config)}")

model = RFDETR(config, pretrained_backbone=PRETRAINED_BACKBONE).to(device)
num_params = model.num_params()
print(f"Parameters: {num_params / 1e6:.1f}M")

# Separate backbone and transformer parameters for different LRs
backbone_params = (
    list(model.bb_stem.parameters())
    + list(model.bb_layer2.parameters())
    + list(model.bb_layer3.parameters())
    + list(model.bb_layer4.parameters())
)
transformer_params = [p for p in model.parameters() if not any(p is q for q in backbone_params)]

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": LEARNING_RATE * BACKBONE_LR_SCALE},
        {"params": transformer_params, "lr": LEARNING_RATE},
    ],
    weight_decay=WEIGHT_DECAY,
)

train_loader = iter(make_dataloader("train", BATCH_SIZE, num_workers=4))

print(f"Time budget: {TIME_BUDGET}s  |  batch size: {BATCH_SIZE}")


def get_lr_multiplier(progress):
    """Warmup → flat → cosine warmdown schedule (all based on training progress ∈ [0,1])."""
    if WARMUP_RATIO > 0 and progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    cooldown_start = 1.0 - WARMDOWN_RATIO
    if progress < cooldown_start:
        return 1.0
    frac = (1.0 - progress) / WARMDOWN_RATIO  # 1 → 0
    return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * frac

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
smooth_loss = 0.0
smooth_l1 = 0.0
step = 0

model.train()

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    try:
        images, boxes_t, labels_t, counts = next(train_loader)
    except StopIteration:
        train_loader = iter(make_dataloader("train", BATCH_SIZE, num_workers=4))
        images, boxes_t, labels_t, counts = next(train_loader)

    images = images.to(device)
    boxes_t = boxes_t.to(device)
    labels_t = labels_t.to(device)

    outputs = model(images)
    loss, l1_val = compute_loss(outputs, boxes_t, labels_t, counts)

    optimizer.zero_grad()
    loss.backward()
    if CLIP_GRAD_NORM > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Only start counting time after the first step (compilation warmup)
    if step > 0:
        total_training_time += dt

    # LR schedule based on training progress
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    optimizer.param_groups[0]["lr"] = LEARNING_RATE * BACKBONE_LR_SCALE * lrm
    optimizer.param_groups[1]["lr"] = LEARNING_RATE * lrm

    loss_val = loss.item()
    if not math.isfinite(loss_val):
        print("FAIL: non-finite loss")
        raise SystemExit(1)

    # EMA smoothing for cleaner log display
    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
    smooth_l1 = ema * smooth_l1 + (1 - ema) * l1_val
    debiased_loss = smooth_loss / (1 - ema ** (step + 1))
    debiased_l1 = smooth_l1 / (1 - ema ** (step + 1))
    pct = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:04d} ({pct:.1f}%) | "
        f"loss: {debiased_loss:.4f} | l1: {debiased_l1:.4f} | "
        f"lrm: {lrm:.3f} | dt: {dt*1000:.0f}ms | "
        f"remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    step += 1

    if step > 0 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r log

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
val_l1 = evaluate_l1(model, BATCH_SIZE)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_l1:           {val_l1:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"d_model:          {D_MODEL}")
print(f"n_enc_layers:     {N_ENC_LAYERS}")
print(f"n_dec_layers:     {N_DEC_LAYERS}")
