"""
One-time data preparation for RF-DETR autoresearch experiments.
Downloads COCO 2017 val data for object detection / segmentation experiments.

Usage:
    python prepare.py

Data is stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import json
import zipfile
import math
import argparse
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMG_SIZE = 640          # input image size (square)
TIME_BUDGET = 300       # training time budget in seconds (5 minutes)
EVAL_IMAGES = 1000      # number of validation images for eval (fixed)
NUM_CLASSES = 80        # COCO 80-category object detection
NUM_QUERIES = 100       # number of object queries in the DETR decoder
MAX_OBJECTS = 100       # maximum objects per image (padded to this length)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "val2017")
ANN_DIR = os.path.join(DATA_DIR, "annotations")
ANN_FILE = os.path.join(ANN_DIR, "instances_val2017.json")
SPLIT_FILE = os.path.join(DATA_DIR, "split.json")

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

NUM_TRAIN = 4000  # images reserved for training (rest = validation)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_file(url, dest_path, description=""):
    """Download a file with progress reporting. Returns True on success."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {description or Path(dest_path).name}")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    print(f"  Downloading {description or url} ...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        elapsed = time.time() - t0
                        speed = downloaded / elapsed / 1024 ** 2 if elapsed > 0 else 0
                        pct = 100 * downloaded / total
                        print(
                            f"\r    {pct:.1f}%  "
                            f"({downloaded / 1024**2:.0f}/{total / 1024**2:.0f} MB  "
                            f"{speed:.1f} MB/s)",
                            end="",
                            flush=True,
                        )
        print()
        os.rename(temp_path, dest_path)
        return True
    except (requests.RequestException, IOError) as exc:
        print(f"\n  Download failed: {exc}")
        for p in [temp_path, dest_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return False


def _extract_zip(zip_path, dest_dir, description=""):
    """Extract a zip archive."""
    print(f"  Extracting {description or Path(zip_path).name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}")


def download_data():
    """Download COCO 2017 val images and annotations."""
    os.makedirs(DATA_DIR, exist_ok=True)

    images_ready = os.path.exists(IMAGES_DIR) and len(list(Path(IMAGES_DIR).glob("*.jpg"))) > 100
    ann_ready = os.path.exists(ANN_FILE)

    if images_ready and ann_ready:
        n = len(list(Path(IMAGES_DIR).glob("*.jpg")))
        print(f"Data: {n} COCO val2017 images + annotations already at {DATA_DIR}")
        _make_split()
        return

    print("Data: downloading COCO 2017 val ...")

    # Annotations (~242 MB zipped)
    ann_zip = os.path.join(DATA_DIR, "annotations_trainval2017.zip")
    if not _download_file(COCO_ANN_URL, ann_zip, "annotations_trainval2017.zip (~242 MB)"):
        print("ERROR: failed to download annotations. Check network connectivity.")
        sys.exit(1)
    if not ann_ready:
        _extract_zip(ann_zip, DATA_DIR, "annotations")

    # Images (~778 MB zipped)
    img_zip = os.path.join(DATA_DIR, "val2017.zip")
    if not _download_file(COCO_IMAGES_URL, img_zip, "val2017.zip (~778 MB)"):
        print("ERROR: failed to download val2017 images. Check network connectivity.")
        sys.exit(1)
    if not images_ready:
        _extract_zip(img_zip, DATA_DIR, "val2017 images")

    n = len(list(Path(IMAGES_DIR).glob("*.jpg")))
    print(f"Data: {n} images ready at {IMAGES_DIR}")
    _make_split()


def _make_split():
    """Create and persist a deterministic train/val split of image IDs."""
    if os.path.exists(SPLIT_FILE):
        return
    with open(ANN_FILE) as f:
        data = json.load(f)
    all_ids = sorted(img["id"] for img in data["images"])
    # Deterministic split: first NUM_TRAIN for training, rest for validation
    train_ids = all_ids[:NUM_TRAIN]
    val_ids = all_ids[NUM_TRAIN : NUM_TRAIN + EVAL_IMAGES]
    with open(SPLIT_FILE, "w") as f:
        json.dump({"train": train_ids, "val": val_ids}, f)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val images")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ImageNet normalisation (standard for pretrained backbones)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class COCODetection(Dataset):
    """
    COCO detection dataset.  Returns image tensors and bounding-box targets.
    Boxes are stored as [cx, cy, w, h] normalised to [0, 1].
    """

    def __init__(self, ann_file, images_dir, img_ids, augment=False):
        with open(ann_file) as f:
            data = json.load(f)

        # category id → 0-indexed label
        cats = sorted(data["categories"], key=lambda c: c["id"])
        self._cat_to_label = {c["id"]: i for i, c in enumerate(cats)}

        id_to_meta = {img["id"]: img for img in data["images"]}

        # group annotations by image id
        ann_by_img: dict[int, list] = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            ann_by_img.setdefault(iid, []).append(ann)

        self._samples = []
        for iid in img_ids:
            if iid not in id_to_meta:
                continue
            meta = id_to_meta[iid]
            self._samples.append(
                dict(
                    file=os.path.join(images_dir, meta["file_name"]),
                    width=meta["width"],
                    height=meta["height"],
                    anns=ann_by_img.get(iid, []),
                )
            )

        self._transform = _TRAIN_TRANSFORM if augment else _EVAL_TRANSFORM
        self._augment = augment

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        img = Image.open(s["file"]).convert("RGB")
        W, H = img.size
        img_t = self._transform(img)

        # Optional horizontal flip
        flip = self._augment and torch.rand(1).item() > 0.5
        if flip:
            img_t = torch.flip(img_t, dims=[2])

        boxes, labels = [], []
        for ann in s["anns"]:
            if ann.get("iscrowd", 0):
                continue
            cat_id = ann["category_id"]
            if cat_id not in self._cat_to_label:
                continue
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            if flip:
                cx = 1.0 - cx
            boxes.append([cx, cy, nw, nh])
            labels.append(self._cat_to_label[cat_id])
            if len(boxes) >= MAX_OBJECTS:
                break

        n = len(boxes)
        boxes_t = torch.zeros(MAX_OBJECTS, 4, dtype=torch.float32)
        labels_t = torch.full((MAX_OBJECTS,), -1, dtype=torch.long)
        if n > 0:
            boxes_t[:n] = torch.tensor(boxes, dtype=torch.float32)
            labels_t[:n] = torch.tensor(labels, dtype=torch.long)
        return img_t, boxes_t, labels_t, torch.tensor(n, dtype=torch.long)


def _collate(batch):
    images, boxes, labels, counts = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(boxes),
        torch.stack(labels),
        torch.stack(counts),
    )


def make_dataloader(split, batch_size, num_workers=4, shuffle=None):
    """
    Build a DataLoader for the COCO detection dataset.

    Args:
        split: "train" or "val"
        batch_size: images per batch
        num_workers: DataLoader worker processes
        shuffle: override shuffle default (True for train, False for val)
    """
    assert split in ("train", "val"), f"Unknown split: {split!r}"
    assert os.path.exists(SPLIT_FILE), (
        f"Split file not found at {SPLIT_FILE}. Run prepare.py first."
    )
    with open(SPLIT_FILE) as f:
        split_data = json.load(f)
    img_ids = split_data[split]

    dataset = COCODetection(
        ann_file=ANN_FILE,
        images_dir=IMAGES_DIR,
        img_ids=img_ids,
        augment=(split == "train"),
    )
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=(split == "train"),
    )

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def _box_cxcywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _generalized_box_iou(boxes1, boxes2):
    """
    Generalised IoU between two sets of boxes (xyxy format).
    Returns a [N, M] matrix.
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(min=1e-6)

    giou = iou - (enc_area - union_area) / enc_area
    return giou


@torch.no_grad()
def evaluate_l1(model, batch_size):
    """
    Average per-box L1 loss over the fixed validation set.
    Uses Hungarian matching to pair each prediction to the nearest ground truth.
    Lower is better.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("scipy is required for evaluate_l1. Run: pip install scipy") from exc

    val_loader = make_dataloader("val", batch_size, num_workers=2, shuffle=False)

    was_training = model.training
    model.eval()
    total_l1 = 0.0
    total_boxes = 0

    for images, boxes, labels, counts in val_loader:
        images = images.cuda()
        boxes = boxes.cuda()
        labels = labels.cuda()

        outputs = model(images)
        pred_boxes = outputs["pred_boxes"]    # (B, Q, 4)
        pred_logits = outputs["pred_logits"]  # (B, Q, num_classes+1)

        B = images.shape[0]
        for i in range(B):
            n = counts[i].item()
            if n == 0:
                continue
            tgt_boxes = boxes[i, :n]    # (n, 4)
            tgt_labels = labels[i, :n]  # (n,)
            pb = pred_boxes[i]           # (Q, 4)
            pl = pred_logits[i]          # (Q, num_classes+1)

            # Cost matrix
            prob = pl.softmax(-1)
            cls_cost = -prob[:, tgt_labels]                    # (Q, n)
            l1_cost = torch.cdist(pb, tgt_boxes, p=1)         # (Q, n)
            giou_cost = -_generalized_box_iou(
                _box_cxcywh_to_xyxy(pb),
                _box_cxcywh_to_xyxy(tgt_boxes),
            )                                                  # (Q, n)
            cost = 2.0 * cls_cost + 5.0 * l1_cost + 2.0 * giou_cost

            pred_idx, tgt_idx = linear_sum_assignment(cost.cpu().float().numpy())

            if len(pred_idx) > 0:
                matched_pred = pb[torch.tensor(pred_idx, device=pb.device)]
                matched_tgt = tgt_boxes[torch.tensor(tgt_idx, device=tgt_boxes.device)]
                total_l1 += F.l1_loss(matched_pred, matched_tgt, reduction="sum").item()
                total_boxes += len(pred_idx)

    if was_training:
        model.train()

    return total_l1 / max(total_boxes, 1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare COCO data for RF-DETR autoresearch")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_data()

    print()
    print("Done! Ready to train.")

