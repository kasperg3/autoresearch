"""
One-time data preparation for autoresearch CV experiments.
Downloads COCO 2017 validation dataset and provides dataloader + evaluation.

Usage:
    python prepare.py                  # download val2017 + annotations
    python prepare.py --train          # also download train2017 (~18 GB)

Data is stored in ~/.cache/autoresearch/coco/.
"""

import os
import json
import zipfile
import argparse

import requests
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300    # training time budget in seconds (5 minutes)
IMG_SIZE    = 640    # input image size (square resize)
NUM_CLASSES = 80     # COCO has 80 named object categories, mapped to labels [1, 80]
MAX_QUERIES = 100    # default number of object queries for DETR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR  = os.path.join(CACHE_DIR, "coco")

COCO_VAL_URL   = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def _download_file(url, dest_path):
    """Download url to dest_path with progress reporting."""
    if os.path.exists(dest_path):
        return
    print(f"Downloading {os.path.basename(dest_path)} ...")
    temp_path = dest_path + ".tmp"
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done  = 0
        with open(temp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(
                            f"\r  {100 * done / total:.1f}%  "
                            f"({done >> 20} / {total >> 20} MB)",
                            end="", flush=True,
                        )
        print()
        os.rename(temp_path, dest_path)
    except Exception as exc:
        for p in (temp_path, dest_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def _extract_zip(zip_path, dest_dir):
    """Extract a zip archive into dest_dir."""
    print(f"Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_coco(include_train=False):
    """Download and extract COCO 2017 images and annotations."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Annotations (required for both splits)
    ann_zip = os.path.join(DATA_DIR, "annotations_trainval2017.zip")
    ann_dir = os.path.join(DATA_DIR, "annotations")
    if not os.path.isdir(ann_dir):
        _download_file(COCO_ANN_URL, ann_zip)
        _extract_zip(ann_zip, DATA_DIR)
    else:
        print("Annotations: already present")

    # Validation images (required)
    val_zip = os.path.join(DATA_DIR, "val2017.zip")
    val_dir = os.path.join(DATA_DIR, "val2017")
    if not os.path.isdir(val_dir):
        _download_file(COCO_VAL_URL, val_zip)
        _extract_zip(val_zip, DATA_DIR)
    else:
        print("val2017: already present")

    # Training images (optional, ~18 GB)
    if include_train:
        train_zip = os.path.join(DATA_DIR, "train2017.zip")
        train_dir = os.path.join(DATA_DIR, "train2017")
        if not os.path.isdir(train_dir):
            _download_file(COCO_TRAIN_URL, train_zip)
            _extract_zip(train_zip, DATA_DIR)
        else:
            print("train2017: already present")

    print(f"Data ready at {DATA_DIR}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCODetection(Dataset):
    """
    COCO object detection dataset.

    Returns (image, boxes, labels) where:
      image:  (3, IMG_SIZE, IMG_SIZE) float32, ImageNet-normalised
      boxes:  (M, 4) float32, normalised [cx, cy, w, h] in [0, 1]
      labels: (M,)   int64, 1-indexed labels in [1, 80]
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: str = "val", img_size: int = IMG_SIZE):
        assert split in ("train", "val"), f"Unknown split: {split!r}"
        self.split    = split
        self.img_size = img_size
        self.img_dir  = os.path.join(DATA_DIR, f"{split}2017")

        ann_path = os.path.join(
            DATA_DIR, "annotations", f"instances_{split}2017.json"
        )
        with open(ann_path) as f:
            data = json.load(f)

        self._id_to_file: dict = {
            img["id"]: img["file_name"] for img in data["images"]
        }
        self.image_ids = sorted(self._id_to_file.keys())

        # Group annotations by image id
        self._annotations: dict = {}
        for ann in data["annotations"]:
            self._annotations.setdefault(ann["image_id"], []).append(ann)

        # Map COCO category ids -> contiguous 1-indexed labels
        cats = sorted(c["id"] for c in data["categories"])
        self._cat_to_idx: dict = {cid: i + 1 for i, cid in enumerate(cats)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id   = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, self._id_to_file[img_id])

        img          = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to square and normalise
        img = TF.resize(img, [self.img_size, self.img_size])
        img = TF.to_tensor(img)
        img = TF.normalize(img, self._MEAN, self._STD)

        # Parse annotations -> normalised [cx, cy, w, h]
        anns   = self._annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx = (x + w * 0.5) / orig_w
            cy = (y + h * 0.5) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self._cat_to_idx[ann["category_id"]])

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.long)

        return img, boxes_t, labels_t


def _collate_fn(batch):
    """Stack images; keep per-image boxes and labels as lists."""
    imgs, boxes, labels = zip(*batch)
    return torch.stack(imgs), list(boxes), list(labels)


def make_dataloader(
    split: str,
    batch_size: int,
    img_size: int = IMG_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    """Return a DataLoader for the specified COCO split."""
    ds = COCODetection(split=split, img_size=img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_l1(model, batch_size: int = 4, device: str = "cuda") -> float:
    """
    Validation L1 loss on matched bounding boxes (val_l1).

    For each image the model's predicted boxes are matched to the ground-truth
    boxes using the Hungarian algorithm (minimising L1 cost).  The mean per-box
    L1 distance over all matched pairs in the validation set is returned.
    Lower is better.

    model(images) must return (pred_boxes, pred_logits) where
      pred_boxes:  (B, Q, 4)  normalised [cx, cy, w, h]
      pred_logits: (B, Q, C+1) — only pred_boxes is used here
    """
    from scipy.optimize import linear_sum_assignment

    val_loader    = make_dataloader("val", batch_size=batch_size, num_workers=2)
    model.eval()
    total_l1      = 0.0
    total_matched = 0

    for images, boxes_list, _ in val_loader:
        images = images.to(device)
        pred_boxes, _ = model(images)   # (B, Q, 4)

        for i in range(len(images)):
            gt   = boxes_list[i].to(device)   # (M, 4)
            if gt.shape[0] == 0:
                continue
            pred = pred_boxes[i]              # (Q, 4)

            # L1 cost matrix: (Q, M)
            cost = (pred[:, None, :] - gt[None, :, :]).abs().sum(-1)
            row_ind, col_ind = linear_sum_assignment(cost.cpu().float().numpy())

            total_l1      += (pred[row_ind] - gt[col_ind]).abs().sum().item()
            total_matched += len(row_ind)

    return total_l1 / total_matched if total_matched else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare COCO 2017 data for autoresearch CV experiments"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Also download train2017 images (~18 GB)",
    )
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()
    download_coco(include_train=args.train)
    print()
    print("Done! Ready to train.")
