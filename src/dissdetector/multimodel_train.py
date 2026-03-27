import json
import os
import random
import glob
import shutil
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO


# =========================
# CONFIG
# =========================
SEED = 42

COCO_ANNOTATIONS = r"C:\Users\User\OneDrive\Desktop\waste final\result.json"
IMAGE_DIR        = r"C:\Users\User\OneDrive\Desktop\waste final\images"

MASK_DIR     = r"C:\Users\User\OneDrive\Desktop\Traning2\masks"
DATASET_ROOT = r"C:\Users\User\OneDrive\Desktop\Traning2\dataset"
TILED_ROOT   = r"C:\Users\User\OneDrive\Desktop\Traning2\tiled_dataset"
CHECKPOINTS  = r"C:\Users\User\OneDrive\Desktop\Traning2\checkpoints_stable_v2"

# Tiling
TILE_SIZE = 512
KEEP_EMPTY_RATIO_TRAIN = 0.20
KEEP_EMPTY_RATIO_VAL   = 1.00

# Training
BATCH_SIZE = 6
EPOCHS = 60
LR_FROZEN = 1e-4        # decoder-only
LR_UNFROZEN = 3e-5      # after unfreezing encoder
FREEZE_EPOCHS = 4
EARLY_STOP = 10

# DataLoader (Windows-safe)
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# Prefer GPU; fail with clear message if GPU requested but unavailable
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using device: CUDA (GPU)")
elif USE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "GPU requested but CUDA is not available. "
        "Install PyTorch with CUDA: https://pytorch.org/get-started/locally/ "
        "(e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118)"
    )
else:
    DEVICE = torch.device("cpu")
    print("Using device: CPU")


# =========================
# Seed
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# Build multi-class masks (only if you need it)
# =========================
def build_masks():
    coco = COCO(COCO_ANNOTATIONS)
    cat_ids = sorted(coco.getCatIds())
    cat_map = {cid: i+1 for i, cid in enumerate(cat_ids)}
    os.makedirs(MASK_DIR, exist_ok=True)

    for img_id in tqdm(coco.getImgIds(), desc="Building masks"):
        info = coco.loadImgs(img_id)[0]
        fname = os.path.basename(info["file_name"])
        img = cv2.imread(os.path.join(IMAGE_DIR, fname))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            cls = cat_map[ann["category_id"]]
            m = coco.annToMask(ann)
            mask[m == 1] = cls

        cv2.imwrite(os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".png"), mask)


# =========================
# Tiling
# =========================
def tile_split(img_dir, mask_dir, out_dir, tile_size=512, keep_empty_ratio=0.2):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    kept = 0
    skipped = 0

    for fname in tqdm(files, desc=f"Tiling {os.path.basename(out_dir)}"):
        img = cv2.imread(os.path.join(img_dir, fname))
        mpath = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        h, w = img.shape[:2]
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size

        if pad_h or pad_w:
            img  = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        H, W = img.shape[:2]
        stem = os.path.splitext(fname)[0]

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                img_t = img[y:y+tile_size, x:x+tile_size]
                m_t   = mask[y:y+tile_size, x:x+tile_size]

                waste_ratio = float(np.mean(m_t > 0))

                # keep all waste tiles, keep only some empty tiles
                if waste_ratio == 0.0 and random.random() > keep_empty_ratio:
                    skipped += 1
                    continue

                out_name = f"{stem}_{y}_{x}.png"
                cv2.imwrite(os.path.join(out_dir, "images", out_name), img_t)
                cv2.imwrite(os.path.join(out_dir, "masks",  out_name), m_t)
                kept += 1

    print(f"✅ {out_dir}: kept={kept}, skipped_empty={skipped}")


# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.files = []
        for f in os.listdir(img_dir):
           m = cv2.imread(os.path.join(mask_dir, f), 0)
           if m is None: 
               continue
           waste_ratio = np.mean(m > 0)
           if waste_ratio > 0:
               self.files.append(f)
               if waste_ratio > 0.02:   # more than 2% waste pixels
                   self.files.append(f)  # duplicate (oversample)
                   self.files.append(f)  # duplicate again

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            return self.__getitem__(random.randint(0, len(self.files) - 1))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].long()


# =========================
# Weights (moderate, stable)
# =========================
def compute_weights(mask_dir, num_classes):
    counts = np.zeros(num_classes)
    for p in glob.glob(os.path.join(mask_dir, "*.png")):
        m = cv2.imread(p, 0)
        for c in range(num_classes):
            counts[c] += np.sum(m == c)

    counts = np.maximum(counts, 1)
    freq = counts / counts.sum()

    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()

    # clamp to avoid extreme imbalance
    weights = np.clip(weights, 0.2, 3.0)

    print("Clipped weights:", weights)
    return torch.tensor(weights, dtype=torch.float32)


# =========================
# IoU metric (mean over classes 1..C-1)
# =========================
@torch.no_grad()
def mean_iou(logits, target, num_classes):
    pred = torch.argmax(logits, dim=1)  # [B,H,W]
    ious = []

    for cls in range(1, num_classes):   # ignore background in score
        pred_i = (pred == cls)
        targ_i = (target == cls)

        inter = (pred_i & targ_i).sum().float()
        union = (pred_i | targ_i).sum().float()

        if union > 0:
            ious.append(inter / union)

    if not ious:
        return torch.tensor(0.0, device=logits.device)
    return torch.mean(torch.stack(ious))


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    coco = COCO(COCO_ANNOTATIONS)
    NUM_CLASSES = 1 + len(coco.getCatIds())
    print("✅ NUM_CLASSES:", NUM_CLASSES)

    # (Optional) If you still need to build masks:
    # build_masks()

    # Tile train/val (this overwrites/extends your tiled folders)
    tile_split(
        os.path.join(DATASET_ROOT, "images", "train"),
        os.path.join(DATASET_ROOT, "masks",  "train"),
        os.path.join(TILED_ROOT, "train"),
        tile_size=TILE_SIZE,
        keep_empty_ratio=KEEP_EMPTY_RATIO_TRAIN
    )
    tile_split(
        os.path.join(DATASET_ROOT, "images", "val"),
        os.path.join(DATASET_ROOT, "masks",  "val"),
        os.path.join(TILED_ROOT, "val"),
        tile_size=TILE_SIZE,
        keep_empty_ratio=KEEP_EMPTY_RATIO_VAL
    )

    # Stronger satellite-safe augmentation
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.RandomBrightnessContrast(p=0.35),
        A.RandomGamma(p=0.20),
        A.GaussNoise(p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.12),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=20, p=0.5,
                   border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomFog(p=0.10),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    train_ds = SegDataset(
        os.path.join(TILED_ROOT, "train", "images"),
        os.path.join(TILED_ROOT, "train", "masks"),
        train_tf
    )
    val_ds = SegDataset(
        os.path.join(TILED_ROOT, "val", "images"),
        os.path.join(TILED_ROOT, "val", "masks"),
        val_tf
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model (ResNet34 is a solid baseline)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    # Freeze encoder initially
    for p in model.encoder.parameters():
        p.requires_grad = False

    weights = compute_weights(os.path.join(TILED_ROOT, "train", "masks"), NUM_CLASSES).to(DEVICE)
    focal = smp.losses.FocalLoss(mode="multiclass")
    dice  = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def loss_fn(logits, target):
        return 0.7 * focal(logits, target) + 0.3 * dice(logits, target)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FROZEN
    )

    # Scheduler (big improvement)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
     optimizer, mode="min", patience=3, factor=0.3
    )

    os.makedirs(CHECKPOINTS, exist_ok=True)

    best_val = 1e9
    patience = 0

    for epoch in range(EPOCHS):

        # Unfreeze + lower LR
        if epoch == FREEZE_EPOCHS:
            print("🔓 Unfreezing encoder + lowering LR")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LR_UNFROZEN
            )

        # ---- Train ----
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                val_loss += float(loss_fn(logits, masks).item())
                val_iou_sum += float(mean_iou(logits, masks, NUM_CLASSES).item())

        val_loss /= max(1, len(val_loader))
        val_iou = val_iou_sum / max(1, len(val_loader))

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | mIoU {val_iou:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS, "best.pt"))
            with open(os.path.join(CHECKPOINTS, "best_summary.json"), "w") as f:
                json.dump({
                    "best_epoch": epoch+1,
                    "best_val_loss": best_val,
                    "best_val_mIoU": val_iou,
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)
            print("✅ Saved new best")
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print("⏹ Early stopping")
                break

    print("✅ Training done.")
    print("Best checkpoint:", os.path.join(CHECKPOINTS, "best.pt"))
