"""
APPROACH 4:
Leaf Counting Pipeline:
1. HSV segmentation → extract masked leaf region
2. Feed masked image into EfficientNetV2-S
3. Last 3 CNN blocks unfrozen, rest frozen (backbone)
4. Train on leaf count regression

Run: python3 leaf_count_hsv_efficientnet.py
Run inference: python3 leaf_count_hsv_efficientnet.py --infer
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = "/home/cvpr_int_1/Nishant/Dataset/hack/leaf_estimation_dataset"
TRAIN_CSV  = os.path.join(BASE_DIR, "train.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test.csv")
SAVE_PATH  = "/home/cvpr_int_1/Nishant/Dataset/hack/best_model_hsv_effv2s.pth"
OUTPUT_CSV = "/home/cvpr_int_1/Nishant/Dataset/hack/submission_hsv_effv2s.csv"

IMG_SIZE   = 384
BATCH      = 16
EPOCHS     = 50
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42

CROP2IDX = {"mustard": 0, "okra": 1, "radish": 2, "wheat": 3}

# Per-crop HSV ranges for better segmentation
# Radish has dark/purple leaves → different range
CROP_HSV_RANGES = {
    "mustard": {"lower": np.array([25, 40, 40]),  "upper": np.array([90, 255, 255])},
    "okra":    {"lower": np.array([25, 40, 40]),  "upper": np.array([90, 255, 255])},
    "wheat":   {"lower": np.array([25, 40, 40]),  "upper": np.array([90, 255, 255])},
    "radish":  {"lower": np.array([100, 50, 30]), "upper": np.array([170, 255, 200])},  # purple/dark
}
DEFAULT_HSV = {
    "lower": np.array([25, 40, 40]),
    "upper": np.array([90, 255, 255])
}

def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# HSV MASKING FUNCTION
# ─────────────────────────────────────────────
def apply_hsv_mask(image_np, crop_type=None):
    """
    Input:  image_np — numpy array HxWx3 RGB uint8
    Output: masked_image — numpy array HxWx3 RGB uint8
            Only the green (leaf) region is kept, rest is black.
    """
    # Get crop-specific HSV range
    if crop_type and crop_type in CROP_HSV_RANGES:
        hsv_range = CROP_HSV_RANGES[crop_type]
    else:
        hsv_range = DEFAULT_HSV

    # Convert RGB → HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Create green mask
    mask = cv2.inRange(blur, hsv_range["lower"], hsv_range["upper"])

    # For radish: also add a general green mask and combine
    if crop_type == "radish":
        green_mask = cv2.inRange(blur, np.array([25, 40, 40]), np.array([90, 255, 255]))
        mask = cv2.bitwise_or(mask, green_mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilate slightly to capture leaf edges
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply mask to original RGB image
    mask_3ch = cv2.merge([mask, mask, mask])
    masked_image = cv2.bitwise_and(image_np, mask_3ch)

    return masked_image, mask


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class LeafDataset(Dataset):
    def __init__(self, df, base_dir, transform=None, is_test=False):
        self.df       = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.is_test  = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        fname = row["filename"]
        # CSV filenames don't have train/ prefix
        if not self.is_test and not fname.startswith("train/"):
            fname = "train/" + fname
        img_path = os.path.join(self.base_dir, fname)

        # Load image
        image_np = np.array(Image.open(img_path).convert("RGB"))

        # ── KEY STEP: Apply HSV mask ──
        crop_type = row.get("crop", None) if not self.is_test else None
        masked_image, _ = apply_hsv_mask(image_np, crop_type=crop_type)
        # masked_image: RGB with only leaf pixels, rest black

        if self.transform:
            masked_image = self.transform(image=masked_image)["image"]

        if self.is_test:
            return masked_image, row["filename"]

        leaf_count = float(row["leaf_count"])
        crop_idx   = CROP2IDX.get(row["crop"], 0)
        day_int  = int(str(row["day"]).replace("d","").strip())
        day_norm   = float(day_int) / 40.0

        return (
            masked_image,
            torch.tensor(leaf_count, dtype=torch.float32),
            torch.tensor(crop_idx,   dtype=torch.long),
            torch.tensor(day_norm,   dtype=torch.float32),
        )


# ─────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-30, 30), p=0.5),
    # Color jitter is mild — we don't want to destroy the HSV mask signal
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(num_holes_range=(1,4), hole_height_range=(16,32), hole_width_range=(16,32),
                    fill=0, p=0.2),  # fill=0 (black) matches masked background
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# MODEL — EfficientNetV2-S, last 3 blocks unfrozen
# ─────────────────────────────────────────────
class LeafCounterHSV(nn.Module):
    def __init__(self):
        super().__init__()

        # Load EfficientNetV2-S with ImageNet21k pretraining
        try:
            backbone = timm.create_model(
                "tf_efficientnetv2_s_in21ft1k",
                pretrained=True,
                num_classes=0,
                global_pool="avg"
            )
            print("[Model] Using tf_efficientnetv2_s_in21ft1k (ImageNet21k pretrained)")
        except Exception:
            backbone = timm.create_model(
                "efficientnetv2_s",
                pretrained=True,
                num_classes=0,
                global_pool="avg"
            )
            print("[Model] Using efficientnetv2_s (ImageNet1k pretrained)")

        self.backbone  = backbone
        feat_dim       = self.backbone.num_features  # 1280

        # ── FREEZE ALL backbone layers first ──
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── UNFREEZE last 3 blocks of EfficientNetV2-S ──
        # EfficientNetV2-S blocks: blocks[0] ... blocks[5]  (6 stages)
        # We unfreeze blocks[3], blocks[4], blocks[5] + bn2 + conv_head
        blocks_to_unfreeze = ["blocks.3", "blocks.4", "blocks.5",
                               "bn2", "conv_head"]
        unfrozen_params = 0
        for name, param in self.backbone.named_parameters():
            for layer_key in blocks_to_unfreeze:
                if name.startswith(layer_key):
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                    break

        total_params  = sum(p.numel() for p in self.backbone.parameters())
        frozen_params = total_params - unfrozen_params
        print(f"[Model] Total backbone params : {total_params:,}")
        print(f"[Model] Frozen params         : {frozen_params:,}")
        print(f"[Model] Unfrozen params       : {unfrozen_params:,}")

        # Day embedding
        self.day_embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64)
        )

        # Crop classifier head (auxiliary — only used during training)
        self.crop_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(256, 4)
        )

        # Per-crop regression heads
        self.leaf_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feat_dim + 64),
                nn.Linear(feat_dim + 64, 512), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(512, 128), nn.SiLU(),
                nn.Linear(128, 1),
                nn.ReLU()  # leaf count >= 0
            ) for _ in range(4)
        ])

    def forward(self, image, day=None, crop_idx=None):
        B    = image.shape[0]
        feat = self.backbone(image)  # [B, 1280]

        # Crop logits
        crop_logits = self.crop_head(feat)  # [B, 4]

        # Day embedding
        if day is not None:
            d = self.day_embed(day.float().unsqueeze(-1))
        else:
            d = torch.zeros(B, 64, device=feat.device)

        combined  = torch.cat([feat, d], dim=1)  # [B, 1344]
        leaf_pred = torch.zeros(B, device=feat.device)

        if self.training and crop_idx is not None:
            # Use GT crop during training
            for c in range(4):
                mask = (crop_idx == c)
                if mask.any():
                    leaf_pred[mask] = self.leaf_heads[c](combined[mask]).squeeze(-1).float()
        else:
            # Use predicted crop at inference
            pred_crop = crop_logits.argmax(dim=1)
            for c in range(4):
                mask = (pred_crop == c)
                if mask.any():
                    leaf_pred[mask] = self.leaf_heads[c](combined[mask]).squeeze(-1).float()

        return leaf_pred, crop_logits


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    huber = nn.HuberLoss(delta=1.0)
    ce    = nn.CrossEntropyLoss()

    total_loss, total_mae, count = 0.0, 0.0, 0

    for images, leaf_counts, crop_idxs, days in tqdm(loader, desc="  Train"):
        images      = images.to(DEVICE)
        leaf_counts = leaf_counts.to(DEVICE)
        crop_idxs   = crop_idxs.to(DEVICE)
        days        = days.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            pred_leaf, pred_crop = model(images, day=days, crop_idx=crop_idxs)
            loss_leaf = huber(pred_leaf, leaf_counts)
            loss_crop = ce(pred_crop, crop_idxs)
            loss      = loss_leaf + 0.3 * loss_crop

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        mae         = (pred_leaf.detach() - leaf_counts).abs().mean().item()
        total_loss += loss.item() * len(images)
        total_mae  += mae * len(images)
        count      += len(images)

    return total_loss / count, total_mae / count


def validate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, leaf_counts, crop_idxs, days in tqdm(loader, desc="  Val  "):
            images = images.to(DEVICE)
            days   = days.to(DEVICE)
            pred_leaf, _ = model(images, day=days)
            all_preds.extend(pred_leaf.cpu().numpy())
            all_labels.extend(leaf_counts.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    mae    = np.abs(preds - labels).mean()
    rmse   = np.sqrt(((preds - labels) ** 2).mean())
    return mae, rmse


def main():
    seed_everything(SEED)
    print(f"\n[Device] {DEVICE}")
    if torch.cuda.is_available():
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    df = pd.read_csv(TRAIN_CSV)
    print(f"\n[Data] train.csv shape: {df.shape}")
    print(df.head(3))
    print(df["crop"].value_counts())
    print(df["leaf_count"].describe())

    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["crop"], random_state=SEED
    )
    print(f"\n[Data] Train: {len(train_df)}  Val: {len(val_df)}")

    train_ds = LeafDataset(train_df, BASE_DIR, train_transform)
    val_ds   = LeafDataset(val_df,   BASE_DIR, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ──
    print()
    model = LeafCounterHSV().to(DEVICE)

    # Only optimize params that require grad
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-2
    )

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup   = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3)
    cosine   = CosineAnnealingLR(optimizer, T_max=EPOCHS - 3, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])

    scaler = torch.amp.GradScaler('cuda')

    best_mae      = float("inf")
    patience_count = 0
    PATIENCE      = 15

    print(f"\n[Training] Starting for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:03d}/{EPOCHS}")
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, scaler)
        val_mae, val_rmse     = validate(model, val_loader)
        scheduler.step()

        print(f"  TrainLoss={train_loss:.4f}  TrainMAE={train_mae:.4f}"
              f"  ValMAE={val_mae:.4f}  ValRMSE={val_rmse:.4f}")

        if val_mae < best_mae:
            best_mae       = val_mae
            patience_count = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ Saved best model → ValMAE={best_mae:.4f}")
        else:
            patience_count += 1
            print(f"  ⏳ No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"\n[Early Stop] Stopping at epoch {epoch}")
                break

    print(f"\n[Done] Best Val MAE: {best_mae:.4f}")
    print(f"[Done] Model saved: {SAVE_PATH}")


# ─────────────────────────────────────────────
# TTA INFERENCE
# ─────────────────────────────────────────────
def build_tta_transforms():
    norm = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    return [
        A.Compose([A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.Resize(height=IMG_SIZE, width=IMG_SIZE), A.HorizontalFlip(p=1.0)] + norm),
        A.Compose([A.Resize(height=IMG_SIZE, width=IMG_SIZE), A.VerticalFlip(p=1.0)] + norm),
        A.Compose([A.Resize(height=IMG_SIZE, width=IMG_SIZE), A.RandomRotate90(p=1.0)] + norm),
        A.Compose([A.Rotate(limit=(90,90), p=1.0),  A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.Rotate(limit=(180,180), p=1.0), A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.Rotate(limit=(270,270), p=1.0), A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(90,90), p=1.0),
                   A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.CenterCrop(height=int(IMG_SIZE*0.90), width=int(IMG_SIZE*0.90)),
                   A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.CenterCrop(height=int(IMG_SIZE*0.85), width=int(IMG_SIZE*0.85)),
                   A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.CenterCrop(height=int(IMG_SIZE*0.80), width=int(IMG_SIZE*0.80)),
                   A.Resize(height=IMG_SIZE, width=IMG_SIZE)] + norm),
        A.Compose([A.Resize(height=448, width=448), A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE)] + norm),
    ]


def tta_predict(model, masked_image_np, tta_transforms):
    """Run TTA on an already-masked numpy image."""
    model.eval()
    preds = []
    with torch.no_grad():
        for aug in tta_transforms:
            tensor = aug(image=masked_image_np)["image"].unsqueeze(0).to(DEVICE)
            pred, _ = model(tensor, day=None)
            preds.append(pred.item())
    return preds


def run_inference():
    print(f"\n[Device] {DEVICE}")
    if torch.cuda.is_available():
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # Load model
    model = LeafCounterHSV().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded from {SAVE_PATH}")

    # Load crop bounds from train.csv for clipping
    train_df   = pd.read_csv(TRAIN_CSV)
    global_min = int(train_df["leaf_count"].min()) - 1
    global_max = int(train_df["leaf_count"].max()) + 2
    print(f"[Bounds] Leaf count clip range: [{global_min}, {global_max}]")

    # Load test
    test_df = pd.read_csv(TEST_CSV)
    print(f"[Data] Test images: {len(test_df)}")

    tta_transforms = build_tta_transforms()
    results        = []
    all_stds       = []
    high_conf      = 0
    low_conf       = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        img_path = os.path.join(BASE_DIR, row["filename"])
        image_np = np.array(Image.open(img_path).convert("RGB"))

        # Apply HSV mask (crop unknown at test time → use default green range)
        masked_image, _ = apply_hsv_mask(image_np, crop_type=None)

        # TTA predictions
        preds   = tta_predict(model, masked_image, tta_transforms)
        std_dev = np.std(preds)
        all_stds.append(std_dev)

        # Confidence-based aggregation
        if std_dev > 2.0:
            raw_pred = np.median(preds)
            low_conf += 1
        else:
            raw_pred = np.mean(preds)
            high_conf += 1

        # Integer rounding + clipping
        final_pred = int(round(raw_pred))
        final_pred = max(1, final_pred)
        final_pred = int(np.clip(final_pred, global_min, global_max))

        results.append({
            "filename":             row["filename"],
            "predicted_leaf_count": final_pred
        })

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"[Output] Saved: {OUTPUT_CSV}")
    print(f"[Output] Total predictions : {len(results)}")
    print(f"[Output] High confidence   : {high_conf} ({100*high_conf/len(results):.1f}%)")
    print(f"[Output] Low confidence    : {low_conf}  ({100*low_conf/len(results):.1f}%)")
    print(f"[Output] Avg pred std      : {np.mean(all_stds):.3f}")
    print(f"\n[Output] Prediction distribution:")
    print(out_df["predicted_leaf_count"].describe())
    print("="*55)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true",
                        help="Run inference instead of training")
    args = parser.parse_args()

    if args.infer:
        run_inference()
    else:
        main()

# ─────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────
# Install deps (if needed):
#   pip3 install timm albumentations scikit-learn tqdm
#
# Train:
#   python3 leaf_count_hsv_efficientnet.py
#
# Inference:
#   python3 leaf_count_hsv_efficientnet.py --infer
#
# Quick sanity check (1 image):
#   python3 -c "
#   import cv2, numpy as np
#   from PIL import Image
#   from leaf_count_hsv_efficientnet import apply_hsv_mask
#   img = np.array(Image.open('leaf_estimation_dataset/train/mustard/d23/mustard_d23_225.png').convert('RGB'))
#   masked, mask = apply_hsv_mask(img, 'mustard')
#   cv2.imwrite('sanity_masked.png', cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
#   print('Saved sanity_masked.png — check if leaves are highlighted')
#   "