"""
APPROACH 5:
=============================================================================
Leaf Counting — Final Solution (Union Mask Version)
=============================================================================
Key design decisions:

1.  Per-crop HSV ranges (TRAINING) — each crop gets tuned thresholds
    radish is purple/dark → completely different range
2.  Union mask (INFERENCE) — all 4 crop ranges OR'd together
    → no crop label needed at test time, still better than single range
3.  4-channel input (RGB + mask) — Fan et al. 2022
    → full texture preserved in RGB, clean foreground signal in ch4
4.  EfficientNetV2-S backbone — last 3 blocks unfrozen, rest frozen
    → fast convergence, strong features, low overfit risk
5.  Per-crop regression heads (4 separate MLPs)
    → leaf count distributions differ strongly across crops
6.  Auxiliary crop classifier head
    → trained jointly, improves shared features
    → at inference: soft-weighted average across all heads
7.  Group-aware validation split — splits by (crop, day)
    → same plant's 24 angles never leak between train and val
8.  Day feature REMOVED — not available at test time
9.  TTA with std-based aggregation
    → mean when confident, median when uncertain
10. Warmup (3 epochs) + cosine annealing LR

Directory layout expected:
    leaf_estimation_dataset/
        train/
            mustard/d01/mustard_d01_000.png ...
            okra/...  radish/...  wheat/...
        test/images/img_100425.png ...
        train.csv   (filename, leaf_count, crop, day, angle)
        test.csv    (filename)

Usage:
    # Step 1 — verify masks FIRST (do this before training):
    python leaf_counting_final.py --debug
    # Open mask_debug.png — leaves should be white/dark-green, pot = black
    # The debug shows BOTH per-crop mask AND union mask side by side
    # If union mask catches pot pixels, increase UNION_OPEN_ITER below

    # Step 2 — train:
    python leaf_counting_final.py

    # Step 3 — inference:
    python leaf_counting_final.py --infer
=============================================================================
"""

import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# =============================================================================
# CONFIG  — edit paths here
# =============================================================================

CFG = dict(
    root         = "../leaf_estimation_dataset",
    train_csv    = "../leaf_estimation_dataset/train.csv",
    test_csv     = "../leaf_estimation_dataset/test.csv",
    checkpoint   = "./final_best_leaf_model.pth",
    output_csv   = "./submission_final.csv",

    img_size     = 384,
    batch_size   = 16,
    epochs       = 50,
    lr           = 1e-4,
    weight_decay = 1e-2,
    val_split    = 0.15,
    seed         = 42,
    num_workers  = 4,
    patience     = 15,

    # TTA: if std of predictions across augmentations exceeds this → use median
    tta_std_threshold = 2.0,

    # Backbone blocks to unfreeze (EfficientNetV2-S: blocks.0 … blocks.5)
    unfreeze_blocks   = ["blocks.3", "blocks.4", "blocks.5",
                         "bn2", "conv_head"],

    # Auxiliary crop-classifier loss weight
    crop_loss_weight  = 0.3,

    # Morphological cleanup iterations for union mask at test time
    # Increase if pot/soil pixels appear in debug union mask
    union_close_iter  = 2,
    union_open_iter   = 4,   # higher = more aggressive noise removal
    union_dilate_iter = 1,
)

CROP2IDX = {"mustard": 0, "okra": 1, "radish": 2, "wheat": 3}
IDX2CROP = {v: k for k, v in CROP2IDX.items()}
N_CROPS  = 4


# =============================================================================
# PER-CROP HSV RANGES  (tuned from debug.png observations)
#
# Observations from debug.png:
#   mustard: pale/young leaves missed → lower sat threshold (20), wider hue
#   okra:    small seedlings missed   → lower sat threshold (25)
#   radish:  working well             → keep purple range, add green backup
#   wheat:   thin blades, fragmented  → lower thresholds, smaller kernel
# =============================================================================

CROP_HSV = {
    # hue 20-95 catches yellow-green to deep green
    # sat 20 catches pale young mustard leaves under studio light
    "mustard": (np.array([35, 35, 50],  dtype=np.uint8),
                np.array([95, 255, 255], dtype=np.uint8)),

    # similar to mustard but slightly tighter hue
    "okra":    (np.array([27, 32, 40],  dtype=np.uint8),
                np.array([95, 255, 255], dtype=np.uint8)),

    # radish leaves are purple/dark — completely different hue range
    # secondary green pass handles any green portions of radish
    "radish":  (np.array([104, 44, 34], dtype=np.uint8),
                np.array([170, 255, 200], dtype=np.uint8)),

    # wheat blades are very pale/narrow — needs lowest saturation threshold
    "wheat":   (np.array([27, 25, 35],  dtype=np.uint8),
                np.array([95, 255, 255], dtype=np.uint8)),
}

# Default used when crop is unknown (test time with single-range fallback)
DEFAULT_HSV = (np.array([20, 20, 35],  dtype=np.uint8),
               np.array([95, 255, 255], dtype=np.uint8))

# Crop-specific morphological kernel sizes
# Wheat has very thin blades — large kernels destroy them
CROP_KERNEL = {
    "mustard": (7, 2),   # (kernel_size, dilate_iterations)
    "okra":    (7, 2),
    "radish":  (5, 1),
    "wheat":   (3, 1),   # small kernel preserves thin blade structure
}
DEFAULT_KERNEL = (5, 1)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# MASK GENERATION — two modes
# =============================================================================

def generate_mask_single(image_rgb: np.ndarray,
                         crop_type: str = None) -> np.ndarray:
    """
    Per-crop HSV mask used during TRAINING (crop label is known).
    Returns (H, W) uint8 array: 255 = leaf, 0 = background.
    """
    lower, upper   = CROP_HSV.get(crop_type, DEFAULT_HSV)
    ksz, dil_iter  = CROP_KERNEL.get(crop_type, DEFAULT_KERNEL)

    hsv  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask = cv2.inRange(blur, lower, upper)

    # Radish: also catch any green portions
    if crop_type == "radish":
        gl, gu = DEFAULT_HSV
        green_mask = cv2.inRange(blur, gl, gu)
        mask = cv2.bitwise_or(mask, green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=dil_iter)

    return mask


def generate_mask_union(image_rgb: np.ndarray,
                        cfg: dict = CFG) -> np.ndarray:
    """
    Union of ALL crop-specific HSV masks — used during INFERENCE
    when crop type is unknown.

    Why union works:
      - Each crop range is tuned to catch that crop's leaves
      - OR-ing them catches whatever any crop could look like
      - Post-processing (open with larger kernel) removes noise from
        the extra false-positives the union introduces
      - Still far better than single DEFAULT_HSV because it includes
        radish's purple range + wheat's low-saturation range

    Tune cfg["union_open_iter"] if pot pixels appear in debug output.
    """
    hsv  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # OR all crop masks together
    combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for lower, upper in CROP_HSV.values():
        m        = cv2.inRange(blur, lower, upper)
        combined = cv2.bitwise_or(combined, m)

    # Moderate kernel — compromise between wheat (small) and mustard (large)
    # Open with more iterations to remove noise introduced by the union
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel,
                                iterations=cfg["union_close_iter"])
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel,
                                iterations=cfg["union_open_iter"])
    combined = cv2.dilate(combined, kernel,
                          iterations=cfg["union_dilate_iter"])

    return combined


# =============================================================================
# 4-CHANNEL IMAGE BUILDER
# =============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise_and_to_tensor(img_4ch_hwc: np.ndarray) -> torch.Tensor:
    """
    img_4ch_hwc : (H, W, 4) — channels 0-2 = RGB uint8, channel 3 = mask uint8
    Returns     : (4, H, W) float32 tensor
                    channels 0-2 : ImageNet-normalised
                    channel  3   : float in [0, 1]
    """
    img = img_4ch_hwc.astype(np.float32)
    img[:, :, :3] = (img[:, :, :3] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img[:, :, 3]  =  img[:, :, 3] / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1))   # CHW


# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================

def get_train_aug(img_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1,
                 scale=(0.85, 1.15),
                 rotate=(-30, 30), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(16, 32),
                        hole_width_range=(16, 32),
                        fill=0, p=0.2),
    ])


def get_val_aug(img_size: int) -> A.Compose:
    return A.Compose([A.Resize(img_size, img_size)])


def get_colour_jitter() -> A.Compose:
    """Applied to RGB BEFORE mask generation so HSV thresholds stay valid."""
    return A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(p=0.2),
    ])


# =============================================================================
# DATASET
# =============================================================================

class LeafDataset(Dataset):
    """
    Training: uses per-crop HSV mask (crop label known)
    Test:     uses union HSV mask (crop label unknown)

    Each sample returns a (4, H, W) tensor — RGB + binary mask.
    Day feature deliberately excluded — not available at test time.
    """

    def __init__(self, root: str, df: pd.DataFrame,
                 img_size: int = 384,
                 augment: bool = False,
                 is_test: bool = False,
                 cfg: dict = CFG):
        self.root        = Path(root)
        self.df          = df.reset_index(drop=True)
        self.img_size    = img_size
        self.augment     = augment
        self.is_test     = is_test
        self.cfg         = cfg
        self.spatial_aug = get_train_aug(img_size) if augment else get_val_aug(img_size)
        self.colour_aug  = get_colour_jitter() if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        fname = row["filename"]

        if not self.is_test and not fname.startswith("train/"):
            fname = "train/" + fname

        image_rgb = np.array(
            Image.open(self.root / fname).convert("RGB"))   # (H,W,3) uint8

        crop_type = None if self.is_test else row.get("crop", None)

        # ── Colour jitter on RGB BEFORE masking ──────────────────────────
        if self.colour_aug is not None:
            image_rgb = self.colour_aug(image=image_rgb)["image"]

        # ── Generate mask ─────────────────────────────────────────────────
        if self.is_test:
            mask = generate_mask_union(image_rgb, self.cfg)   # union at test
        else:
            mask = generate_mask_single(image_rgb, crop_type) # per-crop at train

        # ── Stack to 4-channel (H,W,4) ───────────────────────────────────
        img_4ch = np.concatenate(
            [image_rgb, mask[:, :, np.newaxis]], axis=-1)

        # ── Spatial augmentation (applied to all 4 channels) ─────────────
        img_4ch = self.spatial_aug(image=img_4ch)["image"]    # (H,W,4)

        # ── Normalise + tensor ────────────────────────────────────────────
        tensor = normalise_and_to_tensor(img_4ch)              # (4,H,W)

        if self.is_test:
            return tensor, row["filename"]

        leaf_count = torch.tensor(float(row["leaf_count"]), dtype=torch.float32)
        crop_idx   = torch.tensor(CROP2IDX.get(str(crop_type), 0), dtype=torch.long)
        return tensor, leaf_count, crop_idx


# =============================================================================
# MODEL
# =============================================================================

class LeafCountNet(nn.Module):
    """
    EfficientNetV2-S with 4-channel input.

    Heads:
      crop_head  : auxiliary cross-entropy (training only, improves features)
      leaf_heads : 4 separate regression MLPs (one per crop)

    Inference: soft-weighted average across all 4 heads using crop softmax
    → robust even if crop classifier makes a mistake
    """

    def __init__(self, cfg: dict = CFG):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        try:
            backbone = timm.create_model(
                "tf_efficientnetv2_s_in21ft1k",
                pretrained=True, num_classes=0, global_pool="avg")
            print("[Model] tf_efficientnetv2_s_in21ft1k (ImageNet21k)")
        except Exception:
            backbone = timm.create_model(
                "efficientnetv2_s",
                pretrained=True, num_classes=0, global_pool="avg")
            print("[Model] efficientnetv2_s (ImageNet1k)")

        feat_dim = backbone.num_features   # 1280

        # ── Patch first conv: 3ch → 4ch ───────────────────────────────────
        old_conv = backbone.conv_stem
        out_ch, _, kH, kW = old_conv.weight.shape
        new_conv = nn.Conv2d(4, out_ch,
                             kernel_size=(kH, kW),
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            # Copy pretrained RGB weights
            new_conv.weight[:, :3] = old_conv.weight
            # Initialise 4th channel as mean of RGB — preserves scale
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.conv_stem = new_conv

        # ── Freeze all, then selectively unfreeze ─────────────────────────
        for p in backbone.parameters():
            p.requires_grad = False

        unfrozen = 0
        for name, p in backbone.named_parameters():
            if any(name.startswith(k) for k in cfg["unfreeze_blocks"]):
                p.requires_grad = True
                unfrozen += p.numel()

        total  = sum(p.numel() for p in backbone.parameters())
        print(f"[Model] Backbone: total={total:,} | "
              f"frozen={total-unfrozen:,} | unfrozen={unfrozen:,}")

        self.backbone = backbone
        self.feat_dim = feat_dim

        # ── Auxiliary crop head (training only) ───────────────────────────
        self.crop_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, N_CROPS),
        )

        # ── Per-crop regression heads ─────────────────────────────────────
        def _head():
            return nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 512), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(512, 128),      nn.SiLU(), nn.Dropout(0.15),
                nn.Linear(128, 1),
                nn.ReLU(),   # leaf count >= 0
            )
        self.leaf_heads = nn.ModuleList([_head() for _ in range(N_CROPS)])

    def forward(self, x: torch.Tensor,
                crop_idx: torch.Tensor = None):
        """
        x        : (B, 4, H, W)
        crop_idx : (B,) long — GT crop (training only, None at inference)
        Returns  : leaf_pred (B,), crop_logits (B, N_CROPS)
        """
        B    = x.shape[0]
        feat = self.backbone(x)                     # (B, 1280)
        crop_logits = self.crop_head(feat)           # (B, 4)
        leaf_pred = torch.zeros(B, device=feat.device, dtype=feat.dtype)

        if self.training and crop_idx is not None:
            # Training: use GT crop → route to correct head
            for c in range(N_CROPS):
                m = (crop_idx == c)
                if m.any():
                    leaf_pred[m] = self.leaf_heads[c](feat[m]).squeeze(-1)
        else:
            # Inference: soft-weighted average across all 4 heads
            # → robust to crop classification errors
            crop_probs = torch.softmax(crop_logits, dim=1)          # (B, 4)
            head_preds = torch.stack(
                [self.leaf_heads[c](feat).squeeze(-1) for c in range(N_CROPS)],
                dim=1)                                               # (B, 4)
            leaf_pred = (crop_probs * head_preds).sum(dim=1)        # (B,)

        return leaf_pred, crop_logits


# =============================================================================
# LOSS
# =============================================================================

class HybridLoss(nn.Module):
    def __init__(self, crop_weight: float = 0.3, huber_delta: float = 1.0):
        super().__init__()
        self.crop_weight = crop_weight
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, leaf_pred, leaf_gt, crop_logits, crop_gt):
        l_leaf = self.huber(leaf_pred, leaf_gt)
        l_crop = self.ce(crop_logits, crop_gt)
        return l_leaf + self.crop_weight * l_crop, l_leaf, l_crop


# =============================================================================
# GROUP-AWARE VALIDATION SPLIT
# =============================================================================

def group_aware_split(df: pd.DataFrame,
                      val_frac: float = 0.15,
                      seed: int = 42):
    """
    Splits by (crop, day) so the same plant's 24 angles
    never appear in both train and val. Prevents data leakage.
    """
    df = df.copy()
    df["_group"] = df["crop"] + "_" + df["day"].astype(str)
    groups    = df["_group"].unique()
    rng       = np.random.default_rng(seed)
    n_val     = max(1, int(len(groups) * val_frac))
    val_grps  = set(rng.choice(groups, size=n_val, replace=False))
    mask      = df["_group"].isin(val_grps)
    train_df  = df[~mask].drop(columns=["_group"]).reset_index(drop=True)
    val_df    = df[ mask].drop(columns=["_group"]).reset_index(drop=True)
    return train_df, val_df


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = total_mae = count = 0

    for images, leaf_counts, crop_idxs in tqdm(loader, desc="  Train", leave=False):
        images      = images.to(device)
        leaf_counts = leaf_counts.to(device)
        crop_idxs   = crop_idxs.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            pred_leaf, pred_crop = model(images, crop_idx=crop_idxs)
            loss, _, _ = criterion(pred_leaf, leaf_counts, pred_crop, crop_idxs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        b           = len(images)
        total_loss += loss.item() * b
        total_mae  += (pred_leaf.detach() - leaf_counts).abs().mean().item() * b
        count      += b

    return total_loss / count, total_mae / count


def train_one_epoch_cpu(model, loader, criterion, optimizer, device):
    """CPU fallback — no AMP scaler."""
    model.train()
    total_loss = total_mae = count = 0

    for images, leaf_counts, crop_idxs in tqdm(loader, desc="  Train", leave=False):
        images      = images.to(device)
        leaf_counts = leaf_counts.to(device)
        crop_idxs   = crop_idxs.to(device)

        optimizer.zero_grad()
        pred_leaf, pred_crop = model(images, crop_idx=crop_idxs)
        loss, _, _ = criterion(pred_leaf, leaf_counts, pred_crop, crop_idxs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        b           = len(images)
        total_loss += loss.item() * b
        total_mae  += (pred_leaf.detach() - leaf_counts).abs().mean().item() * b
        count      += b

    return total_loss / count, total_mae / count


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, leaf_counts, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        pred_leaf, _ = model(images)
        all_preds.extend(pred_leaf.cpu().numpy())
        all_labels.extend(leaf_counts.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    mae    = np.abs(preds - labels).mean()
    rmse   = np.sqrt(((preds - labels) ** 2).mean())
    return mae, rmse


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================

def train(cfg: dict = CFG):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(cfg["train_csv"])
    print(f"\n[Data] train.csv: {df.shape}")
    print(df["crop"].value_counts())
    print(df["leaf_count"].describe())

    train_df, val_df = group_aware_split(df, cfg["val_split"], cfg["seed"])
    print(f"\n[Split] Train: {len(train_df)} | Val: {len(val_df)} (group-aware)")

    train_ds = LeafDataset(cfg["root"], train_df,
                           img_size=cfg["img_size"], augment=True,
                           is_test=False, cfg=cfg)
    val_ds   = LeafDataset(cfg["root"], val_df,
                           img_size=cfg["img_size"], augment=False,
                           is_test=False, cfg=cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=True)

    model     = LeafCountNet(cfg).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=3)
    cosine    = CosineAnnealingLR(optimizer,
                                  T_max=cfg["epochs"] - 3, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[3])
    criterion = HybridLoss(crop_weight=cfg["crop_loss_weight"])

    use_amp   = torch.cuda.is_available()
    scaler    = GradScaler("cuda") if use_amp else None

    best_mae     = float("inf")
    patience_cnt = 0

    print(f"\n[Training] epochs={cfg['epochs']}  patience={cfg['patience']}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"Epoch {epoch:03d}/{cfg['epochs']}")

        if use_amp:
            tr_loss, tr_mae = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device)
        else:
            tr_loss, tr_mae = train_one_epoch_cpu(
                model, train_loader, criterion, optimizer, device)

        val_mae, val_rmse = validate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  loss={tr_loss:.4f}  trainMAE={tr_mae:.3f} | "
              f"valMAE={val_mae:.3f}  valRMSE={val_rmse:.3f}  lr={lr_now:.2e}")

        if val_mae < best_mae:
            best_mae     = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), cfg["checkpoint"])
            print(f"  ✓ Saved checkpoint  (valMAE={best_mae:.4f})")
        else:
            patience_cnt += 1
            print(f"  — no improvement ({patience_cnt}/{cfg['patience']})")
            if patience_cnt >= cfg["patience"]:
                print(f"[Early stop] epoch {epoch}")
                break

    print(f"\n[Done] Best val MAE : {best_mae:.4f}")
    print(f"[Done] Checkpoint   : {cfg['checkpoint']}")


# =============================================================================
# TTA INFERENCE
# =============================================================================

def build_tta_transforms(img_size: int) -> list:
    """12 deterministic TTA variants — spatial only, no colour jitter."""
    s = img_size
    return [
        A.Compose([A.Resize(s, s)]),
        A.Compose([A.Resize(s, s), A.HorizontalFlip(p=1.0)]),
        A.Compose([A.Resize(s, s), A.VerticalFlip(p=1.0)]),
        A.Compose([A.Resize(s, s), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        A.Compose([A.Rotate(limit=(90,  90),  p=1.0), A.Resize(s, s)]),
        A.Compose([A.Rotate(limit=(180, 180), p=1.0), A.Resize(s, s)]),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), A.Resize(s, s)]),
        A.Compose([A.Rotate(limit=(90,  90),  p=1.0),
                   A.HorizontalFlip(p=1.0), A.Resize(s, s)]),
        A.Compose([A.CenterCrop(int(s * 0.90), int(s * 0.90)), A.Resize(s, s)]),
        A.Compose([A.CenterCrop(int(s * 0.85), int(s * 0.85)), A.Resize(s, s)]),
        A.Compose([A.CenterCrop(int(s * 0.80), int(s * 0.80)), A.Resize(s, s)]),
        A.Compose([A.Resize(448, 448), A.CenterCrop(s, s)]),
    ]


def infer(cfg: dict = CFG):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = LeafCountNet(cfg).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    model.eval()
    print(f"[Model] Loaded from {cfg['checkpoint']}")

    # Clip predictions to training distribution
    train_df  = pd.read_csv(cfg["train_csv"])
    clip_min  = max(1, int(train_df["leaf_count"].min()) - 1)
    clip_max  = int(train_df["leaf_count"].max()) + 2
    print(f"[Bounds] clip=[{clip_min}, {clip_max}]")

    test_df        = pd.read_csv(cfg["test_csv"])
    tta_transforms = build_tta_transforms(cfg["img_size"])
    results        = []
    high_conf = low_conf = 0
    all_stds  = []

    print(f"[Inference] {len(test_df)} images × {len(tta_transforms)} TTA\n")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path  = Path(cfg["root"]) / row["filename"]
        image_rgb = np.array(Image.open(img_path).convert("RGB"))

        # ── Union mask (crop unknown at test time) ────────────────────────
        mask    = generate_mask_union(image_rgb, cfg)
        img_4ch = np.concatenate(
            [image_rgb, mask[:, :, np.newaxis]], axis=-1)   # (H,W,4)

        # ── TTA predictions ───────────────────────────────────────────────
        preds = []
        with torch.no_grad():
            for aug in tta_transforms:
                aug_img = aug(image=img_4ch)["image"]
                tensor  = normalise_and_to_tensor(aug_img).unsqueeze(0).to(device)
                pred, _ = model(tensor)
                preds.append(pred.item())

        std = np.std(preds)
        all_stds.append(std)

        if std >= cfg["tta_std_threshold"]:
            raw = np.median(preds)
            low_conf += 1
        else:
            raw = np.mean(preds)
            high_conf += 1

        final = int(np.clip(max(1, round(raw)), clip_min, clip_max))
        results.append({"filename": row["filename"],
                        "predicted_leaf_count": final})

    out_df = pd.DataFrame(results)
    out_df.to_csv(cfg["output_csv"], index=False)

    print(f"\n{'='*55}")
    print(f"[Output] {cfg['output_csv']}")
    print(f"[Output] Total          : {len(results)}")
    print(f"[Output] High confidence: {high_conf} ({100*high_conf/len(results):.1f}%)")
    print(f"[Output] Low  confidence: {low_conf}  ({100*low_conf/len(results):.1f}%)")
    print(f"[Output] Mean TTA std   : {np.mean(all_stds):.3f}")
    print(f"\n[Predictions]")
    print(out_df["predicted_leaf_count"].describe())
    print("="*55)


# =============================================================================
# DEBUG — visualise BOTH per-crop and union masks
# This is the most important step before training.
# Check:  leaves = white/dark green,  pot + background = black
# =============================================================================

def debug_masks(cfg: dict = CFG, n_per_crop: int = 2):
    """
    Saves mask_debug.png showing for each sampled image:
      Col 1: Original RGB
      Col 2: Per-crop HSV mask  (used during training)
      Col 3: Union HSV mask     (used during inference)
      Col 4: Masked RGB (union) — visual sanity check

    How to interpret:
      GOOD: leaves are clearly white/dark in mask, pot is black
      BAD : pot pixels appear white → increase union_open_iter in CFG
      BAD : leaves are missing     → lower saturation lower-bound in CROP_HSV
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(cfg["train_csv"])
    samples = (df.groupby("crop")
                 .apply(lambda g: g.sample(
                     min(n_per_crop, len(g)), random_state=42))
                 .reset_index(drop=True))
    n = len(samples)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original RGB",
                  "Per-crop mask\n(training)",
                  "Union mask\n(inference / test time)",
                  "Masked RGB (union)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, pad=6)

    for i, (_, row) in enumerate(samples.iterrows()):
        fname = row["filename"]
        if not fname.startswith("train/"):
            fname = "train/" + fname

        img_path  = Path(cfg["root"]) / fname
        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        crop_type = row["crop"]

        mask_single = generate_mask_single(image_rgb, crop_type)
        mask_union  = generate_mask_union(image_rgb, cfg)

        # Masked RGB using union mask
        mask_3ch   = np.stack([mask_union] * 3, axis=-1)
        masked_rgb = cv2.bitwise_and(image_rgb, mask_3ch)

        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_ylabel(
            f"{crop_type}\nleaves={row['leaf_count']}", fontsize=8)

        axes[i, 1].imshow(mask_single, cmap="Greens")
        axes[i, 2].imshow(mask_union,  cmap="Greens")
        axes[i, 3].imshow(masked_rgb)

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        "HSV Mask Debug\n"
        "GOOD: leaves = dark green / white  |  pot + bg = pale/white (background)\n"
        "BAD (per-crop): leaves missing → lower sat in CROP_HSV\n"
        "BAD (union): pot pixels dark → increase union_open_iter in CFG",
        fontsize=9, y=1.01)

    plt.tight_layout()
    out = "mask_debug.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"\n[Debug] Saved → {out}")
    print("  Col 2 (per-crop) = what training sees")
    print("  Col 3 (union)    = what inference sees — THIS is what matters most")
    print("  If union catches pot:    increase CFG['union_open_iter'] (try 3 or 4)")
    print("  If union misses leaves:  lower CROP_HSV saturation lower-bound")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer",  action="store_true", help="Run inference")
    parser.add_argument("--debug",  action="store_true",
                        help="Visualise masks (run this first!)")
    parser.add_argument("--root",   default=CFG["root"])
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch",  type=int, default=CFG["batch_size"])
    args = parser.parse_args()

    CFG["root"]       = args.root
    CFG["epochs"]     = args.epochs
    CFG["batch_size"] = args.batch

    if args.debug:
        debug_masks(CFG)
    elif args.infer:
        infer(CFG)
    else:
        train(CFG)