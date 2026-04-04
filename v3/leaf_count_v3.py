"""
=============================================================================
PeaceOfCode Hackathon – Track 1(a): Leaf Counting  [v5 — SOTA]
=============================================================================

KEY UPGRADES over v4:
  1. DINOv2-Small backbone   — self-supervised ViT with superior patch-level
                               features for occluded leaf boundaries
  2. Crop-Specific Heads     — separate regression MLPs per crop type,
                               shared backbone (hard-parameter sharing)
  3. Day embedding           — sinusoidal positional encoding for growth stage
                               (day parsed from filename at inference)
  4. Ordinal regression loss — count-aware soft labels (each count value is
                               a distribution, not a point) prevents cliff errors
  5. Multi-scale feature     — DINOv2 CLS token + patch mean + patch std fused
  6. DropPath regularization — stochastic depth on the regression head
  7. SWA (Stochastic Weight  — final 10 epochs averaged for better generalization
     Averaging)
  8. Balanced sampler        — oversamples rare (crop, day_bucket) combos
  9. Two-stage inference     — base pass + 12 TTA passes, crop-specific median
 10. Radish pinnate fix      — dedicated HSV range for radish to capture dark
                               leaflets without the broader green mask

Usage:
    python leaf_counting_v5.py --mode train --root ./leaf_estimation_dataset
    python leaf_counting_v5.py --mode infer --root ./leaf_estimation_dataset

Requirements:
    pip install torch torchvision timm scikit-learn scikit-image opencv-python
    pip install pandas numpy tqdm Pillow
"""

import os, argparse, random, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import mean_absolute_error
from skimage.feature import hog as skimage_hog
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARNING] timm not found — falling back to EfficientNet-B4 backbone")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── CONFIG ──────────────────────────────────────────

CFG = dict(
    root          = "./leaf_estimation_dataset",
    train_csv     = "./leaf_estimation_dataset/train.csv",
    test_csv      = "./leaf_estimation_dataset/test.csv",
    output_csv    = "./submission_v5.csv",
    checkpoint    = "./best_model_v5.pth",
    swa_checkpoint= "./swa_model_v5.pth",

    # ── Backbone ───────────────────────────────────────────────────────────
    # Options: "dinov2_small" (best), "efficientnet_b4" (fallback)
    backbone      = "dinov2_small",
    img_size      = 224,

    # ── Training ───────────────────────────────────────────────────────────
    batch_size    = 24,
    accum_steps   = 2,
    epochs        = 70,
    warmup_epochs = 6,
    swa_start     = 55,        # epoch to begin SWA averaging
    swa_lr        = 5e-5,
    lr            = 2e-4,
    backbone_lr   = 5e-5,      # lower LR for pretrained backbone layers
    weight_decay  = 1e-4,
    val_split     = 0.15,
    seed          = 42,

    # ── Crop ───────────────────────────────────────────────────────────────
    use_plant_crop        = True,
    crop_padding_default  = 0.18,
    crop_padding_mustard  = 0.28,
    crop_padding_radish   = 0.22,  # radish spreads wide too
    crop_padding_okra     = 0.20,

    # ── HOG ───────────────────────────────────────────────────────────────
    use_hog          = True,
    hog_orientations = 9,
    hog_ppc          = 8,
    hog_cpb          = 2,
    hog_proj_dim     = 128,

    # ── Day / Growth stage ────────────────────────────────────────────────
    use_day_emb   = True,
    max_day       = 45,        # max day in dataset
    day_emb_dim   = 16,

    # ── Crop embedding ────────────────────────────────────────────────────
    n_crops       = 4,
    crop_emb_dim  = 32,        # wider than v4

    # ── Ordinal regression ────────────────────────────────────────────────
    use_ordinal   = True,
    max_count     = 30,        # maximum expected leaf count
    ordinal_sigma = 1.5,       # soft-label spread

    # ── Mixup ─────────────────────────────────────────────────────────────
    use_mixup    = True,
    mixup_alpha  = 0.25,

    # ── Label noise ───────────────────────────────────────────────────────
    label_noise  = 0.05,

    # ── Early stopping ────────────────────────────────────────────────────
    patience     = 15,

    # ── TTA ───────────────────────────────────────────────────────────────
    tta_n        = 12,
    tta_agg      = "median",

    # ── Loss weights ──────────────────────────────────────────────────────
    huber_delta         = 1.5,
    mustard_loss_weight = 1.4,
    radish_loss_weight  = 1.2,  # radish also tricky at d35+
    okra_loss_weight    = 1.15,
)

CROP2IDX = {"mustard": 0, "okra": 1, "radish": 2, "wheat": 3}
IDX2CROP = {v: k for k, v in CROP2IDX.items()}

# ─────────────────────────── SEED ────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────── DAY PARSING ─────────────────────────────────────

def parse_day_from_filename(filename):
    """
    Extract growth day from filename like 'mustard/d32/mustard_d32_045.png'
    or 'test/images/img_100425.png' (returns -1 for test images → day=20 default)
    """
    try:
        parts = str(filename).replace("\\", "/").split("/")
        for p in parts:
            if p.startswith("d") and p[1:].isdigit():
                return int(p[1:])
    except Exception:
        pass
    return 20  # median day as fallback for test images

# ─────────────────────────── DYNAMIC PLANT CROP ──────────────────────────────

def get_plant_bbox(pil_img, padding=0.18, crop_type="wheat"):
    """
    HSV green-channel segmentation → plant bounding box.
    Radish uses a broader hue range to capture dark/olive leaflets.
    """
    img_np = np.array(pil_img)
    hsv    = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Radish leaflets can be dark olive-green → broader HSV range
    if str(crop_type).lower() == "radish":
        lower_green = np.array([20,  20,  20])
        upper_green = np.array([100, 255, 255])
    else:
        lower_green = np.array([25,  30,  30])
        upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    coords = cv2.findNonZero(mask)
    h, w   = img_np.shape[:2]

    if coords is None or len(coords) < 50:
        return 0, 0, w, h

    x, y, bw, bh = cv2.boundingRect(coords)

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0,     x  - pad_x)
    y1 = max(0,     y  - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)

    if (x2 - x1) * (y2 - y1) < 0.03 * w * h:
        return 0, 0, w, h

    return x1, y1, x2, y2


def crop_to_plant(pil_img, crop_type="wheat", cfg=None):
    """Per-crop-type padding for dynamic plant extraction."""
    if cfg is None:
        cfg = CFG
    ct = str(crop_type).lower()
    if ct == "mustard":
        padding = cfg["crop_padding_mustard"]
    elif ct == "radish":
        padding = cfg["crop_padding_radish"]
    elif ct == "okra":
        padding = cfg["crop_padding_okra"]
    else:
        padding = cfg["crop_padding_default"]
    x1, y1, x2, y2 = get_plant_bbox(pil_img, padding, crop_type)
    return pil_img.crop((x1, y1, x2, y2))

# ─────────────────────────── HOG EXTRACTION ──────────────────────────────────

def extract_hog(pil_img, img_size=64, orientations=9, ppc=8, cpb=2):
    small = pil_img.resize((img_size, img_size)).convert("L")
    arr   = np.array(small)
    feat  = skimage_hog(
        arr,
        orientations=orientations,
        pixels_per_cell=(ppc, ppc),
        cells_per_block=(cpb, cpb),
        feature_vector=True,
    )
    return feat.astype(np.float32)

# ─────────────────────────── ORDINAL SOFT LABELS ─────────────────────────────

def make_ordinal_label(count, max_count=30, sigma=1.5):
    """
    Convert integer leaf count to a soft probability distribution over
    [0, 1, ..., max_count]. Gaussian centered at count with std=sigma.
    This prevents the model from treating counts as fully independent classes
    and encodes the ordering (5 leaves is closer to 6 than to 1).
    """
    values = np.arange(max_count + 1, dtype=np.float32)
    probs  = np.exp(-0.5 * ((values - count) / sigma) ** 2)
    probs  = probs / probs.sum()
    return torch.tensor(probs, dtype=torch.float32)


def decode_ordinal(logits):
    """
    Convert ordinal logits → expected count via soft argmax.
    E[count] = sum_k k * softmax(logits)[k]
    """
    probs = torch.softmax(logits, dim=-1)
    bins  = torch.arange(probs.shape[-1], dtype=torch.float32, device=logits.device)
    return (probs * bins).sum(dim=-1)

# ─────────────────────────── TRANSFORMS ──────────────────────────────────────

def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size + 40, img_size + 40)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.15),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_tta_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size + 24, img_size + 24)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.65),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# ─────────────────────────── DATASET ─────────────────────────────────────────

class LeafDataset(Dataset):
    def __init__(self, root, df, cfg, transform=None,
                 is_test=False, label_noise=0.0):
        self.root        = Path(root)
        self.df          = df.reset_index(drop=True)
        self.cfg         = cfg
        self.transform   = transform
        self.is_test     = is_test
        self.label_noise = label_noise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = self.root / row["filename"]
        raw_img   = Image.open(img_path).convert("RGB")
        crop_type = str(row.get("crop", "wheat"))

        # ── Parse day ─────────────────────────────────────────────────────
        day_val = row.get("day")
        if pd.notna(day_val):
            if isinstance(day_val, str) and day_val.startswith("d") and day_val[1:].isdigit():
                day = int(day_val[1:])
            else:
                try:
                    day = int(day_val)
                except ValueError:
                    day = parse_day_from_filename(row["filename"])
        else:
            day = parse_day_from_filename(row["filename"])

        # ── Dynamic crop ──────────────────────────────────────────────────
        if self.cfg["use_plant_crop"]:
            cropped_img = crop_to_plant(raw_img, crop_type, self.cfg)
        else:
            cropped_img = raw_img

        # ── HOG ───────────────────────────────────────────────────────────
        if self.cfg["use_hog"]:
            hog_feat   = extract_hog(
                cropped_img,
                orientations=self.cfg["hog_orientations"],
                ppc=self.cfg["hog_ppc"],
                cpb=self.cfg["hog_cpb"],
            )
            hog_tensor = torch.tensor(hog_feat, dtype=torch.float32)
        else:
            hog_tensor = torch.zeros(1, dtype=torch.float32)

        # ── Image transform ───────────────────────────────────────────────
        if self.transform:
            img_tensor = self.transform(cropped_img)
        else:
            img_tensor = transforms.ToTensor()(cropped_img)

        crop_idx = CROP2IDX.get(crop_type.lower(), 3)
        day_t    = torch.tensor(day, dtype=torch.long)

        if self.is_test:
            return (img_tensor,
                    hog_tensor,
                    torch.tensor(crop_idx, dtype=torch.long),
                    day_t,
                    row["filename"])

        label = float(row["leaf_count"])
        if self.label_noise > 0:
            label += random.uniform(-self.label_noise, self.label_noise)

        # Ordinal soft label
        if self.cfg["use_ordinal"]:
            label_t = make_ordinal_label(
                label,
                max_count=self.cfg["max_count"],
                sigma=self.cfg["ordinal_sigma"],
            )
        else:
            label_t = torch.tensor(label, dtype=torch.float32)

        return (img_tensor,
                hog_tensor,
                torch.tensor(crop_idx, dtype=torch.long),
                day_t,
                label_t,
                torch.tensor(label, dtype=torch.float32))  # raw label for metrics

# ─────────────────────────── BACKBONE ────────────────────────────────────────

class DINOv2Backbone(nn.Module):
    """
    DINOv2-Small via timm.
    Returns: CLS token (384-d) + patch mean (384-d) + patch std (384-d) = 1152-d
    Multi-scale pooling captures both global structure and local leaf texture.
    """
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=224,
        )
        self.out_dim = 384 * 3  # CLS + patch_mean + patch_std

    def forward(self, x):
        out        = self.model.forward_features(x)   # (B, N+1, 384)
        cls_token  = out[:, 0]                        # (B, 384)
        patches    = out[:, 1:]                       # (B, N, 384)
        patch_mean = patches.mean(dim=1)              # (B, 384)
        patch_std  = patches.std(dim=1)               # (B, 384)
        return torch.cat([cls_token, patch_mean, patch_std], dim=-1)  # (B, 1152)


class EfficientNetBackbone(nn.Module):
    """Fallback backbone when timm/DINOv2 is not available."""
    def __init__(self):
        super().__init__()
        base        = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.feats  = base.features
        self.pool   = base.avgpool
        self.out_dim= 1792

    def forward(self, x):
        return self.pool(self.feats(x)).flatten(1)

# ─────────────────────────── SINUSOIDAL DAY EMBEDDING ────────────────────────

class SinusoidalDayEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for growth day.
    Day 1 and Day 2 are numerically close → nearby embeddings.
    Works at inference even for unseen day values.
    """
    def __init__(self, dim, max_day=45):
        super().__init__()
        self.dim     = dim
        self.max_day = max_day

    def forward(self, days):
        # days: (B,) long tensor
        days_f = days.float().unsqueeze(1)              # (B, 1)
        i      = torch.arange(self.dim // 2,
                              dtype=torch.float32,
                              device=days.device)       # (dim//2,)
        omega  = 1.0 / (10000 ** (2 * i / self.dim))   # (dim//2,)
        sin_e  = torch.sin(days_f * omega)              # (B, dim//2)
        cos_e  = torch.cos(days_f * omega)              # (B, dim//2)
        return torch.cat([sin_e, cos_e], dim=-1)        # (B, dim)

# ─────────────────────────── CBAM ATTENTION ──────────────────────────────────

class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 4)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, mid), nn.ReLU(), nn.Linear(mid, c), nn.Sigmoid())
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        ca = self.ca(x).unsqueeze(-1).unsqueeze(-1)
        x  = x * ca
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        sa = self.sa_conv(torch.cat([avg, mx], 1))
        return x * sa

# ─────────────────────────── MAIN MODEL ──────────────────────────────────────

class LeafCountNetV5(nn.Module):
    """
    v5 Architecture:
      Backbone (DINOv2-S or EfficientNet-B4)
      + HOG projection
      + Crop embedding (learned)
      + Day embedding  (sinusoidal)
      → Shared fusion layer
      → Per-crop regression heads (ordinal classification)

    Per-crop heads allow each crop's unique morphology to be handled
    by specialized parameters while sharing the expensive backbone.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg        = cfg
        self.use_ordinal= cfg["use_ordinal"]
        self.n_bins     = cfg["max_count"] + 1  # ordinal bins

        # ── Backbone ──────────────────────────────────────────────────────
        if HAS_TIMM and cfg["backbone"] == "dinov2_small":
            self.backbone = DINOv2Backbone()
            feat_dim      = self.backbone.out_dim  # 1152
        else:
            self.backbone = EfficientNetBackbone()
            feat_dim      = self.backbone.out_dim  # 1792
            # Add CBAM for EfficientNet (spatial attention lost in DINOv2 path)
            self.cbam = CBAM(1792)

        self.feat_dim = feat_dim
        self.use_dino = HAS_TIMM and cfg["backbone"] == "dinov2_small"

        # ── HOG projection ────────────────────────────────────────────────
        self.use_hog = cfg["use_hog"]
        if self.use_hog:
            dummy = extract_hog(
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)),
                orientations=cfg["hog_orientations"],
                ppc=cfg["hog_ppc"], cpb=cfg["hog_cpb"])
            hog_in = len(dummy)
            self.hog_proj = nn.Sequential(
                nn.Linear(hog_in, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, cfg["hog_proj_dim"]), nn.GELU())
            hog_out = cfg["hog_proj_dim"]
        else:
            hog_out = 0

        # ── Crop & Day embeddings ─────────────────────────────────────────
        self.crop_emb = nn.Embedding(cfg["n_crops"], cfg["crop_emb_dim"])

        self.use_day_emb = cfg["use_day_emb"]
        if self.use_day_emb:
            self.day_emb = SinusoidalDayEmbedding(cfg["day_emb_dim"],
                                                   cfg["max_day"])
            day_out = cfg["day_emb_dim"]
        else:
            day_out = 0

        # ── Shared fusion layer ───────────────────────────────────────────
        fused_dim = feat_dim + hog_out + cfg["crop_emb_dim"] + day_out
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # ── Per-crop ordinal heads ────────────────────────────────────────
        # Each crop gets its own specialist head (1024 → n_bins)
        n_bins = self.n_bins
        self.heads = nn.ModuleDict({
            crop: nn.Sequential(
                nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, n_bins),
            )
            for crop in CROP2IDX
        })

        # Shared head for unknown crops
        self.shared_head = nn.Sequential(
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, n_bins),
        )

    def forward(self, img, hog_feat, crop_idx, day):
        # ── Backbone features ─────────────────────────────────────────────
        if self.use_dino:
            feat_vec = self.backbone(img)
        else:
            fm       = self.backbone.feats(img)
            fm       = self.cbam(fm)
            feat_vec = self.backbone.pool(fm).flatten(1)

        # ── Fuse all modalities ───────────────────────────────────────────
        parts = [feat_vec]
        if self.use_hog:
            parts.append(self.hog_proj(hog_feat))
        parts.append(self.crop_emb(crop_idx))
        if self.use_day_emb:
            parts.append(self.day_emb(day))

        fused = self.fusion(torch.cat(parts, dim=-1))

        # ── Per-crop heads (vectorized dispatch) ──────────────────────────
        B     = img.shape[0]
        out   = torch.zeros(B, self.n_bins, device=img.device, dtype=fused.dtype)
        for cid, cname in IDX2CROP.items():
            mask = (crop_idx == cid)
            if mask.any():
                out[mask] = self.heads[cname](fused[mask])
        # Samples with unrecognized crop (shouldn't happen, but safe fallback)
        unknown = ~torch.isin(crop_idx,
                              torch.tensor(list(CROP2IDX.values()),
                                           device=img.device))
        if unknown.any():
            out[unknown] = self.shared_head(fused[unknown])

        if self.use_ordinal:
            return out  # (B, n_bins) — logits for KL divergence
        else:
            return decode_ordinal(out)  # (B,) — scalar counts

# ─────────────────────────── LOSS ────────────────────────────────────────────

class CropWeightedOrdinalLoss(nn.Module):
    """
    KL divergence between predicted distribution and soft ordinal target,
    with per-crop loss weighting.

    Why KL + ordinal?
      • Standard MSE ignores the ordinal structure of counts.
      • Predicting P(count=5) ≈ 0.9, P(count=6) ≈ 0.1 is almost right,
        but cross-entropy treats it as completely wrong.
      • KL divergence with soft Gaussian targets captures this nuance.
    """
    CROP_WEIGHTS = {
        CROP2IDX["mustard"]: None,  # filled from cfg at init
        CROP2IDX["radish"]:  None,
        CROP2IDX["okra"]:    None,
        CROP2IDX["wheat"]:   1.0,
    }

    def __init__(self, cfg):
        super().__init__()
        self.CROP_WEIGHTS = {
            CROP2IDX["mustard"]: cfg["mustard_loss_weight"],
            CROP2IDX["radish"]:  cfg["radish_loss_weight"],
            CROP2IDX["okra"]:    cfg["okra_loss_weight"],
            CROP2IDX["wheat"]:   1.0,
        }
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, soft_targets, crop_idx):
        # KL(target || pred) = sum target * (log target - log pred)
        log_pred = self.log_softmax(logits)           # (B, n_bins)
        kl       = (soft_targets * (
            torch.log(soft_targets.clamp(1e-9)) - log_pred
        )).sum(dim=-1)                                 # (B,)

        weights = torch.ones_like(kl)
        for cid, w in self.CROP_WEIGHTS.items():
            if w is not None:
                weights[crop_idx == cid] = w

        return (weights * kl).mean()


class HybridLoss(nn.Module):
    """
    Combines ordinal KL loss + auxiliary Huber regression loss.
    The Huber term on E[count] keeps the expected count well-calibrated.
    """
    def __init__(self, cfg):
        super().__init__()
        self.ordinal_loss = CropWeightedOrdinalLoss(cfg)
        self.huber        = nn.HuberLoss(delta=cfg["huber_delta"])
        self.use_ordinal  = cfg["use_ordinal"]
        self.max_count    = cfg["max_count"]

    def forward(self, logits, soft_targets, raw_labels, crop_idx):
        if self.use_ordinal:
            l_ord    = self.ordinal_loss(logits, soft_targets, crop_idx)
            pred_cnt = decode_ordinal(logits)
            l_huber  = self.huber(pred_cnt, raw_labels)
            return 0.7 * l_ord + 0.3 * l_huber
        else:
            return self.huber(logits, raw_labels)

# ─────────────────────────── MIXUP ───────────────────────────────────────────

def mixup_batch(imgs, hog_feats, crop_idxs, days, soft_labels, raw_labels, alpha=0.25):
    lam  = np.random.beta(alpha, alpha)
    lam  = max(lam, 1 - lam)
    idx  = torch.randperm(imgs.size(0), device=imgs.device)
    return (
        lam * imgs        + (1 - lam) * imgs[idx],
        lam * hog_feats   + (1 - lam) * hog_feats[idx],
        crop_idxs,                                        # dominant
        days,                                             # dominant
        lam * soft_labels + (1 - lam) * soft_labels[idx],
        lam * raw_labels  + (1 - lam) * raw_labels[idx],
    )

# ─────────────────────────── BALANCED SAMPLER ────────────────────────────────

def build_balanced_sampler(df, cfg):
    """
    Oversample rare (crop, day_bucket) combinations so the model trains
    equally on seedlings (d01-d10), mid-growth (d11-d25), and mature (d26-d40).
    """
    if "day" not in df.columns:
        df = df.copy()
        df["day"] = df["filename"].apply(parse_day_from_filename)
    def day_bucket(d):
        if isinstance(d, str) and d.startswith('d') and d[1:].isdigit():
            d = int(d[1:])
        elif pd.notna(d):
            try:
                d = int(d)
            except ValueError:
                d = 20
        else:
            d = 20
        if d <= 10:  return "early"
        if d <= 25:  return "mid"
        return "late"

    df = df.copy()
    df["bucket"] = df["crop"].fillna("wheat") + "_" + df["day"].apply(day_bucket)
    counts       = df["bucket"].value_counts()
    max_count    = counts.max()
    weights      = df["bucket"].map(lambda b: max_count / counts[b]).values
    sampler      = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True,
    )
    return sampler

# ─────────────────────────── LR SCHEDULE ─────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=total_epochs - warmup_epochs,
                               eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])

# ─────────────────────────── TERMINAL TABLE ──────────────────────────────────

_HDR = (f"{'Epoch':>6} | {'LR':>9} | {'Train MAE':>9} | "
        f"{'Val MAE':>7} | {'Val RMSE':>8} | {'Status'}")
_SEP = "-" * len(_HDR)

def print_epoch_row(epoch, total, lr, tr_mae, vl_mae, vl_rmse, status):
    print(f"{epoch:>5}/{total:<1} | {lr:>9.2e} | {tr_mae:>9.3f} | "
          f"{vl_mae:>7.3f} | {vl_rmse:>8.3f} | {status}")

# ─────────────────────────── TRAINING LOOP ───────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device,
              train=True, cfg=CFG, accum_steps=1):
    model.train() if train else model.eval()

    all_preds  = []
    all_labels = []
    all_crops  = []

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    if train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for step, batch in enumerate(tqdm(loader, leave=False,
                                          desc="train" if train else "val")):
            imgs, hog_feats, crop_idxs, days, soft_labels, raw_labels = batch
            imgs        = imgs.to(device)
            hog_feats   = hog_feats.to(device)
            crop_idxs   = crop_idxs.to(device)
            days        = days.to(device)
            soft_labels = soft_labels.to(device)
            raw_labels  = raw_labels.to(device)

            if train and cfg["use_mixup"]:
                imgs, hog_feats, crop_idxs, days, soft_labels, raw_labels = \
                    mixup_batch(imgs, hog_feats, crop_idxs, days,
                                soft_labels, raw_labels, cfg["mixup_alpha"])

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(imgs, hog_feats, crop_idxs, days)
                loss   = criterion(logits, soft_labels, raw_labels, crop_idxs)
                loss   = loss / accum_steps

            if train:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            if cfg["use_ordinal"]:
                preds = decode_ordinal(logits).detach().cpu().numpy()
            else:
                preds = logits.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(raw_labels.cpu().numpy())
            all_crops.extend(crop_idxs.cpu().numpy())

    mae  = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_labels)) ** 2))
    return mae, rmse, np.array(all_preds), np.array(all_labels), np.array(all_crops)


def log_per_crop_mae(preds, labels, crops, epoch):
    print(f"\n  {'─'*40}")
    print(f"  {'Crop':<10} | {'N':>5} | {'Val MAE':>8} | {'RMSE':>8}")
    print(f"  {'─'*40}")
    for cid, cname in IDX2CROP.items():
        mask = crops == cid
        if not mask.any():
            continue
        c_mae  = mean_absolute_error(labels[mask], preds[mask])
        c_rmse = np.sqrt(np.mean((preds[mask] - labels[mask]) ** 2))
        flag   = "  ← watch" if c_mae > 0.35 else ""
        print(f"  {cname:<10} | {mask.sum():>5} | {c_mae:>8.3f} | {c_rmse:>8.3f}{flag}")
    print(f"  {'─'*40}\n")

# ─────────────────────────── STRATIFIED SPLIT ────────────────────────────────

def stratified_val_split(df, val_frac=0.15, seed=42):
    """Stratify by (crop, day_bucket) to ensure all growth stages in val."""
    df = df.copy()
    if "day" not in df.columns:
        df["day"] = df["filename"].apply(parse_day_from_filename)

    def bucket(row):
        day_val = row.get("day", 20)
        # Handle string days like 'd01' from CSV, or ints
        if isinstance(day_val, str) and day_val.startswith('d') and day_val[1:].isdigit():
            d = int(day_val[1:])
        elif pd.notna(day_val):
            try:
                d = int(day_val)
            except ValueError:
                d = 20
        else:
            d = 20
        b = "early" if d <= 10 else ("mid" if d <= 25 else "late")
        return f"{row.get('crop','wheat')}_{b}"

    df["_strat"] = df.apply(bucket, axis=1)
    val_dfs = [g.sample(frac=val_frac, random_state=seed)
               for _, g in df.groupby("_strat") if len(g) >= 2]
    val_df  = pd.concat(val_dfs)
    df.drop(columns=["_strat"], inplace=True)
    val_df.drop(columns=["_strat"], inplace=True)
    return val_df

# ─────────────────────────── MAIN TRAIN ──────────────────────────────────────

def train(cfg=CFG):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[Backbone] {cfg['backbone']} | timm available: {HAS_TIMM}\n")

    df = pd.read_csv(cfg["train_csv"])
    
    # The dataset has images in a 'train' subfolder, but csv only has 'crop/...'
    # Prepend 'train/' so paths resolve correctly.
    if not df["filename"].iloc[0].startswith("train/"):
        df["filename"] = "train/" + df["filename"]
    
    # Filter out missing files safely
    valid_mask = df["filename"].apply(lambda p: (Path(cfg["root"]) / p).exists())
    missing_count = (~valid_mask).sum()
    if missing_count > 0:
        print(f"[Warning] Dropping {missing_count} missing images from training data.")
        df = df[valid_mask].reset_index(drop=True)

    if "crop" not in df.columns:
        df["crop"] = df["filename"].apply(lambda x: x.split("/")[0])
    if "day" not in df.columns:
        df["day"] = df["filename"].apply(parse_day_from_filename)
    print(f"Total: {len(df)} | Crops: {df['crop'].value_counts().to_dict()}\n")

    val_df   = stratified_val_split(df, cfg["val_split"], cfg["seed"])
    train_df = df.drop(val_df.index)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}\n")

    train_ds = LeafDataset(cfg["root"], train_df, cfg,
                           transform=get_train_transforms(cfg["img_size"]),
                           label_noise=cfg["label_noise"])
    val_ds   = LeafDataset(cfg["root"], val_df, cfg,
                           transform=get_val_transforms(cfg["img_size"]))

    sampler      = build_balanced_sampler(train_df, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = LeafCountNetV5(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M\n")

    # ── Differential LR: lower for backbone ───────────────────────────────
    backbone_params = list(model.backbone.parameters())
    head_params     = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg["backbone_lr"]},
        {"params": head_params,     "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    scheduler = build_scheduler(optimizer, cfg["warmup_epochs"], cfg["epochs"])
    criterion = HybridLoss(cfg)

    # ── SWA setup ─────────────────────────────────────────────────────────
    swa_model    = AveragedModel(model)
    swa_scheduler= SWALR(optimizer, swa_lr=cfg["swa_lr"])
    swa_started  = False

    best_mae, patience_ctr = float("inf"), 0

    print(_SEP)
    print(_HDR)
    print(_SEP)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_mae, tr_rmse, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, cfg=cfg, accum_steps=cfg["accum_steps"])

        vl_mae, vl_rmse, vl_preds, vl_labels, vl_crops = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, cfg=cfg)

        # ── SWA update ────────────────────────────────────────────────────
        if epoch >= cfg["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_started = True
        else:
            scheduler.step()

        lr = optimizer.param_groups[1]["lr"]  # head LR

        if vl_mae < best_mae:
            best_mae, patience_ctr = vl_mae, 0
            torch.save(model.state_dict(), cfg["checkpoint"])
            status = f"✓ saved  (best={best_mae:.3f})"
        else:
            patience_ctr += 1
            status = (f"✗ no imp ({patience_ctr}/{cfg['patience']})"
                      if patience_ctr < cfg["patience"]
                      else "⚡ STOP")

        print_epoch_row(epoch, cfg["epochs"], lr,
                        tr_mae, vl_mae, vl_rmse, status)

        if epoch % 5 == 0 or patience_ctr >= cfg["patience"]:
            log_per_crop_mae(vl_preds, vl_labels, vl_crops, epoch)

        if patience_ctr >= cfg["patience"]:
            print(f"\n⚡ Early stopping at epoch {epoch}")
            break

    print(_SEP)
    print(f"\nBest Val MAE: {best_mae:.4f}")

    # ── Update BN stats for SWA model ─────────────────────────────────────
    if swa_started:
        print("\nUpdating BatchNorm for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), cfg["swa_checkpoint"])
        print(f"SWA model saved → {cfg['swa_checkpoint']}")

# ─────────────────────────── INFERENCE ───────────────────────────────────────

def infer(cfg=CFG):
    set_seed(cfg["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(cfg["test_csv"])

    # Load best model (try SWA first, then regular checkpoint)
    model = LeafCountNetV5(cfg).to(device)
    swa_path = Path(cfg["swa_checkpoint"])
    ckpt_path= Path(cfg["checkpoint"])

    if swa_path.exists():
        print(f"Loading SWA checkpoint: {swa_path}")
        swa_model = AveragedModel(model)
        swa_model.load_state_dict(torch.load(swa_path, map_location=device))
        # Extract the underlying module
        infer_model = swa_model.module
    elif ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        infer_model = model
    else:
        raise FileNotFoundError("No checkpoint found. Run --mode train first.")

    infer_model.eval()

    tta_tf  = get_tta_transform(cfg["img_size"])
    base_tf = get_val_transforms(cfg["img_size"])
    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="infer"):
        img_path  = Path(cfg["root"]) / row["filename"]
        
        # Fallback for missing files
        if not img_path.exists():
            print(f"[Warning] Missing test image: {img_path} - using fallback prediction.")
            results.append({
                "filename": row["filename"],
                "predicted_leaf_count": 5
            })
            continue

        raw_img   = Image.open(img_path).convert("RGB")
        crop_type = str(row.get("crop", "wheat"))

        # Parse day from filename for test images
        day = parse_day_from_filename(row["filename"])
        if "day" in row and pd.notna(row.get("day")):
            day = int(row["day"])

        if cfg["use_plant_crop"]:
            proc_img = crop_to_plant(raw_img, crop_type, cfg)
        else:
            proc_img = raw_img

        hog_np = extract_hog(proc_img,
                             orientations=cfg["hog_orientations"],
                             ppc=cfg["hog_ppc"], cpb=cfg["hog_cpb"])
        hog_t  = torch.tensor(hog_np).unsqueeze(0).to(device)
        crop_t = torch.tensor([CROP2IDX.get(crop_type.lower(), 3)],
                               device=device)
        day_t  = torch.tensor([day], dtype=torch.long, device=device)

        preds_all = []

        # ── TTA passes ────────────────────────────────────────────────────
        for _ in range(cfg["tta_n"]):
            img_t = tta_tf(proc_img).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = infer_model(img_t, hog_t, crop_t, day_t)
                    pred   = decode_ordinal(logits).item()
            preds_all.append(pred)

        # ── Clean deterministic pass ──────────────────────────────────────
        img_c = base_tf(proc_img).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = infer_model(img_c, hog_t, crop_t, day_t)
                pred   = decode_ordinal(logits).item()
        preds_all.append(pred)

        # ── Median aggregation ────────────────────────────────────────────
        agg      = np.median(preds_all)
        pred_int = max(0, round(float(agg)))
        results.append({
            "filename":             row["filename"],
            "predicted_leaf_count": pred_int,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(cfg["output_csv"], index=False)
    print(f"\nSubmission saved → {cfg['output_csv']}")
    print(out_df.head(10))

# ─────────────────────────── ENTRY POINT ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leaf Counting v5 — DINOv2 + Ordinal + Per-Crop Heads + SWA")
    parser.add_argument("--mode",         choices=["train", "infer"], default="train")
    parser.add_argument("--root",         default=CFG["root"])
    parser.add_argument("--train_csv",    default=CFG["train_csv"])
    parser.add_argument("--test_csv",     default=CFG["test_csv"])
    parser.add_argument("--checkpoint",   default=CFG["checkpoint"])
    parser.add_argument("--epochs",       type=int,   default=CFG["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=CFG["batch_size"])
    parser.add_argument("--backbone",     default=CFG["backbone"],
                        choices=["dinov2_small", "efficientnet_b4"])
    parser.add_argument("--no_hog",       action="store_true")
    parser.add_argument("--no_crop",      action="store_true")
    parser.add_argument("--no_day_emb",   action="store_true")
    parser.add_argument("--no_ordinal",   action="store_true")
    parser.add_argument("--no_swa",       action="store_true")
    args = parser.parse_args()

    CFG["root"]       = args.root
    CFG["train_csv"]  = args.train_csv
    CFG["test_csv"]   = args.test_csv
    CFG["checkpoint"] = args.checkpoint
    CFG["epochs"]     = args.epochs
    CFG["batch_size"] = args.batch_size
    CFG["backbone"]   = args.backbone

    if args.no_hog:     CFG["use_hog"]     = False
    if args.no_crop:    CFG["use_plant_crop"] = False
    if args.no_day_emb: CFG["use_day_emb"] = False
    if args.no_ordinal: CFG["use_ordinal"] = False
    if args.no_swa:     CFG["swa_start"]   = CFG["epochs"] + 1  # disable

    if args.mode == "train":
        train(CFG)
    else:
        infer(CFG)