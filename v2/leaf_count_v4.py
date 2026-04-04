"""
=============================================================================
PeaceOfCode Hackathon – Track 1(a): Leaf Counting
Competition-Ready Single-Image Inference Pipeline
=============================================================================
KEY DESIGN DECISIONS
--------------------
1. Single-image dataset – each training row is ONE image (angle-agnostic).
2. EfficientNetV2-S backbone (pretrained on ImageNet) + regression head.
3. HIGH RESOLUTION: 384×384 input (up from 224) for better small-leaf detail.
4. MULTI-TASK TRAINING:
     - Main head:      regression (leaf count) → MAE loss
     - Auxiliary head: crop classifier → CrossEntropy loss (training only)
     - total_loss = mae_loss + 0.3 * crop_loss
   The crop head is DROPPED at test time — backbone learns crop-aware
   features implicitly; no crop label needed at inference.
5. Heavy augmentation during training to simulate viewpoint invariance.
6. MAE + MSE compound loss with gradient clipping for stable training.
7. TTA (Test-Time Augmentation) at inference — 8 stochastic passes averaged.
8. Optional pseudo-labelling pass after first inference.

Directory layout expected (matches hackathon spec):
    leaf_estimation_dataset/
        train/
            mustard/d01/mustard_d01_000.png ...
            okra/...  radish/...  wheat/...
        test/images/img_100425.png ...
        train.csv   (filename, leaf_count, crop, day, angle)
        test.csv    (filename)

Run:
    python leaf_counting_solution.py --mode train
    python leaf_counting_solution.py --mode infer
    
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# ─────────────────────────── CONFIG ──────────────────────────────────────────

CFG = dict(
    root        = "./leaf_estimation_dataset/train",
    train_csv   = "./leaf_estimation_dataset/train.csv",
    test_csv    = "./leaf_estimation_dataset/test.csv",
    output_csv  = "./submission_v5.csv",
    checkpoint  = "./best_model_v4.pth",

    # EfficientNetV2-S is designed for 384px natively
    backbone    = "efficientnet_v2_s",
    img_size    = 384,          # ← HIGH RESOLUTION (was 224)

    batch_size  = 16,           # reduced slightly for 384px GPU memory
    epochs      = 40,
    lr          = 5e-5,
    weight_decay= 1e-4,
    val_split   = 0.15,
    seed        = 42,

    # TTA: number of augmented copies averaged at test time
    tta_n       = 8,

    # Crop classes (mustard/okra/radish/wheat)
    n_crops     = 4,

    # Loss weights
    loss_alpha  = 0.5,          # alpha*MAE + (1-alpha)*MSE  for count head
    crop_loss_w = 0.3,          # weight for auxiliary crop classifier loss
)

CROP2IDX = {"mustard": 0, "okra": 1, "radish": 2, "wheat": 3}

# ─────────────────────────── REPRODUCIBILITY ─────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────── TRANSFORMS ──────────────────────────────────────

def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_tta_transform(img_size):
    """Single stochastic augmentation used during TTA."""
    return transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# ─────────────────────────── DATASET ─────────────────────────────────────────

class LeafDataset(Dataset):
    """
    Each row in train.csv = one image → one label.
    Every angle is a separate training sample for natural viewpoint invariance.
    """

    def __init__(self, root, df, transform=None, is_test=False):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        crop_idx = CROP2IDX.get(row.get("crop", "wheat"), 3)

        if self.is_test:
            return image, torch.tensor(crop_idx, dtype=torch.long), row["filename"]

        label = torch.tensor(row["leaf_count"], dtype=torch.float32)
        return image, torch.tensor(crop_idx, dtype=torch.long), label


# ─────────────────────────── MODEL ───────────────────────────────────────────

class LeafCountNet(nn.Module):
    """
    EfficientNetV2-S backbone with TWO heads:

      TRAINING flow:
        image ──► backbone ──► features ──┬──► crop_head ──► CrossEntropy loss
                                          └──► count_head ──► MAE/MSE loss
        total_loss = mae_loss + 0.3 * crop_loss

      TEST TIME flow:
        image ──► backbone ──► features ──────► count_head ──► prediction
        (crop_head is NEVER called at inference — no crop label needed)

    The auxiliary crop_head forces the backbone to learn crop-discriminative
    features during training, making count_head predictions more accurate.
    """

    def __init__(self, n_crops=4, backbone="efficientnet_v2_s", dropout=0.3):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        if backbone == "efficientnet_v2_s":
            base = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.DEFAULT
            )
            feat_dim = base.classifier[1].in_features   # 1280
            base.classifier = nn.Identity()

        elif backbone == "efficientnet_b4":
            base = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.DEFAULT
            )
            feat_dim = base.classifier[1].in_features   # 1792
            base.classifier = nn.Identity()

        elif backbone == "efficientnet_b3":
            base = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.DEFAULT
            )
            feat_dim = base.classifier[1].in_features   # 1536
            base.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = base
        self.feat_dim = feat_dim

        # ── Auxiliary Crop Classifier Head (TRAINING ONLY) ────────────────
        # Helps backbone learn crop-discriminative features.
        # This head is NEVER called at test time.
        self.crop_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_crops),    # logits for crop classification
        )

        # ── Leaf Count Regression Head (TRAINING + TEST) ──────────────────
        self.count_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.ReLU(),                  # leaf count ≥ 0
        )

    def forward(self, img, return_crop_logits=False):
        """
        Args:
            img:               (B, 3, H, W) input image tensor
            return_crop_logits: set True only during training to get crop loss

        Returns:
            count (B,)  — always
            crop_logits (B, n_crops)  — only when return_crop_logits=True
        """
        features = self.backbone(img)                   # (B, feat_dim)

        count = self.count_head(features).squeeze(-1)   # (B,)

        if return_crop_logits:
            crop_logits = self.crop_head(features)      # (B, n_crops)
            return count, crop_logits

        return count                                    # inference path


# ─────────────────────────── LOSS ────────────────────────────────────────────

class HybridCountLoss(nn.Module):
    """alpha * MAE + (1 - alpha) * MSE  for leaf count regression."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + \
               (1 - self.alpha) * self.mse(pred, target)


# ─────────────────────────── TRAINING LOOP ───────────────────────────────────

def run_epoch(model, loader, count_criterion, crop_criterion,
              optimizer, device, crop_loss_w, train=True):
    """
    One full pass over the dataset.

    Loss formula:
        total_loss = count_loss + crop_loss_w * crop_loss

    crop_head is called here during training ONLY.
    """
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, crop_idxs, labels in tqdm(loader, leave=False):
            imgs      = imgs.to(device)
            crop_idxs = crop_idxs.to(device)
            labels    = labels.to(device)

            if train:
                # ── TRAINING: use both heads ──────────────────────────────
                count_pred, crop_logits = model(imgs, return_crop_logits=True)

                count_loss = count_criterion(count_pred, labels)
                crop_loss  = crop_criterion(crop_logits, crop_idxs)
                loss       = count_loss + crop_loss_w * crop_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            else:
                # ── VALIDATION: count head only (mirrors test-time flow) ──
                count_pred = model(imgs, return_crop_logits=False)
                count_loss = count_criterion(count_pred, labels)
                loss       = count_loss

            total_loss += loss.item() * len(labels)
            all_preds.extend(count_pred.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n    = len(all_labels)
    mae  = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_labels)) ** 2))
    return total_loss / n, mae, rmse


def train(cfg=CFG):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Resolution] {cfg['img_size']}×{cfg['img_size']} px")

    # ── Data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(cfg["train_csv"])
    if "crop" not in df.columns:
        df["crop"] = df["filename"].apply(lambda x: x.split("/")[0])

    val_df   = df.sample(frac=cfg["val_split"], random_state=cfg["seed"])
    train_df = df.drop(val_df.index)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = LeafDataset(cfg["root"], train_df,
                           transform=get_train_transforms(cfg["img_size"]))
    val_ds   = LeafDataset(cfg["root"], val_df,
                           transform=get_val_transforms(cfg["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LeafCountNet(
        n_crops  = cfg["n_crops"],
        backbone = cfg["backbone"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler     = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    count_criterion = HybridCountLoss(alpha=cfg["loss_alpha"])
    crop_criterion  = nn.CrossEntropyLoss()

    best_mae = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_mae, tr_rmse = run_epoch(
            model, train_loader, count_criterion, crop_criterion,
            optimizer, device, cfg["crop_loss_w"], train=True)

        vl_loss, vl_mae, vl_rmse = run_epoch(
            model, val_loader, count_criterion, crop_criterion,
            optimizer, device, cfg["crop_loss_w"], train=False)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train MAE={tr_mae:.3f} RMSE={tr_rmse:.3f} | "
              f"Val   MAE={vl_mae:.3f} RMSE={vl_rmse:.3f}")

        if vl_mae < best_mae:
            best_mae = vl_mae
            torch.save(model.state_dict(), cfg["checkpoint"])
            print(f"  ✓ Saved checkpoint (val MAE={best_mae:.3f})")

    print(f"\nBest Val MAE: {best_mae:.4f}")


# ─────────────────────────── INFERENCE (TTA, no crop head) ───────────────────

def infer(cfg=CFG):
    """
    TEST TIME:
        image → backbone → features → count_head → prediction

    The crop_head is NEVER called here. No crop label needed.
    TTA: tta_n stochastic augmentations are averaged for stability.
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(cfg["test_csv"])

    model = LeafCountNet(
        n_crops  = cfg["n_crops"],
        backbone = cfg["backbone"],
    ).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    model.eval()

    tta_tf  = get_tta_transform(cfg["img_size"])
    base_tf = get_val_transforms(cfg["img_size"])

    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = Path(cfg["root"]) / row["filename"]
        raw_img  = Image.open(img_path).convert("RGB")

        preds = []

        # ── TTA passes: tta_n stochastic augmentations ────────────────────
        for _ in range(cfg["tta_n"]):
            img_t = tta_tf(raw_img).unsqueeze(0).to(device)
            with torch.no_grad():
                # return_crop_logits=False → crop head never called
                p = model(img_t, return_crop_logits=False).item()
            preds.append(p)

        # ── One clean (deterministic) pass ────────────────────────────────
        img_clean = base_tf(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds.append(model(img_clean, return_crop_logits=False).item())

        pred_int = max(0, round(float(np.mean(preds))))
        results.append({
            "filename":             row["filename"],
            "predicted_leaf_count": pred_int,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(cfg["output_csv"], index=False)
    print(f"Submission saved → {cfg['output_csv']}")
    print(out_df.head())


# ─────────────────────────── PSEUDO-LABELLING ────────────────────────────────

def pseudo_label_pass(cfg=CFG, confidence_threshold=1.5):
    """
    After first inference, add high-confidence test predictions back to
    training data for a second training pass (semi-supervised trick).
    confidence_threshold: only include samples where std across TTA passes
    is below this value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(cfg["test_csv"])
    tta_tf  = get_tta_transform(cfg["img_size"])

    model = LeafCountNet(
        n_crops  = cfg["n_crops"],
        backbone = cfg["backbone"],
    ).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    model.eval()

    pseudo = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = Path(cfg["root"]) / row["filename"]
        raw_img  = Image.open(img_path).convert("RGB")

        preds = []
        for _ in range(16):   # more passes for reliable std estimate
            img_t = tta_tf(raw_img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds.append(model(img_t, return_crop_logits=False).item())

        std = np.std(preds)
        if std < confidence_threshold:
            pseudo.append({
                "filename":   row["filename"],
                "leaf_count": round(float(np.mean(preds))),
                "crop":       "wheat",  # unknown; backbone handles it implicitly
            })

    print(f"Pseudo-labelled {len(pseudo)}/{len(test_df)} test images")
    pd.DataFrame(pseudo).to_csv("pseudo_labels.csv", index=False)
    return pseudo


# ─────────────────────────── ENTRY POINT ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "infer", "pseudo"],
                                        default="train")
    parser.add_argument("--root",       default=CFG["root"])
    parser.add_argument("--train_csv",  default=CFG["train_csv"])
    parser.add_argument("--test_csv",   default=CFG["test_csv"])
    parser.add_argument("--checkpoint", default=CFG["checkpoint"])
    parser.add_argument("--epochs",     type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--backbone",   default=CFG["backbone"])
    parser.add_argument("--img_size",   type=int, default=CFG["img_size"])
    args = parser.parse_args()

    CFG.update({k: v for k, v in vars(args).items() if v is not None})

    if args.mode == "train":
        train(CFG)
    elif args.mode == "infer":
        infer(CFG)
    elif args.mode == "pseudo":
        pseudo_label_pass(CFG)