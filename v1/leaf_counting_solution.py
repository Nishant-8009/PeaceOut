"""
APPROACH 1:
=============================================================================
PeaceOfCode Hackathon – Track 1(a): Leaf Counting
Competition-Ready Single-Image Inference Pipeline
=============================================================================
KEY DESIGN DECISIONS
--------------------
1. Single-image dataset – each training row is ONE image (angle-agnostic).
2. EfficientNet-B3 backbone (pretrained on ImageNet) + regression head.
3. Crop-type embedding concatenated to image features (multi-task aware).
4. Heavy augmentation during training to simulate viewpoint invariance.
5. MAE + MSE compound loss with gradient clipping for stable training.
6. TTA (Test-Time Augmentation) at inference for better MAE/RMSE.
7. Optional pseudo-labelling pass after first inference.

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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# ─────────────────────────── CONFIG ──────────────────────────────────────────

CFG = dict(
    root        = "./leaf_estimation_dataset",  # <── change on Kaggle
    train_csv   = "./leaf_estimation_dataset/train.csv",
    test_csv    = "./leaf_estimation_dataset/test.csv",
    output_csv  = "./submission.csv",
    checkpoint  = "./best_model.pth",

    backbone    = "efficientnet_b3",   # or "convnext_small" / "vit_b_16"
    img_size    = 224,
    batch_size  = 32,
    epochs      = 40,
    lr          = 1e-5,             
    weight_decay= 1e-4,
    val_split   = 0.15,
    seed        = 42,

    # TTA: number of augmented copies averaged at test time
    tta_n       = 8,

    # Crop-type embedding size (mustard/okra/radish/wheat → 4 classes)
    n_crops     = 4,
    crop_emb_dim= 16,

    # Loss: alpha*MAE + (1-alpha)*RMSE
    loss_alpha  = 0.5,
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

# ─────────────────────────── DATASET ─────────────────────────────────────────

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


class LeafDataset(Dataset):
    """
    Each row in train.csv = one image → one label.
    The model sees single images (compliant with the test constraint).
    We DON'T group by angle; every angle is a separate training sample,
    giving the model natural viewpoint-invariance supervision.
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

        if self.is_test:
            crop_idx = CROP2IDX.get(row.get("crop", "wheat"), 3)
            return image, torch.tensor(crop_idx, dtype=torch.long), row["filename"]

        crop_idx = CROP2IDX.get(row["crop"], 3)
        label = torch.tensor(row["leaf_count"], dtype=torch.float32)
        return image, torch.tensor(crop_idx, dtype=torch.long), label


# ─────────────────────────── MODEL ───────────────────────────────────────────

class LeafCountNet(nn.Module):
    """
    EfficientNet-B3 backbone + crop-type embedding → regression head.

    Architecture flow:
        image ──► EfficientNet-B3 (1536-d) ──┐
                                              cat ──► MLP ──► scalar
        crop_id ──► Embedding (16-d) ─────────┘
    """

    def __init__(self, n_crops=4, crop_emb_dim=16, backbone="efficientnet_b3",
                 dropout=0.3):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        if backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            feat_dim = base.classifier[1].in_features   # 1536
            base.classifier = nn.Identity()
        elif backbone == "convnext_small":
            base = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
            feat_dim = base.classifier[2].in_features   # 768
            base.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = base
        self.feat_dim = feat_dim

        # ── Crop Embedding ────────────────────────────────────────────────
        self.crop_emb = nn.Embedding(n_crops, crop_emb_dim)

        # ── Regression Head ───────────────────────────────────────────────
        in_dim = feat_dim + crop_emb_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.ReLU(),          # leaf count ≥ 0
        )

    def forward(self, img, crop_idx):
        feat = self.backbone(img)                          # (B, feat_dim)
        emb  = self.crop_emb(crop_idx)                    # (B, crop_emb_dim)
        x    = torch.cat([feat, emb], dim=-1)
        out  = self.head(x).squeeze(-1)                   # (B,)
        return out


# ─────────────────────────── LOSS ────────────────────────────────────────────

class HybridLoss(nn.Module):
    """alpha * MAE + (1-alpha) * MSE"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + \
               (1 - self.alpha) * self.mse(pred, target)


# ─────────────────────────── TRAINING ────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, crop_idxs, labels in tqdm(loader, leave=False):
            imgs      = imgs.to(device)
            crop_idxs = crop_idxs.to(device)
            labels    = labels.to(device)

            preds = model(imgs, crop_idxs)
            loss  = criterion(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n      = len(all_labels)
    mae    = mean_absolute_error(all_labels, all_preds)
    rmse   = np.sqrt(np.mean((np.array(all_preds) - np.array(all_labels))**2))
    return total_loss / n, mae, rmse


def train(cfg=CFG):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(cfg["train_csv"])
    # Infer crop from filename if column missing (the hackathon CSV has it)
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
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LeafCountNet(n_crops=cfg["n_crops"],
                         crop_emb_dim=cfg["crop_emb_dim"],
                         backbone=cfg["backbone"]).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    criterion = HybridLoss(alpha=cfg["loss_alpha"])

    best_mae  = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_mae, tr_rmse = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_mae, vl_rmse = run_epoch(
            model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train MAE={tr_mae:.3f} RMSE={tr_rmse:.3f} | "
              f"Val   MAE={vl_mae:.3f} RMSE={vl_rmse:.3f}")

        if vl_mae < best_mae:
            best_mae = vl_mae
            torch.save(model.state_dict(), cfg["checkpoint"])
            print(f"  ✓ Saved checkpoint (val MAE={best_mae:.3f})")

    print(f"\nBest Val MAE: {best_mae:.4f}")


# ─────────────────────────── INFERENCE (with TTA) ────────────────────────────

def infer(cfg=CFG):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine crop for test images from filename prefix
    test_df = pd.read_csv(cfg["test_csv"])
    # test.csv has only "filename"; infer crop from path structure if possible
    # Hackathon test images are all in test/images/ so crop is unknown.
    # Use a fallback: average prediction across all 4 crop embeddings.
    test_df["crop"] = "wheat"   # placeholder; we'll ensemble over crop embeddings

    model = LeafCountNet(n_crops=cfg["n_crops"],
                         crop_emb_dim=cfg["crop_emb_dim"],
                         backbone=cfg["backbone"]).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    model.eval()

    tta_tf  = get_tta_transform(cfg["img_size"])
    base_tf = get_val_transforms(cfg["img_size"])

    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = Path(cfg["root"]) / row["filename"]
        raw_img  = Image.open(img_path).convert("RGB")

        preds_all = []

        # TTA: tta_n stochastic augmentations
        for _ in range(cfg["tta_n"]):
            img_t   = tta_tf(raw_img).unsqueeze(0).to(device)

            # Ensemble over all 4 crop-type embeddings (unknown at test time)
            for crop_id in range(cfg["n_crops"]):
                c = torch.tensor([crop_id], device=device)
                with torch.no_grad():
                    p = model(img_t, c).item()
                preds_all.append(p)

        # Also run clean image
        img_clean = base_tf(raw_img).unsqueeze(0).to(device)
        for crop_id in range(cfg["n_crops"]):
            c = torch.tensor([crop_id], device=device)
            with torch.no_grad():
                preds_all.append(model(img_clean, c).item())

        pred = np.mean(preds_all)
        # Round to nearest integer (leaf count is discrete)
        pred_int = max(0, round(pred))
        results.append({"filename": row["filename"],
                         "predicted_leaf_count": pred_int})

    out_df = pd.DataFrame(results)
    out_df.to_csv(cfg["output_csv"], index=False)
    print(f"Submission saved → {cfg['output_csv']}")
    print(out_df.head())


# ─────────────────────────── PSEUDO-LABELLING ────────────────────────────────

def pseudo_label_pass(cfg=CFG, confidence_threshold=1.5):
    """
    After first inference, add high-confidence test predictions back to
    training data for a second training pass (semi-supervised trick).
    confidence_threshold: only include samples where std across TTA < threshold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(cfg["test_csv"])
    tta_tf  = get_tta_transform(cfg["img_size"])

    model = LeafCountNet(n_crops=cfg["n_crops"],
                         crop_emb_dim=cfg["crop_emb_dim"],
                         backbone=cfg["backbone"]).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    model.eval()

    pseudo = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = Path(cfg["root"]) / row["filename"]
        raw_img  = Image.open(img_path).convert("RGB")
        preds = []
        for _ in range(16):  # more passes for std estimation
            img_t = tta_tf(raw_img).unsqueeze(0).to(device)
            for cid in range(cfg["n_crops"]):
                c = torch.tensor([cid], device=device)
                with torch.no_grad():
                    preds.append(model(img_t, c).item())
        std = np.std(preds)
        if std < confidence_threshold:
            pseudo.append({
                "filename": row["filename"],
                "leaf_count": round(np.mean(preds)),
                "crop": "wheat",   # unknown; model will ensemble anyway
            })

    print(f"Pseudo-labelled {len(pseudo)}/{len(test_df)} test images")
    pd.DataFrame(pseudo).to_csv("pseudo_labels.csv", index=False)
    return pseudo


# ─────────────────────────── ENTRY POINT ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "pseudo"],
                        default="train")
    parser.add_argument("--root",       default=CFG["root"])
    parser.add_argument("--train_csv",  default=CFG["train_csv"])
    parser.add_argument("--test_csv",   default=CFG["test_csv"])
    parser.add_argument("--checkpoint", default=CFG["checkpoint"])
    parser.add_argument("--epochs",     type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--backbone",   default=CFG["backbone"])
    args = parser.parse_args()

    CFG.update({k: v for k, v in vars(args).items() if v is not None})

    if args.mode == "train":
        train(CFG)
    elif args.mode == "infer":
        infer(CFG)
    elif args.mode == "pseudo":
        pseudo_label_pass(CFG)
