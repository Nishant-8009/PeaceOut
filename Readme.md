# 🌿 Leaf Count Estimation — PeaceOfCode Hackathon

Single-Image Leaf Counting using Deep Learning and Computer Vision

---

## 📌 Overview

This repository contains our complete solution pipeline for the **PeaceOfCode Hackathon — Track 1(a): Leaf Counting**.

The challenge focused on estimating the number of leaves from a **single RGB image** of a plant captured from arbitrary viewpoints.

The primary objective was to design models capable of:

* Handling viewpoint variations
* Dealing with occlusions and overlapping leaves
* Generalizing across multiple crop types
* Operating under strict single-image inference constraints

We explored and benchmarked **six different approaches**, ranging from EfficientNet baselines to Vision Transformers, HSV-based segmentation pipelines, and crop-aware multi-task learning systems.

---

# 🎯 Problem Statement

Given a single image of a plant:

```math
f(I_θ) = y
```

where:

* `I_θ` = plant image captured at viewpoint `θ`
* `y ∈ ℕ` = leaf count

The model must remain robust to:

* Viewpoint changes
* Partial visibility
* Self-occlusion
* Morphological variation across crops

while operating entirely under a **single-image inference setting**.

---

# 🌱 Key Challenges

## 1. Occlusion

Leaves frequently overlap or hide behind each other, especially during:

* Early growth stages
* Dense canopies
* Multi-angle captures

## 2. Growth Stage Variability

Different growth stages produce:

* Tiny emerging leaves
* Dense late-stage clusters
* Ambiguous germination regions

## 3. Crop Morphology

Different crops exhibit fundamentally different structures:

* Radish / Mustard → compound leaves
* Wheat → tillers and elongated structures
* Okra → broad sparse leaves

---

# 🧠 Approaches Explored

We experimented with six progressively advanced approaches.

---

# 🚀 Approach 1 — EfficientNet-B3 Baseline

## Key Features

* EfficientNet-B3 backbone
* ImageNet pretrained weights
* Crop embedding concatenated with image features
* MAE + MSE compound loss
* Heavy augmentations
* Test-Time Augmentation (TTA)

## Design Philosophy

The objective was to create a strong baseline using:

* Multi-task crop awareness
* Robust viewpoint invariance
* Stable regression training

## Validation Performance

| Metric     | Score |
| ---------- | ----- |
| Train MAE  | 0.716 |
| Train RMSE | 1.074 |
| Val MAE    | 0.243 |
| Val RMSE   | 0.333 |

---

# 🚀 Approach 2 — High-Resolution Multi-Task EfficientNetV2-S

## Key Improvements over A1

* EfficientNetV2-S backbone
* Higher resolution input (`384×384`)
* Auxiliary crop classification head
* Implicit crop-aware representation learning
* 8-pass TTA
* Multi-task optimization

## Multi-Task Learning Strategy

The network jointly optimized:

### Main Task

* Leaf count regression

### Auxiliary Task

* Crop classification

```python
loss = mae_loss + 0.3 * crop_loss
```

The crop head was discarded during inference.

This allowed the backbone to learn:

* Crop-specific features
* Morphology-aware representations
* Better feature disentanglement

without requiring crop labels at test time.

## Validation Performance

| Metric     | Score |
| ---------- | ----- |
| Train MAE  | 0.640 |
| Train RMSE | 0.990 |
| Val MAE    | 0.217 |
| Val RMSE   | 0.282 |

✅ **Best Performing Model Overall**

---

# 🚀 Approach 3 — DINOv2 Vision Transformer Pipeline

## Major Innovations

* DINOv2-Small self-supervised ViT backbone
* Crop-specific regression heads
* Sinusoidal day embeddings
* Ordinal regression loss
* Multi-scale patch feature fusion
* DropPath regularization
* SWA (Stochastic Weight Averaging)
* Balanced sampling strategy
* 12-pass TTA

## Why DINOv2?

Transformers provide superior:

* Patch-level reasoning
* Occlusion understanding
* Global context aggregation

which are highly relevant for dense leaf structures.

## Validation Performance

| Metric  | Score  |
| ------- | ------ |
| Val MAE | 0.2592 |

## Key Observation

Despite architectural sophistication, the ViT-based pipeline likely remained data-limited.

---

# 🚀 Approach 4 — HSV Segmentation + EfficientNetV2-S

## Pipeline

1. HSV segmentation
2. Leaf mask extraction
3. Masked RGB generation
4. EfficientNetV2-S regression

## Motivation

We hypothesized that explicitly isolating leaf regions would:

* Reduce background noise
* Improve feature localization
* Help the model focus on plant structure

## Validation Performance

| Metric       | Score  |
| ------------ | ------ |
| Best Val MAE | 0.2580 |

## Observation

Classical computer vision preprocessing remained surprisingly competitive.

However, HSV thresholds were sensitive to:

* Lighting conditions
* Crop-specific color distributions
* Shadows and soil regions

---

# 🚀 Approach 5 — Per-Crop Regression Heads + 4-Channel Input

## Major Components

* RGB + segmentation mask input
* Union HSV mask during inference
* EfficientNetV2-S backbone
* 4 separate crop-specific regression heads
* Auxiliary crop classifier
* Soft-weighted prediction aggregation
* Cosine annealing scheduler
* Warmup strategy
* TTA with uncertainty-aware aggregation

## Key Idea

Different crops exhibit very different:

* Leaf distributions
* Shapes
* Growth dynamics

Thus, specialized regression heads were introduced.

## Validation Performance

| Metric    | Score |
| --------- | ----- |
| Train MAE | 0.533 |
| Val MAE   | 0.482 |
| Val RMSE  | 0.739 |

## Observation

This was the most complex pipeline but suffered from:

* Strong overfitting
* Insufficient data for specialization
* Increased optimization difficulty

---

# 🚀 Approach 6 — EfficientNet-B4 Direct Regression Baseline

## Simplified Design

* EfficientNet-B4 backbone
* Direct regression setup
* L1 loss optimization
* AdamW optimizer
* Group-aware validation split
* Strong augmentations

## Validation Performance

| Metric       | Score  |
| ------------ | ------ |
| Best Val MAE | 0.6223 |

## Observation

This clean ablation study demonstrated:

✅ Auxiliary supervision is highly important.

Pure regression without crop-aware guidance underperformed significantly.

---

# 📊 Comparative Results

| Approach | Backbone         | Key Innovation                   | Val MAE   |
| -------- | ---------------- | -------------------------------- | --------- |
| A1       | EfficientNet-B3  | Crop embedding + TTA             | 0.243     |
| A2       | EfficientNetV2-S | Multi-task crop guidance         | **0.217** |
| A3       | DINOv2-Small     | Ordinal loss + SWA               | 0.259     |
| A4       | EfficientNetV2-S | HSV segmentation                 | 0.258     |
| A5       | EfficientNetV2-S | Per-crop heads + 4-channel input | 0.482     |
| A6       | EfficientNet-B4  | Direct regression baseline       | 0.622     |

---

# 🏆 Best Approach — A2

The strongest model combined:

* High-resolution inputs
* EfficientNetV2-S backbone
* Auxiliary crop supervision
* Implicit crop-aware feature learning
* TTA-based robust inference

## Why It Worked Best

### High Resolution

384×384 images preserved:

* Small leaves
* Fine boundaries
* Early growth-stage structures

### Implicit Multi-Task Learning

The auxiliary crop classifier encouraged:

* Better feature disentanglement
* Morphology-aware representations
* Cross-crop generalization

without introducing inference dependencies.

---

# 🔍 Important Insight — Multi-View Aggregation

We also explored the idea of:

```math
f({I_θ1, I_θ2, ..., I_θk}) = y
```

where multiple viewpoints of the same plant are aggregated.

## Why It Is Powerful

* Resolves occlusion naturally
* Provides richer geometric context
* Reduces viewpoint variance

## Why It Was Not Used

The competition strictly required:

✅ Single-image inference

Thus, multi-view architectures would not generalize to leaderboard conditions.

---

# 🛠️ Tech Stack

* Python
* PyTorch
* timm
* OpenCV
* scikit-learn
* NumPy
* Pandas
* torchvision
* Pillow
* DINOv2
* EfficientNet

---

# 📈 Key Learnings

✅ Higher image resolution consistently improved performance.

✅ Auxiliary supervision was critical for robust generalization.

✅ Vision Transformers showed promise but remained data-limited.

✅ Classical CV preprocessing remained surprisingly competitive.

✅ Increasing complexity does not necessarily improve performance.

---

# 🔮 Future Improvements

Potential future directions include:

* Ensemble A2 + A4
* Semi-supervised learning
* Self-supervised pretraining
* Better occlusion reasoning
* Cross-crop augmentation strategies
* Larger and more diverse datasets
* Lightweight deployment optimization

---

# ⭐ Conclusion

This project explored a wide spectrum of modern deep learning techniques for single-image leaf counting.

From CNN baselines to Vision Transformers and segmentation-assisted pipelines, the experiments demonstrated that:

* Careful architectural design matters more than raw complexity
* Multi-task supervision significantly improves representation quality
* Higher-resolution inputs are crucial for plant phenotyping
* Robust augmentation and TTA substantially improve generalization

The final EfficientNetV2-S multi-task pipeline achieved the best validation MAE of **0.217**, providing a strong and practical solution for real-world plant phenotyping tasks.
