<div align="center">

```
                                                                                                          ██████╗  █████╗ ██████╗ ███╗   ███╗ ██████╗ ███████╗██╗  ██╗
                                                                                                          ██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔═══██╗██╔════╝██║  ██║
                                                                                                          ██████╔╝███████║██║  ██║██╔████╔██║██║   ██║███████╗███████║
                                                                                                          ██╔══██╗██╔══██║██║  ██║██║╚██╔╝██║██║   ██║╚════██║██╔══██║
                                                                                                          ██████╔╝██║  ██║██████╔╝██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║
                                                                                                          ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
                                                                                                                                   C O D E R S
```

# 🏜️ Offroad Semantic Segmentation
### Duality AI GHR 2.0 Hackathon 2025

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![DINOv2](https://img.shields.io/badge/DINOv2-ViT--B/14-0068C1?style=for-the-badge)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Pixel-perfect desert terrain understanding using self-supervised vision transformers*

</div>

---

## 📊 Results at a Glance

| Metric | Score |
|--------|-------|
| **Val mIoU** | `0.5668` |
| **mAP@50** | — |
| **mAP@50:95** | — |
| **Backbone** | DINOv2 ViT-B/14 (86M params) |
| **Head** | FPN Decoder (4.3M params) |
| **Training Images** | 2,857 |
| **Test Images** | 1,002 |
| **Classes** | 11 |
| **Epochs** | 15 |
| **GPU** | NVIDIA T4 |

---

## 🏗️ Architecture

```
Input Image (952×532×3)
        │
        ▼
┌─────────────────────────────────────────┐
│         DINOv2 ViT-B/14 Backbone        │
│              86M Parameters             │
│                                         │
│  Block 2  ──► Stage 0 (texture/edges)   │
│  Block 5  ──► Stage 1 (local structure) │
│  Block 8  ──► Stage 2 (semantic parts)  │
│  Block 11 ──► Stage 3 (global context)  │
│                                         │
│  Blocks 6–11 fine-tuned @ lr=3e-5       │
└──────────────┬──────────────────────────┘
               │  4× (B, 2584, 768) feature maps
               ▼
┌─────────────────────────────────────────┐
│           FPN Decoder Head              │
│              4.3M Parameters            │
│                                         │
│  Lateral projections (768 → 256 each)   │
│  Top-down fusion (deep guides shallow)  │
│  3× Progressive upsampling (2×, 2×, 2×) │
│  Dropout(0.1) + 1×1 Classifier          │
└──────────────┬──────────────────────────┘
               │  (B, 11, H, W)
               ▼
┌─────────────────────────────────────────┐
│    Bilinear Interpolation to 952×532    │
└──────────────┬──────────────────────────┘
               ▼
     Output Mask (11 classes, per pixel)
```

---

## 🎨 Class Definitions

| ID | Class | Color | Weight |
|----|-------|-------|--------|
| 0 | Background | ⬛ `#0F0F0F` | 0.4× |
| 1 | Trees | 🟩 `#228B22` | 1.0× |
| 2 | Lush Bushes | 🟢 `#00D26E` | 1.2× |
| 3 | Dry Grass | 🟡 `#D2B478` | 1.0× |
| 4 | Dry Bushes | 🟫 `#A06428` | 2.0× |
| 5 | Ground Clutter | 🔘 `#787888` | 3.0× |
| 6 | Flowers | 🌸 `#FF64B4` | 4.0× |
| 7 | Logs | 🟤 `#5A3719` | 4.0× |
| 8 | Rocks | ⚪ `#B4AFAA` | 2.0× |
| 9 | Landscape | 🟧 `#C2AA6E` | 0.4× |
| 10 | Sky | 🔵 `#64B4F0` | 0.4× |

---

## 🔧 Key Improvements

### 1. 🧠 FPN Multi-Scale Decoder
Taps into 4 intermediate DINOv2 transformer blocks simultaneously. A top-down pathway merges deep semantic context with shallow texture features — producing sharper class boundaries than a single-layer decoder.

### 2. 🎯 Focal Loss (γ=2)
Down-weights easy, well-classified pixels so the gradient budget concentrates on hard misclassifications like Logs and Ground Clutter. Combined with per-class weights for rare classes.

### 3. 🔓 Deeper Backbone Fine-Tuning
Blocks 6–11 unfrozen (doubled from original 9–11 only). Gradient clipping at `max_norm=1.0` keeps training stable while allowing richer domain adaptation to desert terrain.

### 4. 📈 OneCycleLR Scheduler
LR ramps up for the first 30% of training then anneals sharply — finds a better minimum in fewer epochs vs CosineAnnealingLR.

### 5. 🔄 Test-Time Augmentation (TTA)
Validation averages original and horizontally-flipped predictions. Zero training cost, free +1–2 mIoU at inference time.

### 6. 🎲 Richer Augmentation
Random rotation (±10°), resized crop (60–100% zoom), Gaussian blur, color jitter, and random grayscale — all synchronized between image and mask.

---

## ⚙️ Training Configuration

```python
# Backbone
backbone     = "DINOv2 ViT-B/14"
unfrozen     = "blocks 6–11"
embed_dim    = 768

# Head
fpn_dim      = 256
decoder_ups  = 3   # 3× progressive 2× upsample

# Training
epochs       = 15
batch_size   = 2
lr_head      = 3e-4
lr_backbone  = 3e-5
weight_decay = 1e-4
optimizer    = "AdamW"
scheduler    = "OneCycleLR (pct_start=0.3)"
grad_clip    = 1.0
loss         = "0.6 × FocalLoss(γ=2) + 0.4 × DiceLoss"
label_smooth = 0.05
resolution   = "952 × 532"
amp          = True
```

---

## 🚀 Setup & Usage

### Installation
```bash
git clone https://github.com/pratyushmathur05/badmosh-coders.git
cd badmosh-coders
pip install torch torchvision tqdm matplotlib pillow
```

### Training
```bash
# Update paths in train_fpn.py:
# TRAIN_DIR, VAL_DIR, RUNS_DIR

python train_fpn.py
```

### Testing
```bash
# Update paths in test.py:
# TEST_DIR, RUNS_DIR, OUTPUT_DIR

python test.py
```

### Outputs
```
test_outputs/
├── summary_card.png             ← mIoU + mAP gauges overview
├── iou_bar_chart.png            ← per-class IoU horizontal bars
├── map_chart.png                ← per-class AP@50
├── class_performance_tiles.png  ← circular gauge per class
├── iou_distribution.png         ← histogram of per-image scores
├── predictions/
│   └── *_pred.png               ← input | prediction | GT grids
└── test_results.txt             ← full numeric results
```

---

## 📁 Repository Structure

```
badmosh-coders/
├── train_fpn.py          ← main training script (FPN + DINOv2)
├── test.py               ← inference + mIoU + mAP50 evaluation
├── index.html            ← hackathon presentation website
├── document.txt          ← full technical report
└── README.md
```

---

## 🏆 Hackathon

**Event:** Duality AI GHR 2.0 Hackathon 2025
**Team:** Badmosh Coders
**Platform:** Falcon Digital Twin (synthetic desert data)
**Stack:** PyTorch · DINOv2 · Google Colab T4 · Falcon Platform

---

<div align="center">

**Built with ◈ by Badmosh Coders**
*DINOv2 ViT-B/14 · FPN Decoder · 11 Classes · Synthetic Desert Data*

</div>
