# Duality AI Offroad Semantic Scene Segmentation

**Hackathon:** Duality AI Offroad Autonomy Segmentation Challenge  
**Model:** DINOv2 ViT-S/14 + Custom Segmentation Head  
**Framework:** PyTorch  
**Platform:** Google Colab (CUDA GPU)

---

## Repository Structure

```
duality-segmentation/
│
├── README.md                        ← You are here
│
├── train.py                         ← Main training script
│
├── test.py                          ← Inference script for test images
│
├── documentation.txt                ← Full detailed project report
│
├── requirements.txt                 ← Python dependencies
│
├── config/
│   └── config.yaml                  ← Training hyperparameters and paths
│
├── scripts/
│   └── visualize.py                 ← Script to visualize segmentation output
│
├── runs/                            ← Generated after training (not committed)
│   ├── best_segmentation_head.pth   ← Best model weights ✅ SUBMIT THIS
│   ├── checkpoint_latest.pth        ← Latest training checkpoint
│   ├── checkpoint_epoch_5.pth       ← Milestone checkpoint
│   ├── checkpoint_epoch_10.pth      ← Milestone checkpoint
│   ├── checkpoint_epoch_15.pth      ← Milestone checkpoint
│   ├── checkpoint_epoch_20.pth      ← Milestone checkpoint
│   ├── loss_curve.png               ← Training loss graph
│   └── iou_curve.png                ← Validation mIoU graph
│
└── .gitignore                       ← Excludes large files from git
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/duality-segmentation.git
cd duality-segmentation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Run these cells in Google Colab:
```python
import gdown

gdown.download("https://drive.google.com/uc?id=1cvwY89SsWYWmP86SVyYd4sDMz_sZ5RmU",
               "/content/dataset1.zip", quiet=False)

gdown.download("https://drive.google.com/uc?id=1-6MC_mG3ra8Faw85jfoP55A2Jgixccud",
               "/content/dataset2.zip", quiet=False)

import zipfile
with zipfile.ZipFile("/content/dataset1.zip", "r") as z:
    z.extractall("/content/dataset1")
with zipfile.ZipFile("/content/dataset2.zip", "r") as z:
    z.extractall("/content/dataset2")
```

### 4. Train the Model
```bash
python train.py
```

Training will automatically resume from the last checkpoint if the session
disconnects. Checkpoints are saved to `/content/runs/` after every epoch.

### 5. Run Inference on Test Images
```bash
python test.py
```

Predictions will be saved to `/content/runs/predictions/`

---

## Dataset Structure (after extraction)

```
/content/
├── dataset1/
│   └── Offroad_Segmentation_testImages/
│       └── Segmentation/            ← Test images (no labels)
│
└── dataset2/
    └── Offroad_Segmentation_Training_Dataset/
        ├── train/
        │   ├── Color_Images/        ← RGB training images
        │   └── Segmentation/        ← Segmentation masks
        └── val/
            ├── Color_Images/        ← RGB validation images
            └── Segmentation/        ← Segmentation masks
```

---

## Class Labels

| Class Index | Raw Pixel Value | Class Name      |
|-------------|-----------------|-----------------|
| 0           | 0               | Background      |
| 1           | 100             | Trees           |
| 2           | 200             | Lush Bushes     |
| 3           | 300             | Dry Grass       |
| 4           | 500             | Dry Bushes      |
| 5           | 550             | Ground Clutter  |
| 6           | 600             | Flowers         |
| 7           | 700             | Logs            |
| 8           | 800             | Rocks           |
| 9           | 7100            | Landscape       |
| 10          | 10000           | Sky             |

---

## Model Architecture

```
Input Image (3 x 266 x 476)
        ↓
DINOv2 ViT-S/14 Backbone [FROZEN]
  - Splits image into 14x14 patches
  - Outputs patch embeddings (384-dim)
        ↓
Segmentation Head [TRAINABLE]
  - Conv2d(384 → 256) + BatchNorm + GELU
  - Conv2d(256 → 256) + BatchNorm + GELU
  - Conv2d(256 → 11)
        ↓
Bilinear Upsample → (11 x 266 x 476)
        ↓
Argmax → Per-pixel class prediction
```

---

## Training Configuration

| Parameter      | Value                        |
|----------------|------------------------------|
| Epochs         | 20                           |
| Batch Size     | 4                            |
| Learning Rate  | 3e-4 (CosineAnnealingLR)     |
| Optimizer      | AdamW (weight_decay=1e-4)    |
| Loss Function  | CrossEntropy + Dice Loss     |
| Image Size     | 476 x 266                    |
| AMP            | Enabled (GPU only)           |

---

## Results

| Metric         | Score  |
|----------------|--------|
| Best Val mIoU  | [fill after training] |
| Final Epoch    | 20     |

Loss and IoU curves are saved to `runs/loss_curve.png` and `runs/iou_curve.png`

---

## Requirements

See `requirements.txt`:
```
torch>=1.13
torchvision>=0.14
numpy>=1.21
Pillow>=9.0
tqdm>=4.64
matplotlib>=3.5
PyYAML>=6.0
```

---

## Notes

- Do NOT use test images for training — this will result in disqualification
- The `runs/` folder is excluded from git via `.gitignore` (too large)
- Upload `best_segmentation_head.pth` separately or via Git LFS
- If resuming training after a Colab disconnect, just re-run `train.py` —
  it will automatically detect and load `checkpoint_latest.pth`

---

## Acknowledgements

- [Duality AI](https://falcon.duality.ai) for the dataset and challenge
- [Meta AI DINOv2](https://github.com/facebookresearch/dinov2) for the backbone
