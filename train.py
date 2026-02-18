"""
Segmentation Training Script
DINOv2 + Custom Segmentation Head
Duality AI Offroad Segmentation Hackathon

Features:
  - Correct 11-class value map (includes Flowers class 600)
  - Combined CrossEntropy + Dice Loss
  - CosineAnnealingLR scheduler
  - Mixed precision (AMP) on GPU, safe fallback on CPU
  - Auto checkpoint: resumes if session disconnects
  - Milestone checkpoints every 5 epochs
  - Loss + IoU plots saved to /content/runs/
"""

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.switch_backend('Agg')

# ============================================================
# Paths — update these if your folder structure is different
# ============================================================

TRAIN_DIR = "/content/dataset2/Offroad_Segmentation_Training_Dataset/train"
VAL_DIR   = "/content/dataset2/Offroad_Segmentation_Training_Dataset/val"
RUNS_DIR  = "/content/runs"

# ============================================================
# Class Mapping  (raw pixel value -> class index)
# ============================================================

value_map = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape", "Sky"
]

n_classes = len(value_map)  # 11


def convert_mask(mask):
    """Convert raw segmentation pixel values to class indices."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================
# Dataset
# ============================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = self._get_valid_ids()
        print(f"  Found {len(self.data_ids)} valid image-mask pairs in {data_dir}")

    def _get_valid_ids(self):
        """Pair images with masks, handling extension mismatches."""
        all_images = os.listdir(self.image_dir)
        mask_files = set(os.listdir(self.masks_dir))
        valid = []
        for fname in all_images:
            base = os.path.splitext(fname)[0]
            if fname in mask_files:
                valid.append((fname, fname))
            else:
                match = next(
                    (m for m in mask_files if os.path.splitext(m)[0] == base), None
                )
                if match:
                    valid.append((fname, match))
                else:
                    print(f"  [Warning] No mask found for: {fname} -- skipping.")
        return valid

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        img_name, mask_name = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, mask_name))
        mask  = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask  = self.mask_transform(mask) * 255

        return image, mask


# ============================================================
# Segmentation Head
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H = tokenH
        self.W = tokenW

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.block(x)


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = (
            F.one_hot(targets, num_classes=n_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ============================================================
# IoU Metric
# ============================================================

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(n_classes):
        pred_inds   = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


# ============================================================
# Save Training Plots
# ============================================================

def save_plots(train_losses, val_ious, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_ious, marker='o', color='green', label='Val mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Validation mIoU')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curve.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")


# ============================================================
# Checkpoint Helpers
# ============================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_iou, train_loss_history, val_iou_history):
    torch.save({
        "epoch":              epoch,
        "model_state":        model.state_dict(),
        "optimizer_state":    optimizer.state_dict(),
        "scheduler_state":    scheduler.state_dict(),
        "scaler_state":       scaler.state_dict(),
        "best_iou":           best_iou,
        "train_loss_history": train_loss_history,
        "val_iou_history":    val_iou_history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt["best_iou"],
        ckpt.get("train_loss_history", []),
        ckpt.get("val_iou_history", []),
    )


# ============================================================
# Main
# ============================================================

def main():

    # Device
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}  |  AMP: {use_amp}")

    os.makedirs(RUNS_DIR, exist_ok=True)

    # Hyperparameters
    batch_size = 4
    lr         = 3e-4
    n_epochs   = 20

    # Image size (must be divisible by 14 for DINOv2)
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Datasets
    print("\nLoading datasets...")
    trainset = MaskDataset(TRAIN_DIR, transform, mask_transform)
    valset   = MaskDataset(VAL_DIR,   transform, mask_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=use_amp)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=use_amp)

    # Backbone
    print("\nLoading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad = False   # freeze backbone

    # Probe embedding dim
    with torch.no_grad():
        sample    = next(iter(train_loader))[0][:1].to(device)
        probe     = backbone.forward_features(sample)["x_norm_patchtokens"]
        embed_dim = probe.shape[2]
    print(f"DINOv2 embedding dim: {embed_dim}")

    # Model
    model = SegmentationHead(
        in_channels=embed_dim,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14,
    ).to(device)

    # Loss, Optimizer, Scheduler, Scaler
    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint if available
    start_epoch        = 0
    best_iou           = 0.0
    train_loss_history = []
    val_iou_history    = []

    latest_ckpt = os.path.join(RUNS_DIR, "checkpoint_latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"\nCheckpoint found -- resuming from {latest_ckpt}")
        start_epoch, best_iou, train_loss_history, val_iou_history = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler, device
        )
        print(f"Resuming at epoch {start_epoch}  |  Best IoU so far: {best_iou:.4f}")
    else:
        print("\nNo checkpoint found -- starting fresh.")

    # Training Loop
    print(f"\nStarting training for {n_epochs} epochs...\n")

    for epoch in range(start_epoch, n_epochs):

        # Train
        model.train()
        train_loss = 0.0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(feats)
                logits = F.interpolate(logits, size=imgs.shape[2:],
                                       mode="bilinear", align_corners=False)
                loss = ce_loss(logits, labels) + dice_loss(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        # Validate
        model.eval()
        val_ious = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Epoch {epoch+1}/{n_epochs} [Val]  "):
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)
                feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(feats)
                    logits = F.interpolate(logits, size=imgs.shape[2:],
                                           mode="bilinear", align_corners=False)

                val_ious.append(compute_iou(logits, labels))

        mean_iou = float(np.mean(val_ious))
        val_iou_history.append(mean_iou)

        print(f"\n  Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  Val mIoU   : {mean_iou:.4f}")
        print(f"  LR         : {scheduler.get_last_lr()[0]:.6f}")

        # Save latest checkpoint every epoch
        save_checkpoint(
            os.path.join(RUNS_DIR, "checkpoint_latest.pth"),
            epoch, model, optimizer, scheduler, scaler,
            best_iou, train_loss_history, val_iou_history
        )
        print("  >> Latest checkpoint saved.")

        # Save milestone checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                os.path.join(RUNS_DIR, f"checkpoint_epoch_{epoch+1}.pth"),
                epoch, model, optimizer, scheduler, scaler,
                best_iou, train_loss_history, val_iou_history
            )
            print(f"  >> Milestone checkpoint saved (epoch {epoch+1}).")

        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                model.state_dict(),
                os.path.join(RUNS_DIR, "best_segmentation_head.pth")
            )
            print(f"  >> New best model saved! (IoU: {best_iou:.4f})")

    # Done
    print(f"\nTraining complete!  Best Val mIoU: {best_iou:.4f}")
    save_plots(train_loss_history, val_iou_history, RUNS_DIR)

    # Auto-backup runs to Google Drive if mounted
    gdrive_runs = "/content/drive/MyDrive/segmentation_runs"
    if os.path.exists("/content/drive/MyDrive"):
        import shutil
        shutil.copytree(RUNS_DIR, gdrive_runs, dirs_exist_ok=True)
        print(f"Runs backed up to Google Drive: {gdrive_runs}")


if __name__ == "__main__":
    main()
