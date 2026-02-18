import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.switch_backend('Agg')

# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "/content/dataset2/Offroad_Segmentation_Training_Dataset/train"
VAL_DIR   = "/content/dataset2/Offroad_Segmentation_Training_Dataset/val"
RUNS_DIR  = "/content/runs"

# ============================================================
# Class Mapping
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

CLASS_WEIGHTS = [
    0.4,   # 0  Background
    1.0,   # 1  Trees
    1.2,   # 2  Lush Bushes
    1.0,   # 3  Dry Grass
    2.0,   # 4  Dry Bushes
    3.0,   # 5  Ground Clutter
    4.0,   # 6  Flowers
    4.0,   # 7  Logs
    2.0,   # 8  Rocks
    0.4,   # 9  Landscape
    0.4,   # 10 Sky
]


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================
# Dataset
# ============================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_size, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.augment   = augment
        self.h, self.w = img_size
        self.data_ids  = self._get_valid_ids()
        print(f"  Found {len(self.data_ids)} valid pairs in {data_dir}  "
              f"[augment={augment}]")

    def _get_valid_ids(self):
        all_images = os.listdir(self.image_dir)
        mask_files = set(os.listdir(self.masks_dir))
        valid = []
        for fname in all_images:
            base = os.path.splitext(fname)[0]
            if fname in mask_files:
                valid.append((fname, fname))
            else:
                match = next(
                    (m for m in mask_files if os.path.splitext(m)[0] == base),
                    None
                )
                if match:
                    valid.append((fname, match))
                else:
                    print(f"  [Warning] No mask for {fname} — skipping.")
        return valid

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        img_name, mask_name = self.data_ids[idx]
        image = Image.open(
            os.path.join(self.image_dir, img_name)).convert("RGB")
        mask  = Image.open(
            os.path.join(self.masks_dir, mask_name))
        mask  = convert_mask(mask)

        image = TF.resize(image, (self.h, self.w))
        mask  = TF.resize(mask,  (self.h, self.w),
                          interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            if torch.rand(1) > 0.5:
                image = TF.hflip(image);  mask = TF.hflip(mask)

            if torch.rand(1) > 0.5:
                image = TF.vflip(image);  mask = TF.vflip(mask)

            if torch.rand(1) > 0.5:
                angle = float(torch.empty(1).uniform_(-10, 10))
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask,  angle,
                                  interpolation=transforms.InterpolationMode.NEAREST)

            if torch.rand(1) > 0.5:
                i, j, ch, cw = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.6, 1.0), ratio=(3/4, 4/3))
                image = TF.resized_crop(image, i, j, ch, cw, size=(self.h, self.w))
                mask  = TF.resized_crop(mask,  i, j, ch, cw, size=(self.h, self.w),
                                        interpolation=transforms.InterpolationMode.NEAREST)

            image = transforms.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.15)(image)

            if torch.rand(1) > 0.85:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

            if torch.rand(1) > 0.5:
                radius = int(torch.randint(1, 4, (1,)).item()) * 2 + 1
                image = TF.gaussian_blur(image, kernel_size=radius)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        mask  = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================
# DINOv2 Multi-Scale Feature Extractor
# ============================================================

class DINOv2MultiScale(nn.Module):
    """
    Wraps DINOv2 ViT-B/14 and returns patch-token feature maps
    from 4 intermediate transformer blocks via forward hooks.

    Extraction points (ViT-B has 12 blocks, 0-indexed):
        Stage 0 → block  2  (early,  low-level texture)
        Stage 1 → block  5  (mid-low, local structure)
        Stage 2 → block  8  (mid-high, semantic parts)
        Stage 3 → block 11  (deep,    global semantics)

    All stages share the SAME spatial resolution (H/14 × W/14)
    but differ in semantic depth — the FPN decoder exploits this.
    """
    HOOK_BLOCKS = [2, 5, 8, 11]

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self._features = {}
        self._hooks    = []
        self._register_hooks()

    def _register_hooks(self):
        for stage, block_idx in enumerate(self.HOOK_BLOCKS):
            block = self.backbone.blocks[block_idx]
            # Capture the output of each target block
            handle = block.register_forward_hook(
                lambda module, inp, out, s=stage: self._save(s, out)
            )
            self._hooks.append(handle)

    def _save(self, stage, output):
        # DINOv2 block output: (B, 1+N, C)  — strip CLS token
        self._features[stage] = output[:, 1:, :]   # (B, N, C)

    def forward(self, x):
        self._features.clear()
        self.backbone.forward_features(x)           # triggers all hooks
        # Return in shallow→deep order: [s0, s1, s2, s3]
        return [self._features[i] for i in range(len(self.HOOK_BLOCKS))]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ============================================================
# Building Blocks
# ============================================================

class ConvBnGelu(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=p),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


# ============================================================
# FPN Segmentation Decoder
# ============================================================

class FPNSegmentationHead(nn.Module):
    """
    Feature Pyramid Network decoder for ViT patch tokens.

    Architecture:
        1. Lateral projections  — map each stage's C-dim to fpn_dim
        2. Top-down pathway     — merge deep→shallow with element-wise add
        3. Progressive upsampling decoder  — 8× upsample in 3 steps
        4. Final classifier

                  Stage 3 (deepest)
                       │  lateral_proj[3] → P3 (fpn_dim)
                       ↓
              upsample + add
                  Stage 2  lateral_proj[2] → P2
                       ↓
              upsample + add
                  Stage 1  lateral_proj[1] → P1
                       ↓
              upsample + add
                  Stage 0  lateral_proj[0] → P0
                       │
                  decoder (8× upsample)
                       │
                  classifier
    """

    def __init__(self, embed_dim, out_channels, tokenW, tokenH,
                 fpn_dim=256):
        super().__init__()
        self.H       = tokenH
        self.W       = tokenW
        self.fpn_dim = fpn_dim

        # ── Lateral projections (1×1 conv, one per stage) ──
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, fpn_dim, kernel_size=1),
                nn.BatchNorm2d(fpn_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        # ── Top-down refinement convs (after add) ──────────
        self.td_conv = nn.ModuleList([
            ConvBnGelu(fpn_dim, fpn_dim)
            for _ in range(3)      # 3 merge steps (3←2, 2←1, 1←0)
        ])

        # ── Progressive decoder (8× total upsampling) ──────
        # Each stage doubles spatial resolution and halves channels
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnGelu(fpn_dim, 256),
            ConvBnGelu(256,     256),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnGelu(256, 128),
            ConvBnGelu(128, 128),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnGelu(128, 64),
            ConvBnGelu(64,  64),
        )

        self.dropout    = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(64, out_channels, kernel_size=1)

    # ── Helper: reshape N patch tokens → spatial grid ──────
    def _to_spatial(self, tokens):
        B, N, C = tokens.shape
        return tokens.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

    def forward(self, stage_features):
        """
        stage_features: list of 4 tensors, each (B, N, C)
                        ordered shallow→deep: [s0, s1, s2, s3]
        """
        # 1. Lateral projections → spatial feature maps
        laterals = [
            self.lateral[i](self._to_spatial(stage_features[i]))
            for i in range(4)
        ]  # all: (B, fpn_dim, H, W)

        # 2. Top-down pathway: start from deepest (index 3) → shallowest (0)
        #    P3 = lateral[3]  (no top-down input at the top)
        #    P2 = td_conv(lateral[2] + upsample(P3))
        #    P1 = td_conv(lateral[1] + upsample(P2))
        #    P0 = td_conv(lateral[0] + upsample(P1))
        p = laterals[3]
        for i in range(2, -1, -1):           # i = 2, 1, 0
            p_up = F.interpolate(p, size=laterals[i].shape[2:],
                                 mode='bilinear', align_corners=False)
            p    = self.td_conv[2 - i](laterals[i] + p_up)

        # p is now P0: the finest merged feature map (B, fpn_dim, H, W)

        # 3. Progressive 8× upsampling decoder
        x = self.up1(p)    # → (B, 256, 2H, 2W)
        x = self.up2(x)    # → (B, 128, 4H, 4W)
        x = self.up3(x)    # → (B,  64, 8H, 8W)
        x = self.dropout(x)
        return self.classifier(x)   # → (B, n_classes, 8H, 8W)


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets,
                                weight=self.weight,
                                label_smoothing=self.label_smoothing,
                                reduction='none')
        pt    = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        t_oh  = F.one_hot(targets, num_classes=n_classes).permute(0,3,1,2).float()
        inter = (probs * t_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + t_oh.sum(dim=(2, 3))
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ============================================================
# IoU Metric
# ============================================================

def compute_iou(pred_probs, target):
    """pred_probs can be raw logits or averaged softmax probs."""
    pred = torch.argmax(pred_probs, dim=1)
    ious = {}
    for cls in range(n_classes):
        p = pred   == cls
        t = target == cls
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union == 0:
            continue
        ious[cls] = (inter / union).item()
    mean_iou = np.mean(list(ious.values())) if ious else 0.0
    return mean_iou, ious


# ============================================================
# Test-Time Augmentation
# ============================================================

def tta_forward(extractor, head, imgs, original_size):
    def single_pass(x):
        feats  = extractor(x)
        logits = head(feats)
        logits = F.interpolate(logits, size=original_size,
                               mode='bilinear', align_corners=False)
        return F.softmax(logits, dim=1)

    probs      = single_pass(imgs)
    probs_hflip = single_pass(torch.flip(imgs, dims=[3]))
    probs_hflip = torch.flip(probs_hflip, dims=[3])   # un-flip

    return (probs + probs_hflip) / 2.0


# ============================================================
# Save Plots
# ============================================================

def save_plots(train_losses, val_ious, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png')); plt.close()

    plt.figure()
    plt.plot(epochs, val_ious, marker='o', color='green', label='Val mIoU')
    plt.xlabel('Epoch'); plt.ylabel('Mean IoU')
    plt.title('Validation mIoU'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curve.png')); plt.close()

    print(f"Plots saved to {output_dir}")


# ============================================================
# Checkpoint Helpers
# ============================================================

def save_checkpoint(path, epoch, backbone, extractor, head,
                    optimizer, scheduler, scaler,
                    best_iou, train_loss_history, val_iou_history):
    torch.save({
        "epoch":              epoch,
        "backbone_state":     backbone.state_dict(),
        "head_state":         head.state_dict(),
        "optimizer_state":    optimizer.state_dict(),
        "scheduler_state":    scheduler.state_dict(),
        "scaler_state":       scaler.state_dict(),
        "best_iou":           best_iou,
        "train_loss_history": train_loss_history,
        "val_iou_history":    val_iou_history,
    }, path)


def load_checkpoint(path, backbone, head, optimizer,
                    scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Backbone weights are always compatible
    backbone.load_state_dict(ckpt["backbone_state"])
    print("  >> Backbone weights loaded.")

    # Head key changed between old and new script; architecture may also differ
    head_state_key = "head_state" if "head_state" in ckpt else "model_state"
    head_loaded = False
    if head_state_key in ckpt:
        try:
            head.load_state_dict(ckpt[head_state_key])
            print("  >> FPN head weights loaded.")
            head_loaded = True
        except RuntimeError as e:
            print(f"  >> [Warning] Head weights incompatible (old architecture) — "
                  f"training head from scratch. Reason: {e}")
    else:
        print("  >> [Warning] No head weights in checkpoint — training head from scratch.")

    # Only restore optimizer/scheduler/scaler when head was fully loaded;
    # otherwise the step counts are wrong for the new head.
    if head_loaded:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            print("  >> Optimizer / scheduler / scaler state restored.")
        except Exception as e:
            print(f"  >> [Warning] Could not restore optimizer state — resetting. Reason: {e}")
    else:
        print("  >> Optimizer / scheduler / scaler reset (head trained from scratch).")

    return (
        ckpt["epoch"] + 1,
        ckpt["best_iou"],
        ckpt.get("train_loss_history", []),
        ckpt.get("val_iou_history",    []),
    )


# ============================================================
# Main
# ============================================================

def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}  |  AMP: {use_amp}")

    os.makedirs(RUNS_DIR, exist_ok=True)

    # ── Hyperparameters ──────────────────────────────────────
    batch_size  = 2
    lr_head     = 3e-4
    lr_backbone = 3e-5
    n_epochs    = 15

    w = int((960 // 14) * 14)   # 952
    h = int((540 // 14) * 14)   # 532
    print(f"Input size: {w} × {h}  |  Patch grid: {w//14} × {h//14}")

    # ── Datasets ─────────────────────────────────────────────
    print("\nLoading datasets...")
    trainset = MaskDataset(TRAIN_DIR, img_size=(h, w), augment=True)
    valset   = MaskDataset(VAL_DIR,   img_size=(h, w), augment=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=use_amp)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=use_amp)

    # ── Backbone ─────────────────────────────────────────────
    print("\nLoading DINOv2 ViT-B/14 backbone...")
    backbone_raw = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone_raw.to(device)

    # Freeze all first
    for p in backbone_raw.parameters():
        p.requires_grad = False

    # Unfreeze last 6 blocks for fine-tuning
    for name, param in backbone_raw.named_parameters():
        if any(f"blocks.{i}." in name for i in range(6, 12)):
            param.requires_grad = True

    unfrozen = sum(p.numel() for p in backbone_raw.parameters() if p.requires_grad)
    total    = sum(p.numel() for p in backbone_raw.parameters())
    print(f"Backbone params: {total:,} total  |  {unfrozen:,} unfrozen (blocks 6–11)")

    # ── Multi-Scale Feature Extractor ────────────────────────
    extractor = DINOv2MultiScale(backbone_raw)

    # Probe embed dim
    backbone_raw.eval()
    with torch.no_grad():
        dummy  = torch.zeros(1, 3, h, w, device=device)
        stages = extractor(dummy)
        embed_dim = stages[0].shape[2]
    print(f"Embedding dim: {embed_dim}  |  "
          f"Feature stages: {[s.shape for s in stages]}")

    # ── FPN Decoder ──────────────────────────────────────────
    head = FPNSegmentationHead(
        embed_dim   = embed_dim,
        out_channels= n_classes,
        tokenW      = w // 14,
        tokenH      = h // 14,
        fpn_dim     = 256,
    ).to(device)

    head_params = sum(p.numel() for p in head.parameters())
    print(f"FPN head params: {head_params:,}")

    # ── Loss ─────────────────────────────────────────────────
    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
    focal_loss    = FocalLoss(gamma=2.0, weight=class_weights,
                              label_smoothing=0.05)
    dice_loss     = DiceLoss()

    def combined_loss(logits, labels):
        return 0.6 * focal_loss(logits, labels) + 0.4 * dice_loss(logits, labels)

    # ── Optimizer ────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": [p for p in backbone_raw.parameters() if p.requires_grad],
         "lr": lr_backbone},
        {"params": head.parameters(),
         "lr": lr_head},
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = [lr_backbone, lr_head],
        steps_per_epoch = steps_per_epoch,
        epochs          = n_epochs,
        pct_start       = 0.3,
        anneal_strategy = 'cos',
        div_factor      = 10,
        final_div_factor= 1e3,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ───────────────────────────────────────────────
    start_epoch        = 0
    best_iou           = 0.0
    train_loss_history = []
    val_iou_history    = []

    latest_ckpt = os.path.join(RUNS_DIR, "checkpoint_latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"\nCheckpoint found — resuming from {latest_ckpt}")
        start_epoch, best_iou, train_loss_history, val_iou_history = \
            load_checkpoint(latest_ckpt, backbone_raw, head,
                            optimizer, scheduler, scaler, device)
        print(f"Resuming at epoch {start_epoch}  |  Best IoU: {best_iou:.4f}")
    else:
        print("\nNo checkpoint found — starting fresh.")

    # ── Training Loop ─────────────────────────────────────────
    print(f"\nStarting training for {n_epochs} epochs...\n")

    for epoch in range(start_epoch, n_epochs):

        # ─── Train ────────────────────────────────────────────
        backbone_raw.train()
        head.train()
        train_loss = 0.0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                stage_feats = extractor(imgs)
                logits      = head(stage_feats)
                logits      = F.interpolate(logits, size=imgs.shape[2:],
                                            mode='bilinear', align_corners=False)
                loss        = combined_loss(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone_raw.parameters()) + list(head.parameters()),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        # ─── Validate (with TTA) ──────────────────────────────
        backbone_raw.eval()
        head.eval()
        val_ious      = []
        per_class_iou = {i: [] for i in range(n_classes)}

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Epoch {epoch+1}/{n_epochs} [Val]  "):
                imgs   = imgs.to(device)
                labels = labels.to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    avg_probs = tta_forward(extractor, head, imgs,
                                            original_size=imgs.shape[2:])

                mean_iou, cls_ious = compute_iou(avg_probs, labels)
                val_ious.append(mean_iou)
                for cls, iou in cls_ious.items():
                    per_class_iou[cls].append(iou)

        mean_iou = float(np.mean(val_ious))
        val_iou_history.append(mean_iou)

        last_lrs = scheduler.get_last_lr()
        is_best  = mean_iou > best_iou
        print(f"\n  Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss   : {avg_loss:.4f}")
        print(f"  Val mIoU     : {mean_iou:.4f}  {'★ NEW BEST' if is_best else ''}")
        print(f"  LR (backbone): {last_lrs[0]:.7f}  |  LR (head): {last_lrs[1]:.6f}")
        print("  Per-class IoU:")
        for cls in range(n_classes):
            scores = per_class_iou[cls]
            if scores:
                print(f"    [{cls:2d}] {CLASS_NAMES[cls]:<16}: {np.mean(scores):.4f}")
            else:
                print(f"    [{cls:2d}] {CLASS_NAMES[cls]:<16}: N/A")

        # Save checkpoints
        save_checkpoint(
            os.path.join(RUNS_DIR, "checkpoint_latest.pth"),
            epoch, backbone_raw, extractor, head,
            optimizer, scheduler, scaler,
            best_iou, train_loss_history, val_iou_history
        )
        print("  >> Latest checkpoint saved.")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                os.path.join(RUNS_DIR, f"checkpoint_epoch_{epoch+1}.pth"),
                epoch, backbone_raw, extractor, head,
                optimizer, scheduler, scaler,
                best_iou, train_loss_history, val_iou_history
            )
            print(f"  >> Milestone checkpoint saved (epoch {epoch+1}).")

        if is_best:
            best_iou = mean_iou
            torch.save(head.state_dict(),
                       os.path.join(RUNS_DIR, "best_fpn_head.pth"))
            torch.save(backbone_raw.state_dict(),
                       os.path.join(RUNS_DIR, "best_backbone.pth"))
            print(f"  >> New best model saved! (mIoU: {best_iou:.4f})")

    # ── Done ──────────────────────────────────────────────────
    print(f"\nTraining complete!  Best Val mIoU: {best_iou:.4f}")
    save_plots(train_loss_history, val_iou_history, RUNS_DIR)

    gdrive_runs = "/content/drive/MyDrive/segmentation_runs"
    if os.path.exists("/content/drive/MyDrive"):
        import shutil
        shutil.copytree(RUNS_DIR, gdrive_runs, dirs_exist_ok=True)
        print(f"Runs backed up to Google Drive: {gdrive_runs}")


if __name__ == "__main__":
    main()
