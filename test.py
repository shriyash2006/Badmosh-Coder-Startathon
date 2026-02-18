"""
test.py — Duality AI Offroad Segmentation | Badmosh Coders
Metrics: mIoU + mAP50  |  Beautiful visualizations for report & README
"""

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Global dark theme ───────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  '#0D1117',
    'axes.facecolor':    '#0D1117',
    'text.color':        '#E6EDF3',
    'axes.labelcolor':   '#E6EDF3',
    'xtick.color':       '#8B949E',
    'ytick.color':       '#8B949E',
    'axes.edgecolor':    '#30363D',
    'grid.color':        '#21262D',
    'grid.alpha':        0.6,
})

DARK_BG  = '#0D1117'
PANEL_BG = '#161B22'
BORDER   = '#30363D'
BLUE     = '#58A6FF'
GREEN    = '#3FB950'
ORANGE   = '#F0883E'
RED      = '#F78166'
MUTED    = '#8B949E'
DIM      = '#484F58'

# ============================================================
# Paths
# ============================================================

TEST_DIR   = "/content/testImages/Offroad_Segmentation_testImages"
RUNS_DIR   = "/content/runs"
OUTPUT_DIR = "/content/test_outputs"

# ============================================================
# Classes & Colors
# ============================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape", "Sky"
]

CLASS_COLORS = np.array([
    [15,  15,  15],
    [34,  139, 34],
    [0,   210, 110],
    [210, 180, 120],
    [160, 100, 40],
    [120, 120, 140],
    [255, 100, 180],
    [90,  55,  25],
    [180, 175, 170],
    [194, 170, 110],
    [100, 180, 240],
], dtype=np.uint8)

n_classes = len(CLASS_NAMES)

IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # 0.50 → 0.95 for mAP


def convert_mask(mask):
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


def mask_to_color(pred_mask):
    return CLASS_COLORS[pred_mask]


# ============================================================
# IoU + mAP50 Metrics
# ============================================================

def compute_iou(pred, target):
    """Per-class IoU. Returns (mean_iou, {cls: iou})."""
    ious = {}
    for cls in range(n_classes):
        p, t  = pred == cls, target == cls
        inter = int((p & t).sum())
        union = int((p | t).sum())
        if union == 0:
            continue
        ious[cls] = inter / union
    return (np.mean(list(ious.values())) if ious else 0.0), ious


def compute_ap_per_class(cls_ious_list, iou_threshold=0.50):
    """
    Compute Average Precision for a single class at a given IoU threshold.

    For semantic segmentation we treat each image as a single 'detection':
      - TP if class is present in GT  AND  IoU >= threshold
      - FP if class is predicted      AND  (not in GT  OR  IoU < threshold)
      - FN if class is in GT          AND  not predicted (IoU == 0 or missing)

    We rank by predicted confidence (max softmax prob for that class) and
    compute the area under the Precision-Recall curve.

    Args:
        cls_ious_list : list of (iou, confidence, gt_present) per image
        iou_threshold : float, default 0.50
    Returns:
        ap : float in [0, 1]
    """
    if not cls_ious_list:
        return 0.0

    # Sort by confidence descending
    cls_ious_list = sorted(cls_ious_list, key=lambda x: x[1], reverse=True)

    tp_list, fp_list = [], []
    n_gt = sum(1 for _, _, gt in cls_ious_list if gt)

    if n_gt == 0:
        return 0.0

    for iou, conf, gt_present in cls_ious_list:
        if gt_present and iou >= iou_threshold:
            tp_list.append(1); fp_list.append(0)
        else:
            tp_list.append(0); fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls    = tp_cum / (n_gt + 1e-9)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Append sentinel values
    recalls    = np.concatenate([[0], recalls,    [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # Monotonic precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Area under PR curve
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap  = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


def compute_map(per_class_ap_data, iou_threshold=0.50):
    """
    Compute mAP at a given IoU threshold across all classes.
    per_class_ap_data: dict {cls_idx: [(iou, confidence, gt_present), ...]}
    """
    aps = []
    for cls in range(n_classes):
        data = per_class_ap_data.get(cls, [])
        ap   = compute_ap_per_class(data, iou_threshold)
        aps.append(ap)
    return float(np.mean(aps)), aps


# ============================================================
# Model (identical architecture to train.py)
# ============================================================

class DINOv2MultiScale(nn.Module):
    HOOK_BLOCKS = [2, 5, 8, 11]

    def __init__(self, backbone):
        super().__init__()
        self.backbone  = backbone
        self._features = {}
        self._hooks    = []
        for stage, bi in enumerate(self.HOOK_BLOCKS):
            h = backbone.blocks[bi].register_forward_hook(
                lambda m, i, o, s=stage: self._features.update({s: o[:, 1:, :]}))
            self._hooks.append(h)

    def forward(self, x):
        self._features.clear()
        self.backbone.forward_features(x)
        return [self._features[i] for i in range(4)]


class ConvBnGelu(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p),
            nn.BatchNorm2d(oc), nn.GELU())

    def forward(self, x):
        return self.block(x)


class FPNSegmentationHead(nn.Module):
    def __init__(self, embed_dim, out_channels, tokenW, tokenH, fpn_dim=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, fpn_dim, 1),
                          nn.BatchNorm2d(fpn_dim), nn.GELU())
            for _ in range(4)])
        self.td_conv = nn.ModuleList([ConvBnGelu(fpn_dim, fpn_dim) for _ in range(3)])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnGelu(fpn_dim, 256), ConvBnGelu(256, 256))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnGelu(256, 128), ConvBnGelu(128, 128))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnGelu(128, 64), ConvBnGelu(64, 64))
        self.dropout    = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(64, out_channels, 1)

    def _to_spatial(self, t):
        B, N, C = t.shape
        return t.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

    def forward(self, sf):
        lats = [self.lateral[i](self._to_spatial(sf[i])) for i in range(4)]
        p = lats[3]
        for i in range(2, -1, -1):
            p = self.td_conv[2 - i](lats[i] + F.interpolate(
                p, size=lats[i].shape[2:], mode='bilinear', align_corners=False))
        return self.classifier(self.dropout(self.up3(self.up2(self.up1(p)))))


# ============================================================
# Visualization Functions
# ============================================================

def save_prediction_grid(image, color_mask, gt_color, iou, out_path):
    has_gt = gt_color is not None
    ncols  = 3 if has_gt else 2
    fig    = plt.figure(figsize=(7 * ncols, 7), facecolor=DARK_BG)
    iou_str = f"  |  IoU: {iou:.3f}" if iou is not None else ""
    fig.suptitle(f"Badmosh Coders — Offroad Segmentation{iou_str}",
                 fontsize=14, color=BLUE, fontweight='bold', y=1.01)

    items = ([(image, "Input Image", MUTED),
              (color_mask, "Predicted Segmentation", BLUE),
              (gt_color, "Ground Truth", GREEN)]
             if has_gt else
             [(image, "Input Image", MUTED),
              (color_mask, "Predicted Segmentation", BLUE)])

    for idx, (img_data, title, color) in enumerate(items):
        ax = fig.add_subplot(1, ncols, idx + 1)
        ax.set_facecolor(DARK_BG)
        ax.imshow(img_data)
        ax.set_title(title, color=color, fontsize=12, pad=10)
        ax.axis('off')

    patches = [mpatches.Patch(facecolor=[c / 255 for c in CLASS_COLORS[i]],
                              label=CLASS_NAMES[i],
                              edgecolor=BORDER, linewidth=0.5)
               for i in range(n_classes)]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), facecolor=PANEL_BG,
               edgecolor=BORDER, labelcolor='#E6EDF3', framealpha=0.95)
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()


def save_iou_bar_chart(per_class_ious, mean_iou, out_path):
    classes, scores = [], []
    for cls in range(n_classes):
        vals = per_class_ious[cls]
        if vals:
            classes.append(CLASS_NAMES[cls])
            scores.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    y_pos  = np.arange(len(classes))
    colors = [CLASS_COLORS[CLASS_NAMES.index(c)] / 255.0 for c in classes]

    ax.barh(y_pos, scores, color=colors, height=0.85, alpha=0.12)
    bars = ax.barh(y_pos, scores, color=colors, height=0.62,
                   edgecolor=BORDER, linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(min(score + 0.013, 0.96), bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', ha='left',
                color='#E6EDF3', fontsize=10, fontweight='bold')

    ax.axvline(mean_iou, color=RED, linewidth=2, linestyle='--', alpha=0.9, zorder=5)
    ax.text(mean_iou + 0.008, len(classes) - 0.6,
            f'Mean mIoU: {mean_iou:.3f}', color=RED, fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=11, color='#E6EDF3')
    ax.set_xlabel('IoU Score', fontsize=12, color=MUTED, labelpad=10)
    ax.set_xlim(0, 1.08)
    ax.set_title('Per-Class IoU — Test Set Results',
                 fontsize=16, color=BLUE, fontweight='bold', pad=20)
    ax.grid(axis='x', color='#21262D', linewidth=0.8, alpha=0.7)
    ax.tick_params(colors=MUTED)
    fig.text(0.5, -0.02,
             'Badmosh Coders  ·  DINOv2 ViT-B/14 + FPN  ·  Duality AI GHR 2.0 Hackathon',
             ha='center', fontsize=9, color=DIM)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out_path}")


def save_map_chart(per_class_aps, map50, map50_95, out_path):
    """Bar chart showing per-class AP@50 alongside mAP summary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG,
                                   gridspec_kw={'width_ratios': [2.5, 1]})

    # ── Per-class AP bars ──────────────────────────────────
    ax1.set_facecolor(DARK_BG)
    y_pos  = np.arange(n_classes)
    colors = [CLASS_COLORS[i] / 255.0 for i in range(n_classes)]

    ax1.barh(y_pos, per_class_aps, color=colors, height=0.85, alpha=0.12)
    bars = ax1.barh(y_pos, per_class_aps, color=colors, height=0.62,
                    edgecolor=BORDER, linewidth=0.5)

    for bar, score in zip(bars, per_class_aps):
        ax1.text(min(score + 0.013, 0.96), bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', va='center', ha='left',
                 color='#E6EDF3', fontsize=9, fontweight='bold')

    ax1.axvline(map50, color=RED, linewidth=2, linestyle='--', alpha=0.9, zorder=5)
    ax1.text(map50 + 0.01, n_classes - 0.8,
             f'mAP@50: {map50:.3f}', color=RED, fontsize=10, fontweight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(CLASS_NAMES, fontsize=11, color='#E6EDF3')
    ax1.set_xlabel('AP@50', fontsize=12, color=MUTED, labelpad=10)
    ax1.set_xlim(0, 1.08)
    ax1.set_title('Per-Class AP@IoU=0.50', fontsize=14, color=BLUE,
                  fontweight='bold', pad=16)
    ax1.grid(axis='x', color='#21262D', linewidth=0.8, alpha=0.7)
    ax1.tick_params(colors=MUTED)

    # ── Summary panel ──────────────────────────────────────
    ax2.set_facecolor(PANEL_BG)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')

    metrics = [
        (map50,    'mAP@50',    BLUE),
        (map50_95, 'mAP@50:95', ORANGE),
    ]
    for i, (val, label, color) in enumerate(metrics):
        y_center = 0.72 - i * 0.38
        r = 0.14
        tf = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 200)
        ti = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * val, 200)
        ax2.plot(0.5 + r * np.cos(tf), y_center + r * np.sin(tf),
                 color='#21262D', linewidth=14, transform=ax2.transAxes)
        ax2.plot(0.5 + r * np.cos(ti), y_center + r * np.sin(ti),
                 color=color, linewidth=14, solid_capstyle='round',
                 transform=ax2.transAxes)
        ax2.text(0.5, y_center + 0.01, f'{val:.3f}', ha='center', va='center',
                 fontsize=22, fontweight='bold', color='#E6EDF3',
                 transform=ax2.transAxes)
        ax2.text(0.5, y_center - 0.10, label, ha='center', va='center',
                 fontsize=10, color=MUTED, transform=ax2.transAxes)

    ax2.set_title('Summary', color=BLUE, fontsize=12, fontweight='bold', pad=10)

    fig.suptitle('Mean Average Precision — Test Set',
                 fontsize=16, color='#E6EDF3', fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
             'Badmosh Coders  ·  DINOv2 ViT-B/14 + FPN  ·  Duality AI GHR 2.0 Hackathon',
             ha='center', fontsize=9, color=DIM)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out_path}")


def save_class_tiles(per_class_ious, out_path):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle('Class Performance Summary',
                 fontsize=16, color=BLUE, fontweight='bold', y=1.02)

    for idx in range(n_classes):
        ax    = axes.flatten()[idx]
        vals  = per_class_ious[idx]
        score = np.mean(vals) if vals else 0.0
        color = CLASS_COLORS[idx] / 255.0
        tile  = '#1F3A2A' if score > 0.6 else ('#1C2A1C' if score > 0.3 else '#1C2128')
        ax.set_facecolor(tile)

        theta_full = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 200)
        theta_fill = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * score, 200)
        ax.plot(np.cos(theta_full), np.sin(theta_full),
                color=BORDER, linewidth=7, solid_capstyle='round')
        ax.plot(np.cos(theta_fill), np.sin(theta_fill),
                color=color, linewidth=7, solid_capstyle='round')

        sc = GREEN if score >= 0.5 else (ORANGE if score >= 0.25 else RED)
        ax.text(0, 0.12, f'{score:.2f}', ha='center', va='center',
                fontsize=17, fontweight='bold', color=sc)
        ax.text(0, -0.38, CLASS_NAMES[idx], ha='center', va='center',
                fontsize=8.5, color=MUTED)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect('equal'); ax.axis('off')

    axes.flatten()[-1].set_visible(False)
    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out_path}")


def save_iou_distribution(all_ious, mean_iou, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    n, bins, patches_ = ax.hist(all_ious, bins=20, edgecolor=DARK_BG,
                                linewidth=0.8, alpha=0.9)
    for patch, left in zip(patches_, bins[:-1]):
        if   left < 0.3: patch.set_facecolor(RED)
        elif left < 0.5: patch.set_facecolor(ORANGE)
        elif left < 0.7: patch.set_facecolor(BLUE)
        else:            patch.set_facecolor(GREEN)

    ax.axvline(mean_iou, color='#FFFFFF', linewidth=2, linestyle='--',
               label=f'Mean IoU: {mean_iou:.3f}')
    ax.legend(fontsize=11, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor='#E6EDF3')
    ax.set_xlabel('IoU Score per Image', fontsize=12, color=MUTED)
    ax.set_ylabel('Number of Images',    fontsize=12, color=MUTED)
    ax.set_title('Distribution of Per-Image IoU Scores',
                 fontsize=15, color=BLUE, fontweight='bold', pad=15)
    ax.grid(axis='y', color='#21262D', linewidth=0.8)

    ymax = ax.get_ylim()[1]
    for x, label, c in [(0.15, 'Poor\n<0.3', RED), (0.4, 'Fair\n0.3–0.5', ORANGE),
                         (0.6, 'Good\n0.5–0.7', BLUE), (0.85, 'Great\n>0.7', GREEN)]:
        ax.text(x, ymax * 0.90, label, ha='center', fontsize=8.5, color=c, alpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary_card(mean_iou, map50, map50_95, n_images, per_class_ious, out_path):
    fig = plt.figure(figsize=(18, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.45,
                            left=0.05, right=0.97, top=0.86, bottom=0.08)

    def _gauge(ax, value, label, color, sub=""):
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        tf = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 200)
        ti = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * value, 200)
        cx, cy, r = 0.5, 0.58, 0.32
        ax.plot(cx + r * np.cos(tf), cy + r * np.sin(tf), color='#21262D', linewidth=20)
        ax.plot(cx + r * np.cos(ti), cy + r * np.sin(ti), color=color,     linewidth=20,
                solid_capstyle='round')
        gc = GREEN if value >= 0.6 else (ORANGE if value >= 0.4 else RED)
        ax.text(cx, cy + 0.04, f'{value:.3f}', ha='center', va='center',
                fontsize=30, fontweight='bold', color='#E6EDF3',
                transform=ax.transAxes)
        ax.text(cx, cy - 0.13, label, ha='center', va='center',
                fontsize=11, color=MUTED, transform=ax.transAxes)
        if sub:
            ax.text(cx, 0.10, sub, ha='center', fontsize=9,
                    color=DIM, transform=ax.transAxes)

    # Gauge: mIoU
    ax0 = fig.add_subplot(gs[:, 0])
    _gauge(ax0, mean_iou, 'Mean IoU', BLUE, f'{n_images} images')
    ax0.set_title('mIoU', color=BLUE, fontsize=12, fontweight='bold', pad=10)

    # Gauge: mAP50
    ax1 = fig.add_subplot(gs[0, 1])
    _gauge(ax1, map50, 'mAP@50', GREEN)
    ax1.set_title('mAP@50', color=GREEN, fontsize=10, fontweight='bold', pad=8)

    # Gauge: mAP50-95
    ax2 = fig.add_subplot(gs[1, 1])
    _gauge(ax2, map50_95, 'mAP@50:95', ORANGE)
    ax2.set_title('mAP@50:95', color=ORANGE, fontsize=10, fontweight='bold', pad=8)

    # Top 5 classes
    scored = sorted([(CLASS_NAMES[c], np.mean(v), c)
                     for c, v in per_class_ious.items() if v],
                    key=lambda x: x[1], reverse=True)
    ax_top = fig.add_subplot(gs[0, 2])
    ax_top.set_facecolor(DARK_BG)
    top5 = scored[:5]
    ax_top.barh(range(5), [x[1] for x in top5],
                color=[CLASS_COLORS[x[2]] / 255.0 for x in top5],
                height=0.6, edgecolor=BORDER, linewidth=0.4)
    ax_top.set_yticks(range(5))
    ax_top.set_yticklabels([x[0] for x in top5], fontsize=9, color='#E6EDF3')
    ax_top.set_xlim(0, 1.0)
    ax_top.set_title('Top 5 Classes', color=GREEN, fontsize=10, fontweight='bold')
    ax_top.grid(axis='x', color='#21262D', linewidth=0.6)
    for i, x in enumerate(top5):
        ax_top.text(x[1] + 0.01, i, f'{x[1]:.2f}', va='center', fontsize=8, color=MUTED)

    # Bottom 5 classes
    ax_bot = fig.add_subplot(gs[1, 2])
    ax_bot.set_facecolor(DARK_BG)
    bot5 = scored[-5:][::-1]
    ax_bot.barh(range(len(bot5)), [x[1] for x in bot5],
                color=[CLASS_COLORS[x[2]] / 255.0 for x in bot5],
                height=0.6, edgecolor=BORDER, linewidth=0.4)
    ax_bot.set_yticks(range(len(bot5)))
    ax_bot.set_yticklabels([x[0] for x in bot5], fontsize=9, color='#E6EDF3')
    ax_bot.set_xlim(0, 1.0)
    ax_bot.set_title('Bottom 5 Classes', color=RED, fontsize=10, fontweight='bold')
    ax_bot.grid(axis='x', color='#21262D', linewidth=0.6)
    for i, x in enumerate(bot5):
        ax_bot.text(x[1] + 0.01, i, f'{x[1]:.2f}', va='center', fontsize=8, color=MUTED)

    # Full heatmap
    ax_heat = fig.add_subplot(gs[:, 3])
    ax_heat.set_facecolor(DARK_BG)
    all_scores = [np.mean(per_class_ious[c]) if per_class_ious[c] else 0.0
                  for c in range(n_classes)]
    cmap = LinearSegmentedColormap.from_list('c', [RED, ORANGE, BLUE, GREEN])
    ax_heat.barh(range(n_classes), all_scores,
                 color=[cmap(s) for s in all_scores],
                 height=0.7, edgecolor=BORDER, linewidth=0.4)
    ax_heat.set_yticks(range(n_classes))
    ax_heat.set_yticklabels(CLASS_NAMES, fontsize=9, color='#E6EDF3')
    ax_heat.set_xlim(0, 1.08)
    ax_heat.set_title('All Classes IoU', color=BLUE, fontsize=10, fontweight='bold')
    ax_heat.axvline(mean_iou, color='#FFFFFF', linewidth=1.5, linestyle='--', alpha=0.5)
    ax_heat.grid(axis='x', color='#21262D', linewidth=0.6)
    for i, v in enumerate(all_scores):
        ax_heat.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8, color=MUTED)

    fig.text(0.5, 0.95, 'Badmosh Coders — Duality AI Offroad Segmentation',
             ha='center', fontsize=17, fontweight='bold', color='#E6EDF3')
    fig.text(0.5, 0.91, 'DINOv2 ViT-B/14  +  FPN Decoder  ·  GHR 2.0 Hackathon',
             ha='center', fontsize=10, color=DIM)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n{'='*60}")
    print(f"  Badmosh Coders — Offroad Segmentation Test")
    print(f"  Device: {device}  |  AMP: {use_amp}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

    w = int((960 // 14) * 14)
    h = int((540 // 14) * 14)

    # ── Load model ───────────────────────────────────────────
    print("Loading DINOv2 backbone...")
    backbone_raw = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone_raw.load_state_dict(
        torch.load(os.path.join(RUNS_DIR, "best_backbone.pth"),
                   map_location=device, weights_only=True))
    backbone_raw.to(device).eval()
    extractor = DINOv2MultiScale(backbone_raw)

    head = FPNSegmentationHead(768, n_classes, w // 14, h // 14, 256).to(device)
    head.load_state_dict(
        torch.load(os.path.join(RUNS_DIR, "best_fpn_head.pth"),
                   map_location=device, weights_only=True))
    head.eval()
    print("  Model loaded ✓\n")

    # ── Find images ──────────────────────────────────────────
    img_dir  = os.path.join(TEST_DIR, "Color_Images")
    mask_dir = os.path.join(TEST_DIR, "Segmentation")
    has_masks = os.path.isdir(mask_dir)

    image_files = sorted([f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  Found {len(image_files)} test images  |  GT masks: {has_masks}\n")

    all_ious        = []
    per_class_ious  = {i: [] for i in range(n_classes)}

    # mAP data: {cls: [(iou, confidence, gt_present), ...]}
    per_class_ap_data = {i: [] for i in range(n_classes)}

    saved_grids = 0

    for fname in tqdm(image_files, desc="Inference"):
        image          = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        orig_w, orig_h = image.size

        inp = TF.normalize(TF.to_tensor(TF.resize(image, (h, w))),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        inp = inp.unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            def _pass(x):
                return F.softmax(F.interpolate(
                    head(extractor(x)), (h, w),
                    mode='bilinear', align_corners=False), dim=1)
            probs = (_pass(inp) + torch.flip(_pass(torch.flip(inp, [3])), [3])) / 2

        # probs: (1, n_classes, H, W)
        probs_np   = probs.squeeze(0).cpu().numpy()          # (n_classes, H, W)
        pred       = np.argmax(probs_np, axis=0)             # (H, W)
        pred_full  = np.array(Image.fromarray(pred.astype(np.uint8))
                              .resize((orig_w, orig_h), Image.NEAREST))
        color_mask = mask_to_color(pred_full)

        # Per-class max confidence (used as detection score for AP)
        cls_confidence = probs_np.max(axis=(1, 2))           # (n_classes,)

        gt_color = None
        img_iou  = None

        if has_masks:
            base     = os.path.splitext(fname)[0]
            gt_fname = next((m for m in os.listdir(mask_dir)
                             if os.path.splitext(m)[0] == base), None)
            if gt_fname:
                gt = np.array(convert_mask(
                    Image.open(os.path.join(mask_dir, gt_fname)))
                    .resize((orig_w, orig_h), Image.NEAREST))
                gt_color = mask_to_color(gt)

                img_iou, cls_ious = compute_iou(pred_full, gt)
                all_ious.append(img_iou)
                for cls, iou in cls_ious.items():
                    per_class_ious[cls].append(iou)

                # Collect AP data per class
                for cls in range(n_classes):
                    gt_present  = bool((gt == cls).any())
                    pred_present = bool((pred_full == cls).any())
                    iou_val = cls_ious.get(cls, 0.0)
                    conf    = float(cls_confidence[cls])
                    # Only add entry if class is predicted OR present in GT
                    if pred_present or gt_present:
                        per_class_ap_data[cls].append((iou_val, conf, gt_present))

        if saved_grids < 8:
            out_path = os.path.join(OUTPUT_DIR, "predictions",
                                    f"{os.path.splitext(fname)[0]}_pred.png")
            save_prediction_grid(image, color_mask, gt_color, img_iou, out_path)
            saved_grids += 1

    # ── Compute final metrics ─────────────────────────────────
    print("\nComputing metrics...")

    mean_iou = np.mean(all_ious) if all_ious else 0.0

    # mAP@50
    map50, per_class_aps_50 = compute_map(per_class_ap_data, iou_threshold=0.50)

    # mAP@50:95 (average over thresholds 0.50 → 0.95 step 0.05)
    map_per_threshold = []
    for thr in IOU_THRESHOLDS:
        m, _ = compute_map(per_class_ap_data, iou_threshold=thr)
        map_per_threshold.append(m)
    map50_95 = float(np.mean(map_per_threshold))

    # ── Generate visualizations ───────────────────────────────
    print("\nGenerating report visualizations...")

    save_summary_card(mean_iou, map50, map50_95, len(all_ious),
                      per_class_ious,
                      os.path.join(OUTPUT_DIR, "summary_card.png"))

    save_iou_bar_chart(per_class_ious, mean_iou,
                       os.path.join(OUTPUT_DIR, "iou_bar_chart.png"))

    save_map_chart(per_class_aps_50, map50, map50_95,
                   os.path.join(OUTPUT_DIR, "map_chart.png"))

    save_class_tiles(per_class_ious,
                     os.path.join(OUTPUT_DIR, "class_performance_tiles.png"))

    save_iou_distribution(all_ious, mean_iou,
                          os.path.join(OUTPUT_DIR, "iou_distribution.png"))

    # ── Print + save results ──────────────────────────────────
    lines = [
        "=" * 60,
        "  BADMOSH CODERS — TEST RESULTS",
        "  DINOv2 ViT-B/14 + FPN  |  Duality AI GHR 2.0",
        "=" * 60,
        f"  Images evaluated : {len(all_ious)}",
        f"  Mean IoU (mIoU)  : {mean_iou:.4f}",
        f"  mAP@50           : {map50:.4f}",
        f"  mAP@50:95        : {map50_95:.4f}",
        "",
        "  Per-class IoU & AP@50:",
        f"  {'Class':<18} {'IoU':>7}  {'AP@50':>7}  Bar",
        "  " + "-" * 50,
    ]
    for cls in range(n_classes):
        iou_vals = per_class_ious[cls]
        iou_v    = np.mean(iou_vals) if iou_vals else 0.0
        ap_v     = per_class_aps_50[cls]
        bar      = "█" * int(iou_v * 25)
        lines.append(f"  [{cls:2d}] {CLASS_NAMES[cls]:<14} {iou_v:>7.4f}  {ap_v:>7.4f}  {bar}")
    lines.append("=" * 60)

    result_str = "\n".join(lines)
    print("\n" + result_str)

    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
        f.write(result_str)

    print(f"\n{'='*60}")
    print(f"  Output directory : {OUTPUT_DIR}/")
    print(f"  Files for report :")
    print(f"    summary_card.png             <- mIoU + mAP gauges + class breakdown")
    print(f"    iou_bar_chart.png            <- per-class IoU bars")
    print(f"    map_chart.png                <- per-class AP@50 bars")
    print(f"    class_performance_tiles.png  <- circular gauges")
    print(f"    iou_distribution.png         <- score distribution")
    print(f"    predictions/*.png            <- side-by-side images")
    print(f"    test_results.txt             <- numeric results")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
