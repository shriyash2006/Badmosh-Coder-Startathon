"""
Visualization Script
Overlays segmentation predictions on original images using high-contrast colors
Duality AI Offroad Segmentation Hackathon
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Class colors (high contrast palette)
# ============================================================

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape", "Sky"
]

CLASS_COLORS = [
    (0,   0,   0),    # 0  Background   - Black
    (34,  139, 34),   # 1  Trees        - Forest Green
    (0,   255, 0),    # 2  Lush Bushes  - Lime Green
    (255, 255, 0),    # 3  Dry Grass    - Yellow
    (139, 90,  43),   # 4  Dry Bushes   - Brown
    (128, 128, 128),  # 5  Clutter      - Gray
    (255, 0,   255),  # 6  Flowers      - Magenta
    (160, 82,  45),   # 7  Logs         - Sienna
    (169, 169, 169),  # 8  Rocks        - Dark Gray
    (210, 180, 140),  # 9  Landscape    - Tan
    (135, 206, 235),  # 10 Sky          - Sky Blue
]

n_classes = len(CLASS_NAMES)


def colorize_mask(mask_array):
    """Convert class index array to RGB color image."""
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask_array == cls_idx] = color
    return Image.fromarray(color_mask)


def overlay_mask(original_image, color_mask, alpha=0.5):
    """Blend original image with color mask."""
    original = original_image.convert("RGB")
    color    = color_mask.convert("RGB")
    original = original.resize(color.size)
    blended  = Image.blend(original, color, alpha=alpha)
    return blended


def visualize_prediction(image_path, pred_mask_array, save_path=None, show=False):
    """
    Create a 3-panel visualization:
      Left:   Original image
      Middle: Predicted segmentation mask
      Right:  Overlay (image + mask blended)
    """
    original   = Image.open(image_path).convert("RGB")
    color_mask = colorize_mask(pred_mask_array)
    overlay    = overlay_mask(original, color_mask, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Segmentation", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]], label=CLASS_NAMES[i])
        for i in range(n_classes)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("Visualize script loaded.")
    print("Use visualize_prediction(image_path, pred_mask_array, save_path) to generate visuals.")
    print(f"Classes: {CLASS_NAMES}")
