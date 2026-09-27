"""
make_test_images.py
====================
Generates a synthetic "clean" base image (so the project is fully
self-contained and reproducible without needing external downloads),
then derives 5 degraded test images that represent the classic problem
categories requested:

    1. dark.png          - underexposed image
    2. bright.png        - overexposed / washed-out image
    3. low_contrast.png  - compressed dynamic range
    4. blurred.png       - Gaussian-blurred (defocus-like) image
    5. noisy.png         - salt-and-pepper noise (impulsive noise)

Run this once to populate images/input/. Everything downstream
(the notebook, the report figures) is generated from these files, so
re-running is fully reproducible.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "input")
os.makedirs(OUT_DIR, exist_ok=True)


def make_base_image(size=(512, 512)):
    """
    Build a synthetic scene with a mix of smooth gradients, hard edges,
    circles, and text -- good for testing both smoothing and sharpening,
    since it contains flat regions, edges, and fine detail.
    """
    w, h = size
    img = Image.new("RGB", size, (30, 30, 40))
    draw = ImageDraw.Draw(img)

    # Sky-like vertical gradient (smooth region -> tests noise/smoothing)
    for y in range(h // 2):
        t = y / (h / 2)
        color = (int(40 + 120 * t), int(80 + 100 * t), int(160 + 60 * t))
        draw.line([(0, y), (w, y)], fill=color)

    # Ground gradient
    for y in range(h // 2, h):
        t = (y - h / 2) / (h / 2)
        color = (int(60 - 20 * t), int(110 - 40 * t), int(60 - 20 * t))
        draw.line([(0, y), (w, y)], fill=color)

    # "Sun" - circle with sharp edge (edge/contrast test)
    draw.ellipse([w - 160, 40, w - 60, 140], fill=(255, 230, 150))

    # Building silhouettes (hard vertical/horizontal edges -> sharpening test)
    rng = np.random.RandomState(1)
    x = 30
    while x < w - 40:
        bw = rng.randint(30, 70)
        bh = rng.randint(80, 220)
        draw.rectangle([x, h // 2 - bh, x + bw, h // 2], fill=(20, 20, 25))
        # windows (fine detail -> tests blur/sharpen quality)
        for wy in range(h // 2 - bh + 10, h // 2 - 10, 18):
            for wx in range(x + 6, x + bw - 6, 14):
                if rng.rand() > 0.4:
                    draw.rectangle([wx, wy, wx + 6, wy + 10], fill=(240, 220, 120))
        x += bw + rng.randint(10, 25)

    # High-frequency texture patch (checkerboard) -> stresses low-pass/high-pass filters
    tex_x0, tex_y0, tex_size, cell = 20, h - 140, 120, 8
    for i in range(tex_size // cell):
        for j in range(tex_size // cell):
            if (i + j) % 2 == 0:
                draw.rectangle([tex_x0 + i * cell, tex_y0 + j * cell,
                                 tex_x0 + (i + 1) * cell, tex_y0 + (j + 1) * cell],
                                fill=(230, 230, 230))
            else:
                draw.rectangle([tex_x0 + i * cell, tex_y0 + j * cell,
                                 tex_x0 + (i + 1) * cell, tex_y0 + (j + 1) * cell],
                                fill=(10, 10, 10))

    # Text (fine detail, good for visually judging sharpening/blur)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((w // 2 - 90, h - 40), "IMAGE ENHANCEMENT", fill=(255, 255, 255), font=font)

    return np.array(img)


def make_dark(img, factor=0.28):
    """Simulate underexposure: multiply intensities down."""
    out = (img.astype(np.float64) * factor).clip(0, 255).astype(np.uint8)
    return out


def make_bright(img, factor=1.9, offset=60):
    """Simulate overexposure: scale up and add offset, causing clipping (washed out)."""
    out = (img.astype(np.float64) * factor + offset).clip(0, 255).astype(np.uint8)
    return out


def make_low_contrast(img, low=90, high=150):
    """Compress the dynamic range into a narrow band [low, high]."""
    img = img.astype(np.float64)
    normalized = (img - img.min()) / (img.max() - img.min())
    out = (normalized * (high - low) + low).clip(0, 255).astype(np.uint8)
    return out


def make_blurred(img, sigma=3.5):
    from scipy import ndimage
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma=sigma)
    return out


def make_noisy(img, amount=0.06):
    """Add salt-and-pepper impulsive noise."""
    out = img.copy()
    h, w = img.shape[:2]
    n_salt = int(amount * h * w / 2)
    n_pepper = int(amount * h * w / 2)

    rng = np.random.RandomState(7)
    ys = rng.randint(0, h, n_salt)
    xs = rng.randint(0, w, n_salt)
    out[ys, xs] = 255

    ys = rng.randint(0, h, n_pepper)
    xs = rng.randint(0, w, n_pepper)
    out[ys, xs] = 0
    return out


def main():
    base = make_base_image()
    Image.fromarray(base).save(os.path.join(OUT_DIR, "original.png"))

    variants = {
        "dark.png": make_dark(base),
        "bright.png": make_bright(base),
        "low_contrast.png": make_low_contrast(base),
        "blurred.png": make_blurred(base),
        "noisy.png": make_noisy(base),
    }
    for name, arr in variants.items():
        Image.fromarray(arr).save(os.path.join(OUT_DIR, name))
        print("saved", name)


if __name__ == "__main__":
    main()
