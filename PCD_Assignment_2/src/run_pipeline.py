"""
run_pipeline.py
================
End-to-end demo: loads each degraded test image from images/input/,
runs the most suitable enhancement technique(s) on it, saves:
  - a side-by-side comparison figure (images/output/*_comparison.png)
  - the enhanced image alone (images/output/*_enhanced.png)
  - a frequency-domain filtering demo for the blurred/noisy cases
  - a CSV/markdown metrics table (report/metrics.md)

This is the same logic exposed in the Colab notebook
(notebook/image_enhancement_colab.ipynb), just runnable as a plain script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import enhancement as ie  # local module

BASE = os.path.dirname(__file__)
IN_DIR = os.path.join(BASE, "..", "images", "input")
OUT_DIR = os.path.join(BASE, "..", "images", "output")
REPORT_DIR = os.path.join(BASE, "..", "report")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load(name):
    return np.array(Image.open(os.path.join(IN_DIR, name)).convert("RGB"))


def save(arr, name):
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(os.path.join(OUT_DIR, name))


def side_by_side(images, titles, filename, cmap=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2 and cmap is None:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=130)
    plt.close(fig)


results = []  # collect metrics rows

original = load("original.png")

# --------------------------------------------------------------------
# 1) DARK IMAGE -> Gamma correction (point processing)
# --------------------------------------------------------------------
dark = load("dark.png")
dark_gamma = ie.gamma_correction(dark, gamma=0.45)
dark_he = ie.histogram_equalization(dark)
save(dark_gamma, "dark_enhanced_gamma.png")
save(dark_he, "dark_enhanced_histeq.png")
side_by_side([dark, dark_gamma, dark_he],
             ["Dark (input)", "Gamma correction (γ=0.45)", "Histogram equalization"],
             "dark_comparison.png")
results.append(["Dark image", "Gamma correction (γ=0.45)",
                 ie.rms_contrast(dark), ie.rms_contrast(dark_gamma),
                 ie.entropy(dark), ie.entropy(dark_gamma)])
results.append(["Dark image", "Histogram equalization",
                 ie.rms_contrast(dark), ie.rms_contrast(dark_he),
                 ie.entropy(dark), ie.entropy(dark_he)])

# --------------------------------------------------------------------
# 2) BRIGHT / OVEREXPOSED IMAGE -> Gamma correction (gamma > 1)
# --------------------------------------------------------------------
bright = load("bright.png")
bright_gamma = ie.gamma_correction(bright, gamma=2.2)
bright_stretch = ie.contrast_stretching(bright, 1, 99)
save(bright_gamma, "bright_enhanced_gamma.png")
save(bright_stretch, "bright_enhanced_stretch.png")
side_by_side([bright, bright_gamma, bright_stretch],
             ["Bright (input)", "Gamma correction (γ=2.2)", "Contrast stretching"],
             "bright_comparison.png")
results.append(["Bright image", "Gamma correction (γ=2.2)",
                 ie.rms_contrast(bright), ie.rms_contrast(bright_gamma),
                 ie.entropy(bright), ie.entropy(bright_gamma)])
results.append(["Bright image", "Contrast stretching",
                 ie.rms_contrast(bright), ie.rms_contrast(bright_stretch),
                 ie.entropy(bright), ie.entropy(bright_stretch)])

# --------------------------------------------------------------------
# 3) LOW-CONTRAST IMAGE -> Contrast stretching, Histogram Eq, CLAHE
# --------------------------------------------------------------------
lowc = load("low_contrast.png")
lowc_stretch = ie.contrast_stretching(lowc, 2, 98)
lowc_he = ie.histogram_equalization(lowc)
lowc_clahe = ie.clahe(lowc, clip_limit=3.0)
save(lowc_stretch, "low_contrast_enhanced_stretch.png")
save(lowc_he, "low_contrast_enhanced_histeq.png")
save(lowc_clahe, "low_contrast_enhanced_clahe.png")
side_by_side([lowc, lowc_stretch, lowc_he, lowc_clahe],
             ["Low-contrast (input)", "Contrast stretching",
              "Histogram equalization", "CLAHE"],
             "low_contrast_comparison.png")
for name, out in [("Contrast stretching", lowc_stretch),
                   ("Histogram equalization", lowc_he),
                   ("CLAHE", lowc_clahe)]:
    results.append(["Low-contrast image", name,
                     ie.rms_contrast(lowc), ie.rms_contrast(out),
                     ie.entropy(lowc), ie.entropy(out)])

# --------------------------------------------------------------------
# 4) BLURRED IMAGE -> Unsharp mask, Laplacian sharpen, Freq-domain HPF
# --------------------------------------------------------------------
blurred = load("blurred.png")
blur_unsharp = ie.unsharp_mask(blurred, sigma=2.5, amount=2.0)
blur_laplace = ie.laplacian_sharpen(blurred, weight=1.2)
blur_freq_hp = ie.freq_highpass(blurred, d0=25, kind="gaussian", boost=1.5)
save(blur_unsharp, "blurred_enhanced_unsharp.png")
save(blur_laplace, "blurred_enhanced_laplacian.png")
save(blur_freq_hp, "blurred_enhanced_freqHPF.png")
side_by_side([blurred, blur_unsharp, blur_laplace, blur_freq_hp],
             ["Blurred (input)", "Unsharp masking", "Laplacian sharpening",
              "Freq-domain high-boost"],
             "blurred_comparison.png")
for name, out in [("Unsharp masking", blur_unsharp),
                   ("Laplacian sharpening", blur_laplace),
                   ("Freq-domain high-boost", blur_freq_hp)]:
    p = ie.psnr(original, out if out.ndim == 3 else np.stack([out]*3, -1))
    results.append(["Blurred image", name,
                     ie.psnr(original, blurred), p,
                     ie.entropy(blurred), ie.entropy(out)])

# --------------------------------------------------------------------
# 5) NOISY IMAGE (salt & pepper) -> Median filter vs Mean/Gaussian
# --------------------------------------------------------------------
noisy = load("noisy.png")
noisy_median = ie.median_filter(noisy, ksize=3)
noisy_mean = ie.mean_filter(noisy, ksize=3)
noisy_gauss = ie.gaussian_filter(noisy, sigma=1.2)
save(noisy_median, "noisy_enhanced_median.png")
save(noisy_mean, "noisy_enhanced_mean.png")
save(noisy_gauss, "noisy_enhanced_gaussian.png")
side_by_side([noisy, noisy_median, noisy_mean, noisy_gauss],
             ["Noisy (input, salt & pepper)", "Median filter (3x3)",
              "Mean filter (3x3)", "Gaussian filter (sigma=1.2)"],
             "noisy_comparison.png")
for name, out in [("Median filter (3x3)", noisy_median),
                   ("Mean filter (3x3)", noisy_mean),
                   ("Gaussian filter", noisy_gauss)]:
    results.append(["Noisy image", name,
                     ie.psnr(original, noisy), ie.psnr(original, out),
                     ie.entropy(noisy), ie.entropy(out)])

# --------------------------------------------------------------------
# 6) Frequency-domain spectrum visualization (educational, matches slides)
# --------------------------------------------------------------------
spec_orig = ie.freq_spectrum_image(original)
spec_blur = ie.freq_spectrum_image(blurred)
spec_noisy = ie.freq_spectrum_image(noisy)
side_by_side([spec_orig, spec_blur, spec_noisy],
             ["Spectrum: original", "Spectrum: blurred (energy concentrated at low-f)",
              "Spectrum: noisy (energy spread to high-f)"],
             "fourier_spectra_comparison.png")

lpf_demo = ie.freq_lowpass(noisy, d0=35, kind="gaussian")
hpf_demo = ie.freq_highpass(blurred, d0=25, kind="butterworth", order=2, boost=1.3)
side_by_side([noisy, lpf_demo], ["Noisy (input)", "Gaussian Low-Pass Filter (D0=35)"],
             "freq_lowpass_demo.png")
side_by_side([blurred, hpf_demo], ["Blurred (input)", "Butterworth High-Pass boost (D0=25,n=2)"],
             "freq_highpass_demo.png")

# --------------------------------------------------------------------
# 7) auto_enhance() demo across all 5 degraded images
# --------------------------------------------------------------------
auto_imgs, auto_titles = [], []
for name in ["dark.png", "bright.png", "low_contrast.png", "blurred.png", "noisy.png"]:
    im = load(name)
    out, problem = ie.auto_enhance(im, verbose=False)
    auto_imgs += [im, out]
    auto_titles += [f"{name}\n(detected: {problem})", f"{name} -> auto-enhanced"]

fig, axes = plt.subplots(5, 2, figsize=(8, 20))
for ax, im, t in zip(axes.flatten(), auto_imgs, auto_titles):
    ax.imshow(im)
    ax.set_title(t, fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "auto_enhance_all.png"), dpi=120)
plt.close(fig)

# --------------------------------------------------------------------
# Write metrics table (markdown) for the report
# --------------------------------------------------------------------
md_lines = [
    "| Test image | Method | Metric (before) | Metric (after) | Entropy (before) | Entropy (after) |",
    "|---|---|---:|---:|---:|---:|",
]
for row in results:
    cat, method, before, after, ent_b, ent_a = row
    md_lines.append(f"| {cat} | {method} | {before:.2f} | {after:.2f} | {ent_b:.2f} | {ent_a:.2f} |")

with open(os.path.join(REPORT_DIR, "metrics.md"), "w") as f:
    f.write("\n".join(md_lines))

print("Pipeline complete.")
print(f"Outputs saved to: {OUT_DIR}")
print(f"Metrics table saved to: {os.path.join(REPORT_DIR, 'metrics.md')}")
