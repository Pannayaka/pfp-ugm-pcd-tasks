"""
build_notebook.py — generates notebook/image_enhancement_colab.ipynb
by assembling markdown explanations + the actual source code from
src/enhancement.py and src/make_test_images.py, so the notebook and the
plain .py source files never drift out of sync.
"""
import nbformat as nbf
import os

BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, "src", "enhancement.py")) as f:
    ENHANCEMENT_CODE = f.read()

with open(os.path.join(BASE, "src", "make_test_images.py")) as f:
    MAKE_IMAGES_CODE = f.read()

# Strip the "if __name__" runner from make_test_images so we control execution
MAKE_IMAGES_CODE = MAKE_IMAGES_CODE.replace(
    'if __name__ == "__main__":\n    main()', ""
)
# The path logic in make_test_images.py assumes running from src/, adjust for Colab flat layout
MAKE_IMAGES_CODE = MAKE_IMAGES_CODE.replace(
    'OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "input")',
    'OUT_DIR = "images/input"'
)

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------
md(r"""
# Image Enhancement Based on Filtering — Google Colab Implementation

**Course topic:** *Image Enhancement Based on Filtering* (Spatial-domain and
Frequency-domain filtering)

This notebook implements and demonstrates image-enhancement techniques for
several common real-world image problems:

| Problem | Technique(s) applied |
|---|---|
| Dark / underexposed image | Gamma correction, Histogram Equalization |
| Bright / overexposed image | Gamma correction, Contrast stretching |
| Low-contrast image | Contrast stretching, Histogram Equalization, CLAHE |
| Blurred image | Unsharp masking, Laplacian sharpening, Frequency-domain high-pass boosting |
| Noisy image (salt & pepper) | Median filter, (compared against Mean & Gaussian filters) |

It also implements the **frequency-domain (Fourier / DFT-based)** filters
covered in the lecture slides: Ideal, Gaussian, and Butterworth low-pass and
high-pass filters, plus an `auto_enhance()` function that automatically
diagnoses an arbitrary input image and applies the most suitable correction.

> **How to run:** `Runtime → Run all`. No external downloads are required —
> a synthetic test image and its degraded variants are generated
> programmatically, so results are 100% reproducible. At the bottom of the
> notebook you can also upload **your own photo** and run `auto_enhance()`
> on it.
""")

md("## 0. Setup")
code("""\
!pip -q install scipy scikit-image opencv-python-headless matplotlib pillow numpy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

np.random.seed(42)
os.makedirs("images/input", exist_ok=True)
os.makedirs("images/output", exist_ok=True)
print("Setup complete.")
""")

# ---------------------------------------------------------------------
md("""## 1. Core enhancement library

The cell below defines every enhancement function used in this notebook —
identical to `src/enhancement.py` in the project repository — covering:

- **Point processing:** gamma correction, contrast stretching, histogram
  equalization, CLAHE
- **Spatial smoothing:** mean filter, Gaussian filter, median filter
- **Spatial sharpening:** Laplacian filter, unsharp masking
- **Frequency domain:** 2D FFT, Ideal / Gaussian / Butterworth low-pass and
  high-pass filters, spectrum visualization
- **Automatic diagnosis + routing:** `diagnose()` and `auto_enhance()`
""")
code(ENHANCEMENT_CODE)

# ---------------------------------------------------------------------
md("""## 2. Generate synthetic test images

To make the notebook fully self-contained and reproducible (no external
image downloads needed), we synthesize one clean "base" image and derive
five degraded versions of it: **dark, bright, low-contrast, blurred,** and
**noisy (salt & pepper)**.

> Want to test your *own* image instead? Skip to **Section 8** near the end
> of the notebook.
""")
code(MAKE_IMAGES_CODE)
code("""\
base = make_base_image()
Image.fromarray(base).save("images/input/original.png")

variants = {
    "dark.png": make_dark(base),
    "bright.png": make_bright(base),
    "low_contrast.png": make_low_contrast(base),
    "blurred.png": make_blurred(base),
    "noisy.png": make_noisy(base),
}
for name, arr in variants.items():
    Image.fromarray(arr).save(f"images/input/{name}")

fig, axes = plt.subplots(1, 6, figsize=(22, 4))
for ax, (name, arr) in zip(axes, [("original", base)] + list(variants.items())):
    ax.imshow(arr); ax.set_title(name); ax.axis("off")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------
def load_cell():
    return """\
def load(name):
    return np.array(Image.open(f"images/input/{name}").convert("RGB"))

def show(images, titles, figsize_per=4.2):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per*n, figsize_per))
    if n == 1: axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10); ax.axis("off")
    plt.tight_layout(); plt.show()

original = load("original.png")
"""

md("## 3. Dark / underexposed image → Gamma correction & Histogram Equalization")
code(load_cell())
code("""\
dark = load("dark.png")
dark_gamma = gamma_correction(dark, gamma=0.45)
dark_he = histogram_equalization(dark)

show([dark, dark_gamma, dark_he],
     ["Dark (input)", "Gamma correction (γ=0.45)", "Histogram equalization"])

print(f"RMS contrast  before: {rms_contrast(dark):.2f}   after (gamma): {rms_contrast(dark_gamma):.2f}   after (HE): {rms_contrast(dark_he):.2f}")
print(f"Entropy       before: {entropy(dark):.2f}   after (gamma): {entropy(dark_gamma):.2f}   after (HE): {entropy(dark_he):.2f}")
""")

md("""**Observation:** Gamma correction (γ<1) expands the dark tonal range
smoothly, preserving natural-looking gradients. Global Histogram
Equalization boosts contrast more aggressively but can introduce visible
**banding** in smooth gradients (e.g. the sky) because it forces the
histogram to be (approximately) uniform.""")

# ---------------------------------------------------------------------
md("## 4. Bright / overexposed image → Gamma correction & Contrast stretching")
code("""\
bright = load("bright.png")
bright_gamma = gamma_correction(bright, gamma=2.2)
bright_stretch = contrast_stretching(bright, 1, 99)

show([bright, bright_gamma, bright_stretch],
     ["Bright (input)", "Gamma correction (γ=2.2)", "Contrast stretching"])

print(f"RMS contrast  before: {rms_contrast(bright):.2f}   after (gamma): {rms_contrast(bright_gamma):.2f}   after (stretch): {rms_contrast(bright_stretch):.2f}")
""")

md("""**Observation:** With γ>1, gamma correction compresses the bright end
of the range, recovering detail lost to clipping/over-exposure. Contrast
stretching re-maps the (compressed) percentile range back across the full
0–255 span.""")

# ---------------------------------------------------------------------
md("## 5. Low-contrast image → Contrast stretching, Histogram Equalization, CLAHE")
code("""\
lowc = load("low_contrast.png")
lowc_stretch = contrast_stretching(lowc, 2, 98)
lowc_he = histogram_equalization(lowc)
lowc_clahe = clahe(lowc, clip_limit=3.0)

show([lowc, lowc_stretch, lowc_he, lowc_clahe],
     ["Low-contrast (input)", "Contrast stretching", "Histogram equalization", "CLAHE"])

for name, out in [("stretch", lowc_stretch), ("HE", lowc_he), ("CLAHE", lowc_clahe)]:
    print(f"{name:8s}  RMS contrast: {rms_contrast(out):6.2f}   entropy: {entropy(out):.2f}")
print(f"{'input':8s}  RMS contrast: {rms_contrast(lowc):6.2f}   entropy: {entropy(lowc):.2f}")
""")

md("""**Observation:** All three methods substantially increase contrast.
Global histogram equalization achieves the biggest *numerical* contrast gain
but at the cost of banding artifacts; **CLAHE** trades a smaller contrast
gain for locally-adaptive, more natural-looking results, since it limits
contrast amplification per tile.""")

# ---------------------------------------------------------------------
md("""## 6. Blurred image → Spatial sharpening & Frequency-domain high-pass boosting

This section directly implements the *"Sharpening revisited"* and
*"Sharpen filter"* concepts from the lecture: `sharpened = original +
alpha*(original - blurred)`, its Laplacian-kernel equivalent, and the
frequency-domain high-boost filter `G(u,v) = F(u,v)·H(u,v)`.""")
code("""\
blurred = load("blurred.png")
blur_unsharp  = unsharp_mask(blurred, sigma=2.5, amount=2.0)
blur_laplace  = laplacian_sharpen(blurred, weight=1.2)
blur_freq_hp  = freq_highpass(blurred, d0=25, kind="gaussian", boost=1.5)

show([blurred, blur_unsharp, blur_laplace],
     ["Blurred (input)", "Unsharp masking", "Laplacian sharpening"])
show([blurred, np.stack([blur_freq_hp]*3, -1)],
     ["Blurred (input)", "Frequency-domain high-boost (Gaussian HPF, D0=25)"])

print(f"PSNR vs. clean original — blurred: {psnr(original, blurred):.2f} dB")
print(f"PSNR vs. clean original — unsharp: {psnr(original, blur_unsharp):.2f} dB")
print(f"PSNR vs. clean original — laplacian: {psnr(original, blur_laplace):.2f} dB")
print(f"PSNR vs. clean original — freq high-boost: {psnr(original, np.stack([blur_freq_hp]*3,-1)):.2f} dB")
""")

md("""**Observation:** Sharpening cannot perfectly undo a Gaussian blur
(that would require true deconvolution), but it *perceptually* restores
edge crispness by boosting high-frequency content. Overly aggressive
high-pass boosting can amplify noise/ringing (visible as haloing around
edges) and can even *reduce* PSNR relative to the blurred image, which is
why the sharpening amount/D0 must be tuned per image.""")

# ---------------------------------------------------------------------
md("""## 7. Noisy image (salt & pepper) → Median filter vs. Mean / Gaussian filters

This reproduces the *"Median filters"* / *"Local Statistic Filters"* slide
comparison directly.""")
code("""\
noisy = load("noisy.png")
noisy_median = median_filter(noisy, ksize=3)
noisy_mean   = mean_filter(noisy, ksize=3)
noisy_gauss  = gaussian_filter(noisy, sigma=1.2)

show([noisy, noisy_median, noisy_mean, noisy_gauss],
     ["Noisy (salt & pepper)", "Median filter (3x3)", "Mean filter (3x3)", "Gaussian filter"])

for name, out in [("median", noisy_median), ("mean", noisy_mean), ("gaussian", noisy_gauss)]:
    print(f"{name:8s}  PSNR vs clean original: {psnr(original, out):.2f} dB")
print(f"{'input':8s}  PSNR vs clean original: {psnr(original, noisy):.2f} dB")
""")

md("""**Observation:** The **median filter** removes salt-and-pepper
impulsive noise almost completely while preserving edges, because it
replaces each pixel with the *median* (not the mean) of its neighborhood —
outlier impulse values are simply discarded rather than averaged in. Mean
and Gaussian filters, being linear, only *smear* the noise across
neighboring pixels rather than removing it, which is visible as residual
speckling plus unwanted blurring of real edges.""")

# ---------------------------------------------------------------------
md("""## 7b. Frequency-domain view: Fourier spectra & Low/High-pass filtering

Matches the *"FT Pair Example"* and *"Low-pass & High-pass Filtering"*
slides: blur concentrates energy near the DC/low-frequency center of the
spectrum, while noise spreads energy into high frequencies.""")
code("""\
spec_orig  = freq_spectrum_image(original)
spec_blur  = freq_spectrum_image(blurred)
spec_noisy = freq_spectrum_image(noisy)
show([spec_orig, spec_blur, spec_noisy],
     ["Spectrum: original", "Spectrum: blurred\\n(energy concentrated at center)",
      "Spectrum: noisy\\n(energy spread to high-f)"])

lpf_demo = freq_lowpass(noisy, d0=35, kind="gaussian")
hpf_demo = freq_highpass(blurred, d0=25, kind="butterworth", order=2, boost=1.3)
show([noisy, lpf_demo], ["Noisy (input)", "Gaussian Low-Pass Filter (D0=35)\\n(denoises but blurs)"])
show([blurred, hpf_demo], ["Blurred (input)", "Butterworth High-Pass boost (D0=25, n=2)"])
""")

# ---------------------------------------------------------------------
md("""## 8. Bonus: fully automatic enhancement (`auto_enhance`)

`diagnose()` inspects mean brightness, RMS contrast, an impulsive-noise
estimate, and Laplacian variance (a standard blur metric) to classify an
arbitrary image, and `auto_enhance()` then routes it to the matching
correction — useful as a one-call "auto-fix" utility.

**Try it on your own photo:** run the cell below in Colab, click *Choose
Files*, and upload any image (dark, backlit, hazy, blurry phone photo,
etc.).""")
code("""\
try:
    from google.colab import files
    uploaded = files.upload()
    user_path = list(uploaded.keys())[0]
    user_img = np.array(Image.open(user_path).convert("RGB"))
    enhanced, problem = auto_enhance(user_img)
    show([user_img, enhanced], [f"Your image (diagnosed: {problem})", "Auto-enhanced"])
except ImportError:
    print("google.colab not available (not running in Colab) — skipping upload demo.")
    print("You can still call auto_enhance(your_numpy_image) directly.")
""")

code("""\
# Demo across all 5 synthetic degraded images
for name in ["dark.png", "bright.png", "low_contrast.png", "blurred.png", "noisy.png"]:
    im = load(name)
    out, problem = auto_enhance(im)
    show([im, out], [f"{name} (diagnosed: {problem})", "auto_enhance() output"])
""")

# ---------------------------------------------------------------------
md("""## 9. Save all outputs (for the GitHub report package)""")
code("""\
import shutil

def save(arr, name):
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(f"images/output/{name}")

save(dark_gamma, "dark_enhanced_gamma.png")
save(dark_he, "dark_enhanced_histeq.png")
save(bright_gamma, "bright_enhanced_gamma.png")
save(bright_stretch, "bright_enhanced_stretch.png")
save(lowc_stretch, "low_contrast_enhanced_stretch.png")
save(lowc_he, "low_contrast_enhanced_histeq.png")
save(lowc_clahe, "low_contrast_enhanced_clahe.png")
save(blur_unsharp, "blurred_enhanced_unsharp.png")
save(blur_laplace, "blurred_enhanced_laplacian.png")
save(blur_freq_hp, "blurred_enhanced_freqHPF.png")
save(noisy_median, "noisy_enhanced_median.png")
save(noisy_mean, "noisy_enhanced_mean.png")
save(noisy_gauss, "noisy_enhanced_gaussian.png")

shutil.make_archive("image_enhancement_outputs", "zip", "images")
print("Saved all outputs and zipped them into image_enhancement_outputs.zip")

try:
    from google.colab import files
    files.download("image_enhancement_outputs.zip")
except ImportError:
    print("(Not in Colab — zip is available in the local file browser instead.)")
""")

md("""---
## Summary

| Degradation | Best technique in this notebook | Why |
|---|---|---|
| Dark | Gamma correction (γ<1) | Smooth tonal expansion, no banding |
| Bright | Gamma correction (γ>1) / contrast stretch | Recovers clipped highlight detail |
| Low contrast | CLAHE | Locally adaptive, avoids over-amplification |
| Blurred | Unsharp masking | Best perceptual sharpness gain without heavy ringing |
| Noisy (salt & pepper) | Median filter | Removes impulses instead of averaging them in |

See `report/REPORT.md` in the repository for the full written analysis,
including quantitative metrics (PSNR / RMS contrast / entropy) for every
method tested.
""")

nb['cells'] = cells
os.makedirs(os.path.join(BASE, "notebook"), exist_ok=True)
with open(os.path.join(BASE, "notebook", "image_enhancement_colab.ipynb"), "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebook/image_enhancement_colab.ipynb")
print(f"Total cells: {len(cells)}")
