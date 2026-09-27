# Image Enhancement Based on Filtering — Analysis Report

**Project:** Implementation of spatial-domain and frequency-domain image
enhancement techniques in Google Colab
**Reference material:** *Image Enhancement Based on Filtering*, Wahyono,
Ph.D., Dept. of Computer Science and Electronics, Universitas Gadjah Mada
**Deliverables:** `notebook/image_enhancement_colab.ipynb`, `src/*.py`,
`images/input/`, `images/output/`, this report

---

## 1. Objective

The goal of this project is to implement and empirically compare classical
image-enhancement techniques — covering both the **spatial domain**
(point processing, linear/non-linear convolution filters) and the
**frequency domain** (2D DFT-based filtering) — and to apply them
correctly to five common categories of degraded images: **dark
(underexposed)**, **bright (overexposed)**, **low-contrast**, **blurred**,
and **noisy (salt-and-pepper)** images.

## 2. Methodology

### 2.1 Test data

To keep the project fully self-contained and reproducible without relying
on external image downloads, a synthetic 512×512 "city skyline" base image
was generated programmatically (`src/make_test_images.py`), containing
smooth gradients (sky/ground), hard geometric edges (buildings, sun), fine
periodic detail (a checkerboard patch, window grids), and text — a scene
designed to stress-test both smoothing and sharpening operators. Five
degraded variants were then derived from it by controlled, reproducible
transformations:

| File | Degradation | How it was produced |
|---|---|---|
| `dark.png` | Underexposure | Multiply intensities by 0.28 |
| `bright.png` | Overexposure | Multiply by 1.9, add offset 60 (causes clipping) |
| `low_contrast.png` | Compressed dynamic range | Rescale intensities into the narrow band [90,150] |
| `blurred.png` | Defocus-like blur | Gaussian blur, σ = 3.5 |
| `noisy.png` | Impulsive noise | 6% salt-and-pepper noise |

Because the *undegraded* original is known, this design allows both
**no-reference** metrics (RMS contrast, entropy) and **full-reference**
metrics (PSNR against the clean original) to be computed for every method.

### 2.2 Techniques implemented (`src/enhancement.py`)

**Spatial domain — point processing** (for dark/bright/low-contrast images):
- *Gamma correction:* g = 255·(f/255)^γ. γ<1 brightens shadows; γ>1
  compresses highlights.
- *Contrast stretching:* linear percentile-based rescaling of intensities
  to [0,255].
- *Global histogram equalization:* remaps intensities via the cumulative
  distribution function so the output histogram is approximately uniform.
- *CLAHE* (Contrast-Limited Adaptive Histogram Equalization): histogram
  equalization applied per local tile with a clip limit, avoiding the
  over-amplification and banding of the global method.

**Spatial domain — convolution filters** (following the lecture's
`G[i,j] = Σ H[u,v]·F[i-u,j-v]` definition, implemented directly with
`scipy.signal.convolve2d`):
- *Mean/box filter* and *Gaussian filter* — linear smoothing (low-pass) for
  noise suppression.
- *Median filter* — a non-linear, local-statistic filter, expected to
  remove impulsive noise without smearing edges the way linear filters do.
- *Laplacian sharpening* — `g = f − w·∇²f` using the `[[0,1,0],[1,-4,1],[0,1,0]]`
  kernel from the slides.
- *Unsharp masking* — `g = f + α·(f − blur(f))`, i.e. adding back the
  high-frequency "detail" layer extracted by subtracting a blurred copy.

**Frequency domain** (2D FFT-based, matching the "Basic Steps" slide:
`F(u,v)=FFT{f}` → `G(u,v)=F(u,v)·H(u,v)` → `g=IFFT{G}`):
- Ideal, Gaussian, and Butterworth **low-pass filters** `H(u,v)`, using the
  exact formulas given in the slides (e.g. Butterworth:
  `H(u,v)=1/(1+[D(u,v)/D₀]^2n)`).
- Ideal, Gaussian, and Butterworth **high-pass filters**, and a
  high-frequency-emphasis ("high-boost") sharpening built from them.
- A log-magnitude **Fourier spectrum** visualizer for qualitative analysis.

**Automatic routing:** `diagnose()` classifies an arbitrary input image
using simple statistics (mean brightness, RMS contrast, a noise estimate
from a median-filter residual, and Laplacian variance as a blur proxy), and
`auto_enhance()` applies the matching correction — allowing the pipeline to
be run on an arbitrary uploaded photo with a single call.

## 3. Results

### 3.1 Quantitative metrics

*"Metric" is RMS contrast for dark/bright/low-contrast images, and PSNR
(dB, vs. the clean original) for blurred/noisy images.*

| Test image | Method | Before | After | Entropy before | Entropy after |
|---|---|---:|---:|---:|---:|
| Dark | Gamma correction (γ=0.45) | 15.43 | 26.25 | 4.32 | 4.73 |
| Dark | Histogram equalization | 15.43 | 69.04 | 4.32 | 4.50 |
| Bright | Gamma correction (γ=2.2) | 55.66 | 77.12 | 4.98 | 5.18 |
| Bright | Contrast stretching | 55.66 | 80.76 | 4.98 | 5.18 |
| Low-contrast | Contrast stretching | 13.42 | 63.26 | 4.21 | 4.70 |
| Low-contrast | Histogram equalization | 13.42 | 68.84 | 4.21 | 4.36 |
| Low-contrast | CLAHE | 13.42 | 17.11 | 4.21 | 4.93 |
| Blurred | Unsharp masking | 17.26 dB | 18.69 dB | 6.63 | 6.75 |
| Blurred | Laplacian sharpening | 17.26 dB | 17.65 dB | 6.63 | 6.67 |
| Blurred | Freq-domain high-boost | 17.26 dB | 15.99 dB | 6.63 | 6.93 |
| Noisy | Median filter (3×3) | 17.03 dB | **26.84 dB** | 5.61 | 5.55 |
| Noisy | Mean filter (3×3) | 17.03 dB | 21.11 dB | 5.61 | 6.44 |
| Noisy | Gaussian filter | 17.03 dB | 21.29 dB | 5.61 | 7.03 |

(Full machine-generated copy: `report/metrics.md`. Side-by-side visual
comparisons for every row above are in `images/output/*_comparison.png`.)

### 3.2 Discussion by category

**Dark images.** Both gamma correction and histogram equalization
dramatically raise visibility (RMS contrast 15.4 → 26–69). However,
histogram equalization visibly introduces **banding** in the smooth sky
gradient (`images/output/dark_comparison.png`), because it forces
under-represented tonal ranges to expand non-linearly according to the
histogram's CDF, which is harsh on images with large smooth regions. Gamma
correction (γ=0.45) gives a more modest but visually natural brightening
with no artifacts — the better general-purpose choice for underexposed
photographs.

**Bright / overexposed images.** Gamma correction (γ>1) and percentile
contrast stretching perform similarly well here, since the source
degradation (linear scale-up with clipping) is close to their inverse
operation; both recover the "washed-out" building silhouettes clearly.

**Low-contrast images.** All three point-processing methods substantially
increase RMS contrast, but **CLAHE** achieves this most conservatively
(13.4 → 17.1) while producing the most natural-looking result
(`images/output/low_contrast_comparison.png`) — global histogram
equalization again introduces sky-banding, at the cost of the largest raw
contrast number. This illustrates an important general lesson: **a bigger
contrast/entropy number is not automatically a better image** — CLAHE's
local, clip-limited approach avoids over-amplifying noise/gradients the
way the global method does.

**Blurred images.** Spatial sharpening (unsharp masking, Laplacian) and
frequency-domain high-boost filtering all *perceptually* sharpen edges, but
none can fully invert a Gaussian blur (true deconvolution would be
required for that). Unsharp masking gave the best PSNR gain (17.26 → 18.69
dB) with the fewest artifacts. The frequency-domain high-boost filter
actually **reduced** PSNR (17.26 → 15.99 dB) despite visibly sharper
edges — a textbook illustration of the smoothing/sharpening–noise
trade-off and of **ringing artifacts** from a hard cutoff in the frequency
domain (see `images/output/blurred_enhanced_freqHPF.png`), matching the
"Ringing and Blurring" slide in the reference material. In practice, the
boost factor and cutoff radius D₀ must be tuned conservatively.

**Noisy (salt-and-pepper) images.** This is the clearest result in the
project: the **median filter** improved PSNR from 17.03 dB to **26.84 dB**
and visually removed essentially all impulsive noise while preserving
edges and the checkerboard texture (`images/output/noisy_comparison.png`).
The linear mean and Gaussian filters only reached 21.1–21.3 dB — they
*smear* the salt-and-pepper impulses into their neighborhoods instead of
removing them, simultaneously blurring genuine edges. This directly
confirms the lecture's point that median (a non-linear, order-statistic
filter) is specifically suited to impulsive noise, whereas linear
smoothing filters are better suited to additive Gaussian-type noise.

### 3.3 Frequency-domain analysis

The log-magnitude Fourier spectra (`images/output/fourier_spectra_comparison.png`)
visually confirm the theory from the lecture: the **blurred** image's
spectrum energy is tightly concentrated near the DC/low-frequency center
(a Gaussian blur is itself a low-pass filter, attenuating high frequencies),
while the **noisy** image's spectrum shows energy spread broadly across
high frequencies (impulsive noise injects broadband content). This is
precisely why a **low-pass** filter suppresses noise (at the cost of
blurring) and a **high-pass** filter sharpens blur (at the cost of noise
amplification/ringing) — the two operations are, in a sense, complementary
duals of each other, and choosing the right cutoff frequency D₀ is a
direct trade-off between detail preservation and artifact suppression.

## 4. Conclusion

No single filter is universally "best" — the correct enhancement technique
depends entirely on the type of degradation:

| Degradation | Recommended technique | Rationale |
|---|---|---|
| Dark | Gamma correction (γ<1) | Natural tonal expansion, no banding |
| Bright | Gamma correction (γ>1) / contrast stretch | Recovers clipped highlight detail |
| Low contrast | CLAHE | Locally adaptive; avoids over-amplification |
| Blurred | Unsharp masking | Best perceptual/PSNR sharpening with fewest artifacts |
| Salt-and-pepper noise | Median filter | Removes outliers instead of averaging them in |

The `auto_enhance()` utility implemented in this project operationalizes
this table: it diagnoses the dominant defect in an arbitrary image and
automatically routes it to the corresponding correction, demonstrated
successfully on all five synthetic test images and available for
user-uploaded photos directly inside the Colab notebook.

## 5. References

1. Wahyono, *Image Enhancement Based on Filtering*, Dept. of Computer
   Science and Electronics, Universitas Gadjah Mada (lecture slides
   provided for this assignment).
2. R. C. Gonzalez & R. E. Woods, *Digital Image Processing*, 3rd/4th ed.
   (source of the Laplacian, Butterworth, and Gaussian filter figures
   referenced in the lecture material).
3. `scipy.ndimage`, `numpy.fft`, and `opencv-python` documentation, used
   for the underlying convolution, FFT, and CLAHE implementations.
