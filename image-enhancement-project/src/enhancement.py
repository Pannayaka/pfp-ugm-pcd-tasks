"""
enhancement.py
================
Image Enhancement Based on Filtering
-------------------------------------
Implements the spatial-domain and frequency-domain techniques covered in
"Image Enhancement Based on Filtering" (Wahyono, Ph.D., UGM):

Spatial domain:
    - Linear convolution filtering (generic kernel convolution)
    - Smoothing filters: mean (box) filter, Gaussian filter
    - Median filter (non-linear, local-statistic filter)
    - Sharpening: Laplacian filter, unsharp masking (F + alpha*(F - blur(F)))
    - Point-processing: gamma correction, contrast stretching,
      histogram equalization (for dark / bright / low-contrast images)

Frequency domain:
    - 2D DFT / FFT
    - Ideal Low-Pass / High-Pass Filter (ILPF / IHPF)
    - Gaussian Low-Pass / High-Pass Filter (GLPF / GHPF)
    - Butterworth Low-Pass / High-Pass Filter (BLPF / BHPF)
    - Laplacian-in-frequency-domain sharpening

Also includes:
    - Automatic problem detection (dark / bright / low-contrast / blurred /
      noisy) so a single `auto_enhance()` call can route an arbitrary image
      to the most suitable technique.
    - Simple quality metrics (PSNR, contrast/entropy) for before/after
      comparison.

Author: (project generated for course assignment on Image Enhancement)
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


# --------------------------------------------------------------------------
# 0. Utility functions
# --------------------------------------------------------------------------

def to_float(img):
    """Convert an image (uint8, 0-255) to float64 in range [0, 255]."""
    return img.astype(np.float64)


def to_uint8(img):
    """Clip to [0,255] and convert back to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def to_gray(img):
    """Convert an RGB image to grayscale (if needed)."""
    img = np.asarray(img)
    if img.ndim == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] +
                0.114 * img[..., 2])
    return img.astype(np.float64)


def psnr(original, processed):
    """Peak Signal-to-Noise Ratio between two same-shape images."""
    original = to_float(original)
    processed = to_float(processed)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def entropy(img):
    """Shannon entropy of the image histogram - a proxy for information/contrast."""
    gray = to_gray(img).astype(np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def rms_contrast(img):
    """RMS contrast = standard deviation of pixel intensities."""
    return float(np.std(to_gray(img)))


# --------------------------------------------------------------------------
# 1. Generic spatial convolution (as taught: rotate kernel 180deg then slide)
# --------------------------------------------------------------------------

def apply_kernel(img, kernel, per_channel=True):
    """
    Direct implementation of the convolution shown in the slides:
        G = H * F  (kernel rotated 180 deg, slid across image)
    Uses scipy.signal.convolve2d with 'same' boundary handling for speed,
    matching the mathematical definition G[i,j] = sum H[u,v] F[i-u, j-v].
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    img = to_float(img)

    if img.ndim == 2 or not per_channel:
        return convolve2d(img, kernel, mode="same", boundary="symm")

    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = convolve2d(img[..., c], kernel, mode="same", boundary="symm")
    return out


# --------------------------------------------------------------------------
# 2. Point processing: for DARK / BRIGHT / LOW-CONTRAST images
# --------------------------------------------------------------------------

def gamma_correction(img, gamma):
    """
    Power-law (gamma) transform: g = 255 * (f/255)^gamma
        gamma < 1  -> brightens dark images (expands dark tones)
        gamma > 1  -> darkens bright / washed-out images
    """
    img = to_float(img)
    normalized = img / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return to_uint8(corrected)


def contrast_stretching(img, low_pct=2, high_pct=98):
    """
    Linear contrast stretching (min-max / percentile stretch).
    Maps the [low_pct, high_pct] percentile range of intensities to [0,255].
    Ideal for LOW-CONTRAST images whose histogram occupies a narrow band.
    """
    img = to_float(img)
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            lo, hi = np.percentile(img[..., c], [low_pct, high_pct])
            if hi - lo < 1e-6:
                hi = lo + 1
            out[..., c] = (img[..., c] - lo) * 255.0 / (hi - lo)
        return to_uint8(out)
    else:
        lo, hi = np.percentile(img, [low_pct, high_pct])
        if hi - lo < 1e-6:
            hi = lo + 1
        out = (img - lo) * 255.0 / (hi - lo)
        return to_uint8(out)


def histogram_equalization(img):
    """
    Classic global histogram equalization.
    For color images, equalizes the luminance (Y) channel only, in the
    YCrCb color space, to avoid color-shift artifacts.
    """
    img = np.asarray(img)
    if img.ndim == 2:
        return _equalize_channel(img)

    # Color: work in YCrCb, equalize Y, convert back
    import cv2
    ycrcb = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = _equalize_channel(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _equalize_channel(channel):
    channel = to_uint8(channel)
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / \
                 (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_masked, 0).astype("uint8")
    return cdf_final[channel]


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Better than global HE for images with uneven / local lighting
    (e.g. partially shadowed low-contrast images).
    """
    import cv2
    img = np.asarray(img)
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if img.ndim == 2:
        return clahe_op.apply(to_uint8(img))
    ycrcb = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = clahe_op.apply(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# --------------------------------------------------------------------------
# 3. Spatial smoothing (noise removal) & sharpening (deblurring)
# --------------------------------------------------------------------------

def mean_filter(img, ksize=3):
    """Averaging/box filter - simple smoothing (slide: 'Smoothing Linear Filters')."""
    kernel = np.ones((ksize, ksize)) / (ksize * ksize)
    return to_uint8(apply_kernel(img, kernel))


def gaussian_filter(img, sigma=1.0):
    """
    Gaussian smoothing filter: G_sigma = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/2*sigma^2)
    Removes high-frequency noise (low-pass) while preserving structure
    better than the mean filter.
    """
    img = to_float(img)
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[..., c] = ndimage.gaussian_filter(img[..., c], sigma=sigma)
        return to_uint8(out)
    return to_uint8(ndimage.gaussian_filter(img, sigma=sigma))


def median_filter(img, ksize=3):
    """
    Median filter - a local-statistic filter, excellent for removing
    impulsive (salt-and-pepper) noise without excessive blurring.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[..., c] = ndimage.median_filter(img[..., c], size=ksize)
        return out
    return ndimage.median_filter(img, size=ksize)


def laplacian_sharpen(img, weight=1.0):
    """
    Laplacian sharpening, following the slide's 2D Laplacian kernel
    [[0,1,0],[1,-4,1],[0,1,0]]:
        g(x,y) = f(x,y) - w * Laplacian(f)   (center coefficient negative)
    """
    kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=np.float64)
    lap = apply_kernel(img, kernel)
    sharpened = to_float(img) - weight * lap
    return to_uint8(sharpened)


def unsharp_mask(img, sigma=2.0, amount=1.0):
    """
    Unsharp masking, exactly the 'Sharpening revisited' slide:
        detail = original - blurred
        sharpened = original + alpha * detail
    Effective for BLURRED images: recovers edges/detail lost to blur.
    """
    img = to_float(img)
    blurred = to_float(gaussian_filter(to_uint8(img), sigma=sigma))
    detail = img - blurred
    sharpened = img + amount * detail
    return to_uint8(sharpened)


# --------------------------------------------------------------------------
# 4. Frequency-domain filtering (2D FFT based)
# --------------------------------------------------------------------------

def _distance_grid(shape):
    """D(u,v): distance of every frequency-domain point from the DC center."""
    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)


def ideal_filter(shape, d0, highpass=False):
    D = _distance_grid(shape)
    H = (D > d0).astype(np.float64) if highpass else (D <= d0).astype(np.float64)
    return H


def gaussian_freq_filter(shape, d0, highpass=False):
    D = _distance_grid(shape)
    H = np.exp(-(D ** 2) / (2 * (d0 ** 2)))
    return 1 - H if highpass else H


def butterworth_freq_filter(shape, d0, order=2, highpass=False):
    D = _distance_grid(shape)
    D[D == 0] = 1e-6  # avoid divide-by-zero at DC for highpass
    if highpass:
        H = 1 / (1 + (d0 / D) ** (2 * order))
    else:
        H = 1 / (1 + (D / d0) ** (2 * order))
    return H


def apply_freq_filter(img_gray, H):
    """
    Basic steps for filtering in the frequency domain (as in slide
    'Filtering in Frequency Domain: Basic Steps'):
        1-2. F(u,v) = FFT{f(x,y)}, shifted so DC is centered
        3.   G(u,v) = F(u,v) * H(u,v)
        4-5. g(x,y) = Re{ IFFT{ G(u,v) } }
    """
    F = np.fft.fftshift(np.fft.fft2(img_gray))
    G = F * H
    g = np.fft.ifft2(np.fft.ifftshift(G))
    return np.real(g)


def freq_lowpass(img, d0=40, kind="gaussian", order=2):
    """Apply a frequency-domain low-pass filter (denoise / smooth)."""
    gray = to_gray(img)
    if kind == "ideal":
        H = ideal_filter(gray.shape, d0)
    elif kind == "butterworth":
        H = butterworth_freq_filter(gray.shape, d0, order=order)
    else:
        H = gaussian_freq_filter(gray.shape, d0)
    return to_uint8(apply_freq_filter(gray, H))


def freq_highpass(img, d0=30, kind="gaussian", order=2, boost=1.0):
    """
    Apply a frequency-domain high-pass filter and add it back to the
    original (high-frequency emphasis) to sharpen a blurred image.
    """
    gray = to_gray(img)
    if kind == "ideal":
        H = ideal_filter(gray.shape, d0, highpass=True)
    elif kind == "butterworth":
        H = butterworth_freq_filter(gray.shape, d0, order=order, highpass=True)
    else:
        H = gaussian_freq_filter(gray.shape, d0, highpass=True)
    high = apply_freq_filter(gray, H)
    sharpened = gray + boost * high
    return to_uint8(sharpened)


def freq_spectrum_image(img):
    """Return a log-scaled magnitude spectrum, useful for visualization."""
    gray = to_gray(img)
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9) * 255
    return mag.astype(np.uint8)


# --------------------------------------------------------------------------
# 5. Automatic diagnosis + routing
# --------------------------------------------------------------------------

def diagnose(img):
    """
    Very simple heuristic diagnosis of the dominant defect in an image,
    based on mean brightness, RMS contrast, and high-frequency energy
    (Laplacian variance, a common blur metric).
    Returns one of: 'dark', 'bright', 'low_contrast', 'blurred', 'noisy', 'ok'
    """
    gray = to_gray(img)
    mean_brightness = gray.mean()
    contrast = gray.std()
    lap_var = ndimage.laplace(gray).var()  # low -> blurry
    # crude noise estimate: high-frequency energy from a median-filtered residual
    residual = gray - ndimage.median_filter(gray, size=3)
    noise_estimate = residual.std()

    if mean_brightness < 80:
        return "dark"
    if mean_brightness > 180:
        return "bright"
    if contrast < 35:
        return "low_contrast"
    if noise_estimate > 18:
        return "noisy"
    if lap_var < 120:
        return "blurred"
    return "ok"


def auto_enhance(img, verbose=True):
    """
    Detects the dominant problem in `img` and applies the most appropriate
    enhancement pipeline. Returns (enhanced_image, diagnosis_string).
    """
    problem = diagnose(img)
    if problem == "dark":
        out = gamma_correction(img, gamma=0.5)
    elif problem == "bright":
        out = gamma_correction(img, gamma=1.8)
    elif problem == "low_contrast":
        out = clahe(img, clip_limit=2.5)
    elif problem == "noisy":
        out = median_filter(img, ksize=3)
    elif problem == "blurred":
        out = unsharp_mask(img, sigma=2.0, amount=1.5)
    else:
        out = img
    if verbose:
        print(f"[auto_enhance] Detected problem: '{problem}' -> "
              f"applied {'no change' if problem == 'ok' else problem + ' correction'}")
    return out, problem
