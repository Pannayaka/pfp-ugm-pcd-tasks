| Test image | Method | Metric (before) | Metric (after) | Entropy (before) | Entropy (after) |
|---|---|---:|---:|---:|---:|
| Dark image | Gamma correction (γ=0.45) | 15.43 | 26.25 | 4.32 | 4.73 |
| Dark image | Histogram equalization | 15.43 | 69.04 | 4.32 | 4.50 |
| Bright image | Gamma correction (γ=2.2) | 55.66 | 77.12 | 4.98 | 5.18 |
| Bright image | Contrast stretching | 55.66 | 80.76 | 4.98 | 5.18 |
| Low-contrast image | Contrast stretching | 13.42 | 63.26 | 4.21 | 4.70 |
| Low-contrast image | Histogram equalization | 13.42 | 68.84 | 4.21 | 4.36 |
| Low-contrast image | CLAHE | 13.42 | 17.11 | 4.21 | 4.93 |
| Blurred image | Unsharp masking | 17.26 | 18.69 | 6.63 | 6.75 |
| Blurred image | Laplacian sharpening | 17.26 | 17.65 | 6.63 | 6.67 |
| Blurred image | Freq-domain high-boost | 17.26 | 15.99 | 6.63 | 6.93 |
| Noisy image | Median filter (3x3) | 17.03 | 26.84 | 5.61 | 5.55 |
| Noisy image | Mean filter (3x3) | 17.03 | 21.11 | 5.61 | 6.44 |
| Noisy image | Gaussian filter | 17.03 | 21.29 | 5.61 | 7.03 |