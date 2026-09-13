# PCD Assignment 01 — Image Down-Sampling and Up-Sampling

Assignment 1 PCD

Implementation and comparative analysis of:
- **Down-Sampling:** Max Pooling, Average Pooling, Median Pooling
- **Up-Sampling:** Nearest Neighbor, Bilinear, Bicubic

## Repository Contents

```
PCD_Assignment01.ipynb   -> Main notebook (open & run in Google Colab)
REPORT.md                -> Full analysis report (min. 1 page)
images/                  -> Original test images (astronaut, cameraman, checkerboard)
outputs/                 -> Generated visualizations & quantitative metrics
  ├── downsampling_astronaut.png
  ├── downsampling_checkerboard.png
  ├── downsampling_comparison.png
  ├── upsampling_camera.png
  ├── upsampling_zoom_astronaut.png
  ├── upsampling_comparison.png
  ├── psnr_barchart.png
  └── metrics_results.csv     -> PSNR/SSIM table for every method combination
```

## How to Run

1. Open `PCD_Assignment01.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook, or open it directly from GitHub via Colab: `File > Open notebook > GitHub`, then paste this repo's URL).
2. Run all cells in order: **Runtime → Run all**.
3. The notebook automatically loads the test images, runs every down-sampling and up-sampling experiment, produces the visualizations, and computes the PSNR/SSIM metrics.

## Results Summary

| Down-Sampling | Average PSNR (dB) | Average SSIM |
|---|---|---|
| Max Pooling | 14.58 | 0.59 |
| Average Pooling | 20.62 | 0.66 |
| Median Pooling | 20.26 | 0.69 |

| Up-Sampling | Average PSNR (dB) | Average SSIM |
|---|---|---|
| Nearest Neighbor | 17.56 | 0.64 |
| Bilinear | 18.74 | 0.64 |
| Bicubic | 19.16 | 0.66 |

See `REPORT.md` for the full analysis.
