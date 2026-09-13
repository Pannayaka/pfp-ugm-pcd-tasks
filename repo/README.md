# PCD Assignment 01 — Image Down-Sampling & Up-Sampling

Tugas 1 — Pengolahan Citra Digital (Ilmu Komputer, Semester 3).

Implementasi dan analisis perbandingan metode:
- **Down Sampling:** Max Pooling, Average Pooling, Median Pooling
- **Up Sampling:** Nearest Neighbor, Bilinear, Bicubic

## Isi Repository

```
PCD_Assignment01.ipynb   -> Notebook utama (buka & jalankan di Google Colab)
REPORT.md                -> Laporan analisis lengkap (min. 1 halaman)
images/                  -> Citra uji asli (astronaut, cameraman, checkerboard)
outputs/                 -> Hasil visualisasi & metrik kuantitatif
  ├── downsampling_astronaut.png
  ├── downsampling_checkerboard.png
  ├── downsampling_comparison.png
  ├── upsampling_camera.png
  ├── upsampling_zoom_astronaut.png
  ├── upsampling_comparison.png
  ├── psnr_barchart.png
  └── metrics_results.csv     -> Tabel PSNR/SSIM untuk semua kombinasi metode
```

## Cara Menjalankan

1. Buka `PCD_Assignment01.ipynb` di [Google Colab](https://colab.research.google.com/) (File → Upload notebook, atau buka langsung dari GitHub via Colab: `File > Open notebook > GitHub` lalu tempel URL repo ini).
2. Jalankan seluruh sel secara berurutan: **Runtime → Run all**.
3. Notebook otomatis memuat citra uji, menjalankan seluruh eksperimen down-sampling & up-sampling, menghasilkan visualisasi, serta menghitung metrik PSNR/SSIM.

## Ringkasan Hasil

| Down-Sampling | Rata-rata PSNR (dB) | Rata-rata SSIM |
|---|---|---|
| Max Pooling | 14.58 | 0.59 |
| Average Pooling | 20.62 | 0.66 |
| Median Pooling | 20.26 | 0.69 |

| Up-Sampling | Rata-rata PSNR (dB) | Rata-rata SSIM |
|---|---|---|
| Nearest Neighbor | 17.56 | 0.64 |
| Bilinear | 18.74 | 0.64 |
| Bicubic | 19.16 | 0.66 |

Lihat `REPORT.md` untuk analisis lengkap.
