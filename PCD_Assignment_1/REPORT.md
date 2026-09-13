# Laporan Analisis — PCD Assignment 01
## Down Sampling & Up Sampling Citra Digital

**Mata Kuliah:** Pengolahan Citra Digital
**Semester:** 3 (Ilmu Komputer)
**Tools:** Google Colab, Python (OpenCV, NumPy, scikit-image, Matplotlib, Pandas)

---

## 1. Tujuan

Praktikum ini bertujuan untuk mengimplementasikan dan membandingkan performa
beberapa metode **down-sampling** (Max Pooling, Average Pooling, Median Pooling)
dan **up-sampling** (Nearest Neighbor, Bilinear, Bicubic) pada citra digital, serta
menganalisis pengaruh masing-masing metode terhadap kualitas visual dan kuantitatif
citra hasil.

## 2. Metodologi

Tiga citra uji dengan karakteristik berbeda digunakan agar analisis mencakup
berbagai jenis konten visual:

| Citra | Karakteristik |
|---|---|
| **Astronaut** (512×512, RGB) | Foto natural berwarna dengan tekstur halus, kulit, dan tepi objek jelas |
| **Cameraman** (512×512, grayscale) | Citra klasik dengan gradasi kontras dan detail halus |
| **Checkerboard** (200×200) | Pola kotak-kotak berulang berfrekuensi tinggi — cocok untuk mengamati *aliasing* |

**Down-sampling** diimplementasikan secara manual dengan membagi citra menjadi
blok berukuran `factor × factor` piksel, kemudian setiap blok direduksi menjadi
satu nilai piksel menggunakan operasi *max*, *mean* (average), atau *median*.

**Up-sampling** diimplementasikan menggunakan `cv2.resize()` dengan tiga jenis
interpolasi: `INTER_NEAREST` (NN), `INTER_LINEAR` (bilinear), dan `INTER_CUBIC`
(bicubic).

Setiap citra di-down-sample dengan faktor **4×** dan **8×**, kemudian di-up-sample
kembali ke ukuran semula. Kualitas hasil rekonstruksi dibandingkan terhadap citra
asli menggunakan dua metrik kuantitatif:
- **PSNR** (Peak Signal-to-Noise Ratio, dB) — semakin tinggi semakin mirip.
- **SSIM** (Structural Similarity Index, rentang 0–1) — semakin mendekati 1 semakin mirip secara struktural.

## 3. Hasil

### 3.1 Down Sampling
Secara visual (lihat `outputs/downsampling_astronaut.png` dan
`outputs/downsampling_checkerboard.png`):
- **Max pooling** membuat citra tampak lebih terang dan kehilangan detail bayangan,
  karena selalu memilih nilai piksel tertinggi pada tiap blok.
- **Average pooling** menghasilkan citra yang halus dan proporsional, mendekati
  hasil down-sampling standar pada umumnya.
- **Median pooling** memberikan hasil yang mirip average pooling pada citra
  bertekstur halus, tetapi jauh lebih baik dalam mempertahankan pola tegas pada
  citra checkerboard.

Secara kuantitatif (rata-rata seluruh citra & faktor skala):

| Metode Down-Sampling | Rata-rata PSNR (dB) | Rata-rata SSIM |
|---|---|---|
| Max Pooling | 14.58 | 0.59 |
| Average Pooling | 20.62 | 0.66 |
| Median Pooling | 20.26 | 0.69 |

*Nilai tepat dapat dilihat pada `metrics_results.csv` yang dihasilkan notebook.*

Max pooling secara konsisten memiliki PSNR/SSIM terendah di semua kondisi
pengujian, mengonfirmasi bahwa metode ini paling banyak mendistorsi informasi
citra asli akibat bias ke arah nilai piksel maksimum.

### 3.2 Up Sampling
Secara visual (lihat `outputs/upsampling_camera.png` dan
`outputs/upsampling_zoom_astronaut.png`):
- **Nearest Neighbor** menghasilkan efek kotak-kotak (*blocky/pixelated*) yang
  jelas terlihat, karena tidak melakukan interpolasi nilai piksel.
- **Bilinear** menghasilkan transisi warna yang jauh lebih halus dibanding NN.
- **Bicubic** memberikan hasil paling halus dan tampak paling tajam di antara
  ketiganya, terutama pada tepi objek.

Secara kuantitatif (rata-rata seluruh citra & faktor skala):

| Metode Up-Sampling | Rata-rata PSNR (dB) | Rata-rata SSIM |
|---|---|---|
| Nearest Neighbor (NN) | 17.56 | 0.64 |
| Bilinear | 18.74 | 0.64 |
| Bicubic | 19.16 | 0.66 |

Urutan performa ini (**NN < Bilinear < Bicubic**) konsisten pada seluruh kombinasi
citra dan faktor skala yang diuji.

## 4. Analisis Pengaruh Karakteristik Citra

Citra **checkerboard** (pola frekuensi tinggi) menunjukkan penurunan PSNR/SSIM
yang jauh lebih tajam dibanding citra natural (astronaut, cameraman) pada faktor
downsampling yang sama. Hal ini terjadi karena pola berulang dengan transisi
kontras tinggi sangat rentan terhadap **aliasing** — informasi frekuensi tinggi
"terlipat" menjadi pola baru yang salah ketika sampling rate diturunkan tanpa
low-pass filtering yang memadai. Sebaliknya, citra dengan gradasi warna halus
(astronaut, cameraman) lebih toleran terhadap penurunan resolusi karena tidak
banyak mengandung komponen frekuensi tinggi.

Semakin besar faktor down-sampling (8× dibanding 4×), semakin besar pula
penurunan kualitas pada seluruh kombinasi metode — sesuai ekspektasi teoritis
karena semakin banyak informasi piksel asli yang dibuang secara permanen dan
tidak dapat direkonstruksi sepenuhnya oleh interpolasi up-sampling apa pun.

## 5. Kesimpulan

1. Untuk **down-sampling**, **Average Pooling** dan **Median Pooling** memberikan
   hasil terbaik secara umum; **Median Pooling** lebih unggul pada citra dengan
   pola/tepi tegas, sedangkan **Average Pooling** sedikit lebih baik pada citra
   dengan gradasi halus. **Max Pooling** sebaiknya dihindari untuk kompresi/
   resize citra karena bias kecerahan yang signifikan, meskipun berguna dalam
   konteks lain seperti *feature extraction* pada CNN.
2. Untuk **up-sampling**, **Bicubic** memberikan kualitas visual dan kuantitatif
   terbaik namun dengan biaya komputasi tertinggi; **Nearest Neighbor** tercepat
   namun menghasilkan artefak blocky yang signifikan; **Bilinear** merupakan
   kompromi yang baik antara kecepatan dan kualitas.
3. Pemilihan metode terbaik bergantung pada **karakteristik citra** (halus vs.
   frekuensi tinggi) dan **kebutuhan aplikasi** (real-time vs. kualitas akhir).
   Tidak ada metode tunggal yang optimal untuk semua kasus.

---
*Seluruh kode, citra uji, dan hasil eksperimen (grafik & CSV) tersedia pada
notebook `PCD_Assignment01.ipynb` dan folder `outputs/` di repository ini.*
