# Digital Image Processing — Semester Assignments

This repository contains all assignments given throughout the semester for the
**Digital Image Processing** (Pengolahan Citra Digital) course — Computer
Science program, Semester 3.

Each assignment lives in its own folder, self-contained with its own notebook,
report, test images, and generated outputs, so any single assignment can be
opened and run independently.

## Repository Structure

```
.
├── README.md                    <- you are here (course-level index)
├── PCD_Assignment01/             <- Down-Sampling & Up-Sampling
│   ├── PCD_Assignment01.ipynb
│   ├── README.md
│   ├── REPORT.md
│   ├── requirements.txt
│   ├── images/
│   └── outputs/
├── PCD_Assignment02/              <- (added when assigned)
├── PCD_Assignment03/              <- (added when assigned)
└── ...
```

As new assignments are given, a new `PCD_AssignmentNN/` folder is added
following the same pattern, and a new row is added to the table below.

## Assignment Index

| # | Title | Topic | Status | Link |
|---|---|---|---|---|
| 01 | Image Down-Sampling & Up-Sampling | Max/Average/Median pooling, Nearest Neighbor/Bilinear/Bicubic interpolation | ✅ Done | [PCD_Assignment01/](./PCD_Assignment01) |
| 02 | *TBA* | *TBA* | ⏳ Not started | — |
| 03 | *TBA* | *TBA* | ⏳ Not started | — |

## Conventions Used in This Repository

- **Notebook naming:** each assignment's main notebook is named
  `PCD_AssignmentNN.ipynb` (matching the assignment number), so it can be run
  directly from Google Colab.
- **Report:** every assignment includes a `REPORT.md` with a written analysis
  (objective, methodology, results and discussion, conclusion), independent of
  the notebook itself.
- **Images:** any image used as input for an assignment is placed in that
  assignment's `images/` subfolder.
- **Outputs:** all generated figures, charts, and result tables (e.g. `.csv`
  metrics) are placed in that assignment's `outputs/` subfolder, not committed
  loose at the repo root.
- **Reproducibility:** every notebook is written to run top-to-bottom via
  **Runtime → Run all** in Google Colab without manual intervention, including
  automatic installation of required libraries.

## Tools & Environment

- **Platform:** Google Colab
- **Language:** Python 3
- **Common libraries:** OpenCV, NumPy, Matplotlib, scikit-image, Pandas
  (exact list per assignment is listed in each folder's `requirements.txt`)

## Author

- **Name:** *Pannayaka Janggleng Renggo Loekito*
- **Student ID (NIM):** *25/554804/PA/23241*
- **Program:** Computer Science, Semester 3
- **Course:** Digital Image Processing (Pengolahan Citra Digital)
- **Lecturer:** *Mr. Why*
