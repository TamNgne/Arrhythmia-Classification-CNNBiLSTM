# ECG Arrhythmia Classification — Chapman-Shaoxing (4-class)

> Multimodal arrhythmia prediction combining 12-lead ECG signal and demographic
> features (age, sex). Benchmarked on the **Chapman-Shaoxing** dataset (45,151
> 10-second records) with **4-class** classification: **AFIB / GSVT / SB / SR**.

Accepted at the **ISKE 2026 Conference**.

---

## 📋 Table of Contents
- [Pipeline Overview](#-pipeline-overview)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Expected Results](#-expected-results)
- [Web Inference App](#-web-inference-app)
- [Citation](#-citation)

---

## 🔄 Pipeline Overview

```
                                Chapman-Shaoxing
                          (PhysioNet Challenge 2020)
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────┐
            │  preprocessing/process_ecg.py                     │
            │    raw .hea/.mat                                  │
            │    → read                                          │
            │    → impute corrupted leads (limb-lead recovery)  │
            │    → clean (resample → length → normalize         │
            │             → bandpass → SWT denoise)             │
            │    → save .npy  (12 × 5000 float32)               │
            └──────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────────────────────────────┐
            │  preprocessing/process_demographics.py            │
            │    .hea  → extract age / sex / SNOMED Dx          │
            │          → map codes → 11-class → 4-class label   │
            │          → impute age/sex                          │
            │          → stratified split 70/20/10              │
            └──────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────┐
            │  scripts/03_train_chapman.py                      │
            │    Stage 1: ECG_CNN_BiLSTM end-to-end             │
            │    Stage 2: freeze encoder                        │
            │             → SMOTE-Tomek on embeddings (512-D)  │
            │             → train head (512 → 128 → 4)          │
            └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                            AFIB | GSVT | SB | SR
```

---

## 📁 Project Structure

```
ECG-Chapman-Thesis/
├── preprocessing/             # Signal + label preprocessing
│   ├── process_ecg.py
│   └── process_demographics.py
├── dataset/
│   └── data.py                # ECGDataset (loads .npy)
├── models/
│   ├── encoder_ECG.py         # ECGResNet, ECG_CNN_BiLSTM
│   └── classifier_ECG.py
├── utils/
│   ├── metrics.py             # F1, AUROC, confusion matrices
│   └── loss.py                # FocalLoss, AsymmetricLoss
├── scripts/
│   ├── 01_preprocess_ecg.py
│   ├── 02_preprocess_demographics.py
│   ├── 03_train_chapman.py
│   └── 04_evaluate_all.py
├── web_app/
│   ├── app.py                 # Flask inference UI
│   └── templates/index.html
├── checkpoints/               # Trained .pth (gitignored)
├── data/                      # Raw + processed (gitignored)
│   ├── raw/chapman/WFDBRecords/
│   ├── snomed_ct_grouped.csv
│   └── processed/
│       ├── chapman_dataset/{train,val,test,demographics}.csv
│       └── denoised_chapman/<record_id>.npy
├── results/                   # Confusion matrices, metric tables
├── notebooks/                 # Exploration + figure generation
├── docs/                      # Thesis PDF, poster, references
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone <repo-url>
cd ECG-Chapman-Thesis
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Chapman dataset

Download from **PhysioNet Challenge 2020**:
https://physionet.org/content/challenge-2020/1.0.2/

Extract Chapman records to:
```
data/raw/chapman/WFDBRecords/
```

Also download the SNOMED-CT mapping CSV (provided alongside the challenge):
```
data/snomed_ct_grouped.csv
```

---

## 🚀 Usage

### Step 1 — Preprocess ECG signals (one-time, ~50 minutes on 8 cores)

```bash
python -m preprocessing.process_ecg
```

Outputs:
- `data/processed/denoised_chapman/<record_id>.npy`       — denoised signal (12 × 5000)
- `data/processed/denoised_chapman/<record_id>_mask.npy`  — lead repair mask
- `data/processed/denoised_chapman/denoise_metrics.csv`   — SNR / PSNR per record × lead

### Step 2 — Build demographics + stratified split

```bash
python -m preprocessing.process_demographics
```

Outputs:
- `data/processed/chapman_dataset/demographics.csv`
- `data/processed/chapman_dataset/train.csv`  (70% — 31,605 records)
- `data/processed/chapman_dataset/val.csv`    (20% —  9,030 records)
- `data/processed/chapman_dataset/test.csv`   (10% —  4,516 records)

### Step 3 — Train models (run all 4 experiments)

```bash
python -m scripts.03_train_chapman --exp ecg_only_baseline
python -m scripts.03_train_chapman --exp ecg_only_smote_tomek
python -m scripts.03_train_chapman --exp ecg_tab_baseline
python -m scripts.03_train_chapman --exp ecg_tab_smote_tomek
```

### Step 4 — Aggregate results

```bash
python -m scripts.04_evaluate_all
```

Output: `results/tables/final_comparison.csv` — used directly for Thesis Chapter 4.

---

## 🧪 Experiments

| Experiment | Encoder Input | Tabular | SMOTE-Tomek | Head |
|---|---|---|---|---|
| `ecg_only_baseline`     | ECG (12 × 5000) |  ❌  |  ❌  | 512 → 128 → 4 |
| `ecg_only_smote_tomek`  | ECG             |  ❌  |  ✅  | 512 → 128 → 4 |
| `ecg_tab_baseline`      | ECG + (age, sex) |  ✅  |  ❌  | 514 → 128 → 4 |
| `ecg_tab_smote_tomek`   | ECG + (age, sex) |  ✅  |  ✅  | 514 → 128 → 4 |

### Training stages

- **Stage 1**: end-to-end supervised training (`ECG_CNN_BiLSTM`) on imbalanced
  data, optimized with `AdamW + CosineAnnealingLR` + early stopping on val macro-F1.
- **Stage 2** *(only for `*_smote_tomek`)*: freeze encoder → extract 512-D
  embeddings → apply **SMOTE-Tomek on embedding space** → train a new MLP head
  (512 → 128 → 4). SMOTE is applied on embeddings (not raw signal) because
  raw 60k-D features make k-NN intractable and would synthesize non-physiological
  ECG waveforms.

---

## 📊 Expected Results

### Denoising quality (Step 1)

| Metric  | Mean (across 12 leads × 45k records) |
|---|---|
| SNR (dB)   | 22 – 32 |
| PSNR (dB)  | 38 – 50 |
| RMSE       | ~ 0.03  |
| SNR (%)    | 93 – 98 |

### Classification (Test set — 4,516 records)

Reference benchmark: Yıldırım et al. 2020 reports ~94% accuracy on Chapman 4-class
with SWT denoising + CNN. Results are reported per experiment in
`results/tables/`.

---

## 🌐 Web Inference App

A Flask UI lets you upload a `.hea + .mat` pair, runs the same preprocessing
pipeline used in training, and displays both raw + denoised waveforms plus
4-class predictions from both `ecg_only` and `ecg_tab` models.

```bash
python web_app/app.py
# open http://localhost:5000
```

Endpoint `POST /predict` returns:

```json
{
  "record_id": "JS00001",
  "age": 62.0,
  "sex": 1,
  "prediction": {
    "ecg_only":      {"AFIB": 0.05, "GSVT": 0.10, "SB": 0.80, "SR": 0.05},
    "ecg_tab":       {"AFIB": 0.03, "GSVT": 0.08, "SB": 0.85, "SR": 0.04},
    "pred_ecg_only": "SB",
    "pred_ecg_tab":  "SB"
  },
  "lead_repair": {"n_repaired": 1, "actions": {"III": "reconstruct_III"}},
  "plot_raw":   "<base64 PNG>",
  "plot_clean": "<base64 PNG>"
}
```

---

## 📚 Citation

If you use this code or pipeline, please cite:

```bibtex
@inproceedings{nguyen2026multimodal,
  title     = {Multimodal Arrhythmia Prediction with ECG and Demographic
               Features on the Chapman-Shaoxing Dataset},
  author    = {Nguyen, Tam},
  booktitle = {Proc. ISKE 2026},
  year      = {2026}
}
```

---

## 📝 License

For academic / research use only. Chapman-Shaoxing data is governed by the
PhysioNet Credentialed Health Data License — see the dataset page for details.

---

## 🙋 Contact

**Tam Nguyen** — Data Scientist & Data Activation Engineer
- 📧 your.email@example.com
- 🏫 Thesis advisor: [advisor name]
