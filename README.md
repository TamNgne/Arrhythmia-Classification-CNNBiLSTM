# ECG Arrhythmia Classification — Chapman-Shaoxing (4-class)

> Multimodal arrhythmia prediction combining 12-lead ECG signal and demographic
> features (age, sex). Benchmarked on the **Chapman-Shaoxing** dataset (45,152
> 10-second records) with **4-class** classification: **AFIB / GSVT / SB / SR**.

Accepted at the **FLINS-ISKE 2026 Conference**.
                           
## 📁 Project Structure

Arrhythmia Classification/
├── Preprocessing/             # Signal + label preprocessing
│   ├── process_ecg.py
│   └── process_demographics.py
├── dataset/
│   └── data.py                # ECGDataset (loads .npy)
├── Models/
│   ├── encoder_ECG.py         # ECGResNet, ECG_CNN_BiLSTM
│   └── classifier_ECG.py
├── utils/
│   └── metrics.py             # F1, AUROC, confusion matrices          
├── scripts/
│   └── train.py
├── web_app/
│   ├── app.py                 # Flask inference UI
│   └── templates/index.html
├── checkpoints/               # Trained .pth (gitignored)
├── data/                      # Raw + processed (gitignored)
│   ├── raw/chapman/WFDBRecords/
│   ├── ConditionNames_SNOMED-CT.csv
│   └── denoised_signals/
│       └── denoised_chapman/<record_id>.npy
├── requirements.txt
└── README.md

##Setup

### 1. Clone repo

```bash
git clone 'https://github.com/TamNgne/Arrhythmia-Classification-CNNBiLSTM.git
cd Arrhythmia Classification
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
data/chapman/WFDBRecords/
```

Also download the SNOMED-CT mapping CSV (provided alongside the challenge):
```
data/chapman/ConditionNames_SNOMED-CT.csv
```

---


### Step 1 — Preprocess ECG signals (one-time, ~50 minutes on 8 cores)

```bash
python -m Preprocessing.process_ecg
```

Outputs:
- `data/processed/denoised_chapman/<record_id>.npy`       — denoised signal (12 × 5000)
- `data/processed/denoised_chapman/<record_id>_mask.npy`  — lead repair mask
- `data/processed/denoised_chapman/denoise_metrics.csv`   — SNR / PSNR per record × lead

### Step 2 — Build demographics + stratified split

```bash
python -m Preprocessing.process_demographics
```

Outputs:
- `data/demographics/demographics.csv`
- `data/demographics/train.csv`  (70% — 31,605 records)
- `data/demographics/val.csv`    (20% —  9,030 records)
- `data/demographics/test.csv`   (10% —  4,516 records)

### Step 3 — Train models (run all 4 experiments)

```bash
python train.py --exp ecg_only_baseline
python train.py --exp ecg_only_smote_tomek
python train.py --exp ecg_tab_baseline
python train.py --exp ecg_tab_smote_tomek
```

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

## Expected Results

### Denoising quality (Step 1)

| Metric  | Mean (across 12 leads × 45k records) |
|---|---|
| SNR (dB)   | 22 – 32 |
| PSNR (dB)  | 38 – 50 |
| RMSE       | ~ 0.03  |
| SNR (%)    | 93 – 98 |

## Web Inference App

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
## Contact
**Tam Nguyen** 
- 📧 tamnguyen.work@gmail.com
