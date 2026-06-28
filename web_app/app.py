import os
import io
import sys
import base64
import shutil
import tempfile
import traceback

import numpy as np
import copy
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ── Resolve project root (app is in web_app/) ───────────────────────
APP_DIR  = Path(__file__).resolve().parent          # .../web_app
BASE_DIR = APP_DIR.parent                            # .../<project_root>
sys.path.insert(0, str(BASE_DIR))                    # enable top-level imports

# ── Denoising pipeline (replaces utils.processing) ──────────────────
from Preprocessing.process_ecg import (
    read_record               as read_record_robust_for_failed_cases,
    impute_corrupted_leads    as repair_leads_without_blind_synthesis,
    clean_signal,
    _parse_hea_basic          as parse_chapman_hea_basic,
)
from Preprocessing.process_demographics import (
    _safe_int  as safe_int,
    _norm_sex,
    extract_hea_metadata
)

import wfdb
from Models.encoder_ECG import ECG_CNN_BiLSTM, ECGResNet
from scipy.signal import filtfilt, resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# CONFIG (EDIT THESE TO MATCH YOUR TRAINING)
# ============================================================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "web_app", "uploads")
MODEL_DIR     = os.path.join(BASE_DIR, "Checkpoints")

ALLOWED_EXTS = {"hea", "dat", "mat"}

NUM_CLASSES = 4
CLASS_NAMES = ["AFIB", "GSVT", "SB", "SR"]

# ---- Signal settings ----
FS_TARGET = 500
SIG_LEN   = 5000
NUM_LEADS = 12

# ---- Filter settings ----
BP_LOWCUT  = 0.5
BP_HIGHCUT = 40.0
BP_ORDER   = 4

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH_ECGTAB = os.path.join(MODEL_DIR, "ecg_tab_smote_tomek_stage2.pth")
CKPT_PATH_ECG    = os.path.join(MODEL_DIR, "ecg_only_baseline_stage1_best.pth")


# ============================================================
# APP INIT
# ============================================================
app = Flask(__name__,
            template_folder=str(APP_DIR / "templates"),
            static_folder  =str(APP_DIR / "static"))
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR,     exist_ok=True)


def preprocess_signal(x, fs_orig):
    """
    Run the same denoising pipeline as training (process_ecg.clean_signal).
    Input  : raw signal  (12, N)
    Output : cleaned     (12, 5000) float32
    """
    cleaned, _ref = clean_signal(x, fs_orig)
    return cleaned


def preprocess_tabular(age_raw, sex_raw):
    """Build [[sex, age]] feature vector — same as utils.processing.preprocess_tabular."""
    sex_norm = _norm_sex(sex_raw)
    sex      = int(sex_norm) if sex_norm is not None else 0
    age      = float(safe_int(age_raw)) if safe_int(age_raw) is not None else 0.0
    return np.array([[sex, age]], dtype=np.float32)


cnn = ECGResNet(input_channels=12)
# --- Load ECG-tab model ---
model_both = ECG_CNN_BiLSTM(
    cnn_encoder = cnn,
    lstm_hidden = 256,
    lstm_layers = 2,
    tabular_dim = 2,
    num_classes = NUM_CLASSES,
    dropout     = 0.3,
).to(DEVICE)

# --- Load ECG-only ---
model_ecg = ECG_CNN_BiLSTM(
    cnn_encoder = copy.deepcopy(cnn),
    lstm_hidden = 256,
    lstm_layers = 2,
    tabular_dim = 0,
    num_classes = NUM_CLASSES,
    dropout     = 0.3,
).to(DEVICE)

if os.path.isfile(CKPT_PATH_ECGTAB):
    state = torch.load(CKPT_PATH_ECGTAB, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model_both.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model_both.load_state_dict(state["model"], strict=True)
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH_ECGTAB}")
else:
    print(f"[WARN] No checkpoint found at: {CKPT_PATH_ECGTAB}")
    print("[WARN] Predictions will be meaningless until you provide a trained model.")

model_both.eval()

if os.path.isfile(CKPT_PATH_ECG):
    state = torch.load(CKPT_PATH_ECG, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model_ecg.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model_ecg.load_state_dict(state, strict=True)
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH_ECG}")
else:
    print(f"[WARN] No checkpoint found at: {CKPT_PATH_ECG}")
    print("[WARN] Predictions will be meaningless until you provide a trained model.")

model_ecg.eval()


# ============================================================
# PLOTTING
# ============================================================
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_ecg(signal, fs, ch_names, title="ECG"):
    """
    Plot 12-lead ECG and return as base64 PNG string.
    """
    n_leads = signal.shape[0]
    t = np.arange(signal.shape[1]) / fs
    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 1.2 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, signal[i], linewidth=0.7, color="black")
        ax.set_ylabel(ch_names[i] if i < len(ch_names) else f"ch{i}",
                      rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xticks(np.arange(0, t[-1] + 1, 1))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ═══════════════════════════════════════════════
        # 1. Receive uploaded file(s)
        # ═══════════════════════════════════════════════
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        files = request.files.getlist("file")
        hea_path = None
        for f in files:
            if not f or not f.filename:
                continue
            name = secure_filename(f.filename)
            ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in ALLOWED_EXTS:
                continue
            dst = os.path.join(UPLOAD_FOLDER, name)
            f.save(dst)
            if ext == "hea":
                hea_path = Path(dst)

        if hea_path is None:
            return jsonify({"error": "No .hea file uploaded"}), 400

        # ═══════════════════════════════════════════════
        # 2. Read + repair + denoise (same as training)
        # ═══════════════════════════════════════════════
        raw_sig, fs_orig, ch_names = read_record_robust_for_failed_cases(hea_path)
        sig, lead_mask, repair_log = repair_leads_without_blind_synthesis(raw_sig, ch_names)
        cleaned                    = preprocess_signal(sig, fs_orig)

        # ═══════════════════════════════════════════════
        # 3. Tabular features from .hea header
        # ═══════════════════════════════════════════════
        meta = extract_hea_metadata(hea_path)
        tab  = preprocess_tabular(meta.get("age"), meta.get("sex"))
        print(f"debug: meta: {meta}, tabular features: {tab}, lead_mask: {lead_mask}, repair_log: {repair_log}")

        # ═══════════════════════════════════════════════
        # 4. Inference
        # ═══════════════════════════════════════════════
        ecg_t = torch.from_numpy(cleaned).unsqueeze(0).to(DEVICE).float()
        tab_t = torch.from_numpy(tab).to(DEVICE).float()

        with torch.no_grad():
            logits_ecg  = model_ecg(ecg_t)
            probs_ecg   = torch.softmax(logits_ecg, dim=1).cpu().numpy()[0]
            logits_both = model_both(ecg_t, tab_t)
            probs_both  = torch.softmax(logits_both, dim=1).cpu().numpy()[0]

        # ═══════════════════════════════════════════════
        # 5. Visualizations
        # ═══════════════════════════════════════════════
        plot_raw   = plot_ecg(raw_sig, fs_orig,  ch_names,  title="Raw ECG")
        plot_clean = plot_ecg(cleaned, FS_TARGET, LEAD_NAMES, title="Preprocessed ECG")

        return jsonify({
                        "record_id":   hea_path.stem,
                        "age":         float(tab[0, 1]),
                        "sex":         int(tab[0, 0]),
                        "fs_orig":     int(fs_orig),
                        "n_leads":     int(raw_sig.shape[0]),
                        "n_samples_raw":       int(raw_sig.shape[1]),
                        "n_samples_processed": int(cleaned.shape[1]),
                        "lead_repair": {
                                    "n_repaired": int((lead_mask == 0).sum()),
                                    "actions":    {LEAD_NAMES[i] if i < NUM_LEADS else f"ch{i}": act
                                    for i, act in repair_log.items()},
                        },
                        "prediction": {
                        "ecg_only": {CLASS_NAMES[i]: float(probs_ecg[i])  for i in range(NUM_CLASSES)},
                        "ecg_tab":  {CLASS_NAMES[i]: float(probs_both[i]) for i in range(NUM_CLASSES)},
                        "pred_ecg_only": CLASS_NAMES[int(np.argmax(probs_ecg))],
                        "pred_ecg_tab":  CLASS_NAMES[int(np.argmax(probs_both))],
                
                        },
                        "plot_raw":   plot_raw,
                        "plot_clean": plot_clean,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
