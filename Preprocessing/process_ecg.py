"""
ECG processing pipeline (Chapman-Shaoxing, 12-lead, 500 Hz).

Flow (matches original web-app pipeline):
    raw .hea/.mat  ->  read  ->  impute corrupted leads
                   ->  clean (resample -> length -> normalize -> bandpass -> SWT)
                   ->  save .npy  (shape: 12 x 5000, float32)
                   ->  compute denoising metrics (SNR / PSNR / MSE / RMSE / SNR%)

Reference for denoising metrics = bandpass output (pre-SWT).
SWT denoise: hard-threshold @ k*sigma, with amplitude restoration.

"""

import os
from pathlib import Path
from math import gcd
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import pywt
import wfdb
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly
from joblib import Parallel, delayed
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
TARGET_FS         = 500
TARGET_LENGTH     = 5000          # 10 s @ 500 Hz
LOWCUT, HIGHCUT   = 0.5, 40.0
WAVELET           = "rbio3.9"
SWT_LEVEL         = 3
SWT_K             = 3.0           # threshold = k * sigma  (hard)
NORM_METHOD       = "zscore"
LENGTH_STRATEGY   = "center_crop"

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


# ============================================================
# 1. READ  --  load raw signal from .hea / .mat
# ============================================================
def _parse_hea_basic(hea_path: Path) -> Dict:
    info = {"record_id": hea_path.stem, "fs": TARGET_FS,
            "ch_names": list(STANDARD_12_LEADS)}
    try:
        with open(hea_path, "r", errors="replace") as f:
            first = f.readline().split()
            if len(first) >= 3:
                info["fs"] = int(float(first[2]))
    except Exception:
        pass
    return info


def _load_mat(mat_path: Path) -> np.ndarray:
    data = loadmat(str(mat_path))
    for key in ("val", "ECG", "ecg", "data", "signal"):
        if key in data:
            sig = np.asarray(data[key], dtype=np.float32)
            if sig.ndim == 1:
                sig = sig.reshape(1, -1)
            if sig.shape[0] > sig.shape[1]:
                sig = sig.T
            return sig
    raise ValueError(f"No signal key in {mat_path}")


def read_record(hea_path: Path) -> Tuple[np.ndarray, int, List[str]]:
    """Read one Chapman record -> (signal[12, N], fs, ch_names)."""
    hea_path = Path(hea_path)
    try:
        rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
        sig = rec.p_signal.T.astype(np.float32)        # (12, N)
        return sig, int(rec.fs), list(rec.sig_name)
    except Exception:
        info = _parse_hea_basic(hea_path)
        sig  = _load_mat(hea_path.with_suffix(".mat"))
        return sig, info["fs"], info["ch_names"]


# ============================================================
# 2. IMPUTE  --  detect & repair corrupted leads
# ============================================================
def _is_bad_lead(x: np.ndarray) -> bool:
    if x is None or len(x) == 0:                          return True
    if np.all(np.isnan(x)) or np.all(np.isinf(x)):        return True
    if np.nanstd(x) < 1e-6:                               return True
    if len(np.unique(x[~np.isnan(x)])) / len(x) < 1e-3:   return True
    return False


def _lead_idx(ch_names: List[str], name: str) -> Optional[int]:
    name = name.lower().replace(" ", "")
    for i, n in enumerate(ch_names):
        if n.lower().replace(" ", "") == name:
            return i
    return None


def _reconstruct_limb(signal, ch_names, lead, valid_mask):
    """III = II-I ; aVR = -(I+II)/2 ; aVL = I-II/2 ; aVF = II-I/2"""
    iI, iII = _lead_idx(ch_names, "I"), _lead_idx(ch_names, "II")
    def ok(i): return i is not None and valid_mask[i]
    if   lead == "III" and ok(iI)  and ok(iII): return signal[iII] - signal[iI]
    elif lead == "aVR" and ok(iI)  and ok(iII): return -(signal[iI] + signal[iII]) / 2
    elif lead == "aVL" and ok(iI)  and ok(iII): return signal[iI]  - signal[iII] / 2
    elif lead == "aVF" and ok(iII) and ok(iI):  return signal[iII] - signal[iI]  / 2
    return None


def impute_corrupted_leads(signal: np.ndarray,
                           ch_names: List[str]
                           ) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Returns:
        signal_fixed (12, N)
        lead_mask    (12,)   1 = original, 0 = imputed/placeholder
        repair_log   {lead_idx: action}
    """
    n_leads, n_samp = signal.shape
    valid_mask  = np.array([not _is_bad_lead(signal[i]) for i in range(n_leads)])
    repair_log: Dict[int, str] = {}
    out = signal.copy()

    for i in range(n_leads):
        if valid_mask[i]:
            continue
        lead = ch_names[i] if i < len(ch_names) else STANDARD_12_LEADS[i]
        rec  = _reconstruct_limb(signal, ch_names, lead, valid_mask)
        if rec is not None and not _is_bad_lead(rec):
            out[i] = rec
            repair_log[i] = f"reconstruct_{lead}"
        else:
            valid = out[valid_mask] if valid_mask.any() else None
            out[i] = np.mean(valid, axis=0) if valid is not None \
                     else np.zeros(n_samp, dtype=np.float32)
            repair_log[i] = "placeholder"

    return out.astype(np.float32), valid_mask.astype(np.uint8), repair_log


# ============================================================
# 3. CLEAN  --  resample -> length -> normalize -> bandpass -> SWT
# ============================================================
def _resample(x, src_fs, dst_fs=TARGET_FS):
    if src_fs == dst_fs:
        return x.astype(np.float32)
    g = gcd(int(src_fs), int(dst_fs))
    return resample_poly(x, dst_fs // g, src_fs // g).astype(np.float32)


def _bandpass(x, fs, lo=LOWCUT, hi=HIGHCUT, order=4):
    nyq = 0.5 * fs
    b, a = butter(order,
                  [max(lo / nyq, 1e-5), min(hi / nyq, 1 - 1e-5)],
                  btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x.astype(np.float32)
    return filtfilt(b, a, x).astype(np.float32)


def _swt_denoise(x, wavelet=WAVELET, level=SWT_LEVEL, k=SWT_K):
    """
    Stationary Wavelet Transform denoising.  Matches original pipeline:
      - default `norm=False`           (avoids biorthogonal warning)
      - threshold = k * sigma_MAD      (no sqrt(2 log N) factor)
      - mode      = "hard"             (kill small detail coeffs only)
      - amplitude restoration after thresholding
    """
    x      = np.asarray(x, dtype=np.float32)
    n      = len(x)
    factor = 2 ** level
    pad    = int(np.ceil(n / factor) * factor) - n
    xp     = np.pad(x, (0, pad), mode="reflect") if pad else x

    coeffs = pywt.swt(xp, wavelet=wavelet, level=level)

    thresholded = []
    for cA, cD in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745
        thr   = k * sigma
        cD_t  = pywt.threshold(cD, thr, mode="hard")
        thresholded.append((cA, cD_t))

    rec = pywt.iswt(thresholded, wavelet=wavelet)[:n].astype(np.float32)

    # Amplitude restoration (preserves QRS amplitude after thresholding)
    if np.std(rec) > 0:
        rec = rec * float(np.std(x) / np.std(rec))
    return rec.astype(np.float32)


def _standardize_length(x, target_len=TARGET_LENGTH, strategy=LENGTH_STRATEGY):
    n = len(x)
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2 if strategy == "center_crop" else 0
        return x[start:start + target_len]
    diff = target_len - n
    left, right = diff // 2, diff - diff // 2
    return np.pad(x, (left, right), mode="constant", constant_values=0)


def _normalize(x, method=NORM_METHOD):
    if method == "zscore":
        mu, sd = float(np.mean(x)), float(np.std(x))
        return ((x - mu) / (sd if sd > 1e-12 else 1.0)).astype(np.float32)
    if method == "minmax":
        mn, mx = float(np.min(x)), float(np.max(x))
        rng = mx - mn if mx - mn > 1e-12 else 1.0
        return ((x - mn) / rng).astype(np.float32)
    return x.astype(np.float32)


def clean_signal(signal: np.ndarray, src_fs: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per lead pipeline:
        Resample -> Length -> Normalize -> Bandpass -> SWT denoise

    Returns:
        cleaned   (12, TARGET_LENGTH) -- model input (post-SWT, amplitude restored)
        reference (12, TARGET_LENGTH) -- bandpass output (pre-SWT)
                                         used as denoising-metric reference
    """
    out = np.zeros((signal.shape[0], TARGET_LENGTH), dtype=np.float32)
    ref = np.zeros_like(out)
    for i, lead in enumerate(signal):
        x     = _resample(lead, src_fs, TARGET_FS)
        x     = _standardize_length(x, TARGET_LENGTH)
        x     = _normalize(x, NORM_METHOD)        # normalize BEFORE bandpass
        x_bp  = _bandpass(x, TARGET_FS)           # reference for metrics
        x_swt = _swt_denoise(x_bp)                # final cleaned signal
        out[i] = x_swt
        ref[i] = x_bp
    return out, ref


# ============================================================
# 4. METRICS  --  signal-level denoising evaluation
# ============================================================
def _mse(orig: np.ndarray, denoised: np.ndarray) -> float:
    return float(np.mean((orig - denoised) ** 2))


def _rmse(orig: np.ndarray, denoised: np.ndarray) -> float:
    return float(np.sqrt(_mse(orig, denoised)))


def _snr(orig: np.ndarray, denoised: np.ndarray, eps: float = 1e-12) -> float:
    """SNR (dB) = 10 log10(P_signal / P_noise).  inf if perfect, nan if flat."""
    p_sig   = float(np.mean(orig ** 2))
    p_noise = float(np.mean((orig - denoised) ** 2))
    if p_sig   < eps: return float("nan")
    if p_noise < eps: return float("inf")
    return float(10.0 * np.log10(p_sig / p_noise))


def _psnr(orig: np.ndarray, denoised: np.ndarray, eps: float = 1e-12) -> float:
    """PSNR (dB) using robust 99.9-percentile peak."""
    peak    = float(np.percentile(np.abs(orig), 99.9))
    mse_val = _mse(orig, denoised)
    if peak    < eps: return float("nan")
    if mse_val < eps: return float("inf")
    return float(10.0 * np.log10(peak ** 2 / mse_val))


def _snr_percentage(orig: np.ndarray, denoised: np.ndarray,
                    eps: float = 1e-12) -> float:
    """SNR% = 100 * (1 - RMS_noise / RMS_signal). 100 = perfect, 0 = no improvement."""
    rms_sig   = float(np.sqrt(np.mean(orig ** 2)))
    rms_noise = float(np.sqrt(np.mean((orig - denoised) ** 2)))
    if rms_sig < eps: return 0.0
    return float(100.0 * (1.0 - rms_noise / rms_sig))


def evaluate_lead(orig: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
    """Compute all denoising metrics for one lead."""
    return {
        "snr_db":  _snr(orig, denoised),
        "psnr_db": _psnr(orig, denoised),
        "mse":     _mse(orig, denoised),
        "rmse":    _rmse(orig, denoised),
        "snr_pct": _snr_percentage(orig, denoised),
    }


def aggregate_metrics(per_record: Dict[str, Dict[str, Dict[str, float]]]
                     ) -> Dict[str, float]:
    """
    Aggregate metrics across records x leads.

    per_record : {record_id: {lead_name: {metric: value}}}
    Returns mean per metric, ignoring inf / nan.
    """
    bucket = {"snr_db": [], "psnr_db": [], "mse": [], "rmse": [], "snr_pct": []}
    for lead_dict in per_record.values():
        for m in lead_dict.values():
            for k in bucket:
                v = m.get(k, np.nan)
                if np.isfinite(v):
                    bucket[k].append(v)
    return {k: float(np.mean(v)) if v else float("nan")
            for k, v in bucket.items()}


# ============================================================
# 5. SAVE
# ============================================================
def save_record(record_name: str, signal: np.ndarray,
                lead_mask: np.ndarray, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{record_name}.npy",      signal.astype(np.float32))
    np.save(out_dir / f"{record_name}_mask.npy", lead_mask.astype(np.uint8))


# ============================================================
# Per-record pipeline
# ============================================================
def process_one_record(hea_path: Path, out_dir: Path,
                       compute_metrics: bool = True) -> Dict:
    hea_path = Path(hea_path)
    rec_id   = hea_path.stem
    try:
        sig, fs, ch_names = read_record(hea_path)
        sig, mask, log    = impute_corrupted_leads(sig, ch_names)
        cleaned, ref      = clean_signal(sig, fs)
        save_record(rec_id, cleaned, mask, out_dir)

        result = {
            "record_id":  rec_id,
            "status":     "ok",
            "n_repaired": int((mask == 0).sum()),
            "repair_log": log,
        }
        if compute_metrics:
            # Only score leads that are ORIGINAL (mask=1).
            # Imputed / placeholder leads would bias the metrics.
            lead_metrics = {}
            for i, name in enumerate(ch_names[:cleaned.shape[0]]):
                if mask[i] == 1:
                    lead_metrics[name] = evaluate_lead(ref[i], cleaned[i])
            result["metrics"] = lead_metrics
        return result
    except Exception as e:
        return {"record_id": rec_id, "status": "fail", "error": str(e)}


# ============================================================
# Batch
# ============================================================
def process_folder(folder_path: str, out_dir: str,
                   n_jobs: int = -1,
                   compute_metrics: bool = True,
                   metrics_csv: Optional[str] = None) -> Dict:
    hea_files = sorted(Path(folder_path).rglob("*.hea"))
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_record)(hp, Path(out_dir), compute_metrics)
        for hp in tqdm(hea_files, desc="Processing ECG")
    )

    ok_results = [r for r in results if r["status"] == "ok"]
    n_ok, n_fail = len(ok_results), len(results) - len(ok_results)
    print(f"[ECG] ok={n_ok}  fail={n_fail}  total={len(results)}")

    summary = {"results": results, "n_ok": n_ok, "n_fail": n_fail}

    # ---- Aggregate denoising metrics ------------------------
    if compute_metrics and ok_results:
        per_record = {r["record_id"]: r.get("metrics", {}) for r in ok_results}
        agg = aggregate_metrics(per_record)
        summary["aggregate_metrics"] = agg

        print("\n-- Denoising quality (bandpass-normalized vs SWT-denoised) --")
        for k, v in agg.items():
            print(f"  {k:>10s} : {v:>10.4f}")

        # Save per-record x per-lead metrics to CSV for thesis table
        if metrics_csv:
            rows = []
            for rec_id, leads in per_record.items():
                for lead, m in leads.items():
                    rows.append({"record_id": rec_id, "lead": lead, **m})
            pd.DataFrame(rows).to_csv(metrics_csv, index=False)
            print(f"  -> per-lead metrics saved: {metrics_csv}")

    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    CHAPMAN_ROOT = r"C:\Users\nguye\Project\ECG\input\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\chapman\WFDBRecords"
    OUT_DIR      = r"C:\Users\nguye\Project\ECG\test"
    METRICS_CSV  = r"C:\Users\nguye\Project\ECG\test\denoise_metrics.csv"

    process_folder(
        CHAPMAN_ROOT, OUT_DIR,
        n_jobs=-1,
        compute_metrics=True,
        metrics_csv=METRICS_CSV,
    )
