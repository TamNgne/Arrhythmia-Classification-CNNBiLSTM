import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    ECG dataset for Chapman 4-class classification.

    Loads preprocessed signals saved by `process_ecg.py`:
        <ecg_folder>/<record_id>.npy        shape (12, 5000), float32
        <ecg_folder>/<record_id>_mask.npy   shape (12,)    uint8  (optional)

    Args:
        diagnosis_file : CSV with columns [record_id, label or label_enc, ...]
        ecg_folder     : folder containing *.npy files
        feature_cols   : list of tabular columns (e.g. ["sex", "age"]).
                         If None or [] → RAW-ONLY (no tabular features).
        label_map      : optional {label_str -> int}.  If absent, build from
                         the `label` column.
    """
    SIG_LEN  = 5000
    N_LEADS  = 12

    def __init__(self, diagnosis_file, ecg_folder,
                 feature_cols=None, label_map=None):
        self.df         = pd.read_csv(diagnosis_file)
        self.ecg_folder = ecg_folder
        self.df["record_id"] = self.df["record_id"].astype(str)

        if feature_cols is None:
            feature_cols = []
        self.feature_cols = feature_cols

        # ── Label encoding ───────────────────────────────────────
        if "label_enc" in self.df.columns:
            self.labels = self.df["label_enc"].values.astype(int)
            if label_map is not None:
                self.label_map = label_map
            else:
                n_cls = int(self.df["label_enc"].nunique())
                self.label_map = {str(i): i for i in range(n_cls)}
        elif "label" in self.df.columns:
            if label_map is None:
                unique = sorted(self.df["label"].unique())
                label_map = {lab: i for i, lab in enumerate(unique)}
            self.label_map = label_map
            self.labels    = self.df["label"].map(label_map).values.astype(int)
        else:
            # legacy fallback: 'dx' column with raw strings
            unique = sorted(self.df["dx"].unique())
            self.label_map = {lab: i for i, lab in enumerate(unique)}
            self.labels    = self.df["dx"].map(self.label_map).values.astype(int)

        self.idx_to_label = {idx: lab for lab, idx in self.label_map.items()}

        # ── Tabular features (optional) ──────────────────────────
        if len(self.feature_cols) > 0:
            self.features = self.df[self.feature_cols].values.astype(np.float32)
        else:
            self.features = np.zeros((len(self.df), 0), dtype=np.float32)

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    # --------------------------------------------------------------
    def _load_ecg(self, file_id):
        """
        Load preprocessed ECG from .npy file.

        Expected:
            shape (12, T) float32   (already denoised + length-standardized
                                     by process_ecg.py)
        Robust fallbacks:
            - transpose if shape is (T, 12)
            - pad / truncate to SIG_LEN (5000) just in case
            - replace NaN / Inf with 0.0
        """
        npy_path = os.path.join(self.ecg_folder, f"{file_id}.npy")
        ecg = np.load(npy_path).astype(np.float32)

        # Shape handling
        if ecg.ndim == 1:
            ecg = ecg.reshape(1, -1)
        if ecg.shape[0] != self.N_LEADS and ecg.shape[1] == self.N_LEADS:
            ecg = ecg.T

        # Length handling
        if ecg.shape[1] < self.SIG_LEN:
            ecg = np.pad(ecg, ((0, 0), (0, self.SIG_LEN - ecg.shape[1])))
        elif ecg.shape[1] > self.SIG_LEN:
            ecg = ecg[:, :self.SIG_LEN]

        # NaN / Inf guard
        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        return ecg

    # --------------------------------------------------------------
    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        ecg      = self._load_ecg(row["record_id"])
        features = self.features[idx]
        label    = self.labels[idx]
        return {
            "ecg":     torch.tensor(ecg,      dtype=torch.float32),
            "feature": torch.tensor(features, dtype=torch.float32),
            "label":   torch.tensor(label,    dtype=torch.long),
        }
