"""
Demographics pipeline (Chapman-Shaoxing, 4-class).

Flow:
    .hea  ->  extract age / sex / SNOMED-CT Dx codes
          ->  map SNOMED codes -> abbreviations (via snomed_ct.csv)
          ->  abbreviations  -> 11-class rhythm labels
          ->  11-class       -> 4-class final label (AFIB / GSVT / SB / SR)
          ->  impute missing age & sex
          ->  save demographics.csv (full audit columns)
          ->  stratified split  train / val / test  (70 / 20 / 10)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Config
# ============================================================
RANDOM_STATE   = 1
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.20
TEST_RATIO     = 0.10
DEFAULT_RHYTHM = "SR"          # fallback when NO rhythm info is found


# ============================================================
# Label vocabularies (from Chapman thesis convention)
# ============================================================
# --- 11-class rhythm vocabulary ---
RHYTHM_CLASSES_11 = [
    "SR", "SI", "SB",
    "AFIB", "AF",
    "ST", "SVT", "AT", "AVNRT", "AVRT", "SAAWR",
]
RHYTHM_SET_11 = set(RHYTHM_CLASSES_11)

# --- Abbreviation -> 11-class rhythm ---
ABBR_TO_RHYTHM_11 = {
    "NSR":   "SR",
    "SA":    "SI",
    "SB":    "SB",
    "AF":    "AFIB",
    "AFL":   "AF",
    "AFAFL": "AFIB",
    "CAF":   "AFIB",
    "PAF":   "AFIB",
    "RAF":   "AFIB",
    "STach": "ST",
    "SVT":   "SVT",
    "PSVT":  "SVT",
    "ATach": "AT",
    "AVNRT": "AVNRT",
    "AVRT":  "AVRT",
    "SAAWR": "SAAWR",
}

# --- Final 4-class set (pass-through if already final) ---
FINAL4_SET = {"AFIB", "GSVT", "SB", "SR"}

# --- 11-class -> 4-class collapse map ---
GSVT_SET = {"ST", "SVT", "AT", "AVNRT", "AVRT", "SAAWR"}

# --- Extra rhythm-like abbreviations -> 4-class (not in 11-class) ---
EXTRA_RHYTHM_TO_FINAL4 = {
    # Pacing
    "PR":    "SR",
    # Bradycardia variants
    "Brady": "SB", "SARR": "SB", "SAB": "SB", "SND": "SB",
    # Other supraventricular / atrial / junctional rhythms
    "ARH":  "GSVT", "AAR":  "GSVT", "AJR":  "GSVT", "AVJR": "GSVT",
    "JTach":"GSVT", "WAP":  "GSVT",
    # Ventricular rhythms -> bucket as GSVT ("other abnormal rhythm")
    "VF":   "GSVT", "VFL":  "GSVT", "VTach":"GSVT", "PVT":  "GSVT",
    "AIVR": "GSVT", "IR":   "GSVT", "VEsR": "GSVT", "VEsB": "GSVT",
}


# ============================================================
# 1. READ .hea  ->  age, sex, SNOMED Dx codes
# ============================================================
def _safe_int(x) -> Optional[int]:
    try:
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return int(float(s))
    except Exception:
        return None


def _norm_sex(s) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("male",   "m", "1"): return 1
    if s in ("female", "f", "0"): return 0
    return None


def extract_hea_metadata(hea_path: Path) -> Dict:
    """Parse age, sex, SNOMED-CT Dx codes from a .hea comment header."""
    meta = {"record_id": hea_path.stem,
            "age": None, "sex": None, "dx_codes": []}
    try:
        with open(hea_path, "r", errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    continue
                comment = line.lstrip("# ").strip()
                if ":" not in comment:
                    continue
                key, val = (s.strip() for s in comment.split(":", 1))
                key_lo = key.lower()
                if   key_lo == "age": meta["age"] = _safe_int(val)
                elif key_lo == "sex": meta["sex"] = _norm_sex(val)
                elif key_lo == "dx":
                    meta["dx_codes"] = [
                        c.strip() for c in val.split(",") if c.strip()
                    ]
    except Exception:
        pass
    return meta


# ============================================================
# 2. SNOMED lookup + abbreviation mapping
# ============================================================
def build_code_to_abbr(snomed_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build {SNOMED_CT_code -> Abbreviation} dictionary.

    Auto-detects column names from common variants:
        code column   : 'Snomed_CT' | 'SNOMED CT Code' | 'code'
        abbr column   : 'Acronym Name' | 'Abbreviation' | 'abbr'
    """
    cols = {c.lower().strip(): c for c in snomed_df.columns}
    code_col = next((cols[k] for k in
                     ("snomed_ct", "snomed ct code", "code", "snomedct")
                     if k in cols), None)
    abbr_col = next((cols[k] for k in
                     ("acronym name", "abbreviation", "abbr", "acronym")
                     if k in cols), None)
    if code_col is None or abbr_col is None:
        raise ValueError(
            f"SNOMED CSV must contain code + abbreviation columns. "
            f"Found columns: {list(snomed_df.columns)}"
        )
    return {
        str(row[code_col]).strip(): str(row[abbr_col]).strip()
        for _, row in snomed_df.iterrows()
        if pd.notna(row[code_col]) and pd.notna(row[abbr_col])
    }


def parse_tokens(x) -> List[str]:
    """Parse any label format: SNOMED codes, abbreviations, or final labels."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [t.strip() for t in s.split(",") if t.strip()]


def tokens_to_abbrs(tokens: List[str], code_to_abbr: Dict[str, str]) -> List[str]:
    """Numeric tokens -> SNOMED abbr ; non-numeric kept as-is."""
    out = []
    for t in tokens:
        t = str(t).strip()
        if not t:
            continue
        if re.fullmatch(r"\d{6,}", t):
            ab = code_to_abbr.get(t)
            if ab:
                out.append(ab)
        else:
            out.append(t)
    return out


def abbrs_to_rhythm11(abbrs: List[str]) -> List[str]:
    """Map abbreviations -> 11-class rhythm labels (dedupe preserve order)."""
    out = []
    for ab in abbrs:
        ab = str(ab).strip()
        if ab in RHYTHM_SET_11:
            out.append(ab); continue
        r = ABBR_TO_RHYTHM_11.get(ab)
        if r is not None:
            out.append(r)
    return list(dict.fromkeys(out))


def assign_final4(row) -> str:
    """
    Final 4-class label.  Priority:
      1) 11-class rhythm list  -> collapse to 4-class
      2) Direct final labels in tokens (AFIB/GSVT/SB/SR)
      3) Extra rhythm-like tokens (PR / Brady / VF / ARH / ...)
      4) Default -> SR (no rhythm info = sinus rhythm)
    """
    rhythm11 = row["label_list_11"]
    tokset   = {str(x).strip() for x in row["tokens"]}
    abbrset  = {str(x).strip() for x in row["abbrs"]}

    # Step 1: 11-class list
    if len(rhythm11) > 0:
        s = set(rhythm11)
        if "AFIB" in s or "AF" in s:        return "AFIB"
        if s & GSVT_SET:                    return "GSVT"
        if "SB" in s:                        return "SB"
        if "SR" in s or "SI" in s:           return "SR"

    # Step 2: already-final labels
    for cls in ("AFIB", "GSVT", "SB", "SR"):
        if cls in tokset:                    return cls

    # Step 3: extra rhythm-like tokens
    for tok in (tokset | abbrset):
        mapped = EXTRA_RHYTHM_TO_FINAL4.get(tok)
        if mapped is not None:               return mapped

    # Step 4: default
    return DEFAULT_RHYTHM


# ============================================================
# 3. IMPUTE missing age / sex
# ============================================================
def impute_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_missing"] = df["age"].isna().astype(np.int8)
    df["sex_missing"] = df["sex"].isna().astype(np.int8)
    df["age"] = df["age"].fillna(df["age"].median()).astype(np.float32)
    df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0]).astype(np.int8)
    return df


# ============================================================
# 4. BUILD demographics.csv
# ============================================================
def build_demographics(hea_dir: str,
                       snomed_csv: str,
                       out_csv: str) -> pd.DataFrame:
    """
    Walk all .hea files under `hea_dir`, extract metadata, map SNOMED codes
    -> abbreviations -> 11-class -> 4-class label, save full demographics.csv.

    Output columns:
        record_id, age, sex, age_missing, sex_missing,
        dx_codes, abbrs, label_list_11, label
    """
    # -- Load SNOMED lookup ---------------------------------------
    snomed_df = pd.read_csv(snomed_csv)
    code_to_abbr = build_code_to_abbr(snomed_df)
    print(f"[snomed] loaded {len(code_to_abbr)} code -> abbr entries")

    # -- Walk .hea files ------------------------------------------
    hea_files = sorted(Path(hea_dir).rglob("*.hea"))
    print(f"[hea]    found {len(hea_files)} records under {hea_dir}")

    rows = []
    for hp in hea_files:
        m = extract_hea_metadata(hp)
        tokens = m["dx_codes"]                           # raw SNOMED codes
        abbrs  = tokens_to_abbrs(tokens, code_to_abbr)
        rhythm11 = abbrs_to_rhythm11(abbrs)
        rows.append({
            "record_id":     m["record_id"],
            "age":           m["age"],
            "sex":           m["sex"],
            "dx_codes":      ";".join(tokens),
            "tokens":        tokens,
            "abbrs":         abbrs,
            "label_list_11": rhythm11,
        })

    df = pd.DataFrame(rows)
    df["label"] = df.apply(assign_final4, axis=1)
    df = impute_demographics(df)

    # -- Save -----------------------------------------------------
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    save_df = df.copy()
    save_df["abbrs"]         = save_df["abbrs"].apply(lambda x: ",".join(x))
    save_df["label_list_11"] = save_df["label_list_11"].apply(lambda x: ",".join(x))
    save_df = save_df.drop(columns=["tokens"])
    save_df.to_csv(out_csv, index=False)

    print(f"[demographics] saved {len(save_df)} records -> {out_csv}")
    print("\n4-class distribution:")
    print(df["label"].value_counts().to_string())
    return df


# ============================================================
# 5. SPLIT  train / val / test  (stratified by 4-class label)
# ============================================================
def split_train_val_test(df: pd.DataFrame,
                         out_dir: str,
                         keep_cols: Tuple[str, ...] = ("record_id", "age", "sex", "label"),
                         train_ratio: float = TRAIN_RATIO,
                         val_ratio:   float = VAL_RATIO,
                         test_ratio:  float = TEST_RATIO,
                         random_state: int = RANDOM_STATE
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split: 70 / 20 / 10 by default."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "ratios must sum to 1"

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df_final = df[list(keep_cols)].copy()

    # Split 1: train vs (val + test)
    train_df, temp_df = train_test_split(
        df_final,
        test_size    = 1.0 - train_ratio,
        random_state = random_state,
        stratify     = df_final["label"],
    )
    # Split 2: val vs test (relative to temp)
    inner_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size    = inner_test,
        random_state = random_state,
        stratify     = temp_df["label"],
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv  (out_dir / "val.csv",   index=False)
    test_df.to_csv (out_dir / "test.csv",  index=False)

    print(f"\n[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
          f"  total={len(train_df)+len(val_df)+len(test_df)}")
    for name, sdf in (("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)):
        print(f"  [{name}] {sdf['label'].value_counts().to_dict()}")
    return train_df, val_df, test_df


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    HEA_DIR    = r"C:\Users\nguye\Project\ECG\input\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\chapman\WFDBRecords"
    SNOMED_CSV = r"C:\Users\nguye\Project\ECG\input\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\chapman\ConditionNames_SNOMED-CT.csv"
    OUT_DIR    = r"C:\Users\nguye\Project\ECG\test_demo"

    df = build_demographics(
        hea_dir    = HEA_DIR,
        snomed_csv = SNOMED_CSV,
        out_csv    = f"{OUT_DIR}/demographics.csv",
    )
    split_train_val_test(df, OUT_DIR)
