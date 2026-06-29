import os
import argparse
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from imblearn.combine import SMOTETomek
APP_DIR  = Path(__file__).resolve().parent          # .../web_app
BASE_DIR = APP_DIR.parent                            # .../<project_root>
sys.path.insert(0, str(BASE_DIR))                    # enable top-level imports

from Models.encoder_ECG import ECG_CNN_BiLSTM, ECGResNet, freeze_module
from Dataset.data       import ECGDataset
from utils.metrics      import (
    classification_metrics_per_class,
    plot_confusion_matrix,
    plot_per_class_confusion,
)


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # Paths (outputs from process_demographics.py + process_ecg.py)
    "train_csv":  r"C:\Users\nguye\Project\ECG\Arrhythmia classification\data\demographics\train.csv",
    "val_csv":    r"C:\Users\nguye\Project\ECG\Arrhythmia classification\data\demographics\val.csv",
    "test_csv":   r"C:\Users\nguye\Project\ECG\Arrhythmia classification\data\demographics\test.csv",
    "ecg_folder": r"C:\Users\nguye\Project\ECG\Arrhythmia classification\data\denoised_signals",
    "output_dir": r"C:\Users\nguye\Project\ECG\Arrhythmia classification\Checkpoints",

    # Training
    "batch_size":     64,
    "num_workers":    4,
    "epochs_stage1":  100,
    "epochs_stage2":  50,
    "lr_stage1":      1e-3,
    "lr_stage2":      5e-4,
    "weight_decay":   1e-4,
    "patience":       7,                 # early-stop on val macro-F1
    "dropout":        0.3,
    "lstm_hidden":    256,
    "lstm_layers":    2,
    "num_classes":    4,
    "class_names":    ["AFIB", "GSVT", "SB", "SR"],
    "label_to_idx":   {"AFIB": 0, "GSVT": 1, "SB": 2, "SR": 3},
    "seed":           42,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================
# Helpers
# ============================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def uses_tabular(exp_name: str) -> bool:
    return exp_name.startswith("ecg_tab")


def uses_smote(exp_name: str) -> bool:
    return exp_name.endswith("smote_tomek")


def plot_loss(history, title="Loss", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history, linewidth=1.5)
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()


def print_metrics(metrics: dict, title: str = ""):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
    print("\nPer-Class:")
    for cls, vals in metrics["per_class"].items():
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in vals.items()]
        print(f"  {cls:>30s}  {' | '.join(parts)}")
    print("\nMacro Average:")
    for k, v in metrics["macro_avg"].items():
        if isinstance(v, float):
            print(f"  {k:<15s}: {v:.4f}")


# ============================================================
# Dataset factory  (uses ECGDataset directly — loads .npy)
# ============================================================
def build_dataset(csv_path: str, ecg_folder: str, use_tab: bool, label_map: dict):
    """Wrap ECGDataset with feature_cols based on experiment type."""
    feature_cols = ["sex", "age"] if use_tab else []
    return ECGDataset(
        diagnosis_file = csv_path,
        ecg_folder     = ecg_folder,
        feature_cols   = feature_cols,
        label_map      = label_map,
    )


# ============================================================
# Embedding extraction
# ============================================================
@torch.no_grad()
def extract_embeddings(model, dataset, device, use_tab,
                       batch_size=64, num_workers=0):
    """
    Run forward pass through encoder → capture embedding z via forward hook
    on the last nn.Linear (classification head).

    Returns:
        Z_ecg : (N, 512)     ECG embeddings (always)
        Tab   : (N, tab_dim) raw tabular (if use_tab else None)
        Y     : (N,)         integer labels
    """
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None, "No nn.Linear found in model"

    captured = {"z": None}
    def _hook(module, inp, out):
        captured["z"] = inp[0].detach().cpu()
    handle = last_linear.register_forward_hook(_hook)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    model.eval()
    Z_list, T_list, Y_list = [], [], []
    for batch in loader:
        ecg = batch["ecg"].to(device).float()
        tab = batch["feature"].to(device).float()
        y   = batch["label"]
        if use_tab:
            _ = model(ecg, tab)
        else:
            _ = model(ecg)
        z = captured["z"]                                # (B, 512+tab_dim)
        if use_tab and z.shape[1] > 512:
            Z_list.append(z[:, :512].numpy())
            T_list.append(tab.detach().cpu().numpy())
        else:
            Z_list.append(z.numpy())
        Y_list.append(y.numpy())

    handle.remove()
    Z = np.concatenate(Z_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    T = np.concatenate(T_list, axis=0) if use_tab else None
    return Z, T, Y


# ============================================================
# SMOTE-Tomek balanced loader
# ============================================================
def build_balanced_loader(Z_ecg, Tab, Y, use_tab,
                          batch_size=32, random_state=42):
    """
    Apply SMOTE-Tomek on EMBEDDING SPACE (not raw signal).

    Why embedding space?
      - Raw ECG: 60k features per sample, k-NN intractable
      - Embeddings: 512-D, semantically smooth, suited for SMOTE
      - Tomek-links remove ambiguous borderline samples
    """
    if use_tab:
        X = np.hstack([Z_ecg, Tab])
    else:
        X = Z_ecg

    print(f"\n[SMOTE-Tomek] before: {dict(zip(*np.unique(Y, return_counts=True)))}")
    smt = SMOTETomek(random_state=random_state, n_jobs=-1)
    X_res, Y_res = smt.fit_resample(X, Y)
    print(f"[SMOTE-Tomek] after : {dict(zip(*np.unique(Y_res, return_counts=True)))}")

    if use_tab:
        Z_res = X_res[:, :Z_ecg.shape[1]]
        T_res = X_res[:, Z_ecg.shape[1]:]
        ds = TensorDataset(
            torch.from_numpy(Z_res).float(),
            torch.from_numpy(T_res).float(),
            torch.from_numpy(Y_res).long(),
        )
    else:
        ds = TensorDataset(
            torch.from_numpy(X_res).float(),
            torch.from_numpy(Y_res).long(),
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


# ============================================================
# Classifier head used in Stage 2
# ============================================================
class Head(nn.Module):
    """Two-layer MLP head: in_dim → 128 → num_classes."""
    def __init__(self, in_dim: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# Stage 1 — End-to-end supervised training
# ============================================================
def train_stage1(model, train_ds, val_ds, device, cfg, exp_name, use_tab):
    ckpt_path = os.path.join(cfg["output_dir"], f"{exp_name}_stage1.pth")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr_stage1"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs_stage1"])
    criterion = nn.CrossEntropyLoss()

    history, best_f1, patience = [], -1.0, 0
    for epoch in range(1, cfg["epochs_stage1"] + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            ecg = batch["ecg"].to(device).float()
            tab = batch["feature"].to(device).float()
            y   = batch["label"].to(device).long()
            logits = model(ecg, tab) if use_tab else model(ecg)
            loss   = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * ecg.size(0)
        scheduler.step()
        epoch_loss = running / len(train_ds)
        history.append(epoch_loss)

        val_metrics = evaluate_model(model, val_ds, device, cfg["class_names"],
                                     use_tab, batch_size=cfg["batch_size"])
        val_f1 = val_metrics["macro_avg"]["f1"]
        print(f"[Stage1][{epoch:02d}/{cfg['epochs_stage1']}] "
              f"loss={epoch_loss:.4f}  val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_f1": val_f1}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"[Stage1] early stop @ epoch {epoch}")
                break

    plot_loss(history, "Stage 1 Loss",
              os.path.join(cfg["output_dir"], f"{exp_name}_stage1_loss.png"))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model


# ============================================================
# Stage 2 — SMOTE-Tomek fine-tuning (head only)
# ============================================================
def train_stage2(encoder_model, train_ds, val_ds, device, cfg, exp_name, use_tab):
    """Freeze encoder → extract embeddings → SMOTE-Tomek → train new head."""
    print("\n[Stage2] freezing encoder, extracting embeddings…")
    freeze_module(encoder_model)
    encoder_model.eval()

    Z_tr, T_tr, Y_tr = extract_embeddings(encoder_model, train_ds, device, use_tab,
                                          batch_size=cfg["batch_size"])
    Z_va, T_va, Y_va = extract_embeddings(encoder_model, val_ds,   device, use_tab,
                                          batch_size=cfg["batch_size"])

    train_loader = build_balanced_loader(Z_tr, T_tr, Y_tr, use_tab,
                                         batch_size=cfg["batch_size"],
                                         random_state=cfg["seed"])

    in_dim = Z_tr.shape[1] + (T_tr.shape[1] if use_tab else 0)
    head = Head(in_dim, cfg["num_classes"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(),
                                  lr=cfg["lr_stage2"],
                                  weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(cfg["output_dir"], f"{exp_name}_stage2.pth")

    Xv = np.hstack([Z_va, T_va]) if use_tab else Z_va
    Xv_t = torch.from_numpy(Xv).float().to(device)

    history, best_f1, patience = [], -1.0, 0
    for epoch in range(1, cfg["epochs_stage2"] + 1):
        head.train(); running = 0.0; n = 0
        for batch in train_loader:
            if use_tab:
                z, t, y = (b.to(device) for b in batch)
                x = torch.cat([z, t], dim=1)
            else:
                x, y = (b.to(device) for b in batch)
            logits = head(x); loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * x.size(0); n += x.size(0)
        epoch_loss = running / n
        history.append(epoch_loss)

        head.eval()
        with torch.no_grad():
            preds = head(Xv_t).argmax(dim=1).cpu().numpy()
        m = classification_metrics_per_class(Y_va, preds, cfg["class_names"])
        val_f1 = m["macro_avg"]["f1"]
        print(f"[Stage2][{epoch:02d}/{cfg['epochs_stage2']}] "
              f"loss={epoch_loss:.4f}  val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            torch.save({"encoder_state_dict": encoder_model.state_dict(),
                        "head_state_dict":    head.state_dict(),
                        "epoch": epoch, "val_f1": val_f1}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"[Stage2] early stop @ epoch {epoch}")
                break

    plot_loss(history, "Stage 2 Loss",
              os.path.join(cfg["output_dir"], f"{exp_name}_stage2_loss.png"))
    state = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(state["head_state_dict"])
    return encoder_model, head


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_model(model, dataset, device, label_names, use_tab,
                   batch_size=32, num_workers=0, head=None):
    """
    Evaluate either:
      - full model (head=None): logits = model(ecg[, tab])
      - encoder + head pair    : embedding -> head -> logits
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    if head is not None:
        head.eval()
        Z, T, Y = extract_embeddings(model, dataset, device, use_tab,
                                     batch_size=batch_size, num_workers=num_workers)
        X = np.hstack([Z, T]) if use_tab else Z
        preds   = head(torch.from_numpy(X).float().to(device)).argmax(dim=1).cpu().numpy()
        targets = Y
    else:
        preds_list, tgt_list = [], []
        for batch in loader:
            ecg = batch["ecg"].to(device).float()
            tab = batch["feature"].to(device).float()
            y   = batch["label"]
            logits = model(ecg, tab) if use_tab else model(ecg)
            preds_list.append(logits.argmax(dim=1).cpu().numpy())
            tgt_list.append(y.numpy())
        preds   = np.concatenate(preds_list)
        targets = np.concatenate(tgt_list)
    return classification_metrics_per_class(targets, preds, label_names)


# ============================================================
# Main
# ============================================================
ALL_EXPS = [
    "ecg_only_baseline",
    "ecg_only_smote_tomek",
    "ecg_tab_baseline",
    "ecg_tab_smote_tomek",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=ALL_EXPS)
    args = parser.parse_args()

    cfg, exp = CONFIG, args.exp
    use_tab  = uses_tabular(exp)
    do_smote = uses_smote(exp)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("=" * 70)
    print(f"  Experiment : {exp}")
    print(f"  Device     : {cfg['device']}")
    print(f"  Use tab    : {use_tab}")
    print(f"  Use SMOTE  : {do_smote}")
    print(f"  Time       : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # ── Datasets ─────────────────────────────────────────────
    label_map = cfg["label_to_idx"]
    train_ds  = build_dataset(cfg["train_csv"], cfg["ecg_folder"], use_tab, label_map)
    val_ds    = build_dataset(cfg["val_csv"],   cfg["ecg_folder"], use_tab, label_map)
    test_ds   = build_dataset(cfg["test_csv"],  cfg["ecg_folder"], use_tab, label_map)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── Model ────────────────────────────────────────────────
    cnn   = ECGResNet(input_channels=12)
    model = ECG_CNN_BiLSTM(
        cnn_encoder = cnn,
        tabular_dim = 2 if use_tab else 0,
        lstm_hidden = cfg["lstm_hidden"],
        lstm_layers = cfg["lstm_layers"],
        num_classes = cfg["num_classes"],
        dropout     = cfg["dropout"],
    ).to(cfg["device"])

    # ── Stage 1 ──────────────────────────────────────────────
    model = train_stage1(model, train_ds, val_ds, cfg["device"], cfg, exp, use_tab)
    metrics_s1 = evaluate_model(model, test_ds, cfg["device"],
                                cfg["class_names"], use_tab,
                                batch_size=cfg["batch_size"])
    print_metrics(metrics_s1, f"Stage 1 — Test metrics  ({exp})")

    # ── Stage 2 (SMOTE-Tomek), only for *_smote_tomek experiments ─
    if do_smote:
        model, head = train_stage2(model, train_ds, val_ds, cfg["device"],
                                   cfg, exp, use_tab)
        metrics_s2 = evaluate_model(model, test_ds, cfg["device"],
                                    cfg["class_names"], use_tab,
                                    batch_size=cfg["batch_size"], head=head)
        print_metrics(metrics_s2, f"Stage 2 — Test metrics  ({exp})")

    # ── Confusion matrices ───────────────────────────────────
    final = metrics_s2 if do_smote else metrics_s1
    cm    = final["confusion_matrix"]
    plot_confusion_matrix(cm, cfg["class_names"],
                          title=f"Confusion Matrix — {exp}")
    plt.savefig(os.path.join(cfg["output_dir"], f"{exp}_cm.png"), dpi=150)
    plt.close()
    plot_per_class_confusion(cm, cfg["class_names"])
    plt.savefig(os.path.join(cfg["output_dir"], f"{exp}_cm_per_class.png"), dpi=150)
    plt.close()

    print(f"\n[done] outputs saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
