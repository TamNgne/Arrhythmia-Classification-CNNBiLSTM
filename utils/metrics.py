import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix
)
# def nlp_metrics(references, predictions):
#     """
#     references: List of ground truth strings.
#     predictions: List of predicted strings.
#     """
#     smoothie = SmoothingFunction().method4 
#     rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

#     bleu_scores = []
#     rouge1_f1 = []
#     rouge2_f1 = []
#     rougeL_f1 = []

#     # 2. Loop through all samples
#     for ref, pred in zip(references, predictions):
#         # --- BLEU ---
#         # NLTK expects tokenized lists: [['Sinus', 'Rhythm']]
#         ref_tokens = [ref.split()] 
#         pred_tokens = pred.split()
        
#         b_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
#         bleu_scores.append(b_score)

#         # --- ROUGE ---
#         r_score = rouge_scorer_obj.score(ref, pred)
#         rouge1_f1.append(r_score['rouge1'].fmeasure)
#         rouge2_f1.append(r_score['rouge2'].fmeasure)
#         rougeL_f1.append(r_score['rougeL'].fmeasure)

#     # 3. Aggregate Results
#     return {
#         "BLEU": np.mean(bleu_scores),
#         "ROUGE-1": np.mean(rouge1_f1),
#         "ROUGE-2": np.mean(rouge2_f1),
#         "ROUGE-L": np.mean(rougeL_f1)
#     }
def f1_score_macro(logits, targets, num_classes):
    """
    Multi-class macro F1 score
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = targets.cpu().numpy()

    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))

    f1_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return np.mean(f1_scores)

def classification_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    precision = []
    sensitivity = []  # recall
    specificity = []
    f1 = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        s = tn / (tn + fp + 1e-8)

        precision.append(p)
        sensitivity.append(r)
        specificity.append(s)
        f1.append(2 * p * r / (p + r + 1e-8))

    accuracy = np.trace(cm) / np.sum(cm)

    return {
        "accuracy": accuracy,
        "precision_macro": np.mean(precision),
        "sensitivity_macro": np.mean(sensitivity),
        "specificity_macro": np.mean(specificity),
        "f1_macro": np.mean(f1),
        "confusion_matrix": cm
    }

def classification_metrics_per_class(y_true, y_pred, label_names=None):
    """
    Computes detailed metrics for each class.

    Args:
        y_true: array-like of shape (N,)
        y_pred: array-like of shape (N,)
        label_names: optional list of class names (index-aligned)

    Returns:
        dict with per-class metrics and macro averages
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class = {}
    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)          # sensitivity
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = cm[i, :].sum()

        class_name = label_names[i] if label_names else str(i)

        per_class[class_name] = {
            "precision": precision,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support
        }

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_list.append(f1)

    return {
        "per_class": per_class,
        "macro_avg": {
            "precision": np.mean(precision_list),
            "sensitivity": np.mean(recall_list),
            "specificity": np.mean(specificity_list),
            "f1": np.mean(f1_list)
        },
        "confusion_matrix": cm
    }

def multi_label_metrics(logits, targets, num_classes, label_names=None, threshold=0.5):
    """
    Computes comprehensive multi-label metrics including AUROC and AUPRC.
    
    Args:
        logits: Tensor [B, C] (raw model outputs before activation)
        targets: Tensor [B, C] (multi-hot ground truth vectors)
        num_classes: int (e.g., 27)
        label_names: list of strings (optional)
        threshold: float (probability threshold for binary predictions)
        
    Returns:
        dict: Macro averages and per-class metrics
    """
    # Sigmoid to get probabilities
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    y_true = targets.cpu().detach().numpy()
    
    # 2. Apply threshold to get binary predictions 
    y_pred = (probs > threshold).astype(int)

    if label_names is None:
        label_names = [f"Class_{i}" for i in range(num_classes)]

    # 3. Calculate Macro Metrics across all 27 classes
    try:
        auroc_macro = roc_auc_score(y_true, probs, average='macro')
    except ValueError:
        auroc_macro = np.nan
        
    auprc_macro = average_precision_score(y_true, probs, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 4. Generate the Multi-Label Confusion Matrix
    # Returns an array (num_classes, 2, 2)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    per_class = {}

    # 5. Calculate Per-Class Metrics
    for i in range(num_classes):
        # Extract TN, FP, FN, TP for this specific class
        tn, fp, fn, tp = mcm[i].ravel()
        
        # Class-specific AUROC/AUPRC
        try:
            class_auroc = roc_auc_score(y_true[:, i], probs[:, i])
        except ValueError:
            class_auroc = np.nan
            
        try:
            class_auprc = average_precision_score(y_true[:, i], probs[:, i])
        except ValueError:
            class_auprc = np.nan

        # Standard clinical metrics
        precision = tp / (tp + fp + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8) # Recall
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
        
        per_class[label_names[i]] = {
            "AUROC": class_auroc,
            "AUPRC": class_auprc, 
            "F1": f1,
            "Precision": precision,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Support": int(y_true[:, i].sum()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
        }

    return {
        "macro_AUROC": auroc_macro,
        "macro_AUPRC": auprc_macro,
        "macro_F1": f1_macro,
        "per_class": per_class,
        "confusion_matrices": mcm # For plotting later
    }

def plot_confusion_matrix(cm, class_names, normalize=False, title="Confusion Matrix"):
    """
    cm: 2D array (confusion matrix)
    class_names: list of class labels
    normalize: bool, normalize per true class
    """

    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar(im)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def plot_per_class_confusion(cm, class_names, normalize=False):
    num_classes = len(class_names)

    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        mat = np.array([[TP, FN],
                        [FP, TN]], dtype=float)

        if normalize:
            mat = mat / mat.sum(axis=1, keepdims=True)
            mat = np.nan_to_num(mat)

        plt.figure(figsize=(4, 4))
        plt.imshow(mat, cmap="Blues")
        plt.title(f"{cls} vs Rest")
        plt.colorbar()

        labels = ["Positive", "Negative"]
        plt.xticks([0, 1], labels)
        plt.yticks([0, 1], labels)

        for r in range(2):
            for c in range(2):
                plt.text(
                    c, r,
                    f"{mat[r, c]:.2f}" if normalize else int(mat[r, c]),
                    ha="center", va="center",
                    color="white" if mat[r, c] > mat.max()/2 else "black"
                )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()