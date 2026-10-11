"""
metrics.py — Evaluation helpers shared by every model (report §8).
"""
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def tune_thresholds(Y_true, P, grid=np.arange(0.05, 0.95, 0.01)):
    """For each label, pick the probability cut-off that maximizes F1 on
    validation data. Rare labels (threat, identity_hate) usually want a
    threshold well below 0.5."""
    best = []
    for j in range(Y_true.shape[1]):
        if Y_true[:, j].sum() == 0:          # label absent in val: keep default
            best.append(0.5)
            continue
        scores = [f1_score(Y_true[:, j], P[:, j] >= t, zero_division=0) for t in grid]
        best.append(float(grid[int(np.argmax(scores))]))
    return np.array(best)


def _safe(fn, y, p):
    """ROC-AUC / PR-AUC are undefined when a label has only one class present."""
    return fn(y, p) if 0 < y.sum() < len(y) else float("nan")


def multilabel_report(Y_true, P, thresholds, labels):
    """Model 1: per-label P/R/F1 + ROC-AUC + PR-AUC, then macro & micro F1."""
    Y_pred = (P >= thresholds).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0)
    print(f"{'label':<14}{'prec':>7}{'rec':>7}{'f1':>7}{'rocAUC':>8}{'prAUC':>8}{'support':>9}")
    for j, lab in enumerate(labels):
        print(f"{lab:<14}{prec[j]:>7.3f}{rec[j]:>7.3f}{f1[j]:>7.3f}"
              f"{_safe(roc_auc_score, Y_true[:, j], P[:, j]):>8.3f}"
              f"{_safe(average_precision_score, Y_true[:, j], P[:, j]):>8.3f}{sup[j]:>9}")
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    print(f"{'macro-F1':<14}{macro:>21.3f}\n{'micro-F1':<14}{micro:>21.3f}")
    per_label = {lab: {"precision": float(prec[j]), "recall": float(rec[j]), "f1": float(f1[j]),
                       "roc_auc": float(_safe(roc_auc_score, Y_true[:, j], P[:, j])),
                       "pr_auc": float(_safe(average_precision_score, Y_true[:, j], P[:, j])),
                       "support": int(sup[j]), "threshold": float(thresholds[j])}
                 for j, lab in enumerate(labels)}
    return {"macro_f1": float(macro), "micro_f1": float(micro), "per_label": per_label}


def multiclass_report(y_true, y_pred, classes):
    """Model 2: accuracy, per-class P/R/F1, macro-F1, confusion matrix."""
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(" " * 11 + "".join(f"{c:>11}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<11}" + "".join(f"{v:>11}" for v in cm[i]))
    macro = f1_score(y_true, y_pred, average="macro")
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    per_class = {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                     "support": int(sup[i])} for i, c in enumerate(classes)}
    return {"macro_f1": float(macro),
            "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "per_class": per_class, "confusion_matrix": cm.tolist()}
