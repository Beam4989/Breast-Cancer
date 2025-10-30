from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt

def evaluate_classifier(clf, X_test, y_test, *, needs_scaled=False, X_test_s=None):
    """
    ถ้าโมเดลต้องใช้สเกล (LR / SVM / MLP) ให้ส่ง needs_scaled=True และ X_test_s
    """
    if needs_scaled and X_test_s is not None:
        proba = clf.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
    else:
        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "_proba": proba,
    }

def save_roc_curve(name: str, y_test, proba, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"roc_{name}.png"

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend(loc="lower right")
    plt.savefig(fp, bbox_inches="tight", dpi=150)
    plt.close()
    return fp, roc_auc

def save_metrics_json(results: dict, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
