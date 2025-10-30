from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

COLS = [
    "clump_thickness",
    "uniform_cell_size",
    "uniform_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]

def load_artifacts(art_dir: str | Path = "artifacts"):
    art_dir = Path(art_dir)
    pre = joblib.load(art_dir / "preprocessor.pkl")
    model = joblib.load(art_dir / "best_model.pkl")
    return pre, model

def predict_one(sample: dict, art_dir: str | Path = "artifacts") -> dict:
    pre, model = load_artifacts(art_dir)
    X = pd.DataFrame([sample], columns=COLS)
    X_s = pre.transform(X)
    proba = model.predict_proba(X_s)[:, 1][0] if hasattr(model, "predict_proba") else float("nan")
    pred = bool(proba >= 0.5)
    return {"proba_malignant": float(proba), "pred_malignant": pred}

if __name__ == "__main__":
    # ตัวอย่าง
    ex = {c: 3 for c in COLS}
    ex["bare_nuclei"] = 8
    print(predict_one(ex))
