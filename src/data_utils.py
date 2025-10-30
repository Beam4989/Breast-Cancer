from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ชื่อคอลัมน์ตามชุด Wisconsin (Original)
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
ALL_COLS = ["sample_id"] + COLS + ["class"]

def load_original_from_dir(data_dir: str | Path) -> pd.DataFrame:
    """
    อ่านชุด Wisconsin (Original) จากโฟลเดอร์ที่แตกไฟล์แล้ว
    ต้องมีไฟล์ 'breast-cancer-wisconsin.data' อยู่ในโฟลเดอร์
    """
    data_dir = Path(data_dir)
    data_path = data_dir / "breast-cancer-wisconsin.data"
    if not data_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {data_path}")

    df = pd.read_csv(data_path, header=None)
    df.columns = ALL_COLS
    # '?' -> NaN
    df = df.replace("?", np.nan)
    # แปลง feature เป็นตัวเลข
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # target: 2=benign->0, 4=malignant->1
    df["target"] = df["class"].map({2: 0, 4: 1}).astype(int)
    return df

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    จัดการ missing values (median impute) + clip outliers 1–99 percentile (robust)
    คืนค่า X, y พร้อมใช้งาน
    """
    X_raw = df[COLS].copy()
    y = df["target"].astype(int)

    # Impute median
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=COLS, index=df.index)

    # Clip 1st–99th percentile เพื่อลดผลกระทบ outliers
    X = X_imp.copy()
    for c in COLS:
        lo, hi = np.percentile(X[c], [1, 99])
        X[c] = X[c].clip(lo, hi)

    return X, y
