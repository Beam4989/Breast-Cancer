from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from data_utils import load_original_from_dir, preprocess, COLS
from metrics_utils import evaluate_classifier, save_roc_curve, save_metrics_json

ART_DIR = Path("artifacts")
DATA_DIR = Path("data/breast+cancer+wisconsin+original")  # ← โฟลเดอร์ที่ unzip แล้ว

SELECT_BEST_BY = "roc_auc"  # เปลี่ยนเป็น "recall" ได้ถ้าเน้นด้านการแพทย์

def main():
    # 1) Load & preprocess
    df = load_original_from_dir(DATA_DIR)
    X, y = preprocess(df)

    # 2) Train/test split 80/20 (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Scaling สำหรับโมเดลที่ต้องการ
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) Feature Selection (MI)
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi, index=COLS).sort_values(ascending=False)

    # 5) Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", n_jobs=4
        ),
        "SVM_RBF": SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=2000, random_state=42),
    }

    results: dict[str, dict] = {}
    trained = {}

    for name, clf in models.items():
        if name in ["LogisticRegression", "SVM_RBF", "MLP"]:
            clf.fit(X_train_s, y_train)
            metrics = evaluate_classifier(clf, X_test, y_test, needs_scaled=True, X_test_s=X_test_s)
        else:
            clf.fit(X_train, y_train)
            metrics = evaluate_classifier(clf, X_test, y_test)

        roc_path, _ = save_roc_curve(name, y_test, metrics["_proba"], ART_DIR)
        metrics["roc_path"] = str(roc_path)
        metrics.pop("_proba")
        results[name] = metrics
        trained[name] = clf

    # 6) Explainability: permutation + native (RF/XGB)
    rf = trained["RandomForest"]
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_series = pd.Series(perm.importances_mean, index=COLS).sort_values(ascending=False)
    rf_native = pd.Series(rf.feature_importances_, index=COLS).sort_values(ascending=False)

    xgb_native = None
    if "XGBoost" in trained:
        xgb_native = pd.Series(trained["XGBoost"].feature_importances_, index=COLS).sort_values(ascending=False)

    feat_table = pd.DataFrame({
        "MI": mi_series,
        "RF_Permutation": perm_series,
        "RF_Native": rf_native
    }).sort_values("RF_Permutation", ascending=False)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    feat_table.to_csv(ART_DIR / "top_features.csv")

    # 7) เลือก Best model
    df_res = pd.DataFrame(results).T
    best_name = df_res.sort_values(SELECT_BEST_BY, ascending=False).index[0]
    best_model = trained[best_name]

    # 8) บันทึก artifacts
    joblib.dump(scaler, ART_DIR / "preprocessor.pkl")
    joblib.dump(best_model, ART_DIR / "best_model.pkl")
    save_metrics_json(results, ART_DIR / "metrics.json")
    (ART_DIR / "README.txt").write_text(
        f"Best model: {best_name}\nSelected by: {SELECT_BEST_BY}\nTest {SELECT_BEST_BY}: {df_res.loc[best_name, SELECT_BEST_BY]:.4f}\n",
        encoding="utf-8"
    )

    # log สรุป
    print("=== Training finished ===")
    print(df_res.sort_values(SELECT_BEST_BY, ascending=False).round(4))
    print("\nTop features (head):")
    print(feat_table.head().round(4))

if __name__ == "__main__":
    main()
