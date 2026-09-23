"""
Train a Logistic Regression model on the Pima Indians Diabetes Dataset.

Pipeline:
  1. Load data from data/diabetes.csv (download if absent).
  2. Replace physiologically invalid zeros with NaN for selected features.
  3. Impute missing values with the training-set median (via SimpleImputer).
  4. Scale features with StandardScaler.
  5. Train LogisticRegression.
  6. Evaluate on a stratified 20% hold-out set.
  7. Save the fitted pipeline to model/diabetes_model.pkl.
  8. Save evaluation plots to report_images/.

Features with physiologically impossible zero values that are imputed:
  Glucose, BloodPressure, SkinThickness, Insulin, BMI
  (Pregnancies and Age can legitimately be 0/non-zero small, DiabetesPedigreeFunction
   is never 0 in practice, so only the five above are treated.)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "diabetes.csv")
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
IMG_DIR    = "report_images"

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COL = "Outcome"

# Features that cannot physiologically be zero
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_data():
    if not os.path.exists(DATA_PATH):
        print("Dataset not found — downloading …")
        import download_dataset
        download_dataset.download()
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def preprocess(df: pd.DataFrame):
    """Replace invalid zeros with NaN; return X, y."""
    df = df.copy()
    for col in ZERO_AS_NAN:
        n = (df[col] == 0).sum()
        if n:
            print(f"  {col}: {n} zero(s) -> NaN")
        df.loc[df[col] == 0, col] = np.nan
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_roc_curve(y_true, y_prob, auc_val, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR,   exist_ok=True)

    # 1. Load
    df = load_data()

    # 2. Preprocess
    print("\nReplacing physiologically invalid zeros with NaN:")
    X, y = preprocess(df)

    # 3. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Class balance (train) – 0: {(y_train==0).sum()}  1: {(y_train==1).sum()}")
    print(f"Class balance (test)  – 0: {(y_test==0).sum()}   1: {(y_test==1).sum()}")

    # 4. Train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("\nModel training complete.")

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "="*52)
    print("  MODEL EVALUATION RESULTS")
    print("="*52)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("="*52)
    print(f"\n  Confusion Matrix:\n{cm}\n")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # 6. Save plots
    print("Saving evaluation plots …")
    save_confusion_matrix(y_test, y_pred, os.path.join(IMG_DIR, "confusion_matrix.png"))
    save_roc_curve(y_test, y_prob, auc,   os.path.join(IMG_DIR, "roc_curve.png"))

    # 7. Save pipeline
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline saved to {MODEL_PATH}")

    # 8. Save metrics to a text file for the report
    metrics_path = os.path.join(IMG_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    print(f"Metrics saved to {metrics_path}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc, confusion_matrix=cm)


if __name__ == "__main__":
    main()
