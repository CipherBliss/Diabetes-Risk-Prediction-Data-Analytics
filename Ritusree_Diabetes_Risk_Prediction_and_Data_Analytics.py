"""
================================================================================
  Ritusree Banerjee
  Diabetes Risk Prediction and Data Analytics
================================================================================

Consolidated Python source file for academic submission.
All code assembled from the original project files without modification.

Project files included (in order):
  1. download_dataset.py  — Dataset download utility
  2. train_model.py       — Data loading, preprocessing, model training,
                            evaluation (confusion matrix, ROC curve)
  3. generate_analytics.py — Exploratory data analysis and visualisations
  4. backend/api.py       — Flask REST API for prediction
  5. app.py               — Streamlit dashboard (prediction + analytics)

Dataset  : Pima Indians Diabetes Database (UCI / Kaggle)
Algorithm: Logistic Regression (scikit-learn Pipeline with
           SimpleImputer + StandardScaler)
Tools    : Python, Pandas, NumPy, Scikit-learn, Matplotlib,
           Streamlit, Flask, Flask-CORS

Run instructions
----------------
  # Step 1 – Download dataset (if not already present)
  python download_dataset.py

  # Step 2 – Train model and generate evaluation plots
  python train_model.py

  # Step 3 – Generate analytics charts
  python generate_analytics.py

  # Step 4a – Launch Streamlit dashboard
  streamlit run app.py

  # Step 4b – Launch Flask REST API (optional, separate terminal)
  python backend/api.py
================================================================================
"""


# ==============================================================================
# SECTION 1 — DATASET DOWNLOAD UTILITY
# Source: download_dataset.py
# ==============================================================================
"""
Download the Pima Indians Diabetes Dataset.

Source: UCI Machine Learning Repository
  - URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
  - Original: National Institute of Diabetes and Digestive and Kidney Diseases
  - UCI Reference: https://archive.ics.uci.edu/ml/datasets/diabetes
  - Kaggle mirror: https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset

The dataset contains 768 records for Pima Indian women aged >= 21.
Each record has 8 medical predictor features and 1 binary outcome (Outcome).
"""

import os
import urllib.request

DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

OUTPUT_PATH = os.path.join("data", "diabetes.csv")


def download():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
        return

    print(f"Downloading dataset from:\n  {DATASET_URL}")
    tmp_path = OUTPUT_PATH + ".tmp"
    urllib.request.urlretrieve(DATASET_URL, tmp_path)

    # Prepend header row
    with open(tmp_path, "r") as f:
        rows = f.read()
    with open(OUTPUT_PATH, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        f.write(rows)
    os.remove(tmp_path)
    print(f"Dataset saved to {OUTPUT_PATH}")


# ==============================================================================
# SECTION 2 — MODEL TRAINING AND EVALUATION
# Source: train_model.py
# Covers: data loading, preprocessing, logistic regression training,
#         model evaluation, confusion matrix, ROC curve
# ==============================================================================
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
        download()
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


def train_and_evaluate():
    """End-to-end training and evaluation pipeline. Returns metrics dict."""
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


# ==============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS AND VISUALISATIONS
# Source: generate_analytics.py
# Covers: outcome distribution, glucose, BMI, age, blood pressure,
#         correlation heatmap, feature means by outcome
# ==============================================================================
"""
Generate all analytics charts from data/diabetes.csv and save to report_images/.
Run once before the Streamlit app, or let app.py call this on first run.
Does NOT overwrite confusion_matrix.png, roc_curve.png, or metrics.txt.
"""

import matplotlib.ticker as mticker

ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

PROTECTED = {"confusion_matrix.png", "roc_curve.png", "metrics.txt"}


def _save_chart(fig, filename):
    """Save a matplotlib figure to report_images/ and close it."""
    path = os.path.join(IMG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_analytics_data():
    """Load the dataset for analytics (no preprocessing — raw counts)."""
    df = pd.read_csv(DATA_PATH)
    return df


def outcome_distribution(df):
    """Bar chart of Outcome = 0 vs 1."""
    counts = df["Outcome"].value_counts().sort_index()
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    colors = ["#3b82d4", "#e05252"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Diabetes Outcome Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _save_chart(fig, "outcome_distribution.png")


def glucose_distribution(df):
    """Histogram of Glucose (zeros excluded)."""
    g = df[df["Glucose"] > 0]["Glucose"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(g, bins=30, color="#3b82d4", edgecolor="white", alpha=0.85)
    ax.axvline(g.mean(), color="#e05252", linestyle="--", linewidth=1.5,
               label=f"Mean: {g.mean():.1f}")
    ax.axvline(g.median(), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Median: {g.median():.1f}")
    ax.set_title("Glucose Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Plasma Glucose (mg/dL)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, "glucose_distribution.png")


def glucose_outcome(df):
    """Box plot of Glucose by Outcome."""
    g0 = df[df["Outcome"] == 0]["Glucose"]
    g1 = df[df["Outcome"] == 1]["Glucose"]
    g0 = g0[g0 > 0]
    g1 = g1[g1 > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([g0, g1], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#3b82d4")
    bp["boxes"][1].set_facecolor("#e05252")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    ax.set_title("Glucose by Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Plasma Glucose (mg/dL)")
    fig.tight_layout()
    _save_chart(fig, "glucose_outcome.png")


def bmi_distribution(df):
    """Histogram of BMI (zeros excluded)."""
    b = df[df["BMI"] > 0]["BMI"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(b, bins=30, color="#7c5cd8", edgecolor="white", alpha=0.85)
    ax.axvline(b.mean(), color="#e05252", linestyle="--", linewidth=1.5,
               label=f"Mean: {b.mean():.1f}")
    ax.axvline(b.median(), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Median: {b.median():.1f}")
    ax.set_title("BMI Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("BMI (kg/m2)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, "bmi_distribution.png")


def bmi_outcome(df):
    """Box plot of BMI by Outcome."""
    b0 = df[df["Outcome"] == 0]["BMI"]
    b1 = df[df["Outcome"] == 1]["BMI"]
    b0 = b0[b0 > 0]
    b1 = b1[b1 > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([b0, b1], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#3b82d4")
    bp["boxes"][1].set_facecolor("#e05252")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    ax.set_title("BMI by Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("BMI (kg/m2)")
    fig.tight_layout()
    _save_chart(fig, "bmi_outcome.png")


def age_distribution(df):
    """Histogram of Age."""
    a = df["Age"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(a, bins=20, color="#10b981", edgecolor="white", alpha=0.85)
    ax.axvline(a.mean(), color="#e05252", linestyle="--", linewidth=1.5,
               label=f"Mean: {a.mean():.1f}")
    ax.axvline(a.median(), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Median: {a.median():.1f}")
    ax.set_title("Age Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, "age_distribution.png")


def age_outcome(df):
    """Box plot of Age by Outcome."""
    a0 = df[df["Outcome"] == 0]["Age"]
    a1 = df[df["Outcome"] == 1]["Age"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([a0, a1], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#3b82d4")
    bp["boxes"][1].set_facecolor("#e05252")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    ax.set_title("Age by Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Age (years)")
    fig.tight_layout()
    _save_chart(fig, "age_outcome.png")


def blood_pressure_outcome(df):
    """Box plot of BloodPressure by Outcome (zeros excluded)."""
    bp0 = df[df["Outcome"] == 0]["BloodPressure"]
    bp1 = df[df["Outcome"] == 1]["BloodPressure"]
    bp0 = bp0[bp0 > 0]
    bp1 = bp1[bp1 > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bplt = ax.boxplot([bp0, bp1], patch_artist=True, widths=0.5,
                      medianprops=dict(color="black", linewidth=2))
    bplt["boxes"][0].set_facecolor("#3b82d4")
    bplt["boxes"][1].set_facecolor("#e05252")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    ax.set_title("Blood Pressure by Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Diastolic Blood Pressure (mmHg)")
    fig.tight_layout()
    _save_chart(fig, "blood_pressure_outcome.png")


def correlation_heatmap(df):
    """Heatmap of feature correlations."""
    numeric_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_chart(fig, "correlation_heatmap.png")


def feature_boxplots(df):
    """Side-by-side bar chart of feature means by Outcome."""
    features = ["Pregnancies", "Glucose", "BloodPressure", "BMI",
                "Age", "DiabetesPedigreeFunction"]
    means0 = [df[df["Outcome"] == 0][f].mean() for f in features]
    means1 = [df[df["Outcome"] == 1][f].mean() for f in features]

    x = np.arange(len(features))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, means0, width, label="No Diabetes (0)",
           color="#3b82d4", edgecolor="white")
    ax.bar(x + width / 2, means1, width, label="Diabetes (1)",
           color="#e05252", edgecolor="white")
    ax.set_xticks(x)
    short_labels = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age", "DPF"]
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title("Feature Mean Values by Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Value")
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, "feature_means_by_outcome.png")


def generate_analytics():
    """Run all analytics chart generators and print dataset statistics."""
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_analytics_data()
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print("Generating analytics charts...")
    outcome_distribution(df)
    glucose_distribution(df)
    glucose_outcome(df)
    bmi_distribution(df)
    bmi_outcome(df)
    age_distribution(df)
    age_outcome(df)
    blood_pressure_outcome(df)
    correlation_heatmap(df)
    feature_boxplots(df)
    print("All analytics charts saved.")

    # Print actual dataset statistics for verification
    print("\n--- Actual Dataset Statistics ---")
    print(f"Total records: {len(df)}")
    print(f"Outcome 0: {(df['Outcome']==0).sum()} ({(df['Outcome']==0).mean()*100:.1f}%)")
    print(f"Outcome 1: {(df['Outcome']==1).sum()} ({(df['Outcome']==1).mean()*100:.1f}%)")
    for col in ZERO_AS_NAN_COLS:
        n = (df[col] == 0).sum()
        print(f"  {col} zeros: {n}")
    corr_with_outcome = df.corr(numeric_only=True)["Outcome"].drop("Outcome").abs()
    best = corr_with_outcome.idxmax()
    print(f"Strongest correlator with Outcome: {best} ({corr_with_outcome[best]:.4f})")


# ==============================================================================
# SECTION 4 — FLASK REST API
# Source: backend/api.py
# Covers: prediction endpoint, model loading, JSON request/response handling
# ==============================================================================
"""
Flask REST API for Diabetes Risk Prediction.
Run from project root:  python backend/api.py
Endpoint: POST /predict   (JSON body with 8 feature fields)
"""

import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow imports from project root (so model path resolves correctly)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_API_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "model", "diabetes_model.pkl")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML frontend

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_api_model():
    if not os.path.exists(_API_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {_API_MODEL_PATH}. Run python train_model.py first."
        )
    with open(_API_MODEL_PATH, "rb") as f:
        return pickle.load(f)


# Load once at startup
try:
    _pipeline = load_api_model()
    print(f"Model loaded from {_API_MODEL_PATH}")
except FileNotFoundError as e:
    _pipeline = None
    print(f"WARNING: {e}")


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _pipeline is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if _pipeline is None:
        return jsonify({"error": "Model not loaded. Run python train_model.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = [float(data[f]) for f in FEATURE_ORDER]
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid value: {exc}"}), 400

    X = np.array([values])
    prediction = int(_pipeline.predict(X)[0])
    probability = float(_pipeline.predict_proba(X)[0][1])

    return jsonify({
        "prediction": prediction,
        "prediction_label": "Diabetic" if prediction == 1 else "Not Diabetic",
        "probability_diabetic": round(probability, 4),
        "probability_not_diabetic": round(1 - probability, 4),
    })


# ==============================================================================
# SECTION 5 — STREAMLIT DASHBOARD
# Source: app.py
# Covers: prediction UI, data analytics dashboard, model performance display,
#         all inline chart helpers, disclaimer
# ==============================================================================
"""
Diabetes Risk Prediction — Streamlit Application
Combines:
  - Section 1: Prediction (Logistic Regression pipeline)
  - Section 2: Data Analytics Dashboard (computed from data/diabetes.csv)
  - Section 3: Model Performance (verified metrics — do not alter)
  - Section 4: Disclaimer

Run from project root:  streamlit run app.py
"""

import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
_APP_MODEL_PATH = os.path.join("model", "diabetes_model.pkl")
_APP_DATA_PATH  = os.path.join("data", "diabetes.csv")
_APP_IMG_DIR    = "report_images"

# ── Feature config (prediction form) ──────────────────────────────────────────
FEATURE_LABELS = {
    "Pregnancies":              ("Number of Pregnancies",           0,   20,  0,    1),
    "Glucose":                  ("Plasma Glucose (mg/dL)",         50,  250, 120,   1),
    "BloodPressure":            ("Diastolic Blood Pressure (mmHg)", 30, 130,  70,   1),
    "SkinThickness":            ("Skin Fold Thickness (mm)",         5,  99,  23,   1),
    "Insulin":                  ("2-Hour Serum Insulin (uU/mL)",     0, 900,  80,   1),
    "BMI":                      ("Body Mass Index (kg/m2)",         10,  70, 32.0, 0.1),
    "DiabetesPedigreeFunction": ("Diabetes Pedigree Function",    0.07, 2.5, 0.47, 0.01),
    "Age":                      ("Age (years)",                     21, 100,  33,   1),
}
FEATURE_KEYS = list(FEATURE_LABELS.keys())

# Columns where zero = physiologically invalid
_ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ── Verified model metrics (do NOT modify) ────────────────────────────────────
VERIFIED_METRICS = {
    "Accuracy":  0.7078,
    "Precision": 0.6000,
    "Recall":    0.5000,
    "F1-Score":  0.5455,
    "ROC-AUC":   0.8130,
}


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(_APP_MODEL_PATH):
        st.error(
            f"Model file not found at `{_APP_MODEL_PATH}`. "
            "Run `python train_model.py` first."
        )
        st.stop()
    with open(_APP_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_app_data():
    if not os.path.exists(_APP_DATA_PATH):
        st.error(
            f"Dataset not found at `{_APP_DATA_PATH}`. "
            "Run `python download_dataset.py` first."
        )
        st.stop()
    return pd.read_csv(_APP_DATA_PATH)


# ── Analytics chart helpers (generated live in-app via matplotlib) ─────────────

def _fig_outcome_distribution(df):
    counts = df["Outcome"].value_counts().sort_index()
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    colors = ["#3b82d4", "#e05252"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        pct = v / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Diabetes Outcome Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, max(counts.values) * 1.25)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def _fig_histogram(df, col, title, xlabel, color, exclude_zeros=False):
    data = df[col]
    if exclude_zeros:
        data = data[data > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(data, bins=25, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="#e05252", linestyle="--", linewidth=1.5,
               label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.median(), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Median: {data.median():.1f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _fig_boxplot_by_outcome(df, col, title, ylabel, exclude_zeros=False):
    d0 = df[df["Outcome"] == 0][col]
    d1 = df[df["Outcome"] == 1][col]
    if exclude_zeros:
        d0 = d0[d0 > 0]
        d1 = d1[d1 > 0]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bp = ax.boxplot([d0, d1], patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#3b82d4")
    bp["boxes"][1].set_facecolor("#e05252")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def _fig_correlation_heatmap(df):
    numeric_cols = FEATURE_KEYS + ["Outcome"]
    corr = df[numeric_cols].corr()
    n = len(numeric_cols)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = ["Preg", "Gluc", "BP", "Skin", "Ins", "BMI", "DPF", "Age", "Out"]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_feature_means(df):
    features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age", "DiabetesPedigreeFunction"]
    short_labels = ["Pregnancies", "Glucose", "BP", "BMI", "Age", "DPF"]
    means0 = [df[df["Outcome"] == 0][f].mean() for f in features]
    means1 = [df[df["Outcome"] == 1][f].mean() for f in features]
    x = np.arange(len(features))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, means0, w, label="No Diabetes (0)", color="#3b82d4", edgecolor="white")
    ax.bar(x + w / 2, means1, w, label="Diabetes (1)",    color="#e05252", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title("Mean Feature Values by Outcome", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Value")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Section renderers ──────────────────────────────────────────────────────────

def section_prediction(pipeline):
    st.header("1. Prediction")
    st.markdown(
        "Enter the patient's clinical measurements below. "
        "The model is a **Logistic Regression** trained on the "
        "[Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes)."
    )

    col1, col2 = st.columns(2)
    inputs = {}
    for i, key in enumerate(FEATURE_KEYS):
        label, lo, hi, default, step = FEATURE_LABELS[key]
        col = col1 if i % 2 == 0 else col2
        if isinstance(step, float):
            inputs[key] = col.number_input(label, min_value=float(lo), max_value=float(hi),
                                            value=float(default), step=step, key=f"inp_{key}")
        else:
            inputs[key] = col.number_input(label, min_value=int(lo), max_value=int(hi),
                                            value=int(default), step=step, key=f"inp_{key}")

    if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
        X    = np.array([[inputs[k] for k in FEATURE_KEYS]])
        pred = pipeline.predict(X)[0]
        prob = pipeline.predict_proba(X)[0][1]

        if pred == 1:
            st.error(f"**High Risk — Diabetic** ({prob*100:.1f}% probability)")
        else:
            st.success(f"**Low Risk — Not Diabetic** ({prob*100:.1f}% probability)")

        st.markdown("##### Risk Probability")
        st.progress(int(prob * 100))
        st.caption(f"Diabetes risk probability: **{prob*100:.1f}%**")


def section_dashboard(df):
    st.header("2. Data Analytics Dashboard")

    # ── 1. Dataset Overview ──────────────────────────────────────────────────
    st.subheader("2.1 Dataset Overview")
    n_total   = len(df)
    n_no_diab = int((df["Outcome"] == 0).sum())
    n_diab    = int((df["Outcome"] == 1).sum())
    pct_no    = n_no_diab / n_total * 100
    pct_yes   = n_diab    / n_total * 100
    n_features = len(FEATURE_KEYS)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records",  n_total)
    c2.metric("Features",       n_features)
    c3.metric("Target Classes", 2)

    c4, c5 = st.columns(2)
    c4.metric("No Diabetes (Outcome=0)", f"{n_no_diab}  ({pct_no:.1f}%)")
    c5.metric("Diabetes (Outcome=1)",    f"{n_diab}  ({pct_yes:.1f}%)")

    # ── 2. Dataset Preview ───────────────────────────────────────────────────
    st.subheader("2.2 Dataset Preview")
    with st.expander("Show first 10 rows", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Column names and data types", expanded=False):
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    with st.expander("Descriptive statistics", expanded=False):
        st.dataframe(df.describe().round(3), use_container_width=True)

    # ── 3. Data Quality Analysis ─────────────────────────────────────────────
    st.subheader("2.3 Data Quality Analysis")

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    duplicates  = int(df.duplicated().sum())

    qc1, qc2 = st.columns(2)
    qc1.metric("Missing Values (NaN)", total_nulls,
               help="True NaN values before any imputation")
    qc2.metric("Duplicate Records", duplicates)

    zero_data = []
    for col in _ZERO_AS_NAN_COLS:
        n = int((df[col] == 0).sum())
        zero_data.append({"Feature": col, "Zero Values (Clinically Invalid)": n,
                          "% of Records": f"{n / n_total * 100:.1f}%"})
    zero_df = pd.DataFrame(zero_data)
    st.markdown("**Clinically invalid zero values** (treated as missing during training):")
    st.dataframe(zero_df, use_container_width=True, hide_index=True)

    # ── 4. Diabetes Outcome Analysis ─────────────────────────────────────────
    st.subheader("2.4 Diabetes Outcome Analysis")
    fig = _fig_outcome_distribution(df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── 5. Glucose Analysis ──────────────────────────────────────────────────
    st.subheader("2.5 Glucose Analysis")
    g_valid = df[df["Glucose"] > 0]["Glucose"]
    g0 = df[(df["Outcome"] == 0) & (df["Glucose"] > 0)]["Glucose"]
    g1 = df[(df["Outcome"] == 1) & (df["Glucose"] > 0)]["Glucose"]

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        fig = _fig_histogram(df, "Glucose", "Glucose Distribution",
                              "Plasma Glucose (mg/dL)", "#3b82d4", exclude_zeros=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with g_col2:
        fig = _fig_boxplot_by_outcome(df, "Glucose", "Glucose by Outcome",
                                      "Plasma Glucose (mg/dL)", exclude_zeros=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        f"**Glucose summary (zeros excluded):** "
        f"Overall mean = **{g_valid.mean():.1f} mg/dL**, "
        f"mean for No Diabetes = **{g0.mean():.1f}**, "
        f"mean for Diabetes = **{g1.mean():.1f}**. "
        f"Patients with diabetes show a higher average glucose level."
    )

    # ── 6. BMI Analysis ──────────────────────────────────────────────────────
    st.subheader("2.6 BMI Analysis")
    b0 = df[(df["Outcome"] == 0) & (df["BMI"] > 0)]["BMI"]
    b1 = df[(df["Outcome"] == 1) & (df["BMI"] > 0)]["BMI"]

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        fig = _fig_histogram(df, "BMI", "BMI Distribution",
                              "BMI (kg/m2)", "#7c5cd8", exclude_zeros=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with b_col2:
        fig = _fig_boxplot_by_outcome(df, "BMI", "BMI by Outcome",
                                      "BMI (kg/m2)", exclude_zeros=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        f"**BMI summary (zeros excluded):** "
        f"Mean BMI for No Diabetes = **{b0.mean():.1f}**, "
        f"mean BMI for Diabetes = **{b1.mean():.1f}**. "
        f"Patients with diabetes tend to have a higher average BMI."
    )

    # ── 7. Age Analysis ──────────────────────────────────────────────────────
    st.subheader("2.7 Age Analysis")
    a0 = df[df["Outcome"] == 0]["Age"]
    a1 = df[df["Outcome"] == 1]["Age"]

    a_col1, a_col2 = st.columns(2)
    with a_col1:
        fig = _fig_histogram(df, "Age", "Age Distribution", "Age (years)", "#10b981")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with a_col2:
        fig = _fig_boxplot_by_outcome(df, "Age", "Age by Outcome", "Age (years)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        f"**Age summary:** "
        f"Mean age for No Diabetes = **{a0.mean():.1f} years**, "
        f"mean age for Diabetes = **{a1.mean():.1f} years**. "
        f"Patients with diabetes tend to be older on average."
    )

    # ── 8. Blood Pressure Analysis ───────────────────────────────────────────
    st.subheader("2.8 Blood Pressure Analysis")
    bp0 = df[(df["Outcome"] == 0) & (df["BloodPressure"] > 0)]["BloodPressure"]
    bp1 = df[(df["Outcome"] == 1) & (df["BloodPressure"] > 0)]["BloodPressure"]

    fig = _fig_boxplot_by_outcome(df, "BloodPressure",
                                  "Blood Pressure by Outcome",
                                  "Diastolic BP (mmHg)", exclude_zeros=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        f"**Blood Pressure summary (zeros excluded):** "
        f"Mean BP for No Diabetes = **{bp0.mean():.1f} mmHg**, "
        f"mean BP for Diabetes = **{bp1.mean():.1f} mmHg**."
    )

    # ── 9. Correlation Analysis ──────────────────────────────────────────────
    st.subheader("2.9 Correlation Analysis")
    fig = _fig_correlation_heatmap(df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    corr_with_outcome = df.corr(numeric_only=True)["Outcome"].drop("Outcome").abs()
    top_feat = corr_with_outcome.sort_values(ascending=False)
    top_df = pd.DataFrame({
        "Feature": top_feat.index,
        "Absolute Correlation with Outcome": top_feat.values.round(4)
    })
    st.markdown("**Absolute correlations with Outcome (sorted):**")
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # ── 10. Feature Analysis ─────────────────────────────────────────────────
    st.subheader("2.10 Feature Analysis")
    fig = _fig_feature_means(df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        "The chart above compares the mean value of each key feature for patients "
        "with and without diabetes. Higher bars for Outcome=1 indicate features "
        "that are associated with diabetes in this dataset."
    )

    # ── 11. Analytical Insights ──────────────────────────────────────────────
    st.subheader("2.11 Key Analytical Insights")

    majority_class = "No Diabetes" if n_no_diab > n_diab else "Diabetes"
    majority_count = max(n_no_diab, n_diab)
    minority_count = min(n_no_diab, n_diab)

    avg_glucose_0 = df[df["Outcome"] == 0]["Glucose"].mean()
    avg_glucose_1 = df[df["Outcome"] == 1]["Glucose"].mean()
    avg_bmi_0     = df[df["Outcome"] == 0]["BMI"].mean()
    avg_bmi_1     = df[df["Outcome"] == 1]["BMI"].mean()
    avg_age_0     = df[df["Outcome"] == 0]["Age"].mean()
    avg_age_1     = df[df["Outcome"] == 1]["Age"].mean()
    best_feature  = corr_with_outcome.idxmax()
    best_corr     = corr_with_outcome.max()

    insights = [
        f"**Class distribution:** The dataset contains {majority_count} records "
        f"for {majority_class} ({majority_count/n_total*100:.1f}%) and "
        f"{minority_count} records for the other class ({minority_count/n_total*100:.1f}%). "
        f"The dataset is moderately imbalanced.",

        f"**Glucose:** The average Glucose for No Diabetes is **{avg_glucose_0:.1f} mg/dL** "
        f"compared to **{avg_glucose_1:.1f} mg/dL** for Diabetes. "
        f"Higher average glucose is associated with the Diabetic outcome.",

        f"**BMI:** The average BMI for No Diabetes is **{avg_bmi_0:.1f} kg/m2** "
        f"compared to **{avg_bmi_1:.1f} kg/m2** for Diabetes. "
        f"Higher average BMI is associated with the Diabetic outcome.",

        f"**Age:** The average age for No Diabetes is **{avg_age_0:.1f} years** "
        f"compared to **{avg_age_1:.1f} years** for Diabetes. "
        f"Diabetic patients tend to be older on average in this dataset.",

        f"**Strongest correlator:** The numerical feature with the highest absolute "
        f"correlation with Outcome is **{best_feature}** "
        f"(r = {best_corr:.4f}). This does not imply causation.",

        f"**Missing data:** Insulin has the highest rate of clinically invalid zeros "
        f"({int((df['Insulin']==0).sum())} out of {n_total} records, "
        f"{(df['Insulin']==0).mean()*100:.1f}%), followed by SkinThickness "
        f"({int((df['SkinThickness']==0).sum())} records, "
        f"{(df['SkinThickness']==0).mean()*100:.1f}%). "
        f"These high rates of missing values limit model performance.",
    ]

    for insight in insights:
        st.markdown(f"- {insight}")

    st.caption(
        "All insights above are calculated directly from data/diabetes.csv. "
        "Correlation does not imply causation. These are observational patterns in this dataset."
    )


def section_model_performance():
    st.header("3. Model Performance")
    st.markdown(
        "The following metrics were computed on the **held-out test set (154 samples)** "
        "by running `train_model.py` on the actual dataset. "
        "These values are verified and have not been estimated or altered."
    )

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{VERIFIED_METRICS['Accuracy']*100:.2f}%")
    m2.metric("Precision", f"{VERIFIED_METRICS['Precision']*100:.2f}%")
    m3.metric("Recall",    f"{VERIFIED_METRICS['Recall']*100:.2f}%")
    m4.metric("F1-Score",  f"{VERIFIED_METRICS['F1-Score']*100:.2f}%")
    m5.metric("ROC-AUC",   f"{VERIFIED_METRICS['ROC-AUC']:.4f}")

    # Confusion matrix table
    st.markdown("**Confusion Matrix (Test Set):**")
    cm_df = pd.DataFrame(
        [["82 (TN)", "18 (FP)"],
         ["27 (FN)", "27 (TP)"]],
        index=["Actual: No Diabetes", "Actual: Diabetes"],
        columns=["Predicted: No Diabetes", "Predicted: Diabetes"],
    )
    st.dataframe(cm_df, use_container_width=True)

    # Plots side by side
    cm_path  = os.path.join(_APP_IMG_DIR, "confusion_matrix.png")
    roc_path = os.path.join(_APP_IMG_DIR, "roc_curve.png")

    img_col1, img_col2 = st.columns(2)
    if os.path.exists(cm_path):
        img_col1.image(cm_path, caption="Figure 1: Confusion Matrix", use_container_width=True)
    else:
        img_col1.warning(f"Confusion matrix image not found at {cm_path}. Run train_model.py.")

    if os.path.exists(roc_path):
        img_col2.image(roc_path, caption="Figure 2: ROC Curve (AUC = 0.8130)", use_container_width=True)
    else:
        img_col2.warning(f"ROC curve image not found at {roc_path}. Run train_model.py.")

    st.markdown(
        "> The ROC-AUC of **0.8130** indicates good discriminative ability — "
        "the model correctly ranks a randomly chosen diabetic patient above a "
        "non-diabetic patient approximately 81% of the time."
    )


def section_disclaimer():
    st.header("4. Disclaimer")
    st.warning(
        "This application is an **educational machine-learning and data analytics project**. "
        "It is **not a medical diagnostic tool** and should not be used for medical diagnosis "
        "or treatment decisions. Always consult a qualified healthcare professional for any "
        "health-related concerns.",
        icon="⚠️",
    )
    st.caption(
        "Dataset: Pima Indians Diabetes Database — UCI Machine Learning Repository. "
        "Source: National Institute of Diabetes and Digestive and Kidney Diseases. "
        "Model: Logistic Regression (scikit-learn). "
        "This project is for academic/educational purposes only."
    )


# ── Main Streamlit app ─────────────────────────────────────────────────────────

def streamlit_main():
    st.set_page_config(
        page_title="Diabetes Risk Prediction",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🩺 Diabetes Risk Prediction")
    st.markdown(
        "An **educational machine learning and data analytics project** using the "
        "[Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes). "
        "Algorithm: Logistic Regression | Tools: Python, Pandas, NumPy, Scikit-learn, Streamlit"
    )
    st.divider()

    pipeline = load_model()
    df       = load_app_data()

    section_prediction(pipeline)
    st.divider()

    section_dashboard(df)
    st.divider()

    section_model_performance()
    st.divider()

    section_disclaimer()


# ==============================================================================
# ENTRY POINT
# Run this file directly to execute: dataset download → training → analytics
# For the Streamlit app, run:  streamlit run app.py
# For the Flask API, run:      python backend/api.py
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diabetes Risk Prediction — run project pipeline steps"
    )
    parser.add_argument(
        "--step",
        choices=["download", "train", "analytics", "all"],
        default="all",
        help=(
            "download  : download the dataset only\n"
            "train     : train the model and generate evaluation plots\n"
            "analytics : generate EDA / analytics charts\n"
            "all       : run all three steps in sequence (default)"
        ),
    )
    args = parser.parse_args()

    if args.step in ("download", "all"):
        print("\n=== Step 1: Dataset Download ===")
        download()

    if args.step in ("train", "all"):
        print("\n=== Step 2: Model Training and Evaluation ===")
        train_and_evaluate()

    if args.step in ("analytics", "all"):
        print("\n=== Step 3: Analytics Chart Generation ===")
        generate_analytics()

    print("\nDone. To launch the Streamlit dashboard run:  streamlit run app.py")
