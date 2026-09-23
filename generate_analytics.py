"""
Generate all analytics charts from data/diabetes.csv and save to report_images/.
Run once before the Streamlit app, or let app.py call this on first run.
Does NOT overwrite confusion_matrix.png, roc_curve.png, or metrics.txt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_PATH = os.path.join("data", "diabetes.csv")
IMG_DIR   = "report_images"

ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

PROTECTED = {"confusion_matrix.png", "roc_curve.png", "metrics.txt"}


def save(fig, filename):
    path = os.path.join(IMG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_data():
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
    save(fig, "outcome_distribution.png")


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
    save(fig, "glucose_distribution.png")


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
    save(fig, "glucose_outcome.png")


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
    save(fig, "bmi_distribution.png")


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
    save(fig, "bmi_outcome.png")


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
    save(fig, "age_distribution.png")


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
    save(fig, "age_outcome.png")


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
    save(fig, "blood_pressure_outcome.png")


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
    save(fig, "correlation_heatmap.png")


def feature_boxplots(df):
    """Side-by-side bar chart of feature means by Outcome."""
    features = ["Pregnancies", "Glucose", "BloodPressure", "BMI",
                "Age", "DiabetesPedigreeFunction"]
    means0 = [df[df["Outcome"] == 0][f].mean() for f in features]
    means1 = [df[df["Outcome"] == 1][f].mean() for f in features]

    x = np.arange(len(features))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars0 = ax.bar(x - width / 2, means0, width, label="No Diabetes (0)",
                   color="#3b82d4", edgecolor="white")
    bars1 = ax.bar(x + width / 2, means1, width, label="Diabetes (1)",
                   color="#e05252", edgecolor="white")
    ax.set_xticks(x)
    short_labels = ["Pregnancies", "Glucose", "BloodPressure", "BMI",
                    "Age", "DPF"]
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title("Feature Mean Values by Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Value")
    ax.legend()
    fig.tight_layout()
    save(fig, "feature_means_by_outcome.png")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_data()
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


if __name__ == "__main__":
    main()
