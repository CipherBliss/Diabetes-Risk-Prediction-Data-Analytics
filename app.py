"""
Diabetes Risk Prediction — Streamlit Application
Combines:
  - Section 1: Prediction (Logistic Regression pipeline)
  - Section 2: Data Analytics Dashboard (computed from data/diabetes.csv)
  - Section 3: Model Performance (verified metrics — do not alter)
  - Section 4: Disclaimer

Run from project root:  streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "diabetes_model.pkl")
DATA_PATH   = os.path.join("data", "diabetes.csv")
IMG_DIR     = "report_images"

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
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

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
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Run `python train_model.py` first."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Run `python download_dataset.py` first."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


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
    for col in ZERO_AS_NAN_COLS:
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
    cm_path  = os.path.join(IMG_DIR, "confusion_matrix.png")
    roc_path = os.path.join(IMG_DIR, "roc_curve.png")

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


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
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
    df       = load_data()

    section_prediction(pipeline)
    st.divider()

    section_dashboard(df)
    st.divider()

    section_model_performance()
    st.divider()

    section_disclaimer()


if __name__ == "__main__":
    main()
