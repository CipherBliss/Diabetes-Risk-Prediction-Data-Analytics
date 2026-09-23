# Diabetes Risk Prediction

An academic machine learning project that predicts diabetes risk using the Pima Indians Diabetes Dataset and Logistic Regression.

---

## Dataset

| Property | Detail |
|---|---|
| Name | Pima Indians Diabetes Database |
| Source | National Institute of Diabetes and Digestive and Kidney Diseases |
| UCI Repository | https://archive.ics.uci.edu/ml/datasets/diabetes |
| Kaggle Mirror |https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset |
| Raw CSV | https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv |
| Records | 768 |
| Features | 8 numeric clinical measurements |
| Target | Outcome (1 = Diabetic, 0 = Not Diabetic) |

---

## Project Structure

```
Diabetes_Risk_Prediction_Project/
├── backend/
│   └── api.py               # Flask REST API
├── data/
│   └── diabetes.csv         # Downloaded dataset
├── frontend/
│   └── index.html           # Static HTML/JS frontend
├── model/
│   └── diabetes_model.pkl   # Trained pipeline (generated)
├── report_images/
│   ├── confusion_matrix.png # Test-set confusion matrix (generated)
│   ├── roc_curve.png        # ROC curve (generated)
│   └── metrics.txt          # Plain-text metrics (generated)
├── reports/
│   └── report.md            # Academic report
├── app.py                   # Streamlit application
├── download_dataset.py      # Dataset downloader
├── train_model.py           # Model training script
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python download_dataset.py
```

### 3. Train the model

```bash
python train_model.py
```

This will:
- Replace physiologically invalid zeros with NaN and impute with training-median
- Perform an 80/20 stratified split
- Train a Logistic Regression pipeline
- Print evaluation metrics
- Save `model/diabetes_model.pkl`
- Save plots to `report_images/`

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Run the Flask REST API

```bash
python backend/api.py
```

API runs on http://localhost:5000

#### Example API request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"Pregnancies\":2,\"Glucose\":138,\"BloodPressure\":70,\"SkinThickness\":28,\"Insulin\":94,\"BMI\":33.6,\"DiabetesPedigreeFunction\":0.627,\"Age\":45}"
```

Expected response:
```json
{
  "prediction": 1,
  "prediction_label": "Diabetic",
  "probability_diabetic": 0.8421,
  "probability_not_diabetic": 0.1579
}
```

### 6. Open the HTML frontend

Open `frontend/index.html` in a browser **while the Flask API is running**.

---

## Model Performance (Actual — Test Set, 20% Hold-out)

| Metric | Value |
|---|---|
| Accuracy | 0.7078 |
| Precision (Diabetic) | 0.6000 |
| Recall (Diabetic) | 0.5000 |
| F1-Score (Diabetic) | 0.5455 |
| ROC-AUC | 0.8130 |

### Confusion Matrix

|  | Predicted: No Diabetes | Predicted: Diabetes |
|---|---|---|
| **Actual: No Diabetes** | 82 (TN) | 18 (FP) |
| **Actual: Diabetes** | 27 (FN) | 27 (TP) |

---

## Preprocessing Details

| Feature | Zero Treatment |
|---|---|
| Pregnancies | Zero is valid (no change) |
| Glucose | Zero → NaN → imputed with training median |
| BloodPressure | Zero → NaN → imputed with training median |
| SkinThickness | Zero → NaN → imputed with training median |
| Insulin | Zero → NaN → imputed with training median |
| BMI | Zero → NaN → imputed with training median |
| DiabetesPedigreeFunction | No zeros in dataset |
| Age | No zero-value treatment needed |

---

## Libraries Used

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Pipeline, imputer, scaler, Logistic Regression, metrics |
| matplotlib | Generating evaluation plots |
| streamlit | Interactive web application |
| flask + flask-cors | REST API |

---

## Notes

- All metrics were computed on the actual held-out test set after training. No values are estimated or invented.
- The dataset and model are fully reproducible: fixed `random_state=42`, fixed split ratio.
- This project is for academic/educational purposes only. Not for clinical use.
