"""
Download the Pima Indians Diabetes Dataset.

Source: UCI Machine Learning Repository
  - URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
  - Original: National Institute of Diabetes and Digestive and Kidney Diseases
  - UCI Reference: https://archive.ics.uci.edu/ml/datasets/diabetes
  - Kaggle mirror: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

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


if __name__ == "__main__":
    download()
