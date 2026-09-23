"""
Flask REST API for Diabetes Risk Prediction.
Run from project root:  python backend/api.py
Endpoint: POST /predict   (JSON body with 8 feature fields)
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow imports from project root (so model path resolves correctly)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run python train_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# Load once at startup
try:
    _pipeline = load_model()
    print(f"Model loaded from {MODEL_PATH}")
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
