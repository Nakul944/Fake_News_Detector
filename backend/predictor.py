from pathlib import Path

import joblib


# ---------------------------------------------------------
# Model path
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "baseline_model.pkl"


# ---------------------------------------------------------
# Load model only once
# ---------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)

    print(f"Model loaded successfully from: {MODEL_PATH}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_claim(claim: str) -> dict:

    if claim is None:
        raise ValueError("Claim cannot be empty.")

    claim = claim.strip()

    if not claim:
        raise ValueError("Claim cannot be empty.")

    # Predict label
    prediction = model.predict([claim])[0]

    # Get probabilities
    probabilities = model.predict_proba([claim])[0]

    classes = model.classes_

    probability_dict = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    # Confidence = probability of predicted class
    confidence = probability_dict[str(prediction)]

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "probabilities": probability_dict
    }