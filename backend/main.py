from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.predictor import predict_claim
from backend.database import (
    save_prediction,
    get_prediction_history
)


# ---------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Fake News Detection API",
    description=(
        "Backend API for the Fake News Detection using NLP project."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request Schema
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    claim: str


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "status": "running"
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "TF-IDF + Logistic Regression",
        "version": "1.0.0"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        claim = request.claim.strip()

        if not claim:
            raise HTTPException(
                status_code=400,
                detail="Claim cannot be empty."
            )

        # -------------------------------------------------
        # Run model
        # -------------------------------------------------

        result = predict_claim(claim)

        prediction = result["prediction"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]


        # -------------------------------------------------
        # Save prediction in MySQL
        # -------------------------------------------------

        prediction_id = save_prediction(
            claim=claim,
            predicted_label=prediction,
            confidence=confidence
        )


        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return {
            "success": True,
            "prediction_id": prediction_id,
            "claim": claim,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }

    except HTTPException:
        raise

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        print(f"Prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the claim."
        )


# ---------------------------------------------------------
# Prediction History Endpoint
# ---------------------------------------------------------

@app.get("/history")
def history(limit: int = 20):

    if limit < 1:
        limit = 1

    if limit > 100:
        limit = 100

    results = get_prediction_history(limit)

    return {
        "success": True,
        "count": len(results),
        "predictions": results
    }