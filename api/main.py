import os
import sys
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field
import pandas as pd

# Make the project root importable regardless of how/where uvicorn is
# launched from, so "from src.predict import ..." below always works.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.predict import load_pipeline_with_fallback, predict_churn

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded ONCE when the server starts, not on every request. Reuses
    # the exact same fallback logic built in Lab 9 -- if the primary
    # unified pipeline is missing, the API degrades gracefully instead
    # of refusing to start at all.
    pipeline, mode = load_pipeline_with_fallback()
    ml_model["pipeline"] = pipeline
    ml_model["mode"] = mode
    print(f"[STARTUP] Model loaded successfully. Mode: {mode}")
    yield
    ml_model.clear()
    print("[SHUTDOWN] Model cleared.")


app = FastAPI(
    title="Churn Prediction API",
    description="Lab 10 - ML Model Serving with FastAPI",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------
# Stage 6: Standardized error handling.
#
# Without these, three different kinds of failure would return three
# different JSON shapes: Pydantic validation errors, our own
# HTTPException calls, and any genuinely unexpected server error. These
# three handlers force EVERY error response through the same shape:
#   { "error": true, "status_code": <int>, "message": <str>, "details": <optional> }
# so a client never has to special-case how to parse an error.
# ---------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status_code": 422,
            "message": "Request validation failed.",
            "details": exc.errors()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    # Catches BOTH our own raise HTTPException(...) calls AND FastAPI's
    # built-in ones (like 404 for an unmatched route).
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": str(exc.detail),
            "details": None
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # The genuine last resort: something broke that none of the above
    # anticipated. Never leak a raw Python traceback to a client --
    # return a clean 500 in the same standardized shape instead.
    print(f"[UNHANDLED ERROR] {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "An unexpected internal error occurred.",
            "details": None
        }
    )


@app.get("/health")
def health_check():
    """Basic liveness/readiness check: is the server up, and is a
    usable model actually loaded (primary or fallback)?"""
    return {
        "status": "ok",
        "model_loaded": "pipeline" in ml_model,
        "model_mode": ml_model.get("mode")
    }


# ---------------------------------------------------------------------
# Stage 2: Input/output schema contracts.
#
# These constraints are NOT invented -- they're pulled directly from
# the same Pandera schema already enforced in src/validate_data.py
# (get_telco_schema()), so the API's contract and the pipeline's data
# validation agree with each other rather than drifting apart.
# ---------------------------------------------------------------------

class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=72, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0, le=1000)
    TotalCharges: str = Field(..., description="Numeric string; blank allowed for new customers")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
                "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
            }
        }


class PredictionOutput(BaseModel):
    predicted_churn: int = Field(..., description="0 = No churn, 1 = Churn")
    churn_probability: float = Field(..., ge=0, le=1)
    model_mode: Literal["primary", "fallback"]


# ---------------------------------------------------------------------
# Stage 3: Single prediction endpoint.
#
# Deliberately thin -- all the real prediction logic (defensive column
# cleanup, calling .predict()/.predict_proba()) already exists and was
# already tested in src/predict.py (Lab 7). This endpoint is just an
# HTTP adapter on top of it, not a reimplementation.
# ---------------------------------------------------------------------

@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not currently loaded.")
    mode = ml_model["mode"]

    df = pd.DataFrame([customer.dict()])
    result = predict_churn(df, pipeline=pipeline)

    return PredictionOutput(
        predicted_churn=int(result["Predicted_Churn"].iloc[0]),
        churn_probability=round(float(result["Churn_Probability"].iloc[0]), 4),
        model_mode=mode
    )


# ---------------------------------------------------------------------
# Stage 4: Metadata endpoint.
#
# Reads the REAL Lab 7 evaluation report at request time (not baked in
# at startup), so /metadata always reflects whatever the most recent
# evaluation run actually produced -- if you retrain and re-evaluate,
# this endpoint's numbers update without restarting the server.
# ---------------------------------------------------------------------

EVAL_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..',
    'reports', 'lab7_full_pipeline_evaluation_report.json'
)


@app.get("/metadata")
def get_metadata():
    metrics = None
    evaluated_on = None
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH) as f:
            report = json.load(f)
        metrics = report.get("metrics")
        evaluated_on = report.get("evaluated_on")

    return {
        "model_name": "Churn Prediction - Unified Pipeline",
        "model_mode_active": ml_model.get("mode"),
        "model_type": "RandomForestClassifier (sklearn Pipeline: preprocessing + classifier)",
        "expected_input_fields": list(CustomerInput.model_fields.keys()),
        "output_fields": list(PredictionOutput.model_fields.keys()),
        "latest_evaluation_metrics": metrics,
        "evaluated_on": evaluated_on
    }


# ---------------------------------------------------------------------
# Stage 5: Batch inference endpoint.
#
# Same schema contract as /predict, just wrapped in a List[]. Reuses
# predict_churn() with the FULL batch as one DataFrame in a single call
# -- not a Python loop calling /predict's logic N times -- since the
# underlying sklearn pipeline is already vectorized and handles many
# rows at once far more efficiently than row-by-row.
# ---------------------------------------------------------------------

@app.post("/predict/batch", response_model=list[PredictionOutput])
def predict_batch(customers: list[CustomerInput]):
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Batch request contained zero customers.")

    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not currently loaded.")
    mode = ml_model["mode"]

    df = pd.DataFrame([c.dict() for c in customers])
    result = predict_churn(df, pipeline=pipeline)

    return [
        PredictionOutput(
            predicted_churn=int(row["Predicted_Churn"]),
            churn_probability=round(float(row["Churn_Probability"]), 4),
            model_mode=mode
        )
        for _, row in result.iterrows()
    ]