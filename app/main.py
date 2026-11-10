import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import BatchRequest, BatchResponse, PredictionResponse, StudentFeatures
from src.config import FEATURES, METRICS_PATH, MODEL_PATH

state = {"model": None}


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    yield
    state["model"] = None


app = FastAPI(
    title="Student Placement Prediction API",
    description="Predicts whether a student will be placed, based on academic and skill attributes.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model():
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.train` first.")
    return model


def _to_response(placed: int, prob: float) -> PredictionResponse:
    return PredictionResponse(
        placed=bool(placed),
        prediction="Placed" if placed else "Not Placed",
        placement_probability=round(float(prob), 4),
    )


@app.get("/", tags=["General"])
def root():
    return {"message": "Student Placement Prediction API is running", "docs": "/docs"}


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model-info", tags=["General"])
def model_info():
    get_model()
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {"features": FEATURES}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentFeatures):
    model = get_model()
    X = pd.DataFrame([student.model_dump()])[FEATURES]
    proba = model.predict_proba(X)[0, 1]
    return _to_response(int(proba >= 0.5), proba)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest):
    model = get_model()
    X = pd.DataFrame([s.model_dump() for s in req.students])[FEATURES]
    probas = model.predict_proba(X)[:, 1]
    preds = [_to_response(int(p >= 0.5), p) for p in probas]
    return BatchResponse(count=len(preds), predictions=preds)
