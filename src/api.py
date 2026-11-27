"""Prediction API for the taxi trip-duration model."""

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"
LOG_FIELDS = ["timestamp", "PULocationID", "DOLocationID", "trip_distance", "prediction"]

PREDICTIONS = Counter("taxi_predictions_total", "Number of predictions made.")
LATENCY = Histogram("taxi_prediction_latency_seconds", "Time to make one prediction.")
PREDICTED_MINUTES = Histogram(
    "taxi_predicted_duration_minutes",
    "Predicted trip duration in minutes.",
    buckets=[5, 10, 15, 20, 30, 45, 60, 90],
)
INPUT_DISTANCE = Histogram(
    "taxi_input_trip_distance_miles",
    "Trip distance sent by the caller, in miles.",
    buckets=[0.5, 1, 2, 3, 5, 10, 20, 50, 100],
)

app = FastAPI(
    title="Taxi trip-duration API",
    version="1.0.0",
    description=(
        "Predicts the duration of a NYC green taxi trip in minutes. "
        "trip_distance is the taximeter distance measured DURING the trip, "
        "so it is not known at pickup."
    ),
)
logger = logging.getLogger(__name__)
log_lock = threading.Lock()


class TripRequest(BaseModel):
    """Input of one prediction."""

    PULocationID: int = Field(..., ge=1, examples=[166])
    DOLocationID: int = Field(..., ge=1, examples=[239])
    trip_distance: float = Field(..., ge=0.1, le=100, allow_inf_nan=False, examples=[2.5])


class TripResponse(BaseModel):
    """Output of one prediction."""

    predicted_duration_minutes: float


def load_model() -> Any:
    """Load the trained model from disk once, at startup."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model file is missing. Run: uv run python -m src.train"
        )
    return joblib.load(MODEL_PATH)


MODEL = load_model()


def append_request_log(row: dict) -> None:
    """Append one request and its prediction to the log used by Evidently."""
    with log_lock:
        REQUEST_LOG_PATH.parent.mkdir(exist_ok=True)
        new_file = not REQUEST_LOG_PATH.exists()
        with REQUEST_LOG_PATH.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. It does not call the model."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose the metrics in the Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=TripResponse)
def predict(trip: TripRequest) -> TripResponse:
    """Predict the trip duration in minutes and log the request."""
    start = perf_counter()
    try:
        # Training turned the zone IDs into strings, so we must do the same.
        record = {
            "PULocationID": str(trip.PULocationID),
            "DOLocationID": str(trip.DOLocationID),
            "trip_distance": trip.trip_distance,
        }
        minutes = float(MODEL.predict([record])[0])
        now = datetime.now(timezone.utc).isoformat()
        append_request_log({"timestamp": now, **record, "prediction": minutes})
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
    PREDICTIONS.inc()
    INPUT_DISTANCE.observe(trip.trip_distance)
    PREDICTED_MINUTES.observe(minutes)
    LATENCY.observe(perf_counter() - start)
    return TripResponse(predicted_duration_minutes=minutes)
