"""Check the API without a server and without touching the real request log."""

import asyncio
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from . import api

TRIP = {"PULocationID": 166, "DOLocationID": 239, "trip_distance": 2.5}


async def check_api() -> float:
    """Exercise health, prediction, validation, metrics and the request log."""
    with TemporaryDirectory() as directory:
        api.REQUEST_LOG_PATH = Path(directory) / "requests.csv"
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            response = await client.post("/predict", json=TRIP)
            assert response.status_code == 200
            minutes = response.json()["predicted_duration_minutes"]
            assert 0 < minutes < 120, minutes

            bad_requests = (
                {**TRIP, "trip_distance": 0},
                {**TRIP, "trip_distance": 1000},
                {**TRIP, "PULocationID": 0},
                {"PULocationID": 166},
            )
            for bad in bad_requests:
                result = await client.post("/predict", json=bad)
                assert result.status_code == 422, bad

            metrics = await client.get("/metrics")
            assert "taxi_predictions_total 1.0" in metrics.text

        with api.REQUEST_LOG_PATH.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]["PULocationID"] == "166"
    return minutes


if __name__ == "__main__":
    example = asyncio.run(check_api())
    print(f"API checks passed. Example prediction: {example:.2f} minutes.")
