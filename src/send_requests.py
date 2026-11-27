"""Send a batch of real taxi trips to the running API, normal or shifted."""

import time
from statistics import mean

import click
import requests

from src.prepare_data import FEATURES, load_training_data

API_URL = "http://127.0.0.1:8000/predict"


def build_payloads(year: int, month: int, n: int, seed: int, shift: float) -> list:
    """Sample real trips. A shift above 1 stretches the distances."""
    df = load_training_data(year, month)[FEATURES].sample(n=n, random_state=seed)
    df["trip_distance"] = (df["trip_distance"] * shift).clip(0.1, 100)
    return [
        {
            "PULocationID": int(row.PULocationID),
            "DOLocationID": int(row.DOLocationID),
            "trip_distance": round(float(row.trip_distance), 2),
        }
        for row in df.itertuples()
    ]


@click.command()
@click.option("--year", default=2024, show_default=True, type=int)
@click.option("--month", default=1, show_default=True, type=int)
@click.option("--n", default=300, show_default=True, type=int, help="Number of trips.")
@click.option("--shift", default=1.0, show_default=True, type=float,
              help="Multiply every distance by this factor.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--pause", default=0.1, show_default=True, type=float,
              help="Seconds between two requests.")
@click.option("--url", default=API_URL, show_default=True)
def main(year, month, n, shift, seed, pause, url) -> None:
    """Send the trips one by one and print a short summary."""
    payloads = build_payloads(year, month, n, seed, shift)
    codes: dict = {}
    predictions = []
    started = time.time()
    for payload in payloads:
        response = requests.post(url, json=payload, timeout=10)
        codes[response.status_code] = codes.get(response.status_code, 0) + 1
        if response.ok:
            predictions.append(response.json()["predicted_duration_minutes"])
        time.sleep(pause)
    print(f"sent: {len(payloads)} | status codes: {codes}")
    print(f"mean input distance: {mean(p['trip_distance'] for p in payloads):.2f} miles")
    if predictions:
        print(f"mean predicted duration: {mean(predictions):.2f} minutes")
    print(f"time: {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
