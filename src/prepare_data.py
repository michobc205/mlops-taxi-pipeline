"""Download and prepare the NYC Green Taxi data for training."""

from pathlib import Path
from urllib.request import urlretrieve

import click
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"


def raw_data_path(year: int, month: int, color: str = "green") -> Path:
    """Return the local path of one monthly trip file."""
    return DATA_DIR / f"{color}_tripdata_{year}-{month:02d}.parquet"


def download_data(year: int, month: int, color: str = "green") -> Path:
    """Download one monthly file if it is not there yet."""
    path = raw_data_path(year, month, color)
    if path.exists():
        return path

    DATA_DIR.mkdir(exist_ok=True)
    temp_path = path.with_name(path.name + ".part")
    urlretrieve(f"{BASE_URL}/{path.name}", temp_path)
    temp_path.replace(path)
    return path


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the duration and keep sensible trips.

    Works on a copy, so the table of the caller is never changed.
    """
    df = df.copy()
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    location_ids = ["PULocationID", "DOLocationID"]
    df[location_ids] = df[location_ids].astype(str)
    return df[FEATURES + [TARGET]]


def load_training_data(year: int, month: int, color: str = "green") -> pd.DataFrame:
    """Download (if needed), read and preprocess one month."""
    return preprocess(pd.read_parquet(download_data(year, month, color)))


@click.command()
@click.option("--year", default=2024, show_default=True, type=int)
@click.option("--month", default=1, show_default=True, type=int)
@click.option("--color", default="green", show_default=True)
def main(year: int, month: int, color: str) -> None:
    """Download one month and print a short summary."""
    path = download_data(year, month, color)
    raw = pd.read_parquet(path)
    columns_before = raw.shape[1]
    prepared = preprocess(raw)
    print(f"file: {path.relative_to(ROOT_DIR)}")
    print(f"raw rows: {len(raw)} | raw columns before: {columns_before}")
    print(f"raw columns after preprocess (must be the same): {raw.shape[1]}")
    print(f"prepared rows: {len(prepared)} | columns: {list(prepared.columns)}")


if __name__ == "__main__":
    main()
