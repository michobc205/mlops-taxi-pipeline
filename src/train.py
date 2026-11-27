"""Train the taxi trip-duration model and track the run with MLflow."""

import json
import os
from pathlib import Path

import click
import joblib
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.prepare_data import FEATURES, TARGET, load_training_data

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "train_metrics.json"

EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(df):
    """Split the prepared table into a train part and a test part."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE)


def train_pipeline(X_train, y_train):
    """Fit one-hot encoding and linear regression as one pipeline."""
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(X_train.to_dict(orient="records"), y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test) -> float:
    """Return the RMSE (in minutes) on the test part."""
    predictions = pipeline.predict(X_test.to_dict(orient="records"))
    return float(root_mean_squared_error(y_test, predictions))


def save_artifacts(pipeline, metrics: dict) -> None:
    """Save the model and the metrics at a predictable place."""
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def register_model(model_info, run_id: str, promote: bool) -> str:
    """Register a new model version. Set the alias only when asked."""
    client = MlflowClient()
    try:
        client.create_registered_model(
            name=MODEL_NAME,
            tags={"framework": "scikit-learn"},
            description="Model to predict ride duration in NYC",
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    version = client.create_model_version(
        name=MODEL_NAME,
        source=model_info.model_uri,
        model_id=model_info.model_id,
        run_id=run_id,
    )
    if promote:
        client.set_registered_model_alias(
            name=MODEL_NAME, version=version.version, alias="production"
        )
    return version.version


def run_training(year: int, month: int, developer: str, promote: bool) -> dict:
    """Run the full training and log everything to MLflow."""
    df = load_training_data(year, month)
    X_train, X_test, y_train, y_test = split_data(df)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "developer": developer,
                "dataset": "green-taxi",
                "entrypoint": "src.train",
                "features": FEATURES,
                "target": TARGET,
            }
        )
        mlflow.log_params(
            {
                "year": year,
                "month": month,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )

        pipeline = train_pipeline(X_train, y_train)
        rmse = evaluate(pipeline, X_test, y_test)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))

        metrics = {
            "rmse": rmse,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        save_artifacts(pipeline, metrics)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )
        version = register_model(model_info, run.info.run_id, promote)

    return {**metrics, "run_id": run.info.run_id, "model_version": version}


@click.command()
@click.option("--year", default=2024, show_default=True, type=int)
@click.option("--month", default=1, show_default=True, type=int)
@click.option(
    "--developer",
    default=lambda: os.getenv("DEVELOPER", "unknown"),
    show_default="DEVELOPER from .env",
    help="Name saved as a tag on the MLflow run.",
)
@click.option("--promote", is_flag=True, help="Set the alias production.")
def main(year: int, month: int, developer: str, promote: bool) -> None:
    """Train the model and print the result as JSON."""
    print(json.dumps(run_training(year, month, developer, promote), indent=2))


if __name__ == "__main__":
    main()
