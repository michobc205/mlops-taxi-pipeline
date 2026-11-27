"""Compare logged API requests with the training data and save an Evidently report."""

import warnings
from pathlib import Path

import click
import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.prepare_data import FEATURES, load_training_data

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
REPORT_DIR = ROOT_DIR / "monitoring" / "reports"
ZONES = ["PULocationID", "DOLocationID"]
COLUMNS = FEATURES + ["prediction"]
MIN_ROWS = 30  # a demo guard, not a statistical rule

DEFINITION = DataDefinition(
    numerical_columns=["trip_distance", "prediction"],
    categorical_columns=ZONES,
)


def build_reference(year: int, month: int) -> pd.DataFrame:
    """Training trips plus the prediction of the current model."""
    reference = load_training_data(year, month)[FEATURES].copy()
    model = joblib.load(MODEL_PATH)
    reference["prediction"] = model.predict(reference.to_dict(orient="records"))
    return reference[COLUMNS]


def load_current(log_path: Path) -> pd.DataFrame:
    """Logged requests. Zone IDs must be text, like in training."""
    current = pd.read_csv(log_path)[COLUMNS].copy()
    current[ZONES] = current[ZONES].astype(str)
    return current


@click.command()
@click.option("--log", "log_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True, help="Name of the HTML file.")
@click.option("--year", default=2024, show_default=True, type=int)
@click.option("--month", default=1, show_default=True, type=int)
def main(log_path: Path, name: str, year: int, month: int) -> None:
    """Run the drift report and print the results."""
    current = load_current(log_path)
    if len(current) < MIN_ROWS:
        raise click.ClickException(f"The log has {len(current)} rows, need {MIN_ROWS}.")
    reference = build_reference(year, month)
    print(f"reference rows: {len(reference)} | current rows: {len(current)}")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference, data_definition=DEFINITION),
        current_data=Dataset.from_pandas(current, data_definition=DEFINITION),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}.html"
    snapshot.save_html(str(path))
    print(f"report: {path.relative_to(ROOT_DIR)} ({path.stat().st_size} bytes)")
    for metric in snapshot.dict()["metrics"]:
        print(f"  {metric['metric_name']} -> {metric['value']}")


if __name__ == "__main__":
    main()
