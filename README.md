# Taxi trip-duration MLOps pipeline

A neuefische bootcamp project (Module 4, MLOps). It turns a notebook that predicts the duration of a NYC green taxi trip into a small, repeatable pipeline: training code, data and run tracking, automatic training in CI, a prediction API, and monitoring.

> **Scope.** The model predicts the trip duration in minutes from the pickup zone, the drop-off zone and the trip distance. The distance is the taximeter distance measured **during** the trip, so it is not known at pickup. This is a learning project, not a production system.

## The big picture

~~~mermaid
flowchart LR
  N[Notebook] --> T[src/train.py]
  D[(Data, DVC pointer)] --> T
  T --> M[models/model.joblib]
  T --> F[MLflow runs and registry]
  G[GitHub Actions] -. runs .-> T
  M --> A[FastAPI /predict]
  A --> L[monitoring/requests.csv]
  A -- /metrics --> P[Prometheus]
  P --> R[Grafana dashboard]
  L --> E[Evidently drift report]
~~~

## What is in this repository

| Step | What | Where |
|---|---|---|
| 1 | Training code outside the notebook | `src/prepare_data.py`, `src/train.py` |
| 2 | Data versioning and run tracking | `data/*.dvc`, `.dvc/`, MLflow (`mlflow.db`, not committed) |
| 3 | Automatic training and API smoke test in CI | `.github/workflows/train.yaml` |
| 4 | Prediction API | `src/api.py`, `src/check_api.py` |
| 5 | Monitoring | `docker-compose.yaml`, `Dockerfile`, `monitoring/`, `src/send_requests.py`, `src/generate_report.py` |

All code in this repository is my own work. The notebook that this project started from is part of the school's project template (https://github.com/neuefische/mle-mlops-project) and is not included here.

## Quick start

You need [uv](https://docs.astral.sh/uv/) 0.13.5 or newer (it can install Python 3.13) and, for the monitoring stack, Docker Desktop. The commands were run with Git Bash on Windows.

### 1. Install and configure

~~~bash
uv sync
cp .env.example .env
~~~

Put your name after `DEVELOPER=` in `.env`. It is saved as a tag on each MLflow run. The `.env` file is not committed.

### 2. Train

~~~bash
uv run python -m src.train
~~~

This downloads the January 2024 green taxi file when it is missing, trains a linear regression with one-hot encoded zones, and writes `models/model.joblib` and `reports/train_metrics.json`. It logs the run to MLflow (`mlflow.db`) and registers a model version. Add `--promote` to set the alias `production` on that version.

### 3. The data pointer

DVC tracks the data file with a small pointer, `data/green_tripdata_2024-01.parquet.dvc`. It holds the md5 and the size of the file that was used. The data file itself is not in Git. The DVC remote of this project is a local folder outside the repository, so `dvc pull` only works on the machine that has that folder.

### 4. Start the API

~~~bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

In a second terminal:

~~~bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"PULocationID": 166, "DOLocationID": 239, "trip_distance": 2.5}'
~~~

The answer is about `{"predicted_duration_minutes":13.87}`. The interactive description is at `http://127.0.0.1:8000/docs`.

### 5. Test the API without a server

~~~bash
uv run python -m src.check_api
~~~

### 6. Start the monitoring stack

~~~bash
docker compose up --build -d
~~~

| Service | Address | Note |
|---|---|---|
| API | http://127.0.0.1:8000 | needs `models/model.joblib` (step 2) |
| Prometheus | http://127.0.0.1:9090 | reads the API metrics every 5 seconds |
| Grafana | http://127.0.0.1:3000/d/taxi-api-monitoring | login `admin` / `admin`, local only |

The first build downloads about 190 packages and took about 12 minutes on the author's laptop. `docker compose down` keeps the stored data. `docker compose down -v` deletes it. The dashboard is loaded from `monitoring/grafana/dashboards/taxi-api.json`.

### 7. Send request batches and make a drift report

The API appends every request to `monitoring/requests.csv`. Rename that file after each batch, for example to `monitoring/batch1_normal.csv`, so the batches stay apart.

~~~bash
uv run python -m src.send_requests --n 300 --shift 1.0
uv run python -m src.send_requests --n 300 --shift 3.0
uv run python -m src.generate_report --log monitoring/batch1_normal.csv --name batch1_normal
~~~

`--shift 3.0` multiplies every distance by 3. The report is written to `monitoring/reports/` and is not committed. It compares the logged requests with the training data and with the predictions of the current model.

## Results

Measured on the author's laptop on 27 November 2026.

| What | Result |
|---|---|
| Error (RMSE) of the notebook and of `src/train.py` | 5.068848811406987 minutes, identical |
| Rows | 56,551 raw, 52,386 after the rules, 41,908 train, 10,478 test |
| API example (zone 166 to 239, 2.5 miles) | 13.87 minutes |
| Normal batch, 300 requests | mean distance 2.65 miles, mean prediction 13.45 minutes, Evidently flags 2 of 4 columns (the two zone columns, a false alarm) |
| Shifted batch, 300 requests | mean distance 7.95 miles, mean prediction 26.56 minutes, longest 157.3, 18 above 60 minutes, Evidently flags 4 of 4 columns |
| Samples from the training data itself | 300 rows: 2 of 4 flagged. 1,000: 2. 3,000: 1. 10,000: 0. |
| CI (in the private working repository where this was built) | 3 pull requests, all green, 31 to 35 seconds |

## Things to know

- The API takes whole numbers for the zones and turns them into text, as in training. A model that gets numbers instead of text silently ignores the zones and answers about 0.95 minutes too low.
- The model is linear and has no ceiling. For stretched trips it answered up to 157.3 minutes, and the API does not warn.
- A drift report with few rows gives false alarms. Use thousands of rows.
- The model is loaded from `models/model.joblib`. These files are Python pickles: load only files that you made yourself. Loading the model by its MLflow alias would be better.
- CI trains the model and runs the API smoke test. It does not build the Docker image, and it does not run the monitoring stack.
- The dashboard covers traffic, latency and availability. It has no errors panel, no saturation and no alert routing.
- Grafana uses the default login and all ports are bound to 127.0.0.1. This is for local learning, not for a public deployment.

## Data and licence

The data are the public trip records of the NYC Taxi and Limousine Commission (green taxis). They are downloaded when needed and are not stored in this repository. This project is not affiliated with or endorsed by the commission. The code is released under the MIT licence (see `LICENSE`).
