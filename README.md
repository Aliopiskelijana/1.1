# Predictive Maintenance ML System

A production-ready machine failure prediction system using the **AI4I 2020** dataset.

## Architecture

```
train.py              ← Training entry point
src/
  data/
    loader.py         ← Dataset loading (real + synthetic fallback)
    preprocessor.py   ← Feature engineering + StandardScaler
  models/
    trainer.py        ← RF + XGBoost + SMOTE + MLflow tracking
    explainer.py      ← SHAP explainability (EU AI Act)
  api/
    app.py            ← FastAPI application factory
    routes.py         ← /predict, /explain, /health
    schemas.py        ← Pydantic request/response models
    auth.py           ← API key authentication
    model_store.py    ← Singleton model loader
models/saved/         ← Persisted model artifacts
tests/                ← Pytest test suite
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add real dataset
#    Download from: https://archive.ics.uci.edu/dataset/601
cp ai4i2020.csv data/raw/

# 3. Train
python train.py

# 4. Start API
uvicorn src.api.app:app --reload --port 8000

# 5. Run tests
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict` | Predict failure probability |
| POST | `/api/v1/explain` | Predict + SHAP explanation |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.5,
    "Rotational speed [rpm]": 1450,
    "Torque [Nm]": 55.0,
    "Tool wear [min]": 210
  }'
```

### Example Response

```json
{
  "failure_predicted": true,
  "failure_probability": 0.6823,
  "risk_level": "HIGH",
  "threshold_used": 0.3,
  "model_version": "xgboost"
}
```

## Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# MLflow UI: http://localhost:5000
```

## ML Design Decisions

| Topic | Choice | Reason |
|-------|--------|--------|
| Class imbalance | SMOTE + `scale_pos_weight` | 3.4% failure rate |
| Evaluation | Precision-Recall AUC | Better than ROC for imbalanced data |
| Threshold | F2-score tuning (β=2) | Recall > Precision for safety-critical systems |
| Explainability | SHAP TreeExplainer | EU AI Act / audit trail |
| Tracking | MLflow | Experiment comparison, model registry |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (none) | API authentication key |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow server URI |
| `LOG_LEVEL` | `INFO` | Logging level |
