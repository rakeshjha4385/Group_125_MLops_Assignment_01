# Heart Disease Analysis – End-to-End MLOps Pipeline

## Project Overview
This project implements a complete Machine Learning Operations (MLOps) pipeline for predicting heart disease using the UCI Heart Disease Cleveland dataset. The solution covers data ingestion, exploratory data analysis (EDA), model training, containerization, CI/CD automation, and real-time monitoring.

The objective is to demonstrate industry-standard practices for building, deploying, and operating ML systems reliably on local Kubernetes.

---

### Machine Learning
- Data ingestion from UCI repository
- Detailed EDA with statistical analysis
- Preprocessing using `ColumnTransformer`
- Logistic Regression and Random Forest models
- Hyperparameter tuning using `GridSearchCV`
- Metrics comparison across models
- Model serialization using `joblib`

### MLOps
- Dockerized FastAPI inference API
- Kubernetes deployment using Helm charts
- Prometheus metrics instrumentation
- Grafana dashboards for observability
- GitHub Actions CI pipeline:
  - Linting (Black + Flake8)
  - Unit tests (Pytest)
  - Dry-run training validation
- Centralized logging
- Reproducible experiments tracked via MLflow

---

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker Desktop
- Kubernetes (enabled in Docker Desktop)
- Helm
- Git

---

### Local Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/Group_125_MLops_Assignment_01.git
cd Group_125_MLops_Assignment_01

2. Create virtual environment:
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Download dataset:
python -m src.download_data

5. Run EDA:
python notebooks/eda.py

6. Train models:
python -m src.train --dry-run
python -m src.train

7. Launch FastAPI server:
uvicorn src.app:app --reload

8. Deploy to Kubernetes:
helm install heart-api ./helm/heart-disease-chart
kubectl port-forward svc/heart-api-service 8000:8000

9. Setup monitoring stack:
kubectl port-forward svc/prometheus-server 9090:9090
kubectl port-forward svc/grafana 3000:80

Running Tests

black --check src/ tests/
flake8 src/ tests/
python -m pytest -v

Prediction Request Example
POST http://localhost:8000/predict
Sample Payload:
{
  "age": 57,
  "sex": 0,
  "cp": 2,
  "trestbps": 130,
  "chol": 236,
  "fbs": 0,
  "restecg": 0,
  "thalach": 174,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 1,
  "ca": 0,
  "thal": 3
}

### Experiment Tracking Summary ###

    Models tracked using MLflow
    Key metrics logged:
    Accuracy
    Precision
    Recall
    ROC AUC
    Refer to figures/metrics_comparison.csv for detailed results.

Deployment Notes
Prometheus Endpoint
Metrics are exposed at:
    http://localhost:8000/metrics
Prometheus annotations in Helm chart enable scraping automatically.

Grafana
    Import dashboard from helm/grafana-dashboard.json
    Default credentials: admin / admin
    Change password on first login
