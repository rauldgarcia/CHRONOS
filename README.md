# 🏦 CHRONOS: MLOps Forecasting Platform

**Production-Grade Time-Series Forecasting with Automated Retraining**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow)](https://mlflow.org/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-017CEE?style=flat&logo=apache-airflow)](https://airflow.apache.org/)

---

## 📖 Executive Summary

**CHRONOS** is an end-to-end MLOps platform for financial forecasting. It demonstrates production-grade machine learning engineering through automated pipelines, ensemble modeling, drift detection, and seamless integration with the **TITAN** agentic platform.

**Key Features:**

- 🔄 **Automated Retraining**: Self-healing models via drift detection
- 🎯 **Ensemble Voting**: Ridge + XGBoost + LSTM with champion selection
- ⚡ **Big Data Processing**: PySpark for distributed feature engineering
- 📊 **MLOps Stack**: Airflow orchestration + MLflow tracking + Evidently monitoring
- 🚀 **Production API**: FastAPI serving with <100ms latency
- 🤖 **TITAN Integration**: Powers forecast capabilities for autonomous financial agents

---

## 🏗️ System Architecture

\[TITAN Agents\]
↓ HTTP Request
\[CHRONOS API\]
↓
\[Champion Model\] ← MLflow Registry
↓
\[Ensemble: Ridge + XGBoost + LSTM\]

---

## 🛠️ Tech Stack

Component

Technology

**Orchestration**

Apache Airflow 2.8

**Model Tracking**

MLflow 2.10

**Big Data**

PySpark 3.5

**Drift Detection**

Evidently AI 0.4

**API Framework**

FastAPI + Uvicorn

**Database**

PostgreSQL 16

**Deployment**

Docker + Cloud Run

**CI/CD**

GitHub Actions + Poetry

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ with Poetry
- Docker & Docker Compose
- Ubuntu 24.04

### Installation

    # Clone repository
    git clone https://github.com/rauldgarcia/chronos.git
    cd chronos

    # Install dependencies
    poetry install

    # Download historical data
    poetry run python chronos/data/ingestion.py

    # Start API server
    poetry run uvicorn chronos.api.main:app --reload

### Test the API

    # Health check
    curl http://localhost:8000/health

    # Get stock data
    curl http://localhost:8000/data/AAPL

---

## 📊 Project Roadmap

### ✅ Phase 1: Foundation (Week 1-2)

- \[x\] **Day 1**: Project initialization with Poetry + Git setup
- \[x\] **Day 2**: Yahoo Finance data ingestion (5 years historical data)
- \[x\] **Day 3**: Basic FastAPI endpoints (`/health`, `/data/{ticker}`)
- \[x\] **Day 4**: Fix NaN handling in API responses
- \[ \] **Day 5**: Unit tests for data ingestion
- \[ \] **Day 6**: Docker Compose setup (Postgres + MLflow + Airflow)
- \[ \] **Day 7**: Database integration (SQLAlchemy + Alembic migrations)

### 🔄 Phase 2: Feature Engineering (Week 3-4)

- \[ \] PySpark configuration for local development
- \[ \] Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- \[ \] Lag features and time-series transformations
- \[ \] Feature store creation in PostgreSQL
- \[ \] Airflow DAG: Feature engineering pipeline
- \[ \] Data validation with Great Expectations

### 🤖 Phase 3: Model Training (Week 5-6)

- \[ \] Ridge Regression baseline model
- \[ \] XGBoost gradient boosting model
- \[ \] LSTM deep learning model (TensorFlow + CUDA)
- \[ \] Voting Ensemble implementation
- \[ \] MLflow experiment tracking and model registry
- \[ \] Champion model selection logic
- \[ \] Airflow DAG: Training orchestration

### 📈 Phase 4: Monitoring & Deployment (Week 7-8)

- \[ \] Evidently AI drift detection
- \[ \] Automated retraining triggers
- \[ \] FastAPI `/forecast/{ticker}` endpoint
- \[ \] TITAN integration (Forecast Agent)
- \[ \] Docker containerization
- \[ \] Cloud Run deployment
- \[ \] CI/CD pipeline (GitHub Actions)

---

## 📂 Project Structure

chronos/
├── chronos/
│ ├── api/ # FastAPI application
│ ├── data/ # Data ingestion and validation
│ ├── features/ # Feature engineering (PySpark)
│ ├── models/ # ML models (Ridge, XGBoost, LSTM, Ensemble)
│ └── utils/ # Database, MLflow, Spark utilities
├── airflow/
│ └── dags/ # Airflow pipelines
├── tests/ # Pytest unit and integration tests
├── docker/ # Dockerfiles
└── data/ # Local data storage (git ignored)

---

## 🧪 Testing

    # Run all tests
    poetry run pytest -v

    # Run with coverage
    poetry run pytest --cov=chronos

---

## 🔧 Development

    # Format code
    poetry run ruff format .

    # Lint code
    poetry run ruff check .

    # Type check
    poetry run mypy chronos

---

## 📝 License

Private Portfolio Project - Raúl Daniel García Ramón

---

**Built with ❤️ by [Raúl García](https://github.com/rauldgarcia)**
