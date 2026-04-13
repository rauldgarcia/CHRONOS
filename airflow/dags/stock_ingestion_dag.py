"""
CHRONOS MLOps Pipeline — Cloud Composer Standard DAG
====================================================
This DAG follows modern Airflow 2.x best practices:
  - TaskFlow API (@dag, @task)
  - Dynamic Task Mapping (.expand)
  - Task Groups for clean UI
  - Proper documentation and retries

In a real Enterprise GCP environment (Cloud Composer), tasks like
feature engineering and training would be swapped out for
DataprocSubmitJobOperator and VertexAICustomTrainingJobOperator.
For this portfolio project, we execute the Python logic directly
so it can be run "Show, Don't Pay" on a local Docker cluster.
"""

from datetime import timedelta
import pendulum
from airflow.decorators import dag, task, task_group

# Business logic imports
from chronos.data.ingestion import download_stock_data, save_to_postgres
from chronos.features.build_features import run_feature_pipeline
from chronos.utils.logger import log
from chronos.data.validation import run_data_validation
from chronos.models.train import train_models
from chronos.monitoring.drift import detect_drift

TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]

default_args = {
    "owner": "raul.garcia",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="chronos_mlops_pipeline",
    default_args=default_args,
    description="End-to-End MLOps Pipeline: Ingestion, Features, Train, Drift",
    schedule="0 18 * * 1-5",  # Monday-Friday at 18:00 UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "chronos", "composer", "portfolio"],
    doc_md=__doc__,
)
def chronos_pipeline():

    @task_group(group_id="ingestion_phase", tooltip="Download and save raw data")
    def ingestion_phase():
        @task(map_index_template="{{ my_custom_map_index }}")
        def ingest_ticker(ticker: str):
            """Downloads stock data from YFinance and saves to Postgres."""
            log.info(f"🚀 Starting ETL for {ticker}")
            from airflow.operators.python import get_current_context

            context = get_current_context()
            context["my_custom_map_index"] = f"Ingesting: {ticker}"

            try:
                df = download_stock_data(ticker)
                save_to_postgres(df)
                log.info(f"ETL finished for {ticker}")
                return ticker
            except Exception as e:
                log.error(f"ETL failed for {ticker}: {e}")
                raise e

        # Dynamic Task Mapping: Spawns one parallel task per ticker
        return ingest_ticker.expand(ticker=TICKERS)

    @task_group(group_id="feature_engineering_phase")
    def feature_phase(ingested_tickers):
        @task
        def validate_raw_data(tickers):
            """Runs Great Expectations on the raw data."""
            log.info("Starting Data Validation (Great Expectations)")
            run_data_validation()
            return tickers

        @task
        def calculate_features(tickers):
            """Distributed feature engineering with PySpark."""
            log.info("Starting Distributed Feature Engineering (PySpark)")
            run_feature_pipeline()
            return tickers

        # The data validation depends on ingestion finishing
        validated = validate_raw_data(ingested_tickers)
        # Features depend on validation
        return calculate_features(validated)

    @task_group(group_id="ml_training_phase")
    def training_phase(features_ready):
        @task
        def check_data_drift():
            """Runs Evidently AI drift detection."""
            log.info("Starting Data Drift Detection")
            detect_drift("AAPL")

        @task
        def train_champion_model():
            """Trains ML models and uploads Champion to GCS (Prod) or MLflow (Local)."""
            log.info("Starting Model Training Pipeline")
            train_models("AAPL")

        # Drift and Training can run in parallel once features are ready
        check_data_drift()
        train_champion_model()

    # Define the high-level DAG flow
    ingestion_results = ingestion_phase()
    features_results = feature_phase(ingestion_results)
    training_phase(features_results)


# Instantiate the DAG
dag_instance = chronos_pipeline()
