from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum

# Importamos la lógica de negocio directamente de tu paquete
from chronos.data.ingestion import download_stock_data, save_to_postgres
from chronos.features.build_features import run_feature_pipeline
from chronos.utils.logger import log
from chronos.data.validation import run_data_validation

# Configuración del DAG
default_args = {
    "owner": "raul.garcia",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def etl_process(tickers: list):
    """Wrapper function for the ETL process"""
    for ticker in tickers:
        log.info(f"🚀 Starting ETL for {ticker}")
        try:
            df = download_stock_data(ticker)
            save_to_postgres(df)
            log.info(f"✅ ETL finished for {ticker}")
        except Exception as e:
            log.error(f"❌ ETL failed for {ticker}: {e}")
            raise e


def feature_engineering_process():
    """Wrapper function for the PySpark Feature Engineering process"""
    log.info("Starting Distributed Feature Engineering (PySpark)")
    try:
        run_feature_pipeline()
        log.info("Feature Engineering Completed")
    except Exception as e:
        log.error(f"Feature Engineering failed: {e}")


def validation_process():
    """Wrapper function for Data Quality Validation"""
    log.info("Starting Data Validation (Great Expectation)")
    run_data_validation()


with DAG(
    "chronos_stock_ingestion",
    default_args=default_args,
    description="End-to-End MLOps Pipeline: Ingestion & Feature Engineering",
    schedule_interval="0 18 * * 1-5",  # Lunes a Viernes a las 6:00 PM UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ingestion", "features", "stocks", "chronos"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_market_data",
        python_callable=etl_process,
        op_kwargs={"tickers": ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]},
    )

    validate_task = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validation_process,
    )

    feature_task = PythonOperator(
        task_id="calculate_features_pyspark",
        python_callable=feature_engineering_process,
    )

    ingest_task >> validate_task >> feature_task
