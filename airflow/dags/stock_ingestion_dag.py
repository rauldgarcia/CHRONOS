from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum

# Importamos la lógica de negocio directamente de tu paquete
from chronos.data.ingestion import download_stock_data, save_to_postgres
from chronos.utils.logger import log

# Configuración del DAG
default_args = {
    'owner': 'raul.garcia',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
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

with DAG(
    'chronos_stock_ingestion',
    default_args=default_args,
    description='Daily ingestion of stock data from Yahoo Finance',
    schedule_interval='0 18 * * 1-5',  # Lunes a Viernes a las 6:00 PM UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=['ingestion', 'stocks', 'chronos'],
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_market_data',
        python_callable=etl_process,
        op_kwargs={'tickers': ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]},
    )

    ingest_task