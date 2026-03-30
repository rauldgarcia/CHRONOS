import os
import pandas as pd
from datetime import datetime, timedelta
import mlflow

from chronos.utils.db import engine
from chronos.utils.logger import log

if os.environ.get("AIRFLOW_UID"):
    DEFAULT_TRACKING_URI = "http://mlflow:5000"
else:
    DEFAULT_TRACKING_URI = "http://localhost:5000"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def detect_drift(ticker: str):
    """
    Compare the last 7 days (Current) against the previous 30 days (Reference) to detect Data Drift using Evidently AI.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    log.info(f"Starting Drift Detection for {ticker}...")

    query = f"""
        SELECT * 
        FROM stock_features 
        WHERE ticker = '{ticker}' 
        ORDER BY date DESC LIMIT 40
    """

    raw_conn = engine.raw_connection()
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(query, raw_conn)
    finally:
        raw_conn.close()

    if len(df) < 37:
        log.warning(f"Not enough data to compute drift for {ticker}. Skipping.")
        return

    df = df.sort_values(by="date").reset_index(drop=True)

    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "sma_20",
        "volatility_20",
    ]

    reference_data = df.iloc[:30][features]
    current_data = df.iloc[-7:][features]

    log.info("Generating Evidently Data Drift Report...")

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    report_filename = (
        f"/tmp/drift_report_{ticker}_{datetime.now().strftime('%Y%m%d')}.html"
    )
    data_drift_report.save_html(report_filename)

    mlflow.set_experiment("Chronos_Data_Drift_Monitoring")

    with mlflow.start_run(run_name=f"{ticker}_Drift_Check"):
        drift_result = data_drift_report.as_dict()
        dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]

        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("is_drift_detected", int(dataset_drift))
        mlflow.log_artifact(report_filename, "evidently_reports")

        if dataset_drift:
            log.warning(f"DATA DRIFT DETECTED for {ticker}!")
        else:
            log.success(f"No drift detected for {ticker}. Data is stable.")

    if os.path.exists(report_filename):
        os.remove(report_filename)


if __name__ == "__main__":
    detect_drift("AAPL")
