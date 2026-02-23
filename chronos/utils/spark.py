from pyspark.sql import SparkSession
import os
from pathlib import Path
from chronos.utils.logger import log

def get_spark_session(app_name: str = "ChronosFeatureEngineering") -> SparkSession:
    """
    Creates or retrieves a SparkSession configured for Postgres JDBC.
    """
    project_root = Path(__file__).parent.parent.parent
    jar_path = project_root / "jars" / "postgresql-42.7.2.jar"

    if not jar_path.exists():
        log.error(f"Postgres JDBC Driver not found at {jar_path}")
        raise FileNotFoundError("Please download the Postgres JDBC driver into /jars folder")
    
    log.info(f"Starting Spark Session: {app_name}")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars", str(jar_path))
        .config("spark.driver.extraClassPath", str(jar_path))
        .config("spark.sql.shuffle.partitions", "4")
        .master("local[*]")
        .getOrCreate()
    )

    return spark