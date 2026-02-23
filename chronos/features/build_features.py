from pyspark.sql import functions as F
from pyspark.sql.window import Window
from chronos.utils.spark import get_spark_session
from chronos.utils.logger import log
import os

DB_USER = os.getenv("POSTGRES_USER", "chronos_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "chronos_password")
DB_HOST = os.getenv("POSTGRES_SERVER", "localhost")
DB_PORT = "5433"
DB_NAME = os.getenv("POSTGRES_DB", "chronos_db")
JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"

def run_feature_pipeline():
    spark = get_spark_session()

    log.info("Reading raw data from Postgres...")

    raw_df = (
        spark.read.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "stock_data")
        .option("user", DB_USER)
        .option("password", DB_PASS)
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    window_spec = Window.partitionBy("ticker").orderBy("date")

    log.info("Calculating technical indicators...")

    features_df = raw_df.withColumn(
        "daily_return",
        (F.col("close") - F.lag("close", 1).over(window_spec)) / F.lag("close", 1).over(window_spec)
    ).withColumn(
        "sma_20",
        F.avg("close").over(window_spec.rowsBetween(-19, 0))
    ).withColumn(
        "volatility_20",
        F.stddev("close").over(window_spec.rowsBetween(-19, 0))
    )

    features_df = features_df.na.drop()

    log.info("Writing features to stock_features table...")

    mode = "overwrite" # or "append" if it is incremental

    (
        features_df.write
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "stock_features")
        .option("user", DB_USER)
        .option("password", DB_PASS)
        .option("driver", "org.postgresql.Driver")
        .mode(mode)
        .save()
    )

    log.success("Feature Engineering pipeline completed successfully.")
    spark.stop()

if __name__ == "__main__":
    run_feature_pipeline()