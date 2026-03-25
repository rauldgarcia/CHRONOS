from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from chronos.utils.spark import get_spark_session
from chronos.utils.logger import log
import os

from chronos.utils.db import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_SERVER, POSTGRES_PORT, POSTGRES_DB

JDBC_URL = f"jdbc:postgresql://{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

def calculate_features(raw_df: DataFrame) -> DataFrame:
    """
    Pure function: Transform raw stock data into feature-rich data.
    Separated from I/O to allow unit testing.
    """
    window_spec = Window.partitionBy("ticker").orderBy("date")

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

    clean_df = features_df.na.drop()
    count = clean_df.count()
    log.info(f"Features calculated successfully. Total rows after dropping NAs: {count}")
    
    return clean_df

def run_feature_pipeline():
    """Main execution function handling I/O and orchestration."""
    spark = get_spark_session()

    log.info("Reading raw data from Postgres...")

    raw_df = (
        spark.read.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "stock_data")
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    log.info("Calculating technical indicators...")
    features_df = calculate_features(raw_df)

    log.info("Writing features to stock_features table...")

    mode = "overwrite" # or "append" if it is incremental

    (
        features_df.write
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "stock_features")
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .mode(mode)
        .save()
    )

    log.success("Feature Engineering pipeline completed successfully.")
    spark.stop()

if __name__ == "__main__":
    run_feature_pipeline()