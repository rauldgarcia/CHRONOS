import pytest
from pyspark.sql import SparkSession
import os

os.environ["POSTGRES_USER"] = "dummy"
os.environ["POSTGRES_PASSWORD"] = "dummy"
os.environ["POSTGRES_SERVER"] = "localhost"
os.environ["POSTGRES_DB"] = "dummy"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

@pytest.fixture(scope="session")
def spark():
    """
    Creates a local PySpark session for testing.
    Runs once per test session.
    """
    spark_session = (
        SparkSession.builder
        .appName("pytest-pyspark-local")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()