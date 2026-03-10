from datetime import datetime
from chronos.features.build_features import calculate_features
import math


def test_calculate_features_logic(spark):
    """
    Test that technical indicators are calculated correctly using a dummy DataFrame.
    """
    data = [
        ("AAPL", datetime(2024, 1, 1), 100.0),
        ("AAPL", datetime(2024, 1, 2), 110.0),
        ("AAPL", datetime(2024, 1, 3), 105.0),
    ]
    raw_df = spark.createDataFrame(data, ["ticker", "date", "close"])

    result_df = calculate_features(raw_df)
    results = result_df.orderBy("date").collect()

    # The first row is removed by na.drop() (the lag generates a Null)
    assert len(results) == 2

    row_2 = results[0]
    assert row_2.daily_return == 0.10
    assert row_2.sma_20 == 105.0

    row_3 = results[1]
    assert math.isclose(row_3.daily_return, -0.04545, abs_tol=0.0001)
    assert row_3.sma_20 == 105.0
