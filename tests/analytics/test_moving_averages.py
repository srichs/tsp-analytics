from datetime import date

import pytest
from pandas.testing import assert_frame_equal

from tsp import TspIndividualFund, TspAnalytics


def test_get_moving_average_matches_rolling_mean(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    expected = (
        tsp_price.get_price_history(fund=fund)
        .set_index("Date")
        .rolling(window=2, min_periods=1)
        .mean()
        .reset_index()
    )
    moving = tsp_price.get_moving_average(fund=fund, window=2)
    assert_frame_equal(moving, expected)


def test_get_moving_average_respects_date_range(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    moving = tsp_price.get_moving_average(
        fund=fund,
        window=2,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )
    assert moving["Date"].min().date() == date(2024, 1, 2)
    assert moving["Date"].max().date() == date(2024, 1, 3)


def test_get_moving_average_long_returns_tidy_format(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    moving_long = tsp_price.get_moving_average_long(fund=fund, window=2)
    assert moving_long["fund"].unique().tolist() == [fund.value]
    assert "moving_average" in moving_long.columns


def test_get_moving_average_validates_window(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="window must be a positive integer"):
        tsp_price.get_moving_average(window=0)
