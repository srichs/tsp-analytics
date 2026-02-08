from datetime import date
from decimal import Decimal
import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics
from tests.helpers import (
    build_monthly_price_dataframe,
    build_yearly_price_dataframe,
)


def test_correlation_matrix_dict_outputs_metadata(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_correlation_matrix_dict()
    assert payload["start_date"] == "2024-01-02"
    assert payload["end_date"] == "2024-01-04"
    assert TspIndividualFund.G_FUND.value in payload["correlations"]
    g_correlations = payload["correlations"][TspIndividualFund.G_FUND.value]
    assert g_correlations[TspIndividualFund.G_FUND.value] == pytest.approx(1.0)

    as_date = tsp_price.get_correlation_matrix_dict(date_format=None)
    assert as_date["start_date"] == date(2024, 1, 2)
    assert as_date["end_date"] == date(2024, 1, 4)


def test_rolling_correlation_matrix_dict_outputs_window(
    tsp_price: TspAnalytics,
) -> None:
    payload = tsp_price.get_rolling_correlation_matrix_dict(window=2)
    assert payload["window"] == 2
    assert payload["start_date"] == "2024-01-03"
    assert payload["end_date"] == "2024-01-04"
    assert TspLifecycleFund.L_2030.value in payload["correlations"]


def test_price_history_with_metrics_shapes(tsp_price: TspAnalytics) -> None:
    metrics = tsp_price.get_price_history_with_metrics(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert metrics.index.name == "Date"
    assert metrics.columns.nlevels == 2
    assert (TspIndividualFund.G_FUND.value, "price") in metrics.columns
    assert (TspIndividualFund.C_FUND.value, "normalized_price") in metrics.columns

    metrics_long = tsp_price.get_price_history_with_metrics_long(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert set(metrics_long.columns) == {
        "Date",
        "fund",
        "price",
        "return",
        "cumulative_return",
        "normalized_price",
    }


def test_return_distribution_summary_and_dict(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_return_distribution_summary(
        fund=TspIndividualFund.G_FUND,
        percentiles=[0.25, 0.5, 0.75],
    )
    assert summary.index.tolist() == [TspIndividualFund.G_FUND.value]
    assert "win_rate" in summary.columns
    assert summary.loc[TspIndividualFund.G_FUND.value, "win_rate"] == pytest.approx(1.0)

    payload = tsp_price.get_return_distribution_summary_dict(
        fund=TspIndividualFund.G_FUND,
        percentiles=[0.25, 0.5, 0.75],
    )
    assert payload["funds"][TspIndividualFund.G_FUND.value][
        "win_rate"
    ] == pytest.approx(1.0)


def test_return_distribution_summary_validates_percentiles(
    tsp_price: TspAnalytics,
) -> None:
    with pytest.raises(ValueError, match="percentiles must be an iterable"):
        tsp_price.get_return_distribution_summary(percentiles="0.5")
    with pytest.raises(ValueError, match="percentiles must contain at least one value"):
        tsp_price.get_return_distribution_summary(percentiles=[])
    with pytest.raises(ValueError, match="percentiles must be between 0 and 1"):
        tsp_price.get_return_distribution_summary(percentiles=[-0.1, 0.5])


def test_fund_rankings_trailing_return(tsp_price: TspAnalytics) -> None:
    rankings = tsp_price.get_fund_rankings(metric="trailing_return", period=1, top_n=3)
    assert rankings.index.name == "fund"
    assert rankings["metric"].unique().tolist() == ["trailing_return"]
    assert rankings["rank"].min() == 1
    assert TspIndividualFund.G_FUND.value == rankings.index[0]

    payload = tsp_price.get_fund_rankings_dict(
        metric="trailing_return", period=1, top_n=2
    )
    assert payload["metric"] == "trailing_return"
    assert len(payload["rankings"]) == 2

    with pytest.raises(ValueError, match="unsupported ranking metric"):
        tsp_price.get_fund_rankings(metric="not-a-metric")

    with pytest.raises(
        ValueError, match="period is required when ranking by trailing_return"
    ):
        tsp_price.get_fund_rankings(metric="trailing_return")

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_latest_price_report_per_fund_dict(date_format="")


def test_fund_rankings(tsp_price: TspAnalytics) -> None:
    rankings = tsp_price.get_fund_rankings(metric="total_return")
    assert rankings.index.name == "fund"
    assert rankings.iloc[0].name == TspIndividualFund.G_FUND.value
    assert rankings["rank"].min() == 1

    change_percent = tsp_price.get_fund_rankings(metric="change_percent")
    assert change_percent["metric"].unique().tolist() == ["change_percent"]
    assert change_percent.index[0] == TspIndividualFund.G_FUND.value

    days_since = tsp_price.get_fund_rankings(
        metric="days_since",
        reference_date=date(2024, 1, 10),
    )
    assert days_since["metric"].unique().tolist() == ["days_since"]
    assert days_since["value"].nunique() == 1

    trailing = tsp_price.get_fund_rankings(metric="trailing_return", period=1, top_n=3)
    assert len(trailing) == 3
    assert trailing["metric"].unique().tolist() == ["trailing_return"]

    volatility = tsp_price.get_fund_rankings(metric="annualized_volatility")
    assert volatility["value"].tolist() == sorted(volatility["value"].tolist())

    with pytest.raises(
        ValueError, match="period is required when ranking by trailing_return"
    ):
        tsp_price.get_fund_rankings(metric="trailing_return")

    with pytest.raises(ValueError, match="unsupported ranking metric"):
        tsp_price.get_fund_rankings(metric="momentum")

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_fund_rankings(metric="total_return", start_date=date(2024, 1, 1))


def test_fund_rankings_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_fund_rankings_dict(metric="total_return")
    assert payload["metric"] == "total_return"
    assert payload["rankings"][0]["fund"] == TspIndividualFund.G_FUND.value
    assert payload["rankings"][0]["rank"] == 1

    change_payload = tsp_price.get_fund_rankings_dict(metric="change_percent")
    assert change_payload["metric"] == "change_percent"

    trailing = tsp_price.get_fund_rankings_dict(
        metric="trailing_return", period=1, top_n=2
    )
    assert trailing["metric"] == "trailing_return"
    assert trailing["period"] == 1
    assert len(trailing["rankings"]) == 2

    ranged = tsp_price.get_fund_rankings_dict(
        metric="annualized_volatility",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
        top_n=3,
        date_format=None,
    )
    assert ranged["start_date"] == date(2024, 1, 1)
    assert ranged["end_date"] == date(2024, 1, 4)


def test_returns_and_cumulative(tsp_price: TspAnalytics) -> None:
    daily_returns = tsp_price.get_daily_returns()
    expected_returns = (
        tsp_price.dataframe.set_index("Date")
        .dropna(how="all")
        .pct_change()
        .dropna(how="all")
    )
    assert_frame_equal(daily_returns, expected_returns)

    fund = TspIndividualFund.C_FUND
    fund_returns = tsp_price.get_daily_returns(fund=fund)
    expected_fund_returns = (
        tsp_price.dataframe[["Date", fund.value]]
        .set_index("Date")
        .pct_change()
        .dropna(how="all")
    )
    assert_frame_equal(fund_returns, expected_fund_returns)

    cumulative_returns = tsp_price.get_cumulative_returns(fund=fund)
    expected_cumulative = (1 + expected_fund_returns).cumprod() - 1
    assert_frame_equal(cumulative_returns, expected_cumulative)


def test_cumulative_returns_by_date_range(tsp_price: TspAnalytics) -> None:
    start_date = date(2024, 1, 2)
    end_date = date(2024, 1, 4)
    cumulative_returns = tsp_price.get_cumulative_returns_by_date_range(
        start_date, end_date
    )
    expected_returns = tsp_price.get_daily_returns_by_date_range(start_date, end_date)
    expected_cumulative = (1 + expected_returns).cumprod() - 1
    assert_frame_equal(cumulative_returns, expected_cumulative)


def test_normalized_prices_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_normalized_prices(start_date=date(2024, 1, 1))

    with pytest.raises(ValueError, match="base_value must be a positive value"):
        tsp_price.get_normalized_prices(base_value=0)

    with pytest.raises(ValueError, match="base_value must be a positive value"):
        tsp_price.get_normalized_prices(base_value=True)


def test_normalized_prices_long(tsp_price: TspAnalytics) -> None:
    normalized_long = tsp_price.get_normalized_prices_long()
    assert normalized_long.columns.tolist() == ["Date", "fund", "normalized_price"]
    assert normalized_long["fund"].nunique() == len(tsp_price.get_available_funds())

    fund = TspIndividualFund.G_FUND
    normalized_fund = tsp_price.get_normalized_prices_long(fund=fund)
    assert normalized_fund["fund"].unique().tolist() == [fund.value]


def test_trailing_returns_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(
        ValueError, match="periods must contain at least one positive integer"
    ):
        tsp_price.get_trailing_returns(periods=[])

    with pytest.raises(
        ValueError,
        match="periods must be a positive integer or an iterable of positive integers",
    ):
        tsp_price.get_trailing_returns(periods="5")

    with pytest.raises(ValueError, match="periods must be a positive integer"):
        tsp_price.get_trailing_returns(periods=[0])

    with pytest.raises(ValueError, match="periods must be a positive integer"):
        tsp_price.get_trailing_returns(periods=["3"])


def test_daily_returns_by_date_range(tsp_price: TspAnalytics) -> None:
    start_date = date(2024, 1, 2)
    end_date = date(2024, 1, 4)
    returns = tsp_price.get_daily_returns_by_date_range(start_date, end_date)
    expected_prices = tsp_price.dataframe.set_index("Date").dropna(how="all")
    expected_prices = expected_prices.loc[start_date:end_date]
    expected_returns = expected_prices.pct_change().dropna(how="all")
    assert_frame_equal(returns, expected_returns)


def test_trailing_returns_accepts_single_period_int(tsp_price: TspAnalytics) -> None:
    trailing = tsp_price.get_trailing_returns(periods=1)
    assert trailing.index.tolist() == [1]
    assert TspIndividualFund.G_FUND.value in trailing.columns


def test_daily_returns_long(tsp_price: TspAnalytics) -> None:
    returns_long = tsp_price.get_daily_returns_long()
    assert returns_long.columns.tolist() == ["Date", "fund", "return"]
    assert returns_long["fund"].nunique() == len(tsp_price.get_available_funds())

    fund = TspIndividualFund.G_FUND
    fund_returns_long = tsp_price.get_daily_returns_long(
        fund=fund,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )
    assert fund_returns_long["fund"].unique().tolist() == [fund.value]
    assert fund_returns_long["Date"].min().date() == date(2024, 1, 2)

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_daily_returns_long(start_date=date(2024, 1, 2))


def test_daily_returns_do_not_forward_fill_missing_prices() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            TspIndividualFund.G_FUND.value: [100.0, np.nan, 110.0],
            TspIndividualFund.C_FUND.value: [200.0, 210.0, 220.0],
        }
    )
    tsp_price.load_dataframe(dataframe)

    returns = tsp_price.get_daily_returns()
    g_return = returns.loc[pd.Timestamp("2024-01-04"), TspIndividualFund.G_FUND.value]
    assert pd.isna(g_return)


def test_excess_returns(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    benchmark = TspIndividualFund.G_FUND
    returns = tsp_price.get_daily_returns()
    expected = (
        returns[fund.value].sub(returns[benchmark.value]).to_frame(name="excess_return")
    )
    excess = tsp_price.get_excess_returns(fund=fund, benchmark=benchmark)
    assert_frame_equal(excess, expected)

    expected_all = returns.drop(columns=[benchmark.value]).sub(
        returns[benchmark.value], axis=0
    )
    excess_all = tsp_price.get_excess_returns(benchmark=benchmark)
    assert_frame_equal(excess_all, expected_all)


def test_excess_returns_long(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    benchmark = TspIndividualFund.G_FUND
    excess_long = tsp_price.get_excess_returns_long(fund=fund, benchmark=benchmark)
    assert excess_long.columns.tolist() == ["Date", "fund", "excess_return"]
    assert excess_long["fund"].unique().tolist() == [fund.value]


def test_price_and_return_statistics_single_fund(tsp_price: TspAnalytics) -> None:
    price_stats = tsp_price.get_price_statistics(fund=TspIndividualFund.G_FUND)
    assert "median" in price_stats.columns
    assert price_stats.index.tolist() == [TspIndividualFund.G_FUND.value]

    return_stats = tsp_price.get_return_statistics(
        fund=TspIndividualFund.G_FUND,
        trading_days=252,
    )
    assert "annualized_return" in return_stats.columns
    assert "annualized_volatility" in return_stats.columns


def test_correlation_matrix_long(tsp_price: TspAnalytics) -> None:
    correlation_long = tsp_price.get_correlation_matrix_long()
    assert correlation_long.columns.tolist() == ["fund_a", "fund_b", "correlation"]
    available = tsp_price.get_available_funds()
    assert set(correlation_long["fund_a"].unique()) == set(available)
    assert set(correlation_long["fund_b"].unique()) == set(available)


def test_excess_returns_requires_other_funds() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="no funds available for excess returns"):
        tsp_price.get_excess_returns(benchmark=TspIndividualFund.G_FUND)


def test_cumulative_returns_long(tsp_price: TspAnalytics) -> None:
    cumulative_long = tsp_price.get_cumulative_returns_long()
    assert cumulative_long.columns.tolist() == ["Date", "fund", "cumulative_return"]
    assert cumulative_long["fund"].nunique() == len(tsp_price.get_available_funds())

    filtered = tsp_price.get_cumulative_returns_long(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )
    assert filtered["Date"].min().date() == date(2024, 1, 2)

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_cumulative_returns_long(start_date=date(2024, 1, 2))


def test_trailing_returns_calculation(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    trailing_returns = tsp_price.get_trailing_returns(periods=[1, 1, 2], fund=fund)
    price_df = tsp_price.dataframe.set_index("Date")[[fund.value]].dropna(how="all")
    expected = {
        1: price_df.pct_change(periods=1).iloc[-1][fund.value],
        2: price_df.pct_change(periods=2).iloc[-1][fund.value],
    }
    assert trailing_returns.index.tolist() == [1, 2]
    assert trailing_returns.loc[1, fund.value] == pytest.approx(expected[1])
    assert trailing_returns.loc[2, fund.value] == pytest.approx(expected[2])


def test_price_history_with_metrics_long(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    metrics = tsp_price.get_price_history_with_metrics_long(
        funds=[fund],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        base_value=100.0,
    )
    assert metrics.columns.tolist() == [
        "Date",
        "fund",
        "price",
        "return",
        "cumulative_return",
        "normalized_price",
    ]
    fund_metrics = (
        metrics[metrics["fund"] == fund.value]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    assert fund_metrics.loc[0, "normalized_price"] == pytest.approx(100.0)
    assert fund_metrics.loc[1, "return"] == pytest.approx(0.1)
    assert fund_metrics.loc[1, "cumulative_return"] == pytest.approx(0.1)
    assert fund_metrics.loc[1, "normalized_price"] == pytest.approx(110.0)

    with pytest.raises(ValueError, match="base_value must be a positive value"):
        tsp_price.get_price_history_with_metrics_long(base_value=0)


def test_price_history_with_metrics_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_price_history_with_metrics_dict(
        funds=[TspIndividualFund.G_FUND],
        base_value=100.0,
        date_format="%Y-%m-%d",
    )
    records = payload["metrics"]
    assert records[0]["Date"] == "2024-01-01"
    assert set(records[0].keys()) == {
        "Date",
        "fund",
        "price",
        "return",
        "cumulative_return",
        "normalized_price",
    }

    matching = next(record for record in records if record["Date"] == "2024-01-02")
    assert matching["normalized_price"] == pytest.approx(110.0)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_price_history_with_metrics_dict(date_format="")


def test_price_history_with_metrics_long_handles_missing_initial_values(
    tsp_price: TspAnalytics,
) -> None:
    fund = TspIndividualFund.G_FUND
    dataframe = tsp_price.dataframe.copy()
    dataframe.loc[dataframe.index[0], fund.value] = float("nan")
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    metrics = tsp_price.get_price_history_with_metrics_long(
        funds=[fund],
        base_value=100.0,
    )
    fund_metrics = (
        metrics[metrics["fund"] == fund.value]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    first_normalized = fund_metrics["normalized_price"].dropna().iloc[0]
    assert first_normalized == pytest.approx(100.0)


def test_price_history_with_metrics(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    metrics = tsp_price.get_price_history_with_metrics(
        funds=[fund],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
        base_value=100.0,
    )

    assert isinstance(metrics.columns, pd.MultiIndex)
    assert metrics.columns.tolist() == [
        (fund.value, "price"),
        (fund.value, "return"),
        (fund.value, "cumulative_return"),
        (fund.value, "normalized_price"),
    ]
    assert metrics[(fund.value, "normalized_price")].iloc[0] == pytest.approx(100.0)
    assert metrics[(fund.value, "return")].iloc[1] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="base_value must be a positive value"):
        tsp_price.get_price_history_with_metrics(base_value=0)


def test_price_history_with_metrics_handles_missing_initial_values(
    tsp_price: TspAnalytics,
) -> None:
    fund = TspIndividualFund.G_FUND
    dataframe = tsp_price.dataframe.copy()
    dataframe.loc[dataframe.index[0], fund.value] = float("nan")
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    metrics = tsp_price.get_price_history_with_metrics(
        funds=[fund],
        base_value=100.0,
    )
    normalized = metrics[(fund.value, "normalized_price")].dropna()
    assert normalized.iloc[0] == pytest.approx(100.0)


def test_get_fund_snapshot(tsp_price: TspAnalytics) -> None:
    snapshot = tsp_price.get_fund_snapshot(periods=[1, 2], trading_days=252)
    assert snapshot["as_of"].nunique() == 1
    assert snapshot["as_of"].iloc[0] == tsp_price.latest
    assert "latest_price" in snapshot.columns
    assert "change_percent" in snapshot.columns
    assert "trailing_return_1d" in snapshot.columns
    assert "trailing_return_2d" in snapshot.columns
    assert "sortino_ratio" in snapshot.columns
    assert "value_at_risk" in snapshot.columns
    assert "expected_shortfall" in snapshot.columns

    fund = TspIndividualFund.G_FUND
    fund_snapshot = tsp_price.get_fund_snapshot(
        fund=fund, periods=[1], trading_days=252
    )
    price_df = tsp_price.dataframe.set_index("Date")[[fund.value]]
    expected_latest = price_df.iloc[-1][fund.value]
    expected_previous = price_df.iloc[-2][fund.value]
    expected_change = expected_latest - expected_previous
    expected_change_percent = expected_change / expected_previous
    assert fund_snapshot.loc[fund.value, "as_of"] == tsp_price.latest
    assert fund_snapshot.loc[fund.value, "latest_price"] == pytest.approx(
        expected_latest
    )
    assert fund_snapshot.loc[fund.value, "previous_price"] == pytest.approx(
        expected_previous
    )
    assert fund_snapshot.loc[fund.value, "change"] == pytest.approx(expected_change)
    assert fund_snapshot.loc[fund.value, "change_percent"] == pytest.approx(
        expected_change_percent
    )


def test_fund_snapshot_supports_fund_lists(tsp_price: TspAnalytics) -> None:
    funds = [TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    snapshot = tsp_price.get_fund_snapshot(funds=funds, periods=[1, 2])
    assert snapshot.index.tolist() == [fund.value for fund in funds]
    assert "as_of" in snapshot.columns
    assert "latest_price" in snapshot.columns
    assert "trailing_return_1d" in snapshot.columns

    snapshot_long = tsp_price.get_fund_snapshot_long(funds=funds, periods=[1, 2])
    assert snapshot_long["fund"].unique().tolist() == [fund.value for fund in funds]
    assert set(snapshot_long["metric"]).issuperset(
        {"latest_price", "trailing_return_1d"}
    )

    snapshot_dict = tsp_price.get_fund_snapshot_dict(funds=funds, periods=[1, 2])
    assert set(snapshot_dict["funds"].keys()) == {fund.value for fund in funds}

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_fund_snapshot(
            fund=TspIndividualFund.G_FUND,
            funds=[TspLifecycleFund.L_2030],
        )


def test_get_fund_snapshot_risk_metrics(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    trading_days = 252
    mar = 0.02
    confidence = 0.8
    snapshot = tsp_price.get_fund_snapshot(
        fund=fund,
        periods=[1],
        trading_days=trading_days,
        mar=mar,
        confidence=confidence,
    )
    returns = tsp_price.get_daily_returns(fund=fund)[fund.value]
    annualized_return = returns.mean() * trading_days
    daily_mar = (1 + mar) ** (1 / trading_days) - 1
    downside = (returns - daily_mar).where(returns < daily_mar, 0)
    downside_deviation = downside.pow(2).mean() ** 0.5 * (trading_days**0.5)
    expected_sortino = (
        (annualized_return - mar) / downside_deviation
        if downside_deviation
        else float("nan")
    )
    value_at_risk = returns.quantile(1 - confidence)
    expected_shortfall = returns[returns <= value_at_risk].mean()

    if math.isnan(expected_sortino):
        assert math.isnan(snapshot.loc[fund.value, "sortino_ratio"])
    else:
        assert snapshot.loc[fund.value, "sortino_ratio"] == pytest.approx(
            expected_sortino
        )
    assert snapshot.loc[fund.value, "value_at_risk"] == pytest.approx(value_at_risk)
    assert snapshot.loc[fund.value, "expected_shortfall"] == pytest.approx(
        expected_shortfall
    )


def test_get_fund_snapshot_handles_missing_latest_row() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [99.0, 100.0, 101.0],
            TspIndividualFund.C_FUND.value: [199.0, 200.0, np.nan],
        }
    )
    tsp_price.load_dataframe(dataframe)

    snapshot = tsp_price.get_fund_snapshot(periods=[1], trading_days=252)
    assert snapshot.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 3)
    assert snapshot.loc[TspIndividualFund.C_FUND.value, "as_of"] == date(2024, 1, 2)
    assert snapshot.loc[
        TspIndividualFund.C_FUND.value, "latest_price"
    ] == pytest.approx(200.0)


def test_get_fund_snapshot_long(tsp_price: TspAnalytics) -> None:
    snapshot_long = tsp_price.get_fund_snapshot_long(periods=[1, 2], trading_days=252)
    assert snapshot_long.columns.tolist() == ["fund", "as_of", "metric", "value"]
    assert snapshot_long["fund"].nunique() == len(tsp_price.get_available_funds())
    assert snapshot_long["as_of"].nunique() == 1
    assert snapshot_long["as_of"].iloc[0] == tsp_price.latest
    assert "latest_price" in snapshot_long["metric"].unique()
    assert "trailing_return_1d" in snapshot_long["metric"].unique()

    fund = TspIndividualFund.G_FUND
    fund_snapshot_long = tsp_price.get_fund_snapshot_long(
        fund=fund, periods=[1], trading_days=252
    )
    assert fund_snapshot_long["fund"].unique().tolist() == [fund.value]


def test_get_fund_snapshot_dict(tsp_price: TspAnalytics) -> None:
    snapshot = tsp_price.get_fund_snapshot_dict(periods=[1, 2], trading_days=252)
    assert snapshot["as_of"] == "2024-01-04"
    assert TspIndividualFund.G_FUND.value in snapshot["funds"]
    assert snapshot["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-04"

    fund = TspIndividualFund.G_FUND
    fund_snapshot = tsp_price.get_fund_snapshot_dict(
        fund=fund, periods=[1], trading_days=252
    )
    assert fund_snapshot["funds"].keys() == {fund.value}
    latest_value = float(tsp_price.dataframe[fund.value].iloc[-1])
    assert fund_snapshot["funds"][fund.value]["latest_price"] == pytest.approx(
        latest_value
    )

    formatted = tsp_price.get_fund_snapshot_dict(date_format="%b %d, %Y")
    assert formatted["as_of"] == "Jan 04, 2024"

    as_date = tsp_price.get_fund_snapshot_dict(date_format=None)
    assert as_date["as_of"] == date(2024, 1, 4)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_fund_snapshot_dict(date_format="")


def test_get_fund_snapshot_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(
        ValueError, match="periods must contain at least one positive integer"
    ):
        tsp_price.get_fund_snapshot(periods=[])

    with pytest.raises(ValueError, match="periods must be a positive integer"):
        tsp_price.get_fund_snapshot(periods=["3"])

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_fund_snapshot(periods=[1], trading_days=0)

    with pytest.raises(ValueError, match="mar must be a numeric value"):
        tsp_price.get_fund_snapshot(periods=[1], mar="0.01")

    with pytest.raises(ValueError, match="confidence must be a float between 0 and 1"):
        tsp_price.get_fund_snapshot(periods=[1], confidence=1.5)

    single_row = tsp_price.dataframe.head(1)
    tsp_price.dataframe = single_row
    tsp_price.current = single_row.iloc[0]
    tsp_price.latest = tsp_price.current["Date"].date()

    with pytest.raises(ValueError, match="at least two data points"):
        tsp_price.get_fund_snapshot(periods=[1])


def test_rolling_metrics(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.S_FUND
    rolling_mean = tsp_price.get_rolling_mean(fund=fund, window=2)
    expected_mean = (
        tsp_price.dataframe[["Date", fund.value]]
        .set_index("Date")
        .rolling(window=2)
        .mean()
        .dropna(how="all")
    )
    assert_frame_equal(rolling_mean, expected_mean)

    rolling_returns = tsp_price.get_rolling_returns(fund=fund, window=2)
    expected_returns = (
        tsp_price.dataframe[["Date", fund.value]]
        .set_index("Date")
        .pct_change(periods=2)
        .dropna(how="all")
    )
    assert_frame_equal(rolling_returns, expected_returns)

    rolling_volatility = tsp_price.get_rolling_volatility(
        fund=fund, window=2, trading_days=252
    )
    expected_volatility = (
        tsp_price.get_daily_returns(fund=fund)
        .rolling(window=2)
        .std()
        .mul(252**0.5)
        .dropna(how="all")
    )
    assert_frame_equal(rolling_volatility, expected_volatility)

    rolling_max_drawdown = tsp_price.get_rolling_max_drawdown(fund=fund, window=2)
    prices = tsp_price.dataframe[["Date", fund.value]].set_index("Date")[fund.value]
    expected_drawdown = (
        prices.rolling(window=2)
        .apply(lambda values: (values / values.cummax() - 1).min(), raw=False)
        .dropna()
        .to_frame(name="rolling_max_drawdown")
    )
    assert_frame_equal(rolling_max_drawdown, expected_drawdown)


def test_rolling_performance_summary(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    window = 2
    trading_days = 252
    summary = tsp_price.get_rolling_performance_summary(
        fund=fund, window=window, trading_days=trading_days
    )
    returns = tsp_price.get_daily_returns(fund=fund)[fund.value]
    expected_return = returns.rolling(window=window).mean().mul(trading_days)
    expected_volatility = returns.rolling(window=window).std().mul(trading_days**0.5)
    expected_sharpe = expected_return.div(expected_volatility)
    expected = pd.DataFrame(
        {
            "rolling_return": expected_return,
            "rolling_volatility": expected_volatility,
            "rolling_sharpe_ratio": expected_sharpe,
        }
    ).dropna(how="all")
    assert_frame_equal(summary, expected)

    long_summary = tsp_price.get_rolling_performance_summary_long(
        fund=fund, window=window, trading_days=trading_days
    )
    assert long_summary.columns.tolist() == ["Date", "fund", "metric", "value"]
    assert long_summary["fund"].unique().tolist() == [fund.value]
    assert set(long_summary["metric"].unique()) == {
        "rolling_return",
        "rolling_volatility",
        "rolling_sharpe_ratio",
    }

    payload = tsp_price.get_rolling_performance_summary_dict(
        fund=fund,
        window=window,
        trading_days=trading_days,
        date_format=None,
    )
    assert payload["fund"] == fund.value
    assert payload["window"] == window
    assert payload["trading_days"] == trading_days
    assert isinstance(payload["metrics"][0]["Date"], date)

    with pytest.raises(ValueError, match="not enough return data available"):
        tsp_price.get_rolling_performance_summary(fund=fund, window=10)


def test_moving_averages(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    moving_averages = tsp_price.get_moving_averages(
        fund=fund,
        windows=[2, 3],
        method="simple",
    )
    prices = tsp_price.dataframe.set_index("Date")[fund.value]
    expected = pd.DataFrame(
        {
            "sma_2": prices.rolling(window=2, min_periods=2).mean(),
            "sma_3": prices.rolling(window=3, min_periods=3).mean(),
        }
    ).dropna(how="all")
    assert_frame_equal(moving_averages, expected)

    ema = tsp_price.get_moving_averages(
        fund=fund,
        windows=[2],
        method="exponential",
    )
    expected_ema = (
        prices.ewm(span=2, min_periods=2, adjust=False)
        .mean()
        .to_frame(name="ema_2")
        .dropna(how="all")
    )
    assert_frame_equal(ema, expected_ema)

    with pytest.raises(ValueError, match="method must be 'simple' or 'exponential'"):
        tsp_price.get_moving_averages(fund=fund, windows=[2], method="weighted")


def test_rolling_correlation(tsp_price: TspAnalytics) -> None:
    fund_a = TspIndividualFund.G_FUND
    fund_b = TspIndividualFund.F_FUND
    rolling_corr = tsp_price.get_rolling_correlation(fund_a, fund_b, window=2)
    returns = tsp_price.get_daily_returns()
    expected = (
        returns[fund_a.value]
        .rolling(window=2)
        .corr(returns[fund_b.value])
        .dropna()
        .to_frame(name="rolling_correlation")
    )
    assert_frame_equal(rolling_corr, expected)


def test_beta_and_rolling_beta(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    benchmark = TspIndividualFund.S_FUND
    returns = tsp_price.get_daily_returns()
    expected_beta = (
        returns[fund.value].cov(returns[benchmark.value])
        / returns[benchmark.value].var()
    )
    assert tsp_price.get_beta(fund, benchmark) == pytest.approx(expected_beta)

    rolling_beta = tsp_price.get_rolling_beta(fund, benchmark, window=2)
    expected_rolling = (
        returns[fund.value]
        .rolling(window=2)
        .cov(returns[benchmark.value])
        .div(returns[benchmark.value].rolling(window=2).var())
        .dropna()
        .to_frame(name="rolling_beta")
    )
    assert_frame_equal(rolling_beta, expected_rolling)


def test_tracking_error_and_information_ratio(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    benchmark = TspIndividualFund.G_FUND
    trading_days = 252
    excess = tsp_price.get_excess_returns(fund=fund, benchmark=benchmark)
    expected_te = excess["excess_return"].std() * (trading_days**0.5)
    tracking_error = tsp_price.get_tracking_error(
        fund=fund,
        benchmark=benchmark,
        trading_days=trading_days,
    )
    assert tracking_error.loc[fund.value, "tracking_error"] == pytest.approx(
        expected_te
    )

    expected_annualized_excess = excess["excess_return"].mean() * trading_days
    expected_info = expected_annualized_excess / expected_te
    info_ratio = tsp_price.get_information_ratio(
        fund=fund,
        benchmark=benchmark,
        trading_days=trading_days,
    )
    assert info_ratio.loc[fund.value, "annualized_excess_return"] == pytest.approx(
        expected_annualized_excess
    )
    assert info_ratio.loc[fund.value, "tracking_error"] == pytest.approx(expected_te)
    assert info_ratio.loc[fund.value, "information_ratio"] == pytest.approx(
        expected_info
    )

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_tracking_error(fund, benchmark, start_date=date(2024, 1, 2))

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_information_ratio(fund, benchmark, start_date=date(2024, 1, 2))


def test_rolling_tracking_error(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    benchmark = TspIndividualFund.G_FUND
    window = 2
    trading_days = 252
    excess = tsp_price.get_excess_returns(fund=fund, benchmark=benchmark)
    expected = (
        excess["excess_return"]
        .rolling(window=window)
        .std()
        .mul(trading_days**0.5)
        .dropna()
        .to_frame(name="rolling_tracking_error")
    )
    rolling = tsp_price.get_rolling_tracking_error(
        fund=fund,
        benchmark=benchmark,
        window=window,
        trading_days=trading_days,
    )
    assert_frame_equal(rolling, expected)


def test_return_histogram(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    histogram = tsp_price.get_return_histogram(fund=fund, bins=2)
    assert histogram.columns.tolist() == ["bin_left", "bin_right", "count"]
    total_counts = histogram["count"].sum()
    expected_count = len(tsp_price.get_daily_returns(fund=fund))
    assert total_counts == expected_count


def test_sortino_ratio(tsp_price: TspAnalytics) -> None:
    mar = 0.05
    trading_days = 252
    returns = tsp_price.get_daily_returns()
    daily_mar = (1 + mar) ** (1 / trading_days) - 1
    downside = (returns - daily_mar).where(returns < daily_mar, 0)
    downside_dev = downside.pow(2).mean().pow(0.5).mul(trading_days**0.5)
    annualized_return = returns.mean().mul(trading_days)
    expected_sortino = (
        (annualized_return - mar).div(downside_dev).rename("sortino_ratio")
    )

    sortino = tsp_price.get_sortino_ratio(mar=mar, trading_days=trading_days)
    assert_series_equal(sortino["sortino_ratio"], expected_sortino.rename_axis("fund"))

    fund = TspIndividualFund.G_FUND
    fund_sortino = tsp_price.get_sortino_ratio(fund=fund)
    assert fund_sortino.index.tolist() == [fund.value]

    with pytest.raises(ValueError, match="mar must be a numeric value"):
        tsp_price.get_sortino_ratio(mar="0.02")


def test_rolling_sortino_ratio(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.C_FUND
    mar = 0.02
    window = 2
    trading_days = 252
    returns = tsp_price.get_daily_returns(fund=fund)
    daily_mar = (1 + mar) ** (1 / trading_days) - 1
    downside = (returns - daily_mar).where(returns < daily_mar, 0)
    rolling_downside = (
        downside.pow(2).rolling(window=window).mean().pow(0.5).mul(trading_days**0.5)
    )
    rolling_return = returns.rolling(window=window).mean().mul(trading_days)
    expected = (rolling_return - mar).div(rolling_downside).dropna(how="all")

    sortino = tsp_price.get_rolling_sortino_ratio(
        fund=fund,
        window=window,
        trading_days=trading_days,
        mar=mar,
    )
    assert_frame_equal(sortino, expected)


def test_value_at_risk_and_expected_shortfall(tsp_price: TspAnalytics) -> None:
    confidence = 0.75
    returns = tsp_price.get_daily_returns()
    expected_var = returns.quantile(1 - confidence).rename_axis("fund")
    var = tsp_price.get_value_at_risk(confidence=confidence)
    assert_series_equal(var["value_at_risk"], expected_var.rename("value_at_risk"))

    expected_shortfall = {}
    for column in returns.columns:
        threshold = expected_var[column]
        expected_shortfall[column] = returns[column][
            returns[column] <= threshold
        ].mean()
    es = tsp_price.get_expected_shortfall(confidence=confidence)
    assert_series_equal(
        es["expected_shortfall"],
        pd.Series(expected_shortfall, name="expected_shortfall").rename_axis("fund"),
    )

    fund = TspIndividualFund.G_FUND
    fund_var = tsp_price.get_value_at_risk(confidence=confidence, fund=fund)
    assert fund_var.index.tolist() == [fund.value]

    with pytest.raises(ValueError, match="confidence must be a float between 0 and 1"):
        tsp_price.get_value_at_risk(confidence=1.5)

    with pytest.raises(ValueError, match="confidence must be a float between 0 and 1"):
        tsp_price.get_expected_shortfall(confidence=0)


def test_drawdown_summary(tsp_price: TspAnalytics) -> None:
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            TspIndividualFund.G_FUND.value: [100, 110, 90, 95, 120],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    summary = tsp_price.get_drawdown_summary(TspIndividualFund.G_FUND)
    assert summary.index.tolist() == [TspIndividualFund.G_FUND.value]
    assert summary.loc[TspIndividualFund.G_FUND.value, "peak_date"] == date(2024, 1, 2)
    assert summary.loc[TspIndividualFund.G_FUND.value, "trough_date"] == date(
        2024, 1, 3
    )
    assert summary.loc[TspIndividualFund.G_FUND.value, "recovery_date"] == date(
        2024, 1, 5
    )
    assert summary.loc[TspIndividualFund.G_FUND.value, "drawdown_duration_days"] == 1
    assert summary.loc[TspIndividualFund.G_FUND.value, "recovery_duration_days"] == 2


def test_yearly_returns(tsp_price: TspAnalytics) -> None:
    dataframe = build_yearly_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    yearly_returns = tsp_price.get_yearly_returns()
    expected_returns = (
        dataframe.set_index("Date")
        .dropna(how="all")
        .resample("YE")
        .last()
        .pct_change()
        .dropna(how="all")
    )
    assert_frame_equal(yearly_returns, expected_returns)


def test_monthly_returns() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_monthly_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    monthly_returns = tsp_price.get_monthly_returns()
    expected_returns = (
        dataframe.set_index("Date")
        .dropna(how="all")
        .resample("ME")
        .last()
        .pct_change()
        .dropna(how="all")
    )
    assert_frame_equal(monthly_returns, expected_returns)


def test_get_monthly_return_table() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_monthly_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    table = tsp_price.get_monthly_return_table(TspIndividualFund.G_FUND)
    assert table.index.tolist() == [2024]
    assert table.columns.tolist() == [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    assert table.loc[2024, "Feb"] == pytest.approx(0.1)
    assert table.loc[2024, "Mar"] == pytest.approx(0.1)

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_monthly_return_table(
            TspIndividualFund.G_FUND, start_date=date(2024, 1, 1)
        )


def test_price_and_return_statistics_all_funds(tsp_price: TspAnalytics) -> None:
    price_stats = tsp_price.get_price_statistics()
    expected_prices = tsp_price.dataframe.set_index("Date").dropna(how="all")
    expected_stats = expected_prices.describe().T
    expected_stats["median"] = expected_prices.median()
    assert_frame_equal(price_stats, expected_stats)

    return_stats = tsp_price.get_return_statistics()
    returns = tsp_price.get_daily_returns()
    expected_returns = returns.describe().T
    expected_returns["median"] = returns.median()
    expected_returns["skew"] = returns.skew()
    expected_returns["kurtosis"] = returns.kurtosis()
    expected_returns["annualized_return"] = returns.mean() * 252
    expected_returns["annualized_volatility"] = returns.std() * (252**0.5)
    assert_frame_equal(return_stats, expected_returns)


def test_risk_return_summary(tsp_price: TspAnalytics) -> None:
    mar = 0.02
    confidence = 0.75
    trading_days = 252
    summary = tsp_price.get_risk_return_summary(
        trading_days=trading_days,
        mar=mar,
        confidence=confidence,
    )
    returns = tsp_price.get_daily_returns()
    annualized_return = returns.mean().mul(trading_days)
    annualized_volatility = returns.std().mul(trading_days**0.5)
    sharpe_ratio = annualized_return.div(annualized_volatility)
    daily_mar = (1 + mar) ** (1 / trading_days) - 1
    downside = (returns - daily_mar).where(returns < daily_mar, 0)
    downside_deviation = downside.pow(2).mean().pow(0.5).mul(trading_days**0.5)
    sortino_ratio = (annualized_return - mar).div(downside_deviation)
    omega_ratio = {}
    for column in returns.columns:
        gains = (
            (returns[column] - daily_mar).where(returns[column] > daily_mar, 0).sum()
        )
        losses = (
            (daily_mar - returns[column]).where(returns[column] < daily_mar, 0).sum()
        )
        omega_ratio[column] = gains / losses if losses != 0 else float("nan")
    value_at_risk = returns.quantile(1 - confidence)
    expected_shortfall = {
        column: returns[column][returns[column] <= value_at_risk[column]].mean()
        for column in returns.columns
    }
    price_df = tsp_price.dataframe.set_index("Date").dropna(how="all")
    max_drawdown = {
        column: tsp_price._calculate_max_drawdown(price_df[column])
        for column in returns.columns
    }
    max_drawdown_series = pd.Series(max_drawdown, name="max_drawdown").rename_axis(
        "fund"
    )
    calmar_ratio = annualized_return.div(max_drawdown_series.abs()).where(
        max_drawdown_series != 0
    )
    ulcer_index = {
        column: (
            price_df[column].div(price_df[column].cummax()).sub(1).pow(2).mean() ** 0.5
        )
        for column in returns.columns
    }
    drawdown_duration_days = {}
    drawdown_recovery_days = {}
    pain_index = {}
    for column in returns.columns:
        prices = price_df[column].dropna()
        cumulative_max = prices.cummax()
        drawdown = prices.div(cumulative_max).sub(1)
        trough_date = drawdown.idxmin()
        peak_date = prices.loc[:trough_date].idxmax()
        drawdown_duration_days[column] = int((trough_date - peak_date).days)
        peak_price = prices.loc[peak_date]
        recovery_prices = prices.loc[trough_date:]
        recovered = recovery_prices[recovery_prices >= peak_price]
        if recovered.empty:
            drawdown_recovery_days[column] = float("nan")
        else:
            recovery_date = recovered.index[0]
            drawdown_recovery_days[column] = int((recovery_date - trough_date).days)
        pain_index[column] = drawdown.abs().mean()
    pain_index_series = pd.Series(pain_index, name="pain_index").rename_axis("fund")
    pain_ratio = annualized_return.div(pain_index_series).where(pain_index_series != 0)

    assert_series_equal(
        summary["annualized_return"],
        annualized_return.rename("annualized_return").rename_axis("fund"),
    )
    assert_series_equal(
        summary["annualized_volatility"],
        annualized_volatility.rename("annualized_volatility").rename_axis("fund"),
    )
    assert_series_equal(
        summary["sharpe_ratio"],
        sharpe_ratio.rename("sharpe_ratio").rename_axis("fund"),
    )
    assert_series_equal(
        summary["sortino_ratio"],
        sortino_ratio.rename("sortino_ratio").rename_axis("fund"),
    )
    assert_series_equal(
        summary["value_at_risk"],
        value_at_risk.rename("value_at_risk").rename_axis("fund"),
    )
    assert_series_equal(
        summary["expected_shortfall"],
        pd.Series(expected_shortfall, name="expected_shortfall").rename_axis("fund"),
    )
    assert_series_equal(
        summary["max_drawdown"],
        max_drawdown_series,
    )
    assert_series_equal(
        summary["calmar_ratio"],
        calmar_ratio.rename("calmar_ratio").rename_axis("fund"),
    )
    assert_series_equal(
        summary["ulcer_index"],
        pd.Series(ulcer_index, name="ulcer_index").rename_axis("fund"),
    )
    assert_series_equal(
        summary["omega_ratio"],
        pd.Series(omega_ratio, name="omega_ratio").rename_axis("fund"),
    )
    assert_series_equal(
        summary["max_drawdown_duration_days"],
        pd.Series(
            drawdown_duration_days, name="max_drawdown_duration_days"
        ).rename_axis("fund"),
    )
    assert_series_equal(
        summary["max_drawdown_recovery_days"],
        pd.Series(
            drawdown_recovery_days, name="max_drawdown_recovery_days"
        ).rename_axis("fund"),
    )
    assert_series_equal(
        summary["pain_index"],
        pain_index_series,
    )
    assert_series_equal(
        summary["pain_ratio"],
        pain_ratio.rename("pain_ratio").rename_axis("fund"),
    )

    fund = TspIndividualFund.G_FUND
    fund_summary = tsp_price.get_risk_return_summary(fund=fund)
    assert fund_summary.index.tolist() == [fund.value]


def test_risk_return_summary_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_risk_return_summary_dict(
        trading_days=252, mar=0.02, confidence=0.9
    )
    assert payload["trading_days"] == 252
    assert payload["mar"] == pytest.approx(0.02)
    assert payload["confidence"] == pytest.approx(0.9)
    assert TspIndividualFund.G_FUND.value in payload["funds"]

    fund_payload = tsp_price.get_risk_return_summary_dict(fund=TspIndividualFund.G_FUND)
    assert list(fund_payload["funds"].keys()) == [TspIndividualFund.G_FUND.value]


def test_risk_return_summary_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_risk_return_summary(trading_days=0)

    with pytest.raises(ValueError, match="confidence must be a float between 0 and 1"):
        tsp_price.get_risk_return_summary(confidence=1.5)

    with pytest.raises(ValueError, match="mar must be a numeric value"):
        tsp_price.get_risk_return_summary(mar="0.01")


def test_drawdown_and_performance_summary(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    max_drawdown = tsp_price.get_max_drawdown(fund)
    assert max_drawdown == Decimal("0")

    drawdown_series = pd.Series([100, 90, 95, 80])
    assert tsp_price._calculate_max_drawdown(drawdown_series) == pytest.approx(-0.2)

    drawdown = tsp_price.get_drawdown_series(fund=fund)
    price_df = tsp_price.dataframe[["Date", fund.value]].set_index("Date")
    expected_drawdown = price_df.div(price_df.cummax()).sub(1).dropna(how="all")
    assert_frame_equal(drawdown, expected_drawdown)

    summary = tsp_price.get_performance_summary(fund=fund, trading_days=252)
    assert summary.index.tolist() == [fund.value]
    for key in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]:
        assert key in summary.columns

    all_summaries = tsp_price.get_performance_summary()
    for fund_name in tsp_price.dataframe.set_index("Date").columns:
        assert fund_name in all_summaries.index

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_performance_summary(trading_days=0)

    empty_summary = tsp_price._calculate_performance_summary(
        pd.Series([100]), trading_days=252
    )
    assert all(pd.isna(value) for value in empty_summary.values())


def test_performance_summary_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_performance_summary_dict(trading_days=252)
    assert payload["trading_days"] == 252
    assert TspIndividualFund.G_FUND.value in payload["funds"]

    fund_payload = tsp_price.get_performance_summary_dict(fund=TspIndividualFund.G_FUND)
    assert list(fund_payload["funds"].keys()) == [TspIndividualFund.G_FUND.value]


def test_performance_summary_by_date_range(tsp_price: TspAnalytics) -> None:
    start_date = date(2024, 1, 2)
    end_date = date(2024, 1, 4)
    summary = tsp_price.get_performance_summary_by_date_range(start_date, end_date)
    for fund_name in tsp_price.dataframe.set_index("Date").columns:
        assert fund_name in summary.index

    fund_summary = tsp_price.get_performance_summary_by_date_range(
        start_date,
        end_date,
        fund=TspLifecycleFund.L_2030,
    )
    assert fund_summary.index.tolist() == [TspLifecycleFund.L_2030.value]

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_performance_summary_by_date_range(
            start_date, end_date, trading_days=0
        )


def test_performance_summary_by_date_range_dict(tsp_price: TspAnalytics) -> None:
    start_date = date(2024, 1, 2)
    end_date = date(2024, 1, 4)
    payload = tsp_price.get_performance_summary_by_date_range_dict(start_date, end_date)
    assert payload["start_date"] == "2024-01-02"
    assert payload["end_date"] == "2024-01-04"
    assert TspLifecycleFund.L_2030.value in payload["funds"]

    as_date = tsp_price.get_performance_summary_by_date_range_dict(
        start_date,
        end_date,
        date_format=None,
        fund=TspLifecycleFund.L_2030,
    )
    assert as_date["start_date"] == start_date
    assert as_date["end_date"] == end_date


def test_get_cagr(tsp_price: TspAnalytics) -> None:
    dataframe = build_yearly_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    cagr = tsp_price.get_cagr()
    fund = TspIndividualFund.G_FUND.value
    series = dataframe.set_index("Date")[fund]
    expected_years = (series.index[-1] - series.index[0]).days / 365.25
    expected_cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / expected_years) - 1
    assert cagr.loc[fund, "cagr"] == pytest.approx(expected_cagr)
    assert cagr.loc[fund, "start_date"] == series.index[0].date()
    assert cagr.loc[fund, "end_date"] == series.index[-1].date()

    fund_cagr = tsp_price.get_cagr(fund=TspIndividualFund.G_FUND)
    assert fund_cagr.index.tolist() == [fund]

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_cagr(start_date=date(2024, 1, 1))


def test_correlation_matrix(tsp_price: TspAnalytics) -> None:
    correlation = tsp_price.get_correlation_matrix()
    expected = tsp_price.get_daily_returns().corr()
    assert_frame_equal(correlation, expected)

    long = tsp_price.get_correlation_matrix_long()
    assert long.columns.tolist() == ["fund_a", "fund_b", "correlation"]
    sample = long[
        (long["fund_a"] == TspIndividualFund.G_FUND.value)
        & (long["fund_b"] == TspIndividualFund.F_FUND.value)
    ]["correlation"].iloc[0]
    assert sample == pytest.approx(
        correlation.loc[TspIndividualFund.G_FUND.value, TspIndividualFund.F_FUND.value]
    )


def test_rolling_correlation_matrix(tsp_price: TspAnalytics) -> None:
    rolling = tsp_price.get_rolling_correlation_matrix(window=2)
    expected = tsp_price.get_daily_returns().tail(2).corr()
    assert_frame_equal(rolling, expected)

    rolling_long = tsp_price.get_rolling_correlation_matrix_long(window=2)
    assert rolling_long.columns.tolist() == ["fund_a", "fund_b", "correlation"]
    sample = rolling_long[
        (rolling_long["fund_a"] == TspIndividualFund.G_FUND.value)
        & (rolling_long["fund_b"] == TspIndividualFund.F_FUND.value)
    ]["correlation"].iloc[0]
    assert sample == pytest.approx(
        rolling.loc[TspIndividualFund.G_FUND.value, TspIndividualFund.F_FUND.value]
    )

    with pytest.raises(
        ValueError,
        match="not enough return data available for rolling correlation matrix",
    ):
        tsp_price.get_rolling_correlation_matrix(window=10)


def test_drawdown_series_long(tsp_price: TspAnalytics) -> None:
    drawdown_long = tsp_price.get_drawdown_series_long()
    assert drawdown_long.columns.tolist() == ["Date", "fund", "drawdown"]
    assert drawdown_long["fund"].nunique() == len(tsp_price.get_available_funds())

    fund = TspIndividualFund.G_FUND
    fund_drawdown = tsp_price.get_drawdown_series_long(fund=fund)
    assert fund_drawdown["fund"].unique().tolist() == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_drawdown_series_long(
            fund=fund,
            funds=[TspIndividualFund.G_FUND],
        )

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_drawdown_series_long(start_date=date(2024, 1, 1))


def test_get_price_statistics(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_price_statistics()
    assert "mean" in summary.columns
    assert TspIndividualFund.G_FUND.value in summary.index

    fund = TspIndividualFund.G_FUND
    fund_summary = tsp_price.get_price_statistics(fund=fund)
    assert fund_summary.index.tolist() == [fund.value]


def test_get_price_statistics_date_range_validation(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_price_statistics(start_date=date(2024, 1, 1))
    assert summary.index.tolist() != []


def test_get_price_statistics_date_range(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_price_statistics(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
    )
    assert summary.index.tolist() != []


def test_get_return_statistics(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_return_statistics()
    assert "mean" in summary.columns
    assert "annualized_return" in summary.columns
    assert "annualized_volatility" in summary.columns
    assert "skew" in summary.columns
    assert "kurtosis" in summary.columns
    assert TspIndividualFund.G_FUND.value in summary.index

    fund_summary = tsp_price.get_return_statistics(fund=TspIndividualFund.G_FUND)
    assert fund_summary.index.tolist() == [TspIndividualFund.G_FUND.value]

    range_summary = tsp_price.get_return_statistics(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
    )
    assert range_summary.index.tolist() != []

    range_summary_single_bound = tsp_price.get_return_statistics(
        start_date=date(2024, 1, 1)
    )
    assert range_summary_single_bound.index.tolist() != []

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_return_statistics(trading_days=0)


def test_numpy_numeric_inputs(tsp_price: TspAnalytics) -> None:
    normalized = tsp_price.get_normalized_prices(base_value=np.float64(100.0))
    assert not normalized.empty

    var = tsp_price.get_value_at_risk(confidence=np.float64(0.75))
    assert "value_at_risk" in var.columns
