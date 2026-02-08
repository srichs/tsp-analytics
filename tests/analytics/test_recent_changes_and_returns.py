from datetime import date

import numpy as np
import pandas as pd
import pytest

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics


def test_recent_price_change_helpers(tsp_price: TspAnalytics) -> None:
    recent = tsp_price.get_recent_price_changes(days=2, fund=TspIndividualFund.G_FUND)
    assert recent.index[0].date() == date(2024, 1, 3)
    assert recent.index[-1].date() == date(2024, 1, 4)
    assert recent.shape == (2, 1)

    change_1 = (12.0 - 11.0) / 11.0
    change_2 = (13.0 - 12.0) / 12.0
    assert recent.iloc[0, 0] == pytest.approx(change_1)
    assert recent.iloc[1, 0] == pytest.approx(change_2)

    recent_long = tsp_price.get_recent_price_changes_long(
        days=2, fund=TspIndividualFund.G_FUND
    )
    assert recent_long.columns.tolist() == ["Date", "fund", "change_percent"]
    assert set(recent_long["fund"]) == {TspIndividualFund.G_FUND.value}

    recent_dict = tsp_price.get_recent_price_changes_dict(
        days=2, fund=TspIndividualFund.G_FUND
    )
    assert recent_dict["start_date"] == "2024-01-03"
    assert recent_dict["end_date"] == "2024-01-04"
    assert (
        recent_dict["funds"][TspIndividualFund.G_FUND.value][0]["date"] == "2024-01-03"
    )

    summary = tsp_price.get_recent_price_change_summary(
        days=2, fund=TspIndividualFund.G_FUND
    )
    cumulative = (1 + change_1) * (1 + change_2) - 1
    assert summary.loc[
        TspIndividualFund.G_FUND.value, "cumulative_return"
    ] == pytest.approx(cumulative)
    assert summary.loc[TspIndividualFund.G_FUND.value, "start_date"] == date(2024, 1, 3)

    summary_dict = tsp_price.get_recent_price_change_summary_dict(
        days=2, fund=TspIndividualFund.G_FUND
    )
    assert summary_dict["start_date"] == "2024-01-03"
    assert summary_dict["funds"][TspIndividualFund.G_FUND.value][
        "cumulative_return"
    ] == pytest.approx(cumulative)


def test_price_changes_as_of_per_fund(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 3)
    changes = tsp_price.get_price_changes_as_of_per_fund(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030],
    )
    assert changes.columns.tolist() == [
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert changes.index.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]
    assert changes.loc[TspIndividualFund.G_FUND.value, "as_of"] == as_of
    assert changes.loc[TspIndividualFund.G_FUND.value, "previous_as_of"] == date(
        2024, 1, 2
    )

    changes_long = tsp_price.get_price_changes_as_of_per_fund_long(
        as_of,
        funds=[TspIndividualFund.G_FUND],
    )
    assert changes_long.columns.tolist() == [
        "fund",
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]

    payload = tsp_price.get_price_changes_as_of_per_fund_dict(
        as_of,
        funds=[TspIndividualFund.G_FUND],
    )
    assert payload["requested_as_of"] == "2024-01-03"
    assert payload["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-03"
    assert (
        payload["funds"][TspIndividualFund.G_FUND.value]["previous_as_of"]
        == "2024-01-02"
    )


def test_price_changes_as_of_per_fund_handles_missing_data() -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0, 102.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0, np.nan],
        }
    )
    prices.load_dataframe(dataframe)

    changes = prices.get_price_changes_as_of_per_fund(
        date(2024, 1, 3),
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    )
    assert changes.loc[TspIndividualFund.C_FUND.value, "as_of"] == date(2024, 1, 2)
    assert changes.loc[TspIndividualFund.C_FUND.value, "previous_as_of"] == date(
        2024, 1, 1
    )

    with pytest.raises(ValueError, match="at least two data points are required"):
        prices.get_price_changes_as_of_per_fund(date(2024, 1, 1))


def test_monthly_return_table_long_and_dict() -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"]),
            TspIndividualFund.G_FUND.value: [100.0, 110.0, 121.0],
        }
    )
    prices.load_dataframe(dataframe)

    long_table = prices.get_monthly_return_table_long(TspIndividualFund.G_FUND)
    assert long_table.columns.tolist() == [
        "year",
        "month",
        "return",
        "month_num",
        "fund",
    ]
    assert long_table["month"].tolist() == ["Feb", "Mar"]
    assert long_table["month_num"].tolist() == [2, 3]
    assert long_table["fund"].unique().tolist() == [TspIndividualFund.G_FUND.value]
    assert long_table["return"].iloc[0] == pytest.approx(0.1)

    payload = prices.get_monthly_return_table_dict(TspIndividualFund.G_FUND)
    assert payload["fund"] == TspIndividualFund.G_FUND.value
    assert payload["returns"][0]["month"] == "Feb"
    assert payload["returns"][0]["month_num"] == 2


def test_normalized_prices_use_first_valid_prices() -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [np.nan, 100.0, 110.0],
            TspIndividualFund.C_FUND.value: [200.0, 210.0, 220.0],
            TspLifecycleFund.L_2030.value: [np.nan, np.nan, np.nan],
        }
    )
    prices.load_dataframe(dataframe)

    normalized = prices.get_normalized_prices(base_value=100.0)
    assert TspLifecycleFund.L_2030.value not in normalized.columns

    g_fund = TspIndividualFund.G_FUND.value
    c_fund = TspIndividualFund.C_FUND.value

    assert pd.isna(normalized.loc[pd.Timestamp("2024-01-01"), g_fund])
    assert normalized.loc[pd.Timestamp("2024-01-02"), g_fund] == pytest.approx(100.0)
    assert normalized.loc[pd.Timestamp("2024-01-03"), g_fund] == pytest.approx(110.0)

    assert normalized.loc[pd.Timestamp("2024-01-01"), c_fund] == pytest.approx(100.0)
    assert normalized.loc[pd.Timestamp("2024-01-03"), c_fund] == pytest.approx(110.0)


def test_current_prices_per_fund_flag_handles_missing_data() -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0, 102.0],
            TspIndividualFund.C_FUND.value: [200.0, None, None],
        }
    )
    prices.load_dataframe(dataframe)

    per_fund = prices.get_current_prices(per_fund=True)
    assert per_fund.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 3)
    assert per_fund.loc[TspIndividualFund.C_FUND.value, "as_of"] == date(2024, 1, 1)

    per_fund_long = prices.get_current_prices_long(per_fund=True)
    assert set(per_fund_long["fund"]) == {
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    }

    per_fund_dict = prices.get_current_prices_dict(per_fund=True)
    assert (
        per_fund_dict["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-03"
    )
    assert (
        per_fund_dict["funds"][TspIndividualFund.C_FUND.value]["as_of"] == "2024-01-01"
    )


def test_fund_rankings_can_filter_funds(tsp_price: TspAnalytics) -> None:
    rankings = tsp_price.get_fund_rankings(
        metric="trailing_return",
        period=1,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    )
    assert set(rankings.index) == {
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    }


def test_trailing_returns_outputs(tsp_price: TspAnalytics) -> None:
    trailing = tsp_price.get_trailing_returns(periods=1)
    assert trailing.index.tolist() == [1]
    assert TspIndividualFund.G_FUND.value in trailing.columns

    trailing_subset = tsp_price.get_trailing_returns(
        periods=[1, 3], funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert trailing_subset.index.tolist() == [1, 3]
    assert set(trailing_subset.columns) == {
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    }

    trailing_long = tsp_price.get_trailing_returns_long(
        periods=[1, 3], funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert set(trailing_long.columns) == {"period", "fund", "trailing_return"}
    assert set(trailing_long["period"]) == {1, 3}
    assert set(trailing_long["fund"]) == {
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    }

    trailing_dict = tsp_price.get_trailing_returns_dict(
        periods=[1, 3], funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert trailing_dict["periods"] == [1, 3]
    assert TspIndividualFund.G_FUND.value in trailing_dict["funds"]
    assert "1" in trailing_dict["funds"][TspIndividualFund.G_FUND.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_trailing_returns(
            periods=1,
            fund=TspIndividualFund.G_FUND,
            funds=[TspIndividualFund.C_FUND],
        )


def test_return_statistics_dict(tsp_price: TspAnalytics) -> None:
    stats = tsp_price.get_return_statistics_dict(
        fund=TspIndividualFund.G_FUND, trading_days=252
    )
    payload = stats["statistics"][TspIndividualFund.G_FUND.value]
    assert payload["count"] == pytest.approx(3.0)
    assert payload["mean"] == pytest.approx(0.091414, rel=1e-4)
    assert payload["annualized_return"] == pytest.approx(0.091414 * 252, rel=1e-4)
