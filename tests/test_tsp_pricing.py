from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics
from tests.helpers import (
    DummyResponse,
    DummySession,
    build_monthly_price_dataframe,
    build_minimal_price_dataframe,
)


def test_get_price_and_date_queries(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    expected_price = Decimal(str(tsp_price.current[fund.value]))
    assert tsp_price.get_price(fund) == expected_price

    target_date = date(2024, 1, 2)
    prices = tsp_price.get_prices_by_date(target_date)
    assert len(prices) == 1
    assert prices.iloc[0]["Date"].date() == target_date

    price = tsp_price.get_fund_price_by_date(fund, target_date)
    assert price == Decimal(str(prices.iloc[0][fund.value]))
    assert tsp_price.get_fund_price_by_date(fund, date(2023, 12, 31)) is None


def test_get_price_refreshes_latest_row(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    stale_row = tsp_price.dataframe.iloc[0]
    tsp_price.current = stale_row
    tsp_price.latest = stale_row["Date"].date()

    price = tsp_price.get_price(fund)
    latest_row = tsp_price.dataframe.iloc[-1]
    assert price == Decimal(str(latest_row[fund.value]))
    assert tsp_price.latest == latest_row["Date"].date()


def test_get_price_uses_latest_valid_value_when_latest_row_missing(
    tsp_price: TspAnalytics,
) -> None:
    fund = TspIndividualFund.G_FUND
    tsp_price.dataframe.loc[tsp_price.dataframe.index[-1], fund.value] = np.nan
    tsp_price.current = tsp_price.dataframe.loc[tsp_price.dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    price = tsp_price.get_price(fund)
    expected = Decimal(str(tsp_price.dataframe.iloc[-2][fund.value]))
    assert price == expected


def test_get_price_raises_when_fund_has_no_valid_prices(
    tsp_price: TspAnalytics,
) -> None:
    fund = TspIndividualFund.G_FUND
    tsp_price.dataframe[fund.value] = np.nan
    tsp_price.current = tsp_price.dataframe.loc[tsp_price.dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    with pytest.raises(ValueError, match="fund not available in data: G Fund"):
        tsp_price.get_price(fund)


def test_date_range_helpers(tsp_price: TspAnalytics) -> None:
    prices_by_month = tsp_price.get_prices_by_month(2024, 1)
    assert len(prices_by_month) == 4

    prices_by_year = tsp_price.get_prices_by_year(2024)
    assert len(prices_by_year) == 4

    fund = TspLifecycleFund.L_2030
    prices_for_fund_month = tsp_price.get_fund_prices_by_month(fund, 2024, 1)
    assert len(prices_for_fund_month) == 4
    prices_for_fund_year = tsp_price.get_fund_prices_by_year(fund, 2024)
    assert len(prices_for_fund_year) == 4


def test_price_history_allows_single_date_bound(tsp_price: TspAnalytics) -> None:
    start_only = tsp_price.get_price_history(start_date=date(2024, 1, 2))
    assert start_only["Date"].min().date() == date(2024, 1, 2)
    assert start_only["Date"].max().date() == date(2024, 1, 4)

    end_only = tsp_price.get_price_history(end_date=date(2024, 1, 3))
    assert end_only["Date"].min().date() == date(2024, 1, 1)
    assert end_only["Date"].max().date() == date(2024, 1, 3)


def test_price_history_accepts_fund_parameter(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    history = tsp_price.get_price_history(fund=fund)
    assert history.columns.tolist() == ["Date", fund.value]

    history_long = tsp_price.get_price_history_long(fund=fund)
    assert history_long["fund"].unique().tolist() == [fund.value]

    history_dict = tsp_price.get_price_history_long_dict(fund=fund)
    assert {record["fund"] for record in history_dict["prices"]} == {fund.value}

    metrics_wide = tsp_price.get_price_history_with_metrics(fund=fund)
    assert metrics_wide.columns.get_level_values(0).unique().tolist() == [fund.value]

    metrics_long = tsp_price.get_price_history_with_metrics_long(fund=fund)
    assert metrics_long["fund"].unique().tolist() == [fund.value]

    metrics_dict = tsp_price.get_price_history_with_metrics_dict(fund=fund)
    assert {record["fund"] for record in metrics_dict["metrics"]} == {fund.value}

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_price_history(fund=fund, funds=[fund])


def test_price_history_drops_rows_with_all_missing_funds() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            TspIndividualFund.G_FUND.value: [100.0, np.nan, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, np.nan, 201.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    history = tsp_price.get_price_history(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert history["Date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 4)]

    history_g = tsp_price.get_price_history(fund=TspIndividualFund.G_FUND)
    assert history_g["Date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 4)]

    history_long = tsp_price.get_price_history_long(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert set(history_long["Date"].dt.date.unique()) == {
        date(2024, 1, 2),
        date(2024, 1, 4),
    }


def test_month_and_year_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="year must be a positive integer"):
        tsp_price.get_prices_by_year(0)

    with pytest.raises(ValueError, match="month must be an integer between 1 and 12"):
        tsp_price.get_prices_by_month(2024, 13)

    with pytest.raises(ValueError, match="month must be an integer between 1 and 12"):
        tsp_price.get_fund_prices_by_month(TspIndividualFund.G_FUND, 2024, 0)

    with pytest.raises(ValueError, match="year must be a positive integer"):
        tsp_price.get_fund_prices_by_year(TspLifecycleFund.L_2030, True)


def test_latest_prices_and_changes(tsp_price: TspAnalytics) -> None:
    latest_prices = tsp_price.get_latest_prices()
    assert latest_prices.index[0].date() == tsp_price.latest
    assert TspIndividualFund.G_FUND.value in latest_prices.columns

    fund = TspIndividualFund.G_FUND
    latest_fund_price = tsp_price.get_latest_prices(fund=fund)
    assert latest_fund_price.columns.tolist() == [fund.value]

    latest_funds = tsp_price.get_latest_prices(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert latest_funds.columns.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]

    changes = tsp_price.get_latest_price_changes(fund=fund)
    assert changes.index.name == "fund"
    latest_price = float(tsp_price.dataframe[fund.value].iloc[-1])
    previous_price = float(tsp_price.dataframe[fund.value].iloc[-2])
    assert changes.loc[fund.value, "latest_price"] == pytest.approx(latest_price)
    assert changes.loc[fund.value, "previous_price"] == pytest.approx(previous_price)
    assert changes.loc[fund.value, "change"] == pytest.approx(
        latest_price - previous_price
    )
    assert changes.loc[fund.value, "change_percent"] == pytest.approx(
        (latest_price - previous_price) / previous_price
    )

    changes_subset = tsp_price.get_latest_price_changes(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert changes_subset.index.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_latest_prices(
            fund=TspIndividualFund.G_FUND,
            funds=[TspIndividualFund.G_FUND],
        )

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_latest_price_changes(
            fund=TspIndividualFund.G_FUND,
            funds=[TspIndividualFund.G_FUND],
        )


def test_current_prices_accepts_as_of_date(tsp_price: TspAnalytics) -> None:
    target_date = date(2024, 1, 2)
    as_of_prices = tsp_price.get_current_prices(as_of=target_date)
    expected = tsp_price.get_prices_as_of(target_date)
    assert_frame_equal(as_of_prices, expected)

    as_of_long = tsp_price.get_current_prices_long(as_of=target_date)
    expected_long = (
        expected.reset_index()
        .melt(id_vars="Date", var_name="fund", value_name="price")
        .dropna(subset=["price"])
    )
    assert_frame_equal(
        as_of_long.reset_index(drop=True), expected_long.reset_index(drop=True)
    )

    as_of_dict = tsp_price.get_current_prices_dict(as_of=target_date)
    expected_dict = tsp_price.get_prices_as_of_dict(target_date)
    assert as_of_dict == expected_dict


def test_current_prices_reject_invalid_flags(tsp_price: TspAnalytics) -> None:
    with pytest.raises(
        ValueError, match="require_all_funds cannot be used when per_fund is True"
    ):
        tsp_price.get_current_prices(per_fund=True, require_all_funds=True)

    with pytest.raises(
        ValueError, match="allow_missing can only be used when per_fund is True"
    ):
        tsp_price.get_current_prices(allow_missing=True)

    with pytest.raises(
        ValueError, match="require_all_funds cannot be used when per_fund is True"
    ):
        tsp_price.get_current_prices_long(per_fund=True, require_all_funds=True)

    with pytest.raises(
        ValueError, match="allow_missing can only be used when per_fund is True"
    ):
        tsp_price.get_current_prices_long(allow_missing=True)

    with pytest.raises(
        ValueError, match="require_all_funds cannot be used when per_fund is True"
    ):
        tsp_price.get_current_prices_dict(per_fund=True, require_all_funds=True)

    with pytest.raises(
        ValueError, match="allow_missing can only be used when per_fund is True"
    ):
        tsp_price.get_current_prices_dict(allow_missing=True)


def test_current_prices_per_fund_accepts_as_of_date(tsp_price: TspAnalytics) -> None:
    target_date = date(2024, 1, 2)
    as_of_per_fund = tsp_price.get_current_prices_per_fund(as_of=target_date)
    expected = tsp_price.get_prices_as_of_per_fund(target_date)
    assert_frame_equal(as_of_per_fund, expected)

    as_of_per_fund_long = tsp_price.get_current_prices_per_fund_long(as_of=target_date)
    expected_long = tsp_price.get_prices_as_of_per_fund_long(target_date)
    assert_frame_equal(as_of_per_fund_long, expected_long)

    as_of_per_fund_dict = tsp_price.get_current_prices_per_fund_dict(as_of=target_date)
    expected_dict = tsp_price.get_prices_as_of_per_fund_dict(target_date)
    assert as_of_per_fund_dict == expected_dict


def test_get_prices_as_of_uses_last_valid_value_for_fund() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, np.nan],
            TspIndividualFund.C_FUND.value: [200.0, 201.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    as_of = date(2024, 1, 4)
    prices = tsp_price.get_prices_as_of(as_of, fund=TspIndividualFund.G_FUND)
    assert prices.index[0].date() == date(2024, 1, 2)
    assert prices.iloc[0][TspIndividualFund.G_FUND.value] == pytest.approx(100.0)

    price = tsp_price.get_price_as_of(TspIndividualFund.G_FUND, as_of)
    assert price == Decimal("100.0")


def test_get_prices_as_of_skips_rows_with_all_missing_funds() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, np.nan],
            TspIndividualFund.C_FUND.value: [200.0, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    as_of = date(2024, 1, 4)
    prices = tsp_price.get_prices_as_of(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    )
    assert prices.index[0].date() == date(2024, 1, 2)
    assert prices.iloc[0][TspIndividualFund.G_FUND.value] == pytest.approx(100.0)
    assert prices.iloc[0][TspIndividualFund.C_FUND.value] == pytest.approx(200.0)


def test_get_prices_as_of_require_all_funds() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    as_of = date(2024, 1, 3)
    any_funds = tsp_price.get_prices_as_of(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    )
    assert any_funds.index[0].date() == date(2024, 1, 3)

    all_funds = tsp_price.get_prices_as_of(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        require_all_funds=True,
    )
    assert all_funds.index[0].date() == date(2024, 1, 2)

    as_of_payload = tsp_price.get_prices_as_of_dict(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        require_all_funds=True,
    )
    assert as_of_payload["as_of"] == "2024-01-02"

    current_payload = tsp_price.get_current_prices_dict(
        as_of=as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        require_all_funds=True,
    )
    assert current_payload["as_of"] == "2024-01-02"


def test_get_prices_as_of_ignores_non_fund_columns() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0],
            "Notes": ["a", "b"],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    as_of = date(2024, 1, 3)
    prices = tsp_price.get_prices_as_of(as_of)
    assert prices.columns.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    ]

    current = tsp_price.get_current_prices()
    assert current.columns.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    ]


def test_fund_aliases_and_metadata(tsp_price: TspAnalytics) -> None:
    aliases = tsp_price.get_fund_aliases()
    g_fund = TspIndividualFund.G_FUND.value
    assert "g" in aliases[g_fund]
    assert "g fund" in aliases[g_fund]

    metadata = tsp_price.get_fund_metadata()
    assert g_fund in metadata.index
    assert metadata.loc[g_fund, "category"] == "individual"
    assert bool(metadata.loc[g_fund, "available"]) is True

    metadata_dict = tsp_price.get_fund_metadata_dict()
    assert metadata_dict["funds"][g_fund]["category"] == "individual"
    assert metadata_dict["funds"][g_fund]["available"] is True


def test_fund_aliases_resolve_for_latest_prices(tsp_price: TspAnalytics) -> None:
    latest = tsp_price.get_latest_prices(funds="g-fund")
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]
    lifecycle = tsp_price.get_latest_prices(funds="L2050")
    assert lifecycle.columns.tolist() == [TspLifecycleFund.L_2050.value]
    lifecycle_named = tsp_price.get_latest_prices(funds="Lifecycle 2050")
    assert lifecycle_named.columns.tolist() == [TspLifecycleFund.L_2050.value]


def test_fund_aliases_handle_compact_lifecycle_names(tsp_price: TspAnalytics) -> None:
    compact = tsp_price.get_latest_prices(funds="L2050 fund")
    assert compact.columns.tolist() == [TspLifecycleFund.L_2050.value]
    collapsed = tsp_price.get_latest_prices(funds="l2050fund")
    assert collapsed.columns.tolist() == [TspLifecycleFund.L_2050.value]


def test_get_fund_aliases_returns_normalized_aliases(tsp_price: TspAnalytics) -> None:
    aliases = tsp_price.get_fund_aliases()
    assert "g" in aliases[TspIndividualFund.G_FUND.value]
    assert "g fund" in aliases[TspIndividualFund.G_FUND.value]
    assert "l2050" in aliases[TspLifecycleFund.L_2050.value]
    assert "l 2050 fund" in aliases[TspLifecycleFund.L_2050.value]
    assert "lifecycle 2050" in aliases[TspLifecycleFund.L_2050.value]


def test_get_fund_metadata_includes_aliases_and_availability(
    tsp_price: TspAnalytics,
) -> None:
    metadata = tsp_price.get_fund_metadata()
    assert metadata.index.name == "fund"
    assert metadata.loc[TspIndividualFund.G_FUND.value, "category"] == "individual"
    assert metadata.loc[TspLifecycleFund.L_2030.value, "category"] == "lifecycle"
    assert "g" in metadata.loc[TspIndividualFund.G_FUND.value, "aliases"]
    assert bool(metadata.loc[TspIndividualFund.G_FUND.value, "available"]) is True

    without_aliases = tsp_price.get_fund_metadata(include_aliases=False)
    assert "aliases" not in without_aliases.columns

    without_availability = tsp_price.get_fund_metadata(include_availability=False)
    assert "available" not in without_availability.columns


def test_get_fund_metadata_dict_handles_empty_dataframe(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    payload = tsp_price.get_fund_metadata_dict()
    assert TspIndividualFund.G_FUND.value in payload["funds"]
    assert payload["funds"][TspIndividualFund.G_FUND.value]["available"] is False
    assert payload["funds"][TspIndividualFund.G_FUND.value]["category"] == "individual"

    payload_no_alias = tsp_price.get_fund_metadata_dict(include_aliases=False)
    assert "aliases" not in payload_no_alias["funds"][TspIndividualFund.G_FUND.value]


def test_latest_prices_per_fund_handles_missing_latest() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    per_fund = tsp_price.get_latest_prices_per_fund(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert per_fund.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 3)
    assert per_fund.loc[TspIndividualFund.C_FUND.value, "as_of"] == date(2024, 1, 2)
    assert per_fund.loc[TspIndividualFund.G_FUND.value, "price"] == pytest.approx(101.0)
    assert per_fund.loc[TspIndividualFund.C_FUND.value, "price"] == pytest.approx(200.0)

    current = tsp_price.get_current_prices_per_fund()
    assert_frame_equal(current, per_fund)


def test_latest_prices_per_fund_allow_missing_skips_empty_funds() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [np.nan, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="funds not available in data"):
        tsp_price.get_latest_prices_per_fund(
            funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
        )

    per_fund = tsp_price.get_latest_prices_per_fund(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        allow_missing=True,
    )
    assert per_fund.index.tolist() == [TspIndividualFund.G_FUND.value]

    payload = tsp_price.get_latest_prices_per_fund_dict(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        allow_missing=True,
    )
    assert payload["missing_funds"] == [TspIndividualFund.C_FUND.value]

    as_of = date(2024, 1, 3)
    as_of_prices = tsp_price.get_prices_as_of_per_fund(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        allow_missing=True,
    )
    assert as_of_prices.index.tolist() == [TspIndividualFund.G_FUND.value]

    as_of_payload = tsp_price.get_prices_as_of_per_fund_dict(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        allow_missing=True,
    )
    assert as_of_payload["missing_funds"] == [TspIndividualFund.C_FUND.value]

    current_per_fund = tsp_price.get_current_prices(per_fund=True, allow_missing=True)
    assert_frame_equal(current_per_fund, per_fund)

    current_payload = tsp_price.get_current_prices_dict(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        per_fund=True,
        allow_missing=True,
    )
    assert current_payload["missing_funds"] == [TspIndividualFund.C_FUND.value]


def test_latest_prices_per_fund_long_and_dict(tsp_price: TspAnalytics) -> None:
    per_fund_long = tsp_price.get_latest_prices_per_fund_long(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert per_fund_long.columns.tolist() == ["fund", "as_of", "price"]
    assert per_fund_long["fund"].tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]
    assert per_fund_long["as_of"].nunique() == 1

    payload = tsp_price.get_latest_prices_per_fund_dict(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert payload["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-04"

    formatted = tsp_price.get_latest_prices_per_fund_dict(date_format="%b %d, %Y")
    assert formatted["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "Jan 04, 2024"

    as_date = tsp_price.get_latest_prices_per_fund_dict(date_format=None)
    assert as_date["funds"][TspIndividualFund.G_FUND.value]["as_of"] == date(2024, 1, 4)

    current_long = tsp_price.get_current_prices_per_fund_long()
    assert_frame_equal(current_long, tsp_price.get_latest_prices_per_fund_long())

    current_dict = tsp_price.get_current_prices_per_fund_dict()
    assert current_dict == tsp_price.get_latest_prices_per_fund_dict()

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_latest_prices_per_fund_dict(date_format="")


def test_price_recency_per_fund_handles_missing_latest() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0, 102.0],
            TspIndividualFund.C_FUND.value: [200.0, np.nan, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    recency = tsp_price.get_price_recency(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert recency.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 3)
    assert recency.loc[TspIndividualFund.C_FUND.value, "as_of"] == date(2024, 1, 1)
    assert recency.loc[TspIndividualFund.G_FUND.value, "days_since"] == 0
    assert recency.loc[TspIndividualFund.C_FUND.value, "days_since"] == 2

    recency_as_of = tsp_price.get_price_recency(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        reference_date=date(2024, 1, 2),
    )
    assert recency_as_of.loc[TspIndividualFund.G_FUND.value, "days_since"] == 0
    assert recency_as_of.loc[TspIndividualFund.C_FUND.value, "days_since"] == 1

    payload = tsp_price.get_price_recency_dict(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )
    assert payload["reference_date"] == "2024-01-03"
    assert payload["funds"][TspIndividualFund.C_FUND.value]["days_since"] == 2

    formatted = tsp_price.get_price_recency_dict(date_format="%b %d, %Y")
    assert formatted["reference_date"] == "Jan 03, 2024"

    as_date = tsp_price.get_price_recency_dict(date_format=None)
    assert as_date["reference_date"] == date(2024, 1, 3)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_price_recency_dict(date_format="")


def test_current_price_status_combines_price_and_recency(
    tsp_price: TspAnalytics,
) -> None:
    funds = [TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    status = tsp_price.get_current_price_status(funds=funds)
    assert status.columns.tolist() == ["as_of", "price", "days_since"]
    assert status.index.tolist() == [fund.value for fund in funds]
    assert status["days_since"].eq(0).all()

    status_long = tsp_price.get_current_price_status_long(funds=funds)
    assert status_long.columns.tolist() == ["fund", "as_of", "price", "days_since"]

    status_dict = tsp_price.get_current_price_status_dict(funds=funds)
    assert status_dict["reference_date"] == "2024-01-04"
    assert status_dict["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-04"

    anchored = tsp_price.get_current_price_status(
        funds=[TspIndividualFund.G_FUND],
        as_of=date(2024, 1, 2),
    )
    assert anchored.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 2)
    assert anchored.loc[TspIndividualFund.G_FUND.value, "days_since"] == 0

    with pytest.raises(ValueError, match="reference_date cannot be earlier than as_of"):
        tsp_price.get_current_price_status(
            funds=[TspIndividualFund.G_FUND],
            as_of=date(2024, 1, 3),
            reference_date=date(2024, 1, 2),
        )


def test_fund_aliases_are_normalized(tsp_price: TspAnalytics) -> None:
    latest_g = tsp_price.get_latest_prices(funds=" g fund ")
    assert latest_g.columns.tolist() == [TspIndividualFund.G_FUND.value]

    latest_l = tsp_price.get_latest_prices(funds="L2030")
    assert latest_l.columns.tolist() == [TspLifecycleFund.L_2030.value]

    latest_short = tsp_price.get_latest_prices(funds="G")
    assert latest_short.columns.tolist() == [TspIndividualFund.G_FUND.value]


def test_fund_metadata_includes_availability_and_aliases(
    tsp_price: TspAnalytics,
) -> None:
    metadata = tsp_price.get_fund_metadata()
    assert "category" in metadata.columns
    assert "aliases" in metadata.columns
    assert "available" in metadata.columns
    assert metadata["available"].all()
    assert metadata.loc[TspIndividualFund.G_FUND.value, "category"] == "individual"

    metadata_dict = tsp_price.get_fund_metadata_dict()
    assert "funds" in metadata_dict
    assert TspLifecycleFund.L_2030.value in metadata_dict["funds"]
    assert metadata_dict["funds"][TspLifecycleFund.L_2030.value]["available"] is True


def test_latest_price_changes_per_fund_handles_missing_latest() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0, 102.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    changes = tsp_price.get_latest_price_changes_per_fund(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND]
    )

    assert changes.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 3)
    assert changes.loc[TspIndividualFund.G_FUND.value, "previous_as_of"] == date(
        2024, 1, 2
    )
    assert changes.loc[TspIndividualFund.G_FUND.value, "latest_price"] == pytest.approx(
        102.0
    )
    assert changes.loc[
        TspIndividualFund.G_FUND.value, "previous_price"
    ] == pytest.approx(101.0)
    assert changes.loc[
        TspIndividualFund.G_FUND.value, "change_percent"
    ] == pytest.approx((102.0 - 101.0) / 101.0)

    assert changes.loc[TspIndividualFund.C_FUND.value, "as_of"] == date(2024, 1, 2)
    assert changes.loc[TspIndividualFund.C_FUND.value, "previous_as_of"] == date(
        2024, 1, 1
    )
    assert changes.loc[TspIndividualFund.C_FUND.value, "latest_price"] == pytest.approx(
        201.0
    )
    assert changes.loc[
        TspIndividualFund.C_FUND.value, "previous_price"
    ] == pytest.approx(200.0)

    current_changes = tsp_price.get_current_price_changes_per_fund()
    assert_frame_equal(current_changes, changes)


def test_latest_price_changes_per_fund_requires_two_points() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02"]),
            TspIndividualFund.G_FUND.value: [100.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="at least two data points are required"):
        tsp_price.get_latest_price_changes_per_fund()


def test_latest_price_changes_per_fund_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_latest_price_changes_per_fund_dict(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert TspIndividualFund.G_FUND.value in payload["funds"]
    assert payload["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-04"
    assert (
        payload["funds"][TspIndividualFund.G_FUND.value]["previous_as_of"]
        == "2024-01-03"
    )

    formatted = tsp_price.get_latest_price_changes_per_fund_dict(
        date_format="%b %d, %Y"
    )
    assert formatted["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "Jan 04, 2024"

    as_date = tsp_price.get_latest_price_changes_per_fund_dict(date_format=None)
    assert as_date["funds"][TspIndividualFund.G_FUND.value]["as_of"] == date(2024, 1, 4)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_latest_price_changes_per_fund_dict(date_format="")


def test_current_price_aliases(tsp_price: TspAnalytics) -> None:
    latest = tsp_price.get_latest_prices()
    current = tsp_price.get_current_prices()
    assert_frame_equal(current, latest)

    latest_long = tsp_price.get_latest_prices_long()
    current_long = tsp_price.get_current_prices_long()
    assert_frame_equal(current_long, latest_long)

    latest_dict = tsp_price.get_latest_prices_dict()
    current_dict = tsp_price.get_current_prices_dict()
    assert current_dict == latest_dict

    latest_changes = tsp_price.get_latest_price_changes()
    current_changes = tsp_price.get_current_price_changes()
    assert_frame_equal(current_changes, latest_changes)

    latest_changes_long = tsp_price.get_latest_price_changes_long()
    current_changes_long = tsp_price.get_current_price_changes_long()
    assert_frame_equal(current_changes_long, latest_changes_long)

    latest_snapshot = tsp_price.get_latest_price_snapshot()
    current_snapshot = tsp_price.get_current_price_snapshot()
    assert_frame_equal(current_snapshot, latest_snapshot)

    latest_snapshot_long = tsp_price.get_latest_price_snapshot_long()
    current_snapshot_long = tsp_price.get_current_price_snapshot_long()
    assert_frame_equal(current_snapshot_long, latest_snapshot_long)

    latest_snapshot_dict = tsp_price.get_latest_price_snapshot_dict()
    current_snapshot_dict = tsp_price.get_current_price_snapshot_dict()
    assert current_snapshot_dict == latest_snapshot_dict

    latest_changes_dict = tsp_price.get_latest_price_changes_dict()
    current_changes_dict = tsp_price.get_current_price_changes_dict()
    assert current_changes_dict == latest_changes_dict

    latest_report = tsp_price.get_latest_price_report_dict()
    current_report = tsp_price.get_current_price_report_dict()
    assert current_report == latest_report


def test_latest_price_report_dict(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_latest_price_report_dict()
    assert report["as_of"] == "2024-01-04"
    assert report["previous_as_of"] == "2024-01-03"
    assert TspIndividualFund.G_FUND.value in report["prices"]
    assert TspIndividualFund.G_FUND.value in report["changes"]

    report_with_quality = tsp_price.get_latest_price_report_dict(
        include_data_quality=True
    )
    assert "data_quality" in report_with_quality
    assert report_with_quality["data_quality"]["summary"]["start_date"] == "2024-01-01"

    fund = TspIndividualFund.G_FUND
    report_fund = tsp_price.get_latest_price_report_dict(fund=fund)
    assert list(report_fund["prices"].keys()) == [fund.value]
    assert list(report_fund["changes"].keys()) == [fund.value]

    as_date = tsp_price.get_latest_price_report_dict(date_format=None)
    assert as_date["as_of"] == date(2024, 1, 4)
    assert as_date["previous_as_of"] == date(2024, 1, 3)

    report_with_cache = tsp_price.get_latest_price_report_dict(
        include_cache_status=True
    )
    assert report_with_cache["cache_status"]["exists"] is False
    assert isinstance(report_with_cache["cache_status"]["last_business_day"], str)

    current_report_with_cache = tsp_price.get_current_price_report_dict(
        include_cache_status=True
    )
    assert current_report_with_cache["cache_status"]["exists"] is False

    report_quality_dates = tsp_price.get_latest_price_report_dict(
        include_data_quality=True,
        date_format=None,
    )
    assert report_quality_dates["data_quality"]["summary"]["start_date"] == date(
        2024, 1, 1
    )

    report_cache_dates = tsp_price.get_latest_price_report_dict(
        include_cache_status=True,
        date_format=None,
    )
    assert isinstance(report_cache_dates["cache_status"]["last_business_day"], date)


def test_current_price_report_as_of(tsp_price: TspAnalytics) -> None:
    as_of_date = date(2024, 1, 3)
    report = tsp_price.get_current_price_report(as_of=as_of_date)
    assert report["as_of"].unique().tolist() == [as_of_date]
    assert report["previous_as_of"].unique().tolist() == [date(2024, 1, 2)]

    report_dict = tsp_price.get_current_price_report_dict(as_of=as_of_date)
    assert report_dict["as_of"] == "2024-01-03"
    assert report_dict["previous_as_of"] == "2024-01-02"
    assert report_dict["requested_as_of"] == "2024-01-03"
    assert TspIndividualFund.G_FUND.value in report_dict["prices"]
    assert TspIndividualFund.G_FUND.value in report_dict["changes"]

    report_dict_dates = tsp_price.get_current_price_report_dict(
        as_of=as_of_date, date_format=None
    )
    assert report_dict_dates["requested_as_of"] == as_of_date


def test_latest_price_report_per_fund(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_latest_price_report_per_fund()
    assert report.columns.tolist() == [
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert report.index.name == "fund"

    per_fund_changes = tsp_price.get_latest_price_changes_per_fund()
    assert_frame_equal(report, per_fund_changes)

    subset = tsp_price.get_latest_price_report_per_fund(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert subset.index.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]

    current = tsp_price.get_current_price_report_per_fund()
    assert_frame_equal(current, report)

    as_of_date = date(2024, 1, 3)
    anchored = tsp_price.get_current_price_report_per_fund(as_of=as_of_date)
    assert anchored.loc[TspIndividualFund.G_FUND.value, "as_of"] == as_of_date
    assert anchored.loc[TspIndividualFund.G_FUND.value, "previous_as_of"] == date(
        2024, 1, 2
    )


def test_latest_price_report_per_fund_long_and_dict(tsp_price: TspAnalytics) -> None:
    report_long = tsp_price.get_latest_price_report_per_fund_long(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert report_long.columns.tolist() == [
        "fund",
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]

    report_dict = tsp_price.get_latest_price_report_per_fund_dict(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert report_dict["as_of"] == "2024-01-04"
    assert report_dict["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-04"

    formatted = tsp_price.get_latest_price_report_per_fund_dict(date_format="%b %d, %Y")
    assert formatted["as_of"] == "Jan 04, 2024"

    as_date = tsp_price.get_latest_price_report_per_fund_dict(date_format=None)
    assert as_date["as_of"] == date(2024, 1, 4)

    current_dict = tsp_price.get_current_price_report_per_fund_dict(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert current_dict == report_dict

    anchored_dict = tsp_price.get_current_price_report_per_fund_dict(
        funds=[TspIndividualFund.G_FUND], as_of=date(2024, 1, 3)
    )
    assert anchored_dict["requested_as_of"] == "2024-01-03"
    assert (
        anchored_dict["funds"][TspIndividualFund.G_FUND.value]["as_of"] == "2024-01-03"
    )


def test_fund_overview_combines_price_changes_and_recency(
    tsp_price: TspAnalytics,
) -> None:
    overview = tsp_price.get_fund_overview(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert overview.columns.tolist() == [
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
        "recency_as_of",
        "days_since",
    ]
    assert overview.loc[TspIndividualFund.G_FUND.value, "days_since"] == 0
    assert overview.loc[TspLifecycleFund.L_2030.value, "recency_as_of"] == date(
        2024, 1, 4
    )

    reference_overview = tsp_price.get_fund_overview(
        funds=[TspIndividualFund.G_FUND],
        reference_date=date(2024, 1, 3),
    )
    assert reference_overview.loc[
        TspIndividualFund.G_FUND.value, "recency_as_of"
    ] == date(2024, 1, 3)
    assert reference_overview.loc[TspIndividualFund.G_FUND.value, "days_since"] == 0


def test_fund_overview_long_and_dict(tsp_price: TspAnalytics) -> None:
    overview_long = tsp_price.get_fund_overview_long(
        funds=[TspIndividualFund.G_FUND],
        reference_date=date(2024, 1, 4),
    )
    assert overview_long.columns.tolist() == [
        "fund",
        "as_of",
        "previous_as_of",
        "recency_as_of",
        "days_since",
        "metric",
        "value",
    ]
    assert overview_long["fund"].unique().tolist() == [TspIndividualFund.G_FUND.value]

    overview_dict = tsp_price.get_fund_overview_dict(
        funds=[TspIndividualFund.G_FUND],
        reference_date=date(2024, 1, 4),
    )
    assert overview_dict["as_of"] == "2024-01-04"
    assert overview_dict["reference_date"] == "2024-01-04"
    assert (
        overview_dict["funds"][TspIndividualFund.G_FUND.value]["recency_as_of"]
        == "2024-01-04"
    )

    overview_as_date = tsp_price.get_fund_overview_dict(
        funds=[TspIndividualFund.G_FUND],
        reference_date=date(2024, 1, 4),
        date_format=None,
    )
    assert overview_as_date["reference_date"] == date(2024, 1, 4)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_fund_overview_dict(date_format="")


def test_current_fund_overview_aliases(tsp_price: TspAnalytics) -> None:
    overview = tsp_price.get_fund_overview()
    current_overview = tsp_price.get_current_fund_overview()
    assert_frame_equal(current_overview, overview)

    overview_long = tsp_price.get_fund_overview_long()
    current_overview_long = tsp_price.get_current_fund_overview_long()
    assert_frame_equal(current_overview_long, overview_long)

    overview_dict = tsp_price.get_fund_overview_dict()
    current_overview_dict = tsp_price.get_current_fund_overview_dict()
    assert current_overview_dict == overview_dict

    overview_dates = tsp_price.get_current_fund_overview_dict(date_format=None)
    assert isinstance(overview_dates["as_of"], date)


def test_current_fund_overview_as_of(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    as_of = date(2024, 1, 3)
    overview = tsp_price.get_current_fund_overview(fund=fund, as_of=as_of)
    assert overview.loc[fund.value, "as_of"] == as_of
    assert overview.loc[fund.value, "previous_as_of"] == date(2024, 1, 2)
    assert overview.loc[fund.value, "days_since"] == 0

    long_overview = tsp_price.get_current_fund_overview_long(fund=fund, as_of=as_of)
    assert long_overview.columns.tolist() == [
        "fund",
        "as_of",
        "previous_as_of",
        "recency_as_of",
        "days_since",
        "metric",
        "value",
    ]

    overview_dict = tsp_price.get_current_fund_overview_dict(fund=fund, as_of=as_of)
    assert overview_dict["requested_as_of"] == "2024-01-03"
    assert overview_dict["funds"][fund.value]["recency_as_of"] == "2024-01-03"

    with pytest.raises(ValueError, match="reference_date cannot be earlier than as_of"):
        tsp_price.get_current_fund_overview(
            fund=fund,
            as_of=date(2024, 1, 3),
            reference_date=date(2024, 1, 2),
        )


def test_latest_price_report_dataframe(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_latest_price_report()
    assert report.columns.tolist() == [
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert report["as_of"].nunique() == 1
    assert report["previous_as_of"].nunique() == 1
    assert report["as_of"].iloc[0] == date(2024, 1, 4)
    assert report["previous_as_of"].iloc[0] == date(2024, 1, 3)

    fund = TspIndividualFund.G_FUND
    report_fund = tsp_price.get_latest_price_report(fund=fund)
    assert report_fund.index.tolist() == [fund.value]

    current_report = tsp_price.get_current_price_report(fund=fund)
    assert_frame_equal(current_report, report_fund)


def test_latest_prices_and_changes_long(tsp_price: TspAnalytics) -> None:
    latest_long = tsp_price.get_latest_prices_long()
    assert latest_long.columns.tolist() == ["Date", "fund", "price"]
    assert latest_long["fund"].nunique() == len(tsp_price.get_available_funds())
    assert latest_long["Date"].nunique() == 1

    fund = TspIndividualFund.G_FUND
    latest_single = tsp_price.get_latest_prices_long(fund=fund)
    assert latest_single["fund"].unique().tolist() == [fund.value]

    changes_long = tsp_price.get_latest_price_changes_long(fund=fund)
    assert changes_long.columns.tolist() == [
        "fund",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert changes_long["fund"].tolist() == [fund.value]

    per_fund_long = tsp_price.get_latest_price_changes_per_fund_long(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert per_fund_long.columns.tolist() == [
        "fund",
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]


def test_latest_prices_dict(tsp_price: TspAnalytics) -> None:
    latest = tsp_price.get_latest_prices_dict()
    assert latest["as_of"] == "2024-01-04"
    assert TspIndividualFund.G_FUND.value in latest["prices"]

    fund = TspIndividualFund.G_FUND
    latest_fund = tsp_price.get_latest_prices_dict(fund=fund)
    assert latest_fund["prices"] == {
        fund.value: pytest.approx(float(tsp_price.current[fund.value]))
    }

    formatted = tsp_price.get_latest_prices_dict(date_format="%Y/%m/%d")
    assert formatted["as_of"] == "2024/01/04"

    as_date = tsp_price.get_latest_prices_dict(date_format=None)
    assert as_date["as_of"] == date(2024, 1, 4)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_latest_prices_dict(date_format="")


def test_latest_prices_dict_includes_missing_values() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    payload = tsp_price.get_latest_prices_dict()
    assert payload["prices"][TspIndividualFund.G_FUND.value] == pytest.approx(101.0)
    assert payload["prices"][TspIndividualFund.C_FUND.value] is None


def test_latest_prices_refreshes_stale_current(tsp_price: TspAnalytics) -> None:
    stale_row = tsp_price.dataframe.iloc[0]
    tsp_price.current = stale_row
    tsp_price.latest = stale_row["Date"].date()

    latest = tsp_price.get_latest_prices()
    assert latest.index[0].date() == date(2024, 1, 4)


def test_latest_price_changes_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_latest_price_changes_dict()
    assert payload["as_of"] == "2024-01-04"
    assert payload["previous_as_of"] == "2024-01-03"
    assert TspIndividualFund.G_FUND.value in payload["funds"]
    assert payload["funds"][TspIndividualFund.G_FUND.value][
        "latest_price"
    ] == pytest.approx(
        float(tsp_price.dataframe[TspIndividualFund.G_FUND.value].iloc[-1])
    )

    formatted = tsp_price.get_latest_price_changes_dict(date_format="%b %d, %Y")
    assert formatted["as_of"] == "Jan 04, 2024"

    as_date = tsp_price.get_latest_price_changes_dict(date_format=None)
    assert as_date["as_of"] == date(2024, 1, 4)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_latest_price_changes_dict(date_format="")


def test_latest_price_changes_dict_handles_zero_previous_price() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [0.0, 1.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    payload = tsp_price.get_latest_price_changes_dict()
    change = payload["funds"][TspIndividualFund.G_FUND.value]
    assert change["change_percent"] is None


def test_get_prices_as_of_requires_available_date(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="no price data available"):
        tsp_price.get_prices_as_of(date(2023, 12, 31))


def test_fund_aliases_and_long_metrics(tsp_price: TspAnalytics) -> None:
    alias_price = tsp_price.get_price("g-fund")
    expected = Decimal(str(tsp_price.current[TspIndividualFund.G_FUND.value]))
    assert alias_price == expected

    short_price = tsp_price.get_price("G")
    assert short_price == expected

    lifecycle_price = tsp_price.get_price("L2050")
    expected_lifecycle = Decimal(str(tsp_price.current[TspLifecycleFund.L_2050.value]))
    assert lifecycle_price == expected_lifecycle

    metrics_long = tsp_price.get_price_history_with_metrics_long(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert metrics_long.columns.tolist() == [
        "Date",
        "fund",
        "price",
        "return",
        "cumulative_return",
        "normalized_price",
    ]
    assert metrics_long["fund"].nunique() == 2


def test_latest_price_snapshot(tsp_price: TspAnalytics) -> None:
    snapshot = tsp_price.get_latest_price_snapshot()
    assert snapshot.index.name == "fund"
    assert snapshot["as_of"].nunique() == 1
    assert snapshot["as_of"].iloc[0] == date(2024, 1, 4)

    fund = TspIndividualFund.G_FUND
    latest_value = float(tsp_price.dataframe[fund.value].iloc[-1])
    previous_value = float(tsp_price.dataframe[fund.value].iloc[-2])
    expected_change = latest_value - previous_value
    expected_change_percent = expected_change / previous_value

    fund_snapshot = tsp_price.get_latest_price_snapshot(fund=fund)
    assert fund_snapshot.index.tolist() == [fund.value]
    assert fund_snapshot.loc[fund.value, "latest_price"] == pytest.approx(latest_value)
    assert fund_snapshot.loc[fund.value, "previous_price"] == pytest.approx(
        previous_value
    )
    assert fund_snapshot.loc[fund.value, "change"] == pytest.approx(expected_change)
    assert fund_snapshot.loc[fund.value, "change_percent"] == pytest.approx(
        expected_change_percent
    )

    snapshot_long = tsp_price.get_latest_price_snapshot_long(fund=fund)
    assert snapshot_long.columns.tolist() == [
        "fund",
        "as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert snapshot_long["fund"].unique().tolist() == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_latest_price_snapshot(
            fund=fund,
            funds=[fund],
        )

    snapshot_dict = tsp_price.get_latest_price_snapshot_dict(fund=fund)
    assert snapshot_dict["as_of"] == "2024-01-04"
    assert snapshot_dict["funds"][fund.value]["latest_price"] == pytest.approx(
        latest_value
    )
    assert snapshot_dict["funds"][fund.value]["previous_price"] == pytest.approx(
        previous_value
    )

    formatted = tsp_price.get_latest_price_snapshot_dict(
        fund=fund, date_format="%b %d, %Y"
    )
    assert formatted["as_of"] == "Jan 04, 2024"

    as_date = tsp_price.get_latest_price_snapshot_dict(fund=fund, date_format=None)
    assert as_date["as_of"] == date(2024, 1, 4)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_latest_price_snapshot_dict(fund=fund, date_format=" ")


def test_price_changes_as_of(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 3)
    fund = TspIndividualFund.G_FUND
    changes = tsp_price.get_price_changes_as_of(as_of, fund=fund)
    expected_latest = float(tsp_price.dataframe.loc[2, fund.value])
    expected_previous = float(tsp_price.dataframe.loc[1, fund.value])
    assert changes.loc[fund.value, "latest_price"] == pytest.approx(expected_latest)
    assert changes.loc[fund.value, "previous_price"] == pytest.approx(expected_previous)
    assert changes.loc[fund.value, "change"] == pytest.approx(
        expected_latest - expected_previous
    )

    changes_long = tsp_price.get_price_changes_as_of_long(as_of, fund=fund)
    assert changes_long.columns.tolist() == [
        "fund",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert changes_long["fund"].tolist() == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_price_changes_as_of(as_of, fund=fund, funds=[fund])

    with pytest.raises(
        ValueError, match="no price data available on or before the requested date"
    ):
        tsp_price.get_price_changes_as_of(date(2023, 12, 31))

    with pytest.raises(
        ValueError,
        match="at least two data points are required to calculate price changes",
    ):
        tsp_price.get_price_changes_as_of(date(2024, 1, 1))


def test_price_changes_as_of_dict(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 3)
    payload = tsp_price.get_price_changes_as_of_dict(
        as_of,
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030],
    )
    assert payload["as_of"] == "2024-01-03"
    assert payload["previous_as_of"] == "2024-01-02"
    assert TspIndividualFund.G_FUND.value in payload["funds"]

    formatted = tsp_price.get_price_changes_as_of_dict(
        as_of, fund=TspIndividualFund.G_FUND, date_format=None
    )
    assert formatted["as_of"] == as_of

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_price_changes_as_of_dict(
            as_of, fund=TspIndividualFund.G_FUND, date_format=""
        )


def test_price_snapshot_as_of(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 3)
    fund = TspIndividualFund.G_FUND
    snapshot = tsp_price.get_price_snapshot_as_of(as_of, fund=fund)
    assert snapshot.loc[fund.value, "as_of"] == as_of
    assert "change_percent" in snapshot.columns

    snapshot_long = tsp_price.get_price_snapshot_as_of_long(as_of, fund=fund)
    assert snapshot_long.columns.tolist() == [
        "fund",
        "as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
    ]
    assert snapshot_long["fund"].tolist() == [fund.value]

    snapshot_dict = tsp_price.get_price_snapshot_as_of_dict(as_of, fund=fund)
    assert snapshot_dict["as_of"] == "2024-01-03"
    assert snapshot_dict["funds"][fund.value]["latest_price"] == pytest.approx(
        float(tsp_price.dataframe.loc[2, fund.value])
    )

    formatted = tsp_price.get_price_snapshot_as_of_dict(
        as_of, fund=fund, date_format="%b %d, %Y"
    )
    assert formatted["as_of"] == "Jan 03, 2024"

    as_date = tsp_price.get_price_snapshot_as_of_dict(
        as_of, fund=fund, date_format=None
    )
    assert as_date["as_of"] == as_of


def test_prices_as_of(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 2)
    prices = tsp_price.get_prices_as_of(as_of)
    assert prices.index[0].date() == as_of

    after_prices = tsp_price.get_prices_as_of(date(2024, 1, 5))
    assert after_prices.index[0].date() == date(2024, 1, 4)

    fund = TspIndividualFund.G_FUND
    fund_prices = tsp_price.get_prices_as_of(as_of, fund=fund)
    assert fund_prices.columns.tolist() == [fund.value]

    subset_prices = tsp_price.get_prices_as_of(
        as_of, funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert subset_prices.columns.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_prices_as_of(as_of, fund=fund, funds=[fund])

    with pytest.raises(
        ValueError, match="no price data available on or before the requested date"
    ):
        tsp_price.get_prices_as_of(date(2023, 12, 31))


def test_price_as_of(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    price = tsp_price.get_price_as_of(fund, date(2024, 1, 2))
    expected = Decimal(str(tsp_price.dataframe.loc[1, fund.value]))
    assert price == expected

    later = tsp_price.get_price_as_of(fund, date(2024, 1, 5))
    expected_latest = Decimal(str(tsp_price.dataframe.loc[3, fund.value]))
    assert later == expected_latest

    with pytest.raises(
        ValueError, match="no price data available on or before the requested date"
    ):
        tsp_price.get_price_as_of(fund, date(2023, 12, 31))


def test_prices_as_of_dict(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 2)
    payload = tsp_price.get_prices_as_of_dict(as_of)
    assert payload["requested_as_of"] == "2024-01-02"
    assert payload["as_of"] == "2024-01-02"
    assert TspIndividualFund.G_FUND.value in payload["prices"]

    formatted = tsp_price.get_prices_as_of_dict(as_of, date_format="%Y/%m/%d")
    assert formatted["requested_as_of"] == "2024/01/02"
    assert formatted["as_of"] == "2024/01/02"

    as_date = tsp_price.get_prices_as_of_dict(as_of, date_format=None)
    assert as_date["requested_as_of"] == as_of
    assert as_date["as_of"] == as_of


def test_prices_as_of_per_fund(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 2)
    per_fund = tsp_price.get_prices_as_of_per_fund(as_of)
    assert per_fund["as_of"].unique().tolist() == [as_of]

    per_fund_long = tsp_price.get_prices_as_of_per_fund_long(as_of)
    assert per_fund_long.columns.tolist() == ["fund", "as_of", "price"]

    later = tsp_price.get_prices_as_of_per_fund(date(2024, 1, 5))
    assert later["as_of"].unique().tolist() == [date(2024, 1, 4)]

    subset = tsp_price.get_prices_as_of_per_fund(
        as_of, funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
    )
    assert subset.index.tolist() == [
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]

    with pytest.raises(
        ValueError, match="no price data available on or before the requested date"
    ):
        tsp_price.get_prices_as_of_per_fund(date(2023, 12, 31))


def test_prices_as_of_per_fund_dict(tsp_price: TspAnalytics) -> None:
    as_of = date(2024, 1, 2)
    payload = tsp_price.get_prices_as_of_per_fund_dict(as_of)
    assert payload["requested_as_of"] == "2024-01-02"
    assert TspIndividualFund.G_FUND.value in payload["funds"]

    formatted = tsp_price.get_prices_as_of_per_fund_dict(as_of, date_format="%Y/%m/%d")
    assert formatted["requested_as_of"] == "2024/01/02"

    as_date = tsp_price.get_prices_as_of_per_fund_dict(as_of, date_format=None)
    assert as_date["requested_as_of"] == as_of


def test_prices_as_of_long(tsp_price: TspAnalytics) -> None:
    as_of_long = tsp_price.get_prices_as_of_long(date(2024, 1, 2))
    assert as_of_long.columns.tolist() == ["Date", "fund", "price"]
    assert as_of_long["Date"].nunique() == 1

    fund = TspIndividualFund.G_FUND
    fund_long = tsp_price.get_prices_as_of_long(date(2024, 1, 2), fund=fund)
    assert fund_long["fund"].unique().tolist() == [fund.value]


def test_latest_price_filters_require_available_funds() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="funds not available in data"):
        tsp_price.get_latest_prices(
            funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
        )

    with pytest.raises(ValueError, match="funds not available in data"):
        tsp_price.get_latest_price_changes(
            funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030]
        )


def test_latest_price_changes_sorts_by_date() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    shuffled = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    tsp_price.dataframe = shuffled
    tsp_price.current = shuffled.loc[shuffled["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    changes = tsp_price.get_latest_price_changes(fund=TspIndividualFund.G_FUND)
    latest_value = dataframe.sort_values("Date").iloc[-1][
        TspIndividualFund.G_FUND.value
    ]
    assert changes.loc[TspIndividualFund.G_FUND.value, "latest_price"] == pytest.approx(
        latest_value
    )


def test_price_history_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        tsp_price.get_price_history(
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 1),
        )

    missing_fund_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    missing_fund_price.dataframe = dataframe
    missing_fund_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    missing_fund_price.latest = missing_fund_price.current["Date"].date()
    missing_fund_price.check = lambda: None
    with pytest.raises(ValueError, match="funds not available in data"):
        missing_fund_price.get_price_history(funds=[TspLifecycleFund.L_2030])

    history = tsp_price.get_price_history(funds=[" g fund "])
    assert history.columns.tolist() == ["Date", TspIndividualFund.G_FUND.value]


def test_price_history_long_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_price_history_long_dict(
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030],
        date_format="%Y-%m-%d",
    )
    records = payload["prices"]
    assert records[0]["Date"] == "2024-01-01"
    assert set(records[0].keys()) == {"Date", "fund", "price"}

    as_date = tsp_price.get_price_history_long_dict(date_format=None)
    assert isinstance(as_date["prices"][0]["Date"], date)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_price_history_long_dict(date_format="")


def test_resolve_fund_accepts_aliases() -> None:
    tsp_price = TspAnalytics()

    assert tsp_price._resolve_fund("g") == TspIndividualFund.G_FUND.value
    assert tsp_price._resolve_fund("g-fund") == TspIndividualFund.G_FUND.value
    assert tsp_price._resolve_fund("L2050") == TspLifecycleFund.L_2050.value
    assert tsp_price._resolve_fund("L Income Fund") == TspLifecycleFund.L_INCOME.value


def test_get_available_funds_excludes_all_nan_columns() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspLifecycleFund.L_2030.value: [np.nan, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    assert tsp_price.get_available_funds() == [TspIndividualFund.G_FUND.value]
    latest = tsp_price.get_latest_prices()
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]
    changes = tsp_price.get_latest_price_changes()
    assert changes.index.tolist() == [TspIndividualFund.G_FUND.value]


def test_latest_prices_require_available_funds() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [np.nan, np.nan],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="no available funds in data"):
        tsp_price.get_latest_prices()

    with pytest.raises(ValueError, match="no available funds in data"):
        tsp_price.get_latest_price_changes()

    with pytest.raises(ValueError, match="no available funds in data"):
        tsp_price.get_latest_prices_per_fund()


def test_get_price_change_by_date_range(tsp_price: TspAnalytics) -> None:
    start_date = date(2024, 1, 1)
    end_date = date(2024, 1, 4)
    changes = tsp_price.get_price_change_by_date_range(start_date, end_date)

    fund = TspIndividualFund.G_FUND.value
    start_price = tsp_price.dataframe.loc[0, fund]
    end_price = tsp_price.dataframe.loc[3, fund]
    assert changes.loc[fund, "start_price"] == pytest.approx(start_price)
    assert changes.loc[fund, "end_price"] == pytest.approx(end_price)
    assert changes.loc[fund, "change"] == pytest.approx(end_price - start_price)

    fund_change = tsp_price.get_price_change_by_date_range(
        start_date, end_date, fund=TspIndividualFund.G_FUND
    )
    assert fund_change.index.tolist() == [TspIndividualFund.G_FUND.value]

    long_changes = tsp_price.get_price_change_by_date_range_long(start_date, end_date)
    assert long_changes.columns.tolist() == [
        "fund",
        "start_price",
        "end_price",
        "change",
        "change_percent",
    ]

    fund_long = tsp_price.get_price_change_by_date_range_long(
        start_date, end_date, fund=TspIndividualFund.G_FUND
    )
    assert fund_long["fund"].unique().tolist() == [TspIndividualFund.G_FUND.value]

    payload = tsp_price.get_price_change_by_date_range_dict(start_date, end_date)
    assert payload["start_date"] == "2024-01-01"
    assert payload["end_date"] == "2024-01-04"
    assert TspIndividualFund.G_FUND.value in payload["funds"]

    fund_payload = tsp_price.get_price_change_by_date_range_dict(
        start_date,
        end_date,
        fund=TspIndividualFund.G_FUND,
        date_format=None,
    )
    assert fund_payload["start_date"] == start_date
    assert fund_payload["end_date"] == end_date
    assert list(fund_payload["funds"].keys()) == [TspIndividualFund.G_FUND.value]

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_price_change_by_date_range_dict(
            start_date,
            end_date,
            date_format="",
        )


def test_get_price_change_by_date_range_empty(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="no price data available"):
        tsp_price.get_price_change_by_date_range(date(2023, 1, 1), date(2023, 1, 2))


def test_get_price_history_validates_inputs(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        tsp_price.get_price_history(
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 1),
        )

    with pytest.raises(ValueError, match="unknown fund"):
        tsp_price.get_price_history(funds=["Unknown Fund"])

    with pytest.raises(
        ValueError, match="funds must be an iterable of fund enums or fund name strings"
    ):
        tsp_price.get_price_history(funds=123)

    with pytest.raises(ValueError, match="funds must contain at least one fund"):
        tsp_price.get_price_history(funds=[])


def test_get_price_history_filters_funds(tsp_price: TspAnalytics) -> None:
    history = tsp_price.get_price_history(
        funds=[TspIndividualFund.C_FUND, TspLifecycleFund.L_2030],
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )
    assert history.columns.tolist() == [
        "Date",
        TspIndividualFund.C_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]
    assert len(history) == 2

    history_long = tsp_price.get_price_history_long(
        funds=[TspIndividualFund.C_FUND, TspLifecycleFund.L_2030],
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )
    assert history_long.columns.tolist() == ["Date", "fund", "price"]
    assert set(history_long["fund"].unique()) == {
        TspIndividualFund.C_FUND.value,
        TspLifecycleFund.L_2030.value,
    }


def test_fund_inputs_accept_string(tsp_price: TspAnalytics) -> None:
    latest = tsp_price.get_latest_prices(funds="G Fund")
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]

    history = tsp_price.get_price_history(funds="G Fund")
    assert history.columns.tolist() == ["Date", TspIndividualFund.G_FUND.value]

    price = tsp_price.get_price("G Fund")
    assert price == Decimal(str(tsp_price.current[TspIndividualFund.G_FUND.value]))

    price_on_date = tsp_price.get_fund_price_by_date("G Fund", date(2024, 1, 2))
    assert price_on_date == Decimal(
        str(tsp_price.dataframe.loc[1, TspIndividualFund.G_FUND.value])
    )

    monthly = tsp_price.get_fund_prices_by_month("G Fund", 2024, 1)
    assert len(monthly) == 4

    monthly_price = TspAnalytics()
    monthly_df = build_monthly_price_dataframe()
    monthly_price.dataframe = monthly_df
    monthly_price.current = monthly_df.loc[monthly_df["Date"].idxmax()]
    monthly_price.latest = monthly_price.current["Date"].date()
    monthly_price.check = lambda: None

    monthly_table = monthly_price.get_monthly_return_table("G Fund")
    assert monthly_table.index.tolist() == [2024]


def test_fund_inputs_accept_case_insensitive_strings(tsp_price: TspAnalytics) -> None:
    latest = tsp_price.get_latest_prices(funds=" g fund ")
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]

    history = tsp_price.get_price_history(funds=["c fund", "l 2030"])
    assert history.columns.tolist() == [
        "Date",
        TspIndividualFund.C_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]

    portfolio = tsp_price.get_portfolio_returns(
        weights={" g fund ": 0.5, "C FUND": 0.5}
    )
    assert "portfolio_return" in portfolio.columns

    allocation = tsp_price.create_allocation_from_shares({" g fund ": 1, "C Fund": 2})
    g_price = Decimal(str(tsp_price.current[TspIndividualFund.G_FUND.value]))
    assert allocation[TspIndividualFund.G_FUND.value]["subtotal"] == f"{g_price:,.2f}"

    returns = tsp_price.get_daily_returns(fund=" c fund ")
    assert returns.columns.tolist() == [TspIndividualFund.C_FUND.value]


def test_fund_inputs_accept_aliases(tsp_price: TspAnalytics) -> None:
    latest = tsp_price.get_latest_prices(funds="g")
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]

    latest = tsp_price.get_latest_prices(funds="g-fund")
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]

    latest = tsp_price.get_latest_prices(funds="g_fund")
    assert latest.columns.tolist() == [TspIndividualFund.G_FUND.value]

    latest = tsp_price.get_latest_prices(funds="l2050")
    assert latest.columns.tolist() == [TspLifecycleFund.L_2050.value]

    latest = tsp_price.get_latest_prices(funds="l-2060")
    assert latest.columns.tolist() == [TspLifecycleFund.L_2060.value]

    latest = tsp_price.get_latest_prices(funds="L_2075")
    assert latest.columns.tolist() == [TspLifecycleFund.L_2075.value]


def test_missing_fund_validation(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    dataframe = tsp_price.dataframe.drop(columns=[fund.value])
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    with pytest.raises(ValueError, match="fund not available in data"):
        tsp_price.get_price(fund)

    with pytest.raises(ValueError, match="fund not available in data"):
        tsp_price.get_daily_returns(fund=fund)

    with pytest.raises(ValueError, match="fund not available in data"):
        tsp_price.get_latest_prices(fund=fund)


def test_price_history_rejects_missing_fund(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    dataframe = tsp_price.dataframe.drop(columns=[fund.value])
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()

    with pytest.raises(ValueError, match="funds not available in data"):
        tsp_price.get_price_history(funds=[fund])


def test_latest_price_changes_requires_two_rows(tsp_price: TspAnalytics) -> None:
    single_row = tsp_price.dataframe.head(1)
    tsp_price.dataframe = single_row
    tsp_price.current = single_row.iloc[0]
    tsp_price.latest = tsp_price.current["Date"].date()

    with pytest.raises(ValueError, match="at least two data points"):
        tsp_price.get_latest_price_changes()


def test_latest_prices_and_changes_refreshes_current(tsp_price: TspAnalytics) -> None:
    stale_row = tsp_price.dataframe.iloc[0]
    tsp_price.current = stale_row
    tsp_price.latest = stale_row["Date"].date()

    latest_prices = tsp_price.get_latest_prices()
    assert latest_prices.index[0].date() == date(2024, 1, 4)


def test_download_csv_sets_user_agent() -> None:
    csv_text = "Date,G Fund\n2024-01-02,100.0\n"
    session = DummySession([DummyResponse(csv_text.encode("utf-8"))], headers=None)
    tsp_price = TspAnalytics(
        auto_update=False,
        session=session,
        csv_url="https://example.com/fund-price-history.csv",
        max_retries=1,
        user_agent="MyAgent/1.0",
    )
    content, _ = tsp_price._download_csv_content()
    assert "Date" in content
    assert session.headers["User-Agent"] == "MyAgent/1.0"
