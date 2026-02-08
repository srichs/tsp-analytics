from datetime import date

import pytest

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics


def test_get_recent_prices_defaults(tsp_price: TspAnalytics) -> None:
    recent = tsp_price.get_recent_prices(days=2)
    assert recent["Date"].dt.date.tolist() == [date(2024, 1, 3), date(2024, 1, 4)]
    assert TspIndividualFund.G_FUND.value in recent.columns


def test_get_recent_prices_as_of_anchor(tsp_price: TspAnalytics) -> None:
    recent = tsp_price.get_recent_prices(days=2, as_of=date(2024, 1, 3))
    assert recent["Date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]


def test_get_recent_prices_fund_filters(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    recent = tsp_price.get_recent_prices(days=3, fund=fund)
    assert recent.columns.tolist() == ["Date", fund.value]

    recent_funds = tsp_price.get_recent_prices(
        days=3,
        funds=[TspIndividualFund.G_FUND, TspLifecycleFund.L_2030],
    )
    assert recent_funds.columns.tolist() == [
        "Date",
        TspIndividualFund.G_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]


def test_get_recent_prices_long_and_dict(tsp_price: TspAnalytics) -> None:
    recent_long = tsp_price.get_recent_prices_long(
        days=2, fund=TspIndividualFund.G_FUND
    )
    assert recent_long.columns.tolist() == ["Date", "fund", "price"]
    assert recent_long["fund"].unique().tolist() == [TspIndividualFund.G_FUND.value]

    payload = tsp_price.get_recent_prices_dict(days=2, as_of=date(2024, 1, 3))
    assert payload["start_date"] == "2024-01-02"
    assert payload["end_date"] == "2024-01-03"
    assert payload["requested_as_of"] == "2024-01-03"
    assert payload["days"] == 2
    assert {record["fund"] for record in payload["prices"]} == {
        fund.value for fund in TspIndividualFund
    } | {fund.value for fund in TspLifecycleFund}


def test_get_recent_prices_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="days must be a positive integer"):
        tsp_price.get_recent_prices(days=0)

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_recent_prices(
            days=2,
            fund=TspIndividualFund.G_FUND,
            funds=[TspIndividualFund.G_FUND],
        )

    with pytest.raises(
        ValueError, match="no price data available for the requested period"
    ):
        tsp_price.get_recent_prices(days=2, as_of=date(2023, 12, 31))
