from decimal import Decimal

import pytest

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics
from tests.helpers import build_minimal_price_dataframe


def test_create_allocation_uses_latest_prices(tsp_price: TspAnalytics) -> None:
    stale_row = tsp_price.dataframe.iloc[0]
    tsp_price.current = stale_row
    tsp_price.latest = stale_row["Date"].date()

    allocation = tsp_price.create_allocation(g_shares=1)
    latest_row = tsp_price.dataframe.iloc[-1]
    assert allocation["date"] == str(latest_row["Date"].date())
    assert allocation[TspIndividualFund.G_FUND.value]["price"] == str(
        latest_row[TspIndividualFund.G_FUND.value]
    )


def test_create_allocation_with_missing_funds() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    allocation = tsp_price.create_allocation(g_shares=1, c_shares=2)
    assert allocation[TspIndividualFund.G_FUND.value]["price"] == "101.0"
    assert allocation[TspLifecycleFund.L_2030.value]["price"] is None
    assert allocation[TspLifecycleFund.L_2030.value]["subtotal"] == "0.00"
    assert allocation[TspLifecycleFund.L_2030.value]["percent"] == "0.00"

    with pytest.raises(ValueError, match="fund not available in data: L 2030"):
        tsp_price.create_allocation(l_2030_shares=1)


def test_create_allocation_from_shares() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    allocation = tsp_price.create_allocation_from_shares(
        {
            TspIndividualFund.G_FUND: 1,
            "C Fund": 2.5,
        }
    )
    assert allocation[TspIndividualFund.G_FUND.value]["subtotal"] == "101.00"
    assert allocation[TspIndividualFund.C_FUND.value]["subtotal"] == "502.50"

    with pytest.raises(ValueError, match="shares must be a non-empty mapping"):
        tsp_price.create_allocation_from_shares({})

    with pytest.raises(ValueError, match="unknown fund"):
        tsp_price.create_allocation_from_shares({"Unknown Fund": 1})

    with pytest.raises(
        ValueError, match="shares must map fund enums or fund name strings"
    ):
        tsp_price.create_allocation_from_shares({123: 1})


def test_create_allocation_calculations(tsp_price: TspAnalytics) -> None:
    allocation = tsp_price.create_allocation(g_shares=1, f_shares=2)

    g_price = Decimal(str(tsp_price.current[TspIndividualFund.G_FUND.value]))
    f_price = Decimal(str(tsp_price.current[TspIndividualFund.F_FUND.value]))
    expected_total = g_price + (f_price * 2)

    g_percent = (g_price / expected_total) * Decimal(100)
    f_percent = (f_price * 2 / expected_total) * Decimal(100)

    assert allocation["total"] == f"{expected_total:,.2f}"
    assert allocation[TspIndividualFund.G_FUND.value]["percent"] == f"{g_percent:.2f}"
    assert allocation[TspIndividualFund.F_FUND.value]["percent"] == f"{f_percent:.2f}"
    assert allocation["allocation_percent"] == {
        TspIndividualFund.G_FUND.value: f"{g_percent:.2f}",
        TspIndividualFund.F_FUND.value: f"{f_percent:.2f}",
    }


def test_create_allocation_rejects_negative_values(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="g_shares must be a non-negative value"):
        tsp_price.create_allocation(g_shares=-1)
