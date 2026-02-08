import pytest

from tsp import TspIndividualFund, TspAnalytics


def test_price_recency_and_overview(tsp_price: TspAnalytics) -> None:
    recency = tsp_price.get_price_recency()
    assert recency["days_since"].eq(0).all()

    overview = tsp_price.get_fund_overview()
    expected_columns = {
        "as_of",
        "previous_as_of",
        "latest_price",
        "previous_price",
        "change",
        "change_percent",
        "recency_as_of",
        "days_since",
    }
    assert expected_columns.issubset(set(overview.columns))


def test_price_recency_accepts_single_fund(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    recency = tsp_price.get_price_recency(fund=fund)
    assert recency.index.tolist() == [fund.value]

    recency_long = tsp_price.get_price_recency_long(fund=fund)
    assert recency_long["fund"].tolist() == [fund.value]

    recency_dict = tsp_price.get_price_recency_dict(fund=fund)
    assert list(recency_dict["funds"].keys()) == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_price_recency(fund=fund, funds=[fund])


def test_current_price_status_accepts_single_fund(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    status = tsp_price.get_current_price_status(fund=fund)
    assert status.index.tolist() == [fund.value]

    status_dict = tsp_price.get_current_price_status_dict(fund=fund)
    assert list(status_dict["funds"].keys()) == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_current_price_status(fund=fund, funds=[fund])


def test_current_price_alerts_accept_single_fund(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    alerts = tsp_price.get_current_price_alerts(
        fund=fund, stale_days=0, change_threshold=0.05
    )
    assert alerts.index.tolist() == [fund.value]

    alerts_dict = tsp_price.get_current_price_alerts_dict(
        fund=fund, stale_days=0, change_threshold=0.05
    )
    assert list(alerts_dict["funds"].keys()) == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_current_price_alerts(fund=fund, funds=[fund])


def test_fund_overview_accepts_single_fund(tsp_price: TspAnalytics) -> None:
    fund = TspIndividualFund.G_FUND
    overview = tsp_price.get_fund_overview(fund=fund)
    assert overview.index.tolist() == [fund.value]

    overview_dict = tsp_price.get_fund_overview_dict(fund=fund)
    assert list(overview_dict["funds"].keys()) == [fund.value]

    with pytest.raises(ValueError, match="fund and funds cannot both be provided"):
        tsp_price.get_fund_overview(fund=fund, funds=[fund])
