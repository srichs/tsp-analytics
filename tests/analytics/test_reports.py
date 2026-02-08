from datetime import date

import pytest

from tsp import TspIndividualFund, TspAnalytics


def test_fund_analytics_report_bundles_metrics(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_fund_analytics_report(
        TspIndividualFund.G_FUND, start_date=date(2024, 1, 2)
    )
    expected_keys = {
        "price_statistics",
        "return_statistics",
        "performance_summary",
        "drawdown_summary",
        "price_summary",
        "current_overview",
        "date_range",
    }
    assert expected_keys.issubset(report.keys())
    date_range = report["date_range"].loc["date_range"]
    assert date_range["start_date"] == date(2024, 1, 2)
    assert date_range["end_date"] == date(2024, 1, 4)
    assert TspIndividualFund.G_FUND.value in report["price_statistics"].index

    payload = tsp_price.get_fund_analytics_report_dict(
        TspIndividualFund.G_FUND, start_date=date(2024, 1, 2), date_format="iso"
    )
    assert payload["fund"] == TspIndividualFund.G_FUND.value
    assert payload["date_range"]["start_date"] == "2024-01-02"
    assert payload["date_range"]["end_date"] == "2024-01-04"
    assert payload["price_statistics"][TspIndividualFund.G_FUND.value][
        "count"
    ] == pytest.approx(3.0)


def test_current_price_dashboard_includes_snapshot_and_risk(
    tsp_price: TspAnalytics,
) -> None:
    dashboard = tsp_price.get_current_price_dashboard(periods=[1, 2])
    assert "latest_price" in dashboard.columns
    assert "change_percent" in dashboard.columns
    assert "days_since" in dashboard.columns
    assert "trailing_return_1d" in dashboard.columns
    assert "trailing_return_2d" in dashboard.columns
    assert "annualized_return" in dashboard.columns
    assert "annualized_volatility" in dashboard.columns

    payload = tsp_price.get_current_price_dashboard_dict(
        periods=[1, 2], date_format="iso"
    )
    assert payload["periods"] == [1, 2]
    assert payload["trading_days"] == 252
    assert TspIndividualFund.G_FUND.value in payload["funds"]
