from datetime import date

import pytest

from tsp import TspIndividualFund, TspAnalytics


def test_current_price_summary_reference_date(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_current_price_summary(
        reference_date=date(2024, 1, 3), stale_days=0
    )
    row = summary.loc["summary"]
    assert row["as_of"] == date(2024, 1, 3)
    assert row["reference_date"] == date(2024, 1, 3)
    assert row["average_change"] == pytest.approx(1.0)


def test_current_price_summary(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_current_price_summary(stale_days=0)
    row = summary.loc["summary"]
    total_funds = len(tsp_price.get_available_funds())
    assert row["total_funds"] == total_funds
    assert row["stale_funds"] == 0
    assert row["positive_changes"] == total_funds
    assert row["negative_changes"] == 0
    assert row["unchanged_changes"] == 0
    assert row["max_days_since"] == 0

    summary_dict = tsp_price.get_current_price_summary_dict(stale_days=0)
    assert summary_dict["as_of"] == "2024-01-04"
    assert summary_dict["reference_date"] == "2024-01-04"
    assert summary_dict["stale_funds"] == 0


def test_current_price_summary_as_of(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_current_price_summary(as_of=date(2024, 1, 3), stale_days=0)
    row = summary.loc["summary"]
    assert row["as_of"] == date(2024, 1, 3)
    assert row["requested_as_of"] == date(2024, 1, 3)
    assert row["reference_date"] == date(2024, 1, 3)

    summary_dict = tsp_price.get_current_price_summary_dict(
        as_of=date(2024, 1, 3), stale_days=0
    )
    assert summary_dict["as_of"] == "2024-01-03"
    assert summary_dict["requested_as_of"] == "2024-01-03"
    assert summary_dict["reference_date"] == "2024-01-03"


def test_current_price_alerts(tsp_price: TspAnalytics) -> None:
    alerts = tsp_price.get_current_price_alerts(stale_days=0, change_threshold=0.05)
    assert {
        "price",
        "change",
        "change_percent",
        "days_since",
        "is_stale",
        "is_large_move",
    }.issubset(alerts.columns)
    assert alerts["is_stale"].sum() == 0
    assert alerts.loc[TspIndividualFund.G_FUND.value, "is_large_move"]

    as_of_alerts = tsp_price.get_current_price_alerts(
        as_of=date(2024, 1, 3), reference_date=date(2024, 1, 4), stale_days=0
    )
    assert as_of_alerts.loc[TspIndividualFund.G_FUND.value, "as_of"] == date(2024, 1, 3)
    assert as_of_alerts.loc[TspIndividualFund.G_FUND.value, "days_since"] == 1

    payload = tsp_price.get_current_price_alerts_dict(
        stale_days=0, change_threshold=0.05
    )
    assert payload["reference_date"] == "2024-01-04"
    assert payload["stale_threshold_days"] == 0
    assert payload["change_threshold"] == pytest.approx(0.05)
    assert payload["funds"][TspIndividualFund.G_FUND.value]["is_large_move"]


def test_current_price_alert_summary(tsp_price: TspAnalytics) -> None:
    alerts = tsp_price.get_current_price_alerts(stale_days=0, change_threshold=0.05)
    summary = tsp_price.get_current_price_alert_summary(
        stale_days=0, change_threshold=0.05
    )
    row = summary.loc["summary"]
    assert row["total_funds"] == len(alerts)
    assert row["stale_funds"] == int(alerts["is_stale"].sum())
    assert row["large_move_funds"] == int(alerts["is_large_move"].sum())
    assert row["stale_and_large_funds"] == int(
        (alerts["is_stale"] & alerts["is_large_move"]).sum()
    )

    summary_dict = tsp_price.get_current_price_alert_summary_dict(
        stale_days=0, change_threshold=0.05
    )
    assert summary_dict["reference_date"] == "2024-01-04"
    assert summary_dict["stale_threshold_days"] == 0
