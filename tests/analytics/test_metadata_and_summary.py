from datetime import date

import pytest

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics


def test_fund_aliases_and_metadata(tsp_price: TspAnalytics) -> None:
    aliases = tsp_price.get_fund_aliases()
    assert "g" in aliases[TspIndividualFund.G_FUND.value]
    assert "l2050" in aliases[TspLifecycleFund.L_2050.value]

    metadata = tsp_price.get_fund_metadata()
    assert metadata.loc[TspIndividualFund.G_FUND.value, "category"] == "individual"
    assert metadata.loc[TspLifecycleFund.L_2050.value, "category"] == "lifecycle"
    assert metadata["available"].all()


def test_price_summary_helpers(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_price_summary(funds=[TspIndividualFund.G_FUND])
    row = summary.loc[TspIndividualFund.G_FUND.value]
    assert row["first_date"] == date(2024, 1, 1)
    assert row["last_date"] == date(2024, 1, 4)
    assert row["start_price"] == pytest.approx(10.0)
    assert row["end_price"] == pytest.approx(13.0)
    assert row["min_price"] == pytest.approx(10.0)
    assert row["max_price"] == pytest.approx(13.0)
    assert row["mean_price"] == pytest.approx(11.5)
    assert row["median_price"] == pytest.approx(11.5)
    assert row["std_price"] == pytest.approx(1.290994, rel=1e-6)
    assert row["total_return"] == pytest.approx(0.3)

    summary_dict = tsp_price.get_price_summary_dict(funds=[TspIndividualFund.G_FUND])
    payload = summary_dict["funds"][TspIndividualFund.G_FUND.value]
    assert payload["first_date"] == "2024-01-01"
    assert payload["last_date"] == "2024-01-04"
    assert payload["total_return"] == pytest.approx(0.3)


def test_drawdown_summary_dict(tsp_price: TspAnalytics) -> None:
    payload = tsp_price.get_drawdown_summary_dict(TspIndividualFund.G_FUND)
    summary = payload["funds"][TspIndividualFund.G_FUND.value]
    assert summary["max_drawdown"] == pytest.approx(0.0)
    assert summary["peak_date"] == "2024-01-01"


def test_price_statistics_dict(tsp_price: TspAnalytics) -> None:
    stats = tsp_price.get_price_statistics_dict(fund=TspIndividualFund.G_FUND)
    payload = stats["statistics"][TspIndividualFund.G_FUND.value]
    assert payload["count"] == pytest.approx(4.0)
    assert payload["min"] == pytest.approx(10.0)
    assert payload["max"] == pytest.approx(13.0)
    assert payload["median"] == pytest.approx(11.5)


def test_data_summary(tsp_price: TspAnalytics) -> None:
    summary = tsp_price.get_data_summary()
    assert summary["start_date"] == date(2024, 1, 1)
    assert summary["end_date"] == date(2024, 1, 4)
    assert summary["total_rows"] == 4
    assert summary["expected_business_days"] == 4
    assert summary["missing_business_days"] == 0


def test_statistics_support_single_date_bound(tsp_price: TspAnalytics) -> None:
    price_stats_start = tsp_price.get_price_statistics(start_date=date(2024, 1, 3))
    price_stats_end = tsp_price.get_price_statistics(end_date=date(2024, 1, 2))

    assert price_stats_start.loc[TspIndividualFund.G_FUND.value, "count"] == 2
    assert price_stats_end.loc[TspIndividualFund.G_FUND.value, "count"] == 2

    return_stats_start = tsp_price.get_return_statistics(start_date=date(2024, 1, 3))
    return_stats_end = tsp_price.get_return_statistics(end_date=date(2024, 1, 2))

    assert return_stats_start.loc[TspIndividualFund.G_FUND.value, "count"] == 2
    assert return_stats_end.loc[TspIndividualFund.G_FUND.value, "count"] == 1
