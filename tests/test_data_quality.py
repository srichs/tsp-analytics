from __future__ import annotations

from datetime import date

import pandas as pd

from tsp import TspIndividualFund, TspAnalytics


def _build_sparse_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0],
        }
    )


def test_data_summary_and_coverage() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = _build_sparse_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    summary = tsp_price.get_data_summary()

    assert summary["start_date"] == date(2024, 1, 1)
    assert summary["end_date"] == date(2024, 1, 3)
    assert summary["expected_business_days"] == 3
    assert summary["missing_business_days"] == 1
    assert summary["business_day_coverage"] == 2 / 3
    assert TspIndividualFund.G_FUND.value in summary["available_funds"]
    assert TspIndividualFund.S_FUND.value in summary["missing_funds"]

    summary_dict = tsp_price.get_data_summary_dict()
    assert summary_dict["start_date"] == "2024-01-01"
    assert summary_dict["end_date"] == "2024-01-03"
    assert summary_dict["business_day_coverage"] == 2 / 3


def test_missing_business_days_report() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = _build_sparse_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    missing = tsp_price.get_missing_business_days()

    assert missing["Date"].dt.date.tolist() == [date(2024, 1, 2)]

    missing_dict = tsp_price.get_missing_business_days_dict()
    assert missing_dict == [{"date": "2024-01-02"}]


def test_data_quality_report_dict_includes_cache_status() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = _build_sparse_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    report = tsp_price.get_data_quality_report_dict()

    assert "summary" in report
    assert "fund_coverage" in report
    assert "missing_business_days" in report
    assert "cache_status" in report
    assert report["cache_status"]["latest_data_date"] == "2024-01-03"
