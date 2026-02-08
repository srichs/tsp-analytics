import os
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics
from tests.helpers import (
    DummyResponse,
    DummySession,
    build_minimal_price_dataframe,
)


def test_load_csv_text_and_dataframe() -> None:
    dataframe = build_minimal_price_dataframe()
    csv_text = dataframe.to_csv(index=False)

    client = TspAnalytics(auto_update=False)
    client.load_csv_text(csv_text)
    assert client.latest == date(2024, 1, 3)
    assert client.get_available_funds() == [
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    ]

    client.load_dataframe(dataframe)
    assert client.latest == date(2024, 1, 3)

    with pytest.raises(TypeError, match="csv_content must be a string"):
        client.load_csv_text(123)

    with pytest.raises(ValueError, match="csv_content must be a non-empty string"):
        client.load_csv_text("   ")

    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        client.load_dataframe("not a dataframe")


def test_load_dataframe_rejects_negative_prices() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, -101.0],
        }
    )
    with pytest.raises(ValueError, match="negative values"):
        tsp_price.load_dataframe(dataframe)


def test_load_dataframe_accepts_date_index() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    dataframe = pd.DataFrame(
        {
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0],
        },
        index=dates,
    )
    tsp_price.load_dataframe(dataframe)
    assert "Date" in tsp_price.dataframe.columns
    assert tsp_price.latest == date(2024, 1, 3)


def test_load_dataframe_accepts_named_date_index() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    index = pd.Index(["2024-01-02", "2024-01-03"], name="Date")
    dataframe = pd.DataFrame(
        {
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
        },
        index=index,
    )
    tsp_price.load_dataframe(dataframe)
    assert "Date" in tsp_price.dataframe.columns
    assert tsp_price.latest == date(2024, 1, 3)


def test_load_dataframe_rejects_missing_date_column() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
        }
    )
    with pytest.raises(ValueError, match="must include a Date column"):
        tsp_price.load_dataframe(dataframe)


def test_load_csv_normalizes_column_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "tsp_data.csv"
    df = pd.DataFrame(
        {
            " date ": ["2024-01-02"],
            "g-fund": [100.0],
            "C Fund": [200.0],
        }
    )
    df.to_csv(csv_path, index=False)

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    tsp_price.load_csv(csv_path)
    assert "Date" in tsp_price.dataframe.columns
    assert TspIndividualFund.G_FUND.value in tsp_price.dataframe.columns
    assert TspIndividualFund.C_FUND.value in tsp_price.dataframe.columns


def test_load_csv_text_normalizes_data() -> None:
    csv_content = " date ,g-fund,C Fund\n2024-01-02,100.0,200.0\n"
    tsp_price = TspAnalytics(auto_update=False)
    tsp_price.load_csv_text(csv_content)
    assert "Date" in tsp_price.dataframe.columns
    assert TspIndividualFund.G_FUND.value in tsp_price.dataframe.columns
    assert tsp_price.dataframe.loc[0, TspIndividualFund.C_FUND.value] == pytest.approx(
        200.0
    )


def test_load_csv_text_validates_input() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    with pytest.raises(ValueError, match="csv_content must be a non-empty string"):
        tsp_price.load_csv_text("   ")
    with pytest.raises(TypeError, match="csv_content must be a string or bytes"):
        tsp_price.load_csv_text(123)


def test_load_csv_text_accepts_bytes() -> None:
    csv_content = "Date,G Fund\n2024-01-02,100.0\n".encode("utf-8")
    tsp_price = TspAnalytics(auto_update=False)
    tsp_price.load_csv_text(csv_content)
    assert "Date" in tsp_price.dataframe.columns
    assert tsp_price.dataframe.loc[0, TspIndividualFund.G_FUND.value] == pytest.approx(
        100.0
    )


def test_data_dir_requires_non_empty_value() -> None:
    with pytest.raises(ValueError, match="data_dir must be a non-empty directory path"):
        TspAnalytics(data_dir=" ")


def test_load_dataframe_requires_dataframe() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        tsp_price.load_dataframe(["not", "a", "dataframe"])


def test_load_csv_requires_existing_file(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    with pytest.raises(ValueError, match="csv filepath must point to an existing file"):
        tsp_price.load_csv(tmp_path / "missing.csv")


def test_cache_status_reports_file_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "fund-price-history.csv"
    df = build_minimal_price_dataframe()
    df.to_csv(csv_path, index=False)

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    status = tsp_price.get_cache_status()

    assert status["exists"] is True
    assert status["file_size_bytes"] is not None
    assert status["data_start_date"] == date(2024, 1, 2)
    assert status["data_end_date"] == date(2024, 1, 3)
    assert TspIndividualFund.G_FUND.value in status["available_funds"]


def test_download_csv_content_retries_and_sets_user_agent(
    monkeypatch, tmp_path: Path
) -> None:
    csv_content = "Date,G Fund\n2024-01-02,100.0\n".encode("utf-8")
    responses = [
        RuntimeError("network error"),
        DummyResponse(csv_content, encoding=None),
    ]
    session = DummySession(responses, headers=None)
    sleep_calls = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("tsp.tsp.time_module.sleep", fake_sleep)
    tsp_price = TspAnalytics(
        auto_update=False,
        session=session,
        max_retries=2,
        retry_backoff=0.1,
        data_dir=tmp_path,
    )
    content, _ = tsp_price._download_csv_content()
    assert "Date" in content
    assert sleep_calls == [0.1]
    assert session.headers["User-Agent"] == TspAnalytics.USER_AGENT_STR


def test_download_csv_content_uses_apparent_encoding(tmp_path: Path) -> None:
    csv_text = "Date,G Fund\n2024-01-02,100.0\n"
    csv_bytes = csv_text.encode("utf-16")
    response = DummyResponse(csv_bytes, encoding=None, apparent_encoding="utf-16")
    session = DummySession([response], headers={})
    tsp_price = TspAnalytics(auto_update=False, session=session, data_dir=tmp_path)
    content, _ = tsp_price._download_csv_content()
    assert "Date" in content


def test_data_quality_report_includes_expected_sections(
    tsp_price: TspAnalytics,
) -> None:
    report = tsp_price.get_data_quality_report()
    assert set(report.keys()) == {
        "summary",
        "fund_coverage",
        "missing_business_days",
        "cache_status",
    }


def test_load_dataframe_normalizes_columns_basic() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "g_fund": [100.0, 101.0],
            "c fund": [200.0, 201.0],
        }
    )
    tsp_price.load_dataframe(df)
    assert "Date" in tsp_price.dataframe.columns
    assert TspIndividualFund.G_FUND.value in tsp_price.dataframe.columns
    assert TspIndividualFund.C_FUND.value in tsp_price.dataframe.columns
    assert tsp_price.latest == date(2024, 1, 3)


def test_load_csv_requires_file(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False)
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="csv filepath must point to an existing file"):
        tsp_price.load_csv(missing_file)


def test_load_dataframe_and_csv(tmp_path: Path) -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = build_minimal_price_dataframe()
    prices.load_dataframe(dataframe)
    assert prices.latest == dataframe["Date"].max().date()
    assert prices.get_price(TspIndividualFund.G_FUND) == Decimal("101.0")

    csv_path = tmp_path / "fund-price-history.csv"
    dataframe.to_csv(csv_path, index=False)
    prices.load_csv(csv_path)
    assert prices.latest == dataframe["Date"].max().date()

    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        prices.load_dataframe(["not", "a", "dataframe"])  # type: ignore[list-item]


def test_load_dataframe_normalizes_columns_trims_spacing() -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            " date ": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "g_fund": [100.0, 101.0],
            "C-FUND": [200.0, 201.0],
        }
    )
    prices.load_dataframe(dataframe)
    assert prices.get_available_funds() == [
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    ]


def test_update_skips_older_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = build_minimal_price_dataframe()
    csv_path = tmp_path / TspAnalytics.CSV_FILENAME
    existing.to_csv(csv_path, index=False)

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    tsp_price.load_dataframe(existing)

    older = existing.copy()
    older["Date"] = pd.to_datetime(["2024-01-01", "2024-01-02"])
    payload = older.to_csv(index=False).encode("utf-8")

    class FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.encoding = None

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, timeout: float) -> FakeResponse:
            return FakeResponse(payload)

    monkeypatch.setattr("tsp.tsp.Session", FakeSession)

    tsp_price._update()

    refreshed = pd.read_csv(csv_path)
    assert pd.to_datetime(refreshed["Date"]).max().date() == date(2024, 1, 3)
    assert tsp_price.latest == date(2024, 1, 3)


def test_data_quality_report(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_data_quality_report(include_cache_status=False)
    assert "summary" in report
    assert "fund_coverage" in report
    assert "missing_business_days" in report
    assert report["summary"]["total_rows"] == len(tsp_price.dataframe)


def test_normalize_dataframe_handles_bom_date_column() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "\ufeffDate": ["2024-01-02"],
            TspIndividualFund.G_FUND.value: [100.0],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert "Date" in normalized.columns
    assert normalized["Date"].iloc[0].date() == date(2024, 1, 2)


def test_normalize_dataframe_coerces_timezone_and_time() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": ["2024-01-02T15:30:00Z", "2024-01-03 08:15:00+00:00"],
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert normalized["Date"].iloc[0].hour == 0
    assert normalized["Date"].iloc[1].hour == 0
    prices = TspAnalytics(auto_update=False)
    prices.load_dataframe(dataframe)
    assert len(prices.get_prices_by_date(date(2024, 1, 2))) == 1


def test_get_cache_status_without_file(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    tsp_price.check = lambda: None

    status = tsp_price.get_cache_status()

    assert status["exists"] is False
    assert status["file_size_bytes"] is None
    assert status["dataframe_valid"] is False
    assert status["validation_error"] is None
    assert status["latest_data_date"] is None
    assert status["data_start_date"] is None
    assert status["data_end_date"] is None
    assert status["data_span_days"] is None
    assert status["cache_age_days"] is None
    assert status["data_age_days"] is None
    assert status["is_stale"] is None
    assert status["stale_by_days"] is None
    assert status["last_business_day"] == tsp_price._get_last_business_day(date.today())
    assert status["total_rows"] == 0
    assert status["available_funds"] == []


def test_get_cache_status_with_file(tmp_path: Path) -> None:
    dataframe = build_minimal_price_dataframe()
    csv_path = tmp_path / TspAnalytics.CSV_FILENAME
    dataframe.to_csv(csv_path, index=False)

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    tsp_price.check = lambda: None

    status = tsp_price.get_cache_status()

    assert status["exists"] is True
    assert status["csv_filepath"] == str(csv_path)
    assert status["last_updated"] is not None
    assert status["file_size_bytes"] == csv_path.stat().st_size
    assert status["dataframe_valid"] is True
    assert status["validation_error"] is None
    assert status["latest_data_date"] == date(2024, 1, 3)
    assert status["data_start_date"] == date(2024, 1, 2)
    assert status["data_end_date"] == date(2024, 1, 3)
    assert status["data_span_days"] == 1
    assert status["cache_age_days"] is not None
    assert status["data_age_days"] is not None
    assert status["total_rows"] == len(dataframe)
    assert TspIndividualFund.G_FUND.value in status["available_funds"]
    assert TspLifecycleFund.L_2030.value in status["missing_funds"]


def test_get_cache_status_dict(tmp_path: Path) -> None:
    dataframe = build_minimal_price_dataframe()
    csv_path = tmp_path / TspAnalytics.CSV_FILENAME
    dataframe.to_csv(csv_path, index=False)

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    tsp_price.check = lambda: None

    status = tsp_price.get_cache_status_dict()
    assert status["exists"] is True
    assert status["csv_filepath"] == str(csv_path)
    assert isinstance(status["last_updated"], str)
    assert status["latest_data_date"] == "2024-01-03"
    assert status["data_start_date"] == "2024-01-02"
    assert status["data_end_date"] == "2024-01-03"

    as_date = tsp_price.get_cache_status_dict(date_format=None, datetime_format=None)
    assert as_date["latest_data_date"] == date(2024, 1, 3)
    assert isinstance(as_date["last_updated"], datetime)

    with pytest.raises(
        ValueError, match="date_format must be a non-empty string or None"
    ):
        tsp_price.get_cache_status_dict(date_format="")

    with pytest.raises(
        ValueError, match="datetime_format must be a non-empty string or None"
    ):
        tsp_price.get_cache_status_dict(datetime_format="")


def test_get_cache_status_age_and_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataframe = build_minimal_price_dataframe()
    csv_path = tmp_path / TspAnalytics.CSV_FILENAME
    dataframe.to_csv(csv_path, index=False)
    fixed_updated = pd.Timestamp("2024-01-04 12:00:00").timestamp()
    Path(csv_path).touch()
    Path(csv_path).chmod(0o644)
    Path(csv_path).touch()
    os.utime(csv_path, (fixed_updated, fixed_updated))

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2024, 1, 5)

    monkeypatch.setattr("tsp.analytics.date", FixedDate)

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    tsp_price.check = lambda: None

    status = tsp_price.get_cache_status()

    assert status["cache_age_days"] == 1
    assert status["data_age_days"] == 2
    assert status["is_stale"] is True
    assert status["stale_by_days"] == 2
    assert status["last_business_day"] == date(2024, 1, 5)


def test_get_cache_status_invalid_cache(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    csv_path = tmp_path / TspAnalytics.CSV_FILENAME
    csv_path.write_text("not,a,valid,csv\n1,2,3,4\n")
    tsp_price.check = lambda: None

    status = tsp_price.get_cache_status()

    assert status["exists"] is True
    assert status["dataframe_valid"] is False
    assert status["validation_error"] is not None


def test_data_quality_report_dict(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_data_quality_report_dict()
    assert set(report.keys()) == {
        "summary",
        "fund_coverage",
        "missing_business_days",
        "cache_status",
    }
    assert report["summary"]["start_date"] == "2024-01-01"
    assert isinstance(report["fund_coverage"], list)
    assert report["missing_business_days"] == []

    scoped = tsp_price.get_data_quality_report_dict(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
        include_cache_status=False,
        date_format=None,
    )
    assert "cache_status" not in scoped
    assert scoped["summary"]["start_date"] == date(2024, 1, 1)


def test_get_data_quality_report(tsp_price: TspAnalytics) -> None:
    report = tsp_price.get_data_quality_report()
    assert report["summary"]["total_rows"] == len(tsp_price.dataframe)
    assert report["fund_coverage"].index.name == "fund"
    assert report["missing_business_days"].empty is True
    assert "cache_status" in report

    scoped = tsp_price.get_data_quality_report(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
        include_cache_status=False,
    )
    assert "cache_status" not in scoped


def test_get_missing_business_days() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0, 102.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    missing = tsp_price.get_missing_business_days()
    missing_dates = missing["Date"].dt.date.tolist()
    assert missing_dates == [date(2024, 1, 2), date(2024, 1, 4)]

    scoped_missing = tsp_price.get_missing_business_days(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )
    assert scoped_missing["Date"].dt.date.tolist() == [date(2024, 1, 2)]

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_missing_business_days(start_date=date(2024, 1, 1))


def test_get_missing_business_days_empty() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0, 102.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    missing = tsp_price.get_missing_business_days()
    assert missing.empty


def test_ensure_dataframe_raises_without_data() -> None:
    tsp_price = TspAnalytics()
    tsp_price.dataframe = None
    tsp_price.current = None
    tsp_price.latest = None
    tsp_price.check = lambda: None

    with pytest.raises(RuntimeError, match="TSP price data is not available"):
        tsp_price.get_latest_prices()


def test_set_values_validates_required_columns(tmp_path: Path) -> None:
    tsp_price = TspAnalytics()
    missing_date_path = tmp_path / "missing_date.csv"
    missing_date_path.write_text("G Fund\n10\n")
    tsp_price.csv_filepath = str(missing_date_path)
    with pytest.raises(ValueError, match="Date column"):
        tsp_price._set_values()

    missing_fund_path = tmp_path / "missing_fund.csv"
    missing_fund_path.write_text("Date,Other\n2024-01-01,10\n")
    tsp_price.csv_filepath = str(missing_fund_path)
    with pytest.raises(ValueError, match="fund column"):
        tsp_price._set_values()


def test_set_values_returns_false_when_missing(tmp_path: Path) -> None:
    tsp_price = TspAnalytics()
    tsp_price.csv_filepath = str(tmp_path / "missing.csv")
    assert tsp_price._set_values() is False


def test_normalize_dataframe_coerces_and_drops_invalid_rows() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            TspIndividualFund.G_FUND.value: ["bad", "10"],
            TspIndividualFund.F_FUND.value: ["bad", "20"],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert len(normalized) == 1
    assert normalized.iloc[0]["Date"].date() == date(2024, 1, 2)


def test_normalize_dataframe_deduplicates_dates() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            TspIndividualFund.G_FUND.value: [10, 20, 30],
            TspIndividualFund.F_FUND.value: [40, 50, 60],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert len(normalized) == 2
    deduped_price = normalized.loc[
        normalized["Date"].apply(lambda value: value.date()) == date(2024, 1, 1),
        TspIndividualFund.G_FUND.value,
    ].iloc[0]
    assert deduped_price == 20


def test_normalize_dataframe_drops_unknown_columns() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            TspIndividualFund.G_FUND.value: [10, 11],
            "Extra Column": [1, 2],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert "Extra Column" not in normalized.columns


def test_normalize_dataframe_standardizes_column_names() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            " date ": ["2024-01-01", "2024-01-02"],
            " g fund ": [10, 11],
            "C FUND": [20, 21],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert "Date" in normalized.columns
    assert TspIndividualFund.G_FUND.value in normalized.columns
    assert TspIndividualFund.C_FUND.value in normalized.columns


def test_normalize_dataframe_orders_fund_columns() -> None:
    tsp_price = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "C Fund": [200.0, 201.0],
            "G Fund": [100.0, 101.0],
            "L 2030": [300.0, 301.0],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert normalized.columns.tolist() == [
        "Date",
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
        TspLifecycleFund.L_2030.value,
    ]


def test_normalize_dataframe_parses_common_date_formats() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": [" 01/02/2024 ", "2024-01-03"],
            TspIndividualFund.G_FUND.value: [10, 11],
            TspIndividualFund.F_FUND.value: [20, 21],
        }
    )
    normalized = tsp_price._normalize_dataframe(dataframe)
    assert normalized["Date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]


def test_get_available_funds_and_summary(tsp_price: TspAnalytics) -> None:
    funds = tsp_price.get_available_funds()
    expected_funds = [fund.value for fund in TspIndividualFund] + [
        fund.value for fund in TspLifecycleFund
    ]
    assert set(funds) == set(expected_funds)

    summary = tsp_price.get_data_summary()
    assert summary["start_date"] == date(2024, 1, 1)
    assert summary["end_date"] == date(2024, 1, 4)
    assert summary["total_rows"] == 4
    assert summary["missing_funds"] == []
    assert summary["expected_business_days"] == 4
    assert summary["missing_business_days"] == 0
    assert summary["business_day_coverage"] == pytest.approx(1.0)

    coverage = tsp_price.get_fund_coverage_summary()
    assert coverage.index.name == "fund"
    assert coverage.loc[
        TspIndividualFund.G_FUND.value, "coverage_percent"
    ] == pytest.approx(1.0)


def test_get_data_summary_counts_missing_business_days() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    summary = tsp_price.get_data_summary()
    assert summary["expected_business_days"] == 4
    assert summary["missing_business_days"] == 2
    assert summary["business_day_coverage"] == pytest.approx(0.5)

    summary_dict = tsp_price.get_data_summary_dict()
    assert summary_dict["start_date"] == "2024-01-01"
    assert summary_dict["end_date"] == "2024-01-04"
    assert summary_dict["missing_business_days"] == 2
    assert summary_dict["business_day_coverage"] == pytest.approx(0.5)


def test_get_fund_coverage_summary_with_missing_funds() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    coverage = tsp_price.get_fund_coverage_summary()
    assert coverage.loc[TspIndividualFund.G_FUND.value, "available_rows"] == 2
    assert coverage.loc[TspIndividualFund.G_FUND.value, "missing_rows"] == 0
    assert coverage.loc[TspLifecycleFund.L_2030.value, "available_rows"] == 0
    assert coverage.loc[TspLifecycleFund.L_2030.value, "missing_rows"] == 2
    assert coverage.loc[TspLifecycleFund.L_2030.value, "coverage_percent"] == 0.0


def test_data_quality_and_cache_status(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0],
        }
    )
    tsp_price.load_dataframe(dataframe)

    report = tsp_price.get_data_quality_report(include_cache_status=False)
    assert set(report.keys()) == {"summary", "fund_coverage", "missing_business_days"}
    assert report["summary"]["available_funds"] == [
        TspIndividualFund.G_FUND.value,
        TspIndividualFund.C_FUND.value,
    ]

    coverage = report["fund_coverage"]
    assert coverage.loc[TspIndividualFund.G_FUND.value, "available_rows"] == 2

    missing = tsp_price.get_missing_business_days(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
    )
    assert missing["Date"].dt.date.tolist() == [date(2024, 1, 3)]

    cache_status = tsp_price.get_cache_status()
    assert cache_status["exists"] is False
    assert cache_status["data_start_date"] == date(2024, 1, 2)
    assert cache_status["data_end_date"] == date(2024, 1, 4)


def test_data_dir_sets_csv_path(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(data_dir=tmp_path)
    assert Path(tsp_price.csv_filepath).parent == tmp_path


def test_data_dir_rejects_file_path(tmp_path: Path) -> None:
    data_file = tmp_path / "fund-price-history.csv"
    data_file.write_text("Date,G Fund\n2024-01-01,10\n")
    with pytest.raises(ValueError, match="data_dir must be a directory path"):
        TspAnalytics(data_dir=data_file)


def test_request_timeout_validation() -> None:
    with pytest.raises(ValueError, match="request_timeout must be a positive value"):
        TspAnalytics(request_timeout=0)

    with pytest.raises(ValueError, match="request_timeout must be a positive value"):
        TspAnalytics(request_timeout=True)

    tsp_price = TspAnalytics(auto_update=False, request_timeout=np.float64(15.0))
    assert tsp_price.request_timeout == 15.0


def test_retry_settings_validation() -> None:
    with pytest.raises(ValueError, match="max_retries must be a positive integer"):
        TspAnalytics(max_retries=0)

    with pytest.raises(ValueError, match="max_retries must be a positive integer"):
        TspAnalytics(max_retries=True)

    with pytest.raises(ValueError, match="retry_backoff must be a non-negative value"):
        TspAnalytics(retry_backoff=-0.1)


def test_time_hour_validation() -> None:
    with pytest.raises(ValueError, match="time_hour must be a datetime.time value"):
        TspAnalytics(time_hour="19:00")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="time_hour must be a datetime.time value"):
        TspAnalytics(time_hour=19)  # type: ignore[arg-type]

    tsp_price = TspAnalytics(auto_update=False, time_hour=time(hour=18))
    assert tsp_price.time_hour == time(hour=18)


def test_env_data_dir_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TSP_DATA_DIR", str(tmp_path))
    tsp_price = TspAnalytics()
    assert Path(tsp_price.csv_filepath).parent == tmp_path


def test_refresh_invokes_update(monkeypatch: pytest.MonkeyPatch) -> None:
    tsp_price = TspAnalytics()
    called = {"count": 0}

    def _fake_update():
        called["count"] += 1

    monkeypatch.setattr(tsp_price, "_update", _fake_update)
    tsp_price.refresh()
    assert called["count"] == 1


def test_check_skips_auto_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "fund-price-history.csv"
    csv_path.write_text("Date,G Fund\n2024-01-01,10\n")
    tsp_price = TspAnalytics(data_dir=tmp_path, auto_update=False)

    def _fake_update():
        raise AssertionError("update should not be called when auto_update is False")

    monkeypatch.setattr(tsp_price, "_update", _fake_update)
    tsp_price.check()


def test_check_refreshes_invalid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsp_price = TspAnalytics(data_dir=tmp_path, auto_update=True)
    csv_path = Path(tsp_price.csv_filepath)
    csv_path.write_text("not,a,valid,csv\n1,2,3,4\n")

    def _fake_update() -> None:
        dataframe = build_minimal_price_dataframe()
        normalized = tsp_price._normalize_dataframe(dataframe)
        tsp_price._assign_dataframe(normalized)

    monkeypatch.setattr(tsp_price, "_update", _fake_update)
    tsp_price.check()

    assert tsp_price.dataframe is not None
    assert tsp_price.latest == date(2024, 1, 3)


def test_check_invalid_cache_raises_when_auto_update_disabled(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(data_dir=tmp_path, auto_update=False)
    csv_path = Path(tsp_price.csv_filepath)
    csv_path.write_text("not,a,valid,csv\n1,2,3,4\n")

    with pytest.raises(ValueError, match="Date column"):
        tsp_price.check()


def test_update_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_content = "Date,G Fund\n2024-01-01,10\n2024-01-02,11\n"

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.encoding = None

        def raise_for_status(self) -> None:
            return None

    class FlakySession:
        attempts = 0

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self) -> "FlakySession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, timeout: float) -> DummyResponse:
            FlakySession.attempts += 1
            if FlakySession.attempts < 2:
                raise RuntimeError("temporary failure")
            return DummyResponse(csv_content.encode("utf-8"))

    monkeypatch.setattr("tsp.tsp.Session", lambda: FlakySession())

    tsp_price = TspAnalytics(
        auto_update=False, data_dir=tmp_path, max_retries=2, retry_backoff=0.0
    )
    tsp_price._update()

    assert FlakySession.attempts == 2
    assert tsp_price.latest == date(2024, 1, 2)


def test_update_downloads_and_sets_dataframe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_content = "Date,G Fund,F Fund\n2024-01-01,10,20\n2024-01-02,11,21\n"
    record: dict[str, float] = {}

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.encoding = None

        def raise_for_status(self) -> None:
            return None

    class DummySession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, timeout: float) -> DummyResponse:
            record["timeout"] = timeout
            record["url"] = url
            return DummyResponse(csv_content.encode("utf-8"))

    monkeypatch.setattr("tsp.tsp.Session", lambda: DummySession())

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path, request_timeout=7.5)
    tsp_price._update()

    csv_path = tmp_path / TspAnalytics.CSV_FILENAME
    assert csv_path.exists()
    assert record["timeout"] == 7.5
    assert record["url"] == tsp_price.csv_url
    assert tsp_price.latest == date(2024, 1, 2)
    assert tsp_price.dataframe is not None
    assert tsp_price.dataframe.columns.tolist() == ["Date", "G Fund", "F Fund"]


def test_update_uses_custom_session(tmp_path: Path) -> None:
    csv_content = "Date,G Fund\n2024-01-01,10\n2024-01-02,11\n"
    record: dict[str, float] = {}

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.encoding = None

        def raise_for_status(self) -> None:
            return None

    class CustomSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self) -> "CustomSession":
            raise AssertionError(
                "custom session should not be used as a context manager"
            )

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, timeout: float) -> DummyResponse:
            record["url"] = url
            record["timeout"] = timeout
            return DummyResponse(csv_content.encode("utf-8"))

    session = CustomSession()
    tsp_price = TspAnalytics(
        auto_update=False, data_dir=tmp_path, request_timeout=5.0, session=session
    )
    tsp_price._update()

    assert record["url"] == tsp_price.csv_url
    assert record["timeout"] == 5.0
    assert session.headers["User-Agent"] == TspAnalytics.USER_AGENT_STR
    assert tsp_price.latest == date(2024, 1, 2)


def test_csv_url_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_url = "https://example.com/custom.csv"
    record: dict[str, float] = {}

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.encoding = None

        def raise_for_status(self) -> None:
            return None

    class DummySession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, timeout: float) -> DummyResponse:
            record["url"] = url
            return DummyResponse(b"Date,G Fund\n2024-01-01,10\n")

    monkeypatch.setattr("tsp.tsp.Session", lambda: DummySession())

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path, csv_url=custom_url)
    tsp_price._update()

    assert record["url"] == custom_url


def test_csv_url_validation() -> None:
    with pytest.raises(ValueError, match="csv_url must be a non-empty string"):
        TspAnalytics(csv_url="")

    with pytest.raises(ValueError, match="csv_url must be a non-empty string"):
        TspAnalytics(csv_url="   ")

    with pytest.raises(ValueError, match="csv_url must be a non-empty string"):
        TspAnalytics(csv_url=123)
