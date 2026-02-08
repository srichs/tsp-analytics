from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from tsp import TspAnalytics
from tsp import data_io as data_io_module
from tests.helpers import DummyResponse, DummySession


def test_download_csv_content_sets_default_headers(tmp_path: Path) -> None:
    csv_payload = "Date,G Fund\n2024-01-02,100.0\n"
    session = DummySession([DummyResponse(csv_payload.encode("utf-8"))], headers=None)
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path, session=session)

    content, _ = tsp_price._download_csv_content()

    assert "Date" in content
    assert session.headers["User-Agent"] == tsp_price.user_agent
    assert session.headers["Accept"] == "text/csv"


def test_download_csv_content_retries_with_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_payload = "Date,G Fund\n2024-01-02,100.0\n"
    session = DummySession(
        [Exception("boom"), DummyResponse(csv_payload.encode("utf-8"))],
        headers={},
    )
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(data_io_module.time_module, "sleep", fake_sleep)
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        session=session,
        max_retries=2,
        retry_backoff=0.1,
    )

    content, _ = tsp_price._download_csv_content()

    assert "Date" in content
    assert sleep_calls == [0.1]


def test_download_csv_content_raises_after_retries(tmp_path: Path) -> None:
    session = DummySession([Exception("boom"), Exception("boom")], headers={})
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        session=session,
        max_retries=2,
        retry_backoff=0,
    )

    with pytest.raises(Exception, match="boom"):
        tsp_price._download_csv_content()


def test_download_csv_content_rejects_html(tmp_path: Path) -> None:
    html_payload = "<html><body>Service unavailable</body></html>"
    session = DummySession([DummyResponse(html_payload.encode("utf-8"))], headers={})
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        session=session,
        max_retries=1,
        retry_backoff=0,
    )

    with pytest.raises(ValueError, match="does not look like a CSV"):
        tsp_price._download_csv_content()


def test_download_csv_content_requires_date_header(tmp_path: Path) -> None:
    csv_payload = "Fund,Value\nG Fund,100.0\n"
    session = DummySession([DummyResponse(csv_payload.encode("utf-8"))], headers={})
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        session=session,
        max_retries=1,
        retry_backoff=0,
    )

    with pytest.raises(ValueError, match="Date column"):
        tsp_price._download_csv_content()


def test_download_csv_content_requires_fund_columns(tmp_path: Path) -> None:
    csv_payload = "Date,NotAFund\n2024-01-02,100.0\n"
    session = DummySession([DummyResponse(csv_payload.encode("utf-8"))], headers={})
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        session=session,
        max_retries=1,
        retry_backoff=0,
    )

    with pytest.raises(ValueError, match="known fund columns"):
        tsp_price._download_csv_content()


def test_download_csv_content_accepts_sep_header(tmp_path: Path) -> None:
    csv_payload = "sep=,\nDate,G Fund\n2024-01-02,100.0\n"
    session = DummySession([DummyResponse(csv_payload.encode("utf-8"))], headers={})
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        session=session,
        max_retries=1,
        retry_backoff=0,
    )

    content, _ = tsp_price._download_csv_content()

    assert "Date" in content


def test_load_csv_text_accepts_bytes(tmp_path: Path) -> None:
    csv_payload = b"Date,G Fund,C Fund\n2024-01-02,100.0,200.0\n"
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)

    tsp_price.load_csv_text(csv_payload)

    assert tsp_price.dataframe is not None


def test_load_csv_text_rejects_missing_fund_columns(tmp_path: Path) -> None:
    csv_payload = "Date,NotAFund\n2024-01-02,100.0\n"
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)

    with pytest.raises(ValueError, match="known fund columns"):
        tsp_price.load_csv_text(csv_payload)


def test_download_csv_content_respects_etag_metadata(tmp_path: Path) -> None:
    csv_payload = "Date,G Fund\n2024-01-02,100.0\n"
    response = DummyResponse(
        csv_payload.encode("utf-8"),
        status_code=304,
        headers={"ETag": "abc123"},
    )
    session = DummySession([response], headers={})
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path, session=session)
    metadata_path = Path(tsp_price._metadata_filepath())
    metadata_path.write_text(
        '{"csv_url": "https://www.tsp.gov/data/fund-price-history.csv", "etag": "abc123"}'
    )

    content, headers = tsp_price._download_csv_content(cache_exists=True)

    assert content is None
    assert headers["ETag"] == "abc123"
    assert session.headers["If-None-Match"] == "abc123"


def test_update_writes_last_checked_metadata(tmp_path: Path) -> None:
    csv_payload = "Date,G Fund\n2024-01-02,100.0\n"
    csv_path = tmp_path / "fund-price-history.csv"
    csv_path.write_text(csv_payload)
    response = DummyResponse(
        csv_payload.encode("utf-8"),
        status_code=304,
        headers={"ETag": "abc123"},
    )
    session = DummySession([response], headers={})
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path, session=session)

    tsp_price.refresh()

    metadata_path = Path(tsp_price._metadata_filepath())
    metadata = json.loads(metadata_path.read_text())
    assert metadata["etag"] == "abc123"
    assert metadata["last_checked"]


def test_required_funds_enforced_on_load(tmp_path: Path) -> None:
    csv_payload = "Date,G Fund\n2024-01-02,100.0\n"
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        required_funds=["G Fund", "C Fund"],
    )

    with pytest.raises(ValueError, match="missing required funds"):
        tsp_price.load_csv_text(csv_payload)


def test_holiday_calendar_updates_business_day(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(
        auto_update=False,
        data_dir=tmp_path,
        holiday_calendar=[date(2024, 1, 1)],
    )

    assert tsp_price._get_last_business_day(date(2024, 1, 1)) == date(2023, 12, 29)


def test_apply_request_headers_handles_missing_session_headers(tmp_path: Path) -> None:
    class HeaderlessSession:
        def __init__(self) -> None:
            self.headers = None

    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    session = HeaderlessSession()

    headers = tsp_price._apply_request_headers(session)

    assert headers["User-Agent"] == tsp_price.user_agent
    assert headers["Accept"] == "text/csv"
    assert session.headers["User-Agent"] == tsp_price.user_agent


def test_load_cache_metadata_ignores_mismatched_url(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)
    metadata_path = Path(tsp_price._metadata_filepath())
    metadata_path.write_text(
        json.dumps(
            {
                "csv_url": "https://example.com/other.csv",
                "etag": "abc123",
            }
        )
    )

    metadata = tsp_price._load_cache_metadata()

    assert metadata == {}


def test_normalize_holidays_rejects_invalid_values() -> None:
    with pytest.raises(
        ValueError, match="holiday_calendar must contain date or datetime values"
    ):
        TspAnalytics._normalize_holidays([object()])


def test_is_holiday_supports_callable(tmp_path: Path) -> None:
    holiday = date(2024, 1, 2)

    def holiday_calendar(value: date) -> bool:
        return value == holiday

    tsp_price = TspAnalytics(
        auto_update=False, data_dir=tmp_path, holiday_calendar=holiday_calendar
    )

    assert tsp_price._is_holiday(holiday) is True
    assert tsp_price._is_holiday(date(2024, 1, 3)) is False


def test_build_request_headers_includes_cache_metadata(tmp_path: Path) -> None:
    tsp_price = TspAnalytics(auto_update=False, data_dir=tmp_path)

    headers = tsp_price._build_request_headers(
        {"etag": "etag123", "last_modified": "Wed, 01 Jan 2020 00:00:00 GMT"}
    )

    assert headers["If-None-Match"] == "etag123"
    assert headers["If-Modified-Since"] == "Wed, 01 Jan 2020 00:00:00 GMT"


def test_resolve_csv_filepath_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="data_dir must be a non-empty directory path"):
        TspAnalytics(data_dir="")


def test_resolve_user_agent_rejects_empty_string(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="user_agent must be a non-empty string"):
        TspAnalytics(auto_update=False, data_dir=tmp_path, user_agent=" ")
