from types import SimpleNamespace

from tsp.data_providers import RequestsCSVDataProvider


class DummyResponse:
    def __init__(self) -> None:
        self.content = b"data"
        self.headers = {"Content-Type": "text/csv"}
        self.status_code = 200
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self.response = response
        self.headers: dict[str, str] = {}
        self.get_calls: list[tuple[str, float]] = []

    def __enter__(self) -> "DummySession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def get(self, url: str, timeout: float) -> DummyResponse:
        self.get_calls.append((url, timeout))
        return self.response


def test_requests_csv_data_provider_uses_session_factory() -> None:
    response = DummyResponse()
    session = DummySession(response)
    provider = RequestsCSVDataProvider(session_factory=lambda: session)

    result = provider.fetch(
        "https://example.com/data.csv",
        timeout=4.0,
        session=None,
        headers={"User-Agent": "tsp-test"},
    )

    assert response.raise_for_status_called is True
    assert session.get_calls == [("https://example.com/data.csv", 4.0)]
    assert session.headers["User-Agent"] == "tsp-test"
    assert result.content == b"data"
    assert result.headers == {"Content-Type": "text/csv"}
    assert result.status_code == 200
    assert result.encoding == "utf-8"
    assert result.apparent_encoding == "utf-8"


def test_requests_csv_data_provider_uses_existing_session() -> None:
    response = DummyResponse()
    session = DummySession(response)
    provider = RequestsCSVDataProvider(session_factory=lambda: SimpleNamespace())

    result = provider.fetch(
        "https://example.com/data.csv",
        timeout=2.0,
        session=session,
        headers={"User-Agent": "ignored"},
    )

    assert response.raise_for_status_called is True
    assert session.get_calls == [("https://example.com/data.csv", 2.0)]
    assert "User-Agent" not in session.headers
    assert result.content == b"data"
