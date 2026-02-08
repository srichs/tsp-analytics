"""Data provider interfaces for retrieving TSP CSV content."""

from dataclasses import dataclass
from typing import Callable, Protocol

from requests import Session


@dataclass(frozen=True)
class DataFetchResult:
    """Result payload for a CSV fetch operation."""

    content: bytes | None
    headers: dict[str, str]
    status_code: int | None = None
    encoding: str | None = None
    apparent_encoding: str | None = None


class CSVDataProvider(Protocol):
    """Interface for fetching CSV content from a data source."""

    def fetch(
        self,
        url: str,
        timeout: float,
        session: Session | None,
        headers: dict[str, str],
    ) -> DataFetchResult:
        """Fetch CSV content and return a DataFetchResult."""


class RequestsCSVDataProvider:
    """Default provider that uses requests.Session to fetch CSV content."""

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or Session

    def fetch(
        self,
        url: str,
        timeout: float,
        session: Session | None,
        headers: dict[str, str],
    ) -> DataFetchResult:
        if session is None:
            with self._session_factory() as active_session:
                try:
                    response = active_session.get(url, timeout=timeout, headers=headers)
                except TypeError:
                    active_session.headers.update(headers)
                    response = active_session.get(url, timeout=timeout)
        else:
            try:
                response = session.get(url, timeout=timeout, headers=headers)
            except TypeError:
                response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return DataFetchResult(
            content=response.content,
            headers=dict(getattr(response, "headers", {}) or {}),
            status_code=getattr(response, "status_code", None),
            encoding=getattr(response, "encoding", None),
            apparent_encoding=getattr(response, "apparent_encoding", None),
        )
