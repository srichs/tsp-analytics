"""Load, cache, and refresh TSP CSV data from configured sources."""

from datetime import date, datetime, timedelta
from io import StringIO
import json
import os
from os import path
from pathlib import Path
from requests import Session
from tempfile import NamedTemporaryFile
from types import ModuleType
from typing import Callable, Iterable
import time as time_module

from pandas import DataFrame, read_csv

from tsp.data_providers import CSVDataProvider, RequestsCSVDataProvider


class DataIOMixin:
    """Handle data retrieval, caching, and CSV parsing for TSP prices."""

    METADATA_FILENAME = "fund-price-history.metadata.json"
    LOCK_FILENAME = "fund-price-history.lock"

    @staticmethod
    def _normalize_holidays(
        holidays: Iterable[date | datetime] | None,
    ) -> set[date] | None:
        if holidays is None:
            return None
        normalized: set[date] = set()
        for holiday in holidays:
            if isinstance(holiday, datetime):
                normalized.add(holiday.date())
            elif isinstance(holiday, date):
                normalized.add(holiday)
            else:
                raise ValueError(
                    "holiday_calendar must contain date or datetime values"
                )
        return normalized

    def _is_holiday(self, current_date: date) -> bool:
        holiday_calendar = getattr(self, "holiday_calendar", None)
        if holiday_calendar is None:
            return False
        if isinstance(holiday_calendar, set):
            return current_date in holiday_calendar
        if callable(holiday_calendar):
            return bool(holiday_calendar(current_date))
        raise ValueError("holiday_calendar must be a set of dates or a callable")

    def _is_business_day(self, current_date: date) -> bool:
        return current_date.weekday() <= 4 and not self._is_holiday(current_date)

    @staticmethod
    def _validate_downloaded_csv_content(
        csv_content: str,
        expected_funds: list[str] | None = None,
    ) -> None:
        if not csv_content or not csv_content.strip():
            raise ValueError("no csv data was received from server")
        stripped = csv_content.lstrip().lower()
        snippet = stripped[:200]
        if (
            snippet.startswith("<!doctype html")
            or snippet.startswith("<html")
            or "<html" in snippet
        ):
            raise ValueError("downloaded content does not look like a CSV file")
        header_line = ""
        for line in csv_content.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if stripped_line.lower().startswith("sep="):
                continue
            header_line = stripped_line.lstrip("\ufeff").strip()
            break
        if not header_line:
            raise ValueError("downloaded content does not contain a CSV header")
        header_lower = header_line.lower()
        if "date" not in header_lower:
            raise ValueError("downloaded content does not include a Date column header")
        if expected_funds:
            header_columns = {
                column.strip().lower()
                for column in header_line.split(",")
                if column.strip()
            }
            expected_columns = {fund.lower() for fund in expected_funds}
            if header_columns.isdisjoint(expected_columns):
                raise ValueError(
                    "downloaded content does not include known fund columns"
                )

    def _build_request_headers(
        self, metadata: dict[str, str] | None = None
    ) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/csv",
        }
        if metadata:
            etag = metadata.get("etag")
            last_modified = metadata.get("last_modified")
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified
        return headers

    def _apply_request_headers(
        self,
        session: Session,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        session_headers = getattr(session, "headers", None)
        if session_headers is None:
            try:
                session.headers = {}
                session_headers = session.headers
            except Exception:
                return self._build_request_headers(metadata)
        if session_headers is None:
            return self._build_request_headers(metadata)
        headers = self._build_request_headers(metadata)
        session_headers.update(headers)
        return dict(session_headers)

    def _metadata_filepath(self) -> str:
        return str(Path(self.csv_filepath).with_name(self.METADATA_FILENAME))

    def _load_cache_metadata(self) -> dict[str, str]:
        metadata_path = Path(self._metadata_filepath())
        if not metadata_path.is_file():
            return {}
        try:
            data = json.loads(metadata_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        csv_url = data.get("csv_url")
        if csv_url and csv_url != self.csv_url:
            return {}
        return {k: str(v) for k, v in data.items() if isinstance(v, str)}

    def _write_cache_metadata(
        self,
        headers: dict[str, str],
        last_checked: datetime | None = None,
        last_updated: datetime | None = None,
    ) -> None:
        if not headers and last_checked is None and last_updated is None:
            return
        etag = headers.get("ETag") or headers.get("etag")
        last_modified = headers.get("Last-Modified") or headers.get("last-modified")
        if (
            not etag
            and not last_modified
            and last_checked is None
            and last_updated is None
        ):
            return
        payload = {"csv_url": self.csv_url}
        existing = self._load_cache_metadata()
        for key in ("etag", "last_modified", "last_checked", "last_updated"):
            if key in existing:
                payload[key] = existing[key]
        if etag:
            payload["etag"] = etag
        if last_modified:
            payload["last_modified"] = last_modified
        if last_checked is not None:
            payload["last_checked"] = last_checked.astimezone().isoformat()
        if last_updated is not None:
            payload["last_updated"] = last_updated.astimezone().isoformat()
        Path(self._metadata_filepath()).write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )

    def _download_csv_content(
        self, cache_exists: bool = False
    ) -> tuple[str | None, dict[str, str]]:
        from tsp import tsp as tsp_module

        session_cls = getattr(tsp_module, "Session", Session)
        sleep_module = getattr(tsp_module, "time_module", time_module)
        expected_funds = getattr(self, "ALL_FUNDS", None)
        metadata = self._load_cache_metadata() if cache_exists else {}
        request_headers = self._build_request_headers(metadata)
        data_provider: CSVDataProvider = getattr(
            self, "data_provider", None
        ) or RequestsCSVDataProvider(session_factory=session_cls)
        for attempt in range(1, self.max_retries + 1):
            try:
                session = self.session
                if session is not None:
                    request_headers = self._apply_request_headers(session, metadata)
                response = data_provider.fetch(
                    self.csv_url,
                    timeout=self.request_timeout,
                    session=session,
                    headers=request_headers,
                )
                headers = response.headers or {}
                if response.status_code == 304 and cache_exists:
                    return None, dict(headers)
                encoding = response.encoding or response.apparent_encoding or "utf-8"
                csv_content = (response.content or b"").decode(
                    encoding, errors="replace"
                )
                self._validate_downloaded_csv_content(
                    csv_content, expected_funds=expected_funds
                )
                return csv_content, dict(headers)
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise
                wait_seconds = self.retry_backoff * (2 ** (attempt - 1))
                if wait_seconds > 0:
                    self.logger.warning(
                        "download attempt %s failed; retrying in %.2fs",
                        attempt,
                        wait_seconds,
                        exc_info=exc,
                    )
                    sleep_module.sleep(wait_seconds)
                else:
                    self.logger.warning(
                        "download attempt %s failed; retrying immediately",
                        attempt,
                        exc_info=exc,
                    )
        raise RuntimeError("unable to download TSP price data after retries")

    def _emit_event(self, name: str, payload: dict[str, object] | None = None) -> None:
        handler = getattr(self, "event_handler", None)
        if handler is None:
            return
        try:
            handler(name, payload or {})
        except Exception as exc:
            self.logger.debug("event handler error for %s", name, exc_info=exc)

    def check(self) -> None:
        """
        Checks if the current CSV file exists, if it doesn't it calls _update() to download it.
        If it exists, then the method checks if there is possibly a newer version available and calls
        _update() to re-download the TSP CSV file.
        """
        self.logger.debug("check()")
        if not path.isfile(self.csv_filepath):
            self.logger.debug("no csv data exists, checking for download")
            if self.auto_update:
                self._update()
            else:
                self.logger.debug("auto_update disabled; skipping download")
        else:
            self.logger.debug("csv data exists, checking if update is needed")
            try:
                self._set_values()
            except Exception as exc:
                self.logger.warning("cached csv failed validation", exc_info=exc)
                if not self.auto_update:
                    raise
                self._update()
                if (
                    self.dataframe is None
                    or self.current is None
                    or self.latest is None
                ):
                    raise RuntimeError("TSP price data is not available") from exc
                self.logger.debug("check() complete")
                return
            if not self.auto_update:
                self.logger.debug("auto_update disabled; skipping refresh")
                self.logger.debug("check() complete")
                return
            now = datetime.now().time()
            today = date.today()
            last_business_day = self._get_last_business_day(today)

            if self.latest < last_business_day:
                if last_business_day < today or now > self.time_hour:
                    self.logger.debug(
                        "checking for new data based on last business day"
                    )
                    self._update()
        self.logger.debug("check() complete")

    def _get_last_business_day(self, current_date: date) -> date:
        last_business_day = current_date
        while not self._is_business_day(last_business_day):
            last_business_day -= timedelta(days=1)
        return last_business_day

    def _lock_filepath(self) -> str:
        return str(Path(self.csv_filepath).with_name(self.LOCK_FILENAME))

    def _acquire_file_lock(self) -> tuple[object | None, Callable[[], None]]:
        if not getattr(self, "use_file_lock", True):
            return None, lambda: None
        lock_path = Path(self._lock_filepath())
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+")
        try:
            fcntl_module: ModuleType | None
            try:
                import fcntl as fcntl_module
            except ImportError:
                fcntl_module = None
            if fcntl_module is not None:
                fcntl_module.flock(lock_file.fileno(), fcntl_module.LOCK_EX)

                def _release() -> None:
                    fcntl_module.flock(lock_file.fileno(), fcntl_module.LOCK_UN)
                    lock_file.close()

                return lock_file, _release
            msvcrt_module: ModuleType | None
            try:
                import msvcrt as msvcrt_module
            except ImportError:
                msvcrt_module = None
            if msvcrt_module is not None:
                msvcrt_module.locking(lock_file.fileno(), msvcrt_module.LK_LOCK, 1)

                def _release() -> None:
                    msvcrt_module.locking(lock_file.fileno(), msvcrt_module.LK_UNLCK, 1)
                    lock_file.close()

                return lock_file, _release
        except Exception:
            lock_file.close()
            raise
        return lock_file, lock_file.close

    def _resolve_csv_filepath(self, data_dir: str | Path | None) -> str:
        if data_dir is None:
            data_dir = os.getenv(self.ENV_DATA_DIR) or self.DEFAULT_DATA_DIR
        if isinstance(data_dir, str) and not data_dir.strip():
            raise ValueError("data_dir must be a non-empty directory path")
        data_path = Path(data_dir).expanduser()
        if data_path.exists() and not data_path.is_dir():
            raise ValueError("data_dir must be a directory path")
        data_path.mkdir(parents=True, exist_ok=True)
        return str(data_path / self.CSV_FILENAME)

    def _resolve_csv_url(self, csv_url: str | None) -> str:
        if csv_url is None:
            return self.CSV_URL
        if not isinstance(csv_url, str) or not csv_url.strip():
            raise ValueError("csv_url must be a non-empty string")
        return csv_url.strip()

    def _resolve_user_agent(self, user_agent: str | None) -> str:
        if user_agent is None:
            return self.USER_AGENT_STR
        if not isinstance(user_agent, str) or not user_agent.strip():
            raise ValueError("user_agent must be a non-empty string")
        return user_agent.strip()

    def _update(self) -> None:
        csv_file_exists = path.isfile(self.csv_filepath)
        self.logger.debug("_update()")
        lock_handle, release_lock = self._acquire_file_lock()
        try:
            try:
                self._emit_event("download_start", {"csv_url": self.csv_url})
                csv_content, response_headers = self._download_csv_content(
                    cache_exists=csv_file_exists
                )
            except Exception as e:
                if not csv_file_exists:
                    self._emit_event("download_error", {"error": str(e)})
                    raise
                else:
                    self.logger.warning("unable to download new csv file", exc_info=e)
                    self._emit_event("download_error", {"error": str(e)})
                    return

            if csv_content is None:
                self.logger.debug("csv file not modified; using cached data")
                self._write_cache_metadata(
                    response_headers, last_checked=datetime.now()
                )
                self._emit_event("download_not_modified", {"csv_url": self.csv_url})
                if self.dataframe is None:
                    self._set_values()
                return

            if not csv_content:
                self.logger.debug("no csv data was received from server")
                if not csv_file_exists:
                    raise ValueError("no csv data was received from server")
                return
            try:
                dataframe = self._normalize_dataframe(read_csv(StringIO(csv_content)))
            except Exception as e:
                if not csv_file_exists:
                    raise
                self.logger.warning("downloaded csv data failed validation", exc_info=e)
                self._emit_event("download_validation_failed", {"error": str(e)})
                return
            if self.latest is not None:
                new_latest = dataframe["Date"].max().date()
                if new_latest < self.latest:
                    self.logger.warning(
                        "downloaded csv data is older than existing cache; keeping existing data"
                    )
                    self._emit_event(
                        "download_older_than_cache",
                        {
                            "latest_date": str(self.latest),
                            "new_latest_date": str(new_latest),
                        },
                    )
                    return
            temp_file_path = None
            try:
                with NamedTemporaryFile(
                    "w", delete=False, dir=Path(self.csv_filepath).parent
                ) as csv_file:
                    self.logger.debug("writing csv data to temporary file")
                    csv_file.write(csv_content)
                    temp_file_path = csv_file.name

                self.logger.debug("replacing old csv file with new data")
                Path(temp_file_path).replace(self.csv_filepath)
                self._assign_dataframe(dataframe)
                self._write_cache_metadata(
                    response_headers,
                    last_checked=datetime.now(),
                    last_updated=datetime.now(),
                )
                self._emit_event("download_success", {"latest_date": str(self.latest)})
            finally:
                if temp_file_path and path.exists(temp_file_path):
                    Path(temp_file_path).unlink(missing_ok=True)
        finally:
            _ = lock_handle
            release_lock()

    def _set_values(self) -> bool:
        self.logger.debug("_set_values()")
        if path.isfile(self.csv_filepath):
            dataframe = read_csv(self.csv_filepath)
            dataframe = self._normalize_dataframe(dataframe)
            self._assign_dataframe(dataframe)
            return True
        return False

    def refresh(self) -> None:
        """
        Forces a refresh of the cached CSV by downloading the latest data.
        """
        self.logger.debug("refresh()")
        self._update()

    def load_dataframe(self, dataframe: DataFrame) -> None:
        """
        Loads a pandas dataframe into the client after validation and normalization.

        Args:
            dataframe (DataFrame): dataframe containing a Date column and fund prices.

        Raises:
            ValueError: if the dataframe is empty or missing required columns.
            TypeError: if the provided object is not a pandas DataFrame.
        """
        if not isinstance(dataframe, DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        normalized = self._normalize_dataframe(dataframe)
        self._assign_dataframe(normalized)

    def load_csv(self, filepath: str | Path) -> None:
        """
        Loads a CSV file into the client after validation and normalization.

        Args:
            filepath (str | Path): path to the CSV file.

        Raises:
            ValueError: if the file does not exist or cannot be read.
        """
        csv_path = Path(filepath).expanduser()
        if not csv_path.is_file():
            raise ValueError("csv filepath must point to an existing file")
        try:
            dataframe = read_csv(csv_path)
        except Exception as exc:
            raise ValueError("unable to read csv file") from exc
        normalized = self._normalize_dataframe(dataframe)
        self._assign_dataframe(normalized)

    def load_csv_text(self, csv_content: str | bytes | bytearray) -> None:
        """
        Loads CSV content from a string into the client after validation and normalization.

        Args:
            csv_content (str | bytes | bytearray): CSV content as text or raw bytes.

        Raises:
            TypeError: if the provided content is not a string or bytes.
            ValueError: if the content is empty or cannot be parsed as CSV.
        """
        if isinstance(csv_content, (bytes, bytearray)):
            csv_content = csv_content.decode("utf-8-sig", errors="replace")
        if not isinstance(csv_content, str):
            raise TypeError("csv_content must be a string or bytes")
        if not csv_content.strip():
            raise ValueError("csv_content must be a non-empty string")
        self._validate_downloaded_csv_content(
            csv_content,
            expected_funds=getattr(self, "ALL_FUNDS", None),
        )
        try:
            dataframe = read_csv(StringIO(csv_content))
        except Exception as exc:
            raise ValueError("unable to parse csv content") from exc
        normalized = self._normalize_dataframe(dataframe)
        self._assign_dataframe(normalized)
