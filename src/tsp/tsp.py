"""Public TSP price client composed from data, analytics, and charting mixins."""

from collections.abc import Callable, Iterable
from datetime import date, datetime, time
import logging
from pathlib import Path
from requests import Session
import time as time_module

from pandas import DataFrame, Series

from tsp.allocation import AllocationMixin
from tsp.analytics import AnalyticsMixin
from tsp.charts import ChartsMixin
from tsp.data_io import DataIOMixin
from tsp.data_providers import CSVDataProvider
from tsp.fund_metadata import FundMetadataMixin
from tsp.fund_resolution import FundResolutionMixin
from tsp.funds import TspIndividualFund, TspLifecycleFund
from tsp.normalization import NormalizationMixin
from tsp.validation import ValidationMixin


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__all__ = ["TspAnalytics", "time_module"]


class TspAnalytics(
    DataIOMixin,
    FundMetadataMixin,
    FundResolutionMixin,
    NormalizationMixin,
    AllocationMixin,
    ValidationMixin,
    AnalyticsMixin,
    ChartsMixin,
):
    """High-level client for retrieving, analyzing, and charting TSP fund data.

    Fund-specific methods accept either fund enums or fund name strings
    (case-insensitive).
    """

    TSP_URL = "https://www.tsp.gov/"
    CSV_FILENAME = "fund-price-history.csv"
    CSV_URL = f"{TSP_URL}data/{CSV_FILENAME}"
    DEFAULT_DATA_DIR = Path.home() / ".cache" / "tsp"
    DEFAULT_CSV_FILEPATH = str(DEFAULT_DATA_DIR / CSV_FILENAME)
    ENV_DATA_DIR = "TSP_DATA_DIR"
    USER_AGENT_STR = (
        "Mozilla/5.0 (Windows NT 10.0; rv:128.0) Gecko/20100101 Firefox/128.0"
    )
    INDIVIDUAL_FUNDS = [fund.value for fund in TspIndividualFund]
    LIFECYCLE_FUNDS = [fund.value for fund in TspLifecycleFund]
    ALL_FUNDS = INDIVIDUAL_FUNDS + LIFECYCLE_FUNDS

    def __init__(
        self,
        log_level: int = logging.ERROR,
        time_hour: time = time(hour=19),
        data_dir: str | Path | None = None,
        auto_update: bool = True,
        request_timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        csv_url: str | None = None,
        user_agent: str | None = None,
        session: Session | None = None,
        data_provider: CSVDataProvider | None = None,
        holiday_calendar: (
            Iterable[date | datetime] | Callable[[date], bool] | None
        ) = None,
        required_funds: (
            Iterable[str | TspIndividualFund | TspLifecycleFund] | None
        ) = None,
        use_file_lock: bool = True,
        logger: logging.Logger | None = None,
        event_handler: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        """Initialize the TSP client and load cached fund price data if available.

        Args:
            log_level (int): logging level to apply to the internal logger.
            time_hour (time): time of day after which a new trading day is expected.
            data_dir (str | Path | None): directory for caching the CSV file. Defaults
                to ``~/.cache/tsp`` or ``TSP_DATA_DIR`` if set.
            auto_update (bool): whether to download a fresh CSV when cached data is
                stale.
            request_timeout (float): request timeout, in seconds, for CSV downloads.
            max_retries (int): number of download retries before raising an error.
            retry_backoff (float): exponential backoff multiplier (seconds) between
                retries.
            csv_url (str | None): optional override for the CSV download URL.
            user_agent (str | None): optional override for the HTTP ``User-Agent``
                header.
            session (Session | None): optional ``requests`` session to reuse for downloads.
            data_provider (CSVDataProvider | None): optional provider to fetch CSV data.
            holiday_calendar (Iterable[date | datetime] | Callable[[date], bool] | None):
                optional holiday calendar to skip non-trading dates when checking for
                updates.
            required_funds (Iterable[str | TspIndividualFund | TspLifecycleFund] | None):
                optional fund list to require in loaded data.
            use_file_lock (bool): whether to use a file lock during cache updates.
            logger (logging.Logger | None): optional logger instance for TspAnalytics.
            event_handler (Callable[[str, dict[str, object]], None] | None): optional
                handler invoked for download/cache events.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.debug("TspAnalytics initialized")
        self._validate_time_hour(time_hour)
        self.time_hour = time_hour
        self.csv_filepath = self._resolve_csv_filepath(data_dir)
        self.auto_update = auto_update
        self._validate_positive_float(request_timeout, "request_timeout")
        self.request_timeout = float(request_timeout)
        self._validate_positive_int(max_retries, "max_retries")
        self.max_retries = int(max_retries)
        self._validate_non_negative_float(retry_backoff, "retry_backoff")
        self.retry_backoff = float(retry_backoff)
        self.csv_url = self._resolve_csv_url(csv_url)
        self.user_agent = self._resolve_user_agent(user_agent)
        self.session = session
        self.data_provider = data_provider
        self.holiday_calendar: set[date] | Callable[[date], bool] | None
        if holiday_calendar is None:
            self.holiday_calendar = None
        elif callable(holiday_calendar):
            self.holiday_calendar = holiday_calendar
        else:
            self.holiday_calendar = self._normalize_holidays(holiday_calendar)
        self.required_funds = self._normalize_required_funds(required_funds)
        self.use_file_lock = use_file_lock
        self.event_handler = event_handler
        self.dataframe: DataFrame | None = None
        self.current: Series | None = None
        self.latest: date | None = None
        self._set_values()

    def _normalize_required_funds(
        self,
        required_funds: Iterable[str | TspIndividualFund | TspLifecycleFund] | None,
    ) -> list[str] | None:
        if required_funds is None:
            return None
        if isinstance(required_funds, (str, TspIndividualFund, TspLifecycleFund)):
            required_funds = [required_funds]
        if not isinstance(required_funds, Iterable):
            raise ValueError(
                "required_funds must be an iterable of fund names or enums"
            )
        resolved: list[str] = []
        for fund in required_funds:
            resolved.append(self._resolve_fund(fund))
        if not resolved:
            raise ValueError("required_funds must contain at least one fund")
        return list(dict.fromkeys(resolved))
