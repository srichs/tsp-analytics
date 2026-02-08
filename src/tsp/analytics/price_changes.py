"""Price-change analytics for fund recency, deltas, and streak metrics."""

from collections.abc import Iterable
from datetime import date, datetime

from pandas import DataFrame, Series, to_datetime

from tsp.funds import FundInput


class AnalyticsPriceChangesMixin:
    """Provide price-change analytics such as recency and delta calculations."""

    def get_price_recency(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the number of days since each fund's most recent available price.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date. When provided, recency is calculated relative to this
                date using prices on or before it. When None, uses the latest available date in
                the dataset.

        Returns:
            DataFrame: dataframe indexed by fund with as_of date and days_since columns.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        if fund is not None:
            funds = [fund]
        if reference_date is None:
            reference_date = self.latest
            latest = self.get_latest_prices_per_fund(funds=funds)
        else:
            reference_date = to_datetime(reference_date).date()
            latest = self.get_prices_as_of_per_fund(reference_date, funds=funds)
        records: list[dict] = []
        for fund_name, row in latest.iterrows():
            as_of_date = row["as_of"]
            records.append(
                {
                    "fund": fund_name,
                    "as_of": as_of_date,
                    "days_since": int((reference_date - as_of_date).days),
                }
            )
        return DataFrame(records).set_index("fund")

    def get_price_recency_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets per-fund price recency in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date. When provided, recency is calculated relative to this
                date using prices on or before it. When None, uses the latest available date in
                the dataset.

        Returns:
            DataFrame: dataframe with fund, as_of, and days_since columns.
        """
        return self.get_price_recency(
            fund=fund, funds=funds, reference_date=reference_date
        ).reset_index()

    def get_price_recency_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets per-fund price recency as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date. When provided, recency is calculated relative to this
                date using prices on or before it. When None, uses the latest available date in
                the dataset.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with reference_date and per-fund recency metrics.
        """
        recency = self.get_price_recency(
            fund=fund, funds=funds, reference_date=reference_date
        )
        if recency.empty:
            raise ValueError("no price data available for price recency")
        if reference_date is None:
            reference_date = self.latest
        else:
            reference_date = to_datetime(reference_date).date()
        payload: dict[str, dict] = {}
        for fund_name, row in recency.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "days_since": int(row["days_since"]),
            }
        return {
            "reference_date": self._format_date_for_output(reference_date, date_format),
            "funds": payload,
        }

    def get_current_price_status(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the latest per-fund prices with recency information.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to the as_of date when
                provided, otherwise uses the latest available date in the dataset.

        Returns:
            DataFrame: dataframe indexed by fund with as_of, price, and days_since columns.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        if fund is not None:
            funds = [fund]
        if reference_date is not None:
            reference_date_value = to_datetime(reference_date).date()
        elif as_of is not None:
            reference_date_value = to_datetime(as_of).date()
        else:
            reference_date_value = self.latest

        if reference_date_value is None:
            raise ValueError("reference_date could not be resolved")

        if as_of is not None:
            as_of_value = to_datetime(as_of).date()
            if reference_date_value < as_of_value:
                raise ValueError("reference_date cannot be earlier than as_of")

        if as_of is not None:
            status = self.get_prices_as_of_per_fund(as_of, funds=funds)
        else:
            status = self.get_latest_prices_per_fund(funds=funds)

        status = status.copy()
        status["days_since"] = status["as_of"].apply(
            lambda value: int((reference_date_value - value).days)
        )
        return status

    def get_current_price_status_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the current price status in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to the as_of date when
                provided, otherwise uses the latest available date in the dataset.

        Returns:
            DataFrame: dataframe with fund, as_of, price, and days_since columns.
        """
        return self.get_current_price_status(
            fund=fund, funds=funds, as_of=as_of, reference_date=reference_date
        ).reset_index()

    def get_current_price_status_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets the current price status as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to the as_of date when
                provided, otherwise uses the latest available date in the dataset.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with reference_date and per-fund price/recency metrics.
        """
        status = self.get_current_price_status(
            fund=fund, funds=funds, as_of=as_of, reference_date=reference_date
        )
        if reference_date is not None:
            reference_date_value = to_datetime(reference_date).date()
        elif as_of is not None:
            reference_date_value = to_datetime(as_of).date()
        else:
            reference_date_value = self.latest

        payload: dict[str, dict] = {}
        for fund_name, row in status.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "price": self._format_numeric_for_output(row["price"]),
                "days_since": int(row["days_since"]),
            }
        return {
            "reference_date": self._format_date_for_output(
                reference_date_value, date_format
            ),
            "funds": payload,
        }

    def get_current_price_summary(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
    ) -> DataFrame:
        """
        Gets a summary of current prices, recency, and daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date for the price changes summary.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to the latest available
                date in the dataset, or the as_of date when provided.
            stale_days (int): number of days after which a price is considered stale.

        Returns:
            DataFrame: dataframe with a single summary row of recency and change statistics.
        """
        self._ensure_dataframe()
        if (
            isinstance(stale_days, bool)
            or not isinstance(stale_days, int)
            or stale_days < 0
        ):
            raise ValueError("stale_days must be a non-negative integer")
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        if fund is not None:
            funds = [fund]
        fund_list = self._resolve_funds(funds)
        status = self.get_current_price_status(
            funds=fund_list, as_of=as_of, reference_date=reference_date
        )
        if status.empty:
            raise ValueError("no price data available for current price summary")
        if as_of is None:
            if reference_date is None:
                changes = self.get_latest_price_changes_per_fund(funds=fund_list)
            else:
                changes = self.get_price_changes_as_of_per_fund(
                    as_of=reference_date, funds=fund_list
                )
        else:
            changes = self.get_price_changes_as_of_per_fund(
                as_of=as_of, funds=fund_list
            )
        if changes.empty:
            raise ValueError("no price change data available for current price summary")
        if reference_date is not None:
            reference_date_value = to_datetime(reference_date).date()
        elif as_of is not None:
            reference_date_value = to_datetime(as_of).date()
        else:
            reference_date_value = self.latest
        if reference_date_value is None:
            raise ValueError("reference_date could not be resolved")

        if as_of is not None or reference_date is not None:
            as_of_value = changes["as_of"].max()
        else:
            as_of_value = self.latest
        days_since = status["days_since"].astype(float)
        change = changes["change"].astype(float)
        change_percent = changes["change_percent"].astype(float)

        summary = {
            "as_of": as_of_value,
            "reference_date": reference_date_value,
            "stale_threshold_days": stale_days,
            "total_funds": int(len(fund_list)),
            "stale_funds": int((days_since > stale_days).sum()),
            "min_days_since": float(days_since.min()),
            "max_days_since": float(days_since.max()),
            "mean_days_since": float(days_since.mean()),
            "median_days_since": float(days_since.median()),
            "positive_changes": int((change > 0).sum()),
            "negative_changes": int((change < 0).sum()),
            "unchanged_changes": int((change == 0).sum()),
            "average_change": float(change.mean()),
            "average_change_percent": float(change_percent.mean()),
            "min_change": float(change.min()),
            "max_change": float(change.max()),
            "min_change_percent": float(change_percent.min()),
            "max_change_percent": float(change_percent.max()),
        }
        if as_of is not None:
            summary["requested_as_of"] = to_datetime(as_of).date()
        return DataFrame([summary], index=["summary"])

    def get_current_price_alert_summary(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        change_threshold: float | None = 0.02,
    ) -> DataFrame:
        """
        Gets a compact summary of current price alerts (stale data and large moves).

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to as_of when provided,
                otherwise uses the latest available date in the dataset.
            stale_days (int): number of days after which a price is considered stale.
            change_threshold (float | None): absolute daily change threshold for large moves.
                Set to None to disable large-move detection.

        Returns:
            DataFrame: dataframe with a single summary row of alert counts and thresholds.
        """
        alerts = self.get_current_price_alerts(
            fund=fund,
            funds=funds,
            as_of=as_of,
            reference_date=reference_date,
            stale_days=stale_days,
            change_threshold=change_threshold,
        )
        if alerts.empty:
            raise ValueError("no alert data available for current price alert summary")

        if reference_date is not None:
            reference_date_value = to_datetime(reference_date).date()
        elif as_of is not None:
            reference_date_value = to_datetime(as_of).date()
        else:
            reference_date_value = self.latest
        if reference_date_value is None:
            raise ValueError("reference_date could not be resolved")

        stale_mask = alerts["is_stale"]
        large_mask = alerts["is_large_move"]
        total_funds = int(len(alerts))
        stale_funds = int(stale_mask.sum())
        large_moves = int(large_mask.sum())
        stale_and_large = int((stale_mask & large_mask).sum())
        summary = {
            "as_of": alerts["as_of"].max(),
            "reference_date": reference_date_value,
            "stale_threshold_days": stale_days,
            "change_threshold": change_threshold,
            "total_funds": total_funds,
            "stale_funds": stale_funds,
            "large_move_funds": large_moves,
            "stale_only_funds": int((stale_mask & ~large_mask).sum()),
            "large_move_only_funds": int((~stale_mask & large_mask).sum()),
            "stale_and_large_funds": stale_and_large,
        }
        return DataFrame([summary], index=["summary"])

    def get_current_price_alert_summary_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        change_threshold: float | None = 0.02,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets a JSON-friendly summary of current price alerts.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to as_of when provided,
                otherwise uses the latest available date in the dataset.
            stale_days (int): number of days after which a price is considered stale.
            change_threshold (float | None): absolute daily change threshold for large moves.
                Set to None to disable large-move detection.
            date_format (str | None): format for date values. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: summary dictionary with alert counts and thresholds.
        """
        summary = self.get_current_price_alert_summary(
            fund=fund,
            funds=funds,
            as_of=as_of,
            reference_date=reference_date,
            stale_days=stale_days,
            change_threshold=change_threshold,
        ).iloc[0]
        return {
            "as_of": self._format_date_for_output(summary["as_of"], date_format),
            "reference_date": self._format_date_for_output(
                summary["reference_date"], date_format
            ),
            "stale_threshold_days": int(summary["stale_threshold_days"]),
            "change_threshold": (
                None
                if summary["change_threshold"] is None
                else float(summary["change_threshold"])
            ),
            "total_funds": int(summary["total_funds"]),
            "stale_funds": int(summary["stale_funds"]),
            "large_move_funds": int(summary["large_move_funds"]),
            "stale_only_funds": int(summary["stale_only_funds"]),
            "large_move_only_funds": int(summary["large_move_only_funds"]),
            "stale_and_large_funds": int(summary["stale_and_large_funds"]),
        }

    def get_current_price_summary_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        date_format: str | None = "iso",
        as_of: date | None = None,
    ) -> dict:
        """
        Gets a summary of current prices as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to the latest available
                date in the dataset, or the as_of date when provided.
            stale_days (int): number of days after which a price is considered stale.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            as_of (date | None): optional historical anchor date for the price changes summary.

        Returns:
            dict: summary dictionary with recency and change statistics.
        """
        summary = self.get_current_price_summary(
            fund=fund,
            funds=funds,
            as_of=as_of,
            reference_date=reference_date,
            stale_days=stale_days,
        ).iloc[0]
        payload = {
            "as_of": self._format_date_for_output(summary["as_of"], date_format),
            "reference_date": self._format_date_for_output(
                summary["reference_date"], date_format
            ),
            "stale_threshold_days": int(summary["stale_threshold_days"]),
            "total_funds": int(summary["total_funds"]),
            "stale_funds": int(summary["stale_funds"]),
            "min_days_since": self._format_numeric_for_output(
                summary["min_days_since"]
            ),
            "max_days_since": self._format_numeric_for_output(
                summary["max_days_since"]
            ),
            "mean_days_since": self._format_numeric_for_output(
                summary["mean_days_since"]
            ),
            "median_days_since": self._format_numeric_for_output(
                summary["median_days_since"]
            ),
            "positive_changes": int(summary["positive_changes"]),
            "negative_changes": int(summary["negative_changes"]),
            "unchanged_changes": int(summary["unchanged_changes"]),
            "average_change": self._format_numeric_for_output(
                summary["average_change"]
            ),
            "average_change_percent": self._format_numeric_for_output(
                summary["average_change_percent"]
            ),
            "min_change": self._format_numeric_for_output(summary["min_change"]),
            "max_change": self._format_numeric_for_output(summary["max_change"]),
            "min_change_percent": self._format_numeric_for_output(
                summary["min_change_percent"]
            ),
            "max_change_percent": self._format_numeric_for_output(
                summary["max_change_percent"]
            ),
        }
        if "requested_as_of" in summary:
            payload["requested_as_of"] = self._format_date_for_output(
                summary["requested_as_of"], date_format
            )
        return payload

    def get_current_price_alerts(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        change_threshold: float | None = 0.02,
    ) -> DataFrame:
        """
        Gets a per-fund alert table highlighting stale prices or large daily moves.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to as_of when provided,
                otherwise uses the latest available date in the dataset.
            stale_days (int): number of days after which a price is considered stale.
            change_threshold (float | None): absolute daily change threshold for large moves.
                Set to None to disable large-move detection.

        Returns:
            DataFrame: dataframe indexed by fund with price, change, recency, and alert flags.
        """
        self._ensure_dataframe()
        if (
            isinstance(stale_days, bool)
            or not isinstance(stale_days, int)
            or stale_days < 0
        ):
            raise ValueError("stale_days must be a non-negative integer")
        if change_threshold is not None:
            self._validate_non_negative_float(change_threshold, "change_threshold")
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        if fund is not None:
            funds = [fund]

        status = self.get_current_price_status(
            funds=funds, as_of=as_of, reference_date=reference_date
        )
        if as_of is not None:
            changes = self.get_price_changes_as_of_per_fund(as_of=as_of, funds=funds)
        else:
            changes = self.get_latest_price_changes_per_fund(funds=funds)

        alerts = status.join(
            changes[["previous_as_of", "previous_price", "change", "change_percent"]],
            how="left",
        )
        alerts["is_stale"] = alerts["days_since"] > stale_days
        if change_threshold is None:
            alerts["is_large_move"] = False
        else:
            alerts["is_large_move"] = alerts["change_percent"].abs() >= float(
                change_threshold
            )
        return alerts

    def get_current_price_alerts_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        change_threshold: float | None = 0.02,
    ) -> DataFrame:
        """
        Gets current price alerts in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to as_of when provided,
                otherwise uses the latest available date in the dataset.
            stale_days (int): number of days after which a price is considered stale.
            change_threshold (float | None): absolute daily change threshold for large moves.

        Returns:
            DataFrame: dataframe with fund, price, change, recency, and alert flags.
        """
        return self.get_current_price_alerts(
            fund=fund,
            funds=funds,
            as_of=as_of,
            reference_date=reference_date,
            stale_days=stale_days,
            change_threshold=change_threshold,
        ).reset_index()

    def get_current_price_alerts_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        change_threshold: float | None = 0.02,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets current price alerts as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. Defaults to as_of when provided,
                otherwise uses the latest available date in the dataset.
            stale_days (int): number of days after which a price is considered stale.
            change_threshold (float | None): absolute daily change threshold for large moves.
                Set to None to disable large-move detection.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with reference_date, thresholds, and per-fund alert data.
        """
        alerts = self.get_current_price_alerts(
            fund=fund,
            funds=funds,
            as_of=as_of,
            reference_date=reference_date,
            stale_days=stale_days,
            change_threshold=change_threshold,
        )
        if reference_date is not None:
            reference_date_value = to_datetime(reference_date).date()
        elif as_of is not None:
            reference_date_value = to_datetime(as_of).date()
        else:
            reference_date_value = self.latest
        payload: dict[str, dict] = {}
        for fund_name, row in alerts.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "price": self._format_numeric_for_output(row["price"]),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
                "days_since": int(row["days_since"]),
                "is_stale": bool(row["is_stale"]),
                "is_large_move": bool(row["is_large_move"]),
            }
        return {
            "reference_date": self._format_date_for_output(
                reference_date_value, date_format
            ),
            "stale_threshold_days": int(stale_days),
            "change_threshold": (
                float(change_threshold) if change_threshold is not None else None
            ),
            "funds": payload,
        }

    def get_current_price_changes(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) price changes from the previous trading day.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe containing latest price changes indexed by fund.
        """
        if as_of is not None:
            return self.get_price_changes_as_of(as_of=as_of, fund=fund, funds=funds)
        return self.get_latest_price_changes(fund=fund, funds=funds)

    def get_latest_price_changes_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets the latest price change metrics as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with as_of dates and per-fund change metrics.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        latest_date, latest, previous = self._resolve_latest_and_previous(price_df)
        changes = self._build_price_change_frame(
            latest=latest, previous=previous
        ).rename_axis("fund")
        if changes.empty:
            raise ValueError("no price data available for latest price changes")
        payload: dict[str, dict] = {}
        for fund_name, row in changes.iterrows():
            payload[fund_name] = {
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {
            "as_of": self._format_date_for_output(latest_date.date(), date_format),
            "previous_as_of": self._format_date_for_output(
                price_df.index[-2].date(), date_format
            ),
            "funds": payload,
        }

    def get_latest_price_report_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        include_cache_status: bool = False,
        include_data_quality: bool = False,
    ) -> dict:
        """
        Gets a combined latest price report with prices and daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            include_cache_status (bool): whether to include cache metadata.
            include_data_quality (bool): whether to include a data-quality summary payload.

        Returns:
            dict: dictionary with as-of dates, latest prices, and daily change metrics.
        """
        latest_payload = self.get_latest_prices_dict(
            fund=fund, funds=funds, date_format=date_format
        )
        changes_payload = self.get_latest_price_changes_dict(
            fund=fund, funds=funds, date_format=date_format
        )
        payload = {
            "as_of": latest_payload["as_of"],
            "previous_as_of": changes_payload["previous_as_of"],
            "prices": latest_payload["prices"],
            "changes": changes_payload["funds"],
        }
        if include_cache_status:
            datetime_format = None if date_format is None else date_format
            payload["cache_status"] = self.get_cache_status_dict(
                date_format=date_format, datetime_format=datetime_format
            )
        if include_data_quality:
            datetime_format = None if date_format is None else date_format
            payload["data_quality"] = self.get_data_quality_report_dict(
                include_cache_status=False,
                date_format=date_format,
                datetime_format=datetime_format,
            )
        return payload

    def get_latest_price_report_per_fund(
        self, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets a per-fund latest price report using each fund's last two valid prices.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe with per-fund as_of, previous_as_of, prices, and change metrics.
        """
        return self.get_latest_price_changes_per_fund(funds=funds)

    def get_latest_price_report_per_fund_long(
        self, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the per-fund latest price report in long (tidy) format.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, and change metrics columns.
        """
        return self.get_latest_price_report_per_fund(funds=funds).reset_index()

    def get_latest_price_report_per_fund_dict(
        self, funds: Iterable[FundInput] | None = None, date_format: str | None = "iso"
    ) -> dict:
        """
        Gets a per-fund latest price report as a JSON-friendly dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with an overall latest as_of date and per-fund metrics.
        """
        report = self.get_latest_price_report_per_fund(funds=funds)
        if report.empty:
            raise ValueError("no price data available for per-fund price report")
        payload: dict[str, dict] = {}
        for fund_name, row in report.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        overall_as_of = self._format_date_for_output(report["as_of"].max(), date_format)
        return {"as_of": overall_as_of, "funds": payload}

    def get_fund_overview(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets a per-fund overview combining price changes with price recency metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset.

        Returns:
            DataFrame: dataframe with per-fund prices, changes, and recency metrics.
        """
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        if fund is not None:
            funds = [fund]
        report = self.get_latest_price_report_per_fund(funds=funds)
        recency = self.get_price_recency(
            funds=funds, reference_date=reference_date
        ).rename(columns={"as_of": "recency_as_of"})
        overview = report.join(recency[["recency_as_of", "days_since"]], how="left")
        return overview

    def get_fund_overview_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the per-fund overview in long (tidy) format for dashboarding.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, recency_as_of, days_since, and
                metric/value columns.
        """
        overview = self.get_fund_overview(
            fund=fund, funds=funds, reference_date=reference_date
        ).reset_index()
        long_overview = overview.melt(
            id_vars=["fund", "as_of", "previous_as_of", "recency_as_of", "days_since"],
            var_name="metric",
            value_name="value",
        )
        return long_overview.dropna(subset=["value"])

    def get_fund_overview_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets a JSON-friendly per-fund overview with price changes and recency metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with overall as_of date, reference_date, and per-fund overview data.
        """
        overview = self.get_fund_overview(
            fund=fund, funds=funds, reference_date=reference_date
        )
        if overview.empty:
            raise ValueError("no price data available for fund overview")
        payload: dict[str, dict] = {}
        for fund_name, row in overview.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
                "recency_as_of": self._format_date_for_output(
                    row["recency_as_of"], date_format
                ),
                "days_since": int(row["days_since"]),
            }
        overall_as_of = self._format_date_for_output(
            overview["as_of"].max(), date_format
        )
        if reference_date is None:
            reference_date_value = self.latest
        else:
            reference_date_value = to_datetime(reference_date).date()
        reference_date_value = self._format_date_for_output(
            reference_date_value, date_format
        )
        return {
            "as_of": overall_as_of,
            "reference_date": reference_date_value,
            "funds": payload,
        }

    def get_current_fund_overview(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) per-fund overview with recency metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, uses the most recent available
                prices on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset (or the as_of date when provided).

        Returns:
            DataFrame: dataframe with per-fund prices, changes, and recency metrics.
        """
        if as_of is None:
            return self.get_fund_overview(
                fund=fund, funds=funds, reference_date=reference_date
            )
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        if fund is not None:
            funds = [fund]

        as_of_value = to_datetime(as_of).date()
        if reference_date is not None:
            reference_date_value = to_datetime(reference_date).date()
            if reference_date_value < as_of_value:
                raise ValueError("reference_date cannot be earlier than as_of")
        else:
            reference_date_value = as_of_value

        report = self.get_price_changes_as_of_per_fund(as_of_value, funds=funds)
        recency = self.get_price_recency(
            funds=funds, reference_date=reference_date_value
        ).rename(columns={"as_of": "recency_as_of"})
        overview = report.join(recency[["recency_as_of", "days_since"]], how="left")
        return overview

    def get_current_fund_overview_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) per-fund overview in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, uses the most recent available
                prices on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset (or the as_of date when provided).

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, recency_as_of, days_since, and
                metric/value columns.
        """
        return (
            self.get_current_fund_overview(
                fund=fund, funds=funds, as_of=as_of, reference_date=reference_date
            )
            .reset_index()
            .melt(
                id_vars=[
                    "fund",
                    "as_of",
                    "previous_as_of",
                    "recency_as_of",
                    "days_since",
                ],
                var_name="metric",
                value_name="value",
            )
            .dropna(subset=["value"])
        )

    def get_current_fund_overview_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets the current (latest available) per-fund overview as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, uses the most recent available
                prices on or before the requested date.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset (or the as_of date when provided).
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with overall as_of date, reference_date, and per-fund overview data.
        """
        overview = self.get_current_fund_overview(
            fund=fund, funds=funds, as_of=as_of, reference_date=reference_date
        )
        if overview.empty:
            raise ValueError("no price data available for fund overview")
        payload: dict[str, dict] = {}
        for fund_name, row in overview.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
                "recency_as_of": self._format_date_for_output(
                    row["recency_as_of"], date_format
                ),
                "days_since": int(row["days_since"]),
            }
        overall_as_of = self._format_date_for_output(
            overview["as_of"].max(), date_format
        )
        if reference_date is None:
            reference_date_value = as_of if as_of is not None else self.latest
        else:
            reference_date_value = to_datetime(reference_date).date()
        reference_date_value = self._format_date_for_output(
            reference_date_value, date_format
        )
        response = {
            "as_of": overall_as_of,
            "reference_date": reference_date_value,
            "funds": payload,
        }
        if as_of is not None:
            response["requested_as_of"] = self._format_date_for_output(
                as_of, date_format
            )
        return response

    def get_latest_price_report(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets a combined latest price report with prices and daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe containing as-of dates and daily change metrics indexed by fund.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        latest_date, latest, previous = self._resolve_latest_and_previous(price_df)
        report = self._build_price_change_frame(
            latest=latest, previous=previous
        ).rename_axis("fund")
        report.insert(0, "as_of", latest_date.date())
        report.insert(1, "previous_as_of", price_df.index[-2].date())
        return report

    def get_latest_price_report_long(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price report in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, and change metrics.
        """
        return self.get_latest_price_report(fund=fund, funds=funds).reset_index()

    def get_current_price_report(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) price report with daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe containing as-of dates and daily change metrics indexed by fund.
        """
        if as_of is None:
            return self.get_latest_price_report(fund=fund, funds=funds)
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        target_date = to_datetime(as_of).normalize()
        filtered = price_df[price_df.index <= target_date]
        if filtered.empty:
            raise ValueError("no price data available on or before the requested date")
        latest_date, latest, previous = self._resolve_latest_and_previous(filtered)
        report = self._build_price_change_frame(
            latest=latest, previous=previous
        ).rename_axis("fund")
        report.insert(0, "as_of", latest_date.date())
        report.insert(1, "previous_as_of", filtered.index[-2].date())
        return report

    def get_current_price_report_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the current price report in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, and change metrics.
        """
        if as_of is None:
            return self.get_latest_price_report_long(fund=fund, funds=funds)
        return self.get_current_price_report(
            fund=fund, funds=funds, as_of=as_of
        ).reset_index()

    def get_current_price_changes_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        as_of: date | None = None,
    ) -> dict:
        """
        Gets the current (latest available) price change metrics as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            as_of (date | None): optional historical anchor date.

        Returns:
            dict: dictionary with as_of dates and per-fund change metrics.
        """
        if as_of is not None:
            return self.get_price_changes_as_of_dict(
                as_of=as_of, fund=fund, funds=funds, date_format=date_format
            )
        return self.get_latest_price_changes_dict(
            fund=fund, funds=funds, date_format=date_format
        )

    def get_current_price_report_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        include_cache_status: bool = False,
        include_data_quality: bool = False,
        as_of: date | None = None,
    ) -> dict:
        """
        Gets a combined current price report with prices and daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            include_cache_status (bool): whether to include cache metadata.
            include_data_quality (bool): whether to include a data-quality summary payload.
            as_of (date | None): optional historical anchor date.

        Returns:
            dict: dictionary with as-of dates, current prices, and daily change metrics.
        """
        if as_of is None:
            return self.get_latest_price_report_dict(
                fund=fund,
                funds=funds,
                date_format=date_format,
                include_cache_status=include_cache_status,
                include_data_quality=include_data_quality,
            )
        prices_payload = self.get_prices_as_of_dict(
            as_of=as_of, fund=fund, funds=funds, date_format=date_format
        )
        changes_payload = self.get_price_changes_as_of_dict(
            as_of=as_of, fund=fund, funds=funds, date_format=date_format
        )
        payload = {
            "as_of": changes_payload["as_of"],
            "previous_as_of": changes_payload["previous_as_of"],
            "requested_as_of": self._format_date_for_output(as_of, date_format),
            "prices": prices_payload["prices"],
            "changes": changes_payload["funds"],
        }
        if include_cache_status:
            datetime_format = None if date_format is None else date_format
            payload["cache_status"] = self.get_cache_status_dict(
                date_format=date_format, datetime_format=datetime_format
            )
        if include_data_quality:
            datetime_format = None if date_format is None else date_format
            payload["data_quality"] = self.get_data_quality_report_dict(
                include_cache_status=False,
                date_format=date_format,
                datetime_format=datetime_format,
            )
        return payload

    def get_current_price_report_per_fund(
        self, funds: Iterable[FundInput] | None = None, as_of: date | None = None
    ) -> DataFrame:
        """
        Gets the current (latest available) per-fund price report.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with per-fund as_of, previous_as_of, prices, and change metrics.
        """
        if as_of is None:
            return self.get_latest_price_report_per_fund(funds=funds)
        return self.get_price_changes_as_of_per_fund(as_of, funds=funds)

    def get_current_price_report_per_fund_long(
        self, funds: Iterable[FundInput] | None = None, as_of: date | None = None
    ) -> DataFrame:
        """
        Gets the current (latest available) per-fund price report in long format.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, and change metrics columns.
        """
        return self.get_current_price_report_per_fund(
            funds=funds, as_of=as_of
        ).reset_index()

    def get_current_price_report_per_fund_dict(
        self,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        as_of: date | None = None,
    ) -> dict:
        """
        Gets the current (latest available) per-fund price report as a dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            as_of (date | None): optional historical anchor date.

        Returns:
            dict: dictionary with an overall latest as_of date and per-fund metrics.
        """
        report = self.get_current_price_report_per_fund(funds=funds, as_of=as_of)
        if report.empty:
            raise ValueError("no price data available for per-fund price report")
        payload: dict[str, dict] = {}
        for fund_name, row in report.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        overall_as_of = self._format_date_for_output(report["as_of"].max(), date_format)
        response = {"as_of": overall_as_of, "funds": payload}
        if as_of is not None:
            response["requested_as_of"] = self._format_date_for_output(
                as_of, date_format
            )
        return response

    def get_latest_price_changes_per_fund(
        self, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price change per fund using each fund's last two valid prices.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe indexed by fund with as_of, previous_as_of, and change metrics.
        """
        self._ensure_dataframe()
        fund_list = self._resolve_funds(funds)
        self._ensure_funds_available(fund_list)
        records: list[dict] = []
        insufficient: list[str] = []
        for fund_name in fund_list:
            series = (
                self.dataframe[["Date", fund_name]]
                .dropna(subset=[fund_name])
                .sort_values("Date")
            )
            if len(series) < 2:
                insufficient.append(fund_name)
                continue
            latest_row = series.iloc[-1]
            previous_row = series.iloc[-2]
            latest_price = latest_row[fund_name]
            previous_price = previous_row[fund_name]
            change = latest_price - previous_price
            if previous_price == 0:
                change_percent = float("nan")
            else:
                change_percent = change / previous_price
            records.append(
                {
                    "fund": fund_name,
                    "as_of": latest_row["Date"].date(),
                    "previous_as_of": previous_row["Date"].date(),
                    "latest_price": latest_price,
                    "previous_price": previous_price,
                    "change": change,
                    "change_percent": change_percent,
                }
            )
        if insufficient:
            raise ValueError(
                "at least two data points are required to calculate price changes for: "
                f'{", ".join(insufficient)}'
            )
        return DataFrame(records).set_index("fund")

    def get_current_price_changes_per_fund(
        self, funds: Iterable[FundInput] | None = None, as_of: date | None = None
    ) -> DataFrame:
        """
        Gets the current (latest available) price changes per fund using last two valid prices.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe indexed by fund with as_of, previous_as_of, and change metrics.
        """
        if as_of is not None:
            return self.get_price_changes_as_of_per_fund(as_of=as_of, funds=funds)
        return self.get_latest_price_changes_per_fund(funds=funds)

    def get_latest_price_changes_per_fund_long(
        self, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price changes per fund in long (tidy) format.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, latest_price, previous_price, change, change_percent.
        """
        return self.get_latest_price_changes_per_fund(funds=funds).reset_index()

    def get_current_price_changes_per_fund_long(
        self, funds: Iterable[FundInput] | None = None, as_of: date | None = None
    ) -> DataFrame:
        """
        Gets the current (latest available) price changes per fund in long (tidy) format.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, latest_price, previous_price, change, change_percent.
        """
        if as_of is not None:
            return self.get_price_changes_as_of_per_fund_long(as_of=as_of, funds=funds)
        return self.get_latest_price_changes_per_fund_long(funds=funds)

    def get_latest_price_changes_per_fund_dict(
        self, funds: Iterable[FundInput] | None = None, date_format: str | None = "iso"
    ) -> dict:
        """
        Gets the latest price changes per fund as a JSON-friendly dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with per-fund price change metrics and as-of dates.
        """
        changes = self.get_latest_price_changes_per_fund(funds=funds)
        if changes.empty:
            raise ValueError("no price data available for latest per-fund changes")
        payload: dict[str, dict] = {}
        for fund_name, row in changes.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {"funds": payload}

    def get_current_price_changes_per_fund_dict(
        self,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        as_of: date | None = None,
    ) -> dict:
        """
        Gets the current (latest available) price changes per fund as a JSON-friendly dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            as_of (date | None): optional historical anchor date.

        Returns:
            dict: dictionary with per-fund price change metrics and as-of dates.
        """
        if as_of is not None:
            return self.get_price_changes_as_of_per_fund_dict(
                as_of=as_of, funds=funds, date_format=date_format
            )
        return self.get_latest_price_changes_per_fund_dict(
            funds=funds, date_format=date_format
        )

    def get_current_price_changes_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) price changes in a long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with fund, latest_price, previous_price, change, change_percent.
        """
        if as_of is not None:
            return self.get_price_changes_as_of_long(
                as_of=as_of, fund=fund, funds=funds
            )
        return self.get_latest_price_changes_long(fund=fund, funds=funds)

    def get_current_price_snapshot(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) price snapshot with daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with as_of date, latest_price, previous_price, change, change_percent.
        """
        if as_of is None:
            return self.get_latest_price_snapshot(fund=fund, funds=funds)
        return self.get_price_snapshot_as_of(as_of=as_of, fund=fund, funds=funds)

    def get_current_price_snapshot_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the current (latest available) price snapshot in a long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.

        Returns:
            DataFrame: dataframe with fund, as_of, latest_price, previous_price, change, change_percent.
        """
        if as_of is None:
            return self.get_latest_price_snapshot_long(fund=fund, funds=funds)
        return self.get_price_snapshot_as_of_long(as_of=as_of, fund=fund, funds=funds)

    def get_current_price_snapshot_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        as_of: date | None = None,
    ) -> dict:
        """
        Gets the current (latest available) price snapshot as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return a date object.
            as_of (date | None): optional historical anchor date.

        Returns:
            dict: dictionary with as_of date and per-fund change metrics.
        """
        if as_of is None:
            return self.get_latest_price_snapshot_dict(
                fund=fund, funds=funds, date_format=date_format
            )
        return self.get_price_snapshot_as_of_dict(
            as_of=as_of, fund=fund, funds=funds, date_format=date_format
        )

    def get_latest_price_changes(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price change and percent change from the previous trading day.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe containing latest price changes indexed by fund.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        latest_date, latest, previous = self._resolve_latest_and_previous(price_df)
        return self._build_price_change_frame(
            latest=latest, previous=previous
        ).rename_axis("fund")

    def get_latest_price_changes_long(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price changes in a long (tidy) format with a fund column.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with fund, latest_price, previous_price, change, change_percent.
        """
        return self.get_latest_price_changes(fund=fund, funds=funds).reset_index()

    def get_recent_price_changes(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets daily percent changes over the most recent trading days.

        Args:
            days (int): number of recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before this date.

        Returns:
            DataFrame: dataframe indexed by Date with daily percent changes per fund.
        """
        self._ensure_dataframe()
        self._validate_positive_int(days, "days")
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        if as_of is not None:
            target_date = to_datetime(as_of).normalize()
            price_df = price_df[price_df.index <= target_date]
        if price_df.empty:
            raise ValueError("no price data available for requested date range")
        if len(price_df.index) < 2:
            raise ValueError(
                "at least two data points are required to calculate price changes"
            )
        changes = price_df.pct_change(fill_method=None).dropna(how="all")
        recent = changes.tail(days)
        if recent.empty:
            raise ValueError("no price changes available for requested range")
        return recent

    def get_recent_price_changes_long(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets recent daily percent changes in long (tidy) format.

        Args:
            days (int): number of recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before this date.

        Returns:
            DataFrame: dataframe with Date, fund, and change_percent columns.
        """
        changes = self.get_recent_price_changes(
            days=days, fund=fund, funds=funds, as_of=as_of
        ).reset_index()
        return changes.melt(
            id_vars="Date", var_name="fund", value_name="change_percent"
        ).dropna(subset=["change_percent"])

    def get_recent_price_changes_dict(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets recent daily percent changes as a JSON-friendly dictionary.

        Args:
            days (int): number of recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before this date.
            date_format (str | None): format for the dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with requested date range and per-fund change records.
        """
        changes = self.get_recent_price_changes(
            days=days, fund=fund, funds=funds, as_of=as_of
        )
        start_date = self._format_date_for_output(
            changes.index.min().date(), date_format
        )
        end_date = self._format_date_for_output(changes.index.max().date(), date_format)
        payload: dict[str, list[dict]] = {}
        for fund_name in changes.columns:
            records: list[dict] = []
            series = changes[fund_name].dropna()
            for timestamp, value in series.items():
                records.append(
                    {
                        "date": self._format_date_for_output(
                            timestamp.date(), date_format
                        ),
                        "change_percent": self._format_numeric_for_output(value),
                    }
                )
            payload[fund_name] = records
        return {
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "funds": payload,
        }

    def get_recent_price_change_summary(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets summary statistics for recent daily percent changes.

        Args:
            days (int): number of recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before this date.

        Returns:
            DataFrame: dataframe of summary statistics per fund.
        """
        changes = self.get_recent_price_changes(
            days=days, fund=fund, funds=funds, as_of=as_of
        )
        summary = changes.describe().T
        summary["median"] = changes.median()
        summary["cumulative_return"] = (1 + changes).prod() - 1
        start_date = changes.index.min().date()
        end_date = changes.index.max().date()
        summary["start_date"] = start_date
        summary["end_date"] = end_date
        return summary

    def get_recent_price_change_summary_dict(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets summary statistics for recent daily percent changes as a JSON-friendly dict.

        Args:
            days (int): number of recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before this date.
            date_format (str | None): format for the dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with date range and per-fund summary statistics.
        """
        summary = self.get_recent_price_change_summary(
            days=days, fund=fund, funds=funds, as_of=as_of
        )
        start_date = self._format_date_for_output(
            summary["start_date"].iloc[0], date_format
        )
        end_date = self._format_date_for_output(
            summary["end_date"].iloc[0], date_format
        )
        payload = self._format_dataframe_for_output(summary, date_format=date_format)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "funds": payload,
        }

    def get_latest_price_snapshot(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price snapshot with the as-of date and daily change metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with as_of date, latest_price, previous_price, change, change_percent.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        latest_date, latest, previous = self._resolve_latest_and_previous(price_df)
        return self._build_price_snapshot_frame(
            as_of_date=latest_date.date(), latest=latest, previous=previous
        ).rename_axis("fund")

    def get_latest_price_snapshot_long(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest price snapshot in a long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with fund, as_of, latest_price, previous_price, change, change_percent.
        """
        return self.get_latest_price_snapshot(fund=fund, funds=funds).reset_index()

    def get_latest_price_snapshot_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets the latest price snapshot as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return a date object.

        Returns:
            dict: dictionary with as_of date and per-fund price metrics.
        """
        snapshot = self.get_latest_price_snapshot(fund=fund, funds=funds)
        if snapshot.empty:
            raise ValueError("no price data available for latest snapshot")
        as_of = self._format_date_for_output(snapshot["as_of"].iloc[0], date_format)
        metrics = {}
        for fund_name, row in snapshot.iterrows():
            metrics[fund_name] = {
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {"as_of": as_of, "funds": metrics}

    def get_price_snapshot_as_of(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> DataFrame:
        """
        Gets a price snapshot anchored to the most recent prices on or before a specific date.

        Args:
            as_of (date): date to anchor the price snapshot calculation.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with as_of date, latest_price, previous_price, change, change_percent.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        target_date = to_datetime(as_of).normalize()
        filtered = price_df[price_df.index <= target_date]
        if filtered.empty:
            raise ValueError("no price data available on or before the requested date")
        if len(filtered.index) < 2:
            raise ValueError(
                "at least two data points are required to calculate price changes"
            )
        latest_date, latest, previous = self._resolve_latest_and_previous(filtered)
        return self._build_price_snapshot_frame(
            as_of_date=latest_date.date(), latest=latest, previous=previous
        ).rename_axis("fund")

    def get_price_snapshot_as_of_long(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> DataFrame:
        """
        Gets a price snapshot as of a specific date in long (tidy) format.

        Args:
            as_of (date): date to anchor the price snapshot calculation.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with fund, as_of, latest_price, previous_price, change, change_percent.
        """
        return self.get_price_snapshot_as_of(
            as_of, fund=fund, funds=funds
        ).reset_index()

    def get_price_snapshot_as_of_dict(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets a price snapshot as of a specific date as a JSON-friendly dictionary.

        Args:
            as_of (date): date to anchor the price snapshot calculation.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return a date object.

        Returns:
            dict: dictionary with as_of date and per-fund price metrics.
        """
        snapshot = self.get_price_snapshot_as_of(as_of, fund=fund, funds=funds)
        if snapshot.empty:
            raise ValueError("no price data available for as-of snapshot")
        as_of_date = self._format_date_for_output(
            snapshot["as_of"].iloc[0], date_format
        )
        metrics: dict[str, dict] = {}
        for fund_name, row in snapshot.iterrows():
            metrics[fund_name] = {
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {"as_of": as_of_date, "funds": metrics}

    def get_price_changes_as_of(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> DataFrame:
        """
        Gets price changes and percent changes as of a specific date.

        Args:
            as_of (date): date to anchor the price change calculation.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe containing as-of price changes indexed by fund.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        target_date = to_datetime(as_of).normalize()
        filtered = price_df[price_df.index <= target_date]
        if filtered.empty:
            raise ValueError("no price data available on or before the requested date")
        if len(filtered.index) < 2:
            raise ValueError(
                "at least two data points are required to calculate price changes"
            )
        latest_date, latest, previous = self._resolve_latest_and_previous(filtered)
        return self._build_price_change_frame(
            latest=latest, previous=previous
        ).rename_axis("fund")

    def get_price_changes_as_of_long(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> DataFrame:
        """
        Gets price changes as of a specific date in long (tidy) format.

        Args:
            as_of (date): date to anchor the price change calculation.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with fund, latest_price, previous_price, change, change_percent.
        """
        return self.get_price_changes_as_of(as_of, fund=fund, funds=funds).reset_index()

    def get_price_changes_as_of_dict(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets price changes as of a specific date as a JSON-friendly dictionary.

        Args:
            as_of (date): date to anchor the price change calculation.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with as-of dates and per-fund change metrics.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self._select_price_change_frame(fund=fund, funds=funds)
        target_date = to_datetime(as_of).normalize()
        filtered = price_df[price_df.index <= target_date]
        if filtered.empty:
            raise ValueError("no price data available on or before the requested date")
        if len(filtered.index) < 2:
            raise ValueError(
                "at least two data points are required to calculate price changes"
            )
        latest_date, latest, previous = self._resolve_latest_and_previous(filtered)
        changes = self._build_price_change_frame(
            latest=latest, previous=previous
        ).rename_axis("fund")
        payload: dict[str, dict] = {}
        for fund_name, row in changes.iterrows():
            payload[fund_name] = {
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {
            "as_of": self._format_date_for_output(latest_date.date(), date_format),
            "previous_as_of": self._format_date_for_output(
                filtered.index[-2].date(), date_format
            ),
            "funds": payload,
        }

    def get_price_changes_as_of_per_fund(
        self, as_of: date, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets price changes as of a specific date using each fund's last two valid prices.

        Args:
            as_of (date): date to anchor the price change calculation.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe indexed by fund with as_of, previous_as_of, and change metrics.
        """
        self._ensure_dataframe()
        fund_list = self._resolve_funds(funds)
        self._ensure_funds_available(fund_list)
        target_date = to_datetime(as_of).normalize()
        price_df = self.dataframe[self.dataframe["Date"] <= target_date].sort_values(
            "Date"
        )
        if price_df.empty:
            raise ValueError("no price data available on or before the requested date")
        records: list[dict] = []
        insufficient: list[str] = []
        for fund_name in fund_list:
            series = (
                price_df[["Date", fund_name]]
                .dropna(subset=[fund_name])
                .sort_values("Date")
            )
            if len(series) < 2:
                insufficient.append(fund_name)
                continue
            latest_row = series.iloc[-1]
            previous_row = series.iloc[-2]
            latest_price = latest_row[fund_name]
            previous_price = previous_row[fund_name]
            change = latest_price - previous_price
            change_percent = change / previous_price if previous_price else float("nan")
            records.append(
                {
                    "fund": fund_name,
                    "as_of": latest_row["Date"].date(),
                    "previous_as_of": previous_row["Date"].date(),
                    "latest_price": latest_price,
                    "previous_price": previous_price,
                    "change": change,
                    "change_percent": change_percent,
                }
            )
        if insufficient:
            raise ValueError(
                "at least two data points are required to calculate price changes for: "
                f'{", ".join(insufficient)}'
            )
        return DataFrame(records).set_index("fund")

    def get_price_changes_as_of_per_fund_long(
        self, as_of: date, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets price changes as of a specific date in long (tidy) format per fund.

        Args:
            as_of (date): date to anchor the price change calculation.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe with fund, as_of, previous_as_of, and change metrics columns.
        """
        return self.get_price_changes_as_of_per_fund(as_of, funds=funds).reset_index()

    def get_price_changes_as_of_per_fund_dict(
        self,
        as_of: date,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets price changes as of a specific date per fund as a JSON-friendly dictionary.

        Args:
            as_of (date): date to anchor the price change calculation.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with requested_as_of date and per-fund change metrics.
        """
        changes = self.get_price_changes_as_of_per_fund(as_of, funds=funds)
        if changes.empty:
            raise ValueError("no price data available for as-of per-fund changes")
        payload: dict[str, dict] = {}
        for fund_name, row in changes.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "previous_as_of": self._format_date_for_output(
                    row["previous_as_of"], date_format
                ),
                "latest_price": self._format_numeric_for_output(row["latest_price"]),
                "previous_price": self._format_numeric_for_output(
                    row["previous_price"]
                ),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {
            "requested_as_of": self._format_date_for_output(as_of, date_format),
            "funds": payload,
        }

    def _select_price_change_frame(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        price_df = (
            self.dataframe.sort_values("Date").set_index("Date").dropna(how="all")
        )
        if funds is not None:
            fund_list = self._resolve_funds(funds)
            self._ensure_funds_available(fund_list)
            price_df = price_df[fund_list]
        elif fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        else:
            available = self.get_available_funds()
            if not available:
                raise ValueError("no available funds in data")
            price_df = price_df[available]
        if price_df.empty or price_df.shape[1] == 0:
            raise ValueError("no price data available for requested funds")
        return price_df

    @staticmethod
    def _resolve_latest_and_previous(
        price_df: DataFrame,
    ) -> tuple[datetime, Series, Series]:
        if len(price_df.index) < 2:
            raise ValueError(
                "at least two data points are required to calculate price changes"
            )
        latest_date = price_df.index[-1]
        latest = price_df.iloc[-1]
        previous = price_df.iloc[-2]
        return latest_date, latest, previous

    @staticmethod
    def _build_price_change_frame(latest: Series, previous: Series) -> DataFrame:
        change = latest - previous
        change_percent = change.div(previous.replace(0, float("nan")))
        return DataFrame(
            {
                "latest_price": latest,
                "previous_price": previous,
                "change": change,
                "change_percent": change_percent,
            }
        )

    @staticmethod
    def _build_price_snapshot_frame(
        as_of_date: date, latest: Series, previous: Series
    ) -> DataFrame:
        change = latest - previous
        change_percent = change.div(previous.replace(0, float("nan")))
        return DataFrame(
            {
                "as_of": as_of_date,
                "latest_price": latest,
                "previous_price": previous,
                "change": change,
                "change_percent": change_percent,
            }
        )

    def get_price_change_by_date_range(
        self, start_date: date, end_date: date, fund: FundInput | None = None
    ) -> DataFrame:
        """
        Gets price changes between the first and last trading day in a date range.

        Args:
            start_date (date): the start date of the range.
            end_date (date): the end date of the range.
            fund (FundInput | None): optional fund to limit the output.

        Returns:
            DataFrame: dataframe with start/end prices and change information.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        filtered = self._filter_by_date_range(
            start_date, end_date, dataframe=price_df.reset_index()
        )
        if filtered.empty:
            raise ValueError("no price data available for requested date range")
        filtered = filtered.set_index("Date")
        start_prices = filtered.iloc[0]
        end_prices = filtered.iloc[-1]
        change = end_prices - start_prices
        change_percent = change.div(start_prices.replace(0, float("nan")))
        result = DataFrame(
            {
                "start_price": start_prices,
                "end_price": end_prices,
                "change": change,
                "change_percent": change_percent,
            }
        )
        result.index.name = "fund"
        return result

    def get_price_change_by_date_range_long(
        self, start_date: date, end_date: date, fund: FundInput | None = None
    ) -> DataFrame:
        """
        Gets price changes between the first and last trading day in a date range in tidy format.

        Args:
            start_date (date): the start date of the range.
            end_date (date): the end date of the range.
            fund (FundInput | None): optional fund to limit the output.

        Returns:
            DataFrame: dataframe with fund, start/end prices, and change information.
        """
        changes = self.get_price_change_by_date_range(start_date, end_date, fund=fund)
        return changes.reset_index()

    def get_price_change_by_date_range_dict(
        self,
        start_date: date,
        end_date: date,
        fund: FundInput | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets price changes between the first and last trading day in a date range as a dict.

        Args:
            start_date (date): the start date of the range.
            end_date (date): the end date of the range.
            fund (FundInput | None): optional fund to limit the output.
            date_format (str | None): format for the start/end dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with start/end dates and per-fund price change metrics.
        """
        changes = self.get_price_change_by_date_range(start_date, end_date, fund=fund)
        payload: dict[str, dict] = {}
        for fund_name, row in changes.iterrows():
            payload[fund_name] = {
                "start_price": self._format_numeric_for_output(row["start_price"]),
                "end_price": self._format_numeric_for_output(row["end_price"]),
                "change": self._format_numeric_for_output(row["change"]),
                "change_percent": self._format_numeric_for_output(
                    row["change_percent"]
                ),
            }
        return {
            "start_date": self._format_date_for_output(start_date, date_format),
            "end_date": self._format_date_for_output(end_date, date_format),
            "funds": payload,
        }
