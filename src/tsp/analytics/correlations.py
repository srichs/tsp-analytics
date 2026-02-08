"""Correlation analytics for fund return data."""

from pandas import DataFrame, isna


class AnalyticsCorrelationMixin:
    """Compute correlations across fund return series."""

    def get_correlation_matrix(self) -> DataFrame:
        """
        Gets the correlation matrix of daily returns across all funds.

        Returns:
            DataFrame: correlation matrix for daily returns.
        """
        returns = self.get_daily_returns()
        if returns.empty:
            raise ValueError("no return data available for correlation matrix")
        return returns.corr()

    def get_correlation_matrix_dict(self, date_format: str | None = "iso") -> dict:
        """
        Gets the correlation matrix as a JSON-friendly dictionary.

        Args:
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with start/end dates and nested correlation values.
        """
        self._ensure_dataframe()
        returns = self.get_daily_returns()
        if returns.empty:
            raise ValueError("no return data available for correlation matrix")
        correlation = returns.corr()
        if correlation.empty:
            raise ValueError("no correlation data available")
        start_date = returns.index.min().date()
        end_date = returns.index.max().date()
        payload: dict[str, dict] = {}
        for fund_name, row in correlation.iterrows():
            payload[fund_name] = {
                other_fund: self._format_numeric_for_output(value)
                for other_fund, value in row.items()
            }
        return {
            "start_date": self._format_date_for_output(start_date, date_format),
            "end_date": self._format_date_for_output(end_date, date_format),
            "correlations": payload,
        }

    def get_correlation_matrix_long(self) -> DataFrame:
        """
        Gets the correlation matrix in long (tidy) format.

        Returns:
            DataFrame: dataframe with fund_a, fund_b, and correlation columns.
        """
        correlation = self.get_correlation_matrix()
        long = correlation.stack().reset_index()
        long.columns = ["fund_a", "fund_b", "correlation"]
        return long.dropna(subset=["correlation"])

    def get_correlation_pairs(
        self,
        *,
        top_n: int | None = 10,
        absolute: bool = True,
        min_abs_correlation: float | None = None,
    ) -> DataFrame:
        """
        Gets the strongest correlation pairs between funds.

        Args:
            top_n (int | None): optional limit on the number of pairs returned.
            absolute (bool): when True, rank by absolute correlation magnitude.
            min_abs_correlation (float | None): optional threshold between 0 and 1 to filter pairs.

        Returns:
            DataFrame: dataframe with fund pairs and correlation values.
        """
        self._ensure_dataframe()
        if top_n is not None:
            self._validate_positive_int(top_n, "top_n")
        if min_abs_correlation is not None:
            self._validate_numeric(min_abs_correlation, "min_abs_correlation")
            if not 0 <= float(min_abs_correlation) <= 1:
                raise ValueError("min_abs_correlation must be between 0 and 1")

        correlation = self.get_correlation_matrix()
        if correlation.empty:
            raise ValueError("no correlation data available")

        funds = list(correlation.columns)
        records: list[dict] = []
        for idx, fund_a in enumerate(funds):
            for fund_b in funds[idx + 1 :]:
                value = correlation.loc[fund_a, fund_b]
                if isna(value):
                    continue
                value_float = float(value)
                abs_value = abs(value_float)
                if min_abs_correlation is not None and abs_value < float(
                    min_abs_correlation
                ):
                    continue
                records.append(
                    {
                        "fund_a": fund_a,
                        "fund_b": fund_b,
                        "correlation": value_float,
                        "abs_correlation": abs_value,
                    }
                )

        if not records:
            raise ValueError("no correlation pairs available")

        pairs = DataFrame(records)
        sort_key = "abs_correlation" if absolute else "correlation"
        pairs = pairs.sort_values(sort_key, ascending=False).reset_index(drop=True)
        if top_n is not None:
            pairs = pairs.head(top_n).reset_index(drop=True)
        return pairs

    def get_correlation_pairs_dict(
        self,
        *,
        top_n: int | None = 10,
        absolute: bool = True,
        min_abs_correlation: float | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets correlation pairs as a JSON-friendly dictionary.

        Args:
            top_n (int | None): optional limit on the number of pairs returned.
            absolute (bool): when True, rank by absolute correlation magnitude.
            min_abs_correlation (float | None): optional threshold between 0 and 1 to filter pairs.
            date_format (str | None): format for start/end dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with date range metadata and correlation pair records.
        """
        returns = self.get_daily_returns()
        if returns.empty:
            raise ValueError("no return data available for correlation pairs")
        pairs = self.get_correlation_pairs(
            top_n=top_n, absolute=absolute, min_abs_correlation=min_abs_correlation
        )
        start_date = returns.index.min().date()
        end_date = returns.index.max().date()
        payload: list[dict] = []
        for _, row in pairs.iterrows():
            payload.append(
                {
                    "fund_a": row["fund_a"],
                    "fund_b": row["fund_b"],
                    "correlation": self._format_numeric_for_output(row["correlation"]),
                    "abs_correlation": self._format_numeric_for_output(
                        row["abs_correlation"]
                    ),
                }
            )
        return {
            "start_date": self._format_date_for_output(start_date, date_format),
            "end_date": self._format_date_for_output(end_date, date_format),
            "pairs": payload,
        }

    def get_rolling_correlation_matrix(self, window: int = 63) -> DataFrame:
        """
        Gets a correlation matrix for the most recent rolling window of daily returns.

        Args:
            window (int): number of recent trading days to include.

        Returns:
            DataFrame: correlation matrix for the specified window.
        """
        self._ensure_dataframe()
        self._validate_positive_int(window, "window")
        returns = self.get_daily_returns()
        if returns.empty or len(returns.index) < window:
            raise ValueError(
                "not enough return data available for rolling correlation matrix"
            )
        return returns.tail(window).corr()

    def get_rolling_correlation_matrix_long(self, window: int = 63) -> DataFrame:
        """
        Gets a rolling correlation matrix in long (tidy) format.

        Args:
            window (int): number of recent trading days to include.

        Returns:
            DataFrame: dataframe with fund_a, fund_b, and correlation columns.
        """
        correlation = self.get_rolling_correlation_matrix(window=window)
        long = correlation.stack().reset_index()
        long.columns = ["fund_a", "fund_b", "correlation"]
        return long.dropna(subset=["correlation"])

    def get_rolling_correlation_matrix_dict(
        self, window: int = 63, date_format: str | None = "iso"
    ) -> dict:
        """
        Gets the rolling correlation matrix as a JSON-friendly dictionary.

        Args:
            window (int): number of recent trading days to include.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with window metadata and nested correlation values.
        """
        self._ensure_dataframe()
        self._validate_positive_int(window, "window")
        returns = self.get_daily_returns()
        if returns.empty or len(returns.index) < window:
            raise ValueError(
                "not enough return data available for rolling correlation matrix"
            )
        window_returns = returns.tail(window)
        correlation = window_returns.corr()
        if correlation.empty:
            raise ValueError("no correlation data available")
        start_date = window_returns.index.min().date()
        end_date = window_returns.index.max().date()
        payload: dict[str, dict] = {}
        for fund_name, row in correlation.iterrows():
            payload[fund_name] = {
                other_fund: self._format_numeric_for_output(value)
                for other_fund, value in row.items()
            }
        return {
            "window": int(window),
            "start_date": self._format_date_for_output(start_date, date_format),
            "end_date": self._format_date_for_output(end_date, date_format),
            "correlations": payload,
        }
