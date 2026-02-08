"""Analytics report builders for summarizing TSP fund performance."""

from collections.abc import Iterable
from datetime import date

from pandas import DataFrame

from tsp.funds import FundInput


class AnalyticsReportsMixin:
    """Build summary report tables for returns, risk, and correlations."""

    def get_fund_analytics_report(
        self,
        fund: FundInput,
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
    ) -> dict[str, DataFrame]:
        """
        Builds a consolidated analytics report for a single fund.

        Args:
            fund (FundInput): fund to summarize.
            start_date (date | None): optional start date for date-range metrics.
            end_date (date | None): optional end date for date-range metrics.
            trading_days (int): trading days per year used for annualization.

        Returns:
            dict[str, DataFrame]: dictionary of analytics tables keyed by report name.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(trading_days, "trading_days")

        if start_date is not None or end_date is not None:
            history = self.get_price_history(
                fund=fund_name, start_date=start_date, end_date=end_date
            )
            resolved_start = history["Date"].min().date()
            resolved_end = history["Date"].max().date()
            performance = self.get_performance_summary_by_date_range(
                resolved_start, resolved_end, fund=fund_name, trading_days=trading_days
            )
        else:
            history = self.get_price_history(fund=fund_name)
            resolved_start = history["Date"].min().date()
            resolved_end = history["Date"].max().date()
            performance = self.get_performance_summary(
                fund=fund_name, trading_days=trading_days
            )

        return {
            "price_statistics": self.get_price_statistics(
                fund=fund_name, start_date=start_date, end_date=end_date
            ),
            "return_statistics": self.get_return_statistics(
                fund=fund_name,
                start_date=start_date,
                end_date=end_date,
                trading_days=trading_days,
            ),
            "performance_summary": performance,
            "drawdown_summary": self.get_drawdown_summary(fund=fund_name),
            "price_summary": self.get_price_summary(funds=[fund_name]),
            "current_overview": self.get_current_fund_overview(funds=[fund_name]),
            "date_range": DataFrame(
                [
                    {
                        "start_date": resolved_start,
                        "end_date": resolved_end,
                    }
                ],
                index=["date_range"],
            ),
        }

    def get_fund_analytics_report_dict(
        self,
        fund: FundInput,
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Builds a consolidated analytics report for a single fund as a dictionary.

        Args:
            fund (FundInput): fund to summarize.
            start_date (date | None): optional start date for date-range metrics.
            end_date (date | None): optional end date for date-range metrics.
            trading_days (int): trading days per year used for annualization.
            date_format (str | None): format for date values. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary payload with analytics summaries and metadata.
        """
        report = self.get_fund_analytics_report(
            fund=fund,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
        )
        date_range = report["date_range"].iloc[0]
        return {
            "fund": self._ensure_fund_available(fund),
            "date_range": {
                "start_date": self._format_date_for_output(
                    date_range["start_date"], date_format
                ),
                "end_date": self._format_date_for_output(
                    date_range["end_date"], date_format
                ),
            },
            "price_statistics": self._format_dataframe_for_output(
                report["price_statistics"], date_format=date_format
            ),
            "return_statistics": self._format_dataframe_for_output(
                report["return_statistics"], date_format=date_format
            ),
            "performance_summary": self._format_dataframe_for_output(
                report["performance_summary"], date_format=date_format
            ),
            "drawdown_summary": self._format_dataframe_for_output(
                report["drawdown_summary"], date_format=date_format
            ),
            "price_summary": self._format_dataframe_for_output(
                report["price_summary"], date_format=date_format
            ),
            "current_overview": self._format_dataframe_for_output(
                report["current_overview"], date_format=date_format
            ),
        }

    def get_performance_summary(
        self, fund: FundInput | None = None, trading_days: int = 252
    ) -> DataFrame:
        """
        Gets a performance summary for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to summarize. If None, includes all funds.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe of summary statistics.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is None:
            summaries = {}
            for column in price_df.columns:
                summaries[column] = self._calculate_performance_summary(
                    price_df[column], trading_days
                )
            return DataFrame.from_dict(summaries, orient="index")

        fund_name = self._ensure_fund_available(fund)
        return DataFrame(
            [self._calculate_performance_summary(price_df[fund_name], trading_days)],
            index=[fund_name],
        )

    def get_performance_summary_dict(
        self, fund: FundInput | None = None, trading_days: int = 252
    ) -> dict:
        """
        Gets a performance summary as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            trading_days (int): trading days per year used for annualization.

        Returns:
            dict: dictionary with performance summary metrics keyed by fund.
        """
        summary = self.get_performance_summary(fund=fund, trading_days=trading_days)
        if summary.empty:
            raise ValueError("no performance data available")
        return {
            "trading_days": trading_days,
            "funds": self._format_dataframe_for_output(summary),
        }

    def get_performance_summary_by_date_range(
        self,
        start_date: date,
        end_date: date,
        fund: FundInput | None = None,
        trading_days: int = 252,
    ) -> DataFrame:
        """
        Gets a performance summary for the specified fund or all funds over a date range.

        Args:
            start_date (date): the start date to calculate performance for.
            end_date (date): the end date to calculate performance for.
            fund (FundInput | None): the fund to summarize. If None, includes all funds.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe of summary statistics for the specified date range.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        filtered = self._filter_by_date_range(
            start_date, end_date, dataframe=price_df.reset_index()
        )
        filtered = filtered.set_index("Date")
        if fund is None:
            summaries = {}
            for column in filtered.columns:
                summaries[column] = self._calculate_performance_summary(
                    filtered[column], trading_days
                )
            return DataFrame.from_dict(summaries, orient="index")

        fund_name = self._ensure_fund_available(fund)
        return DataFrame(
            [self._calculate_performance_summary(filtered[fund_name], trading_days)],
            index=[fund_name],
        )

    def get_performance_summary_by_date_range_dict(
        self,
        start_date: date,
        end_date: date,
        fund: FundInput | None = None,
        trading_days: int = 252,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets a performance summary for a date range as a JSON-friendly dictionary.

        Args:
            start_date (date): the start date to calculate performance for.
            end_date (date): the end date to calculate performance for.
            fund (FundInput | None): optional fund to limit the output.
            trading_days (int): trading days per year used for annualization.
            date_format (str | None): format for date outputs.

        Returns:
            dict: dictionary with date range metadata and performance summary metrics.
        """
        summary = self.get_performance_summary_by_date_range(
            start_date=start_date,
            end_date=end_date,
            fund=fund,
            trading_days=trading_days,
        )
        if summary.empty:
            raise ValueError("no performance data available for date range")
        return {
            "start_date": self._format_date_for_output(start_date, date_format),
            "end_date": self._format_date_for_output(end_date, date_format),
            "trading_days": trading_days,
            "funds": self._format_dataframe_for_output(summary),
        }

    def get_fund_rankings(
        self,
        metric: str,
        period: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        trading_days: int = 252,
        top_n: int | None = None,
        funds: Iterable[FundInput] | None = None,
        ascending: bool | None = None,
    ) -> DataFrame:
        """
        Ranks funds by a specified performance metric.

        Args:
            metric (str): metric name to rank by. Supported values include
                "total_return", "annualized_return", "annualized_volatility",
                "sharpe_ratio", "max_drawdown", "cagr", "trailing_return",
                "change", "change_percent", "latest_price", or "days_since".
            period (int | None): trailing-return period (required when metric is "trailing_return").
            start_date (date | None): optional start date for performance-based rankings.
            end_date (date | None): optional end date for performance-based rankings.
            as_of (date | None): optional historical anchor date for current price rankings.
            reference_date (date | None): optional reference date for days_since rankings.
            trading_days (int): trading days per year used for annualization.
            top_n (int | None): optional limit on the number of ranked funds returned.
            funds (Iterable[FundInput] | None): optional collection of funds to include.
            ascending (bool | None): override the default sort direction for the metric.

        Returns:
            DataFrame: dataframe indexed by fund with ranking metadata.
        """
        self._ensure_dataframe()
        metric_key = metric.strip().lower().replace(" ", "_")
        supported_metrics = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "cagr",
            "trailing_return",
            "change",
            "change_percent",
            "latest_price",
            "days_since",
        }
        if metric_key not in supported_metrics:
            raise ValueError(f"unsupported ranking metric: {metric}")
        if metric_key == "trailing_return":
            if period is None:
                raise ValueError("period is required when ranking by trailing_return")
            self._validate_positive_int(period, "period")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        if top_n is not None:
            self._validate_positive_int(top_n, "top_n")

        if metric_key in {"change", "change_percent", "latest_price", "days_since"}:
            overview = self.get_current_fund_overview(
                as_of=as_of, reference_date=reference_date
            )
            series = overview[metric_key]
        elif metric_key == "trailing_return":
            trailing = self.get_trailing_returns(periods=[period])
            series = trailing.iloc[0]
        elif metric_key == "cagr":
            summary = self.get_cagr(start_date=start_date, end_date=end_date)
            series = summary["cagr"]
        else:
            if start_date is not None and end_date is not None:
                summary = self.get_performance_summary_by_date_range(
                    start_date=start_date, end_date=end_date, trading_days=trading_days
                )
            else:
                summary = self.get_performance_summary(trading_days=trading_days)
            series = summary[metric_key]

        if funds is not None:
            fund_list = self._resolve_funds(funds)
            self._ensure_funds_available(fund_list)
            series = series.loc[fund_list]

        rankings = DataFrame({"value": series}).dropna(subset=["value"])
        if rankings.empty:
            raise ValueError("no data available to rank funds for the selected metric")

        default_ascending = {"annualized_volatility": True, "days_since": True}
        if ascending is None:
            ascending = default_ascending.get(metric_key, False)

        rankings["rank"] = (
            rankings["value"].rank(ascending=ascending, method="min").astype(int)
        )
        rankings["metric"] = metric_key
        if metric_key == "trailing_return":
            rankings["period"] = period
        if start_date is not None and end_date is not None:
            rankings["start_date"] = start_date
            rankings["end_date"] = end_date

        rankings = rankings.sort_values(["rank", "value"], ascending=[True, ascending])
        if top_n is not None:
            rankings = rankings.head(top_n)
        rankings.index.name = "fund"
        return rankings

    def get_fund_rankings_dict(
        self,
        metric: str,
        period: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        trading_days: int = 252,
        top_n: int | None = None,
        funds: Iterable[FundInput] | None = None,
        ascending: bool | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets fund rankings as a JSON-friendly dictionary.

        Args:
            metric (str): metric name to rank by.
            period (int | None): trailing-return period for "trailing_return".
            start_date (date | None): optional start date for performance-based rankings.
            end_date (date | None): optional end date for performance-based rankings.
            as_of (date | None): optional historical anchor date for current price rankings.
            reference_date (date | None): optional reference date for days_since rankings.
            trading_days (int): trading days per year used for annualization.
            top_n (int | None): optional limit on ranked funds returned.
            funds (Iterable[FundInput] | None): optional collection of funds to include.
            ascending (bool | None): override default sort direction.
            date_format (str | None): format for any date outputs.

        Returns:
            dict: dictionary with ranking metadata and ordered fund rankings.
        """
        rankings = self.get_fund_rankings(
            metric=metric,
            period=period,
            start_date=start_date,
            end_date=end_date,
            as_of=as_of,
            reference_date=reference_date,
            trading_days=trading_days,
            top_n=top_n,
            funds=funds,
            ascending=ascending,
        )
        if rankings.empty:
            raise ValueError("no data available to rank funds for the selected metric")

        records: list[dict] = []
        for fund_name, row in rankings.iterrows():
            record = {"fund": fund_name}
            for column in rankings.columns:
                record[column] = self._format_value_for_output(row[column], date_format)
            records.append(record)

        payload: dict[str, object] = {
            "metric": (
                rankings["metric"].iloc[0] if "metric" in rankings.columns else metric
            ),
            "rankings": records,
        }
        if "period" in rankings.columns:
            payload["period"] = self._format_value_for_output(
                rankings["period"].iloc[0], date_format
            )
        if "start_date" in rankings.columns:
            payload["start_date"] = self._format_value_for_output(
                rankings["start_date"].iloc[0], date_format
            )
        if "end_date" in rankings.columns:
            payload["end_date"] = self._format_value_for_output(
                rankings["end_date"].iloc[0], date_format
            )
        return payload

    def get_cagr(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the compound annual growth rate (CAGR) for the specified fund or all funds.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date for CAGR calculation.
            end_date (date | None): optional end date for CAGR calculation.

        Returns:
            DataFrame: dataframe with start/end dates, prices, and CAGR values.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=price_df.reset_index()
            )
            price_df = filtered.set_index("Date")
        if price_df.empty:
            raise ValueError("no price data available for CAGR")
        results = {}
        for column in price_df.columns:
            series = price_df[column].dropna()
            if len(series) < 2:
                results[column] = {
                    "start_date": None,
                    "end_date": None,
                    "start_price": float("nan"),
                    "end_price": float("nan"),
                    "years": float("nan"),
                    "cagr": float("nan"),
                }
                continue
            start_timestamp = series.index[0]
            end_timestamp = series.index[-1]
            years = (end_timestamp - start_timestamp).days / 365.25
            if years <= 0:
                cagr = float("nan")
            else:
                cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
            results[column] = {
                "start_date": start_timestamp.date(),
                "end_date": end_timestamp.date(),
                "start_price": float(series.iloc[0]),
                "end_price": float(series.iloc[-1]),
                "years": float(years),
                "cagr": float(cagr),
            }
        summary = DataFrame.from_dict(results, orient="index")
        summary.index.name = "fund"
        return summary

    def get_fund_snapshot(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        periods: Iterable[int] = (5, 20, 63, 252),
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
    ) -> DataFrame:
        """
        Gets a snapshot of recent prices, changes, trailing returns, and performance metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            periods (Iterable[int]): trailing return periods to include.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.

        Returns:
            DataFrame: dataframe of snapshot metrics indexed by fund.
            Includes an `as_of` column representing the latest available date per fund.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_numeric(mar, "mar")
        self._validate_confidence(confidence, "confidence")
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
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
            if available:
                price_df = price_df[available]
        if len(price_df.index) < 2:
            raise ValueError(
                "at least two data points are required to build a snapshot"
            )

        periods_list = self._resolve_periods(periods)
        records: list[dict] = []
        for column in price_df.columns:
            series = price_df[column].dropna()
            if len(series) < 2:
                raise ValueError(
                    f"at least two data points are required to build a snapshot for {column}"
                )
            latest = series.iloc[-1]
            previous = series.iloc[-2]
            change = latest - previous
            change_percent = change / previous if previous != 0 else float("nan")

            summary = self._calculate_performance_summary(series, trading_days)
            returns = series.pct_change(fill_method=None).dropna(how="all")
            if returns.empty:
                summary["sortino_ratio"] = float("nan")
                summary["value_at_risk"] = float("nan")
                summary["expected_shortfall"] = float("nan")
            else:
                annualized_return = returns.mean() * trading_days
                daily_mar = self._to_daily_rate(mar, trading_days)
                downside = (returns - daily_mar).where(returns < daily_mar, 0)
                downside_deviation = downside.pow(2).mean() ** 0.5 * (trading_days**0.5)
                sortino_ratio = (
                    (annualized_return - mar) / downside_deviation
                    if downside_deviation
                    else float("nan")
                )
                var_level = 1 - float(confidence)
                value_at_risk = returns.quantile(var_level)
                expected_shortfall = returns[returns <= value_at_risk].mean()
                summary["sortino_ratio"] = float(sortino_ratio)
                summary["value_at_risk"] = float(value_at_risk)
                summary["expected_shortfall"] = float(expected_shortfall)
            summary["as_of"] = series.index[-1].date()
            summary["latest_price"] = latest
            summary["previous_price"] = previous
            summary["change"] = change
            summary["change_percent"] = change_percent
            for period in periods_list:
                summary[f"trailing_return_{period}d"] = series.pct_change(
                    periods=period, fill_method=None
                ).iloc[-1]
            records.append(summary)

        snapshot = DataFrame(records, index=price_df.columns)
        snapshot.index.name = "fund"
        return snapshot

    def get_current_price_dashboard(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        periods: Iterable[int] = (5, 20, 63, 252),
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
        reference_date: date | None = None,
    ) -> DataFrame:
        """
        Gets a dashboard-ready snapshot of current prices, changes, trailing returns, and risk.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            periods (Iterable[int]): trailing return periods to include.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset.

        Returns:
            DataFrame: dataframe of dashboard metrics indexed by fund.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_numeric(mar, "mar")
        self._validate_confidence(confidence, "confidence")
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")

        overview = self.get_current_fund_overview(
            fund=fund, funds=funds, reference_date=reference_date
        )
        if overview.empty:
            raise ValueError("no price data available for current price dashboard")

        periods_list = self._resolve_periods(periods)
        trailing = self.get_trailing_returns(
            periods=periods_list, fund=fund, funds=funds
        )
        trailing = trailing.T
        trailing.columns = [
            f"trailing_return_{int(period)}d" for period in trailing.columns
        ]

        risk_summary = self.get_risk_return_summary(
            fund=fund, trading_days=trading_days, mar=mar, confidence=confidence
        )

        dashboard = overview.join(trailing, how="left").join(risk_summary, how="left")
        return dashboard

    def get_current_price_dashboard_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        periods: Iterable[int] = (5, 20, 63, 252),
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
        reference_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets a JSON-friendly dashboard snapshot of current prices, returns, and risk metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            periods (Iterable[int]): trailing return periods to include.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset.
            date_format (str | None): format for date values. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary payload with per-fund dashboard metrics.
        """
        periods_list = self._resolve_periods(periods)
        dashboard = self.get_current_price_dashboard(
            fund=fund,
            funds=funds,
            periods=periods_list,
            trading_days=trading_days,
            mar=mar,
            confidence=confidence,
            reference_date=reference_date,
        )
        if dashboard.empty:
            raise ValueError("no price data available for current price dashboard")
        return {
            "periods": periods_list,
            "trading_days": trading_days,
            "funds": self._format_dataframe_for_output(
                dashboard, date_format=date_format
            ),
        }

    def get_fund_snapshot_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        periods: Iterable[int] = (5, 20, 63, 252),
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
    ) -> DataFrame:
        """
        Gets a long (tidy) fund snapshot for visualization or export.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            periods (Iterable[int]): trailing return periods to include.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.

        Returns:
            DataFrame: dataframe with fund, as_of, metric, and value columns.
        """
        snapshot = self.get_fund_snapshot(
            fund=fund,
            funds=funds,
            periods=periods,
            trading_days=trading_days,
            mar=mar,
            confidence=confidence,
        ).reset_index()
        long_snapshot = snapshot.melt(
            id_vars=["fund", "as_of"], var_name="metric", value_name="value"
        )
        return long_snapshot.dropna(subset=["value"])

    def get_fund_snapshot_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        periods: Iterable[int] = (5, 20, 63, 252),
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets a JSON-friendly snapshot of recent prices, trailing returns, and performance metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            periods (Iterable[int]): trailing return periods to include.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                pass None to return a date object.

        Returns:
            dict: dictionary with an overall as_of date and per-fund metric payloads.
        """
        snapshot = self.get_fund_snapshot(
            fund=fund,
            funds=funds,
            periods=periods,
            trading_days=trading_days,
            mar=mar,
            confidence=confidence,
        )
        formatted_metrics: dict[str, dict] = {}
        for fund_name, row in snapshot.iterrows():
            formatted_metrics[fund_name] = {
                metric: self._format_numeric_for_output(value)
                for metric, value in row.items()
                if metric != "as_of"
            }
            formatted_metrics[fund_name]["as_of"] = self._format_date_for_output(
                row["as_of"], date_format
            )
        as_of_date = self._format_date_for_output(snapshot["as_of"].max(), date_format)
        return {"as_of": as_of_date, "funds": formatted_metrics}
