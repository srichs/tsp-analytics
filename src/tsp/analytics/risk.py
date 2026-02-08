"""Risk metrics for TSP funds, including drawdowns and volatility."""

from collections.abc import Iterable
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from tsp.funds import FundInput


class AnalyticsRiskMixin:
    """Calculate risk-focused metrics for fund price series."""

    def get_price_statistics(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets descriptive statistics for fund prices.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date to filter prices.
            end_date (date | None): optional end date to filter prices.
                If only one bound is provided, the other defaults to the earliest/latest date
                in the dataset.

        Returns:
            DataFrame: dataframe of descriptive statistics for the requested prices.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        if start_date is not None or end_date is not None:
            min_date = price_df.index.min().date()
            max_date = price_df.index.max().date()
            start = start_date or min_date
            end = end_date or max_date
            filtered = self._filter_by_date_range(
                start, end, dataframe=price_df.reset_index()
            )
            price_df = filtered.set_index("Date")
        if price_df.empty:
            raise ValueError("no price data available for statistics")
        summary = price_df.describe().T
        summary["median"] = price_df.median()
        return summary

    def get_price_statistics_dict(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict[str, dict]:
        """
        Gets descriptive statistics for fund prices as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date to filter prices.
            end_date (date | None): optional end date to filter prices.
            date_format (str | None): optional date formatting for output values.

        Returns:
            dict: dictionary payload keyed by fund name.
        """
        summary = self.get_price_statistics(
            fund=fund, start_date=start_date, end_date=end_date
        )
        payload = self._format_dataframe_for_output(summary, date_format=date_format)
        return {"statistics": payload}

    def get_return_statistics(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
    ) -> DataFrame:
        """
        Gets descriptive statistics for daily returns.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date to filter returns.
            end_date (date | None): optional end date to filter returns.
                If only one bound is provided, the other defaults to the earliest/latest date
                in the dataset.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe of descriptive statistics for daily returns.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        returns = self.get_daily_returns(fund=fund)
        if start_date is not None or end_date is not None:
            min_date = returns.index.min().date()
            max_date = returns.index.max().date()
            start = start_date or min_date
            end = end_date or max_date
            filtered = self._filter_by_date_range(
                start, end, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        if returns.empty:
            raise ValueError("no return data available for statistics")
        summary = returns.describe().T
        summary["median"] = returns.median()
        summary["skew"] = returns.skew()
        summary["kurtosis"] = returns.kurtosis()
        summary["annualized_return"] = returns.mean() * trading_days
        summary["annualized_volatility"] = returns.std() * (trading_days**0.5)
        return summary

    def get_return_statistics_dict(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
        date_format: str | None = "iso",
    ) -> dict[str, dict]:
        """
        Gets descriptive statistics for daily returns as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date to filter returns.
            end_date (date | None): optional end date to filter returns.
            trading_days (int): trading days per year used for annualization.
            date_format (str | None): optional date formatting for output values.

        Returns:
            dict: dictionary payload keyed by fund name.
        """
        summary = self.get_return_statistics(
            fund=fund,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
        )
        payload = self._format_dataframe_for_output(summary, date_format=date_format)
        return {"statistics": payload}

    def _resolve_percentiles(self, percentiles: Iterable[float]) -> list[float]:
        if isinstance(percentiles, (str, bytes)) or not isinstance(
            percentiles, Iterable
        ):
            raise ValueError(
                "percentiles must be an iterable of floats between 0 and 1"
            )
        percentiles_list: list[float] = []
        for percentile in percentiles:
            self._validate_numeric(percentile, "percentiles")
            value = float(percentile)
            if not 0 <= value <= 1:
                raise ValueError("percentiles must be between 0 and 1")
            if value not in percentiles_list:
                percentiles_list.append(value)
        if not percentiles_list:
            raise ValueError("percentiles must contain at least one value")
        return sorted(percentiles_list)

    def get_return_distribution_summary(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        percentiles: Iterable[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    ) -> DataFrame:
        """
        Gets summary statistics for the distribution of daily returns.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            percentiles (Iterable[float]): percentiles to include between 0 and 1.

        Returns:
            DataFrame: dataframe of distribution statistics for daily returns.
        """
        self._ensure_dataframe()
        percentile_values = self._resolve_percentiles(percentiles)
        returns = self.get_daily_returns(fund=fund)
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        if returns.empty:
            raise ValueError("no return data available for distribution summary")
        summary = returns.describe(percentiles=percentile_values).T
        summary["median"] = returns.median()
        summary["skew"] = returns.skew()
        summary["kurtosis"] = returns.kurtosis()
        counts = returns.count()
        summary["win_rate"] = returns.gt(0).sum().div(counts)
        summary["loss_rate"] = returns.lt(0).sum().div(counts)
        summary["zero_rate"] = returns.eq(0).sum().div(counts)
        return summary

    def get_return_distribution_summary_dict(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        percentiles: Iterable[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    ) -> dict:
        """
        Gets a JSON-friendly daily return distribution summary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            percentiles (Iterable[float]): percentiles to include between 0 and 1.

        Returns:
            dict: dictionary of return distribution statistics per fund.
        """
        summary = self.get_return_distribution_summary(
            fund=fund, start_date=start_date, end_date=end_date, percentiles=percentiles
        )
        if summary.empty:
            raise ValueError("no return data available for distribution summary")
        payload: dict[str, dict] = {}
        for fund_name, row in summary.iterrows():
            payload[fund_name] = {
                column: self._format_numeric_for_output(row[column])
                for column in summary.columns
            }
        return {"funds": payload}

    def get_sortino_ratio(
        self, fund: FundInput | None = None, mar: float = 0.0, trading_days: int = 252
    ) -> DataFrame:
        """
        Gets the annualized Sortino ratio for the specified fund or all funds.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            mar (float): minimum acceptable return (annualized) for the Sortino ratio.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe with annualized return, downside deviation, and Sortino ratio.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_numeric(mar, "mar")
        returns = self.get_daily_returns(fund=fund)
        if returns.empty:
            raise ValueError("no return data available for Sortino ratio")
        daily_mar = self._to_daily_rate(mar, trading_days)
        downside = (returns - daily_mar).where(returns < daily_mar, 0)
        downside_deviation = downside.pow(2).mean().pow(0.5).mul(trading_days**0.5)
        annualized_return = returns.mean().mul(trading_days)
        sortino_ratio = (annualized_return - mar).div(downside_deviation)
        return DataFrame(
            {
                "annualized_return": annualized_return,
                "downside_deviation": downside_deviation,
                "sortino_ratio": sortino_ratio,
            }
        ).rename_axis("fund")

    def get_value_at_risk(
        self, confidence: float = 0.95, fund: FundInput | None = None
    ) -> DataFrame:
        """
        Gets the historical Value at Risk (VaR) for daily returns.

        Args:
            confidence (float): confidence level between 0 and 1.
            fund (FundInput | None): optional fund to limit the output.

        Returns:
            DataFrame: dataframe of VaR values indexed by fund.
        """
        self._validate_confidence(confidence, "confidence")
        returns = self.get_daily_returns(fund=fund)
        if returns.empty:
            raise ValueError("no return data available for value at risk")
        var_level = 1 - float(confidence)
        var_values = returns.quantile(var_level)
        return DataFrame({"value_at_risk": var_values}).rename_axis("fund")

    def get_expected_shortfall(
        self, confidence: float = 0.95, fund: FundInput | None = None
    ) -> DataFrame:
        """
        Gets the historical expected shortfall (CVaR) for daily returns.

        Args:
            confidence (float): confidence level between 0 and 1.
            fund (FundInput | None): optional fund to limit the output.

        Returns:
            DataFrame: dataframe of expected shortfall values indexed by fund.
        """
        self._validate_confidence(confidence, "confidence")
        returns = self.get_daily_returns(fund=fund)
        if returns.empty:
            raise ValueError("no return data available for expected shortfall")
        var_level = 1 - float(confidence)
        var_values = returns.quantile(var_level)
        expected_shortfall = {}
        for column in returns.columns:
            threshold = var_values[column]
            expected_shortfall[column] = returns[column][
                returns[column] <= threshold
            ].mean()
        return DataFrame({"expected_shortfall": expected_shortfall}).rename_axis("fund")

    def get_risk_return_summary(
        self,
        fund: FundInput | None = None,
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
    ) -> DataFrame:
        """
        Gets a risk/return summary with common risk metrics for each fund.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.

        Returns:
            DataFrame: dataframe with annualized return/volatility, Sharpe/Sortino,
            max drawdown, Calmar ratio, ulcer index, max drawdown duration,
            max drawdown recovery time, pain index/ratio, Omega ratio, skew/kurtosis,
            Value at Risk, and expected shortfall.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_numeric(mar, "mar")
        self._validate_confidence(confidence, "confidence")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        returns = price_df.pct_change(fill_method=None).dropna(how="all")
        if returns.empty:
            raise ValueError("no return data available for risk summary")
        annualized_return = returns.mean().mul(trading_days)
        annualized_volatility = returns.std().mul(trading_days**0.5)
        sharpe_ratio = annualized_return.div(annualized_volatility)
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        daily_mar = self._to_daily_rate(mar, trading_days)
        downside = (returns - daily_mar).where(returns < daily_mar, 0)
        downside_deviation = downside.pow(2).mean().pow(0.5).mul(trading_days**0.5)
        sortino_ratio = (annualized_return - mar).div(downside_deviation)
        omega_ratio = {}
        drawdown_duration_days = {}
        drawdown_recovery_days = {}
        pain_index = {}
        for column in returns.columns:
            threshold = daily_mar
            gains = (
                (returns[column] - threshold)
                .where(returns[column] > threshold, 0)
                .sum()
            )
            losses = (
                (threshold - returns[column])
                .where(returns[column] < threshold, 0)
                .sum()
            )
            omega_ratio[column] = gains / losses if losses != 0 else float("nan")
        var_level = 1 - float(confidence)
        value_at_risk = returns.quantile(var_level)
        expected_shortfall = {}
        for column in returns.columns:
            threshold = value_at_risk[column]
            expected_shortfall[column] = returns[column][
                returns[column] <= threshold
            ].mean()
        max_drawdown = {
            column: self._calculate_max_drawdown(price_df[column].dropna())
            for column in returns.columns
        }
        for column in returns.columns:
            prices = price_df[column].dropna()
            if prices.empty:
                drawdown_duration_days[column] = float("nan")
                continue
            cumulative_max = prices.cummax()
            drawdown = prices.div(cumulative_max).sub(1)
            trough_date = drawdown.idxmin()
            peak_date = prices.loc[:trough_date].idxmax()
            drawdown_duration_days[column] = int((trough_date - peak_date).days)
            peak_price = prices.loc[peak_date]
            recovery_prices = prices.loc[trough_date:]
            recovered = recovery_prices[recovery_prices >= peak_price]
            if recovered.empty:
                drawdown_recovery_days[column] = float("nan")
            else:
                recovery_date = recovered.index[0]
                drawdown_recovery_days[column] = int((recovery_date - trough_date).days)
            pain_index[column] = drawdown.abs().mean()
        max_drawdown_series = Series(max_drawdown)
        calmar_ratio = annualized_return.div(max_drawdown_series.abs()).where(
            max_drawdown_series != 0
        )
        ulcer_index = {
            column: (
                (price_df[column].div(price_df[column].cummax()).sub(1).pow(2)).mean()
                ** 0.5
            )
            for column in returns.columns
        }
        pain_index_series = Series(pain_index)
        pain_ratio = annualized_return.div(pain_index_series).where(
            pain_index_series != 0
        )
        summary = DataFrame(
            {
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "skew": skew,
                "kurtosis": kurtosis,
                "max_drawdown": max_drawdown_series,
                "calmar_ratio": calmar_ratio,
                "ulcer_index": Series(ulcer_index),
                "max_drawdown_duration_days": Series(drawdown_duration_days),
                "max_drawdown_recovery_days": Series(drawdown_recovery_days),
                "pain_index": pain_index_series,
                "pain_ratio": pain_ratio,
                "omega_ratio": Series(omega_ratio),
                "value_at_risk": value_at_risk,
                "expected_shortfall": Series(expected_shortfall),
            }
        )
        summary.index.name = "fund"
        return summary

    def get_risk_return_summary_dict(
        self,
        fund: FundInput | None = None,
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
    ) -> dict:
        """
        Gets a risk/return summary as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.

        Returns:
            dict: dictionary with risk/return metrics keyed by fund.
        """
        summary = self.get_risk_return_summary(
            fund=fund, trading_days=trading_days, mar=mar, confidence=confidence
        )
        if summary.empty:
            raise ValueError("no risk/return data available")
        return {
            "trading_days": trading_days,
            "mar": self._format_numeric_for_output(mar),
            "confidence": self._format_numeric_for_output(confidence),
            "funds": self._format_dataframe_for_output(summary),
        }

    def get_beta(self, fund: FundInput, benchmark: FundInput) -> float:
        """
        Gets beta for a fund relative to a benchmark fund based on daily returns.

        Args:
            fund (FundInput): the fund to evaluate.
            benchmark (FundInput): the benchmark fund.

        Returns:
            float: beta value.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        returns = self.get_daily_returns()
        if fund_name not in returns.columns or benchmark_name not in returns.columns:
            raise ValueError("fund or benchmark not available in return data")
        benchmark_returns = returns[benchmark_name]
        variance = benchmark_returns.var()
        if variance == 0 or variance != variance:
            return float("nan")
        covariance = returns[fund_name].cov(benchmark_returns)
        return float(covariance / variance)

    def get_rolling_beta(
        self, fund: FundInput, benchmark: FundInput, window: int = 63
    ) -> DataFrame:
        """
        Gets rolling beta for a fund relative to a benchmark fund.

        Args:
            fund (FundInput): the fund to evaluate.
            benchmark (FundInput): the benchmark fund.
            window (int): rolling window size.

        Returns:
            DataFrame: dataframe of rolling beta values.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        self._validate_positive_int(window, "window")
        returns = self.get_daily_returns()
        if fund_name not in returns.columns or benchmark_name not in returns.columns:
            raise ValueError("fund or benchmark not available in return data")
        benchmark_returns = returns[benchmark_name]
        rolling_cov = returns[fund_name].rolling(window=window).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(window=window).var()
        rolling_beta = rolling_cov.div(rolling_var)
        return rolling_beta.dropna(how="all").to_frame(name="rolling_beta")

    def get_rolling_sharpe_ratio(
        self, fund: FundInput, window: int = 63, trading_days: int = 252
    ) -> DataFrame:
        """
        Gets rolling annualized Sharpe ratios for the specified fund.

        Args:
            fund (FundInput): the fund to retrieve rolling Sharpe ratios for.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe of rolling Sharpe ratio values.
        """
        self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        self._validate_positive_int(trading_days, "trading_days")
        returns = self.get_daily_returns(fund=fund)
        rolling_return = returns.rolling(window=window).mean().mul(trading_days)
        rolling_volatility = returns.rolling(window=window).std().mul(trading_days**0.5)
        sharpe_ratio = rolling_return.div(rolling_volatility)
        return sharpe_ratio.dropna(how="all")

    def get_rolling_sortino_ratio(
        self,
        fund: FundInput,
        window: int = 63,
        trading_days: int = 252,
        mar: float = 0.0,
    ) -> DataFrame:
        """
        Gets rolling annualized Sortino ratios for the specified fund.

        Args:
            fund (FundInput): the fund to retrieve rolling Sortino ratios for.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for the Sortino ratio.

        Returns:
            DataFrame: dataframe of rolling Sortino ratio values.
        """
        self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_numeric(mar, "mar")
        returns = self.get_daily_returns(fund=fund)
        daily_mar = self._to_daily_rate(mar, trading_days)
        downside = (returns - daily_mar).where(returns < daily_mar, 0)
        rolling_downside = (
            downside.pow(2)
            .rolling(window=window)
            .mean()
            .pow(0.5)
            .mul(trading_days**0.5)
        )
        rolling_return = returns.rolling(window=window).mean().mul(trading_days)
        sortino_ratio = (rolling_return - mar).div(rolling_downside)
        return sortino_ratio.dropna(how="all")

    def get_rolling_performance_summary(
        self, fund: FundInput, window: int = 63, trading_days: int = 252
    ) -> DataFrame:
        """
        Gets rolling annualized return, volatility, and Sharpe ratio for a fund.

        Args:
            fund (FundInput): the fund to analyze.
            window (int): rolling window size in trading days.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe with rolling return, volatility, and Sharpe ratio columns.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        self._validate_positive_int(trading_days, "trading_days")
        returns = self.get_daily_returns(fund=fund_name)
        if returns.empty or len(returns.index) < window:
            raise ValueError(
                "not enough return data available for rolling performance summary"
            )
        series = returns[fund_name]
        rolling_return = series.rolling(window=window).mean().mul(trading_days)
        rolling_volatility = series.rolling(window=window).std().mul(trading_days**0.5)
        rolling_sharpe = rolling_return.div(rolling_volatility)
        summary = DataFrame(
            {
                "rolling_return": rolling_return,
                "rolling_volatility": rolling_volatility,
                "rolling_sharpe_ratio": rolling_sharpe,
            }
        )
        summary.index.name = "Date"
        return summary.dropna(how="all")

    def get_rolling_performance_summary_long(
        self, fund: FundInput, window: int = 63, trading_days: int = 252
    ) -> DataFrame:
        """
        Gets rolling performance metrics in long (tidy) format.

        Args:
            fund (FundInput): the fund to analyze.
            window (int): rolling window size in trading days.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe with Date, fund, metric, and value columns.
        """
        fund_name = self._ensure_fund_available(fund)
        summary = self.get_rolling_performance_summary(
            fund=fund_name, window=window, trading_days=trading_days
        )
        long_summary = summary.reset_index().melt(
            id_vars="Date", var_name="metric", value_name="value"
        )
        long_summary["fund"] = fund_name
        return long_summary[["Date", "fund", "metric", "value"]].dropna(
            subset=["value"]
        )

    def get_rolling_performance_summary_dict(
        self,
        fund: FundInput,
        window: int = 63,
        trading_days: int = 252,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets rolling performance metrics as a JSON-friendly dictionary.

        Args:
            fund (FundInput): the fund to analyze.
            window (int): rolling window size in trading days.
            trading_days (int): trading days per year used for annualization.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with fund metadata and rolling performance metrics.
        """
        fund_name = self._ensure_fund_available(fund)
        long_summary = self.get_rolling_performance_summary_long(
            fund=fund_name, window=window, trading_days=trading_days
        )
        if long_summary.empty:
            raise ValueError("no rolling performance data available")
        metrics = self._format_long_dataframe_for_output(
            long_summary.drop(columns=["fund"]), date_format
        )
        return {
            "fund": fund_name,
            "window": window,
            "trading_days": trading_days,
            "metrics": metrics,
        }

    def get_drawdown_series(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the drawdown series for the specified fund or funds.

        Args:
            fund (FundInput | None): optional fund to calculate drawdowns for.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe of drawdown values indexed by date.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if funds is not None:
            fund_list = self._resolve_funds(funds)
            self._ensure_funds_available(fund_list)
            price_df = price_df[fund_list]
        elif fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        if price_df.empty:
            raise ValueError("no price data available for drawdown series")
        cumulative_max = price_df.cummax()
        drawdown = (price_df / cumulative_max) - 1
        return drawdown.dropna(how="all")

    def get_drawdown_series_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets drawdown series in long (tidy) format for visualization or export.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with Date, fund, and drawdown columns.
        """
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        drawdown = self.get_drawdown_series(fund=fund, funds=funds).reset_index()
        if start_date is not None and end_date is not None:
            drawdown = self._filter_by_date_range(
                start_date, end_date, dataframe=drawdown
            )
        return drawdown.melt(
            id_vars="Date", var_name="fund", value_name="drawdown"
        ).dropna(subset=["drawdown"])

    def get_rolling_max_drawdown(self, fund: FundInput, window: int = 252) -> DataFrame:
        """
        Gets the rolling maximum drawdown for the specified fund.

        Args:
            fund (FundInput): the fund to analyze.
            window (int): rolling window size.

        Returns:
            DataFrame: dataframe of rolling maximum drawdown values.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        prices = (
            self.dataframe[["Date", fund_name]].set_index("Date")[fund_name].dropna()
        )
        rolling_drawdown = prices.rolling(window=window).apply(
            self._calculate_max_drawdown, raw=False
        )
        return rolling_drawdown.dropna(how="all").to_frame(name="rolling_max_drawdown")

    def get_rolling_volatility(
        self, fund: FundInput, window: int = 20, trading_days: int = 252
    ) -> DataFrame:
        """
        Gets rolling annualized volatility for the specified fund.

        Args:
            fund (FundInput): the fund to retrieve the rolling volatility for.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe of rolling volatility values.
        """
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        self._validate_positive_int(trading_days, "trading_days")
        returns = self.get_daily_returns(fund=fund_name)
        return (
            returns.rolling(window=window)
            .std()
            .mul(trading_days**0.5)
            .dropna(how="all")
        )

    def get_max_drawdown(self, fund: FundInput) -> Decimal:
        """
        Gets the maximum drawdown for the specified fund.

        Args:
            fund (FundInput): the fund to calculate drawdown for.

        Returns:
            Decimal: maximum drawdown value.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        prices = self.dataframe[["Date", fund_name]].set_index("Date")
        cumulative_max = prices.cummax()
        drawdown = (prices / cumulative_max) - 1
        return Decimal(str(drawdown.min().iloc[0]))

    def get_drawdown_summary(self, fund: FundInput) -> DataFrame:
        """
        Gets drawdown summary statistics for the specified fund.

        Args:
            fund (FundInput): the fund to analyze.

        Returns:
            DataFrame: dataframe with drawdown metrics indexed by fund.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        prices = (
            self.dataframe[["Date", fund_name]].set_index("Date")[fund_name].dropna()
        )
        if prices.empty:
            raise ValueError("no price data available for drawdown summary")
        cumulative_max = prices.cummax()
        drawdown = (prices / cumulative_max) - 1
        trough_date = drawdown.idxmin()
        peak_date = prices.loc[:trough_date].idxmax()
        peak_price = prices.loc[peak_date]
        recovery_date = prices.loc[trough_date:][prices >= peak_price].index.min()
        if recovery_date != recovery_date:
            recovery_date = None
        drawdown_duration_days = int((trough_date - peak_date).days)
        recovery_duration_days = None
        if recovery_date is not None:
            recovery_duration_days = int((recovery_date - trough_date).days)
        summary = DataFrame(
            {
                "max_drawdown": [float(drawdown.loc[trough_date])],
                "peak_date": [peak_date.date()],
                "trough_date": [trough_date.date()],
                "recovery_date": [
                    recovery_date.date() if recovery_date is not None else None
                ],
                "drawdown_duration_days": [drawdown_duration_days],
                "recovery_duration_days": [recovery_duration_days],
            },
            index=[fund_name],
        )
        summary.index.name = "fund"
        return summary

    def get_drawdown_summary_dict(
        self, fund: FundInput, date_format: str | None = "iso"
    ) -> dict:
        """
        Gets drawdown summary statistics as a JSON-friendly dictionary.

        Args:
            fund (FundInput): the fund to analyze.
            date_format (str | None): format for date values. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary payload keyed by fund.
        """
        summary = self.get_drawdown_summary(fund=fund)
        if summary.empty:
            raise ValueError("no drawdown data available")
        return {
            "funds": self._format_dataframe_for_output(summary, date_format=date_format)
        }

    @staticmethod
    def _build_regime_labels(regime_type: str, num_regimes: int) -> list[str]:
        if num_regimes <= 0:
            raise ValueError("num_regimes must be a positive integer")
        if regime_type == "volatility":
            if num_regimes == 2:
                return ["low_vol", "high_vol"]
            if num_regimes == 3:
                return ["low_vol", "mid_vol", "high_vol"]
        if regime_type == "trend":
            if num_regimes == 2:
                return ["down", "up"]
            if num_regimes == 3:
                return ["down", "flat", "up"]
        return [f"regime_{idx + 1}" for idx in range(num_regimes)]

    @staticmethod
    def _kmeans_cluster(
        values: np.ndarray, num_regimes: int, max_iter: int = 100
    ) -> np.ndarray:
        if values.size == 0:
            return values
        rng = np.random.default_rng(42)
        initial = np.quantile(values, np.linspace(0, 1, num_regimes + 2)[1:-1], axis=0)
        centers = initial.copy()
        if centers.ndim == 1:
            centers = centers.reshape(num_regimes, 1)
        values_2d = values if values.ndim == 2 else values.reshape(-1, 1)
        for _ in range(max_iter):
            distances = np.linalg.norm(
                values_2d[:, None, :] - centers[None, :, :], axis=2
            )
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for idx in range(num_regimes):
                if np.any(labels == idx):
                    new_centers[idx] = values_2d[labels == idx].mean(axis=0)
                else:
                    new_centers[idx] = values_2d[rng.integers(0, len(values_2d))]
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return labels

    def _assign_regimes(
        self, values: Series, regime_type: str, num_regimes: int, method: str
    ) -> DataFrame:
        if method not in {"quantile", "cluster"}:
            raise ValueError('method must be "quantile" or "cluster"')
        cleaned = values.dropna()
        if cleaned.empty:
            return DataFrame(index=values.index, columns=["regime", "regime_label"])
        if method == "quantile":
            regimes = Series(np.nan, index=values.index, dtype="float")
            raw = pd.qcut(cleaned, q=num_regimes, labels=False, duplicates="drop")
            regimes.loc[cleaned.index] = raw.astype(float).add(1)
        else:
            clustered = self._kmeans_cluster(cleaned.to_numpy(), num_regimes)
            order = np.argsort(
                [
                    (
                        cleaned.to_numpy()[clustered == idx].mean()
                        if np.any(clustered == idx)
                        else float("nan")
                    )
                    for idx in range(num_regimes)
                ]
            )
            reorder_map = {int(old): int(new) for new, old in enumerate(order, start=1)}
            ordered = [reorder_map[int(label)] for label in clustered]
            regimes = Series(np.nan, index=values.index, dtype="float")
            regimes.loc[cleaned.index] = ordered
        max_regime = int(regimes.max()) if regimes.notna().any() else num_regimes
        labels_map = self._build_regime_labels(regime_type, max_regime)
        label_series = regimes.map(
            {idx + 1: label for idx, label in enumerate(labels_map)}
        )
        return DataFrame({"regime": regimes, "regime_label": label_series})

    def get_regime_detection(
        self,
        fund: FundInput | None = None,
        window: int = 63,
        regime_type: str = "volatility",
        num_regimes: int = 3,
        method: str = "quantile",
        trading_days: int = 252,
    ) -> DataFrame:
        """
        Detects market regimes using rolling volatility or trend metrics.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            window (int): rolling window size in trading days.
            regime_type (str): "volatility" or "trend".
            num_regimes (int): number of regimes to classify.
            method (str): "quantile" or "cluster" for regime assignment.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe with regime assignments for each date and fund.
        """
        self._ensure_dataframe()
        self._validate_positive_int(window, "window")
        self._validate_positive_int(num_regimes, "num_regimes")
        self._validate_positive_int(trading_days, "trading_days")
        regime_key = regime_type.strip().lower()
        if regime_key not in {"volatility", "trend"}:
            raise ValueError('regime_type must be "volatility" or "trend"')

        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]

        returns = price_df.pct_change(fill_method=None)
        rolling_volatility = returns.rolling(window=window).std().mul(trading_days**0.5)
        rolling_trend = price_df.pct_change(periods=window)

        records: list[DataFrame] = []
        for column in price_df.columns:
            metric = (
                rolling_volatility[column]
                if regime_key == "volatility"
                else rolling_trend[column]
            )
            regimes = self._assign_regimes(metric, regime_key, num_regimes, method)
            combined = DataFrame(
                {
                    "Date": metric.index,
                    "fund": column,
                    "rolling_volatility": rolling_volatility[column],
                    "rolling_trend": rolling_trend[column],
                    "regime": regimes["regime"],
                    "regime_label": regimes["regime_label"],
                }
            )
            records.append(combined)

        output = pd.concat(records, ignore_index=True) if records else DataFrame()
        if fund is not None:
            return output.drop(columns=["fund"]).set_index("Date")
        return output

    def get_historical_stress_test_summary(
        self,
        windows: dict[str, tuple[date, date]] | None = None,
        fund: FundInput | None = None,
        trading_days: int = 252,
    ) -> DataFrame:
        """
        Runs historical stress tests for specified windows.

        Args:
            windows (dict | None): mapping of window name to (start_date, end_date).
            fund (FundInput | None): optional fund to limit output.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe with stress test performance metrics.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        default_windows = {
            "gfc_2008": (date(2007, 10, 1), date(2009, 3, 31)),
            "covid_2020": (date(2020, 2, 19), date(2020, 3, 23)),
        }
        window_map = windows or default_windows
        records: list[dict] = []
        for window_name, (start_date, end_date) in window_map.items():
            self._validate_date_range(start_date, end_date)
            summary = self.get_performance_summary_by_date_range(
                start_date=start_date,
                end_date=end_date,
                fund=fund,
                trading_days=trading_days,
            )
            if summary.empty:
                continue
            for fund_name, row in summary.iterrows():
                records.append(
                    {
                        "window": window_name,
                        "fund": fund_name,
                        "start_date": start_date,
                        "end_date": end_date,
                        **row.to_dict(),
                    }
                )
        return DataFrame.from_records(records)

    def get_worst_drawdown_windows(
        self, fund: FundInput | None = None, window: int = 63, top_n: int = 5
    ) -> DataFrame:
        """
        Finds the worst rolling return windows for drawdown-style stress analysis.

        Args:
            fund (FundInput | None): optional fund to limit output.
            window (int): rolling window size in trading days.
            top_n (int): number of worst windows to return.

        Returns:
            DataFrame: dataframe with the worst rolling windows.
        """
        self._ensure_dataframe()
        self._validate_positive_int(window, "window")
        self._validate_positive_int(top_n, "top_n")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]

        records: list[dict] = []
        for column in price_df.columns:
            rolling_returns = price_df[column].pct_change(periods=window).dropna()
            worst_windows = rolling_returns.nsmallest(top_n)
            for end_date, total_return in worst_windows.items():
                start_date = end_date - pd.Timedelta(days=window)
                records.append(
                    {
                        "fund": column,
                        "start_date": start_date.date(),
                        "end_date": end_date.date(),
                        "total_return": float(total_return),
                    }
                )
        result = DataFrame.from_records(records)
        if fund is not None and not result.empty:
            return result.drop(columns=["fund"])
        return result

    def get_shock_scenario_analysis(
        self, shocks: dict[str, dict[FundInput, float]], base_date: date | None = None
    ) -> DataFrame:
        """
        Applies defined shock scenarios to fund prices.

        Args:
            shocks (dict): mapping of scenario name to fund shock returns.
            base_date (date | None): optional date for base prices (defaults to latest).

        Returns:
            DataFrame: dataframe of shock scenario impacts by fund.
        """
        self._ensure_dataframe()
        if not shocks:
            raise ValueError("shocks must contain at least one scenario")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if base_date is None:
            base_date = price_df.index.max().date()
        if base_date not in price_df.index.date:
            raise ValueError("base_date not found in price data")
        base_prices = price_df.loc[price_df.index.date == base_date].iloc[-1]
        records: list[dict] = []
        for scenario, shock_map in shocks.items():
            resolved_shocks = {
                self._ensure_fund_available(fund): float(shock)
                for fund, shock in shock_map.items()
            }
            for fund_name in base_prices.index:
                shock = resolved_shocks.get(fund_name, 0.0)
                base_price = base_prices[fund_name]
                shocked_price = base_price * (1 + shock)
                records.append(
                    {
                        "scenario": scenario,
                        "fund": fund_name,
                        "base_date": base_date,
                        "base_price": float(base_price),
                        "shock_return": float(shock),
                        "shocked_price": float(shocked_price),
                    }
                )
        return DataFrame.from_records(records)
