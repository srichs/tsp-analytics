"""Return calculations for TSP funds over configurable date ranges."""

import calendar
from collections.abc import Iterable
from datetime import date

from pandas import DataFrame

from tsp.funds import FundInput, TspIndividualFund


class AnalyticsReturnsMixin:
    """Compute return series and summary return metrics for funds."""

    def get_daily_returns(self, fund: FundInput | None = None) -> DataFrame:
        """
        Gets the daily percentage returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to retrieve returns for.

        Returns:
            DataFrame: dataframe of daily percentage returns.
        """
        self._ensure_dataframe()
        if fund is None:
            price_df = self.dataframe.set_index("Date").dropna(how="all")
            return price_df.pct_change(fill_method=None).dropna(how="all")
        fund_name = self._ensure_fund_available(fund)
        return (
            self.dataframe[["Date", fund_name]]
            .set_index("Date")
            .pct_change(fill_method=None)
            .dropna(how="all")
        )

    def get_daily_returns_long(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets daily returns in long (tidy) format for visualization or export.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with Date, fund, and return columns.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        returns = self.get_daily_returns(fund=fund).reset_index()
        if start_date is not None and end_date is not None:
            returns = self._filter_by_date_range(
                start_date, end_date, dataframe=returns
            )
        return returns.melt(
            id_vars="Date", var_name="fund", value_name="return"
        ).dropna(subset=["return"])

    def get_excess_returns(
        self,
        fund: FundInput | None = None,
        benchmark: FundInput = TspIndividualFund.G_FUND,
    ) -> DataFrame:
        """
        Gets daily excess returns versus a benchmark fund.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            benchmark (FundInput): benchmark fund for excess return calculations.

        Returns:
            DataFrame: dataframe of excess returns.
        """
        self._ensure_dataframe()
        benchmark_name = self._ensure_fund_available(benchmark)
        returns = self.get_daily_returns()
        if returns.empty:
            raise ValueError("no return data available for excess returns")
        if benchmark_name not in returns.columns:
            raise ValueError(
                f"benchmark fund not available in return data: {benchmark_name}"
            )
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            excess = returns[fund_name].sub(returns[benchmark_name])
            return excess.to_frame(name="excess_return")
        excess = returns.drop(columns=[benchmark_name], errors="ignore")
        if excess.empty:
            raise ValueError("no funds available for excess returns")
        return excess.sub(returns[benchmark_name], axis=0)

    def get_excess_returns_long(
        self,
        fund: FundInput | None = None,
        benchmark: FundInput = TspIndividualFund.G_FUND,
    ) -> DataFrame:
        """
        Gets daily excess returns in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            benchmark (FundInput): benchmark fund for excess return calculations.

        Returns:
            DataFrame: dataframe with Date, fund, and excess_return columns.
        """
        excess = self.get_excess_returns(fund=fund, benchmark=benchmark)
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            excess = excess.reset_index()
            excess["fund"] = fund_name
            return excess[["Date", "fund", "excess_return"]].dropna(
                subset=["excess_return"]
            )
        excess = excess.reset_index()
        return excess.melt(
            id_vars="Date", var_name="fund", value_name="excess_return"
        ).dropna(subset=["excess_return"])

    def get_daily_returns_by_date_range(
        self, start_date: date, end_date: date, fund: FundInput | None = None
    ) -> DataFrame:
        """
        Gets the daily percentage returns for the specified date range.

        Args:
            start_date (date): the start date to retrieve returns for.
            end_date (date): the end date to retrieve returns for.
            fund (FundInput | None): the fund to retrieve returns for.

        Returns:
            DataFrame: dataframe of daily percentage returns for the date range.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        filtered = self._filter_by_date_range(
            start_date, end_date, dataframe=price_df.reset_index()
        )
        filtered = filtered.set_index("Date")
        return filtered.pct_change(fill_method=None).dropna(how="all")

    def get_cumulative_returns(self, fund: FundInput | None = None) -> DataFrame:
        """
        Gets cumulative returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to retrieve returns for.

        Returns:
            DataFrame: dataframe of cumulative returns.
        """
        daily_returns = self.get_daily_returns(fund=fund)
        return (1 + daily_returns).cumprod() - 1

    def get_cumulative_returns_long(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets cumulative returns in long (tidy) format for visualization or export.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with Date, fund, and cumulative_return columns.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        cumulative = self.get_cumulative_returns(fund=fund).reset_index()
        if start_date is not None and end_date is not None:
            cumulative = self._filter_by_date_range(
                start_date, end_date, dataframe=cumulative
            )
        return cumulative.melt(
            id_vars="Date", var_name="fund", value_name="cumulative_return"
        ).dropna(subset=["cumulative_return"])

    def get_cumulative_returns_by_date_range(
        self, start_date: date, end_date: date, fund: FundInput | None = None
    ) -> DataFrame:
        """
        Gets cumulative returns for the specified date range.

        Args:
            start_date (date): the start date to retrieve returns for.
            end_date (date): the end date to retrieve returns for.
            fund (FundInput | None): the fund to retrieve returns for.

        Returns:
            DataFrame: dataframe of cumulative returns for the date range.
        """
        daily_returns = self.get_daily_returns_by_date_range(
            start_date, end_date, fund=fund
        )
        return (1 + daily_returns).cumprod() - 1

    def get_normalized_prices(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        base_value: float = 100.0,
    ) -> DataFrame:
        """
        Gets normalized prices rebased to a starting value for easy comparison.

        Args:
            fund (FundInput | None): the fund to normalize. If None, includes all funds.
            start_date (date | None): optional start date to normalize from.
            end_date (date | None): optional end date to normalize to.
            base_value (float): base value to normalize prices to.

        Returns:
            DataFrame: dataframe of normalized prices.
        """
        self._ensure_dataframe()
        self._validate_positive_float(base_value, "base_value")
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        else:
            available = self.get_available_funds()
            if not available:
                raise ValueError("no available funds in data")
            price_df = price_df[available]
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=price_df.reset_index()
            )
            price_df = filtered.set_index("Date")
        if price_df.empty:
            raise ValueError("no price data available for normalization")
        first_valid = self._resolve_first_valid_prices(price_df)
        return price_df.div(first_valid).mul(base_value)

    def get_normalized_prices_long(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        base_value: float = 100.0,
    ) -> DataFrame:
        """
        Gets normalized prices in long (tidy) format for visualization or export.

        Args:
            fund (FundInput | None): the fund to normalize. If None, includes all funds.
            start_date (date | None): optional start date to normalize from.
            end_date (date | None): optional end date to normalize to.
            base_value (float): base value to normalize prices to.

        Returns:
            DataFrame: dataframe with Date, fund, and normalized_price columns.
        """
        normalized = self.get_normalized_prices(
            fund=fund, start_date=start_date, end_date=end_date, base_value=base_value
        ).reset_index()
        return normalized.melt(
            id_vars="Date", var_name="fund", value_name="normalized_price"
        ).dropna(subset=["normalized_price"])

    def get_yearly_returns(self, fund: FundInput | None = None) -> DataFrame:
        """
        Gets yearly returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to retrieve yearly returns for.

        Returns:
            DataFrame: dataframe of yearly returns.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        yearly_prices = price_df.resample("YE").last()
        return yearly_prices.pct_change(fill_method=None).dropna(how="all")

    def get_monthly_returns(self, fund: FundInput | None = None) -> DataFrame:
        """
        Gets monthly returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to retrieve monthly returns for.

        Returns:
            DataFrame: dataframe of monthly returns.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            price_df = price_df[[fund_name]]
        monthly_prices = price_df.resample("ME").last()
        return monthly_prices.pct_change(fill_method=None).dropna(how="all")

    def get_monthly_return_table(
        self,
        fund: FundInput,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets a month-by-month return table for a single fund.

        Args:
            fund (FundInput): fund to analyze.
            start_date (date | None): optional start date for the return table.
            end_date (date | None): optional end date for the return table.

        Returns:
            DataFrame: dataframe with years as rows and month abbreviations as columns.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        returns = self.get_monthly_returns(fund=fund_name)
        if returns.empty:
            raise ValueError("no return data available for monthly return table")
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        if returns.empty:
            raise ValueError("no return data available for monthly return table")
        working = returns.reset_index()
        working["year"] = working["Date"].dt.year
        working["month"] = working["Date"].dt.month
        table = working.pivot(index="year", columns="month", values=fund_name)
        table = table.reindex(columns=range(1, 13))
        table.columns = [calendar.month_abbr[month] for month in table.columns]
        table.index.name = "year"
        return table

    def get_monthly_return_table_long(
        self,
        fund: FundInput,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets a long (tidy) month-by-month return table for a single fund.

        Args:
            fund (FundInput): fund to analyze.
            start_date (date | None): optional start date for the return table.
            end_date (date | None): optional end date for the return table.

        Returns:
            DataFrame: dataframe with year, month, month_num, fund, and return columns.
        """
        fund_name = self._ensure_fund_available(fund)
        table = self.get_monthly_return_table(
            fund=fund_name, start_date=start_date, end_date=end_date
        )
        month_map = {abbr: idx for idx, abbr in enumerate(calendar.month_abbr) if abbr}
        long_table = table.reset_index().melt(
            id_vars="year", var_name="month", value_name="return"
        )
        long_table["month_num"] = long_table["month"].map(month_map)
        long_table["fund"] = fund_name
        return (
            long_table.dropna(subset=["return"])
            .sort_values(["year", "month_num"])
            .reset_index(drop=True)
        )

    def get_monthly_return_table_dict(
        self,
        fund: FundInput,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict:
        """
        Gets a month-by-month return table as a JSON-friendly dictionary.

        Args:
            fund (FundInput): fund to analyze.
            start_date (date | None): optional start date for the return table.
            end_date (date | None): optional end date for the return table.

        Returns:
            dict: dictionary with fund metadata and monthly return records.
        """
        fund_name = self._ensure_fund_available(fund)
        long_table = self.get_monthly_return_table_long(
            fund=fund_name, start_date=start_date, end_date=end_date
        )
        records = [
            {
                "year": int(row["year"]),
                "month": row["month"],
                "month_num": int(row["month_num"]),
                "return": self._format_numeric_for_output(row["return"]),
            }
            for _, row in long_table.iterrows()
        ]
        return {"fund": fund_name, "returns": records}

    def get_trailing_returns(
        self,
        periods: Iterable[int] | int,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> DataFrame:
        """
        Gets trailing returns for the provided periods based on the latest available price.

        Args:
            periods (Iterable[int] | int): period or iterable of periods to calculate.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None): optional collection of funds to include.

        Returns:
            DataFrame: dataframe of trailing returns indexed by period.
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
        else:
            available = self.get_available_funds()
            if not available:
                raise ValueError("no available funds in data")
            price_df = price_df[available]
        if price_df.empty:
            raise ValueError("no price data available for trailing returns")
        periods_list = self._resolve_periods(periods)
        trailing_returns = {}
        for period in periods_list:
            trailing_returns[period] = price_df.pct_change(
                periods=period, fill_method=None
            ).iloc[-1]
        trailing_df = DataFrame(trailing_returns).T
        trailing_df.index.name = "periods"
        return trailing_df

    def get_trailing_returns_long(
        self,
        periods: Iterable[int] | int,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> DataFrame:
        """
        Gets trailing returns in long (tidy) format.

        Args:
            periods (Iterable[int] | int): period or iterable of periods to calculate.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None): optional collection of funds to include.

        Returns:
            DataFrame: dataframe with period, fund, and trailing_return columns.
        """
        trailing = self.get_trailing_returns(periods=periods, fund=fund, funds=funds)
        long = trailing.reset_index().melt(
            id_vars="periods", var_name="fund", value_name="trailing_return"
        )
        return long.rename(columns={"periods": "period"}).dropna(
            subset=["trailing_return"]
        )

    def get_trailing_returns_dict(
        self,
        periods: Iterable[int] | int,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
    ) -> dict:
        """
        Gets trailing returns as a JSON-friendly dictionary.

        Args:
            periods (Iterable[int] | int): period or iterable of periods to calculate.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None): optional collection of funds to include.

        Returns:
            dict: dictionary with periods list and per-fund trailing return mappings.
        """
        trailing = self.get_trailing_returns(periods=periods, fund=fund, funds=funds)
        if trailing.empty:
            raise ValueError("no price data available for trailing returns")
        periods_list = [int(period) for period in trailing.index.tolist()]
        payload: dict[str, dict[str, float | None]] = {}
        for fund_name in trailing.columns:
            payload[fund_name] = {
                str(period): self._format_numeric_for_output(
                    trailing.loc[period, fund_name]
                )
                for period in trailing.index
            }
        return {"periods": periods_list, "funds": payload}

    def get_rolling_mean(self, fund: FundInput, window: int = 20) -> DataFrame:
        """
        Gets a rolling mean for the specified fund.

        Args:
            fund (FundInput): the fund to retrieve the rolling mean for.
            window (int): rolling window size.

        Returns:
            DataFrame: dataframe of rolling mean values.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        prices = self.dataframe[["Date", fund_name]].set_index("Date")
        return prices.rolling(window=window).mean().dropna(how="all")

    def get_moving_averages(
        self, fund: FundInput, windows: Iterable[int] = (20, 50), method: str = "simple"
    ) -> DataFrame:
        """
        Gets moving averages for the specified fund using one or more window sizes.

        Args:
            fund (FundInput): the fund to analyze.
            windows (Iterable[int]): moving average windows in trading days.
            method (str): 'simple' for SMA or 'exponential' for EMA.

        Returns:
            DataFrame: dataframe of moving average values indexed by date.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        windows_list = self._resolve_periods(windows)
        method_key = method.strip().lower()
        if method_key in ("simple", "sma"):
            label = "sma"

            def calculate(series, window):
                return series.rolling(window=window, min_periods=window).mean()

        elif method_key in ("exponential", "ema"):
            label = "ema"

            def calculate(series, window):
                return series.ewm(span=window, min_periods=window, adjust=False).mean()

        else:
            raise ValueError("method must be 'simple' or 'exponential'")

        prices = (
            self.dataframe[["Date", fund_name]].set_index("Date")[fund_name].dropna()
        )
        moving_averages = {
            f"{label}_{window}": calculate(prices, window) for window in windows_list
        }
        return DataFrame(moving_averages).dropna(how="all")

    def get_rolling_returns(self, fund: FundInput, window: int = 20) -> DataFrame:
        """
        Gets rolling returns for the specified fund over the given window.

        Args:
            fund (FundInput): the fund to retrieve rolling returns for.
            window (int): rolling window size.

        Returns:
            DataFrame: dataframe of rolling return values.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(window, "window")
        prices = self.dataframe[["Date", fund_name]].set_index("Date")
        return prices.pct_change(periods=window, fill_method=None).dropna(how="all")

    def get_rolling_correlation(
        self, fund_a: FundInput, fund_b: FundInput, window: int = 63
    ) -> DataFrame:
        """
        Gets rolling correlation of daily returns between two funds.

        Args:
            fund_a (FundInput): first fund to compare.
            fund_b (FundInput): second fund to compare.
            window (int): rolling window size.

        Returns:
            DataFrame: dataframe of rolling correlation values.
        """
        self._validate_positive_int(window, "window")
        fund_a_name = self._ensure_fund_available(fund_a)
        fund_b_name = self._ensure_fund_available(fund_b)
        returns = self.get_daily_returns()
        for fund_name in (fund_a_name, fund_b_name):
            if fund_name not in returns.columns:
                raise ValueError(f"fund not available in return data: {fund_name}")
        correlation = (
            returns[fund_a_name].rolling(window=window).corr(returns[fund_b_name])
        )
        return correlation.dropna(how="all").to_frame(name="rolling_correlation")

    def get_tracking_error(
        self,
        fund: FundInput,
        benchmark: FundInput,
        trading_days: int = 252,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the annualized tracking error of a fund versus a benchmark.

        Args:
            fund (FundInput): fund to evaluate.
            benchmark (FundInput): benchmark fund.
            trading_days (int): trading days per year used for annualization.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with tracking error values indexed by fund.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        excess = self.get_excess_returns(fund=fund_name, benchmark=benchmark_name)
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=excess.reset_index()
            )
            excess = filtered.set_index("Date")
        if excess.empty:
            raise ValueError("no return data available for tracking error")
        tracking_error = excess["excess_return"].std() * (trading_days**0.5)
        return DataFrame(
            {"tracking_error": [float(tracking_error)]}, index=[fund_name]
        ).rename_axis("fund")

    def get_information_ratio(
        self,
        fund: FundInput,
        benchmark: FundInput,
        trading_days: int = 252,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets the annualized information ratio of a fund versus a benchmark.

        Args:
            fund (FundInput): fund to evaluate.
            benchmark (FundInput): benchmark fund.
            trading_days (int): trading days per year used for annualization.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with annualized excess return, tracking error, and information ratio.
        """
        self._ensure_dataframe()
        self._validate_positive_int(trading_days, "trading_days")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        excess = self.get_excess_returns(fund=fund_name, benchmark=benchmark_name)
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=excess.reset_index()
            )
            excess = filtered.set_index("Date")
        if excess.empty:
            raise ValueError("no return data available for information ratio")
        annualized_excess_return = excess["excess_return"].mean() * trading_days
        tracking_error = excess["excess_return"].std() * (trading_days**0.5)
        info_ratio = (
            annualized_excess_return / tracking_error
            if tracking_error
            else float("nan")
        )
        return DataFrame(
            {
                "annualized_excess_return": [float(annualized_excess_return)],
                "tracking_error": [float(tracking_error)],
                "information_ratio": [float(info_ratio)],
            },
            index=[fund_name],
        ).rename_axis("fund")

    def get_rolling_tracking_error(
        self,
        fund: FundInput,
        benchmark: FundInput,
        window: int = 63,
        trading_days: int = 252,
    ) -> DataFrame:
        """
        Gets rolling annualized tracking error of a fund versus a benchmark.

        Args:
            fund (FundInput): fund to evaluate.
            benchmark (FundInput): benchmark fund.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.

        Returns:
            DataFrame: dataframe of rolling tracking error values.
        """
        self._ensure_dataframe()
        self._validate_positive_int(window, "window")
        self._validate_positive_int(trading_days, "trading_days")
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        excess = self.get_excess_returns(fund=fund_name, benchmark=benchmark_name)
        if excess.empty:
            raise ValueError("no return data available for tracking error")
        rolling_te = (
            excess["excess_return"]
            .rolling(window=window)
            .std()
            .mul(trading_days**0.5)
            .dropna(how="all")
        )
        return rolling_te.to_frame(name="rolling_tracking_error")

    def get_return_histogram(self, fund: FundInput, bins: int = 50) -> DataFrame:
        """
        Gets a histogram of daily returns for the specified fund.

        Args:
            fund (FundInput): the fund to calculate histogram data for.
            bins (int): number of histogram bins.

        Returns:
            DataFrame: dataframe with histogram bin edges and counts.
        """
        self._validate_positive_int(bins, "bins")
        fund_name = self._ensure_fund_available(fund)
        returns = self.get_daily_returns(fund=fund_name)
        if returns.empty:
            raise ValueError("no return data available for histogram")
        series = returns[fund_name].dropna()
        histogram = series.value_counts(bins=bins, sort=False)
        counts = histogram.values
        edges = histogram.index
        return DataFrame(
            {
                "bin_left": [interval.left for interval in edges],
                "bin_right": [interval.right for interval in edges],
                "count": counts,
            }
        )
