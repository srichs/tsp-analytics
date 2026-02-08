"""Charting helpers for visualizing TSP prices, returns, and allocations."""

from collections.abc import Iterable, Mapping
from datetime import date
import os

import matplotlib

if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

from matplotlib import pyplot as plt, ticker as mtick
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tsp.funds import FundInput, TspIndividualFund


class ChartsMixin:
    """Generate Matplotlib charts for common TSP analytics outputs."""

    def show_pie_chart(
        self, allocation: dict, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a pie chart with the allocation percentages in each fund.

        Args:
            allocation(dict): a dict describing the share prices and user's allocation.
        """
        self.logger.debug("generating allocation pie chart")
        if (
            "allocation_percent" not in allocation
            or not isinstance(allocation["allocation_percent"], Mapping)
            or not allocation["allocation_percent"]
        ):
            raise ValueError(
                "allocation_percent must contain at least one fund allocation to plot"
            )
        labels = []
        sizes = []

        for fund, percent in allocation["allocation_percent"].items():
            labels.append(fund)
            sizes.append(float(percent))

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%")
        if show:
            plt.show()
        return fig, ax

    def show_individual_price_chart(self, *, show: bool = True) -> tuple[Figure, Axes]:
        """
        Show a line chart with the individual funds prices plotted.
        """
        self._ensure_dataframe()
        self.logger.debug("generating individual funds price line chart")
        df = self.dataframe.drop(self.LIFECYCLE_FUNDS, axis=1, errors="ignore").dropna(
            how="all"
        )
        self._validate_chart_dataframe(df, "individual fund price chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(
            x="Date",
            title="Historical Individual Fund Price Chart",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_lifecycle_price_chart(self, *, show: bool = True) -> tuple[Figure, Axes]:
        """
        Show a line chart with the lifecycle funds prices plotted.
        """
        self._ensure_dataframe()
        self.logger.debug("generating lifecycle funds price line chart")
        df = self.dataframe.drop(self.INDIVIDUAL_FUNDS, axis=1, errors="ignore").dropna(
            how="all"
        )
        self._validate_chart_dataframe(df, "lifecycle fund price chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(
            x="Date",
            title="Historical Lifecycle Fund Price Chart",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_fund_price_chart(
        self, fund: FundInput, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with the specified funds prices plotted.

        Args:
            fund (FundInput): the fund to plot the prices for.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self.logger.debug(f"generating price line chart for {fund_name}")
        df = self.dataframe[["Date", fund_name]].dropna(how="all")
        self._validate_chart_dataframe(df, f"{fund_name} price chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(
            x="Date",
            title=f"{fund_name} Price Chart",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_individual_price_chart_by_date_range(
        self, start_date: date, end_date: date, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with the individual funds prices plotted for the given date range.

        Args:
            start_date (date): the start date of the date range to plot.
            end_date (date): the end date of the date range to plot.
        """
        self._ensure_dataframe()
        self.logger.debug(
            f"generating individual funds line chart for {start_date} to {end_date}"
        )
        df = self.dataframe.drop(self.LIFECYCLE_FUNDS, axis=1, errors="ignore").dropna(
            how="all"
        )
        df = self._filter_by_date_range(start_date, end_date, dataframe=df)
        self._validate_chart_dataframe(df, "individual fund price chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(
            x="Date",
            title=f"Individual Funds Price Chart ({start_date} to {end_date})",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_lifecycle_price_chart_by_date_range(
        self, start_date: date, end_date: date, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with the lifecycle funds prices plotted for the given date range.

        Args:
            start_date (date): the start date of the date range to plot.
            end_date (date): the end date of the date range to plot.
        """
        self._ensure_dataframe()
        self.logger.debug(
            f"generating lifecycle funds line chart for {start_date} to {end_date}"
        )
        df = self.dataframe.drop(self.INDIVIDUAL_FUNDS, axis=1, errors="ignore").dropna(
            how="all"
        )
        df = self._filter_by_date_range(start_date, end_date, dataframe=df)
        self._validate_chart_dataframe(df, "lifecycle fund price chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(
            x="Date",
            title=f"Lifecycle Funds Price Chart ({start_date} to {end_date})",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_fund_price_chart_by_date_range(
        self, start_date: date, end_date: date, fund: FundInput, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with the specified funds prices plotted for the given date range.

        Args:
            start_date (date): the start date of the date range to plot.
            end_date (date): the end date of the date range to plot.
            fund (FundInput): the fund to plot the data for.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self.logger.debug(
            f"generating price line chart for {fund_name} from {start_date} to {end_date}"
        )
        df = self.dataframe[["Date", fund_name]].dropna(how="all")
        df = self._filter_by_date_range(start_date, end_date, dataframe=df)
        self._validate_chart_dataframe(df, f"{fund_name} price chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(
            x="Date",
            title=f"{fund_name} Price Chart ({start_date} to {end_date})",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_cumulative_returns_chart(
        self, fund: FundInput | None = None, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with cumulative returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to plot cumulative returns for.
        """
        self._ensure_dataframe()
        self.logger.debug("generating cumulative returns chart")
        cumulative_returns = self.get_cumulative_returns(fund=fund)
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative_returns.plot(
            title="Cumulative Returns", xlabel="Date", ylabel="Cumulative Return", ax=ax
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_portfolio_value_chart(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        initial_value: float = 10_000.0,
        normalize_weights: bool = True,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart of portfolio value based on weighted daily returns.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            initial_value (float): starting portfolio value.
            normalize_weights (bool): normalize weights to sum to 1.
        """
        self._ensure_dataframe()
        self.logger.debug("generating portfolio value chart")
        portfolio_values = self.get_portfolio_value_history(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            initial_value=initial_value,
            normalize_weights=normalize_weights,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        portfolio_values.plot(
            title="Portfolio Value",
            xlabel="Date",
            ylabel="Value ($)",
            legend=False,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_portfolio_drawdown_chart(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        normalize_weights: bool = True,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart of portfolio drawdowns based on weighted daily returns.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            normalize_weights (bool): normalize weights to sum to 1.
        """
        self._ensure_dataframe()
        self.logger.debug("generating portfolio drawdown chart")
        drawdown = self.get_portfolio_drawdown_series(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        drawdown.plot(
            title="Portfolio Drawdown",
            xlabel="Date",
            ylabel="Drawdown",
            legend=False,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_price_history_chart(
        self,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with price history for selected funds.

        Args:
            funds (Iterable[FundInput] | None):
                funds to plot. When None, plots all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
        """
        self._ensure_dataframe()
        self.logger.debug("generating price history chart")
        price_history = self.get_price_history(
            funds=funds, start_date=start_date, end_date=end_date
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        price_history.plot(
            x="Date",
            title="Fund Price History",
            xlabel="Date",
            ylabel="Price ($)",
            linewidth=1,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_normalized_price_chart(
        self,
        fund: FundInput | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        base_value: float = 100.0,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with normalized prices rebased to a starting value.

        Args:
            fund (FundInput | None): the fund to plot. If None, includes all funds.
            start_date (date | None): optional start date to normalize from.
            end_date (date | None): optional end date to normalize to.
            base_value (float): base value to normalize prices to.
        """
        self._ensure_dataframe()
        self.logger.debug("generating normalized price chart")
        normalized_prices = self.get_normalized_prices(
            fund=fund, start_date=start_date, end_date=end_date, base_value=base_value
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        normalized_prices.plot(
            title="Normalized Prices", xlabel="Date", ylabel="Normalized Price", ax=ax
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_monthly_returns_chart(
        self, fund: FundInput | None = None, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with monthly returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to plot monthly returns for.
        """
        self._ensure_dataframe()
        self.logger.debug("generating monthly returns chart")
        monthly_returns = self.get_monthly_returns(fund=fund)
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_returns.plot(
            kind="bar", title="Monthly Returns", xlabel="Month", ylabel="Return", ax=ax
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_return_histogram_chart(
        self, fund: FundInput, bins: int = 50, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a histogram chart of daily returns for a fund.

        Args:
            fund (FundInput): the fund to plot.
            bins (int): number of histogram bins.
        """
        self._ensure_dataframe()
        self._validate_positive_int(bins, "bins")
        fund_name = self._ensure_fund_available(fund)
        self.logger.debug("generating return histogram chart")
        returns = self.get_daily_returns(fund=fund_name)
        if returns.empty:
            raise ValueError("no return data available for histogram chart")
        series = returns[fund_name].dropna()
        if series.empty:
            raise ValueError("no return data available for histogram chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(series, bins=bins, color="tab:blue", alpha=0.7, edgecolor="white")
        mean = series.mean()
        median = series.median()
        ax.axvline(
            mean,
            color="tab:orange",
            linestyle="--",
            linewidth=1,
            label=f"Mean {mean:.2%}",
        )
        ax.axvline(
            median,
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label=f"Median {median:.2%}",
        )
        ax.set_title(f"{fund_name} Daily Return Distribution")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        if show:
            plt.show()
        return fig, ax

    def show_yearly_returns_chart(
        self, fund: FundInput | None = None, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with yearly returns for the specified fund or all funds.

        Args:
            fund (FundInput | None): the fund to plot yearly returns for.
        """
        self._ensure_dataframe()
        self.logger.debug("generating yearly returns chart")
        yearly_returns = self.get_yearly_returns(fund=fund)
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_returns.plot(
            kind="bar", title="Yearly Returns", xlabel="Year", ylabel="Return", ax=ax
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_monthly_return_heatmap(
        self,
        fund: FundInput,
        start_date: date | None = None,
        end_date: date | None = None,
        cmap: str = "RdYlGn",
        vmin: float = -0.1,
        vmax: float = 0.1,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a heatmap of monthly returns for the specified fund.

        Args:
            fund (FundInput): the fund to visualize.
            start_date (date | None): optional start date for the heatmap.
            end_date (date | None): optional end date for the heatmap.
            cmap (str): Matplotlib colormap name.
            vmin (float): minimum value for the color scale.
            vmax (float): maximum value for the color scale.
        """
        self._ensure_dataframe()
        self.logger.debug("generating monthly return heatmap")
        fund_name = self._ensure_fund_available(fund)
        table = self.get_monthly_return_table(
            fund=fund_name, start_date=start_date, end_date=end_date
        )
        fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.4 + 2)))
        heatmap = ax.imshow(
            table.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        ax.set_title(f"{fund_name} Monthly Returns")
        colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_rolling_mean_chart(
        self, fund: FundInput, window: int = 20, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with a rolling mean for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling mean chart")
        fund_name = self._ensure_fund_available(fund)
        rolling_mean = self.get_rolling_mean(fund=fund_name, window=window)
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_mean.plot(
            title=f"{fund_name} Rolling Mean ({window}D)",
            xlabel="Date",
            ylabel="Price ($)",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_returns_chart(
        self, fund: FundInput, window: int = 20, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling returns for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling returns chart")
        fund_name = self._ensure_fund_available(fund)
        rolling_returns = self.get_rolling_returns(fund=fund_name, window=window)
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_returns.plot(
            title=f"{fund_name} Rolling Returns ({window}D)",
            xlabel="Date",
            ylabel="Return",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_moving_average_chart(
        self,
        fund: FundInput,
        windows: Iterable[int] = (20, 50),
        method: str = "simple",
        show_price: bool = True,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart of moving averages for a fund, optionally overlaying the price series.

        Args:
            fund (FundInput): the fund to plot.
            windows (Iterable[int]): moving average windows in trading days.
            method (str): 'simple' for SMA or 'exponential' for EMA.
            show_price (bool): whether to plot the price series alongside moving averages.
        """
        self._ensure_dataframe()
        self.logger.debug("generating moving average chart")
        fund_name = self._ensure_fund_available(fund)
        moving_averages = self.get_moving_averages(
            fund=fund_name, windows=windows, method=method
        ).reset_index()
        if moving_averages.empty:
            raise ValueError("no data available to plot moving averages")
        if show_price:
            prices = self.dataframe[["Date", fund_name]].dropna(how="all")
            moving_averages = moving_averages.merge(prices, on="Date", how="left")
        self._validate_chart_dataframe(moving_averages, "moving average chart")
        title = f"{fund_name} Moving Averages ({method.title()})"
        fig, ax = plt.subplots(figsize=(10, 6))
        moving_averages.plot(
            x="Date", title=title, xlabel="Date", ylabel="Price ($)", ax=ax
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_correlation_chart(
        self,
        fund_a: FundInput,
        fund_b: FundInput,
        window: int = 63,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling correlations between two funds.

        Args:
            fund_a (FundInput): first fund to compare.
            fund_b (FundInput): second fund to compare.
            window (int): rolling window size.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling correlation chart")
        fund_a_name = self._ensure_fund_available(fund_a)
        fund_b_name = self._ensure_fund_available(fund_b)
        rolling_correlation = self.get_rolling_correlation(
            fund_a=fund_a_name, fund_b=fund_b_name, window=window
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_correlation.plot(
            title=f"{fund_a_name} vs {fund_b_name} Rolling Correlation ({window}D)",
            xlabel="Date",
            ylabel="Correlation",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_tracking_error_chart(
        self,
        fund: FundInput,
        benchmark: FundInput,
        window: int = 63,
        trading_days: int = 252,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling annualized tracking error versus a benchmark.

        Args:
            fund (FundInput): the fund to plot.
            benchmark (FundInput): benchmark fund.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling tracking error chart")
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        rolling_tracking_error = self.get_rolling_tracking_error(
            fund=fund_name,
            benchmark=benchmark_name,
            window=window,
            trading_days=trading_days,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_tracking_error.plot(
            title=f"{fund_name} Rolling Tracking Error vs {benchmark_name} ({window}D)",
            xlabel="Date",
            ylabel="Tracking Error",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_sharpe_ratio_chart(
        self,
        fund: FundInput,
        window: int = 63,
        trading_days: int = 252,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling annualized Sharpe ratios for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling Sharpe ratio chart")
        fund_name = self._ensure_fund_available(fund)
        rolling_sharpe = self.get_rolling_sharpe_ratio(
            fund=fund_name, window=window, trading_days=trading_days
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_sharpe.plot(
            title=f"{fund_name} Rolling Sharpe Ratio ({window}D)",
            xlabel="Date",
            ylabel="Sharpe Ratio",
            ax=ax,
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_sortino_ratio_chart(
        self,
        fund: FundInput,
        window: int = 63,
        trading_days: int = 252,
        mar: float = 0.0,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling annualized Sortino ratios for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for the Sortino ratio.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling Sortino ratio chart")
        fund_name = self._ensure_fund_available(fund)
        rolling_sortino = self.get_rolling_sortino_ratio(
            fund=fund_name, window=window, trading_days=trading_days, mar=mar
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_sortino.plot(
            title=f"{fund_name} Rolling Sortino Ratio ({window}D)",
            xlabel="Date",
            ylabel="Sortino Ratio",
            ax=ax,
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_performance_summary_chart(
        self,
        fund: FundInput,
        window: int = 63,
        trading_days: int = 252,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a chart of rolling annualized return, volatility, and Sharpe ratio.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling performance summary chart")
        fund_name = self._ensure_fund_available(fund)
        summary = self.get_rolling_performance_summary(
            fund=fund_name, window=window, trading_days=trading_days
        )
        self._validate_chart_dataframe(
            summary.reset_index(), "rolling performance summary chart"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        summary[["rolling_return", "rolling_volatility"]].plot(ax=ax)
        ax.set_title(f"{fund_name} Rolling Performance ({window}D)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Return / Volatility")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)

        ax_secondary = ax.twinx()
        summary["rolling_sharpe_ratio"].plot(
            ax=ax_secondary,
            color="tab:green",
            linestyle="--",
            label="Rolling Sharpe Ratio",
        )
        ax_secondary.set_ylabel("Sharpe Ratio")
        ax_secondary.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}"))

        primary_lines, primary_labels = ax.get_legend_handles_labels()
        secondary_lines, secondary_labels = ax_secondary.get_legend_handles_labels()
        ax.legend(
            primary_lines + secondary_lines,
            primary_labels + secondary_labels,
            loc="best",
        )

        if show:
            plt.show()
        return fig, ax

    def show_rolling_volatility_chart(
        self,
        fund: FundInput,
        window: int = 20,
        trading_days: int = 252,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling annualized volatility for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
            trading_days (int): trading days per year used for annualization.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling volatility chart")
        fund_name = self._ensure_fund_available(fund)
        rolling_volatility = self.get_rolling_volatility(
            fund=fund_name, window=window, trading_days=trading_days
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_volatility.plot(
            title=f"{fund_name} Rolling Volatility ({window}D)",
            xlabel="Date",
            ylabel="Volatility",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_drawdown_chart(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart of drawdowns for the specified fund or funds.

        Args:
            fund (FundInput | None): optional fund to plot drawdowns for.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
        """
        self._ensure_dataframe()
        self.logger.debug("generating drawdown chart")
        drawdown = self.get_drawdown_series(fund=fund, funds=funds)
        self._validate_chart_dataframe(drawdown.reset_index(), "drawdown chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        drawdown.plot(title="Drawdown", xlabel="Date", ylabel="Drawdown", ax=ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_excess_returns_chart(
        self,
        fund: FundInput | None = None,
        benchmark: FundInput = TspIndividualFund.G_FUND,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart of daily excess returns versus a benchmark fund.

        Args:
            fund (FundInput | None): optional fund to plot. If None, plots all funds vs benchmark.
            benchmark (FundInput): benchmark fund for excess return calculations.
        """
        self._ensure_dataframe()
        self.logger.debug("generating excess returns chart")
        benchmark_name = self._ensure_fund_available(benchmark)
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            excess = self.get_excess_returns(fund=fund_name, benchmark=benchmark_name)
            title = f"{fund_name} Excess Returns vs {benchmark_name}"
        else:
            excess = self.get_excess_returns(benchmark=benchmark_name)
            title = f"Excess Returns vs {benchmark_name}"
        self._validate_chart_dataframe(excess.reset_index(), "excess returns chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        excess.plot(title=title, xlabel="Date", ylabel="Excess Return", ax=ax)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_max_drawdown_chart(
        self, fund: FundInput, window: int = 252, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart of rolling maximum drawdown for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            window (int): rolling window size.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling max drawdown chart")
        fund_name = self._ensure_fund_available(fund)
        rolling_drawdown = self.get_rolling_max_drawdown(fund=fund_name, window=window)
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_drawdown.plot(
            title=f"{fund_name} Rolling Max Drawdown ({window}D)",
            xlabel="Date",
            ylabel="Rolling Max Drawdown",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_correlation_heatmap(self, *, show: bool = True) -> tuple[Figure, Axes]:
        """
        Show a heatmap of correlations between fund daily returns.
        """
        self._ensure_dataframe()
        self.logger.debug("generating correlation heatmap")
        correlation = self.get_correlation_matrix()
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.index)))
        ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation.index)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        plt.title("Fund Return Correlations")
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_rolling_correlation_heatmap(
        self, window: int = 63, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a heatmap of correlations between fund daily returns for a recent window.

        Args:
            window (int): number of recent trading days to include.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling correlation heatmap")
        correlation = self.get_rolling_correlation_matrix(window=window)
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.index)))
        ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation.index)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        plt.title(f"Fund Return Correlations (Last {window} Days)")
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_correlation_pairs_chart(
        self,
        *,
        top_n: int | None = 10,
        absolute: bool = True,
        min_abs_correlation: float | None = None,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart of the strongest correlation pairs between funds.

        Args:
            top_n (int | None): optional limit on the number of pairs displayed.
            absolute (bool): when True, rank by absolute correlation magnitude.
            min_abs_correlation (float | None): optional threshold between 0 and 1 to filter pairs.
        """
        self._ensure_dataframe()
        self.logger.debug("generating correlation pairs chart")
        pairs = self.get_correlation_pairs(
            top_n=top_n, absolute=absolute, min_abs_correlation=min_abs_correlation
        )
        if pairs.empty:
            raise ValueError("no correlation pairs available to plot")
        labels = [f"{row['fund_a']} vs {row['fund_b']}" for _, row in pairs.iterrows()]
        values = pairs["correlation"]
        colors = ["tab:blue" if value >= 0 else "tab:red" for value in values]
        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4 + 2)))
        ax.barh(labels, values, color=colors)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title("Top Fund Correlation Pairs")
        ax.set_xlabel("Correlation")
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}"))
        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_latest_price_change_chart(
        self, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with the latest daily percent change across all funds.
        """
        self._ensure_dataframe()
        self.logger.debug("generating latest price change chart")
        changes = self.get_latest_price_changes()
        fig, ax = plt.subplots(figsize=(10, 6))
        changes["change_percent"].plot(
            kind="bar",
            title="Latest Daily Price Change",
            xlabel="Fund",
            ylabel="Percent Change",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_latest_price_changes_per_fund_chart(
        self, funds: Iterable[FundInput] | None = None, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with the latest daily percent change per fund.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
        """
        self._ensure_dataframe()
        self.logger.debug("generating latest price changes per fund chart")
        changes = self.get_latest_price_changes_per_fund(funds=funds)
        if changes.empty:
            raise ValueError("no price data available to plot per-fund changes")
        fig, ax = plt.subplots(figsize=(10, 6))
        changes["change_percent"].plot(
            kind="bar",
            title="Latest Daily Price Change (Per Fund)",
            xlabel="Fund",
            ylabel="Percent Change",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_recent_price_change_heatmap(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a heatmap of recent daily percent changes by fund.

        Args:
            days (int): number of recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date.
        """
        self._ensure_dataframe()
        self.logger.debug("generating recent price change heatmap")
        changes = self.get_recent_price_changes(
            days=days, fund=fund, funds=funds, as_of=as_of
        )
        if changes.empty:
            raise ValueError("no price changes available to plot")
        values = changes.values
        max_abs = float(abs(values).max())
        if max_abs == 0:
            max_abs = 1.0
        fig_width = max(8, len(changes.columns) * 1.2)
        fig_height = max(4, len(changes.index) * 0.5 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        heatmap = ax.imshow(
            values, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs
        )
        ax.set_xticks(range(len(changes.columns)))
        ax.set_xticklabels(changes.columns, rotation=45, ha="right")
        date_labels = [value.date().isoformat() for value in changes.index]
        ax.set_yticks(range(len(changes.index)))
        ax.set_yticklabels(date_labels)
        ax.set_title("Recent Daily Price Changes (%)")
        ax.set_xlabel("Fund")
        ax.set_ylabel("Date")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_latest_prices_per_fund_chart(
        self, funds: Iterable[FundInput] | None = None, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with the latest available price per fund.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
        """
        self._ensure_dataframe()
        self.logger.debug("generating latest prices per fund chart")
        latest = self.get_latest_prices_per_fund(funds=funds)
        if latest.empty:
            raise ValueError("no price data available to plot latest prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        latest["price"].plot(
            kind="bar",
            title="Latest Prices (Per Fund)",
            xlabel="Fund",
            ylabel="Price ($)",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_current_prices_per_fund_chart(
        self,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        sort_by: str = "price",
        ascending: bool = False,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with the current (latest available) price per fund.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, uses each fund's most recent
                price on or before the date.
            sort_by (str): sort order for the bars: "price" or "fund".
            ascending (bool): whether to sort in ascending order.
        """
        self._ensure_dataframe()
        self.logger.debug("generating current prices per fund chart")
        current = self.get_current_prices_per_fund(funds=funds, as_of=as_of)
        if current.empty:
            raise ValueError("no price data available to plot current prices")

        sort_key = sort_by.strip().lower()
        if sort_key == "price":
            current = current.sort_values("price", ascending=ascending)
        elif sort_key == "fund":
            current = current.sort_index(ascending=ascending)
        else:
            raise ValueError('sort_by must be "price" or "fund"')

        fig, ax = plt.subplots(figsize=(10, 6))
        current["price"].plot(
            kind="bar",
            title="Current Prices (Per Fund)",
            xlabel="Fund",
            ylabel="Price ($)",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
        subtitle = f"As of {as_of}" if as_of else f"As of {self.latest}"
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        if show:
            plt.show()
        return fig, ax

    def show_price_recency_chart(
        self,
        funds: Iterable[FundInput] | None = None,
        reference_date: date | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart of the number of days since each fund's most recent price.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            reference_date (date | None):
                optional reference date. When provided, recency is calculated relative to this
                date using prices on or before it. When None, uses the latest available date.
        """
        self.logger.debug("generating price recency chart")
        recency = self.get_price_recency(funds=funds, reference_date=reference_date)
        if recency.empty:
            raise ValueError("no price data available to plot price recency")
        recency_sorted = recency.sort_values("days_since", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(recency_sorted.index, recency_sorted["days_since"])
        ref_date = reference_date or self.latest
        ax.set_title(f"Price Recency (as of {ref_date})")
        ax.set_xlabel("Fund")
        ax.set_ylabel("Days Since Last Price")
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        if show:
            plt.show()
        return fig, ax

    def show_current_price_alerts_chart(
        self,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        stale_days: int = 3,
        change_threshold: float | None = 0.02,
        metric: str = "change_percent",
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart of current price alerts (staleness and large moves).

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None): optional historical anchor date.
            reference_date (date | None): optional reference date for recency calculations.
            stale_days (int): number of days after which a price is considered stale.
            change_threshold (float | None): absolute daily change threshold for large moves.
            metric (str): metric to visualize: 'change_percent', 'change', 'days_since', or 'price'.
        """
        self.logger.debug("generating current price alerts chart")
        alerts = self.get_current_price_alerts(
            funds=funds,
            as_of=as_of,
            reference_date=reference_date,
            stale_days=stale_days,
            change_threshold=change_threshold,
        )
        if alerts.empty:
            raise ValueError("no alert data available to plot")

        metric_key = metric.strip().lower().replace(" ", "_")
        metric_map = {
            "change_percent": ("change_percent", "Change (%)"),
            "change": ("change", "Change"),
            "days_since": ("days_since", "Days Since"),
            "price": ("price", "Price ($)"),
        }
        if metric_key not in metric_map:
            raise ValueError(
                "metric must be one of: change_percent, change, days_since, price"
            )
        column, label = metric_map[metric_key]

        values = alerts[column]
        colors = []
        edges = []
        for _, row in alerts.iterrows():
            if row["is_stale"]:
                colors.append("tab:gray")
            elif metric_key in {"change_percent", "change"}:
                colors.append("tab:green" if row[column] >= 0 else "tab:red")
            else:
                colors.append("tab:blue")
            edges.append("black" if row["is_large_move"] else "none")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(alerts.index, values, color=colors, edgecolor=edges)
        ref_date = reference_date or as_of or self.latest
        ax.set_title(f"Current Price Alerts (as of {ref_date})")
        ax.set_xlabel("Fund")
        ax.set_ylabel(label)
        if metric_key == "change_percent":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        elif metric_key == "price":
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_price_change_chart_as_of(
        self, as_of: date, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart with daily percent changes as of a specific date.

        Args:
            as_of (date): date to anchor the price change calculation.
        """
        self._ensure_dataframe()
        self.logger.debug("generating price change chart as of %s", as_of)
        changes = self.get_price_changes_as_of(as_of)
        fig, ax = plt.subplots(figsize=(10, 6))
        changes["change_percent"].plot(
            kind="bar",
            title=f"Price Change as of {as_of}",
            xlabel="Fund",
            ylabel="Percent Change",
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_fund_coverage_chart(self, *, show: bool = True) -> tuple[Figure, Axes]:
        """
        Show a bar chart of data coverage percentage for each fund.
        """
        self._ensure_dataframe()
        self.logger.debug("generating fund coverage chart")
        summary = self.get_fund_coverage_summary()
        fig, ax = plt.subplots(figsize=(10, 6))
        summary["coverage_percent"].plot(
            kind="bar", title="Fund Coverage", xlabel="Fund", ylabel="Coverage", ax=ax
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_missing_business_days_chart(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a timeline chart highlighting missing business days in the dataset.

        Args:
            start_date (date | None): optional start date to check.
            end_date (date | None): optional end date to check.
        """
        self._ensure_dataframe()
        self.logger.debug("generating missing business days chart")
        missing = self.get_missing_business_days(
            start_date=start_date, end_date=end_date
        )
        if missing.empty:
            raise ValueError("no missing business days to plot")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.scatter(missing["Date"], [1] * len(missing), marker="|", color="tab:red")
        ax.set_title("Missing Business Days")
        ax.set_xlabel("Date")
        ax.set_yticks([])
        ax.set_ylim(0.5, 1.5)
        plt.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_current_price_dashboard_metric_chart(
        self,
        metric: str,
        period: int | None = None,
        funds: Iterable[FundInput] | None = None,
        periods: Iterable[int] = (5, 20, 63, 252),
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
        reference_date: date | None = None,
        top_n: int | None = None,
        ascending: bool | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart of a selected metric from the current price dashboard snapshot.

        Args:
            metric (str): dashboard metric to plot (e.g., "change_percent", "days_since").
            period (int | None): trailing-return period when metric is "trailing_return".
            funds (Iterable[FundInput] | None): optional collection of funds to include.
            periods (Iterable[int]): trailing return periods to include in the dashboard.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.
            reference_date (date | None):
                optional reference date for recency calculations. When None, uses the latest
                available date in the dataset.
            top_n (int | None): optional limit of funds to display.
            ascending (bool | None): override the default sort direction.
        """
        self._ensure_dataframe()
        self.logger.debug("generating current price dashboard metric chart")

        metric_key = metric.strip().lower().replace(" ", "_")
        if metric_key == "trailing_return":
            if period is None:
                raise ValueError("period is required when metric is trailing_return")
            resolved_periods = self._resolve_periods(periods)
            if period not in resolved_periods:
                resolved_periods.append(period)
            metric_column = f"trailing_return_{int(period)}d"
            label = f"Trailing Return ({int(period)}D)"
        else:
            resolved_periods = self._resolve_periods(periods)
            metric_column = metric_key
            label = metric_key.replace("_", " ").title()

        dashboard = self.get_current_price_dashboard(
            funds=funds,
            periods=resolved_periods,
            trading_days=trading_days,
            mar=mar,
            confidence=confidence,
            reference_date=reference_date,
        )
        if metric_column not in dashboard.columns:
            raise ValueError(f"metric not available in dashboard: {metric_column}")

        data = dashboard[metric_column].dropna()
        if data.empty:
            raise ValueError("no data available to plot dashboard metric")

        if ascending is None:
            ascending = False
        data = data.sort_values(ascending=ascending)
        if top_n is not None:
            self._validate_positive_int(top_n, "top_n")
            data = data.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(data.index, data.values)
        ax.set_title(f"Current Price Dashboard: {label}")
        ax.set_xlabel("Fund")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)

        percent_metrics = {
            "change_percent",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "value_at_risk",
            "expected_shortfall",
        }
        if metric_key == "trailing_return" or metric_column in percent_metrics:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_risk_return_scatter(
        self,
        fund: FundInput | None = None,
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a scatter plot of annualized volatility vs. annualized return.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.
        """
        self._ensure_dataframe()
        self.logger.debug("generating risk/return scatter plot")
        summary = self.get_risk_return_summary(
            fund=fund, trading_days=trading_days, mar=mar, confidence=confidence
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(summary["annualized_volatility"], summary["annualized_return"])
        for fund_name, row in summary.iterrows():
            ax.annotate(
                fund_name,
                (row["annualized_volatility"], row["annualized_return"]),
                textcoords="offset points",
                xytext=(5, 5),
            )
        ax.set_title("Risk vs. Return")
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_fund_rankings_chart(
        self,
        metric: str,
        period: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of: date | None = None,
        reference_date: date | None = None,
        trading_days: int = 252,
        top_n: int | None = 10,
        funds: Iterable[FundInput] | None = None,
        ascending: bool | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart ranking funds by a specified metric.

        Args:
            metric (str): metric name to rank by.
            period (int | None): trailing-return period for "trailing_return" rankings.
            start_date (date | None): optional start date for performance-based rankings.
            end_date (date | None): optional end date for performance-based rankings.
            as_of (date | None): optional historical anchor date for current price rankings.
            reference_date (date | None): optional reference date for days_since rankings.
            trading_days (int): trading days per year used for annualization.
            top_n (int | None): optional limit of funds to display.
            funds (Iterable[FundInput] | None): optional collection of funds to include.
            ascending (bool | None): override the default sort direction for the metric.
        """
        self._ensure_dataframe()
        self.logger.debug("generating fund ranking chart")
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
            raise ValueError("no data available to plot fund rankings")

        metric_key = metric.strip().lower().replace(" ", "_")
        label = metric_key.replace("_", " ").title()
        if metric_key == "trailing_return" and period is not None:
            label = f"Trailing Return ({period}D)"
        if start_date is not None and end_date is not None:
            title = f"Fund Rankings by {label} ({start_date} to {end_date})"
        else:
            title = f"Fund Rankings by {label}"

        sorted_rankings = rankings.sort_values("rank")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_rankings.index, sorted_rankings["value"])
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.invert_yaxis()

        percent_metrics = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "cagr",
            "trailing_return",
            "change_percent",
        }
        currency_metrics = {"latest_price", "change"}
        if metric_key in percent_metrics:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        elif metric_key in currency_metrics:
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
        plt.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_trailing_returns_chart(
        self,
        periods: Iterable[int] | int = (1, 5, 20, 63, 252),
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a bar chart of trailing returns for one or more funds.

        Args:
            periods (Iterable[int] | int): trailing return periods in trading days.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None): optional collection of funds to include.
        """
        self._ensure_dataframe()
        self.logger.debug("generating trailing returns chart")
        trailing = self.get_trailing_returns(periods=periods, fund=fund, funds=funds)
        if trailing.empty:
            raise ValueError("no trailing return data available to plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        trailing.plot(kind="bar", ax=ax)
        if len(trailing.columns) == 1:
            title = f"Trailing Returns ({trailing.columns[0]})"
        else:
            title = "Trailing Returns"
        ax.set_title(title)
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Trailing Return")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def show_daily_return_histogram(
        self, fund: FundInput, bins: int = 50, *, show: bool = True
    ) -> tuple[Figure, Axes]:
        """
        Show a histogram of daily returns for the specified fund.

        Args:
            fund (FundInput): the fund to plot.
            bins (int): number of histogram bins.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        self._validate_positive_int(bins, "bins")
        self.logger.debug("generating daily returns histogram")
        returns = self.get_daily_returns(fund=fund_name)
        if returns.empty:
            raise ValueError("no return data available for histogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        returns.plot(
            kind="hist",
            bins=bins,
            title=f"{fund_name} Daily Returns",
            xlabel="Daily Return",
            ax=ax,
        )
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax

    def show_rolling_beta_chart(
        self,
        fund: FundInput,
        benchmark: FundInput,
        window: int = 63,
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Show a line chart with rolling beta for a fund relative to a benchmark.

        Args:
            fund (FundInput): the fund to plot.
            benchmark (FundInput): the benchmark fund.
            window (int): rolling window size.
        """
        self._ensure_dataframe()
        self.logger.debug("generating rolling beta chart")
        fund_name = self._ensure_fund_available(fund)
        benchmark_name = self._ensure_fund_available(benchmark)
        rolling_beta = self.get_rolling_beta(
            fund=fund_name, benchmark=benchmark_name, window=window
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_beta.plot(
            title=f"{fund_name} Rolling Beta vs {benchmark_name} ({window}D)",
            xlabel="Date",
            ylabel="Beta",
            ax=ax,
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.grid(True, linestyle="--", alpha=0.6)
        if show:
            plt.show()
        return fig, ax
