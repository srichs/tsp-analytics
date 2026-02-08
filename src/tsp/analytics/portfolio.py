"""Portfolio analytics for weighted allocations and holdings."""

from collections.abc import Iterable, Mapping
from datetime import date

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from tsp.funds import FundInput, TspIndividualFund


class AnalyticsPortfolioMixin:
    """Compute portfolio returns and value series from weighted allocations."""

    def get_portfolio_returns(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Gets daily portfolio returns based on a weighted allocation of funds.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with a portfolio_return column.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        weight_map = self._resolve_weights(weights, normalize_weights=normalize_weights)
        returns = self.get_daily_returns()
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        if returns.empty:
            raise ValueError("no return data available for portfolio")
        weight_series = Series(weight_map)
        portfolio_returns = (
            returns[list(weight_map.keys())].mul(weight_series).sum(axis=1)
        )
        return portfolio_returns.to_frame(name="portfolio_return")

    def get_portfolio_cumulative_returns(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Gets cumulative portfolio returns for the provided weights.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with a portfolio_cumulative_return column.
        """
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )
        cumulative = (1 + portfolio_returns["portfolio_return"]).cumprod() - 1
        return cumulative.to_frame(name="portfolio_cumulative_return")

    def get_portfolio_value_history(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        initial_value: float = 10_000.0,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Gets a portfolio value history series based on weighted daily returns.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            initial_value (float): starting portfolio value.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with a portfolio_value column.
        """
        self._validate_positive_float(initial_value, "initial_value")
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )
        values = (
            (1 + portfolio_returns["portfolio_return"])
            .cumprod()
            .mul(float(initial_value))
        )
        return values.to_frame(name="portfolio_value")

    def get_portfolio_bootstrap_simulation(
        self,
        weights: Mapping[FundInput, float],
        years: int,
        simulations: int = 10_000,
        trading_days: int = 252,
        initial_value: float = 100_000.0,
        start_date: date | None = None,
        end_date: date | None = None,
        normalize_weights: bool = True,
        random_state: int | None = None,
    ) -> DataFrame:
        """
        Simulates long-horizon portfolio outcomes using bootstrap resampling.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            years (int): number of years to simulate.
            simulations (int): number of simulated paths to generate.
            trading_days (int): trading days per year used for sampling length.
            initial_value (float): starting portfolio value.
            start_date (date | None): optional start date for filtering returns.
            end_date (date | None): optional end date for filtering returns.
            normalize_weights (bool): normalize weights to sum to 1.
            random_state (int | None): seed for reproducible sampling.

        Returns:
            DataFrame: dataframe with simulated ending values.
        """
        self._validate_positive_int(years, "years")
        self._validate_positive_int(simulations, "simulations")
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_positive_float(initial_value, "initial_value")
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )["portfolio_return"].dropna()
        if portfolio_returns.empty:
            raise ValueError(
                "no return data available for portfolio bootstrap simulation"
            )
        periods = years * trading_days
        rng = np.random.default_rng(random_state)
        bootstrapped = rng.choice(
            portfolio_returns.values, size=(simulations, periods), replace=True
        )
        ending_values = float(initial_value) * (1 + bootstrapped).prod(axis=1)
        result = DataFrame({"ending_value": ending_values})
        result.index = range(1, simulations + 1)
        result.index.name = "simulation"
        return result

    def get_portfolio_retirement_sequence_analysis(
        self,
        weights: Mapping[FundInput, float],
        initial_value: float,
        annual_withdrawal: float,
        years: int = 30,
        start_date: date | None = None,
        end_date: date | None = None,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Runs a rolling sequence-of-returns retirement drawdown analysis.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            initial_value (float): starting portfolio value.
            annual_withdrawal (float): yearly withdrawal amount.
            years (int): number of retirement years to simulate.
            start_date (date | None): optional start date for filtering returns.
            end_date (date | None): optional end date for filtering returns.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with start dates and ending values for each sequence.
        """
        self._validate_positive_int(years, "years")
        self._validate_positive_float(initial_value, "initial_value")
        self._validate_positive_float(annual_withdrawal, "annual_withdrawal")
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )["portfolio_return"].dropna()
        if portfolio_returns.empty:
            raise ValueError("no return data available for portfolio sequence analysis")
        monthly_returns = portfolio_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_returns = monthly_returns.dropna()
        months = years * 12
        if len(monthly_returns) < months:
            raise ValueError("not enough return history for sequence analysis")
        monthly_withdrawal = float(annual_withdrawal) / 12
        records: list[dict] = []
        for start in range(0, len(monthly_returns) - months + 1):
            sequence = monthly_returns.iloc[start : start + months]
            value = float(initial_value)
            for period_return in sequence:
                value = value * (1 + period_return) - monthly_withdrawal
                if value <= 0:
                    value = 0.0
                    break
            records.append(
                {
                    "start_date": sequence.index[0].date(),
                    "end_date": sequence.index[-1].date(),
                    "ending_value": float(value),
                    "success": value > 0,
                }
            )
        return DataFrame.from_records(records)

    @staticmethod
    def _get_rebalance_dates(
        dates: pd.DatetimeIndex, frequency: str
    ) -> pd.DatetimeIndex:
        freq = frequency.strip().lower()
        if freq not in {"monthly", "quarterly"}:
            raise ValueError('frequency must be "monthly" or "quarterly"')
        period_alias = "M" if freq == "monthly" else "Q"
        periods = dates.to_period(period_alias)
        rebalance_mask = periods != periods.shift(-1)
        return dates[rebalance_mask]

    def _build_trend_weights(
        self, trend_series: Series, base_weights: Series, defensive_fund: str
    ) -> Series:
        if trend_series.isna().all():
            return base_weights
        positive = trend_series[trend_series > 0].index.tolist()
        if not positive:
            weights = Series(0.0, index=base_weights.index)
            if defensive_fund not in weights.index:
                weights[defensive_fund] = 0.0
            weights.loc[defensive_fund] = 1.0
            return weights
        filtered = base_weights.loc[positive]
        total = float(filtered.sum())
        if total <= 0:
            weights = Series(0.0, index=base_weights.index)
            if defensive_fund not in weights.index:
                weights[defensive_fund] = 0.0
            weights.loc[defensive_fund] = 1.0
            return weights
        weights = Series(0.0, index=base_weights.index)
        weights.loc[filtered.index] = filtered.div(total)
        return weights

    def _run_rebalancing_strategy(
        self,
        returns: DataFrame,
        target_weights: Series,
        initial_value: float,
        rebalance_dates: Iterable[pd.Timestamp] | None = None,
        threshold: float | None = None,
        trend_returns: DataFrame | None = None,
        defensive_fund: str | None = None,
    ) -> tuple[Series, Series]:
        if returns.empty:
            raise ValueError("no return data available for portfolio rebalancing")
        weights = target_weights.copy()
        holdings = weights.mul(float(initial_value))
        values: list[float] = []
        turnover_values: list[float] = []
        rebalance_set = set(rebalance_dates) if rebalance_dates is not None else set()

        for current_date, row in returns.iterrows():
            holdings = holdings.mul(1 + row)
            portfolio_value = float(holdings.sum())
            current_weights = (
                holdings.div(portfolio_value) if portfolio_value else weights.copy()
            )
            should_rebalance = current_date in rebalance_set
            if threshold is not None:
                deviations = current_weights.sub(target_weights).abs()
                if deviations.max() >= threshold:
                    should_rebalance = True

            if should_rebalance:
                if trend_returns is not None and defensive_fund is not None:
                    trend_row = trend_returns.loc[current_date]
                    weights = self._build_trend_weights(
                        trend_row, target_weights, defensive_fund
                    )
                else:
                    weights = target_weights.copy()
                turnover = float(weights.sub(current_weights).abs().sum() / 2)
                holdings = weights.mul(portfolio_value)
            else:
                turnover = 0.0
            values.append(portfolio_value)
            turnover_values.append(turnover)

        return Series(values, index=returns.index), Series(
            turnover_values, index=returns.index
        )

    def get_portfolio_rebalancing_backtest(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        initial_value: float = 10_000.0,
        normalize_weights: bool = True,
        trading_days: int = 252,
        rebalance_threshold: float = 0.05,
        trend_window: int = 63,
        trend_frequency: str = "monthly",
        trend_defensive_fund: FundInput = TspIndividualFund.G_FUND,
        strategies: Iterable[str] | None = None,
        include_buy_and_hold: bool = True,
    ) -> dict[str, DataFrame]:
        """
        Backtests portfolio rebalancing strategies with turnover and risk metrics.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering returns.
            end_date (date | None): optional end date for filtering returns.
            initial_value (float): starting portfolio value.
            normalize_weights (bool): normalize weights to sum to 1.
            trading_days (int): trading days per year used for annualization.
            rebalance_threshold (float): deviation threshold for threshold rebalancing.
            trend_window (int): lookback window (trading days) for trend signals.
            trend_frequency (str): "monthly" or "quarterly" trend rebalance schedule.
            trend_defensive_fund (FundInput): fallback fund when all trends are negative.
            strategies (Iterable[str] | None): optional list of strategies to run.
            include_buy_and_hold (bool): include buy-and-hold baseline strategy.

        Returns:
            dict[str, DataFrame]: dictionary with summary, impact, values, drawdowns, and turnover.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        self._validate_positive_float(initial_value, "initial_value")
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_non_negative_float(rebalance_threshold, "rebalance_threshold")
        if rebalance_threshold > 1:
            raise ValueError("rebalance_threshold must be between 0 and 1")
        self._validate_positive_int(trend_window, "trend_window")

        weight_map = self._resolve_weights(weights, normalize_weights=normalize_weights)
        fund_list = list(weight_map.keys())
        returns = self.get_daily_returns().dropna(how="all")
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        returns = returns[fund_list].fillna(0.0)
        if returns.empty:
            raise ValueError("no return data available for portfolio rebalancing")

        defensive_fund = self._resolve_fund(trend_defensive_fund)
        default_strategies = ["monthly", "quarterly", "threshold", "trend"]
        if strategies is None:
            strategies = default_strategies
        strategy_list = [strategy.strip().lower() for strategy in strategies]
        if include_buy_and_hold:
            strategy_list = ["buy_and_hold"] + strategy_list
        if "trend" in strategy_list and defensive_fund not in fund_list:
            fund_list.append(defensive_fund)
            self._ensure_funds_available([defensive_fund])

        target_weights = Series(weight_map).reindex(fund_list, fill_value=0.0)
        returns = returns[fund_list].fillna(0.0)
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        price_df = price_df[fund_list]
        if start_date is not None and end_date is not None:
            filtered_prices = self._filter_by_date_range(
                start_date, end_date, dataframe=price_df.reset_index()
            )
            price_df = filtered_prices.set_index("Date")
        trend_returns = price_df.pct_change(periods=trend_window).reindex(returns.index)

        values_map: dict[str, Series] = {}
        turnover_map: dict[str, Series] = {}

        for strategy in strategy_list:
            if strategy == "buy_and_hold":
                values, turnover = self._run_rebalancing_strategy(
                    returns=returns,
                    target_weights=target_weights,
                    initial_value=initial_value,
                )
            elif strategy in {"monthly", "quarterly"}:
                rebalance_dates = self._get_rebalance_dates(returns.index, strategy)
                values, turnover = self._run_rebalancing_strategy(
                    returns=returns,
                    target_weights=target_weights,
                    initial_value=initial_value,
                    rebalance_dates=rebalance_dates,
                )
            elif strategy == "threshold":
                values, turnover = self._run_rebalancing_strategy(
                    returns=returns,
                    target_weights=target_weights,
                    initial_value=initial_value,
                    threshold=rebalance_threshold,
                )
            elif strategy == "trend":
                rebalance_dates = self._get_rebalance_dates(
                    returns.index, trend_frequency
                )
                values, turnover = self._run_rebalancing_strategy(
                    returns=returns,
                    target_weights=target_weights,
                    initial_value=initial_value,
                    rebalance_dates=rebalance_dates,
                    trend_returns=trend_returns,
                    defensive_fund=defensive_fund,
                )
            else:
                raise ValueError(
                    "strategies must include monthly, quarterly, threshold, trend, or buy_and_hold"
                )
            values_map[strategy] = values
            turnover_map[strategy] = turnover

        values_df = DataFrame(values_map)
        drawdown_df = values_df.div(values_df.cummax()).sub(1)
        turnover_df = DataFrame(turnover_map)

        summary_records: list[dict[str, float | int | str]] = []
        for strategy in values_df.columns:
            stats = self._calculate_performance_summary(
                values_df[strategy], trading_days
            )
            turnover_series = turnover_df[strategy]
            turnover_total = float(turnover_series.sum())
            turnover_count = int((turnover_series > 0).sum())
            turnover_avg = (
                float(turnover_series[turnover_series > 0].mean())
                if turnover_count
                else 0.0
            )
            summary_records.append(
                {
                    "strategy": strategy,
                    **stats,
                    "total_turnover": turnover_total,
                    "average_turnover": turnover_avg,
                    "rebalance_count": turnover_count,
                    "max_drawdown": float(drawdown_df[strategy].min()),
                }
            )
        summary_df = DataFrame.from_records(summary_records).set_index("strategy")

        impact_df = DataFrame()
        if "buy_and_hold" in summary_df.index:
            baseline = summary_df.loc["buy_and_hold"]
            impact_df = (
                summary_df[
                    [
                        "total_return",
                        "annualized_return",
                        "annualized_volatility",
                        "sharpe_ratio",
                        "max_drawdown",
                    ]
                ]
                .sub(baseline)
                .rename(
                    columns={
                        "total_return": "total_return_diff",
                        "annualized_return": "annualized_return_diff",
                        "annualized_volatility": "annualized_volatility_diff",
                        "sharpe_ratio": "sharpe_ratio_diff",
                        "max_drawdown": "max_drawdown_diff",
                    }
                )
            )

        return {
            "summary": summary_df,
            "impact": impact_df,
            "values": values_df,
            "drawdowns": drawdown_df,
            "turnover": turnover_df,
        }

    def get_portfolio_performance_summary(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Gets a performance summary for a weighted portfolio.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            trading_days (int): trading days per year used for annualization.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe of portfolio performance metrics.
        """
        self._validate_positive_int(trading_days, "trading_days")
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )
        price_series = (1 + portfolio_returns["portfolio_return"]).cumprod()
        summary = self._calculate_performance_summary(price_series, trading_days)
        return DataFrame([summary], index=["portfolio"])

    def get_portfolio_drawdown_series(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Gets the portfolio drawdown series based on weighted daily returns.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with a portfolio_drawdown column.
        """
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )
        if portfolio_returns.empty:
            raise ValueError("no return data available for portfolio drawdown")
        portfolio_values = (1 + portfolio_returns["portfolio_return"]).cumprod()
        drawdown = portfolio_values.div(portfolio_values.cummax()).sub(1)
        return drawdown.to_frame(name="portfolio_drawdown")

    def get_portfolio_stress_test_summary(
        self,
        weights: Mapping[FundInput, float],
        windows: dict[str, tuple[date, date]] | None = None,
        trading_days: int = 252,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Runs historical stress tests for a weighted portfolio.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            windows (dict | None): mapping of window name to (start_date, end_date).
            trading_days (int): trading days per year used for annualization.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with stress test performance metrics.
        """
        self._validate_positive_int(trading_days, "trading_days")
        default_windows = {
            "gfc_2008": (date(2007, 10, 1), date(2009, 3, 31)),
            "covid_2020": (date(2020, 2, 19), date(2020, 3, 23)),
        }
        window_map = windows or default_windows
        records: list[dict] = []
        for window_name, (start_date, end_date) in window_map.items():
            self._validate_date_range(start_date, end_date)
            summary = self.get_portfolio_performance_summary(
                weights=weights,
                start_date=start_date,
                end_date=end_date,
                trading_days=trading_days,
                normalize_weights=normalize_weights,
            )
            if summary.empty:
                continue
            record = summary.iloc[0].to_dict()
            record.update(
                {"window": window_name, "start_date": start_date, "end_date": end_date}
            )
            records.append(record)
        return DataFrame.from_records(records)

    def get_portfolio_worst_drawdown_windows(
        self,
        weights: Mapping[FundInput, float],
        window: int = 63,
        top_n: int = 5,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Finds the worst rolling return windows for a weighted portfolio.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            window (int): rolling window size in trading days.
            top_n (int): number of worst windows to return.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with the worst rolling windows.
        """
        self._validate_positive_int(window, "window")
        self._validate_positive_int(top_n, "top_n")
        values = self.get_portfolio_value_history(
            weights=weights, normalize_weights=normalize_weights
        )["portfolio_value"]
        rolling_returns = values.pct_change(periods=window).dropna()
        worst_windows = rolling_returns.nsmallest(top_n)
        records: list[dict] = []
        for end_date, total_return in worst_windows.items():
            start_date = end_date - pd.Timedelta(days=window)
            records.append(
                {
                    "start_date": start_date.date(),
                    "end_date": end_date.date(),
                    "total_return": float(total_return),
                }
            )
        return DataFrame.from_records(records)

    def get_portfolio_shock_scenario_analysis(
        self,
        weights: Mapping[FundInput, float],
        shocks: dict[str, dict[FundInput, float]],
        base_date: date | None = None,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Applies defined shock scenarios to a weighted portfolio.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            shocks (dict): mapping of scenario name to fund shock returns.
            base_date (date | None): optional date for base prices (defaults to latest).
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe of shock scenario impacts for the portfolio.
        """
        if not shocks:
            raise ValueError("shocks must contain at least one scenario")
        weight_map = self._resolve_weights(weights, normalize_weights=normalize_weights)
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
            weighted_shock = 0.0
            for fund_name, weight in weight_map.items():
                weighted_shock += weight * resolved_shocks.get(fund_name, 0.0)
            base_value = 1.0
            shocked_value = base_value * (1 + weighted_shock)
            records.append(
                {
                    "scenario": scenario,
                    "base_date": base_date,
                    "base_value": base_value,
                    "shock_return": float(weighted_shock),
                    "shocked_value": float(shocked_value),
                }
            )
            for fund_name, weight in weight_map.items():
                shock = resolved_shocks.get(fund_name, 0.0)
                records.append(
                    {
                        "scenario": scenario,
                        "base_date": base_date,
                        "fund": fund_name,
                        "weight": float(weight),
                        "base_price": float(base_prices[fund_name]),
                        "shock_return": float(shock),
                        "weighted_shock": float(weight * shock),
                    }
                )
        return DataFrame.from_records(records)

    def get_portfolio_risk_return_summary(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
        mar: float = 0.0,
        confidence: float = 0.95,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Gets a risk/return summary for a weighted portfolio.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            trading_days (int): trading days per year used for annualization.
            mar (float): minimum acceptable return (annualized) for Sortino calculations.
            confidence (float): confidence level between 0 and 1 for VaR/expected shortfall.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with portfolio risk metrics.
        """
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_numeric(mar, "mar")
        self._validate_confidence(confidence, "confidence")
        portfolio_returns = self.get_portfolio_returns(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            normalize_weights=normalize_weights,
        )
        if portfolio_returns.empty:
            raise ValueError("no return data available for portfolio risk summary")
        returns = portfolio_returns["portfolio_return"]
        annualized_return = returns.mean() * trading_days
        annualized_volatility = returns.std() * (trading_days**0.5)
        sharpe_ratio = (
            annualized_return / annualized_volatility
            if annualized_volatility
            else float("nan")
        )
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        daily_mar = self._to_daily_rate(mar, trading_days)
        downside = (returns - daily_mar).where(returns < daily_mar, 0)
        downside_deviation = downside.pow(2).mean() ** 0.5 * (trading_days**0.5)
        sortino_ratio = (
            (annualized_return - mar) / downside_deviation
            if downside_deviation
            else float("nan")
        )
        gains = (returns - daily_mar).where(returns > daily_mar, 0).sum()
        losses = (daily_mar - returns).where(returns < daily_mar, 0).sum()
        omega_ratio = gains / losses if losses != 0 else float("nan")
        var_level = 1 - float(confidence)
        value_at_risk = returns.quantile(var_level)
        expected_shortfall = returns[returns <= value_at_risk].mean()
        portfolio_values = (1 + returns).cumprod()
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        cumulative_max = portfolio_values.cummax()
        drawdown = portfolio_values.div(cumulative_max).sub(1)
        trough_date = drawdown.idxmin()
        peak_date = portfolio_values.loc[:trough_date].idxmax()
        drawdown_duration_days = int((trough_date - peak_date).days)
        peak_value = portfolio_values.loc[peak_date]
        recovery_values = portfolio_values.loc[trough_date:]
        recovered = recovery_values[recovery_values >= peak_value]
        if recovered.empty:
            drawdown_recovery_days = float("nan")
        else:
            recovery_date = recovered.index[0]
            drawdown_recovery_days = int((recovery_date - trough_date).days)
        pain_index = drawdown.abs().mean()
        calmar_ratio = (
            annualized_return / abs(max_drawdown) if max_drawdown != 0 else float("nan")
        )
        ulcer_index = (
            portfolio_values.div(portfolio_values.cummax()).sub(1).pow(2).mean() ** 0.5
        )
        pain_ratio = annualized_return / pain_index if pain_index != 0 else float("nan")
        summary = DataFrame(
            {
                "annualized_return": [float(annualized_return)],
                "annualized_volatility": [float(annualized_volatility)],
                "sharpe_ratio": [float(sharpe_ratio)],
                "sortino_ratio": [float(sortino_ratio)],
                "skew": [float(skew)],
                "kurtosis": [float(kurtosis)],
                "max_drawdown": [float(max_drawdown)],
                "calmar_ratio": [float(calmar_ratio)],
                "ulcer_index": [float(ulcer_index)],
                "max_drawdown_duration_days": [drawdown_duration_days],
                "max_drawdown_recovery_days": [float(drawdown_recovery_days)],
                "pain_index": [float(pain_index)],
                "pain_ratio": [float(pain_ratio)],
                "omega_ratio": [float(omega_ratio)],
                "value_at_risk": [float(value_at_risk)],
                "expected_shortfall": [float(expected_shortfall)],
            },
            index=["portfolio"],
        )
        summary.index.name = "portfolio"
        return summary

    def get_portfolio_contribution_analysis(
        self,
        weights: Mapping[FundInput, float],
        start_date: date | None = None,
        end_date: date | None = None,
        trading_days: int = 252,
        normalize_weights: bool = True,
    ) -> DataFrame:
        """
        Calculates return and volatility contributions for each fund allocation.

        Args:
            weights (Mapping): mapping of fund enums or fund name strings to weights.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            trading_days (int): trading days per year used for annualization.
            normalize_weights (bool): normalize weights to sum to 1.

        Returns:
            DataFrame: dataframe with return and volatility contribution metrics.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        self._validate_positive_int(trading_days, "trading_days")

        weight_map = self._resolve_weights(weights, normalize_weights=normalize_weights)
        fund_list = list(weight_map.keys())
        returns = self.get_daily_returns()
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        returns = returns[fund_list].fillna(0.0)
        if returns.empty:
            raise ValueError(
                "no return data available for portfolio contribution analysis"
            )

        weight_series = Series(weight_map).reindex(fund_list)
        portfolio_returns = returns.mul(weight_series).sum(axis=1)
        annualized_return = portfolio_returns.mean() * trading_days
        fund_annualized_returns = returns.mean() * trading_days
        return_contribution = weight_series.mul(fund_annualized_returns)
        return_contribution_pct = (
            return_contribution.div(annualized_return)
            if annualized_return
            else Series(float("nan"), index=fund_list)
        )

        cov_matrix = returns.cov()
        portfolio_variance = float(
            np.dot(weight_series, np.dot(cov_matrix, weight_series))
        )
        portfolio_daily_volatility = portfolio_variance**0.5
        if portfolio_daily_volatility:
            marginal_contribution = cov_matrix.dot(weight_series)
            volatility_contribution_daily = weight_series.mul(
                marginal_contribution
            ).div(portfolio_daily_volatility)
        else:
            volatility_contribution_daily = Series(0.0, index=fund_list)
        annualization = trading_days**0.5
        portfolio_annualized_volatility = portfolio_daily_volatility * annualization
        volatility_contribution = volatility_contribution_daily.mul(annualization)
        volatility_contribution_pct = (
            volatility_contribution.div(portfolio_annualized_volatility)
            if portfolio_annualized_volatility
            else Series(float("nan"), index=fund_list)
        )

        return DataFrame(
            {
                "weight": weight_series,
                "annualized_return_contribution": return_contribution,
                "annualized_return_contribution_pct": return_contribution_pct,
                "annualized_volatility_contribution": volatility_contribution,
                "annualized_volatility_contribution_pct": volatility_contribution_pct,
            }
        )

    def optimize_portfolio(
        self,
        objective: str = "min_variance",
        funds: list[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        min_weights: Mapping[FundInput, float] | float | None = None,
        max_weights: Mapping[FundInput, float] | float | None = None,
        max_volatility: float | None = None,
        max_drawdown: float | None = None,
        confidence: float = 0.95,
        trading_days: int = 252,
        samples: int = 5000,
        random_state: int | None = None,
    ) -> DataFrame:
        """
        Optimizes a portfolio using mean-variance, risk parity, or CVaR objectives.

        Args:
            objective (str): one of ``min_variance``, ``risk_parity``, or ``cvar``.
            funds (list[FundInput] | None): optional list of funds to include.
            start_date (date | None): optional start date for filtering returns.
            end_date (date | None): optional end date for filtering returns.
            min_weights (Mapping | float | None): minimum weights per fund or scalar floor.
            max_weights (Mapping | float | None): maximum weights per fund or scalar cap.
            max_volatility (float | None): optional annualized volatility limit.
            max_drawdown (float | None): optional maximum drawdown limit (positive value).
            confidence (float): confidence level for CVaR calculations.
            trading_days (int): trading days per year used for annualization.
            samples (int): number of random weight samples to evaluate.
            random_state (int | None): seed for reproducible weight sampling.

        Returns:
            DataFrame: dataframe with optimized weights and portfolio metrics.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_positive_int(samples, "samples")
        if max_volatility is not None:
            self._validate_positive_float(max_volatility, "max_volatility")
        if max_drawdown is not None:
            self._validate_non_negative_float(max_drawdown, "max_drawdown")
        self._validate_confidence(confidence, "confidence")
        objective = objective.lower().strip()
        if objective not in {"min_variance", "risk_parity", "cvar"}:
            raise ValueError(
                "objective must be one of: min_variance, risk_parity, cvar"
            )
        fund_names = self._resolve_funds(funds)
        returns = self.get_daily_returns()
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        if returns.empty:
            raise ValueError("no return data available for portfolio optimization")
        returns = returns[fund_names].dropna(how="all")
        if returns.empty:
            raise ValueError("no return data available for requested funds")
        mean_returns = returns.mean() * trading_days
        covariance = returns.cov() * trading_days
        min_weights_arr, max_weights_arr = self._normalize_weight_bounds(
            fund_names, min_weights=min_weights, max_weights=max_weights
        )
        rng = np.random.default_rng(random_state)
        best_weights = None
        best_metric = None
        best_portfolio_returns = None
        for _ in range(samples):
            weights = self._sample_weight_vector(rng, min_weights_arr, max_weights_arr)
            portfolio_returns = returns.values @ weights
            annualized_volatility = float(
                np.sqrt(weights @ covariance.values @ weights)
            )
            if max_volatility is not None and annualized_volatility > max_volatility:
                continue
            if max_drawdown is not None:
                price_series = Series(
                    (1 + portfolio_returns).cumprod(), index=returns.index
                )
                drawdown = abs(self._calculate_max_drawdown(price_series))
                if drawdown > max_drawdown:
                    continue
            if objective == "min_variance":
                metric = annualized_volatility
            elif objective == "risk_parity":
                if annualized_volatility == 0:
                    continue
                marginal = covariance.values @ weights
                risk_contrib = weights * marginal / annualized_volatility
                target = annualized_volatility / len(weights)
                metric = float(((risk_contrib - target) ** 2).sum())
            else:
                losses = -portfolio_returns
                var_threshold = np.quantile(losses, confidence)
                tail_losses = losses[losses >= var_threshold]
                if tail_losses.size == 0:
                    continue
                metric = float(tail_losses.mean())
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_weights = weights
                best_portfolio_returns = portfolio_returns
        if best_weights is None or best_portfolio_returns is None:
            raise ValueError("no feasible portfolio found for the provided constraints")
        best_portfolio_returns_arr = np.asarray(best_portfolio_returns, dtype=float)
        price_series = Series(
            (1 + best_portfolio_returns_arr).cumprod(), index=returns.index
        )
        max_drawdown_value = self._calculate_max_drawdown(price_series)
        losses = -best_portfolio_returns_arr
        var_threshold = np.quantile(losses, confidence)
        tail_losses = losses[losses >= var_threshold]
        cvar_loss = float(tail_losses.mean()) if tail_losses.size else float("nan")
        summary = {
            "objective": objective,
            "annualized_return": float(np.dot(best_weights, mean_returns.values)),
            "annualized_volatility": float(
                np.sqrt(best_weights @ covariance.values @ best_weights)
            ),
            "max_drawdown": float(max_drawdown_value),
            "cvar_loss": cvar_loss,
            "objective_value": float(best_metric),
        }
        for fund_name, weight in zip(fund_names, best_weights, strict=False):
            summary[fund_name] = float(weight)
        result = DataFrame([summary], index=["portfolio"])
        result.index.name = "portfolio"
        return result

    def get_efficient_frontier(
        self,
        funds: list[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        points: int = 20,
        min_weights: Mapping[FundInput, float] | float | None = None,
        max_weights: Mapping[FundInput, float] | float | None = None,
        max_volatility: float | None = None,
        max_drawdown: float | None = None,
        trading_days: int = 252,
        samples_per_point: int = 2000,
        random_state: int | None = None,
    ) -> DataFrame:
        """
        Generates a mean-variance efficient frontier subject to constraints.

        Args:
            funds (list[FundInput] | None): optional list of funds to include.
            start_date (date | None): optional start date for filtering returns.
            end_date (date | None): optional end date for filtering returns.
            points (int): number of frontier points to compute.
            min_weights (Mapping | float | None): minimum weights per fund or scalar floor.
            max_weights (Mapping | float | None): maximum weights per fund or scalar cap.
            max_volatility (float | None): optional annualized volatility limit.
            max_drawdown (float | None): optional maximum drawdown limit (positive value).
            trading_days (int): trading days per year used for annualization.
            samples_per_point (int): number of random weight samples per frontier point.
            random_state (int | None): seed for reproducible weight sampling.

        Returns:
            DataFrame: dataframe with frontier metrics and weights.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        self._validate_positive_int(points, "points")
        self._validate_positive_int(trading_days, "trading_days")
        self._validate_positive_int(samples_per_point, "samples_per_point")
        if max_volatility is not None:
            self._validate_positive_float(max_volatility, "max_volatility")
        if max_drawdown is not None:
            self._validate_non_negative_float(max_drawdown, "max_drawdown")
        fund_names = self._resolve_funds(funds)
        returns = self.get_daily_returns()
        if start_date is not None and end_date is not None:
            filtered = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = filtered.set_index("Date")
        if returns.empty:
            raise ValueError("no return data available for efficient frontier")
        returns = returns[fund_names].dropna(how="all")
        if returns.empty:
            raise ValueError("no return data available for requested funds")
        mean_returns = returns.mean() * trading_days
        covariance = returns.cov() * trading_days
        min_weights_arr, max_weights_arr = self._normalize_weight_bounds(
            fund_names, min_weights=min_weights, max_weights=max_weights
        )
        rng = np.random.default_rng(random_state)
        min_return = float(mean_returns.min())
        max_return = float(mean_returns.max())
        if min_return == max_return:
            target_returns = np.repeat(min_return, points)
        else:
            target_returns = np.linspace(min_return, max_return, points)
        frontier_rows: list[dict] = []
        for target_return in target_returns:
            best_metric = None
            best_weights = None
            best_volatility = None
            best_drawdown = None
            for _ in range(samples_per_point):
                weights = self._sample_weight_vector(
                    rng, min_weights_arr, max_weights_arr
                )
                expected_return = float(np.dot(weights, mean_returns.values))
                if expected_return < target_return:
                    continue
                volatility = float(np.sqrt(weights @ covariance.values @ weights))
                if max_volatility is not None and volatility > max_volatility:
                    continue
                if max_drawdown is not None:
                    portfolio_returns = returns.values @ weights
                    price_series = Series(
                        (1 + portfolio_returns).cumprod(), index=returns.index
                    )
                    drawdown = abs(self._calculate_max_drawdown(price_series))
                    if drawdown > max_drawdown:
                        continue
                else:
                    drawdown = float("nan")
                if best_metric is None or volatility < best_metric:
                    best_metric = volatility
                    best_weights = weights
                    best_volatility = volatility
                    best_drawdown = drawdown
            if best_weights is None:
                row = {
                    "target_return": float(target_return),
                    "expected_return": float("nan"),
                    "annualized_volatility": float("nan"),
                    "max_drawdown": float("nan"),
                }
                for fund_name in fund_names:
                    row[fund_name] = float("nan")
            else:
                row = {
                    "target_return": float(target_return),
                    "expected_return": float(np.dot(best_weights, mean_returns.values)),
                    "annualized_volatility": float(best_volatility),
                    "max_drawdown": float(best_drawdown),
                }
                for fund_name, weight in zip(fund_names, best_weights, strict=False):
                    row[fund_name] = float(weight)
            frontier_rows.append(row)
        return DataFrame(frontier_rows)

    def _normalize_weight_bounds(
        self,
        fund_names: list[str],
        min_weights: Mapping[FundInput, float] | float | None,
        max_weights: Mapping[FundInput, float] | float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if min_weights is None:
            min_weights_arr = np.zeros(len(fund_names))
        elif isinstance(min_weights, (int, float)):
            self._validate_non_negative_float(min_weights, "min_weights")
            min_weights_arr = np.full(len(fund_names), float(min_weights))
        else:
            if not isinstance(min_weights, Mapping):
                raise ValueError("min_weights must be a mapping, scalar, or None")
            min_weights_arr = np.zeros(len(fund_names))
            resolved = {
                self._resolve_fund(key): value for key, value in min_weights.items()
            }
            for idx, fund in enumerate(fund_names):
                if fund in resolved:
                    self._validate_non_negative_float(
                        resolved[fund], f"{fund} min_weight"
                    )
                    min_weights_arr[idx] = float(resolved[fund])
        if max_weights is None:
            max_weights_arr = np.ones(len(fund_names))
        elif isinstance(max_weights, (int, float)):
            self._validate_non_negative_float(max_weights, "max_weights")
            max_weights_arr = np.full(len(fund_names), float(max_weights))
        else:
            if not isinstance(max_weights, Mapping):
                raise ValueError("max_weights must be a mapping, scalar, or None")
            max_weights_arr = np.ones(len(fund_names))
            resolved = {
                self._resolve_fund(key): value for key, value in max_weights.items()
            }
            for idx, fund in enumerate(fund_names):
                if fund in resolved:
                    self._validate_non_negative_float(
                        resolved[fund], f"{fund} max_weight"
                    )
                    max_weights_arr[idx] = float(resolved[fund])
        if np.any(min_weights_arr > max_weights_arr):
            raise ValueError("min_weights must be less than or equal to max_weights")
        min_sum = float(min_weights_arr.sum())
        max_sum = float(max_weights_arr.sum())
        if min_sum > 1 + 1e-9:
            raise ValueError("min_weights sum to more than 1")
        if max_sum < 1 - 1e-9:
            raise ValueError("max_weights sum to less than 1")
        return min_weights_arr, max_weights_arr

    @staticmethod
    def _sample_weight_vector(
        rng: np.random.Generator,
        min_weights: np.ndarray,
        max_weights: np.ndarray,
        max_attempts: int = 1000,
    ) -> np.ndarray:
        n = len(min_weights)
        min_sum = float(min_weights.sum())
        remaining = 1.0 - min_sum
        if remaining < 0:
            raise ValueError("min_weights sum to more than 1")
        if remaining == 0:
            return min_weights.copy()
        capacity = max_weights - min_weights
        capacity_sum = float(capacity.sum())
        if capacity_sum < remaining - 1e-9:
            raise ValueError("max_weights sum to less than 1")
        for _ in range(max_attempts):
            proportions = rng.dirichlet(np.ones(n))
            extra = proportions * remaining
            if np.all(extra <= capacity + 1e-12):
                return min_weights + extra
        extra = np.zeros(n)
        remaining_allocation = remaining
        for idx in np.argsort(-capacity):
            if remaining_allocation <= 0:
                break
            add = min(capacity[idx], remaining_allocation)
            extra[idx] = add
            remaining_allocation -= add
        return min_weights + extra

    def get_optimal_interfund_transfer_plan(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        starting_fund: FundInput | None = None,
        max_non_g_transfers: int = 2,
        g_fund: FundInput = TspIndividualFund.G_FUND,
    ) -> DataFrame:
        """
        Builds an optimal interfund transfer plan using historical daily returns.

        The plan maximizes cumulative returns with a constraint on the number of
        transfers into non-G funds per month. Moves into the G fund do not count
        toward the monthly transfer limit.

        Args:
            start_date (date | None): optional start date for the analysis range.
            end_date (date | None): optional end date for the analysis range.
            starting_fund (FundInput | None): fund held at the start of the first month.
            max_non_g_transfers (int): maximum number of transfers into non-G funds per month.
            g_fund (FundInput): fund treated as the unlimited transfer destination.

        Returns:
            DataFrame: transfer schedule with dates, fund changes, and cumulative returns.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        if isinstance(max_non_g_transfers, bool) or not isinstance(
            max_non_g_transfers, int
        ):
            raise ValueError("max_non_g_transfers must be a non-negative integer")
        if max_non_g_transfers < 0:
            raise ValueError("max_non_g_transfers must be a non-negative integer")

        returns = self.get_daily_returns()
        if start_date is not None and end_date is not None:
            returns = self._filter_by_date_range(
                start_date, end_date, dataframe=returns.reset_index()
            )
            returns = returns.set_index("Date")
        returns = returns.dropna(how="all")
        if returns.empty:
            raise ValueError("no return data available for interfund transfer plan")

        g_fund_name = self._ensure_fund_available(g_fund)
        if starting_fund is None:
            starting_fund_name = (
                g_fund_name if g_fund_name in returns.columns else returns.columns[0]
            )
        else:
            starting_fund_name = self._ensure_fund_available(starting_fund)
        if starting_fund_name not in returns.columns:
            raise ValueError(
                f"starting fund not available in return data: {starting_fund_name}"
            )

        transfer_records: list[dict] = []
        current_fund = starting_fund_name
        cumulative_value = 1.0

        grouped = returns.groupby([returns.index.year, returns.index.month])
        for (year, month), month_returns in grouped:
            month_returns = month_returns.copy()
            dates = list(month_returns.index)
            if not dates:
                continue
            month_key = f"{year:04d}-{month:02d}"
            daily_rows = [row.dropna() for _, row in month_returns.iterrows()]
            if any(row.empty for row in daily_rows):
                raise ValueError(f"no return data available for {month_key}")

            dp: dict[tuple[str, int], float] = {}
            backpointers: list[dict[tuple[str, int], tuple[str, int] | None]] = []

            first_row = daily_rows[0]
            first_back: dict[tuple[str, int], tuple[str, int] | None] = {}
            for fund, daily_return in first_row.items():
                transfers_used = 0
                if fund != current_fund and fund != g_fund_name:
                    transfers_used = 1
                if transfers_used <= max_non_g_transfers:
                    dp[(fund, transfers_used)] = 1 + daily_return
                    first_back[(fund, transfers_used)] = None
            if not dp:
                raise ValueError(f"no feasible starting allocation for {month_key}")
            backpointers.append(first_back)

            for row in daily_rows[1:]:
                next_dp: dict[tuple[str, int], float] = {}
                next_back: dict[tuple[str, int], tuple[str, int] | None] = {}
                for (prev_fund, transfers_used), prev_value in dp.items():
                    for fund, daily_return in row.items():
                        next_transfers = transfers_used
                        if fund != prev_fund and fund != g_fund_name:
                            next_transfers += 1
                        if next_transfers > max_non_g_transfers:
                            continue
                        value = prev_value * (1 + daily_return)
                        key = (fund, next_transfers)
                        if value > next_dp.get(key, float("-inf")):
                            next_dp[key] = value
                            next_back[key] = (prev_fund, transfers_used)
                if not next_dp:
                    raise ValueError(f"no feasible allocation path for {month_key}")
                dp = next_dp
                backpointers.append(next_back)

            best_state = max(dp, key=dp.get)
            funds_by_day: list[str] = [current_fund] * len(dates)
            state: tuple[str, int] | None = best_state
            for idx in range(len(dates) - 1, -1, -1):
                if state is None:
                    break
                fund, transfers_used = state
                funds_by_day[idx] = fund
                state = backpointers[idx].get((fund, transfers_used))

            monthly_transfer_count = 0
            for idx, day in enumerate(dates):
                fund = funds_by_day[idx]
                if idx == 0:
                    prev_fund = current_fund
                else:
                    prev_fund = funds_by_day[idx - 1]
                daily_return = month_returns.loc[day, fund]
                cumulative_value *= 1 + daily_return
                if fund != prev_fund:
                    if fund != g_fund_name:
                        monthly_transfer_count += 1
                    transfer_records.append(
                        {
                            "date": day.date(),
                            "month": month_key,
                            "from_fund": prev_fund,
                            "to_fund": fund,
                            "transfer_count": monthly_transfer_count,
                            "cumulative_return": cumulative_value - 1,
                        }
                    )

            current_fund = funds_by_day[-1]

        if not transfer_records:
            return DataFrame(
                columns=["date", "month", "from_fund", "to_fund", "transfer_count"]
            )
        return DataFrame(transfer_records)
