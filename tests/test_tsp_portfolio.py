import pandas as pd
import pytest
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics
from tests.helpers import build_minimal_price_dataframe


def test_portfolio_returns(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    portfolio_returns = tsp_price.get_portfolio_returns(weights=weights)
    assert portfolio_returns.columns.tolist() == ["portfolio_return"]
    assert len(portfolio_returns) == len(tsp_price.dataframe) - 1
    latest_return = portfolio_returns.iloc[-1]["portfolio_return"]
    assert latest_return == pytest.approx(
        0.6 * tsp_price.get_daily_returns(fund=TspIndividualFund.G_FUND).iloc[-1, 0]
        + 0.4 * tsp_price.get_daily_returns(fund=TspIndividualFund.C_FUND).iloc[-1, 0]
    )


def test_portfolio_returns_and_cumulative(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    returns = tsp_price.get_portfolio_returns(weights=weights)
    expected_returns = tsp_price.get_daily_returns()[[fund.value for fund in weights]]
    expected_portfolio = expected_returns.mul(
        pd.Series({fund.value: weight for fund, weight in weights.items()})
    ).sum(axis=1)
    assert_series_equal(
        returns["portfolio_return"],
        expected_portfolio.rename("portfolio_return"),
    )

    cumulative = tsp_price.get_portfolio_cumulative_returns(weights=weights)
    expected_cumulative = (1 + expected_portfolio).cumprod() - 1
    assert_series_equal(
        cumulative["portfolio_cumulative_return"],
        expected_cumulative.rename("portfolio_cumulative_return"),
    )


def test_portfolio_value_history_and_summary(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.5,
        TspIndividualFund.C_FUND: 0.5,
    }
    values = tsp_price.get_portfolio_value_history(
        weights=weights, initial_value=1000.0
    )
    assert values.columns.tolist() == ["portfolio_value"]

    summary = tsp_price.get_portfolio_performance_summary(
        weights=weights, trading_days=252
    )
    assert summary.index.tolist() == ["portfolio"]
    assert "total_return" in summary.columns
    assert "annualized_return" in summary.columns


def test_portfolio_drawdown_series(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    drawdown = tsp_price.get_portfolio_drawdown_series(weights=weights)
    returns = tsp_price.get_portfolio_returns(weights=weights)["portfolio_return"]
    portfolio_values = (1 + returns).cumprod()
    expected = (
        portfolio_values.div(portfolio_values.cummax())
        .sub(1)
        .to_frame(name="portfolio_drawdown")
    )
    assert_frame_equal(drawdown, expected)


def test_portfolio_risk_return_summary(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    trading_days = 252
    mar = 0.02
    confidence = 0.8
    summary = tsp_price.get_portfolio_risk_return_summary(
        weights=weights,
        trading_days=trading_days,
        mar=mar,
        confidence=confidence,
    )
    returns = tsp_price.get_portfolio_returns(weights=weights)["portfolio_return"]
    annualized_return = returns.mean() * trading_days
    annualized_volatility = returns.std() * (trading_days**0.5)
    sharpe_ratio = annualized_return / annualized_volatility
    daily_mar = (1 + mar) ** (1 / trading_days) - 1
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
    value_at_risk = returns.quantile(1 - confidence)
    expected_shortfall = returns[returns <= value_at_risk].mean()
    portfolio_values = (1 + returns).cumprod()
    max_drawdown = tsp_price._calculate_max_drawdown(portfolio_values)
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

    assert summary.index.tolist() == ["portfolio"]
    assert summary.loc["portfolio", "annualized_return"] == pytest.approx(
        annualized_return
    )
    assert summary.loc["portfolio", "annualized_volatility"] == pytest.approx(
        annualized_volatility
    )
    assert summary.loc["portfolio", "sharpe_ratio"] == pytest.approx(sharpe_ratio)
    if sortino_ratio != sortino_ratio:
        assert (
            summary.loc["portfolio", "sortino_ratio"]
            != summary.loc["portfolio", "sortino_ratio"]
        )
    else:
        assert summary.loc["portfolio", "sortino_ratio"] == pytest.approx(sortino_ratio)
    assert summary.loc["portfolio", "max_drawdown"] == pytest.approx(max_drawdown)
    if calmar_ratio != calmar_ratio:
        assert (
            summary.loc["portfolio", "calmar_ratio"]
            != summary.loc["portfolio", "calmar_ratio"]
        )
    else:
        assert summary.loc["portfolio", "calmar_ratio"] == pytest.approx(calmar_ratio)
    assert summary.loc["portfolio", "ulcer_index"] == pytest.approx(ulcer_index)
    assert (
        summary.loc["portfolio", "max_drawdown_duration_days"] == drawdown_duration_days
    )
    if drawdown_recovery_days != drawdown_recovery_days:
        assert (
            summary.loc["portfolio", "max_drawdown_recovery_days"]
            != summary.loc["portfolio", "max_drawdown_recovery_days"]
        )
    else:
        assert (
            summary.loc["portfolio", "max_drawdown_recovery_days"]
            == drawdown_recovery_days
        )
    assert summary.loc["portfolio", "pain_index"] == pytest.approx(pain_index)
    if pain_ratio != pain_ratio:
        assert (
            summary.loc["portfolio", "pain_ratio"]
            != summary.loc["portfolio", "pain_ratio"]
        )
    else:
        assert summary.loc["portfolio", "pain_ratio"] == pytest.approx(pain_ratio)
    if omega_ratio != omega_ratio:
        assert (
            summary.loc["portfolio", "omega_ratio"]
            != summary.loc["portfolio", "omega_ratio"]
        )
    else:
        assert summary.loc["portfolio", "omega_ratio"] == pytest.approx(omega_ratio)
    assert summary.loc["portfolio", "value_at_risk"] == pytest.approx(value_at_risk)
    assert summary.loc["portfolio", "expected_shortfall"] == pytest.approx(
        expected_shortfall
    )

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_portfolio_risk_return_summary(weights=weights, trading_days=0)

    with pytest.raises(ValueError, match="mar must be a numeric value"):
        tsp_price.get_portfolio_risk_return_summary(weights=weights, mar="0.01")

    with pytest.raises(ValueError, match="confidence must be a float between 0 and 1"):
        tsp_price.get_portfolio_risk_return_summary(weights=weights, confidence=1.5)


def test_portfolio_weights_validation(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="weights must be a non-empty mapping"):
        tsp_price.get_portfolio_returns(weights={})

    with pytest.raises(ValueError, match="unknown fund"):
        tsp_price.get_portfolio_returns(weights={"Unknown Fund": 0.5})

    with pytest.raises(
        ValueError, match="weights must map fund enums or fund name strings"
    ):
        tsp_price.get_portfolio_returns(weights={123: 0.5})

    with pytest.raises(
        ValueError, match="weights must include at least one positive value"
    ):
        tsp_price.get_portfolio_returns(weights={TspIndividualFund.G_FUND: 0.0})

    with pytest.raises(ValueError, match="weights must sum to 1"):
        tsp_price.get_portfolio_returns(
            weights={TspIndividualFund.G_FUND: 0.8, TspIndividualFund.C_FUND: 0.8},
            normalize_weights=False,
        )

    with pytest.raises(ValueError, match="initial_value must be a positive value"):
        tsp_price.get_portfolio_value_history(
            weights={TspIndividualFund.G_FUND: 1.0},
            initial_value=0,
        )


def test_portfolio_weights_missing_fund() -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="funds not available in data"):
        tsp_price.get_portfolio_returns(weights={TspLifecycleFund.L_2030: 0.5})


def test_portfolio_bootstrap_simulation_is_deterministic(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    simulations = 3
    trading_days = 3
    initial_value = 1000.0
    random_state = 42

    result = tsp_price.get_portfolio_bootstrap_simulation(
        weights=weights,
        years=1,
        simulations=simulations,
        trading_days=trading_days,
        initial_value=initial_value,
        random_state=random_state,
    )

    portfolio_returns = tsp_price.get_portfolio_returns(weights=weights)[
        "portfolio_return"
    ].dropna()
    rng = np.random.default_rng(random_state)
    bootstrapped = rng.choice(
        portfolio_returns.values,
        size=(simulations, trading_days),
        replace=True,
    )
    expected_values = initial_value * (1 + bootstrapped).prod(axis=1)
    expected = pd.DataFrame(
        {"ending_value": expected_values}, index=range(1, simulations + 1)
    )
    expected.index.name = "simulation"
    assert_frame_equal(result, expected)


def test_portfolio_bootstrap_simulation_requires_returns() -> None:
    tsp_price = TspAnalytics()
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02"]),
            TspIndividualFund.G_FUND.value: [100.0],
            TspIndividualFund.C_FUND.value: [200.0],
        }
    )
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    with pytest.raises(ValueError, match="no return data available for portfolio"):
        tsp_price.get_portfolio_bootstrap_simulation(
            weights={TspIndividualFund.G_FUND: 1.0},
            years=1,
            random_state=1,
        )


def test_portfolio_retirement_sequence_requires_history(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    with pytest.raises(ValueError, match="not enough return history"):
        tsp_price.get_portfolio_retirement_sequence_analysis(
            weights=weights,
            initial_value=1000.0,
            annual_withdrawal=100.0,
            years=1,
        )


def test_portfolio_rebalance_dates_validation(tsp_price: TspAnalytics) -> None:
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")

    rebalance_dates = tsp_price._get_rebalance_dates(dates, "monthly")
    assert rebalance_dates[-1] == pd.Timestamp("2024-01-31")

    with pytest.raises(ValueError, match='frequency must be "monthly" or "quarterly"'):
        tsp_price._get_rebalance_dates(dates, "weekly")


def test_portfolio_returns_date_range_requires_pair(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    start_date = tsp_price.latest
    end_date = tsp_price.latest

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_portfolio_returns(weights=weights, start_date=start_date)

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_portfolio_returns(weights=weights, end_date=end_date)


def test_portfolio_returns_normalizes_weights(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 2.0,
        TspIndividualFund.C_FUND: 1.0,
    }
    returns = tsp_price.get_portfolio_returns(weights=weights)

    normalized = {fund.value: weight for fund, weight in weights.items()}
    total = sum(normalized.values())
    expected = (
        tsp_price.get_daily_returns()[list(normalized.keys())]
        .mul(pd.Series({key: value / total for key, value in normalized.items()}))
        .sum(axis=1)
    )

    assert_series_equal(
        returns["portfolio_return"], expected.rename("portfolio_return")
    )


def test_portfolio_returns_filters_date_range(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }
    returns = tsp_price.get_daily_returns()
    start_date = returns.index[1].date()
    end_date = returns.index[-1].date()

    filtered = tsp_price.get_portfolio_returns(
        weights=weights,
        start_date=start_date,
        end_date=end_date,
    )

    expected_returns = returns.loc[start_date:end_date]
    expected = (
        expected_returns[[fund.value for fund in weights]]
        .mul(pd.Series({fund.value: weight for fund, weight in weights.items()}))
        .sum(axis=1)
    )
    assert_series_equal(
        filtered["portfolio_return"], expected.rename("portfolio_return")
    )


def test_portfolio_worst_drawdown_windows_validates_inputs(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    with pytest.raises(ValueError, match="window must be a positive integer"):
        tsp_price.get_portfolio_worst_drawdown_windows(weights=weights, window=0)

    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        tsp_price.get_portfolio_worst_drawdown_windows(weights=weights, top_n=0)


def test_portfolio_bootstrap_simulation_validates_inputs(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    with pytest.raises(ValueError, match="years must be a positive integer"):
        tsp_price.get_portfolio_bootstrap_simulation(weights=weights, years=0)

    with pytest.raises(ValueError, match="simulations must be a positive integer"):
        tsp_price.get_portfolio_bootstrap_simulation(
            weights=weights, years=1, simulations=0
        )

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        tsp_price.get_portfolio_bootstrap_simulation(
            weights=weights, years=1, trading_days=0
        )

    with pytest.raises(ValueError, match="initial_value must be a positive value"):
        tsp_price.get_portfolio_bootstrap_simulation(
            weights=weights, years=1, initial_value=0
        )


def test_portfolio_retirement_sequence_validates_inputs(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    with pytest.raises(ValueError, match="years must be a positive integer"):
        tsp_price.get_portfolio_retirement_sequence_analysis(
            weights=weights,
            initial_value=1000.0,
            annual_withdrawal=100.0,
            years=0,
        )

    with pytest.raises(ValueError, match="annual_withdrawal must be a positive value"):
        tsp_price.get_portfolio_retirement_sequence_analysis(
            weights=weights,
            initial_value=1000.0,
            annual_withdrawal=0.0,
            years=1,
        )

    with pytest.raises(ValueError, match="initial_value must be a positive value"):
        tsp_price.get_portfolio_retirement_sequence_analysis(
            weights=weights,
            initial_value=0.0,
            annual_withdrawal=100.0,
            years=1,
        )


def test_portfolio_contribution_analysis(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    analysis = tsp_price.get_portfolio_contribution_analysis(
        weights=weights, trading_days=252
    )
    assert set(analysis.columns) == {
        "weight",
        "annualized_return_contribution",
        "annualized_return_contribution_pct",
        "annualized_volatility_contribution",
        "annualized_volatility_contribution_pct",
    }
    assert analysis.loc[TspIndividualFund.G_FUND.value, "weight"] == pytest.approx(0.6)
    assert analysis.loc[TspIndividualFund.C_FUND.value, "weight"] == pytest.approx(0.4)
    assert analysis["annualized_return_contribution_pct"].sum() == pytest.approx(1.0)

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_portfolio_contribution_analysis(
            weights=weights,
            start_date=tsp_price.latest,
        )


def test_portfolio_rebalancing_backtest_outputs(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    result = tsp_price.get_portfolio_rebalancing_backtest(
        weights=weights,
        trading_days=252,
        trend_window=2,
        strategies=["monthly"],
        include_buy_and_hold=True,
    )

    assert set(result.keys()) == {
        "summary",
        "impact",
        "values",
        "drawdowns",
        "turnover",
    }
    summary = result["summary"]
    assert {"buy_and_hold", "monthly"}.issubset(summary.index)

    turnover = result["turnover"]
    assert turnover["buy_and_hold"].sum() == pytest.approx(0.0)

    impact = result["impact"]
    assert np.allclose(impact.loc["buy_and_hold"].fillna(0.0).values, 0.0)


def test_portfolio_rebalancing_backtest_threshold_strategy(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    result = tsp_price.get_portfolio_rebalancing_backtest(
        weights=weights,
        trading_days=252,
        rebalance_threshold=0.0,
        strategies=["threshold"],
        include_buy_and_hold=False,
    )

    assert result["summary"].index.tolist() == ["threshold"]
    assert result["values"].columns.tolist() == ["threshold"]


def test_portfolio_rebalancing_backtest_trend_strategy(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.C_FUND: 0.9,
        TspIndividualFund.G_FUND: 0.1,
    }

    result = tsp_price.get_portfolio_rebalancing_backtest(
        weights=weights,
        trading_days=252,
        trend_window=1,
        strategies=["trend"],
        include_buy_and_hold=False,
    )

    summary = result["summary"]
    assert summary.index.tolist() == ["trend"]
    assert result["values"].columns.tolist() == ["trend"]


def test_portfolio_rebalancing_backtest_validates_inputs(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    with pytest.raises(
        ValueError, match="start_date and end_date must be provided together"
    ):
        tsp_price.get_portfolio_rebalancing_backtest(
            weights=weights,
            start_date=tsp_price.latest,
        )

    with pytest.raises(
        ValueError, match="rebalance_threshold must be a non-negative value"
    ):
        tsp_price.get_portfolio_rebalancing_backtest(
            weights=weights,
            rebalance_threshold=-0.1,
        )

    with pytest.raises(ValueError, match="rebalance_threshold must be between 0 and 1"):
        tsp_price.get_portfolio_rebalancing_backtest(
            weights=weights,
            rebalance_threshold=1.5,
        )

    with pytest.raises(ValueError, match="strategies must include"):
        tsp_price.get_portfolio_rebalancing_backtest(
            weights=weights,
            strategies=["weekly"],
        )

    with pytest.raises(ValueError, match='frequency must be "monthly" or "quarterly"'):
        tsp_price.get_portfolio_rebalancing_backtest(
            weights=weights,
            strategies=["trend"],
            include_buy_and_hold=False,
            trend_frequency="weekly",
        )


def test_optimize_portfolio_min_variance(tsp_price: TspAnalytics) -> None:
    result = tsp_price.optimize_portfolio(
        objective="min_variance",
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        samples=200,
        random_state=7,
    )

    assert result.index.tolist() == ["portfolio"]
    assert result.loc["portfolio", "objective"] == "min_variance"
    weights = result.loc[
        "portfolio", [TspIndividualFund.G_FUND.value, TspIndividualFund.C_FUND.value]
    ]
    assert float(weights.sum()) == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert (weights <= 1).all()

    with pytest.raises(ValueError, match="objective must be one of"):
        tsp_price.optimize_portfolio(objective="maximize_returns")

    with pytest.raises(
        ValueError, match="min_weights must be less than or equal to max_weights"
    ):
        tsp_price.optimize_portfolio(
            funds=[TspIndividualFund.G_FUND],
            min_weights={TspIndividualFund.G_FUND: 0.8},
            max_weights={TspIndividualFund.G_FUND: 0.5},
            samples=10,
        )

    with pytest.raises(ValueError, match="max_volatility must be a positive value"):
        tsp_price.optimize_portfolio(
            max_volatility=0,
        )

    with pytest.raises(ValueError, match="samples must be a positive integer"):
        tsp_price.optimize_portfolio(samples=0)

    with pytest.raises(ValueError, match="max_drawdown must be a non-negative value"):
        tsp_price.optimize_portfolio(max_drawdown=-0.1)


def test_portfolio_shock_scenario_analysis_validates_inputs(
    tsp_price: TspAnalytics,
) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    with pytest.raises(ValueError, match="shocks must contain at least one scenario"):
        tsp_price.get_portfolio_shock_scenario_analysis(weights=weights, shocks={})

    with pytest.raises(ValueError, match="base_date not found in price data"):
        tsp_price.get_portfolio_shock_scenario_analysis(
            weights=weights,
            shocks={"shock": {TspIndividualFund.G_FUND: -0.1}},
            base_date=tsp_price.latest.replace(year=tsp_price.latest.year - 1),
        )
