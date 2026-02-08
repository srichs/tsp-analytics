from datetime import date

from tsp import TspIndividualFund


def test_regime_detection_volatility(tsp_price):
    summary = tsp_price.get_regime_detection(
        fund=TspIndividualFund.G_FUND, window=2, num_regimes=2, method="quantile"
    )
    assert not summary.empty
    assert set(
        ["rolling_volatility", "rolling_trend", "regime", "regime_label"]
    ).issubset(summary.columns)


def test_historical_stress_test_summary(tsp_price):
    windows = {"sample": (date(2024, 1, 1), date(2024, 1, 4))}
    summary = tsp_price.get_historical_stress_test_summary(windows=windows)
    assert not summary.empty
    assert "window" in summary.columns


def test_worst_drawdown_windows(tsp_price):
    windows = tsp_price.get_worst_drawdown_windows(
        fund=TspIndividualFund.G_FUND, window=2, top_n=1
    )
    assert len(windows) == 1
    assert "total_return" in windows.columns


def test_shock_scenario_analysis(tsp_price):
    shocks = {"shock": {TspIndividualFund.G_FUND: -0.1, TspIndividualFund.F_FUND: -0.2}}
    analysis = tsp_price.get_shock_scenario_analysis(
        shocks=shocks, base_date=date(2024, 1, 4)
    )
    assert not analysis.empty
    assert "shocked_price" in analysis.columns


def test_portfolio_stress_and_scenarios(tsp_price):
    weights = {TspIndividualFund.G_FUND: 0.6, TspIndividualFund.F_FUND: 0.4}
    windows = {"sample": (date(2024, 1, 1), date(2024, 1, 4))}
    stress = tsp_price.get_portfolio_stress_test_summary(
        weights=weights, windows=windows
    )
    assert not stress.empty
    worst = tsp_price.get_portfolio_worst_drawdown_windows(
        weights=weights, window=2, top_n=1
    )
    assert len(worst) == 1
    shocks = {"shock": {TspIndividualFund.G_FUND: -0.1, TspIndividualFund.F_FUND: -0.2}}
    scenario = tsp_price.get_portfolio_shock_scenario_analysis(
        weights=weights, shocks=shocks, base_date=date(2024, 1, 4)
    )
    assert not scenario.empty
    assert "scenario" in scenario.columns
