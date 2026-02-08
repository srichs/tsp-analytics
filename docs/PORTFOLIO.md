# Portfolio Analytics

This guide shows how to build weighted portfolio analytics from the underlying fund data.
Portfolio helpers accept a mapping of funds to weights (fund enums or fund name strings),
normalize weights by default, and return a Pandas `DataFrame` for downstream analysis or
visualization.

## Portfolio Returns

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

weights = {
    TspIndividualFund.C_FUND: 0.6,
    TspIndividualFund.S_FUND: 0.4,
}

portfolio_returns = prices.get_portfolio_returns(weights=weights)
print(portfolio_returns.tail())
```

By default, weights are normalized to sum to 1. If you pass weights that already sum to 1 and
prefer validation over normalization, set `normalize_weights=False`:

```python
portfolio_returns = prices.get_portfolio_returns(
    weights=weights,
    normalize_weights=False,
)
```

## Portfolio Cumulative Returns

```python
portfolio_cumulative = prices.get_portfolio_cumulative_returns(weights=weights)
```

## Portfolio Value History

Use `get_portfolio_value_history` to track a notional portfolio value:

```python
portfolio_value = prices.get_portfolio_value_history(
    weights=weights,
    initial_value=10_000,
)
```

## Portfolio Performance Summary

```python
summary = prices.get_portfolio_performance_summary(weights=weights)
print(summary)
```

The performance summary includes total return, annualized return, volatility, Sharpe ratio, and
maximum drawdown metrics, computed from the weighted daily returns.

## Portfolio Risk/Return Summary

Use `get_portfolio_risk_return_summary` to capture a broader set of risk metrics including
Calmar ratio, ulcer index, max drawdown duration, max drawdown recovery time, pain index/ratio,
Omega ratio, skew/kurtosis, value at risk (VaR), and expected shortfall:

```python
portfolio_risk = prices.get_portfolio_risk_return_summary(
    weights=weights,
    trading_days=252,
    mar=0.0,
    confidence=0.95,
)
```

## Portfolio Drawdowns

```python
drawdown = prices.get_portfolio_drawdown_series(weights=weights)
```

## Portfolio Visualization

```python
prices.show_portfolio_value_chart(
    weights=weights,
    initial_value=10_000,
)

prices.show_portfolio_drawdown_chart(weights=weights)
```

## Date Ranges

All portfolio helpers accept `start_date` and `end_date` parameters when you want to focus on a
specific period:

```python
from datetime import date

portfolio_returns = prices.get_portfolio_returns(
    weights=weights,
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
```

## Common Validation Errors

- If the weights mapping is empty or all weights are zero, the library raises a `ValueError`.
- If a fund is missing from the cached CSV and has a positive weight, the library raises a
  `ValueError`.
- When `normalize_weights=False`, the weights must sum to 1 (within a small tolerance).
