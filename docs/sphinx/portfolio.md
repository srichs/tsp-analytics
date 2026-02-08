# Portfolio Analytics

This guide introduces the weighted portfolio helpers in `TspAnalytics`.

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

weights = {
    TspIndividualFund.C_FUND: 0.6,
    TspIndividualFund.S_FUND: 0.4,
}

portfolio_returns = prices.get_portfolio_returns(weights=weights)
portfolio_cumulative = prices.get_portfolio_cumulative_returns(weights=weights)
portfolio_value = prices.get_portfolio_value_history(weights=weights, initial_value=10_000)
portfolio_summary = prices.get_portfolio_performance_summary(weights=weights)

prices.show_portfolio_value_chart(weights=weights, initial_value=10_000)
```

Use `normalize_weights=False` if your weights already sum to 1 and you want validation instead of
automatic normalization.
