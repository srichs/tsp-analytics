# Analytics & visualization

`TspAnalytics` includes helpers for returns analysis, rolling metrics, and
visualizations. These examples show typical analysis tasks.

## Returns and performance

```python
from datetime import date
from tsp import TspAnalytics, TspLifecycleFund

prices = TspAnalytics()
summary = prices.get_performance_summary(TspLifecycleFund.L_2040)
print(summary)

changes = prices.get_price_change_by_date_range(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)
print(changes)

changes_dict = prices.get_price_change_by_date_range_dict(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)
print(changes_dict["funds"]["G Fund"])
```

## Rolling metrics

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
rolling_sharpe = prices.get_rolling_sharpe_ratio(TspIndividualFund.C_FUND)
print(rolling_sharpe.tail())

rolling_beta = prices.get_rolling_beta(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.S_FUND,
)
print(rolling_beta.tail())

rolling_drawdown = prices.get_rolling_max_drawdown(
    fund=TspIndividualFund.C_FUND,
    window=252,
)
print(rolling_drawdown.tail())
```

## Charts

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
prices.show_fund_price_chart(TspIndividualFund.C_FUND)
prices.show_correlation_heatmap()
prices.show_rolling_beta_chart(TspIndividualFund.C_FUND, TspIndividualFund.S_FUND)
prices.show_daily_return_histogram(TspIndividualFund.C_FUND, bins=40)
prices.show_rolling_max_drawdown_chart(TspIndividualFund.C_FUND, window=252)
```
