# Usage guide

Use `TspAnalytics` to fetch cached TSP price data, filter it, and visualize it. The
examples below mirror common tasks and are suitable for notebooks or scripts.

## Quick start

```python
from datetime import date
from pathlib import Path
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")

latest_g = prices.get_price(TspIndividualFund.G_FUND)
print(latest_g)

history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
prices.show_price_history_chart(funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND])
```

## Data coverage

```python
from tsp import TspAnalytics

prices = TspAnalytics()
print(prices.get_data_summary())
```

The data summary includes expected business-day counts, missing business days, and a coverage
ratio between the first and last available dates.

## Allocation helper

```python
from tsp import TspAnalytics

prices = TspAnalytics()
allocation = prices.create_allocation(g_shares=10, c_shares=5)
prices.show_pie_chart(allocation)
```

## Benchmarking with beta

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
beta = prices.get_beta(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.S_FUND,
)
print(beta)
```
