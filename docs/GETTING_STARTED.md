# Getting Started

This guide walks through installation, the first data pull, and a few high-value analytics
queries. For deeper examples, see [docs/USAGE.md](USAGE.md) and [docs/ANALYTICS.md](ANALYTICS.md).

## Install

```bash
pip install tsp-analytics
```

For local development:

```bash
pip install -r requirements.txt
pip install -e .
```

## First Run

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
latest = prices.get_price(TspIndividualFund.G_FUND)
print(latest)
```

## Fund Name Shortcuts

Fund inputs accept enums or flexible string aliases:

```python
prices.get_latest_prices(funds="G")        # alias for G Fund
prices.get_latest_prices(funds="g-fund")   # alias for G Fund
prices.get_latest_prices(funds="L2050")    # alias for L 2050
prices.get_latest_prices(funds="L2050fund") # compact lifecycle alias
```

## Common Analytics

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

# Daily returns
returns = prices.get_daily_returns()

# Rolling volatility
rolling_vol = prices.get_rolling_volatility(
    fund=TspIndividualFund.C_FUND,
    window=63,
)

# Risk/return summary
risk_summary = prices.get_risk_return_summary()
```

## Quick Visuals

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

prices.show_fund_price_chart(TspIndividualFund.C_FUND)
prices.show_latest_price_change_chart()
prices.show_correlation_heatmap()
```

## Next Steps

- Review [docs/USAGE.md](USAGE.md) for caching and data refresh controls.
- Explore [docs/ANALYTICS.md](ANALYTICS.md) for advanced analytics and visualization recipes.
- See [docs/CONTRIBUTING.md](CONTRIBUTING.md) if you plan to extend the library.
