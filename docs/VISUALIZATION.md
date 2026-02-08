# Visualization Guide

This guide focuses on visualizing TSP price data and analytics outputs. It complements the
analytics overview in [ANALYTICS.md](ANALYTICS.md) and the recipe-style examples in
[EXAMPLES.md](EXAMPLES.md).

## Built-in Matplotlib Charts

`TspAnalytics` includes convenience plotting helpers based on Matplotlib. These methods are
helpful for quick exploratory analysis or notebooks. Each `show_*` helper returns the
Matplotlib `(fig, ax)` pair and accepts `show=False` to suppress display when you want
to embed or save the figure yourself.

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

# Price history line charts
prices.show_individual_price_chart()
prices.show_lifecycle_price_chart()
prices.show_fund_price_chart(TspIndividualFund.C_FUND)

# Capture a figure for embedding or saving
fig, ax = prices.show_fund_price_chart(TspIndividualFund.C_FUND, show=False)
fig.savefig("c-fund.png", dpi=150, bbox_inches="tight")

# Filter charts with a single date bound (the other defaults to the dataset limits)
prices.show_price_history_chart(start_date=date(2024, 1, 1))
prices.show_price_history_chart(end_date=date(2024, 3, 31))

# Moving averages for trend analysis
prices.show_moving_average_chart(
    fund=TspIndividualFund.C_FUND,
    windows=[20, 50],
    method="simple",
)

# Drawdown and volatility
prices.show_drawdown_chart(fund=TspIndividualFund.C_FUND)
prices.show_rolling_volatility_chart(TspIndividualFund.C_FUND, window=63)
prices.show_rolling_performance_summary_chart(TspIndividualFund.C_FUND, window=63)

# Monthly return heatmap
prices.show_monthly_return_heatmap(TspIndividualFund.C_FUND)

# Daily return distribution histogram
prices.show_return_histogram_chart(TspIndividualFund.C_FUND)

# Latest per-fund daily change chart
prices.show_latest_price_changes_per_fund_chart()

# Recent daily change heatmap (compact window)
prices.show_recent_price_change_heatmap(days=5)

# Latest per-fund prices chart
prices.show_latest_prices_per_fund_chart()

# Current per-fund prices chart (optionally anchored to a historical date)
prices.show_current_prices_per_fund_chart()
prices.show_current_prices_per_fund_chart(as_of=date(2024, 1, 2), sort_by="fund")

# Price recency (freshness) chart
prices.show_price_recency_chart()

# Current dashboard metric chart (prices + analytics snapshot)
prices.show_current_price_dashboard_metric_chart(metric="change_percent")
prices.show_current_price_dashboard_metric_chart(metric="trailing_return", period=20)

# Fund rankings (trailing returns or performance metrics)
prices.show_fund_rankings_chart(metric="trailing_return", period=20, top_n=5)
prices.show_fund_rankings_chart(metric="annualized_volatility", top_n=5)
prices.show_fund_rankings_chart(metric="change_percent", top_n=5)

# Trailing return bar chart across multiple horizons
prices.show_trailing_returns_chart(
    periods=[1, 5, 20, 63],
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
)
```

## Dashboard Snapshots

For quick monitoring dashboards, pair the latest price charts with recency analytics:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

prices.show_latest_price_change_chart()
prices.show_latest_price_changes_per_fund_chart()
prices.show_latest_prices_per_fund_chart()
prices.show_current_prices_per_fund_chart()
prices.show_price_recency_chart()
prices.show_current_price_dashboard_metric_chart(metric="change_percent")
```

## Risk & Correlation Visuals

Use scatter and heatmap helpers to compare funds at a glance:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
prices.show_risk_return_scatter()
prices.show_correlation_heatmap()
prices.show_correlation_pairs_chart(top_n=5)
```

## Data Quality Visuals

Use the built-in helpers to visualize fund coverage and missing business days:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
prices.show_fund_coverage_chart()
prices.show_missing_business_days_chart()
```

## Current Price Alert Visuals

Highlight stale prices and large daily moves with the alerts chart helper:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
prices.show_current_price_alerts_chart(metric="change_percent", change_threshold=0.03)
prices.show_current_price_alerts_chart(metric="days_since", stale_days=2)
```

## Long-Format Data for Custom Charts

Many analytics helpers provide long (tidy) data that works well with Seaborn or Plotly.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

price_long = prices.get_price_history_long()
returns_long = prices.get_daily_returns_long()
metrics_long = prices.get_price_history_with_metrics_long()
trailing_long = prices.get_trailing_returns_long(periods=[1, 5, 20, 63])
report_long = prices.get_current_price_report_long()
moving_average_long = prices.get_moving_average_long(window=20)
recent_prices_long = prices.get_recent_prices_long(days=10)

# Latest prices in long format (great for bar charts)
current_long = prices.get_current_prices_long()
```

### Seaborn Example

```python
import seaborn as sns
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
rolling = prices.get_rolling_volatility(TspIndividualFund.C_FUND, window=63)

ax = sns.lineplot(data=rolling, x=rolling.index, y=rolling[TspIndividualFund.C_FUND.value])
ax.set(title="63-Day Rolling Volatility", xlabel="Date", ylabel="Volatility")
```

### Plotly Example

```python
import plotly.express as px
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
history_long = history.melt(id_vars="Date", var_name="Fund", value_name="Price")

fig = px.line(history_long, x="Date", y="Price", color="Fund", title="TSP Fund Prices")
fig.show()
```

## Portfolio Visualization

Use weighted analytics to visualize a portfolio’s performance.

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {TspIndividualFund.C_FUND: 0.6, TspIndividualFund.S_FUND: 0.4}

prices.show_portfolio_value_chart(weights=weights, initial_value=10_000)
prices.show_portfolio_drawdown_chart(weights=weights)
```
