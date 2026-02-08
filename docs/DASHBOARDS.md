# Dashboard & Reporting Guide

This guide focuses on pulling analytics-friendly datasets for dashboards or reports. It
complements the API reference in [REFERENCE.md](REFERENCE.md) and the visualization recipes in
[VISUALIZATION.md](VISUALIZATION.md).

## Current Price Snapshots

Use the "current" aliases to obtain the most recent trading-day prices in formats that are
easy to serialize or visualize:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

current_prices = prices.get_current_prices()
current_prices_long = prices.get_current_prices_long()
current_prices_dict = prices.get_current_prices_dict()
```

The dictionary output is JSON-friendly and includes an `as_of` date plus a `prices` mapping.

## Latest Price Changes

To build a "daily movers" widget or table, use the latest change snapshot:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
latest_changes = prices.get_latest_price_changes()
latest_snapshot = prices.get_latest_price_snapshot()
latest_snapshot_dict = prices.get_latest_price_snapshot_dict()
```

## Per-Fund Latest Price Report

If you need a per-fund table that always uses each fund’s last two valid prices (useful
when the latest row has missing data), use the per-fund report helpers:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
report_per_fund = prices.get_latest_price_report_per_fund()
report_per_fund_long = prices.get_latest_price_report_per_fund_long()
report_per_fund_dict = prices.get_latest_price_report_per_fund_dict()
```

## Fund Overview (Price Changes + Recency)

To show both price changes and the number of days since each fund’s last price, use the
fund overview helpers:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
overview = prices.get_fund_overview()
overview_long = prices.get_fund_overview_long()
overview_dict = prices.get_fund_overview_dict()

# Optionally anchor recency to a prior date for backtesting snapshots
overview_as_of = prices.get_fund_overview(reference_date=date(2024, 1, 3))
```

## Fund Snapshot Summaries

For a compact analytics widget that includes trailing returns and risk metrics, use the
fund snapshot helpers:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
fund_snapshot = prices.get_fund_snapshot(periods=[1, 5, 20, 63])
fund_snapshot_dict = prices.get_fund_snapshot_dict(periods=[1, 5, 20, 63])
```

## Current Price Dashboard Snapshot

If you want a single payload that merges current prices, recency, trailing returns, and
risk metrics, use the dashboard snapshot helpers:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
dashboard = prices.get_current_price_dashboard(periods=[1, 5, 20, 63])
dashboard_dict = prices.get_current_price_dashboard_dict(periods=[1, 5, 20, 63])
```

## Performance & Risk Summary Payloads

For compact widgets, use the JSON-friendly summary helpers:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

performance = prices.get_performance_summary_dict()
performance_range = prices.get_performance_summary_by_date_range_dict(
    start_date=date(2024, 1, 2),
    end_date=date(2024, 1, 4),
)
risk_summary = prices.get_risk_return_summary_dict()
rankings = prices.get_fund_rankings_dict(metric="total_return", top_n=5)
```

## Long-Format Metrics for Dashboards

Long (tidy) data works well with visualization libraries and reporting tools:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

price_history_long = prices.get_price_history_long()
returns_long = prices.get_daily_returns_long()
drawdown_long = prices.get_drawdown_series_long()
snapshot_long = prices.get_fund_snapshot_long(periods=[1, 5, 20, 63])
metrics_long = prices.get_price_history_with_metrics_long()
```

## Portfolio Reporting

If you have portfolio weights, you can build a compact risk/return summary suitable for a
dashboard card:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {
    TspIndividualFund.C_FUND: 0.6,
    TspIndividualFund.S_FUND: 0.4,
}

portfolio_returns = prices.get_portfolio_returns(weights=weights)
portfolio_value = prices.get_portfolio_value_history(weights=weights, initial_value=10_000)
portfolio_risk = prices.get_portfolio_risk_return_summary(weights=weights)
```

## Exporting Data

Every analytics helper returns a Pandas `DataFrame` or `Series`, which makes it easy to export:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics()
history = prices.get_price_history_long()
history.to_csv(Path("tsp_price_history_long.csv"), index=False)
```
