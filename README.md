# tsp-analytics

[![CI](https://github.com/srichs/tsp-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/srichs/tsp-analytics/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/srichs/tsp-analytics/ci.yml?branch=main&label=docs)](https://github.com/srichs/tsp-analytics/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tsp-analytics)](https://pypi.org/project/tsp-analytics/)

A Python module for retrieving the prices of Thrift Savings Plan (TSP) funds, calculating analytics, and visualizing historical performance.

## Highlights

- Pulls official TSP fund price history from `tsp.gov` and caches the CSV locally.
- Flexible current-price access (latest row, per-fund latest prices, historical as-of anchors).
- Optional completeness checks for as-of pricing (require all requested funds).
- Current price alerts for stale data and large daily moves.
- Analytics for returns, correlations, drawdowns, rolling metrics, risk stats, and rankings.
- Dashboard-ready outputs (dataframes, long/tidy tables, JSON-friendly dictionaries).
- Current price dashboard snapshot (prices, recency, trailing returns, risk stats).
- Built-in Matplotlib visualization helpers for price history, rankings, recency, and risk.
- Correlation pair summaries to spotlight the strongest fund relationships.
- Data-quality tooling: cache status, fund coverage, missing business days, and reports.
- Weighted portfolio analytics for value, drawdowns, and performance summaries.
- Monthly return tables (wide, long, or dict) for seasonality dashboards.
- Recent daily change windows with summary stats and heatmap-friendly outputs.
- Recent price window helpers for the latest N trading days (wide, long, or JSON-ready).
- Flexible fund name aliases (e.g., `G`, `g-fund`, `L2050`, `Lifecycle 2050`).

## Installation

```bash
pip install tsp-analytics
```

For local development:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

If you prefer a single requirements file instead of extras:

```bash
pip install -r requirements-dev.txt
```

## Cache Location

By default, the price history CSV is cached at `~/.cache/tsp/fund-price-history.csv`.
To customize the cache directory, pass `data_dir` or set `TSP_DATA_DIR`:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")
```

```bash
export TSP_DATA_DIR="$HOME/.cache/tsp"
```

## Offline or Manual Refresh

Disable automatic updates when you need offline access or want to control refresh timing:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.refresh()
```

## Quickstart

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
latest_price = prices.get_price(TspIndividualFund.G_FUND)
latest_prices = prices.get_current_prices()
latest_snapshot = prices.get_current_price_snapshot_dict()
```

## Analytics & Visualization Quickstart

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

# Price history (rows with all selected fund prices missing are dropped)
history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2024, 1, 1),
)

# Long-format data for Plotly/Seaborn dashboards
history_long = prices.get_price_history_long(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# Recent trading-day window for dashboards
recent_prices = prices.get_recent_prices(days=10)
recent_prices_long = prices.get_recent_prices_long(days=10)

# Built-in chart helpers (Matplotlib)
prices.show_fund_price_chart(TspIndividualFund.C_FUND)
prices.show_current_prices_per_fund_chart()
```

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/GETTING_STARTED.md)
- [Usage guide](docs/USAGE.md)
- [Current prices & snapshots](docs/CURRENT_PRICES.md)
- [Analytics & visualization](docs/ANALYTICS.md)
- [Visualization guide](docs/VISUALIZATION.md)
- [Portfolio analytics](docs/PORTFOLIO.md)
- [Examples & recipes](docs/EXAMPLES.md)
- [Dashboards & reporting](docs/DASHBOARDS.md)
- [Configuration](docs/CONFIGURATION.md)
- [Data sources & caching](docs/DATA_SOURCES.md)
- [API reference](docs/REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing guide](docs/CONTRIBUTING.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Testing guide](docs/TESTING.md)
- [Architecture overview](docs/ARCHITECTURE.md)

## Building Documentation

To build the Sphinx documentation locally:

```bash
pip install -r requirements-dev.txt
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

## At a Glance

### Current Prices

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
latest = prices.get_current_prices(fund=TspIndividualFund.G_FUND)
latest_per_fund = prices.get_current_prices(per_fund=True)
latest_dict = prices.get_current_prices_dict()
latest_per_fund_dict = prices.get_current_prices_dict(per_fund=True)
latest_report = prices.get_current_price_report()
latest_report_long = prices.get_current_price_report_long()
latest_report_as_of = prices.get_current_price_report(as_of=date(2024, 1, 2))
latest_report_as_of_long = prices.get_current_price_report_long(as_of=date(2024, 1, 2))
latest_changes = prices.get_current_price_changes()
latest_changes_as_of = prices.get_current_price_changes(as_of=date(2024, 1, 2))
latest_changes_dict = prices.get_current_price_changes_dict(as_of=date(2024, 1, 2))
latest_changes_per_fund = prices.get_current_price_changes_per_fund(as_of=date(2024, 1, 2))
snapshot = prices.get_current_price_snapshot()
snapshot_as_of = prices.get_current_price_snapshot(as_of=date(2024, 1, 2))
snapshot_dict = prices.get_current_price_snapshot_dict(as_of=date(2024, 1, 2))
as_of_prices = prices.get_current_prices(as_of=date(2024, 1, 2))
as_of_prices_complete = prices.get_current_prices(
    as_of=date(2024, 1, 2),
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    require_all_funds=True,
)
as_of_per_fund = prices.get_current_prices_per_fund(as_of=date(2024, 1, 2))
as_of_per_fund_dict = prices.get_current_prices_per_fund_dict(as_of=date(2024, 1, 2))
safe_per_fund = prices.get_current_prices_per_fund(
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    allow_missing=True,
)
safe_per_fund_payload = prices.get_current_prices_per_fund_dict(
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    allow_missing=True,
)
status = prices.get_current_price_status()
status_dict = prices.get_current_price_status_dict()
status_g = prices.get_current_price_status(fund=TspIndividualFund.G_FUND)
summary = prices.get_current_price_summary(stale_days=2)
summary_dict = prices.get_current_price_summary_dict(stale_days=2)
summary_as_of = prices.get_current_price_summary(as_of=date(2024, 1, 2), stale_days=2)
summary_as_of_dict = prices.get_current_price_summary_dict(
    as_of=date(2024, 1, 2),
    stale_days=2,
)
alerts = prices.get_current_price_alerts(stale_days=2, change_threshold=0.03)
alerts_dict = prices.get_current_price_alerts_dict(stale_days=2, change_threshold=0.03)
alerts_g = prices.get_current_price_alerts(fund="G", stale_days=2, change_threshold=0.03)
alert_summary = prices.get_current_price_alert_summary(stale_days=2, change_threshold=0.03)
alert_summary_dict = prices.get_current_price_alert_summary_dict(
    stale_days=2,
    change_threshold=0.03,
)
```

> Tip: When requesting `as_of` prices for a single fund, the lookup skips rows where that fund
> has missing prices so you always get the most recent valid value.

### Analytics

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
returns = prices.get_daily_returns()
drawdown = prices.get_drawdown_series(fund=TspIndividualFund.C_FUND)
price_history = prices.get_price_history(start_date=date(2024, 1, 1))
risk_summary = prices.get_risk_return_summary()
performance_summary_dict = prices.get_performance_summary_dict()
risk_summary_dict = prices.get_risk_return_summary_dict()
dashboard = prices.get_current_price_dashboard(periods=[1, 5, 20])
price_summary = prices.get_price_summary()
price_recency = prices.get_price_recency()
correlation_pairs = prices.get_correlation_pairs(top_n=5)
current_overview = prices.get_current_fund_overview()
current_overview_as_of = prices.get_current_fund_overview(as_of=date(2024, 1, 2))
current_change_rank = prices.get_fund_rankings(metric="change_percent")
return_distribution = prices.get_return_distribution_summary(
    fund=TspIndividualFund.C_FUND,
    percentiles=[0.05, 0.5, 0.95],
)
price_stats = prices.get_price_statistics(start_date=date(2024, 1, 1))
return_stats = prices.get_return_statistics(end_date=date(2024, 3, 31))
price_stats_dict = prices.get_price_statistics_dict(fund=TspIndividualFund.G_FUND)
return_stats_dict = prices.get_return_statistics_dict(
    fund=TspIndividualFund.G_FUND,
    trading_days=252,
)
recent_changes = prices.get_recent_price_changes(days=5)
recent_change_summary = prices.get_recent_price_change_summary(days=5)
recent_changes_dict = prices.get_recent_price_changes_dict(days=5)
fund_report = prices.get_fund_analytics_report(
    TspIndividualFund.C_FUND,
    start_date=date(2024, 1, 1),
)
fund_report_dict = prices.get_fund_analytics_report_dict(
    TspIndividualFund.C_FUND,
    start_date=date(2024, 1, 1),
)
drawdown_summary = prices.get_drawdown_summary_dict(TspIndividualFund.C_FUND)
range_changes = prices.get_price_change_by_date_range_dict(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)
as_of_changes = prices.get_price_changes_as_of_per_fund(date(2024, 1, 3))
monthly_return_table = prices.get_monthly_return_table_long(TspIndividualFund.C_FUND)
rolling_volatility = prices.get_rolling_volatility(TspIndividualFund.C_FUND, window=63)
rolling_performance = prices.get_rolling_performance_summary(TspIndividualFund.C_FUND, window=63)
trailing_return = prices.get_trailing_returns(periods=20, fund=TspIndividualFund.C_FUND)
trailing_return_long = prices.get_trailing_returns_long(periods=[1, 5, 20, 63])
trailing_return_dict = prices.get_trailing_returns_dict(periods=[1, 5, 20, 63])
correlation_payload = prices.get_correlation_matrix_dict()
rolling_correlation_payload = prices.get_rolling_correlation_matrix_dict(window=63)
```

### Visualization

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
prices.show_fund_price_chart(TspIndividualFund.C_FUND)
prices.show_latest_price_changes_per_fund_chart()
prices.show_recent_price_change_heatmap(days=5)
prices.show_price_recency_chart()
prices.show_current_prices_per_fund_chart()
prices.show_price_history_chart(start_date=date(2024, 1, 1))
prices.show_return_histogram_chart(TspIndividualFund.C_FUND)
prices.show_correlation_heatmap()
prices.show_correlation_pairs_chart(top_n=5)
prices.show_rolling_performance_summary_chart(TspIndividualFund.C_FUND, window=63)
prices.show_fund_rankings_chart(metric="trailing_return", period=20, top_n=5)
prices.show_current_price_dashboard_metric_chart(metric="change_percent")
prices.show_current_price_alerts_chart(metric="change_percent", change_threshold=0.03)
prices.show_trailing_returns_chart(
    periods=[1, 5, 20, 63],
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
)

# Capture charts for custom dashboards or reports
fig, ax = prices.show_fund_price_chart(TspIndividualFund.C_FUND, show=False)
fig.savefig("c-fund.png", dpi=150, bbox_inches="tight")

# Current price bar chart anchored to a historical date
fig, ax = prices.show_current_prices_per_fund_chart(
    as_of=date(2024, 1, 2),
    sort_by="fund",
    show=False,
)
```

### Data Quality & Cache Status

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
status = prices.get_cache_status()
status_dict = prices.get_cache_status_dict()
report = prices.get_data_quality_report()
report_dict = prices.get_data_quality_report_dict()
```

### Fund Metadata & Aliases

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
aliases = prices.get_fund_aliases()
metadata = prices.get_fund_metadata()
metadata_dict = prices.get_fund_metadata_dict()
```

## Data Sources & Caching

The library downloads the official `fund-price-history.csv` file from `tsp.gov` and caches it
locally. The cache refreshes automatically after the most recent business day and the configured
update time (defaults to 7:00 PM local time). You can override the cache directory with the
`TSP_DATA_DIR` environment variable or the `data_dir` constructor argument:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")
```

If you need to run fully offline or inject a custom HTTP session, disable auto updates and
load data explicitly:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.refresh()  # or prices.load_csv(...), prices.load_csv_text(...)
```

For custom networking (proxies, retries, shared sessions), pass a pre-configured session.
The library ensures a CSV-friendly `Accept` header and a user-agent string are present:

```python
import requests
from tsp import TspAnalytics

session = requests.Session()
session.headers.update({"User-Agent": "my-app/1.0"})
prices = TspAnalytics(session=session, request_timeout=15.0, max_retries=3, retry_backoff=0.5)
```

Downloads are validated to ensure the response looks like a CSV (including a `Date` header
and known fund columns). If the upstream response appears to be HTML or missing expected headers,
the client raises
a validation error and preserves the existing cache. See
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for guidance.

To load a pandas dataframe directly, pass it into `load_dataframe`. The dataframe can include
either a `Date` column or a date-like index (named `Date` or a datetime index):

```python
import pandas as pd
from tsp import TspAnalytics, TspIndividualFund

df = pd.DataFrame(
    {
        TspIndividualFund.G_FUND.value: [100.0, 101.5],
        TspIndividualFund.C_FUND.value: [200.0, 202.0],
    },
    index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
)

prices = TspAnalytics(auto_update=False)
prices.load_dataframe(df)
```

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for full details.

## Analytics & Visualization Tips

- Use the long-format helpers (`get_price_history_long`, `get_daily_returns_long`) when charting
  with Seaborn or Plotly.
- Price history helpers drop rows where all selected fund prices are missing to avoid blank
  records in downstream charts and analytics.
- Use the `show_*` helpers for quick Matplotlib visuals; they return `(fig, ax)` so you can save
  or embed charts without opening a window.
- Use dictionary helpers (like `get_current_price_report_dict`) for JSON-ready dashboard payloads.

Start with [docs/ANALYTICS.md](docs/ANALYTICS.md) and [docs/VISUALIZATION.md](docs/VISUALIZATION.md)
for step-by-step examples.

## Contributing & Development

Interested in contributing? Start here:

- [Contributing guide](CONTRIBUTING.md)
- [Contributor guide](docs/CONTRIBUTING.md)
- [Development guide](docs/DEVELOPMENT.md) (architecture & tooling notes)
- [Testing guide](docs/TESTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

Quick setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

When you add user-facing features, please update the relevant docs in `docs/` and include
tests under `tests/`.
