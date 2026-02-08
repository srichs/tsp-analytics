# Current Prices & Snapshots

This guide focuses on retrieving the latest available TSP prices, daily changes, and
snapshot-style payloads that are ready for dashboards or APIs.

## Latest vs. Current

The library treats "current" prices as the latest available trading-day prices in the
cached dataset. Methods prefixed with `get_current_*` are aliases of the corresponding
`get_latest_*` helpers. Use whichever naming makes your code clearer.

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

latest = prices.get_latest_prices()
current = prices.get_current_prices()

latest_g = prices.get_latest_prices(fund=TspIndividualFund.G_FUND)
current_g = prices.get_current_prices(fund=TspIndividualFund.G_FUND)

# The single-fund helper uses the most recent valid price for that fund.
latest_g_value = prices.get_price(TspIndividualFund.G_FUND)

# Single-letter aliases are supported for individual funds.
latest_g_alias = prices.get_latest_prices(fund="G")
# Compact lifecycle aliases are also normalized (e.g., "L2050fund").
latest_l2050 = prices.get_latest_prices(fund="L2050fund")

# Anchor the "current" prices to a historical date when needed.
current_as_of = prices.get_current_prices(as_of=date(2024, 1, 2))
current_as_of_dict = prices.get_current_prices_dict(as_of=date(2024, 1, 2))

# Return per-fund latest prices without switching to the per-fund helper.
current_per_fund = prices.get_current_prices(per_fund=True)
current_per_fund_dict = prices.get_current_prices_dict(per_fund=True)
```

When you request `as_of` prices for a single fund (for example, via
`get_prices_as_of(..., fund="G")`), the helper skips rows where the fund price is missing.
This ensures you get the most recent valid price on or before the requested date.

When you request `as_of` prices without specifying funds, only recognized TSP fund columns
are returned. Any extra columns in a custom dataframe (for example, notes or annotations)
are ignored to keep the output focused on fund prices.

## Current Prices in Dictionary Form

When you need JSON-friendly payloads (for example, an API response), use the `*_dict`
helpers. The `as_of` field is formatted by default as ISO 8601. Missing numeric values
are converted to `None` so the payload can be safely serialized to JSON.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

payload = prices.get_current_prices_dict()
print(payload["as_of"], payload["prices"])

payload = prices.get_current_prices_dict(date_format=None)

# Per-fund dictionary payloads (each fund has its own as_of date)
payload_per_fund = prices.get_current_prices_dict(per_fund=True)
```

## Per-Fund Latest Prices

If you need the most recent valid price for each fund (even when the latest row contains
missing values for some funds), use the per-fund helpers. Each fund uses its own most recent
date, so the `as_of` column can differ by fund.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

latest_per_fund = prices.get_latest_prices_per_fund()
current_per_fund = prices.get_current_prices_per_fund()
latest_per_fund_long = prices.get_latest_prices_per_fund_long()
latest_per_fund_dict = prices.get_latest_prices_per_fund_dict()
current_per_fund_long = prices.get_current_prices_per_fund_long()
current_per_fund_dict = prices.get_current_prices_per_fund_dict()

# Anchor per-fund prices to a historical date (each fund uses its most recent price on or before).
current_per_fund_as_of = prices.get_current_prices_per_fund(as_of=date(2024, 1, 2))
current_per_fund_as_of_dict = prices.get_current_prices_per_fund_dict(
    as_of=date(2024, 1, 2),
)

# Skip funds that have no available prices and capture missing fund names.
current_per_fund_safe = prices.get_current_prices_per_fund(
    funds=["G", "C"],
    allow_missing=True,
)
current_per_fund_safe_payload = prices.get_current_prices_per_fund_dict(
    funds=["G", "C"],
    allow_missing=True,
)
missing_funds = current_per_fund_safe_payload.get("missing_funds", [])
```

> Note: `allow_missing` is only supported with per-fund helpers (or `per_fund=True`). For
> non-per-fund lookups, use `require_all_funds=True` when you need strict completeness.

## Visualizing Current Prices

To visualize the current (latest available) prices quickly, use the chart helper. You can
anchor the view to a historical date for backtesting or reporting.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
prices.show_current_prices_per_fund_chart()
prices.show_current_prices_per_fund_chart(as_of=date(2024, 1, 2), sort_by="fund")
```

## Price Recency (Freshness)

If you want to understand how fresh each fund's latest price is, use the price
recency helpers. These compute the number of days since each fund's most recent
valid price, using either the latest trading day in the dataset or a custom
reference date.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

recency = prices.get_price_recency()
recency_as_of = prices.get_price_recency(reference_date=date(2024, 1, 2))
recency_g = prices.get_price_recency(fund="G")
recency_dict = prices.get_price_recency_dict()
```

## Current Fund Overview (Prices + Recency)

If you want a dashboard-ready view that combines daily price changes with recency metrics,
use the fund overview helpers. The `get_current_*` aliases emphasize that you're using the
latest available data:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

overview = prices.get_current_fund_overview()
overview_long = prices.get_current_fund_overview_long()
overview_dict = prices.get_current_fund_overview_dict()
overview_g = prices.get_current_fund_overview(fund="G")

# Anchor the overview to a historical date for point-in-time reporting.
overview_as_of = prices.get_current_fund_overview(as_of=date(2024, 1, 2))
overview_as_of_dict = prices.get_current_fund_overview_dict(as_of=date(2024, 1, 2))
```

> Tip: When you provide `as_of`, the overview uses prices on or before the requested date.
> You can also pass a later `reference_date` to compute recency relative to a reporting date.

## Current Price Dashboard Snapshot

For a single payload that combines current prices, recency, trailing returns, and risk metrics,
use the dashboard snapshot helper:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
dashboard = prices.get_current_price_dashboard(periods=[1, 5, 20, 63])
dashboard_dict = prices.get_current_price_dashboard_dict(periods=[1, 5, 20, 63])
```

## Current Price Changes Anchored to a Date

If you need daily price changes but want to anchor them to a historical date, use the
`as_of` parameter. This is useful for backtesting or producing point-in-time reports.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

changes = prices.get_current_price_changes(as_of=date(2024, 1, 2))
changes_long = prices.get_current_price_changes_long(as_of=date(2024, 1, 2))
changes_dict = prices.get_current_price_changes_dict(as_of=date(2024, 1, 2))

# Per-fund changes based on each fund's last two valid prices
changes_per_fund = prices.get_current_price_changes_per_fund(as_of=date(2024, 1, 2))
changes_per_fund_dict = prices.get_current_price_changes_per_fund_dict(as_of=date(2024, 1, 2))
```

## Current Price Snapshots Anchored to a Date

Use price snapshots when you want the latest price, previous price, and daily change metrics
in a single table. The `as_of` parameter lets you anchor snapshots to historical dates.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

snapshot = prices.get_current_price_snapshot(as_of=date(2024, 1, 2))
snapshot_long = prices.get_current_price_snapshot_long(as_of=date(2024, 1, 2))
snapshot_dict = prices.get_current_price_snapshot_dict(as_of=date(2024, 1, 2))
```

## Current Price Status (Prices + Recency Without Changes)

If you want the latest per-fund price with recency metrics but do not need daily change
calculations, use the current price status helpers. These are lighter-weight than the
fund overview when you only need freshness and the latest price:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

status = prices.get_current_price_status()
status_long = prices.get_current_price_status_long()
status_dict = prices.get_current_price_status_dict()
status_g = prices.get_current_price_status(fund="G")

# Anchor to a historical date and use that date as the recency reference.
status_as_of = prices.get_current_price_status(as_of=date(2024, 1, 2))
```

> Note: When you pass both `as_of` and `reference_date`, the reference date must be on or after
> the `as_of` date so recency values stay non-negative.

## Current Price Summary (Freshness + Change Stats)

If you need a compact health check for freshness and daily changes across funds, use the
summary helper. This aggregates the recency metrics, counts of positive/negative changes,
and descriptive stats for the daily changes.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

summary = prices.get_current_price_summary(stale_days=2)
summary_dict = prices.get_current_price_summary_dict(stale_days=2)

# Anchor the summary to a historical date (point-in-time monitoring).
summary_as_of = prices.get_current_price_summary(as_of=date(2024, 1, 2), stale_days=2)
summary_as_of_dict = prices.get_current_price_summary_dict(
    as_of=date(2024, 1, 2),
    stale_days=2,
)
```

## Current Price Alerts (Stale Prices + Large Moves)

The alerts helpers combine recency, daily changes, and alert flags for stale data or
large price moves. This is useful for monitoring dashboards or notifications.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

alerts = prices.get_current_price_alerts(stale_days=2, change_threshold=0.03)
alerts_long = prices.get_current_price_alerts_long(stale_days=2, change_threshold=0.03)
alerts_dict = prices.get_current_price_alerts_dict(stale_days=2, change_threshold=0.03)
alerts_g = prices.get_current_price_alerts(fund="G", stale_days=2, change_threshold=0.03)

# Anchor to a historical date and use a custom reference date for recency.
alerts_as_of = prices.get_current_price_alerts(
    as_of=date(2024, 1, 2),
    reference_date=date(2024, 1, 4),
    stale_days=1,
)
```

## Visualizing Current Price Analytics

Use the chart helpers for a quick visual review of stale prices, large daily moves, and
dashboard metrics. These helpers return `(fig, ax)` so you can save or embed the charts.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

# Bar chart of per-fund staleness and large daily moves.
prices.show_current_price_alerts_chart(metric="change_percent", change_threshold=0.03)

# Highlight a dashboard metric across funds (e.g., trailing returns or volatility).
prices.show_current_price_dashboard_metric_chart(metric="trailing_return_20d")
```

### Alert Summary Snapshot

If you want a compact summary of alert counts (stale prices and large moves), use the
alert summary helpers. This is useful for dashboard badges or monitoring checks:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

summary = prices.get_current_price_alert_summary(stale_days=2, change_threshold=0.03)
summary_dict = prices.get_current_price_alert_summary_dict(
    stale_days=2,
    change_threshold=0.03,
)
```

## Current Price Reports (Prices + Daily Changes)

Use the report helpers to combine prices and daily change metrics. You can also anchor the
report to a historical date when building dashboards or backtesting snapshots.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

report = prices.get_current_price_report()
report_long = prices.get_current_price_report_long()
report_dict = prices.get_current_price_report_dict()

# Anchor to a historical date (uses the most recent data on or before the date).
report_as_of = prices.get_current_price_report(as_of=date(2024, 1, 3))
report_as_of_long = prices.get_current_price_report_long(as_of=date(2024, 1, 3))
report_as_of_dict = prices.get_current_price_report_dict(as_of=date(2024, 1, 3))

# Per-fund report output (each fund uses its last two valid prices).
report_per_fund = prices.get_current_price_report_per_fund()
report_per_fund_as_of = prices.get_current_price_report_per_fund(as_of=date(2024, 1, 3))
report_per_fund_dict = prices.get_current_price_report_per_fund_dict(as_of=date(2024, 1, 3))
```

## Current Price Changes

Use the change helpers to compute daily deltas vs. the previous trading day.

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

changes = prices.get_current_price_changes()
changes_single = prices.get_current_price_changes(fund=TspIndividualFund.C_FUND)
changes_long = prices.get_current_price_changes_long()
changes_dict = prices.get_current_price_changes_dict()
```

## Recent Daily Change Windows

If you want to inspect the most recent trading-day changes across a window (for example, the
last 5 trading days), use the recent change helpers. These return percent changes by day and
can be anchored to a historical `as_of` date:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

recent_changes = prices.get_recent_price_changes(days=5)
recent_changes_long = prices.get_recent_price_changes_long(days=5)
recent_changes_dict = prices.get_recent_price_changes_dict(days=5)

# Anchor the recent window to a historical date if needed
recent_changes_as_of = prices.get_recent_price_changes(days=5, as_of=date(2024, 1, 31))
recent_summary = prices.get_recent_price_change_summary(days=5)
recent_summary_dict = prices.get_recent_price_change_summary_dict(days=5)
```

## Per-Fund Latest Changes

If you need per-fund price changes that account for missing values in the latest row,
use the per-fund helpers. These use each fund's last two valid price points.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

per_fund_changes = prices.get_latest_price_changes_per_fund()
per_fund_changes_long = prices.get_latest_price_changes_per_fund_long()
per_fund_changes_dict = prices.get_latest_price_changes_per_fund_dict()
```

## Per-Fund Latest Price Report

If you want a combined per-fund report that includes both the latest price and the daily
change metrics, use the per-fund report helpers. Each fund uses its own most recent valid
price pair, so the `as_of` and `previous_as_of` dates can vary by fund.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

report_per_fund = prices.get_latest_price_report_per_fund()
report_per_fund_long = prices.get_latest_price_report_per_fund_long()
report_per_fund_dict = prices.get_latest_price_report_per_fund_dict()
```

## Current Price Snapshot

Snapshots bundle the as-of date with price and change metrics. This is helpful when you
want a single payload to drive a dashboard widget.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

snapshot = prices.get_current_price_snapshot()
snapshot_long = prices.get_current_price_snapshot_long()
snapshot_dict = prices.get_current_price_snapshot_dict()
```

## Current Price Report (Combined Prices + Changes)

If you want a single dictionary that includes both the latest prices and the daily changes,
use the combined report helper. This pairs the latest price payload with the daily change
metrics in one response.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

report = prices.get_current_price_report_dict()
report_subset = prices.get_current_price_report_dict(funds=["C Fund", "S Fund"])
report_with_cache = prices.get_current_price_report_dict(include_cache_status=True)
report_with_quality = prices.get_current_price_report_dict(include_data_quality=True)
```

When you set `include_cache_status=True`, the report includes the JSON-friendly cache
metadata from `get_cache_status_dict()`. The cache dates use the same `date_format`
as the rest of the report, and `date_format=None` returns native `date`/`datetime` objects.

## Current Price Report (DataFrame)

If you prefer a dataframe for downstream analytics or exports, use the report dataframe
helper. It includes the as-of and previous-as-of dates alongside the change metrics.

```python
from tsp import TspAnalytics

prices = TspAnalytics()

report = prices.get_current_price_report()
report_subset = prices.get_current_price_report(funds=["C Fund", "S Fund"])
```

## As-of Pricing (Historical Anchor)

Use `get_prices_as_of()` when you need the most recent trading-day prices on or before a
specific date (for example, around holidays).

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

as_of_prices = prices.get_prices_as_of(date(2024, 1, 2))
# Require all requested funds to have data on the resolved as-of date.
as_of_prices_complete = prices.get_prices_as_of(
    date(2024, 1, 2),
    funds=["G Fund", "C Fund"],
    require_all_funds=True,
)
# Dictionary output includes both the requested date and the resolved trading day.
as_of_dict = prices.get_prices_as_of_dict(date(2024, 1, 2))
```

## As-of Per-Fund Pricing (Historical Anchor)

If you need each fund’s most recent valid price on or before a specific date (even when
some funds have missing values on the latest row), use the per-fund helpers. Each fund uses
its own `as_of` date.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

as_of_per_fund = prices.get_prices_as_of_per_fund(date(2024, 1, 2))
as_of_per_fund_long = prices.get_prices_as_of_per_fund_long(date(2024, 1, 2))
as_of_per_fund_dict = prices.get_prices_as_of_per_fund_dict(date(2024, 1, 2))
```

## As-of Price Changes (Historical Anchor)

To calculate daily changes as of a specific date, use the as-of change helpers. The dictionary
form includes both the as-of date and the previous trading day used for the comparison.

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

changes = prices.get_price_changes_as_of(date(2024, 1, 2))
changes_long = prices.get_price_changes_as_of_long(date(2024, 1, 2))
changes_dict = prices.get_price_changes_as_of_dict(
    date(2024, 1, 2),
    fund=TspIndividualFund.C_FUND,
)
```

## Troubleshooting

- If you are offline, initialize with `auto_update=False` and load a cached CSV or dataframe
  with `load_csv()` or `load_dataframe()`.
- Use `get_cache_status()` to confirm whether your cache is stale and what funds are present.
- If you load data that has no usable fund prices (for example, all fund columns are empty),
  the current/latest price helpers will raise a `ValueError` indicating that no funds are
  available. Check `get_available_funds()` and the data-quality report before running
  analytics.
