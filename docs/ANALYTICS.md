# Analytics & Visualization

This project provides analytics helpers for returns, volatility, drawdowns, correlations, and more. All analytics are derived from the official TSP CSV data.

All fund-specific methods accept either fund enums or fund name strings (case-insensitive). For
example, `"C Fund"`, `" c fund "`, and `"C"` work anywhere a `fund` argument is expected.

## Current Price Monitoring

Use the current price helpers when you need a quick operational view of the latest trading-day
prices, daily changes, and recency metrics:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

# Latest prices + daily changes
report = prices.get_current_price_report()
report_as_of = prices.get_current_price_report(as_of=date(2024, 1, 3))
report_dict = prices.get_current_price_report_dict(as_of=date(2024, 1, 3))
report_long = prices.get_current_price_report_long()
report_as_of_long = prices.get_current_price_report_long(as_of=date(2024, 1, 3))

# Per-fund latest prices with last two valid prices for each fund
per_fund_report = prices.get_current_price_report_per_fund()
per_fund_report_as_of = prices.get_current_price_report_per_fund(as_of=date(2024, 1, 3))

# Recency-focused status (price + days since last update)
status = prices.get_current_price_status()
status_dict = prices.get_current_price_status_dict()
status_g = prices.get_current_price_status(fund="G")

# Summary rollup for freshness + daily change stats
summary = prices.get_current_price_summary(stale_days=2)
summary_dict = prices.get_current_price_summary_dict(stale_days=2)

# Combined alerts for stale prices or large daily moves
alerts = prices.get_current_price_alerts(stale_days=2, change_threshold=0.03)
alerts_dict = prices.get_current_price_alerts_dict(stale_days=2, change_threshold=0.03)
alerts_g = prices.get_current_price_alerts(fund="G", stale_days=2, change_threshold=0.03)

# Snapshot view with latest/previous prices and changes
snapshot = prices.get_current_price_snapshot()
snapshot_as_of = prices.get_current_price_snapshot(as_of=date(2024, 1, 3))
snapshot_dict = prices.get_current_price_snapshot_dict(as_of=date(2024, 1, 3))
```

If you need aligned as-of pricing for multiple funds (for example, to compare funds on a
single trading day without missing values), set `require_all_funds=True`:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
aligned = prices.get_current_prices(
    as_of=date(2024, 1, 2),
    funds=["G Fund", "C Fund"],
    require_all_funds=True,
)
```

## Dashboard Payloads (JSON-Friendly)

Use the `*_dict` helpers to produce API-ready payloads for dashboards. Dates are formatted as
ISO 8601 by default and missing numeric values are returned as `None`.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

current_report = prices.get_current_price_report_dict(include_cache_status=True)
dashboard = prices.get_current_price_dashboard_dict(periods=[1, 5, 20, 63])
fund_snapshot = prices.get_fund_snapshot_dict(periods=[1, 5, 20, 63])
correlation_matrix = prices.get_correlation_matrix_dict()
```

## Returns

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

# Daily returns for all funds
returns = prices.get_daily_returns()

# Cumulative returns for a single fund
cumulative = prices.get_cumulative_returns(fund=TspIndividualFund.S_FUND)
```

Returns are computed without forward-filling missing prices. If a fund has gaps in its price
history, the corresponding return rows will remain `NaN` so you can detect missing data rather
than generating synthetic returns.

## Price History & Long-Format Data

Use `get_price_history` for wide tables and `get_price_history_long` for tidy outputs that
work well with Plotly or Seaborn. Rows where all selected fund prices are missing are dropped
so downstream charts and analytics do not include empty records.

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2024, 1, 1),
)
history_long = prices.get_price_history_long(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
```

## Recent Price Windows

For dashboards or short-term charts, use `get_recent_prices` to retrieve the most recent
trading-day prices (optionally anchored to a historical date).

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

# Latest 5 trading days for all funds
recent = prices.get_recent_prices(days=5)

# Latest 10 trading days anchored to a historical date
recent_as_of = prices.get_recent_prices(days=10, as_of=date(2024, 1, 15))

# Long-format output for Plotly or Seaborn
recent_long = prices.get_recent_prices_long(days=5)

# JSON-friendly output for dashboards
recent_dict = prices.get_recent_prices_dict(days=5)
```

## Recent Daily Change Windows

To focus on the most recent trading days (for monitoring dashboards or short-horizon analytics),
use the recent price change helpers. These provide a compact window of daily percent changes and
summary statistics.

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

recent_changes = prices.get_recent_price_changes(days=5)
recent_changes_long = prices.get_recent_price_changes_long(days=5)
recent_changes_dict = prices.get_recent_price_changes_dict(days=5)

recent_summary = prices.get_recent_price_change_summary(days=5)
recent_summary_dict = prices.get_recent_price_change_summary_dict(days=5)
```

## Price & Return Statistics

Descriptive statistics are available for both prices and daily returns, including medians,
skew, kurtosis, and annualized metrics for return series:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

price_stats = prices.get_price_statistics(fund=TspIndividualFund.G_FUND)
return_stats = prices.get_return_statistics(fund=TspIndividualFund.G_FUND, trading_days=252)
```

If you need JSON-friendly outputs (for APIs or dashboards), use the dictionary helpers:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

price_stats_dict = prices.get_price_statistics_dict(fund=TspIndividualFund.G_FUND)
return_stats_dict = prices.get_return_statistics_dict(
    fund=TspIndividualFund.G_FUND,
    trading_days=252,
)
```

You can scope statistics to a date range. If you only supply one bound, the other defaults to
the earliest/latest available date:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

stats_from = prices.get_price_statistics(start_date=date(2024, 1, 1))
stats_through = prices.get_return_statistics(end_date=date(2024, 3, 31))
```

If you want a compact summary of each fund's price history (first/last dates, min/max/mean,
and total return), use the price summary helpers:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

summary = prices.get_price_summary()
summary_g = prices.get_price_summary(funds=[TspIndividualFund.G_FUND])
summary_dict = prices.get_price_summary_dict()
```

## Fund Analytics Report (Bundled)

When you need a single payload that combines multiple analytics tables for one fund,
use the fund analytics report. It bundles price statistics, return statistics,
performance summary, drawdown summary, price summary, and current overview metrics:

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

report = prices.get_fund_analytics_report(
    TspIndividualFund.C_FUND,
    start_date=date(2024, 1, 1),
)
report_dict = prices.get_fund_analytics_report_dict(
    TspIndividualFund.C_FUND,
    start_date=date(2024, 1, 1),
)
```

If you only need drawdown summaries for a single fund in JSON form:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
drawdown = prices.get_drawdown_summary_dict(TspIndividualFund.C_FUND)
```

## Return Distribution Summaries

If you want a distribution-centric view (percentiles plus win/loss rates), use the return
distribution summary helpers:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

distribution = prices.get_return_distribution_summary(
    fund=TspIndividualFund.C_FUND,
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99],
)
distribution_dict = prices.get_return_distribution_summary_dict(
    fund=TspIndividualFund.C_FUND,
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99],
)
```

## Correlation Matrices (Dictionary Output)

Use the correlation matrix helpers when you need JSON-friendly payloads for APIs or dashboards.
They include the date window used for the correlations and format missing values as `None`:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

correlation_payload = prices.get_correlation_matrix_dict()
rolling_payload = prices.get_rolling_correlation_matrix_dict(window=63)
```

If you want to surface the strongest correlation pairs for reporting, use the pair helpers:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

pairs = prices.get_correlation_pairs(top_n=5)
pairs_dict = prices.get_correlation_pairs_dict(top_n=5)
```

## Trailing Returns & Snapshot Metrics

Trailing returns accept either a single integer period or a list of periods. This is useful for
dashboard cards or reports that focus on a few recent horizons:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

trailing_20 = prices.get_trailing_returns(periods=20, fund=TspIndividualFund.C_FUND)
trailing_multi = prices.get_trailing_returns(periods=[1, 5, 20, 63])
trailing_long = prices.get_trailing_returns_long(periods=[1, 5, 20, 63])
trailing_dict = prices.get_trailing_returns_dict(periods=[1, 5, 20, 63])

snapshot = prices.get_fund_snapshot(periods=20)
snapshot_multi = prices.get_fund_snapshot(periods=[1, 5, 20, 63])
```

Snapshots include the latest prices, trailing returns, and performance metrics in one table.
To visualize the trailing return table directly, use the chart helper:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
prices.show_trailing_returns_chart(
    periods=[1, 5, 20, 63],
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
)
```

## Fund Overview & Price Recency

If you want a dashboard-ready view of the latest price changes plus recency (how many days
since each fund's most recent price), use the fund overview helpers. The `get_current_*`
variants are aliases that emphasize you're using the latest available data:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

overview = prices.get_fund_overview()
current_overview = prices.get_current_fund_overview()
overview_long = prices.get_current_fund_overview_long()
overview_dict = prices.get_current_fund_overview_dict()
overview_g = prices.get_current_fund_overview(fund="G")

anchored_overview = prices.get_current_fund_overview(reference_date=date(2024, 1, 3))
```

If you only need the latest price, the as-of date, and freshness (without daily change
calculations), use the current price status helpers:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

status = prices.get_current_price_status()
status_long = prices.get_current_price_status_long()
status_dict = prices.get_current_price_status_dict()
status_g = prices.get_current_price_status(fund="G")

anchored_status = prices.get_current_price_status(as_of=date(2024, 1, 3))
```

## Fund Rankings

Rank funds by common performance metrics or trailing returns. Rankings include rank numbers,
metric metadata, and optional period or date-range context:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

ranked_trailing = prices.get_fund_rankings(metric="trailing_return", period=20, top_n=5)
ranked_cagr = prices.get_fund_rankings(metric="cagr")
ranked_volatility = prices.get_fund_rankings(metric="annualized_volatility", top_n=5)
ranked_subset = prices.get_fund_rankings(
    metric="trailing_return",
    period=20,
    funds=["C Fund", "S Fund"],
)
ranked_changes = prices.get_fund_rankings(metric="change_percent")
ranked_days_since = prices.get_fund_rankings(
    metric="days_since",
    reference_date=date(2024, 1, 10),
)
ranked_changes_as_of = prices.get_fund_rankings(
    metric="change",
    as_of=date(2024, 1, 3),
)

ranked_payload = prices.get_fund_rankings_dict(metric="trailing_return", period=20, top_n=5)
```

## Performance & Risk Summaries (Dictionary Output)

If you need JSON-friendly payloads (for example, an API response), use the `*_dict` summary
helpers. These convert missing numeric values to `None` and include basic metadata:

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

performance = prices.get_performance_summary_dict()
performance_range = prices.get_performance_summary_by_date_range_dict(
    start_date=date(2024, 1, 2),
    end_date=date(2024, 1, 4),
)
risk_summary = prices.get_risk_return_summary_dict(fund=TspIndividualFund.C_FUND)
```

## Long-Format Price History (Dictionary Output)

If you need JSON-friendly long-format payloads for dashboards or APIs, use the `*_dict`
helpers. These format dates and replace missing numeric values with `None` so the payloads
serialize cleanly:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

price_history = prices.get_price_history_long_dict(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
metrics_history = prices.get_price_history_with_metrics_dict(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
```

## Visualization-Ready Exports

If you're building dashboards or notebooks, the long-format helpers pair well with plotting
libraries that expect tidy data. Combine them with the Matplotlib chart helpers for quick
visuals or use them to feed Plotly/Altair:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

price_long = prices.get_price_history_long(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
returns_long = prices.get_price_history_with_metrics_long(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

fig, ax = prices.show_price_history_chart(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    show=False,
)
fig.savefig("price-history.png", dpi=150, bbox_inches="tight")
```

## Flexible Date Ranges for Price History

When you only know one end of a date range, you can supply a single bound. The missing bound
defaults to the earliest or latest date in the dataset:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

from_start = prices.get_price_history(start_date=date(2024, 1, 1))
to_end = prices.get_price_history(end_date=date(2024, 3, 31))
```

You can also request a single fund explicitly with the `fund` argument (use `funds` for a
collection). Passing both `fund` and `funds` raises a `ValueError` to avoid ambiguous filters:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
single_fund_history = prices.get_price_history(fund=TspIndividualFund.C_FUND)
single_fund_long = prices.get_price_history_long(fund=TspIndividualFund.C_FUND)
```

## Data Quality Report

Use `get_data_quality_report()` to consolidate the data summary, fund coverage, and missing
business-day checks (with optional cache metadata):

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
report = prices.get_data_quality_report()

scoped_report = prices.get_data_quality_report(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    include_cache_status=False,
)
```

If you need a JSON-friendly payload for APIs or dashboards, use the dictionary
helper. It formats dates and returns missing values as `None`:

```python
report_dict = prices.get_data_quality_report_dict()
cache_status = prices.get_cache_status_dict()
```

The data summary portion includes expected business-day counts, missing business-day totals, and
coverage ratios between the first and last available dates, which can be helpful for monitoring
data gaps at a glance.

## Current Prices & Snapshots

Use the current price helpers to retrieve latest prices, daily change metrics, and JSON-friendly
snapshots for dashboards or APIs:

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

# Latest prices for a single fund or subset
current_c = prices.get_current_prices(fund=TspIndividualFund.C_FUND)
current_subset = prices.get_current_prices(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# Current price report (daily change metrics included)
report = prices.get_current_price_report()
report_dict = prices.get_current_price_report_dict(include_data_quality=True)

# As-of prices for a point-in-time snapshot
prices_as_of = prices.get_prices_as_of(date(2024, 1, 15))
price_as_of = prices.get_price_as_of(TspIndividualFund.G_FUND, date(2024, 1, 15))
```

## As-of Price Changes

Use as-of helpers to anchor change calculations to a specific date (for example, when you want
to analyze the last trading day before a holiday):

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

changes_as_of = prices.get_price_changes_as_of(date(2024, 1, 3))
snapshot_as_of = prices.get_price_snapshot_as_of(date(2024, 1, 3), fund=TspIndividualFund.C_FUND)
snapshot_as_of_dict = prices.get_price_snapshot_as_of_dict(date(2024, 1, 3), fund=TspIndividualFund.C_FUND)
# As-of dicts include both the requested date and the resolved trading day.
as_of_prices_dict = prices.get_prices_as_of_dict(date(2024, 1, 3))
```

## Current Price Snapshots & Changes

Current price helpers are aliases for the latest available trading-day data. They are useful
when you prefer terminology like \"current\" in dashboards or APIs:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

current_changes = prices.get_current_price_changes()
current_changes_per_fund = prices.get_current_price_changes_per_fund()
current_snapshot = prices.get_current_price_snapshot()
current_snapshot_long = prices.get_current_price_snapshot_long()
current_snapshot_dict = prices.get_current_price_snapshot_dict()
```

## Latest Price Reports (Per Fund)

When you want to compare the most recent value per fund (even if some funds have missing
values in the latest CSV row), use the per-fund helpers. Each fund reports its own `as_of`
date so you can spot stale series:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

per_fund_latest = prices.get_latest_prices_per_fund()
per_fund_latest_long = prices.get_latest_prices_per_fund_long()
per_fund_latest_dict = prices.get_latest_prices_per_fund_dict()

per_fund_changes = prices.get_latest_price_changes_per_fund()
per_fund_report = prices.get_latest_price_report_per_fund()
```

## Fund Overview (Price Changes + Recency)

If you need a single table that blends per-fund price changes with recency metrics, use the
fund overview helpers. These combine the latest price report (per fund) with the days since
each fund’s most recent price:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

overview = prices.get_fund_overview()
overview_long = prices.get_fund_overview_long()
overview_dict = prices.get_fund_overview_dict()

# Anchor recency to a specific date (e.g., for backtesting dashboards)
overview_as_of = prices.get_fund_overview(reference_date=date(2024, 1, 3))
```

## Current Price Dashboard Snapshot

For a single table that combines current prices, recency, trailing returns, and risk metrics,
use the dashboard snapshot helper:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
dashboard = prices.get_current_price_dashboard(periods=[1, 5, 20, 63])
dashboard_dict = prices.get_current_price_dashboard_dict(periods=[1, 5, 20, 63])
```

## Fund Snapshot (Tidy Format)

Generate a snapshot of recent prices, trailing returns, and performance metrics in a
long (tidy) format for dashboarding:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
snapshot_long = prices.get_fund_snapshot_long(periods=[1, 5, 20, 63])
snapshot_subset = prices.get_fund_snapshot_long(
    funds=["C Fund", "S Fund"],
    periods=[1, 5, 20, 63],
)
```

Each fund uses its latest available price and as-of date, which is helpful when some funds have
missing values in the most recent row of the CSV. The snapshot includes risk metrics like the
Sortino ratio, value at risk (VaR), and expected shortfall. You can customize the minimum
acceptable return and confidence level:

```python
snapshot_long = prices.get_fund_snapshot_long(
    periods=[1, 5, 20, 63],
    mar=0.02,
    confidence=0.9,
)
```

## Fund Snapshot (Dictionary Format)

If you need a JSON-friendly payload for an API response or dashboard widget, use the
dictionary helper. Any missing numeric values are returned as `None` so the payload
serializes cleanly to JSON:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
snapshot_dict = prices.get_fund_snapshot_dict(periods=[1, 5, 20, 63])
```

Each fund entry includes its own `as_of` date so you can detect when a fund's most recent
value trails the latest dataset row.

## Fund Rankings

Rank funds by performance metrics (total return, volatility, Sharpe ratio, CAGR) or trailing
returns. This is useful for quickly comparing leaders and laggards across the fund set. You
can also rank by current price change metrics or recency (days since the latest price).

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

# Rank by total return across the full history
rankings = prices.get_fund_rankings(metric="total_return")

# Rank by trailing 20-day returns, limiting to the top 5 funds
trailing_rank = prices.get_fund_rankings(metric="trailing_return", period=20, top_n=5)

# Rank by latest daily change percent
change_rank = prices.get_fund_rankings(metric="change_percent")

# Rank by days since the latest price (useful for identifying stale funds)
recency_rank = prices.get_fund_rankings(
    metric="days_since",
    reference_date=date(2024, 1, 10),
)

# Rank by annualized volatility within a specific date range
range_rank = prices.get_fund_rankings(
    metric="annualized_volatility",
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31),
    top_n=5,
)
```

For dashboard payloads, use the dictionary helper to keep the ordered rankings:

```python
rankings_dict = prices.get_fund_rankings_dict(metric="total_return")
```

### Long-format returns for visualization

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

returns_long = prices.get_daily_returns_long()
cumulative_long = prices.get_cumulative_returns_long()
```

## Risk Metrics

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

var = prices.get_value_at_risk(fund=TspIndividualFund.C_FUND, confidence=0.95)
expected_shortfall = prices.get_expected_shortfall(fund=TspIndividualFund.C_FUND, confidence=0.95)
drawdown = prices.get_drawdown_series(fund=TspIndividualFund.C_FUND)
```

## Normalized Prices

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

normalized = prices.get_normalized_prices(
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
    base_value=100.0,
)

normalized_long = prices.get_normalized_prices_long(
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
    base_value=100.0,
)
```

Normalized prices are rebased using each fund's first available (non-null) price in the selected
range. This keeps normalized series usable even if a fund has missing values at the start of the
window.

Funds with no usable prices in the selected range are excluded so the normalized output contains
only funds with data.

### Long-format price history

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

price_long = prices.get_price_history_long()
```

### Combined price and return metrics

To build a single tidy dataset with price, daily return, cumulative return, and normalized price:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
metrics_long = prices.get_price_history_with_metrics_long()
```

### Combined price and return metrics (wide format)

If you prefer a wide format with a MultiIndex of `(fund, metric)` columns:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
metrics_wide = prices.get_price_history_with_metrics(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
```

## Rolling Metrics

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

rolling_mean = prices.get_rolling_mean(fund=TspIndividualFund.C_FUND, window=20)
rolling_returns = prices.get_rolling_returns(fund=TspIndividualFund.C_FUND, window=20)
rolling_volatility = prices.get_rolling_volatility(
    fund=TspIndividualFund.C_FUND,
    window=20,
    trading_days=252,
)
rolling_performance = prices.get_rolling_performance_summary(
    fund=TspIndividualFund.C_FUND,
    window=63,
    trading_days=252,
)
rolling_performance_long = prices.get_rolling_performance_summary_long(
    fund=TspIndividualFund.C_FUND,
    window=63,
    trading_days=252,
)
rolling_performance_dict = prices.get_rolling_performance_summary_dict(
    fund=TspIndividualFund.C_FUND,
    window=63,
    trading_days=252,
)
rolling_sharpe = prices.get_rolling_sharpe_ratio(
    fund=TspIndividualFund.C_FUND,
    window=63,
    trading_days=252,
)
rolling_sortino = prices.get_rolling_sortino_ratio(
    fund=TspIndividualFund.C_FUND,
    window=63,
    trading_days=252,
    mar=0.0,
)

rolling_corr = prices.get_rolling_correlation(
    fund_a=TspIndividualFund.C_FUND,
    fund_b=TspIndividualFund.S_FUND,
    window=63,
)
```

## Portfolio Analytics

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {TspIndividualFund.C_FUND: 0.6, TspIndividualFund.S_FUND: 0.4}

portfolio_returns = prices.get_portfolio_returns(weights=weights)
portfolio_value = prices.get_portfolio_value_history(weights=weights, initial_value=10_000)
portfolio_summary = prices.get_portfolio_performance_summary(weights=weights)
```

### Monte Carlo Bootstrapping (Long-Horizon Outcomes)

Bootstrap daily portfolio returns to estimate a distribution of ending values:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {TspIndividualFund.C_FUND: 0.6, TspIndividualFund.S_FUND: 0.4}

bootstrapped = prices.get_portfolio_bootstrap_simulation(
    weights=weights,
    years=20,
    simulations=10_000,
    initial_value=100_000,
    random_state=42,
)
percentiles = bootstrapped["ending_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
print(percentiles)
```

### Sequence-of-Returns Risk (Retirement Drawdown)

Run a rolling sequence test on monthly returns to gauge retirement drawdown risk:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {TspIndividualFund.C_FUND: 0.5, TspIndividualFund.S_FUND: 0.3, TspIndividualFund.G_FUND: 0.2}

sequences = prices.get_portfolio_retirement_sequence_analysis(
    weights=weights,
    initial_value=750_000,
    annual_withdrawal=40_000,
    years=30,
)
success_rate = sequences["success"].mean()
worst_cases = sequences.nsmallest(5, "ending_value")[["start_date", "ending_value"]]
print(success_rate)
print(worst_cases)
```

For visualization recipes and plotting helpers, see [VISUALIZATION.md](VISUALIZATION.md).

```python
rolling_corr_matrix = prices.get_rolling_correlation_matrix(window=63)
rolling_corr_matrix_long = prices.get_rolling_correlation_matrix_long(window=63)

rolling_beta = prices.get_rolling_beta(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.S_FUND,
    window=63,
)

rolling_max_drawdown = prices.get_rolling_max_drawdown(
    fund=TspIndividualFund.C_FUND,
    window=252,
)
```

## Correlation Matrix Exports

If you want to build correlation heatmaps in libraries such as Seaborn or Plotly,
use the long-format helpers:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
correlation_long = prices.get_correlation_matrix_long()
rolling_corr_long = prices.get_rolling_correlation_matrix_long(window=63)
```

## Moving Averages

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

multi_fund_ma = prices.get_moving_average(
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    window=20,
)
multi_fund_ma_long = prices.get_moving_average_long(
    funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
    window=20,
)

moving_averages = prices.get_moving_averages(
    fund=TspIndividualFund.C_FUND,
    windows=[20, 50],
    method="simple",
)

prices.show_moving_average_chart(
    fund=TspIndividualFund.C_FUND,
    windows=[20, 50],
    method="exponential",
)
```

## Quick Dashboards & Visuals

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

# Snapshot of the latest prices and daily changes
snapshot = prices.get_latest_price_snapshot()

# Latest prices (aliases to the most recent trading day)
current_prices = prices.get_current_prices()
current_prices_long = prices.get_current_prices_long()
current_prices_dict = prices.get_current_prices_dict()

# Quick risk/return summary for all funds
risk_summary = prices.get_risk_return_summary()

# Visualize latest daily changes across funds
prices.show_latest_price_change_chart()

# Plot a single fund's drawdown series
prices.show_drawdown_chart(fund=TspIndividualFund.C_FUND)
```

## Data Coverage & Quality

To understand how complete the cached data is for each fund, use the coverage summary:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
coverage = prices.get_fund_coverage_summary()
print(coverage)
```

`get_available_funds()` only returns funds with at least one non-null price in the dataset.
If a fund column exists but is entirely empty, it is treated as unavailable so analytics and
visualizations do not include missing data.

You can also visualize coverage with a bar chart:

```python
prices.show_fund_coverage_chart()
```

### Missing Business Days

To identify gaps in the price history (for example, data gaps or missing business-day rows),
use `get_missing_business_days`:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
missing = prices.get_missing_business_days()
scoped_missing = prices.get_missing_business_days(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)
```

You can visualize missing business days as a timeline:

```python
prices.show_missing_business_days_chart()
```

## Tracking Error & Information Ratio

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

tracking_error = prices.get_tracking_error(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)

info_ratio = prices.get_information_ratio(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)

rolling_te = prices.get_rolling_tracking_error(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
    window=63,
)
```

## Portfolio Analytics

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
portfolio_risk = prices.get_portfolio_risk_return_summary(weights=weights)
portfolio_drawdown = prices.get_portfolio_drawdown_series(weights=weights)
```

## Rolling Returns

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
rolling_returns = prices.get_rolling_returns(
    fund=TspIndividualFund.C_FUND,
    window=20,
)
```

## Drawdowns & Performance Summary

```python
from tsp import TspAnalytics, TspLifecycleFund

prices = TspAnalytics()

max_drawdown = prices.get_max_drawdown(TspLifecycleFund.L_2040)
drawdown_series = prices.get_drawdown_series()
drawdown_long = prices.get_drawdown_series_long()
drawdown_summary = prices.get_drawdown_summary(TspLifecycleFund.L_2040)
summary = prices.get_performance_summary(TspLifecycleFund.L_2040)
```

## Fund Snapshot

Use `get_fund_snapshot` to capture the latest price, daily change, trailing returns, and
performance metrics in a single table. The output includes an `as_of` column with the latest
available date:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
snapshot = prices.get_fund_snapshot(periods=[1, 5, 20])
```

## Compound Annual Growth Rate (CAGR)

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

cagr = prices.get_cagr()
fund_cagr = prices.get_cagr(fund=TspIndividualFund.C_FUND)
```

### Performance Summary by Date Range

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

summary = prices.get_performance_summary_by_date_range(
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
```

## Price Statistics

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

stats = prices.get_price_statistics()
fund_stats = prices.get_price_statistics(fund=TspIndividualFund.G_FUND)
```

## Latest Price Snapshots (Wide and Long)

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

latest_wide = prices.get_latest_prices()
latest_long = prices.get_latest_prices_long()
latest_changes = prices.get_latest_price_changes()
latest_changes_long = prices.get_latest_price_changes_long()
latest_snapshot = prices.get_latest_price_snapshot()
latest_snapshot_long = prices.get_latest_price_snapshot_long()
latest_prices_dict = prices.get_latest_prices_dict()
latest_snapshot_dict = prices.get_latest_price_snapshot_dict()

# Most recent available prices on or before a specific date
as_of_prices = prices.get_prices_as_of(date(2024, 1, 2))
as_of_long = prices.get_prices_as_of_long(date(2024, 1, 2))

# Per-fund price changes as of a specific date (uses each fund's last two valid prices)
changes_as_of = prices.get_price_changes_as_of_per_fund(date(2024, 1, 3))
changes_as_of_long = prices.get_price_changes_as_of_per_fund_long(date(2024, 1, 3))
changes_as_of_dict = prices.get_price_changes_as_of_per_fund_dict(date(2024, 1, 3))
```

## Monthly Return Tables

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
table = prices.get_monthly_return_table(TspIndividualFund.C_FUND)
table_long = prices.get_monthly_return_table_long(TspIndividualFund.C_FUND)
table_dict = prices.get_monthly_return_table_dict(TspIndividualFund.C_FUND)
```

## Return Statistics

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

return_stats = prices.get_return_statistics()
fund_return_stats = prices.get_return_statistics(fund=TspIndividualFund.G_FUND)

range_return_stats = prices.get_return_statistics(
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
```

## Price Changes by Date Range

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

changes = prices.get_price_change_by_date_range(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)

g_change = prices.get_price_change_by_date_range(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
    fund=TspIndividualFund.G_FUND,
)

changes_long = prices.get_price_change_by_date_range_long(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)

changes_dict = prices.get_price_change_by_date_range_dict(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)
```

## Latest Price Changes Snapshot

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
latest_changes = prices.get_latest_price_changes()

latest_subset = prices.get_latest_price_changes(
    funds=["G Fund", "C Fund", "S Fund"],
)
```

## Correlations

```python
from tsp import TspAnalytics

prices = TspAnalytics()
correlation = prices.get_correlation_matrix()
```

## Return Distribution

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
histogram = prices.get_return_histogram(
    fund=TspIndividualFund.C_FUND,
    bins=40,
)
```

## Excess Returns vs. a Benchmark

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

excess = prices.get_excess_returns(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)
excess_long = prices.get_excess_returns_long(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)
```

## Risk Metrics

Use historical Value at Risk (VaR) and Expected Shortfall to understand downside risk:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

var = prices.get_value_at_risk(confidence=0.95)
fund_var = prices.get_value_at_risk(confidence=0.9, fund=TspIndividualFund.C_FUND)

expected_shortfall = prices.get_expected_shortfall(confidence=0.95)
fund_es = prices.get_expected_shortfall(confidence=0.9, fund=TspIndividualFund.C_FUND)
```

## Risk/Return Summary

Combine annualized return, volatility, Sharpe/Sortino ratios, max drawdown, Calmar ratio,
ulcer index, max drawdown duration, max drawdown recovery time, pain index/ratio, Omega ratio,
and downside risk metrics into a single snapshot:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()
summary = prices.get_risk_return_summary()
fund_summary = prices.get_risk_return_summary(fund=TspIndividualFund.C_FUND, confidence=0.9)
```

## Downside Risk-Adjusted Returns (Sortino)

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

sortino = prices.get_sortino_ratio()
fund_sortino = prices.get_sortino_ratio(fund=TspIndividualFund.C_FUND, mar=0.02)
```

## Beta

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

beta = prices.get_beta(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.S_FUND,
)
```

## Visualizations

All chart methods use Matplotlib and will open a window in a local environment:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

prices.show_fund_price_chart(TspIndividualFund.G_FUND)
prices.show_cumulative_returns_chart()
prices.show_correlation_heatmap()
prices.show_latest_price_change_chart()
prices.show_monthly_return_heatmap(TspIndividualFund.C_FUND)
prices.show_rolling_correlation_chart(TspIndividualFund.C_FUND, TspIndividualFund.S_FUND)
prices.show_rolling_beta_chart(TspIndividualFund.C_FUND, TspIndividualFund.S_FUND)
prices.show_daily_return_histogram(TspIndividualFund.C_FUND, bins=40)
prices.show_rolling_max_drawdown_chart(TspIndividualFund.C_FUND, window=252)
prices.show_rolling_sortino_ratio_chart(TspIndividualFund.C_FUND, window=63)
prices.show_risk_return_scatter()
prices.show_excess_returns_chart(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)
```

### Multi-Fund Price History

To compare multiple funds over time, you can retrieve price history and plot it in one chart:

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
prices.show_price_history_chart(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
```

## Custom Visualization Workflows

All analytics return `DataFrame` objects, so you can build custom visuals with any plotting
library you prefer.

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
history = prices.get_price_history(funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND])
history_long = history.melt(id_vars="Date", var_name="Fund", value_name="Price")

fig = px.line(history_long, x="Date", y="Price", color="Fund", title="TSP Fund Prices")
fig.show()
```
