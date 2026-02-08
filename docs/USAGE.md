# Usage Guide

## Initialization and Data Refresh

All fund-specific methods accept either fund enums or fund name strings (case-insensitive).
Aliases like `"G"`, `"g-fund"`, `"L2050"`, and `"Lifecycle 2050"` are supported.
Compact lifecycle aliases such as `"L2050fund"` are normalized as well.

```python
from datetime import date
from tsp import TspAnalytics

# Initializes the client and loads cached data if available.
prices = TspAnalytics()

# Force a refresh if needed.
prices.check()
```

By default, `TspAnalytics` stores the CSV file in `~/.cache/tsp/fund-price-history.csv`. It refreshes the file when the last known business day has passed and the configured update time has been reached. You can also set the `TSP_DATA_DIR` environment variable to override the cache location.

If you prefer to store cached data elsewhere (for example, a user-level cache directory), pass `data_dir`:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")
```

`data_dir` must be a directory path. If you pass a file path, `TspAnalytics` raises a `ValueError`.

To configure a global cache directory:

```bash
export TSP_DATA_DIR="$HOME/.cache/tsp"
```

You can adjust the update window by passing `time_hour` (defaults to 7:00 PM local time):

```python
from datetime import time
from tsp import TspAnalytics

prices = TspAnalytics(time_hour=time(hour=20))
```

If you need to tune network behavior, pass `request_timeout` (seconds) to the client:

```python
from tsp import TspAnalytics

prices = TspAnalytics(request_timeout=30.0)
```

If you need to override the default user agent string (for example, for internal proxies),
pass `user_agent`:

```python
from tsp import TspAnalytics

prices = TspAnalytics(user_agent="MyApp/1.0")
```

If you need to point at a different CSV source (for example, a staging URL or a local proxy),
pass `csv_url`:

```python
from tsp import TspAnalytics

prices = TspAnalytics(csv_url="https://example.com/fund-price-history.csv")
```

If you need to provide a preconfigured `requests.Session` (for example, to inject proxies,
custom retries, or custom TLS settings), pass `session`:

```python
import requests
from tsp import TspAnalytics

session = requests.Session()
session.headers.update({"X-My-Header": "example"})

prices = TspAnalytics(session=session)
```

If you need to control network access manually, disable auto updates and call `refresh()` when you
want to download new data:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.refresh()
```

`refresh()` protects the cache from regressions: if the downloaded CSV has an older latest date
than the cached file, the cache is kept and a warning is logged.

### Loading Custom Dataframes

You can inject custom price data directly with `load_dataframe`. The dataframe may include a
`Date` column or a date-like index (named `Date` or a datetime index):

```python
import pandas as pd
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics(auto_update=False)
dataframe = pd.DataFrame(
    {
        TspIndividualFund.G_FUND.value: [100.0, 101.0],
        TspIndividualFund.C_FUND.value: [200.0, 201.0],
    },
    index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
)
prices.load_dataframe(dataframe)
```

`load_dataframe` validates that at least one fund column is present and that no negative prices
appear in the data.

If you have raw CSV text (for example, downloaded in a separate system), you can load it with
`load_csv_text`. The loader performs the same CSV validation used for network downloads,
including checks for a `Date` header and known fund columns.

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.load_csv_text("Date,G Fund\n2024-01-02,100.0\n")
```

### Cache Status

To inspect cache metadata (file path, last update time, and available fund coverage) without
triggering a network update, use `get_cache_status()`:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
status = prices.get_cache_status()
print(status)
```

The cache status payload also includes:

- `cache_age_days`: days since the CSV file was last updated.
- `data_age_days`: days since the latest data point in the cache.
- `is_stale`: whether the cached data trails the most recent business day.
- `stale_by_days`: number of days the cache trails the most recent business day.
- `last_business_day`: most recent business day based on today's date.
- `data_start_date`: earliest available date in the cache.
- `data_end_date`: latest available date in the cache.
- `data_span_days`: number of days between the earliest and latest cached dates.
- `file_size_bytes`: file size of the cached CSV (when present).
- `dataframe_valid`: whether the cached CSV could be loaded into a valid dataframe.
- `validation_error`: a descriptive message when the cached CSV failed validation.

### Data Quality Report

To build a consolidated report with the data summary, fund coverage, missing business days,
and optional cache metadata:

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

For JSON-friendly output, use the dictionary helper:

```python
report_dict = prices.get_data_quality_report_dict()
cache_status = prices.get_cache_status_dict()
```

## Current Prices (Per-Fund As-of)

To retrieve per-fund prices anchored to a historical date (each fund uses its most recent
price on or before the requested date), use the `as_of` parameter on the per-fund helpers:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

per_fund_as_of = prices.get_current_prices_per_fund(as_of=date(2024, 1, 2))
per_fund_as_of_long = prices.get_current_prices_per_fund_long(as_of=date(2024, 1, 2))
per_fund_as_of_dict = prices.get_current_prices_per_fund_dict(as_of=date(2024, 1, 2))
```

## As-of Pricing Completeness

When you request as-of prices for multiple funds, the default behavior is to return the most
recent row with any requested fund data. If you need every requested fund to have a value on
the resolved date (for example, for aligned analytics or reporting), set `require_all_funds=True`:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
aligned = prices.get_current_prices(
    as_of=date(2024, 1, 2),
    funds=["G Fund", "C Fund"],
    require_all_funds=True,
)
aligned_dict = prices.get_prices_as_of_dict(
    date(2024, 1, 2),
    funds=["G Fund", "C Fund"],
    require_all_funds=True,
)
```

When you request `as_of` prices for a single fund, the helper skips rows where that fund's
price is missing, returning the most recent valid price on or before the requested date.

## Current Price Reports & Snapshots

Use the report and snapshot helpers when you need daily change metrics alongside
latest prices:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

# Combined latest prices + daily change metrics
report = prices.get_current_price_report()
report_dict = prices.get_current_price_report_dict(include_cache_status=True)

# Snapshot payload (single as-of date + changes)
snapshot = prices.get_current_price_snapshot()
snapshot_dict = prices.get_current_price_snapshot_dict()

# Alerts for stale prices or large moves
alerts = prices.get_current_price_alerts(stale_days=2, change_threshold=0.03)
alerts_dict = prices.get_current_price_alerts_dict(stale_days=2, change_threshold=0.03)
```

## Analytics & Visualization Quickstart

The analytics helpers return pandas dataframes, and chart helpers return Matplotlib figures so
you can save or embed them in dashboards:

```python
from datetime import date

from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

# Analytics helpers
daily_returns = prices.get_daily_returns()
trailing_20 = prices.get_trailing_returns(periods=20, fund=TspIndividualFund.C_FUND)
performance = prices.get_performance_summary()
correlation = prices.get_correlation_matrix()
correlation_payload = prices.get_correlation_matrix_dict()

# Long-format data (great for Seaborn/Plotly)
history_long = prices.get_price_history_long(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
metrics_long = prices.get_price_history_with_metrics_long(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# Chart helpers (pass show=False to avoid opening a window)
fig, ax = prices.show_fund_price_chart(TspIndividualFund.C_FUND, show=False)
fig.savefig("c-fund.png", dpi=150, bbox_inches="tight")

fig, ax = prices.show_correlation_heatmap(show=False)
fig.savefig("correlation-heatmap.png", dpi=150, bbox_inches="tight")

fig, ax = prices.show_portfolio_value_chart(
    weights={TspIndividualFund.C_FUND: 0.6, TspIndividualFund.S_FUND: 0.4},
    start_date=date(2022, 1, 1),
    show=False,
)
fig.savefig("portfolio.png", dpi=150, bbox_inches="tight")
```

## Loading Custom Data

If you have a dataframe or a CSV file with TSP prices (for example, from a previous export),
you can load it directly for analytics without relying on the network cache:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)

# Load a pandas dataframe
prices.load_dataframe(my_dataframe)

# Or load from a CSV file
prices.load_csv(Path("fund-price-history.csv"))

# Or load CSV content that you fetched elsewhere (string or bytes)
csv_text = "Date,G Fund,C Fund\n2024-01-02,100.0,200.0\n"
prices.load_csv_text(csv_text)

csv_bytes = csv_text.encode("utf-8")
prices.load_csv_text(csv_bytes)
```

Loaded data is normalized and validated the same way as cached data (including column alias
normalization and date parsing). If validation fails, a `ValueError` or `TypeError` is raised.
Dates are normalized to midnight with timezone info stripped so date-based queries behave
consistently across sources.

Price values must be non-negative. If any fund price is negative in the supplied data, the
load helpers raise a `ValueError` to prevent invalid analytics.

## Fetching Prices

```python
from datetime import date
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

# Latest price for a single fund
latest = prices.get_price(TspIndividualFund.C_FUND)

# Latest prices for a subset of funds
latest_subset = prices.get_latest_prices(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# "Current" is an alias for the latest available trading-day prices.
current_subset = prices.get_current_prices(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# If you need per-fund latest prices (each fund uses its own latest date), set per_fund=True.
current_per_fund = prices.get_current_prices(per_fund=True)

# You can also pass fund names directly as strings.
latest_strings = prices.get_latest_prices(funds="G Fund")
latest_aliases = prices.get_latest_prices(funds="g-fund")
latest_single_letter = prices.get_latest_prices(funds="G")
latest_lifecycle = prices.get_latest_prices(funds="L2050")

# Discover normalized aliases for each fund (useful for UI validation)
aliases = prices.get_fund_aliases()
g_aliases = aliases[TspIndividualFund.G_FUND.value]

# Fund metadata with category, aliases, and availability (based on loaded data)
metadata = prices.get_fund_metadata()
metadata_dict = prices.get_fund_metadata_dict()

# Latest prices in long (tidy) format
latest_long = prices.get_latest_prices_long()

# Per-fund latest price changes (each fund uses its last two valid prices)
per_fund_changes = prices.get_latest_price_changes_per_fund()
per_fund_changes_dict = prices.get_latest_price_changes_per_fund_dict()

# Price history with embedded return metrics (wide format)
metrics_wide = prices.get_price_history_with_metrics(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# JSON-friendly latest prices and snapshots (handy for APIs)
latest_dict = prices.get_latest_prices_dict()
snapshot_dict = prices.get_latest_price_snapshot_dict()
current_dict = prices.get_current_prices_dict()

# Combined price + change report as a dataframe
report = prices.get_current_price_report()

# The default output formats the as_of date as an ISO 8601 string.
latest_dict = prices.get_latest_prices_dict(date_format="iso")
snapshot_dict = prices.get_latest_price_snapshot_dict(date_format="%Y-%m-%d")

# If you need a date object, pass date_format=None.
latest_dict = prices.get_latest_prices_dict(date_format=None)

# Most recent available prices on or before a specific date
as_of_prices = prices.get_prices_as_of(date(2024, 1, 2))
as_of_prices_long = prices.get_prices_as_of_long(date(2024, 1, 2))
# Dictionary output includes the requested date and the resolved trading-day date.
as_of_prices_dict = prices.get_prices_as_of_dict(date(2024, 1, 2))
as_of_single_price = prices.get_price_as_of(
    fund=TspIndividualFund.C_FUND,
    as_of=date(2024, 1, 2),
)

# Per-fund as-of prices (each fund uses its own last valid date <= as_of)
as_of_per_fund = prices.get_prices_as_of_per_fund(date(2024, 1, 2))
as_of_per_fund_long = prices.get_prices_as_of_per_fund_long(date(2024, 1, 2))
as_of_per_fund_dict = prices.get_prices_as_of_per_fund_dict(date(2024, 1, 2))

# Per-fund price changes as of a specific date (each fund uses its last two valid prices)
as_of_changes = prices.get_price_changes_as_of_per_fund(date(2024, 1, 3))
as_of_changes_long = prices.get_price_changes_as_of_per_fund_long(date(2024, 1, 3))
as_of_changes_dict = prices.get_price_changes_as_of_per_fund_dict(date(2024, 1, 3))

# Daily change metrics as of a specific date
changes_as_of = prices.get_price_changes_as_of(date(2024, 1, 2))
changes_as_of_long = prices.get_price_changes_as_of_long(date(2024, 1, 2))
changes_as_of_dict = prices.get_price_changes_as_of_dict(
    date(2024, 1, 2),
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)

# Prices for a date range
range_df = prices.get_prices_by_date_range(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 3, 31),
)

# A single fund's price on a specific day
single_price = prices.get_fund_price_by_date(
    fund=TspIndividualFund.C_FUND,
    date=date(2024, 1, 15),
)
```

When requesting latest prices or price changes, provide either `fund` or `funds` (not both). Passing both arguments raises a `ValueError`. Fund name strings are case-insensitive and extra whitespace is ignored.

All fund-specific APIs accept either fund enums (`TspIndividualFund`/`TspLifecycleFund`) or fund
name strings such as `"G Fund"` and `" l 2040 "`, so you can mix and match based on your
application inputs. Aliases like `"G"`, `"g-fund"`, `"g_fund"`, and `"L2050"` are accepted too.
Names like `"L Income Fund"` are also normalized to the lifecycle income fund.

If you need the most recent trading-day prices on or before a given date (for example, if the
date is a weekend or holiday), use `get_prices_as_of()`, `get_prices_as_of_long()`, or
`get_prices_as_of_dict()`.

`get_available_funds()` only returns funds that have at least one non-null price in the
dataset. If a fund column exists but is entirely empty, it is treated as unavailable so
analytics and visualizations do not silently include missing data.

## Monthly & Yearly Data

Monthly and yearly helpers validate that the year is a positive integer and the month is between
1 and 12:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

monthly_prices = prices.get_prices_by_month(2024, 1)
yearly_prices = prices.get_prices_by_year(2024)
fund_monthly = prices.get_fund_prices_by_month(TspIndividualFund.G_FUND, 2024, 1)
fund_yearly = prices.get_fund_prices_by_year(TspIndividualFund.G_FUND, 2024)
```

## Monthly Return Tables

For a quick seasonality view, create a month-by-month return table for a single fund:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
table = prices.get_monthly_return_table(TspIndividualFund.C_FUND)
print(table)

# Fund names can be strings too
string_table = prices.get_monthly_return_table("C Fund")

# Long-format tables are handy for dashboards and APIs
table_long = prices.get_monthly_return_table_long(TspIndividualFund.C_FUND)
table_dict = prices.get_monthly_return_table_dict(TspIndividualFund.C_FUND)
```

You can also visualize the table as a heatmap:

```python
prices.show_monthly_return_heatmap(TspIndividualFund.C_FUND)
```

## Data Coverage & Fund Availability

```python
from tsp import TspAnalytics

prices = TspAnalytics()

funds = prices.get_available_funds()
summary = prices.get_data_summary()
```

When a fund is missing from the cached dataset, fund-specific APIs raise a `ValueError`. Use
`get_available_funds()` or the `available_funds` entry in `get_data_summary()` to confirm coverage.
The data summary also includes expected business days, missing business days, and a coverage ratio
between the first and last available dates to help spot gaps quickly.

Chart helpers that target a fund group (for example, lifecycle funds) raise a `ValueError` if the
underlying data does not contain any columns for that group.

### Fund Coverage Summary

Use `get_fund_coverage_summary()` to understand how much data is available for each fund:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
coverage = prices.get_fund_coverage_summary()
print(coverage)
```

To visualize the coverage directly:

```python
prices.show_fund_coverage_chart()
```

## Data Normalization Notes

During normalization, the CSV loader trims column whitespace, standardizes fund names, converts
prices to numeric values, and parses dates. Common formats such as `YYYY-MM-DD` and `MM/DD/YYYY`
are supported. Any rows without valid dates or prices are removed so downstream analytics remain
consistent.

## Exporting Data for External Analytics

Every analytics helper returns a Pandas `DataFrame` or `Series`, which you can export or use
in other tooling:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics()

history = prices.get_price_history()
history.to_csv(Path("tsp-analytics-history.csv"), index=False)

stats = prices.get_return_statistics()
stats.to_excel(Path("tsp-return-stats.xlsx"))
```

### Long-format data for visualization libraries

If you need tidy data for visualization libraries like Seaborn or Altair, use the long-format
helpers:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

price_long = prices.get_price_history_long()
returns_long = prices.get_daily_returns_long()
cumulative_long = prices.get_cumulative_returns_long()
normalized_long = prices.get_normalized_prices_long()
drawdown_long = prices.get_drawdown_series_long()

# Combined price/return metrics in one tidy dataset
metrics_long = prices.get_price_history_with_metrics_long()
```

## Portfolio Allocation

```python
from tsp import TspAnalytics

prices = TspAnalytics()

allocation = prices.create_allocation(
    g_shares=10,
    f_shares=5,
    c_shares=20,
)

print(allocation["total"])        # Total dollar value
print(allocation["allocation_percent"])  # Percent allocation
```

You can also pass a mapping of fund names or enums to shares:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

allocation = prices.create_allocation_from_shares(
    {
        TspIndividualFund.G_FUND: 10,
        "C Fund": 20,
    }
)
```

If a fund is missing from the cached CSV and the share count is zero, the allocation output
includes the fund with a `None` price and `0.00` subtotal. Non-zero share counts for missing
funds raise a `ValueError`.

Use `show_pie_chart` to visualize the allocation:

```python
prices.show_pie_chart(allocation)
```

## Portfolio Analytics

You can build portfolio analytics by supplying fund weights. The helpers normalize weights by
default and return Pandas `DataFrame` objects for downstream analysis:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

weights = {
    TspIndividualFund.C_FUND: 0.6,
    TspIndividualFund.S_FUND: 0.4,
}

portfolio_returns = prices.get_portfolio_returns(weights=weights)
portfolio_value = prices.get_portfolio_value_history(weights=weights, initial_value=10_000)
portfolio_summary = prices.get_portfolio_performance_summary(weights=weights)

prices.show_portfolio_value_chart(weights=weights, initial_value=10_000)
```

## Latest Price Changes

```python
from tsp import TspAnalytics

prices = TspAnalytics()

changes = prices.get_latest_price_changes()
print(changes)

subset_changes = prices.get_latest_price_changes(
    funds=["G Fund", "C Fund"],
)
print(subset_changes)

snapshot = prices.get_latest_price_snapshot()
snapshot_long = prices.get_latest_price_snapshot_long()

# Price changes anchored to a specific date (uses most recent trading day on or before)
changes_as_of = prices.get_price_changes_as_of(date(2024, 1, 3))
changes_as_of_long = prices.get_price_changes_as_of_long(date(2024, 1, 3))

snapshot_as_of = prices.get_price_snapshot_as_of(date(2024, 1, 3))
snapshot_as_of_long = prices.get_price_snapshot_as_of_long(date(2024, 1, 3))
snapshot_as_of_dict = prices.get_price_snapshot_as_of_dict(date(2024, 1, 3))
```

## Return Statistics

```python
from datetime import date

from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

stats = prices.get_return_statistics()
fund_stats = prices.get_return_statistics(fund=TspIndividualFund.C_FUND)

# Scope the statistics window (if only one bound is provided, the other defaults)
range_stats = prices.get_return_statistics(start_date=date(2024, 1, 1))
range_stats_through = prices.get_return_statistics(end_date=date(2024, 3, 31))
```

## Risk Metrics (VaR & Expected Shortfall)

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

var = prices.get_value_at_risk(confidence=0.95)
fund_var = prices.get_value_at_risk(confidence=0.9, fund=TspIndividualFund.C_FUND)

expected_shortfall = prices.get_expected_shortfall(confidence=0.95)
fund_es = prices.get_expected_shortfall(confidence=0.9, fund=TspIndividualFund.C_FUND)
```

## Risk/Return Summary

Use the risk/return summary for a combined view of annualized return, volatility, Sharpe/Sortino
ratios, max drawdown, and downside risk metrics:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

summary = prices.get_risk_return_summary()
fund_summary = prices.get_risk_return_summary(
    fund=TspIndividualFund.C_FUND,
    confidence=0.9,
)
```

## Excess Returns vs. a Benchmark Fund

Track relative performance by subtracting a benchmark fund's daily return from the target fund:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

excess = prices.get_excess_returns(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)
excess_long = prices.get_excess_returns_long(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)

prices.show_excess_returns_chart(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.G_FUND,
)
```

## Sortino Ratio (Downside Risk Adjusted)

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

sortino = prices.get_sortino_ratio()
fund_sortino = prices.get_sortino_ratio(fund=TspIndividualFund.C_FUND, mar=0.02)
```

## Logging

The library uses a module-level logger (`tsp.tsp`). Configure logging in your application to
see debug or info output.

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Snapshot Analytics

For a quick summary of performance and trailing returns:

```python
from tsp import TspAnalytics, TspIndividualFund

prices = TspAnalytics()

snapshot = prices.get_fund_snapshot(periods=[1, 5, 20])
# The snapshot includes an `as_of` column with the latest data date.
single_fund = prices.get_fund_snapshot(
    fund=TspIndividualFund.G_FUND,
    periods=[1, 20],
)
snapshot_risk = prices.get_fund_snapshot(
    periods=[1, 5, 20],
    mar=0.02,
    confidence=0.9,
)
```

## Rolling Correlation & Return Distributions

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

rolling_corr = prices.get_rolling_correlation(
    fund_a=TspIndividualFund.C_FUND,
    fund_b=TspIndividualFund.S_FUND,
    window=63,
)

prices.show_rolling_correlation_heatmap(window=63)

histogram = prices.get_return_histogram(
    fund=TspIndividualFund.C_FUND,
    bins=40,
)

rolling_drawdown = prices.get_rolling_max_drawdown(
    fund=TspIndividualFund.C_FUND,
    window=252,
)
```

## Moving Averages

Use simple or exponential moving averages for trend analysis:

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

## Benchmarking with Beta

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

beta = prices.get_beta(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.S_FUND,
)

rolling_beta = prices.get_rolling_beta(
    fund=TspIndividualFund.C_FUND,
    benchmark=TspIndividualFund.S_FUND,
    window=63,
)
```

## Price History for Selected Funds

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
print(history.head())
```

To visualize the selected funds, use `show_price_history_chart`:

```python
prices.show_price_history_chart(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)
```

You can also provide a single date bound when you want everything up to (or after) a date:

```python
history_from_start = prices.get_price_history(start_date=date(2020, 1, 1))
history_to_end = prices.get_price_history(end_date=date(2024, 1, 1))
```

## Price Change Over a Date Range

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
changes = prices.get_price_change_by_date_range(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)
print(changes)

changes_long = prices.get_price_change_by_date_range_long(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)
print(changes_long)

changes_dict = prices.get_price_change_by_date_range_dict(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)
print(changes_dict["funds"]["G Fund"])
```
