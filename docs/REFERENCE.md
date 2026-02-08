# API Reference

This reference summarizes the core APIs available in `TspAnalytics`. See `docs/USAGE.md` and `docs/ANALYTICS.md` for step-by-step examples.

For full docstring output, build the Sphinx docs in `docs/sphinx/` or generate a
single-file HTML page with `pydoc`.

## Initialization Options

Common `TspAnalytics` constructor arguments:

- `data_dir`: cache directory for the downloaded CSV.
- `auto_update`: enable/disable automatic refresh checks.
- `time_hour`: refresh window start time (local time).
- `request_timeout`: network timeout for CSV downloads.
- `max_retries`: retry count for CSV downloads.
- `retry_backoff`: base retry delay in seconds (exponential backoff).
- `csv_url`: override the default TSP CSV URL.
- `user_agent`: override the default HTTP user agent string for downloads.
- `session`: pass a pre-configured `requests.Session` for custom networking.

## Core Data Access

| Method | Purpose |
| --- | --- |
| `refresh()` | Forces a download of the latest CSV data. |
| `load_dataframe(dataframe)` | Load and normalize a pandas dataframe as the active price dataset. Accepts a `Date` column or a date-like index. |
| `load_csv(filepath)` | Load and normalize a CSV file as the active price dataset. |
| `load_csv_text(csv_content)` | Load and normalize CSV content provided as a string. |
| `get_cache_status()` | Cache file path, update time, file metadata, validation state, age/staleness indicators, data range (`data_start_date`, `data_end_date`, `data_span_days`), and dataset coverage (no network). |
| `get_cache_status_dict(date_format="iso", datetime_format="iso")` | JSON-friendly cache status payload with formatted dates and datetimes. |
| `get_available_funds()` | List of fund columns available in the cached data. |
| `get_fund_aliases()` | Mapping of canonical fund names to normalized alias strings. |
| `get_fund_metadata(include_aliases=True, include_availability=True)` | Fund metadata with category, alias list, and availability indicators. |
| `get_fund_metadata_dict(include_aliases=True, include_availability=True)` | JSON-friendly fund metadata payload keyed by fund name. |
| `get_data_summary()` | Summary of available data range, fund coverage, and business-day coverage. |
| `get_data_summary_dict(date_format="iso")` | JSON-friendly summary of available data range, fund coverage, and business-day coverage. |
| `get_fund_coverage_summary()` | Coverage statistics per fund (rows, missing rows, coverage percent). |
| `get_fund_coverage_summary_dict(date_format="iso")` | JSON-friendly coverage summaries per fund. |
| `get_missing_business_days(start_date=None, end_date=None)` | Missing business-day dates within the dataset. |
| `get_missing_business_days_dict(start_date=None, end_date=None, date_format="iso")` | Missing business-day dates as JSON-friendly records. |
| `get_data_quality_report_dict(start_date=None, end_date=None, include_cache_status=True, date_format="iso", datetime_format="iso")` | JSON-friendly data-quality report (summary, coverage, missing days, optional cache status). |
| `get_price(fund)` | Latest price for a single fund. |
| `get_prices_by_date(date)` | Prices across all funds on a given date. |
| `get_prices_by_date_range(start_date, end_date)` | Prices for all funds between dates. |
| `get_fund_price_by_date(fund, date)` | Single fund price on a date. |
| `get_fund_prices_by_date_range(fund, start_date, end_date)` | Fund prices between dates. |
| `get_latest_prices(fund=None, funds=None)` | Latest prices (optionally filtered to one fund or a list). |
| `get_latest_prices_long(fund=None, funds=None)` | Latest prices in long (tidy) format. |
| `get_latest_prices_dict(fund=None, funds=None, date_format="iso")` | Latest prices as a JSON-friendly dictionary. |
| `get_current_prices(fund=None, funds=None, as_of=None, per_fund=False, allow_missing=False, require_all_funds=False)` | Alias for latest prices, optionally anchored to an as-of date or returned per fund (with optional missing-fund skipping and aligned-fund enforcement). |
| `get_current_prices_long(fund=None, funds=None, as_of=None, per_fund=False, allow_missing=False, require_all_funds=False)` | Latest prices in long format, optionally anchored to an as-of date or returned per fund (with optional missing-fund skipping and aligned-fund enforcement). |
| `get_current_prices_dict(fund=None, funds=None, date_format="iso", as_of=None, per_fund=False, allow_missing=False, require_all_funds=False)` | JSON-friendly latest prices, optionally anchored to an as-of date or returned per fund (with optional missing-fund skipping and aligned-fund enforcement). |
| `get_latest_prices_per_fund(funds=None, allow_missing=False)` | Latest price per fund using each fund's last valid date (optionally skipping missing). |
| `get_latest_prices_per_fund_long(funds=None, allow_missing=False)` | Latest price per fund in long (tidy) format (optionally skipping missing). |
| `get_latest_prices_per_fund_dict(funds=None, date_format="iso", allow_missing=False)` | Latest price per fund as a JSON-friendly dictionary (optionally skipping missing). |
| `get_current_prices_per_fund(funds=None, as_of=None, allow_missing=False)` | Alias for `get_latest_prices_per_fund` (optionally anchored to an as-of date and skipping missing). |
| `get_current_prices_per_fund_long(funds=None, as_of=None, allow_missing=False)` | Alias for `get_latest_prices_per_fund_long` (optionally anchored to an as-of date and skipping missing). |
| `get_current_prices_per_fund_dict(funds=None, date_format="iso", as_of=None, allow_missing=False)` | Alias for `get_latest_prices_per_fund_dict` (optionally anchored to an as-of date and skipping missing). |
| `get_price_recency(funds=None, reference_date=None)` | Days since each fund's most recent available price. |
| `get_price_recency_long(funds=None, reference_date=None)` | Price recency in long (tidy) format. |
| `get_price_recency_dict(funds=None, reference_date=None, date_format="iso")` | Price recency as a JSON-friendly dictionary. |
| `get_current_price_status(funds=None, as_of=None, reference_date=None)` | Latest per-fund prices with recency metrics (as-of date, price, days_since). |
| `get_current_price_status_long(funds=None, as_of=None, reference_date=None)` | Current price status in long (tidy) format. |
| `get_current_price_status_dict(funds=None, as_of=None, reference_date=None, date_format="iso")` | Current price status as a JSON-friendly dictionary. |
| `get_current_price_summary(funds=None, as_of=None, reference_date=None, stale_days=3)` | Summary of recency metrics and daily change stats across funds (optionally anchored to an as-of date). |
| `get_current_price_summary_dict(funds=None, reference_date=None, stale_days=3, date_format="iso", as_of=None)` | JSON-friendly summary of recency metrics and daily change stats (optionally anchored to an as-of date). |
| `get_current_price_alerts(funds=None, as_of=None, reference_date=None, stale_days=3, change_threshold=0.02)` | Per-fund alerts for stale prices and large daily moves. |
| `get_current_price_alerts_long(funds=None, as_of=None, reference_date=None, stale_days=3, change_threshold=0.02)` | Current price alerts in long (tidy) format. |
| `get_current_price_alerts_dict(funds=None, as_of=None, reference_date=None, stale_days=3, change_threshold=0.02, date_format="iso")` | JSON-friendly current price alerts payload. |
| `get_current_price_alert_summary(funds=None, as_of=None, reference_date=None, stale_days=3, change_threshold=0.02)` | Summary counts for stale prices and large daily moves. |
| `get_current_price_alert_summary_dict(funds=None, as_of=None, reference_date=None, stale_days=3, change_threshold=0.02, date_format="iso")` | JSON-friendly summary of current price alert counts. |
| `get_prices_as_of(as_of, fund=None, funds=None, require_all_funds=False)` | Most recent prices on or before a specific date, optionally requiring all requested funds to have data on the resolved date. |
| `get_price_as_of(fund, as_of)` | Most recent price for a single fund on or before a date. |
| `get_prices_as_of_long(as_of, fund=None, funds=None, require_all_funds=False)` | As-of prices in long (tidy) format, optionally requiring all requested funds to have data on the resolved date. |
| `get_prices_as_of_dict(as_of, fund=None, funds=None, date_format="iso", require_all_funds=False)` | As-of prices as a JSON-friendly dictionary (includes requested_as_of and resolved as_of dates), with optional aligned-fund enforcement. |
| `get_prices_as_of_per_fund(as_of, funds=None, allow_missing=False)` | Most recent price per fund on or before a specific date (optionally skipping missing). |
| `get_prices_as_of_per_fund_long(as_of, funds=None, allow_missing=False)` | As-of per-fund prices in long (tidy) format (optionally skipping missing). |
| `get_prices_as_of_per_fund_dict(as_of, funds=None, date_format="iso", allow_missing=False)` | As-of per-fund prices as a JSON-friendly dictionary (optionally skipping missing). |
| `get_price_history(fund=None, funds=None, start_date=None, end_date=None)` | Price history for selected funds. |
| `get_price_history_long(fund=None, funds=None, start_date=None, end_date=None)` | Price history in long (tidy) format. |
| `get_price_history_long_dict(fund=None, funds=None, start_date=None, end_date=None, date_format="iso")` | Price history in JSON-friendly long-format records. |
| `get_recent_prices(days=5, fund=None, funds=None, as_of=None)` | Most recent trading-day prices (optionally anchored to an as-of date). |
| `get_recent_prices_long(days=5, fund=None, funds=None, as_of=None)` | Recent trading-day prices in long (tidy) format. |
| `get_recent_prices_dict(days=5, fund=None, funds=None, as_of=None, date_format="iso")` | Recent trading-day prices in JSON-friendly long-format records. |
| `get_moving_average(fund=None, funds=None, window=20, start_date=None, end_date=None)` | Rolling moving averages for selected funds (single window). |
| `get_moving_average_long(fund=None, funds=None, window=20, start_date=None, end_date=None)` | Rolling moving averages in long (tidy) format. |
| `get_price_history_with_metrics(fund=None, funds=None, start_date=None, end_date=None, base_value=100.0)` | Wide-format price history with returns, cumulative returns, and normalized prices. |
| `get_price_history_with_metrics_long(fund=None, funds=None, start_date=None, end_date=None, base_value=100.0)` | Long-format price history with returns, cumulative returns, and normalized prices. |
| `get_price_history_with_metrics_dict(fund=None, funds=None, start_date=None, end_date=None, base_value=100.0, date_format="iso")` | Metrics-rich price history as JSON-friendly long-format records. |
| `get_price_summary(funds=None)` | Price history summary statistics per fund (first/last date, min/max, totals). |
| `get_price_summary_dict(funds=None, date_format="iso")` | Price summary statistics as a JSON-friendly dictionary. |
| `get_latest_price_changes(fund=None, funds=None)` | Latest price and percent change vs. the prior trading day. |
| `get_latest_price_changes_long(fund=None, funds=None)` | Latest price changes with an explicit fund column. |
| `get_current_price_changes(fund=None, funds=None, as_of=None)` | Latest price changes, optionally anchored to an as-of date. |
| `get_current_price_changes_long(fund=None, funds=None, as_of=None)` | Current price changes in long (tidy) format with optional as-of anchoring. |
| `get_current_price_changes_dict(fund=None, funds=None, date_format="iso", as_of=None)` | JSON-friendly current price changes, optionally anchored to an as-of date. |
| `get_current_price_changes_per_fund(funds=None, as_of=None)` | Per-fund latest price changes using each fund's last two valid prices. |
| `get_current_price_changes_per_fund_long(funds=None, as_of=None)` | Per-fund latest price changes in long (tidy) format. |
| `get_current_price_changes_per_fund_dict(funds=None, date_format="iso", as_of=None)` | Per-fund latest price changes as a JSON-friendly dictionary. |
| `get_recent_price_changes(days=5, fund=None, funds=None, as_of=None)` | Daily percent changes for the most recent trading days. |
| `get_recent_price_changes_long(days=5, fund=None, funds=None, as_of=None)` | Recent daily percent changes in long (tidy) format. |
| `get_recent_price_changes_dict(days=5, fund=None, funds=None, as_of=None, date_format="iso")` | Recent daily percent changes as a JSON-friendly dictionary. |
| `get_recent_price_change_summary(days=5, fund=None, funds=None, as_of=None)` | Summary statistics for recent daily percent changes. |
| `get_recent_price_change_summary_dict(days=5, fund=None, funds=None, as_of=None, date_format="iso")` | Recent daily change summary as a JSON-friendly dictionary. |
| `get_latest_price_report_dict(fund=None, funds=None, date_format="iso")` | Combined latest prices and daily changes as a JSON-friendly dictionary. |
| `get_latest_price_report_long(fund=None, funds=None)` | Latest price report in long (tidy) format. |
| `get_current_price_report_dict(fund=None, funds=None, date_format="iso", as_of=None)` | Current price report dictionary, with optional as-of anchoring. |
| `get_current_price_report_long(fund=None, funds=None, as_of=None)` | Current price report in long (tidy) format, with optional as-of anchoring. |
| `get_latest_price_report_per_fund(funds=None)` | Per-fund latest price report using each fund's last two valid prices. |
| `get_latest_price_report_per_fund_long(funds=None)` | Per-fund latest price report in long (tidy) format. |
| `get_latest_price_report_per_fund_dict(funds=None, date_format="iso")` | Per-fund latest price report as a JSON-friendly dictionary. |
| `get_fund_overview(funds=None, reference_date=None)` | Per-fund overview combining price changes with recency metrics. |
| `get_fund_overview_long(funds=None, reference_date=None)` | Per-fund overview in long (tidy) format. |
| `get_fund_overview_dict(funds=None, reference_date=None, date_format="iso")` | Per-fund overview as a JSON-friendly dictionary. |
| `get_current_price_report_per_fund(funds=None, as_of=None)` | Current per-fund price report with optional as-of anchoring. |
| `get_current_price_report_per_fund_long(funds=None, as_of=None)` | Current per-fund price report in long format, with optional as-of anchoring. |
| `get_current_price_report_per_fund_dict(funds=None, date_format="iso", as_of=None)` | Current per-fund price report dictionary, with optional as-of anchoring. |
| `get_current_price_snapshot(fund=None, funds=None, as_of=None)` | Current price snapshot (latest/previous prices and changes), optionally anchored to an as-of date. |
| `get_current_price_snapshot_long(fund=None, funds=None, as_of=None)` | Current price snapshot in long (tidy) format, with optional as-of anchoring. |
| `get_current_price_snapshot_dict(fund=None, funds=None, date_format="iso", as_of=None)` | Current price snapshot as a JSON-friendly dictionary, with optional as-of anchoring. |
| `get_latest_price_snapshot(fund=None, funds=None)` | Latest prices with as-of date and daily change metrics. |
| `get_latest_price_snapshot_long(fund=None, funds=None)` | Latest price snapshot with an explicit fund column. |
| `get_latest_price_snapshot_dict(fund=None, funds=None, date_format="iso")` | Latest price snapshot as a JSON-friendly dictionary. |
| `get_price_changes_as_of(as_of, fund=None, funds=None)` | Price changes anchored to a specific date. |
| `get_price_changes_as_of_long(as_of, fund=None, funds=None)` | As-of price changes in long (tidy) format. |
| `get_price_changes_as_of_dict(as_of, fund=None, funds=None, date_format="iso")` | As-of price changes as a JSON-friendly dictionary. |
| `get_price_changes_as_of_per_fund(as_of, funds=None)` | As-of price changes using each fund's last two valid prices. |
| `get_price_changes_as_of_per_fund_long(as_of, funds=None)` | As-of per-fund price changes in long (tidy) format. |
| `get_price_changes_as_of_per_fund_dict(as_of, funds=None, date_format="iso")` | As-of per-fund price changes as a JSON-friendly dictionary. |
| `get_price_snapshot_as_of(as_of, fund=None, funds=None)` | Price snapshot with as-of date and daily changes. |
| `get_price_snapshot_as_of_long(as_of, fund=None, funds=None)` | As-of price snapshot in long (tidy) format. |
| `get_price_snapshot_as_of_dict(as_of, fund=None, funds=None, date_format="iso")` | As-of price snapshot as a JSON-friendly dictionary. |
| `get_price_change_by_date_range(start_date, end_date, fund=None)` | Price changes between the first and last trading day in a range. |
| `get_price_change_by_date_range_long(start_date, end_date, fund=None)` | Price changes in tidy format with a fund column. |
| `get_price_change_by_date_range_dict(start_date, end_date, fund=None, date_format="iso")` | Price changes in JSON-friendly dictionary format. |

`get_latest_prices` and `get_latest_price_changes` accept either a single `fund` or a `funds` collection. Passing both arguments raises a `ValueError` to prevent ambiguous filters.

Fund name strings (for example, `"g fund"` or `"  C Fund  "`) are normalized to be case-insensitive, and extra whitespace is ignored. This applies across all fund-specific APIs, not just the latest-price helpers.
Aliases such as `"G"`, `"L2050"`, or `"L Income Fund"` are accepted in the same way.

### Fund Availability

Fund-specific methods validate that the requested fund exists in the cached CSV. If a fund is missing, the methods raise a `ValueError`. Use `get_available_funds()` or `get_data_summary()` to confirm coverage before requesting a specific fund.

## Returns & Performance

| Method | Purpose |
| --- | --- |
| `get_daily_returns(fund=None)` | Daily percentage returns. |
| `get_daily_returns_long(fund=None, start_date=None, end_date=None)` | Daily returns in long (tidy) format. |
| `get_daily_returns_by_date_range(start_date, end_date, fund=None)` | Daily returns for a date range. |
| `get_excess_returns(fund=None, benchmark=TspIndividualFund.G_FUND)` | Daily excess returns versus a benchmark. |
| `get_excess_returns_long(fund=None, benchmark=TspIndividualFund.G_FUND)` | Excess returns in long (tidy) format. |
| `get_cumulative_returns(fund=None)` | Cumulative returns series. |
| `get_cumulative_returns_by_date_range(start_date, end_date, fund=None)` | Cumulative returns for a date range. |
| `get_cumulative_returns_long(fund=None, start_date=None, end_date=None)` | Cumulative returns in long (tidy) format. |
| `get_monthly_returns(fund=None)` | Monthly returns. |
| `get_yearly_returns(fund=None)` | Yearly returns. |
| `get_monthly_return_table(fund, start_date=None, end_date=None)` | Month-by-month return table for a single fund. |
| `get_monthly_return_table_long(fund, start_date=None, end_date=None)` | Month-by-month return table in long (tidy) format. |
| `get_monthly_return_table_dict(fund, start_date=None, end_date=None)` | Month-by-month return table as a JSON-friendly dictionary. |
| `get_trailing_returns(periods, fund=None)` | Trailing returns for one or more periods. |
| `get_trailing_returns_long(periods, fund=None, funds=None)` | Trailing returns in long (tidy) format. |
| `get_trailing_returns_dict(periods, fund=None, funds=None)` | Trailing returns as a JSON-friendly dictionary. |
| `get_performance_summary(fund=None, trading_days=252)` | Performance summary with returns, volatility, and drawdown. |
| `get_performance_summary_by_date_range(start_date, end_date, fund=None, trading_days=252)` | Performance summary for a date range. |
| `get_performance_summary_dict(fund=None, trading_days=252)` | Performance summary as a JSON-friendly dictionary. |
| `get_performance_summary_by_date_range_dict(start_date, end_date, fund=None, trading_days=252, date_format="iso")` | Performance summary for a date range as a JSON-friendly dictionary. |
| `get_risk_return_summary_dict(fund=None, trading_days=252, mar=0.0, confidence=0.95)` | Risk/return summary as a JSON-friendly dictionary. |
| `get_fund_rankings(metric, period=None, start_date=None, end_date=None, as_of=None, reference_date=None, trading_days=252, top_n=None, funds=None, ascending=None)` | Rank funds by a specified performance, trailing-return, or current-price metric. |
| `get_fund_rankings_dict(metric, period=None, start_date=None, end_date=None, as_of=None, reference_date=None, trading_days=252, top_n=None, funds=None, ascending=None, date_format="iso")` | Rank funds by metric and return ordered results as a JSON-friendly dictionary. |
| `get_cagr(fund=None, start_date=None, end_date=None)` | Compound annual growth rate (CAGR) summary. |
| `get_fund_snapshot(fund=None, funds=None, periods=(5, 20, 63, 252), trading_days=252, mar=0.0, confidence=0.95)` | Snapshot of recent prices, changes, trailing returns, and performance metrics (includes `as_of`). |
| `get_fund_snapshot_long(fund=None, funds=None, periods=(5, 20, 63, 252), trading_days=252, mar=0.0, confidence=0.95)` | Snapshot in long (tidy) format for visualization and export. |
| `get_fund_snapshot_dict(fund=None, funds=None, periods=(5, 20, 63, 252), trading_days=252, mar=0.0, confidence=0.95, date_format='iso')` | Snapshot in JSON-friendly dictionary format for dashboards and APIs. |
| `get_current_price_dashboard(fund=None, funds=None, periods=(5, 20, 63, 252), trading_days=252, mar=0.0, confidence=0.95, reference_date=None)` | Dashboard snapshot combining current prices, recency, trailing returns, and risk metrics. |
| `get_current_price_dashboard_dict(fund=None, funds=None, periods=(5, 20, 63, 252), trading_days=252, mar=0.0, confidence=0.95, reference_date=None, date_format='iso')` | Dashboard snapshot as a JSON-friendly dictionary. |
| `get_fund_analytics_report(fund, start_date=None, end_date=None, trading_days=252)` | Consolidated analytics report with summary, performance, drawdown, and overview metrics. |
| `get_fund_analytics_report_dict(fund, start_date=None, end_date=None, trading_days=252, date_format='iso')` | Consolidated analytics report as a JSON-friendly dictionary. |
| `get_portfolio_returns(weights, start_date=None, end_date=None, normalize_weights=True)` | Weighted daily portfolio returns. |
| `get_portfolio_cumulative_returns(weights, start_date=None, end_date=None, normalize_weights=True)` | Weighted portfolio cumulative returns. |
| `get_portfolio_value_history(weights, start_date=None, end_date=None, initial_value=10000.0, normalize_weights=True)` | Weighted portfolio value series. |
| `get_portfolio_performance_summary(weights, start_date=None, end_date=None, trading_days=252, normalize_weights=True)` | Performance summary for a weighted portfolio. |
| `get_portfolio_drawdown_series(weights, start_date=None, end_date=None, normalize_weights=True)` | Drawdown series for a weighted portfolio. |
| `get_portfolio_risk_return_summary(weights, start_date=None, end_date=None, trading_days=252, mar=0.0, confidence=0.95, normalize_weights=True)` | Risk/return summary for a weighted portfolio. |
| `get_return_statistics(fund=None, start_date=None, end_date=None, trading_days=252)` | Descriptive statistics for daily returns. |
| `get_return_statistics_dict(fund=None, start_date=None, end_date=None, trading_days=252, date_format='iso')` | JSON-friendly descriptive statistics for daily returns. |
| `get_return_distribution_summary(fund=None, start_date=None, end_date=None, percentiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))` | Distribution summary for daily returns (percentiles, skew, win/loss rates). |
| `get_return_distribution_summary_dict(fund=None, start_date=None, end_date=None, percentiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))` | JSON-friendly return distribution summary. |
| `get_normalized_prices_long(fund=None, start_date=None, end_date=None, base_value=100.0)` | Normalized prices in long (tidy) format. |

## Risk & Diagnostics

| Method | Purpose |
| --- | --- |
| `get_price_statistics(fund=None, start_date=None, end_date=None)` | Descriptive statistics for prices. |
| `get_price_statistics_dict(fund=None, start_date=None, end_date=None, date_format='iso')` | JSON-friendly descriptive statistics for prices. |
| `get_return_statistics(fund=None, start_date=None, end_date=None, trading_days=252)` | Descriptive statistics for daily returns. |
| `get_return_statistics_dict(fund=None, start_date=None, end_date=None, trading_days=252, date_format='iso')` | JSON-friendly descriptive statistics for daily returns. |
| `get_correlation_matrix()` | Correlation matrix of daily returns. |
| `get_correlation_matrix_long()` | Correlation matrix in long (tidy) format. |
| `get_correlation_matrix_dict(date_format="iso")` | Correlation matrix as a JSON-friendly dictionary with start/end dates. |
| `get_correlation_pairs(top_n=10, absolute=True, min_abs_correlation=None)` | Strongest correlation pairs between funds. |
| `get_correlation_pairs_dict(top_n=10, absolute=True, min_abs_correlation=None, date_format="iso")` | Correlation pairs as a JSON-friendly dictionary with date metadata. |
| `get_rolling_correlation_matrix(window=63)` | Correlation matrix for the most recent rolling window. |
| `get_rolling_correlation_matrix_long(window=63)` | Rolling correlation matrix in long (tidy) format. |
| `get_rolling_correlation_matrix_dict(window=63, date_format="iso")` | Rolling correlation matrix as a JSON-friendly dictionary with window metadata. |
| `get_drawdown_series(fund=None, funds=None)` | Drawdown series for one or more funds. |
| `get_drawdown_series_long(fund=None, funds=None, start_date=None, end_date=None)` | Drawdown series in long (tidy) format. |
| `get_max_drawdown(fund)` | Maximum drawdown for a fund. |
| `get_drawdown_summary(fund)` | Drawdown summary with peak, trough, recovery dates. |
| `get_drawdown_summary_dict(fund, date_format='iso')` | Drawdown summary as a JSON-friendly dictionary. |
| `get_rolling_max_drawdown(fund, window=252)` | Rolling maximum drawdown over a window. |
| `get_rolling_mean(fund, window=20)` | Rolling mean of prices. |
| `get_moving_averages(fund, windows=(20, 50), method='simple')` | Simple or exponential moving averages. |
| `get_rolling_returns(fund, window=20)` | Rolling returns over a window. |
| `get_rolling_performance_summary(fund, window=63, trading_days=252)` | Rolling annualized return, volatility, and Sharpe ratio summary. |
| `get_rolling_performance_summary_long(fund, window=63, trading_days=252)` | Rolling performance summary in long (tidy) format. |
| `get_rolling_performance_summary_dict(fund, window=63, trading_days=252, date_format='iso')` | Rolling performance summary as a JSON-friendly dictionary. |
| `get_rolling_correlation(fund_a, fund_b, window=63)` | Rolling correlation between two funds' returns. |
| `get_beta(fund, benchmark)` | Beta of a fund relative to a benchmark. |
| `get_rolling_beta(fund, benchmark, window=63)` | Rolling beta between a fund and benchmark. |
| `get_tracking_error(fund, benchmark, trading_days=252, start_date=None, end_date=None)` | Annualized tracking error vs. benchmark. |
| `get_information_ratio(fund, benchmark, trading_days=252, start_date=None, end_date=None)` | Annualized information ratio vs. benchmark. |
| `get_rolling_tracking_error(fund, benchmark, window=63, trading_days=252)` | Rolling annualized tracking error. |
| `get_rolling_volatility(fund, window=20, trading_days=252)` | Rolling annualized volatility. |
| `get_rolling_sharpe_ratio(fund, window=63, trading_days=252)` | Rolling Sharpe ratio. |
| `get_sortino_ratio(fund=None, mar=0.0, trading_days=252)` | Sortino ratio with downside deviation. |
| `get_rolling_sortino_ratio(fund, window=63, trading_days=252, mar=0.0)` | Rolling Sortino ratio. |
| `get_return_histogram(fund, bins=50)` | Histogram data for daily returns. |
| `get_value_at_risk(confidence=0.95, fund=None)` | Historical value at risk (VaR) for daily returns. |
| `get_expected_shortfall(confidence=0.95, fund=None)` | Historical expected shortfall (CVaR) for daily returns. |

## Allocation Tools

| Method | Purpose |
| --- | --- |
| `create_allocation(...)` | Calculates share allocation totals and percentages. |
| `create_allocation_from_shares(shares)` | Calculates allocation totals from a mapping of funds to shares. |
| `show_pie_chart(allocation)` | Plots a pie chart for allocation percentages. |

## Visualization Helpers

| Method | Purpose |
| --- | --- |
| `show_fund_price_chart(fund)` | Price chart for a single fund. |
| `show_individual_price_chart()` | Price chart for all individual funds. |
| `show_lifecycle_price_chart()` | Price chart for all lifecycle funds. |
| `show_price_history_chart(funds=None, start_date=None, end_date=None)` | Price chart for selected funds. |
| `show_cumulative_returns_chart(fund=None)` | Cumulative returns chart. |
| `show_normalized_price_chart(fund=None, start_date=None, end_date=None, base_value=100.0)` | Normalized price chart. |
| `show_monthly_returns_chart(fund=None)` | Monthly returns bar chart. |
| `show_yearly_returns_chart(fund=None)` | Yearly returns bar chart. |
| `show_monthly_return_heatmap(fund, start_date=None, end_date=None, cmap='RdYlGn')` | Monthly return heatmap. |
| `show_return_histogram_chart(fund, bins=50)` | Daily return histogram chart. |
| `show_rolling_mean_chart(fund, window=20)` | Rolling mean chart. |
| `show_moving_average_chart(fund, windows=(20, 50), method='simple', show_price=True)` | Moving average chart with optional price overlay. |
| `show_rolling_returns_chart(fund, window=20)` | Rolling returns chart. |
| `show_rolling_performance_summary_chart(fund, window=63, trading_days=252)` | Rolling performance chart (return, volatility, Sharpe ratio). |
| `show_rolling_correlation_chart(fund_a, fund_b, window=63)` | Rolling correlation chart. |
| `show_rolling_beta_chart(fund, benchmark, window=63)` | Rolling beta chart. |
| `show_rolling_tracking_error_chart(fund, benchmark, window=63, trading_days=252)` | Rolling tracking error chart. |
| `show_rolling_volatility_chart(fund, window=20, trading_days=252)` | Rolling volatility chart. |
| `show_rolling_sharpe_ratio_chart(fund, window=63, trading_days=252)` | Rolling Sharpe ratio chart. |
| `show_rolling_sortino_ratio_chart(fund, window=63, trading_days=252, mar=0.0)` | Rolling Sortino ratio chart. |
| `show_drawdown_chart(fund=None, funds=None)` | Drawdown chart. |
| `show_rolling_max_drawdown_chart(fund, window=252)` | Rolling max drawdown chart. |
| `show_correlation_heatmap()` | Correlation heatmap. |
| `show_rolling_correlation_heatmap(window=63)` | Rolling correlation heatmap for recent returns. |
| `show_correlation_pairs_chart(top_n=10, absolute=True, min_abs_correlation=None)` | Bar chart of strongest correlation pairs. |
| `show_latest_price_change_chart()` | Latest price change bar chart. |
| `show_latest_price_changes_per_fund_chart(funds=None)` | Latest price change bar chart per fund. |
| `show_recent_price_change_heatmap(days=5, fund=None, funds=None, as_of=None)` | Heatmap of recent daily percent changes by fund. |
| `show_latest_prices_per_fund_chart(funds=None)` | Latest price bar chart per fund. |
| `show_current_prices_per_fund_chart(funds=None, as_of=None, sort_by="price", ascending=False)` | Current price bar chart per fund (supports historical anchors). |
| `show_price_recency_chart(funds=None, reference_date=None)` | Price recency (days-since) bar chart per fund. |
| `show_current_price_alerts_chart(funds=None, as_of=None, reference_date=None, stale_days=3, change_threshold=0.02, metric="change_percent")` | Alert chart for stale prices or large daily moves. |
| `show_fund_rankings_chart(metric, period=None, start_date=None, end_date=None, trading_days=252, top_n=10, funds=None, ascending=None)` | Fund ranking bar chart for performance or trailing returns. |
| `show_trailing_returns_chart(periods=(1, 5, 20, 63, 252), fund=None, funds=None)` | Trailing returns bar chart across one or more horizons. |
| `show_price_change_chart_as_of(as_of)` | Price change bar chart anchored to a specific date. |
| `show_fund_coverage_chart()` | Fund coverage bar chart. |
| `show_missing_business_days_chart(start_date=None, end_date=None)` | Timeline chart of missing business days. |
| `show_excess_returns_chart(fund=None, benchmark=TspIndividualFund.G_FUND)` | Excess returns chart versus a benchmark. |
| `show_daily_return_histogram(fund, bins=50)` | Daily returns histogram. |
| `show_portfolio_value_chart(weights, start_date=None, end_date=None, initial_value=10000.0, normalize_weights=True)` | Portfolio value chart. |
| `show_portfolio_drawdown_chart(weights, start_date=None, end_date=None, normalize_weights=True)` | Portfolio drawdown chart. |

Chart helpers that target fund groups (for example, lifecycle funds) raise a `ValueError` if there
is no data available for the requested group.
