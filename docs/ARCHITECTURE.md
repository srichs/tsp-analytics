# Architecture Overview

This document describes how the TSP pricing and analytics module is organized, how data
flows through the system, and where to extend it when adding new functionality.

## High-Level Structure

The project centers on the `TspAnalytics` class, which aggregates three concerns:

- **Data access & caching** (`src/tsp/tsp.py`): downloads the CSV, caches it locally, and
  normalizes the data.
- **Analytics** (`src/tsp/analytics/`): calculates returns, risk metrics, and performance
  summaries.
- **Visualization** (`src/tsp/charts.py`): provides Matplotlib helpers built on the analytics
  output.

The `TspAnalytics` class mixes in `AnalyticsMixin` and `ChartsMixin` so that analytics and
visualizations operate directly on the cached dataframe.

## Data Flow

1. **Initialization**: `TspAnalytics` resolves the cache directory and loads existing CSV data
   if present.
2. **Refresh/Check**: `check()` optionally downloads a new CSV based on the last business
   day and the configured update time. `refresh()` forces an update.
3. **Normalization**: `_normalize_dataframe()` standardizes column names, coerces date and
   numeric columns, removes duplicates, and drops unusable rows.
4. **Analytics**: methods in `AnalyticsMixin` read from the normalized dataframe and return
   `DataFrame` or `Series` objects for downstream visualization or export.

## Adding Analytics

When adding analytics:

- Validate input parameters (dates, positive integers, numeric values).
- Use existing helpers such as `_filter_by_date_range`, `_resolve_funds`, and
  `_resolve_weights` to keep behavior consistent.
- Prefer returning tidy data (`DataFrame`/`Series`) instead of plotting directly. Chart
  helpers should live in `ChartsMixin` and consume analytics outputs.

## Adding Visualizations

Visualization helpers should:

- Call `self._ensure_dataframe()` and validate inputs.
- Use analytics helpers for calculations rather than duplicating logic.
- Rely on Matplotlib for consistent styling with existing charts.

## Testing Strategy

Tests live in `tests/` and exercise:

- Input validation and error handling.
- Core analytics results against deterministic fixtures.
- Plotting helpers with `plt.show` monkeypatched to avoid interactive windows.

Use fixtures like `_build_price_dataframe` to produce stable test data without network
access.
