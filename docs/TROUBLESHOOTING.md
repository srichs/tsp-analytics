# Troubleshooting

## Data Download Issues

If the CSV cannot be downloaded, `TspAnalytics` will keep the most recent cached file. Typical causes:

- Network outages or proxy restrictions.
- The `tsp.gov` endpoint is temporarily unavailable.
- The download returned HTML (for example, a maintenance page) instead of the CSV file.

To confirm you have data available, inspect the cached summary:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
print(prices.get_data_summary())
```

If you want to know whether the cache is stale relative to the latest business day, inspect
`get_cache_status()` which includes `cache_age_days`, `data_age_days`, `is_stale`, and
`stale_by_days` fields. It also reports `dataframe_valid` and `validation_error` if the cached
CSV exists but fails normalization.

If you see errors like `downloaded content does not look like a CSV file` or
`downloaded content does not include a Date column header`, the upstream response likely
returned HTML or a non-CSV payload. Verify access to `https://www.tsp.gov/data/fund-price-history.csv`,
check for captive portals or proxies, and retry.

If the cache is empty, delete the local CSV and retry (the default cache location is
`~/.cache/tsp/fund-price-history.csv` unless `TSP_DATA_DIR` is set):

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics()
Path(prices.csv_filepath).unlink(missing_ok=True)
prices.check()
```

## Date Range Errors

Methods that accept `start_date` and `end_date` require both values and that the start date is not after the end date. Double-check the range if you see:

- `start_date must be on or before end_date`
- `start_date and end_date must be provided together`
- `no price data available for requested date range`

## Fund Availability Errors

If you see `fund not available in data: <fund name>`, your cached CSV does not contain that fund column. Use:

```python
from tsp import TspAnalytics

prices = TspAnalytics()
print(prices.get_available_funds())
```

Update the cache with `prices.check()` or point `data_dir` to a location that includes the latest CSV.

## Matplotlib Charts Not Displaying

Visualization helpers call `plt.show()` which requires a display. In headless environments, configure a non-interactive backend:

```python
import matplotlib
matplotlib.use("Agg")
```

You can then save charts using Matplotlib directly after calling the helper functions or by reusing the underlying data methods.
