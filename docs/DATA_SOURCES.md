# Data Sources & Caching

This project pulls daily TSP fund price history from the official CSV published by `tsp.gov`.
The CSV is cached locally so analytics and visualizations can run offline once data is downloaded.

## Default Source

By default, `TspAnalytics` downloads:

- `https://www.tsp.gov/data/fund-price-history.csv`

The CSV includes historical prices for all individual and lifecycle funds.

The download workflow validates the response to ensure it looks like a CSV (for example,
checking for a `Date` header and at least one known TSP fund column). If the response
appears to be HTML or missing the expected headers, the client raises a validation error
and keeps the existing cache so you can investigate the upstream response.

## Cache Location

The default cache location is:

```
~/.cache/tsp/fund-price-history.csv
```

Override it with `data_dir` or the `TSP_DATA_DIR` environment variable:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")
```

```bash
export TSP_DATA_DIR="$HOME/.cache/tsp"
```

## Offline or Custom Data

If you already have a CSV file or a pandas dataframe, load it directly:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.load_csv("/path/to/fund-price-history.csv")
```

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.load_dataframe(my_dataframe)
```

These methods allow analytics to run in air-gapped environments, CI, or local notebooks without
network access.

## Cache Health Checks

Use the cache helpers to validate the cache without triggering a download:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
status = prices.get_cache_status()
report = prices.get_data_quality_report()
```

`get_cache_status()` returns staleness indicators and file metadata, while
`get_data_quality_report()` summarizes missing business days and fund coverage.
