# Configuration Guide

This guide explains how to configure `TspAnalytics` for caching, refresh timing, and offline workflows.

## Cache Location

By default, `TspAnalytics` stores the downloaded CSV in `~/.cache/tsp/fund-price-history.csv`.
You can override this location either with a constructor argument or an environment variable:

```python
from pathlib import Path
from tsp import TspAnalytics

# Per-instance cache directory.
prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")
```

```bash
export TSP_DATA_DIR="$HOME/.cache/tsp"
```

If `data_dir` points to a file instead of a directory, `TspAnalytics` raises a `ValueError`.

## Refresh Timing

`TspAnalytics` refreshes the cache automatically when a new business day has passed *and* the
configured update time is reached. You can control this window via `time_hour`:

```python
from datetime import time
from tsp import TspAnalytics

# Refresh after 8:00pm local time instead of the default 7:00pm.
prices = TspAnalytics(time_hour=time(hour=20))
```

## Offline or Manual Refresh

Disable automatic refresh for offline use cases and call `refresh()` when you are ready:

```python
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.refresh()
```

You can also inspect the cache status without triggering a network request:

```python
status = prices.get_cache_status()
```

## Network Settings

If you need to tune network behavior or use a different source, supply a timeout or URL:

```python
from tsp import TspAnalytics

prices = TspAnalytics(
    request_timeout=15.0,
    csv_url="https://www.tsp.gov/data/fund-price-history.csv",
    max_retries=3,
    retry_backoff=0.5,
    user_agent="MyApp/1.0",
)
```

`max_retries` controls how many attempts are made before the download fails. `retry_backoff`
is the base delay (in seconds) between retries, applied exponentially (for example, `0.5`,
`1.0`, `2.0`, ...). Set `retry_backoff=0` to retry immediately without waiting.

### Custom Sessions

For advanced networking (proxies, retries, custom adapters), pass a pre-configured
`requests.Session` instance:

```python
import requests
from tsp import TspAnalytics

session = requests.Session()
session.headers.update({"User-Agent": "my-app/1.0"})
prices = TspAnalytics(session=session)
```

When a session is provided, the client ensures a CSV-friendly `Accept` header and a user-agent
string are present if you have not set them already.

## Custom Data Sources

If you already have a CSV or a pandas dataframe, you can load it directly:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.load_csv(Path("fund-price-history.csv"))
```

```python
prices.load_dataframe(my_dataframe)
```

This is useful for testing, offline analysis, or analyzing a curated dataset.
