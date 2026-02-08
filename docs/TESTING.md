# Testing Guide

This project uses `pytest` for automated tests and includes extensive unit coverage for
price retrieval, analytics, and visualization helpers.

## Running Tests

Make sure you have the development dependencies installed:

```bash
pip install -e .[dev]
```

If you prefer a single requirements file instead of extras:

```bash
pip install -r requirements-dev.txt
```

Run the full suite:

```bash
pytest
```

Run a focused test module:

```bash
pytest tests/test_tsp_pricing.py
```

Run a single test by name:

```bash
pytest tests/test_tsp_pricing.py -k test_get_price_and_date_queries
```

Run the test suite with coverage (requires `pytest-cov`):

```bash
pytest --cov=tsp --cov-report=term-missing
```

## Coverage Expectations

When adding new public APIs or analytics helpers, aim to cover:

- The primary success path (expected outputs).
- Validation errors (invalid input types, out-of-range values).
- Edge cases (empty dataframes, missing fund data, or short date ranges).

If new behavior touches data downloads or caching, add tests that mock network responses
and validate fallback behavior.

## Test Environment Notes

- Tests are designed to run offline. Network calls are mocked where needed.
- The Matplotlib backend is set to `Agg` in `tests/conftest.py`, so chart helpers can be
  executed without opening GUI windows.
- A temporary cache directory is set via `TSP_DATA_DIR` in `tests/conftest.py` to avoid
  touching user-level caches.

## Writing New Tests

When adding new analytics or chart helpers:

1. Add tests under `tests/` that validate both expected output and error handling.
2. Use the existing helper dataframes in `tests/test_tsp.py` to keep tests deterministic.
3. Mock network calls (for example, by monkeypatching `tsp.tsp.Session`) when validating
   download or cache behavior.
4. Keep tests small and focused. If you add new public methods, ensure there is at least one
   test for each new success path and each expected validation error.
