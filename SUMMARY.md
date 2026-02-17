# tsp-analytics: 80% Understanding (Sharpened)

## What it is / Who it's for
`tsp-analytics` is a Python library for analysts, investors, and developers interested in exploring, analyzing, and visualizing Thrift Savings Plan (TSP) fund data. It is designed for Python API/programmatic usage, focused on official TSP fund prices and performance analytics.

## Key Features (as directly evidenced)
- **Official Data Handling**: Downloads TSP fund price history (CSV) from tsp.gov, with robust local disk cache and fallback support.
- **Flexible Configuration**: Data/cache directory can be set via the `TSP_DATA_DIR` environment var or `data_dir` constructor argument. Supports specifying fund requirements, trading hour cutoff, holiday calendars, custom user agent/URL/session, and more.
- **Rich Analytics**: Composite analytics mixins offer return, correlation, risk, rolling metrics, drawdown, reporting, and portfolio analytics. Fund-specific lookups are flexible (by name, alias, or enum; strong normalization and alias support confirmed directly in `fund_metadata.py`).
- **Comprehensive Fund Name Resolution**: Fund inputs can be provided as enums or many aliases ("L 2050", "L2050", "L 2050 Fund", "lifecycle 2050", etc.); a large static mapping and normalization logic is present (see `fund_metadata.py`, `fund_resolution.py`). Aliases are exposed programmatically.
- **Ready Visualizations**: Built-in Matplotlib chart routines.
- **Extensible Data Layer**: Pluggable data providers (default is requests-based CSV fetch).
- **Robust Logging & Callback Support**: `log_level`, custom loggers, and event handlers on cache actions.

## Architecture Overview (from code evidence)
- **Class Structure**: All top-level features orchestrated by the `TspAnalytics` class (src/tsp/tsp.py) -- this composes numerous mixins (analytics, fund metadata/resolution, validation, I/O, charts, etc.).
- **Analytics Mixins**: `AnalyticsMixin` (src/tsp/analytics/__init__.py) groups granular mixins: prices, price-changes, returns, correlations, risk, reports, portfolio, and helpers.
- **Fund Metadata & Aliases**: Fund names and aliases are static but very robust. The `FundMetadataMixin` and `FundResolutionMixin` (see `src/tsp/fund_metadata.py`, `src/tsp/fund_resolution.py`) provide normalization, alias mapping, and validation for fund identifiers. These are not fetched externally—they are computed from code-provided lists/enums and normalization rules.
- **Data Providers**: Interface/protocol for CSV data fetching (default via requests) in `src/tsp/data_providers.py`. Results are struct-like objects; can supply custom versions for testing or new sources.
- **Configurable Filepath**: Uses `~/.cache/tsp/fund-price-history.csv` as default cache; can be overridden.

## Execution Model / Entrypoints
- **Library API Only**: Usage is by importing and instantiating `TspAnalytics` in Python: `from tsp import TspAnalytics`.
- **No CLI or web frontend.**

## How to Run Locally (confirmed)
1. Install (`pip install tsp-analytics` or from local source).
2. Create and use `TspAnalytics()` in your Python code.
3. Manipulate with optional args for log level, data dir, fund list, etc.

## Config / Env Vars
- `TSP_DATA_DIR`: Sets cache dir for downloaded fund price history.
- Constructor arguments cover all other runtime customizations (see class docstring for a thorough list).
- No evidence of secrets or required credentials.

## Data Flow
- On usage, attempts to load cached CSV from disk. If missing/stale and `auto_update=True`, will fetch from tsp.gov (with retry logic, configurable timeouts, custom user-agent, etc.).
- Data loaded into in-memory pandas data structures for analytics and charting.
- No external DB or message queue.

## Extension/Implementation Hotspots
- `src/tsp/tsp.py`: Start here—main orchestration (TspAnalytics class, handles init, config, fetch, orchestrates mixins).
- `src/tsp/analytics/`: For in-depth analytics/logic (returns, correlations, risk, etc.).
- `src/tsp/data_providers.py`: For the fetch interface; extensible if using non-default HTTP/data fetching.
- `src/tsp/fund_metadata.py`, `src/tsp/fund_resolution.py`: For fund aliasing logic, normalization rules, and enforcement of input validation.

## Risks / Gotchas
- **Requires Python 3.10+** (uses typing features and annotated types).
- **No CLI/REST/Web interface.**
- If the tsp.gov endpoint or network is unreachable, fetch will retry (up to a limit) with exponential backoff—they'll raise error if unsuccessful (see data_providers/request logic).
- **Fund metadata/aliases are static and coded in helpers.** No evidence of dynamic alias, external alias updating, or fetching—static rules and enums only.       
- **Offline operation uses the last cached file**; user must ensure data currency.

## Suggested First 5 Files to Read
1. `src/tsp/tsp.py` (TspAnalytics logic, mixin composition, config parsing)
2. `src/tsp/analytics/__init__.py` (shows analytics mixin composition and supported analytic features)
3. `src/tsp/data_providers.py` (abstracts and implements the data fetch logic)
4. `src/tsp/fund_metadata.py` (details alias logic, normalization, public fund metadata methods)
5. `src/tsp/fund_resolution.py` (input validation, mapping, and normalization functions)

## Open Questions
- Can fund metadata/aliases be refreshed externally or are they fully static? (Evidence: Fully static, computed from code. No external refresh.)
- Are there any plans to support CLI, REST, or alternate frontends? (No evidence found.)
- How to extend with other fund data sources (is there a plugin registration pattern or just protocol adherence)? (Evidence: Data provider protocol can be swapped, but no plugin registry exists; you supply your own provider object.)

---
**Evidence basis**: All above claims are directly supported by class docstrings, implementations, and argument signatures in the referenced files. In particular, both fund alias logic and alias mapping are static and described in detail; normalization is robust but does not employ dynamic fetching or external config. No CLI/web. Extension is via subclassing/mixins and Python protocol adherence only.


# Reading plan

Get productive fast in the `tsp-analytics` Python library. The plan focuses on core architecture, feature hotspots, and how to safely run and extend the codebase. 

---

1. **README.md**
   *Why it matters:* First overview of usage, main features, and how the library is intended to be used.
   *What to look for:*
   - Installation and basic usage instructions
   - Example code snippets
   - Key project capabilities
   *Time estimate:* 5 min

2. **src/tsp/tsp.py**
   *Why it matters:* Main orchestration logic—defines the `TspAnalytics` class (the core API object) and composes all mixins/features.
   *What to look for:*
   - Constructor arguments and config mechanisms
   - How mixins are combined
   - Key public methods and I/O flows
   *Time estimate:* 20 min

3. **src/tsp/analytics/__init__.py**
   *Why it matters:* Shows which analytics mixins are bundled and provides a high-level map of available analytics features.
   *What to look for:*
   - Which analytics modules are imported/composed
   - Brief docstrings listing analytic capabilities
   *Time estimate:* 7 min

4. **src/tsp/data_providers.py**
   *Why it matters:* Implements/facilitates data fetching, caching, and provider extensibility.
   *What to look for:*
   - DataProvider class/protocol
   - Network access, cache logic, retry/backoff behavior
   - Swapping or extending data sources
   *Time estimate:* 12 min

5. **src/tsp/fund_metadata.py**
   *Why it matters:* Contains all static mappings, aliases, and fund metadata exposed to users.
   *What to look for:*
   - How fund aliases are mapped/normalized
   - Main public metadata methods and structures
   *Time estimate:* 10 min

6. **src/tsp/fund_resolution.py**
   *Why it matters:* Key logic for validating and resolving user input (fund names, enums, aliases).
   *What to look for:*
   - Normalization/validation functions
   - Error handling for invalid input
   *Time estimate:* 8 min

7. **src/tsp/analytics/portfolio.py**
   *Why it matters:* Portfolio analytics (allocations, returns, risk) are key use cases.
   *What to look for:*
   - Portfolio computation logic
   - Integration with fund/metdata/price sources
   *Time estimate:* 10 min

8. **src/tsp/charts.py**
   *Why it matters:* Provides built-in methods for visualizing price and analytics data.
   *What to look for:*
   - Chart-generating functions and their options
   - Use of Matplotlib and integration with analytics results
   *Time estimate:* 7 min

9. **src/tsp/data_io.py**
   *Why it matters:* Handles reading/writing of local price data and related file operations.
   *What to look for:*
   - CSV/file handling logic
   - Interfaces to/from other data structures
   *Time estimate:* 7 min

10. **tests/test_tsp_loading.py**
    *Why it matters:* Demonstrates basic data loading and integration across modules—good for smoke testing.
    *What to look for:*
    - Typical initialization/test flows
    - Which assertions are made on fetch or load
    *Time estimate:* 5 min

11. **tests/test_data_providers.py**
    *Why it matters:* Shows how data fetching is validated and tested, useful for extending providers or fixing bugs.
    *What to look for:*
    - Mock/fake providers
    - How failures/caching are tested
    *Time estimate:* 7 min

12. **pyproject.toml**
    *Why it matters:* Declares dependencies, build requirements, and Python version constraints.
    *What to look for:*
    - Required package versions
    - Build backend and minimum Python version
    *Time estimate:* 3 min

---

## If you only have 30 minutes
1. **README.md** — Grasp high-level purpose, install/use basics (5 min)
2. **src/tsp/tsp.py** — Core class, config, and orchestration (15 min)
3. **src/tsp/analytics/__init__.py** — Understand analytics surface area (7 min)

---

## If you need to make a change safely
- **How to run tests/build:**
  - Run all tests with `pytest` (tests/ directory has test modules; use `pytest` from repo root or `python -m pytest`).
  - Ensure you have dependencies from `requirements.txt` or via `pip install -e .`.

- **Where to add a small change and validate quickly:**
  - Add a change in `src/tsp/fund_metadata.py` (e.g., fund alias mapping) or `src/tsp/tsp.py` (logic tweak).
  - Validate by running relevant test(s), such as `tests/test_fund_metadata_helpers.py` or `tests/test_tsp_loading.py`, for fast feedback.