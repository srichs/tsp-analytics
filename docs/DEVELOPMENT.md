# Development Guide

This guide covers local development, testing, and contributing practices for `python-tsp-priv`.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

For editable installs during development:

```bash
pip install -e .[dev]
```

If you prefer using the requirements file instead of extras, `requirements-dev.txt`
includes the base requirements and development tooling:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest
```

For quick feedback during development:

```bash
pytest tests/test_tsp_pricing.py
```

See [docs/TESTING.md](TESTING.md) for testing conventions and guidance on writing new tests.
See [docs/CONTRIBUTING.md](CONTRIBUTING.md) for the contribution checklist and workflow.

If you are adding analytics or visualization features, include unit tests that validate:

- Input validation and error messages.
- Basic numerical correctness against known data fixtures.
- Edge cases (empty data ranges, missing fund columns).

## Data Cache Location

By default, the price history CSV is cached in `~/.cache/tsp/fund-price-history.csv`. For development (or when packaging), you can set a custom cache directory:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(data_dir=Path.home() / ".cache" / "tsp")
```

You can also set the `TSP_DATA_DIR` environment variable to override the cache location:

```bash
export TSP_DATA_DIR="$HOME/.cache/tsp"
```

## Building Documentation

This project supports two documentation outputs: Sphinx HTML documentation and
single-file HTML from `pydoc`.

### Sphinx HTML

```bash
pip install -r requirements-dev.txt
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

Open `docs/sphinx/_build/html/index.html` in a browser to view the docs.

### pydoc HTML

```bash
python -m pydoc -w tsp.tsp
```

The command writes `tsp.tsp.html` in the current working directory.

## Packaging Notes

This project follows PEP 621 for package metadata in `pyproject.toml`. The
`setup.py` file remains for legacy tooling compatibility and only wires up
package discovery.

## Release Process

When preparing a release:

1. Update the version in `pyproject.toml`.
2. Summarize notable changes in the documentation or release notes.
3. Tag the release in Git with the new version (for example, `v0.1.1`).
4. Publish the updated package to your package index of choice.

## Adding Analytics or Visualizations

1. Add new analytics helpers to `src/tsp/tsp.py` on `TspAnalytics`.
2. Validate inputs (date ranges, numeric parameters) where appropriate.
3. Add or update tests under `tests/`.
4. Document new functionality in `docs/` and update `README.md` if it is user-facing.

## Project Layout

- `src/tsp/`: Library source code.
- `tests/`: Pytest coverage for analytics and validation.
- `docs/`: Usage, analytics, troubleshooting, and development guides.
- `docs/sphinx/`: Sphinx configuration and reference material.

## Documentation Structure

- `docs/USAGE.md`: Core usage and configuration.
- `docs/ANALYTICS.md`: Analytics and visualization examples.
- `docs/EXAMPLES.md`: End-to-end recipes and visualization workflows.
- `docs/DEVELOPMENT.md`: Development and contribution guidance.
- `docs/TESTING.md`: Testing conventions and workflows.
- `docs/TROUBLESHOOTING.md`: Common issues and fixes.
- `docs/ARCHITECTURE.md`: Data flow and code organization overview.
- `docs/sphinx/`: Sphinx site source files.

## Contribution Checklist

- [ ] Tests added or updated.
- [ ] Documentation updated for user-facing changes.
- [ ] `pytest` passes locally.
