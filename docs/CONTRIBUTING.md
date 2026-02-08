# Contributing Guide

Thank you for contributing to the TSP price analytics library! This guide summarizes the
workflow for reporting issues, adding features, and updating documentation. For full project
policies, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Quick Checklist

- [ ] Add or update tests for code changes.
- [ ] Update user-facing documentation in `README.md` and `docs/`.
- [ ] Run the test suite (`pytest`).
- [ ] Keep changes focused and well-documented in commit messages.
- [ ] Document any new public APIs in `docs/REFERENCE.md`.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Project Structure

- `src/tsp/`: main library implementation (client, analytics mixins, charting, utilities).
- `tests/`: pytest suite for unit and integration-style coverage.
- `docs/`: user and contributor documentation (guides, reference, examples).

## Code Style

- Prefer small, composable helpers in mixins (avoid deep inheritance chains).
- Keep error messages explicit and user-facing (they are surfaced in dashboards).
- Avoid try/except around imports (import errors should be surfaced directly).
- Keep docstrings up to date for new parameters or behavior changes.

## Running Tests

```bash
pytest
```

If you are working on analytics or charts, include tests that validate:

- Input validation and error messaging.
- Numerical correctness using known fixtures.
- Edge cases like empty date ranges or missing fund columns.

Targeted test runs for faster feedback:

```bash
pytest tests/test_tsp_pricing.py
pytest tests/test_data_io.py
```

## Documentation Updates

When you add features:

- Update the relevant guide in `docs/` (typically `USAGE.md` or `ANALYTICS.md`).
- Add examples in `docs/EXAMPLES.md` if the feature is user-facing.
- Update `README.md` highlights or quick-start snippets if applicable.
- Update `docs/REFERENCE.md` if you add or rename public APIs.

If you add new parameters (even optional ones), update the relevant sections in:

- `docs/USAGE.md`
- `docs/CURRENT_PRICES.md` or `docs/ANALYTICS.md`
- `README.md` (if the change is user-facing)

## Documentation Builds

```bash
pip install -r requirements-dev.txt
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

## Where to Add Things

- `src/tsp/`: Library source code.
- `tests/`: Pytest test suite.
- `docs/`: User and contributor documentation.

See [docs/DEVELOPMENT.md](DEVELOPMENT.md) for the full development guide.
