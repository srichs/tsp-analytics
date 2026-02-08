# Contributing

Thanks for considering a contribution!

## Getting Started

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .[dev]
   ```
   Alternatively, `pip install -r requirements-dev.txt` installs base and dev
   dependencies without the editable install.
3. Run the test suite:
   ```bash
   pytest
   ```

## Development Tips

- The `TspAnalytics` class is the main entry point for analytics and data retrieval.
- Data is cached locally in `~/.cache/tsp/fund-price-history.csv` by default.
- Add new analytics to `src/tsp/tsp.py` along with unit tests under `tests/`.
- See `docs/DEVELOPMENT.md` for a fuller development guide.

## Documentation Updates

- User-facing examples belong in `docs/USAGE.md` or `docs/ANALYTICS.md`.
- Contributor guidance is maintained in `docs/DEVELOPMENT.md`.
- Add troubleshooting notes to `docs/TROUBLESHOOTING.md` if you address a common issue.

## Code Style

- Keep methods focused and prefer small helper functions.
- Add input validation for public APIs where possible.
- Update documentation in `README.md` and the `docs/` folder if new functionality is added.

## Pull Request Checklist

- [ ] Tests added or updated for changes.
- [ ] Documentation updated when new behavior is added.
- [ ] `pytest` passes locally.
