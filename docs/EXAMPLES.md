# Examples & Recipes

This guide collects end-to-end examples that combine analytics helpers with visualization
workflows. All examples assume you have the official TSP CSV cached locally (handled
automatically by `TspAnalytics`).

## Quick Daily Snapshot

Build a daily snapshot that combines latest prices, percent changes, and trailing returns:

```python
from tsp import TspAnalytics

prices = TspAnalytics()

snapshot = prices.get_fund_snapshot(periods=[1, 5, 20, 63])
# Includes an `as_of` column with the latest data date.
print(snapshot.sort_values("change_percent", ascending=False))

# Tidy snapshot for dashboarding (fund/metric/value)
snapshot_long = prices.get_fund_snapshot_long(periods=[1, 5, 20, 63])
```

## Price Summary Snapshot

Summarize the full price history (first/last dates, min/max, and total return):

```python
from tsp import TspAnalytics

prices = TspAnalytics()
summary = prices.get_price_summary()
summary_dict = prices.get_price_summary_dict()
```

## Recent Price Window

Pull the most recent trading-day prices for quick dashboards or recent-period charts:

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()

recent = prices.get_recent_prices(days=5)
recent_long = prices.get_recent_prices_long(days=5)
recent_as_of = prices.get_recent_prices(days=10, as_of=date(2024, 1, 15))
recent_dict = prices.get_recent_prices_dict(days=5)
```

## Fund Analytics Bundle

Generate a single report that includes summary statistics, performance, drawdowns, and
current overview metrics for a fund:

```python
from datetime import date
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

report = prices.get_fund_analytics_report(
    TspIndividualFund.C_FUND,
    start_date=date(2024, 1, 1),
)
report_dict = prices.get_fund_analytics_report_dict(
    TspIndividualFund.C_FUND,
    start_date=date(2024, 1, 1),
)
```

## Monthly Return Tables for Dashboards

Export a month-by-month return table in long (tidy) format for a heatmap or BI tool:

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
monthly_long = prices.get_monthly_return_table_long(TspIndividualFund.C_FUND)
monthly_dict = prices.get_monthly_return_table_dict(TspIndividualFund.C_FUND)
```

## As-of Price Change Snapshot (Per Fund)

Capture per-fund price changes as of a specific date (each fund uses its last two valid prices):

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
changes_as_of = prices.get_price_changes_as_of_per_fund(date(2024, 1, 3))
changes_as_of_dict = prices.get_price_changes_as_of_per_fund_dict(date(2024, 1, 3))
```

## Rolling Volatility Dashboard (Seaborn)

```python
import seaborn as sns
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
rolling = prices.get_rolling_volatility(TspIndividualFund.C_FUND, window=63)

ax = sns.lineplot(data=rolling, x=rolling.index, y=rolling[TspIndividualFund.C_FUND.value])
ax.set(title="63-Day Rolling Volatility", xlabel="Date", ylabel="Volatility")
```

## Moving Averages (Matplotlib)

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

prices.show_moving_average_chart(
    fund=TspIndividualFund.C_FUND,
    windows=[20, 50],
    method="simple",
)
```

## Fund Comparison with Plotly

```python
import plotly.express as px
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
history = prices.get_price_history(
    funds=[TspIndividualFund.C_FUND, TspIndividualFund.S_FUND],
)
history_long = history.melt(id_vars="Date", var_name="Fund", value_name="Price")

fig = px.line(history_long, x="Date", y="Price", color="Fund", title="TSP Fund Prices")
fig.show()
```

## Portfolio Performance Summary

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()

weights = {
    TspIndividualFund.C_FUND: 0.6,
    TspIndividualFund.S_FUND: 0.4,
}

portfolio_summary = prices.get_portfolio_performance_summary(weights=weights)
portfolio_risk = prices.get_portfolio_risk_return_summary(weights=weights)
print(portfolio_summary)
print(portfolio_risk)
```

## Monte Carlo Bootstrapping (Long-Horizon Outcomes)

Bootstrap daily returns to model potential long-horizon outcomes for a portfolio. This is
useful for estimating the distribution of ending values over multi-year horizons.

```python
import numpy as np
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {TspIndividualFund.C_FUND: 0.6, TspIndividualFund.S_FUND: 0.4}

years = 20
simulations = 10_000
initial_value = 100_000

bootstrapped = prices.get_portfolio_bootstrap_simulation(
    weights=weights,
    years=years,
    simulations=simulations,
    initial_value=initial_value,
    random_state=42,
)

percentiles = np.percentile(bootstrapped["ending_value"], [5, 25, 50, 75, 95])
print(dict(zip([5, 25, 50, 75, 95], percentiles)))
```

## Sequence-of-Returns Risk (Retirement Drawdown)

Estimate how the ordering of returns affects retirement drawdowns by running a rolling
historical sequence test on monthly returns.

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
weights = {TspIndividualFund.C_FUND: 0.5, TspIndividualFund.S_FUND: 0.3, TspIndividualFund.G_FUND: 0.2}

initial_value = 750_000
annual_withdrawal = 40_000
retirement_years = 30

sequences = prices.get_portfolio_retirement_sequence_analysis(
    weights=weights,
    initial_value=initial_value,
    annual_withdrawal=annual_withdrawal,
    years=retirement_years,
)
success_rate = sequences["success"].mean()
worst_cases = sequences.nsmallest(5, "ending_value")[["start_date", "ending_value"]].values.tolist()

print(f"Success rate: {success_rate:.1%}")
print("Worst starting dates:", worst_cases)
```

## Exporting Long-Format Data

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics()

metrics_long = prices.get_price_history_with_metrics_long()
metrics_long.to_csv(Path("tsp_price_metrics_long.csv"), index=False)
```

## Working with a Local CSV

If you already have the TSP CSV cached elsewhere, load it directly:

```python
from pathlib import Path
from tsp import TspAnalytics

prices = TspAnalytics(auto_update=False)
prices.load_csv(Path("fund-price-history.csv"))

# Or load CSV text from an API or other source
csv_text = "Date,G Fund,C Fund\n2024-01-02,100.0,200.0\n"
prices.load_csv_text(csv_text)

summary = prices.get_data_summary()
print(summary)
```

## Heatmap of Monthly Returns

```python
from tsp import TspIndividualFund, TspAnalytics

prices = TspAnalytics()
prices.show_monthly_return_heatmap(TspIndividualFund.C_FUND)
```

## Price Change Bar Chart (Pandas)

```python
from datetime import date
from tsp import TspAnalytics

prices = TspAnalytics()
changes = prices.get_price_change_by_date_range_long(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 29),
)

ax = changes.set_index("fund")["change_percent"].plot(
    kind="bar",
    title="Year-to-Date Change",
    ylabel="Percent Change",
)
ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
```

## Fund Ranking Snapshot

```python
from tsp import TspAnalytics

prices = TspAnalytics()

# Rank by trailing 20-day returns
ranked = prices.get_fund_rankings(metric="trailing_return", period=20)
print(ranked.head(5))

# Visualize the top performers
prices.show_fund_rankings_chart(metric="trailing_return", period=20, top_n=5)
```
