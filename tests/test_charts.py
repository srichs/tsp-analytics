import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tsp import TspIndividualFund, TspAnalytics
from pandas import DataFrame
import pandas as pd

from tests.helpers import (
    build_minimal_price_dataframe,
    build_monthly_price_dataframe,
    build_yearly_price_dataframe,
)


def build_price_with_dataframe(dataframe: DataFrame) -> TspAnalytics:
    tsp_price = TspAnalytics()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None
    return tsp_price


def test_show_fund_price_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_fund_price_chart(TspIndividualFund.G_FUND, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_price_history_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_price_history_chart(
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        start_date=tsp_price.latest,
        show=False,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_moving_average_chart(
        fund=TspIndividualFund.G_FUND,
        windows=(2,),
        show=False,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_moving_average_chart_validates_inputs() -> None:
    tsp_price = build_price_with_dataframe(build_minimal_price_dataframe())

    with pytest.raises(ValueError, match="no data available to plot moving averages"):
        tsp_price.show_moving_average_chart(
            fund=TspIndividualFund.G_FUND,
            windows=(5,),
            show=False,
        )

    with pytest.raises(ValueError, match="method must be 'simple' or 'exponential'"):
        tsp_price.show_moving_average_chart(
            fund=TspIndividualFund.G_FUND,
            windows=(2,),
            method="invalid",
            show=False,
        )


def test_show_latest_price_charts_return_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_latest_prices_per_fund_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_current_prices_per_fund_chart(
        as_of=tsp_price.latest,
        sort_by="fund",
        show=False,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_latest_price_change_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_latest_price_changes_per_fund_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_recent_price_change_heatmap(days=3, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_price_recency_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_current_price_alerts_chart(
        metric="change_percent", change_threshold=0.05, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    with pytest.raises(ValueError, match='sort_by must be "price" or "fund"'):
        tsp_price.show_current_prices_per_fund_chart(sort_by="date", show=False)

    with pytest.raises(ValueError, match="metric must be one of"):
        tsp_price.show_current_price_alerts_chart(metric="unknown", show=False)


def test_show_risk_return_scatter_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_risk_return_scatter(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_current_price_dashboard_metric_chart_returns_figure(
    tsp_price: TspAnalytics,
) -> None:
    fig, ax = tsp_price.show_current_price_dashboard_metric_chart(
        metric="change_percent", show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_current_price_dashboard_metric_chart(
        metric="trailing_return", period=1, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    with pytest.raises(
        ValueError, match="period is required when metric is trailing_return"
    ):
        tsp_price.show_current_price_dashboard_metric_chart(
            metric="trailing_return", show=False
        )

    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        tsp_price.show_current_price_dashboard_metric_chart(
            metric="change_percent", top_n=0, show=False
        )

    with pytest.raises(ValueError, match="metric not available in dashboard"):
        tsp_price.show_current_price_dashboard_metric_chart(
            metric="unknown_metric", show=False
        )


def test_show_return_histogram_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_return_histogram_chart(
        TspIndividualFund.G_FUND, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "G Fund Daily Return Distribution"

    fig, ax = tsp_price.show_daily_return_histogram(
        TspIndividualFund.G_FUND, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_rolling_performance_summary_chart_returns_figure(
    tsp_price: TspAnalytics,
) -> None:
    fig, ax = tsp_price.show_rolling_performance_summary_chart(
        TspIndividualFund.G_FUND, window=2, trading_days=252, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_fund_rankings_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_fund_rankings_chart(
        metric="trailing_return", period=1, top_n=3, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_fund_rankings_chart(
        metric="change_percent", top_n=3, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        tsp_price.show_fund_rankings_chart(metric="change_percent", top_n=0, show=False)

    with pytest.raises(
        ValueError, match="period is required when ranking by trailing_return"
    ):
        tsp_price.show_fund_rankings_chart(metric="trailing_return", show=False)

    with pytest.raises(ValueError, match="unsupported ranking metric"):
        tsp_price.show_fund_rankings_chart(metric="unsupported_metric", show=False)


def test_show_trailing_returns_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_trailing_returns_chart(
        periods=[1, 5, 20],
        funds=[TspIndividualFund.G_FUND, TspIndividualFund.C_FUND],
        show=False,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_excess_returns_and_coverage_charts_return_figure(
    tsp_price: TspAnalytics,
) -> None:
    fig, ax = tsp_price.show_excess_returns_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = tsp_price.show_fund_coverage_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_correlation_pairs_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_correlation_pairs_chart(top_n=3, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    with pytest.raises(ValueError, match="min_abs_correlation must be between 0 and 1"):
        tsp_price.show_correlation_pairs_chart(min_abs_correlation=1.5, show=False)


def test_show_price_change_chart_as_of_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_price_change_chart_as_of(tsp_price.latest, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_show_additional_return_charts(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_cumulative_returns_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Cumulative Returns"

    fig, ax = tsp_price.show_normalized_price_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Normalized Prices"

    monthly_price = build_price_with_dataframe(build_monthly_price_dataframe())

    fig, ax = monthly_price.show_monthly_returns_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Monthly Returns"

    yearly_price = build_price_with_dataframe(build_yearly_price_dataframe())

    fig, ax = yearly_price.show_yearly_returns_chart(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Yearly Returns"


def test_show_rolling_correlation_chart_returns_figure(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_rolling_correlation_chart(
        TspIndividualFund.G_FUND, TspIndividualFund.C_FUND, window=2, show=False
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "G Fund vs C Fund Rolling Correlation (2D)"


def test_show_rolling_correlation_chart_validates_window(
    tsp_price: TspAnalytics,
) -> None:
    with pytest.raises(ValueError, match="window must be a positive integer"):
        tsp_price.show_rolling_correlation_chart(
            TspIndividualFund.G_FUND,
            TspIndividualFund.C_FUND,
            window=0,
            show=False,
        )


def test_show_return_histogram_chart_validates_bins(tsp_price: TspAnalytics) -> None:
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        tsp_price.show_return_histogram_chart(
            TspIndividualFund.G_FUND, bins=0, show=False
        )


def test_show_return_histogram_chart_requires_data() -> None:
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02"]),
            TspIndividualFund.G_FUND.value: [100.0],
        }
    )
    tsp_price = build_price_with_dataframe(dataframe)

    with pytest.raises(
        ValueError, match="no return data available for histogram chart"
    ):
        tsp_price.show_return_histogram_chart(TspIndividualFund.G_FUND, show=False)


def test_show_portfolio_charts_return_figure(tsp_price: TspAnalytics) -> None:
    weights = {
        TspIndividualFund.G_FUND: 0.6,
        TspIndividualFund.C_FUND: 0.4,
    }

    fig, ax = tsp_price.show_portfolio_value_chart(weights=weights, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Portfolio Value"

    fig, ax = tsp_price.show_portfolio_drawdown_chart(weights=weights, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Portfolio Drawdown"


def test_show_pie_chart_validates_allocation(tsp_price: TspAnalytics) -> None:
    fig, ax = tsp_price.show_pie_chart(
        {"allocation_percent": {"G Fund": 50.0, "C Fund": 50.0}},
        show=False,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    with pytest.raises(ValueError, match="allocation_percent must contain"):
        tsp_price.show_pie_chart({}, show=False)
