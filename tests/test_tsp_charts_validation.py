from datetime import date

import pytest

from tsp import TspIndividualFund, TspAnalytics
from tests.helpers import build_minimal_price_dataframe


def test_show_pie_chart_requires_allocation_percent(tsp_price: TspAnalytics) -> None:
    allocation = tsp_price.create_allocation()
    with pytest.raises(
        ValueError, match="allocation_percent must contain at least one fund allocation"
    ):
        tsp_price.show_pie_chart(allocation)


def test_show_pie_chart_returns_axes(
    tsp_price: TspAnalytics, monkeypatch: pytest.MonkeyPatch
) -> None:
    allocation = tsp_price.create_allocation(g_shares=1)
    show_calls = []

    def fake_show() -> None:
        show_calls.append(True)

    monkeypatch.setattr("tsp.charts.plt.show", fake_show)
    fig, ax = tsp_price.show_pie_chart(allocation, show=False)
    assert fig == ax.figure
    assert show_calls == []


def test_price_chart_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    monkeypatch.setattr("tsp.charts.plt.show", lambda: None)

    fig, ax = tsp_price.show_individual_price_chart(show=False)
    assert fig == ax.figure
    fig, ax = tsp_price.show_individual_price_chart_by_date_range(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        show=False,
    )
    assert fig == ax.figure
    fig, ax = tsp_price.show_moving_average_chart(
        fund=TspIndividualFund.G_FUND,
        windows=[2],
        method="simple",
        show=False,
    )
    assert fig == ax.figure
    fig, ax = tsp_price.show_price_change_chart_as_of(date(2024, 1, 3), show=False)
    assert fig == ax.figure
    fig, ax = tsp_price.show_latest_price_changes_per_fund_chart(show=False)
    assert fig == ax.figure

    with pytest.raises(
        ValueError, match="no fund data available to plot lifecycle fund price chart"
    ):
        tsp_price.show_lifecycle_price_chart(show=False)

    with pytest.raises(
        ValueError, match="no fund data available to plot lifecycle fund price chart"
    ):
        tsp_price.show_lifecycle_price_chart_by_date_range(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            show=False,
        )

    with pytest.raises(
        ValueError, match="no data available to plot G Fund price chart"
    ):
        tsp_price.show_fund_price_chart_by_date_range(
            start_date=date(2024, 1, 5),
            end_date=date(2024, 1, 6),
            fund=TspIndividualFund.G_FUND,
            show=False,
        )


def test_risk_return_scatter_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    monkeypatch.setattr("tsp.charts.plt.show", lambda: None)
    fig, ax = tsp_price.show_risk_return_scatter(show=False)
    assert fig == ax.figure


def test_portfolio_drawdown_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    monkeypatch.setattr("tsp.charts.plt.show", lambda: None)
    fig, ax = tsp_price.show_portfolio_drawdown_chart(
        weights={TspIndividualFund.G_FUND: 0.5, TspIndividualFund.C_FUND: 0.5},
        show=False,
    )
    assert fig == ax.figure


def test_fund_coverage_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    tsp_price = TspAnalytics()
    dataframe = build_minimal_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None

    monkeypatch.setattr("tsp.charts.plt.show", lambda: None)
    fig, ax = tsp_price.show_fund_coverage_chart(show=False)
    assert fig == ax.figure
