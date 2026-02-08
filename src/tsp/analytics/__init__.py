"""Aggregate analytics mixin composed of TSP analytic helper classes."""

from datetime import date

from tsp.analytics.helpers import AnalyticsHelpersMixin
from tsp.analytics.prices import AnalyticsPricesMixin
from tsp.analytics.price_changes import AnalyticsPriceChangesMixin
from tsp.analytics.returns import AnalyticsReturnsMixin
from tsp.analytics.correlations import AnalyticsCorrelationMixin
from tsp.analytics.risk import AnalyticsRiskMixin
from tsp.analytics.reports import AnalyticsReportsMixin
from tsp.analytics.portfolio import AnalyticsPortfolioMixin


class AnalyticsMixin(
    AnalyticsPricesMixin,
    AnalyticsPriceChangesMixin,
    AnalyticsReturnsMixin,
    AnalyticsCorrelationMixin,
    AnalyticsRiskMixin,
    AnalyticsReportsMixin,
    AnalyticsPortfolioMixin,
    AnalyticsHelpersMixin,
):
    """Bundle price, return, risk, correlation, and report analytics helpers."""


__all__ = ["AnalyticsMixin", "date"]
