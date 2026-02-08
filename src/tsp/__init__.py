"""Public package exports for the TSP pricing and analytics library."""

from importlib.metadata import PackageNotFoundError, version as _version

from tsp.data_providers import CSVDataProvider, DataFetchResult, RequestsCSVDataProvider
from tsp.tsp import TspIndividualFund, TspLifecycleFund, TspAnalytics

try:
    __version__ = _version("tsp-analytics")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "CSVDataProvider",
    "DataFetchResult",
    "RequestsCSVDataProvider",
    "TspIndividualFund",
    "TspLifecycleFund",
    "TspAnalytics",
    "__version__",
]
