import os
import sys
import warnings
from decimal import Decimal
import importlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

tsp_module = importlib.import_module("tsp")
TspIndividualFund = tsp_module.TspIndividualFund
TspLifecycleFund = tsp_module.TspLifecycleFund
TspAnalytics = tsp_module.TspAnalytics


os.environ.setdefault("TSP_DATA_DIR", str(ROOT / ".tsp_test_data"))
matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="'mode' parameter is deprecated and will be removed in Pillow 13.*",
    category=DeprecationWarning,
    module=r"matplotlib\.backends\._backend_tk",
)


def _build_price_dataframe() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    columns = {"Date": dates}
    funds = [fund.value for fund in TspIndividualFund] + [
        fund.value for fund in TspLifecycleFund
    ]
    for idx, fund in enumerate(funds):
        base = Decimal("10") + Decimal(idx)
        columns[fund] = [float(base + Decimal(offset)) for offset in range(len(dates))]
    return pd.DataFrame(columns)


@pytest.fixture()
def tsp_price() -> TspAnalytics:
    tsp_price = TspAnalytics()
    dataframe = _build_price_dataframe()
    tsp_price.dataframe = dataframe
    tsp_price.current = dataframe.loc[dataframe["Date"].idxmax()]
    tsp_price.latest = tsp_price.current["Date"].date()
    tsp_price.check = lambda: None
    return tsp_price


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    yield
    plt.close("all")
