from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

import pandas as pd

from tsp import TspIndividualFund, TspLifecycleFund


@dataclass
class DummyResponse:
    content: bytes
    status_code: int = 200
    encoding: str | None = None
    apparent_encoding: str | None = None
    headers: dict | None = None

    def raise_for_status(self) -> None:
        return None


class DummySession:
    def __init__(
        self,
        responses: Iterable[DummyResponse | Exception],
        headers: dict | None = None,
    ) -> None:
        self._responses = iter(responses)
        self.headers = headers

    def get(self, url: str, timeout: float) -> DummyResponse:
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def build_yearly_price_dataframe() -> pd.DataFrame:
    dates = pd.to_datetime(["2022-12-30", "2023-12-29", "2024-12-30"])
    columns: dict[str, list[float] | pd.Series] = {"Date": dates}
    funds = [fund.value for fund in TspIndividualFund] + [
        fund.value for fund in TspLifecycleFund
    ]
    for idx, fund in enumerate(funds):
        base = Decimal("100") + Decimal(idx)
        columns[fund] = [
            float(base * (Decimal("1.1") ** offset)) for offset in range(len(dates))
        ]
    return pd.DataFrame(columns)


def build_monthly_price_dataframe() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    columns: dict[str, list[float] | pd.Series] = {"Date": dates}
    funds = [fund.value for fund in TspIndividualFund] + [
        fund.value for fund in TspLifecycleFund
    ]
    for idx, fund in enumerate(funds):
        base = Decimal("100") + Decimal(idx)
        columns[fund] = [
            float(base * (Decimal("1.1") ** offset)) for offset in range(len(dates))
        ]
    return pd.DataFrame(columns)


def build_minimal_price_dataframe() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Date": dates,
            TspIndividualFund.G_FUND.value: [100.0, 101.0],
            TspIndividualFund.C_FUND.value: [200.0, 201.0],
        }
    )
