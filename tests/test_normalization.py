import pandas as pd
import pytest

from tsp.fund_metadata import FundMetadataMixin
from tsp.normalization import NormalizationMixin
from tsp.validation import ValidationMixin


class DummyNormalizer(NormalizationMixin, FundMetadataMixin, ValidationMixin):
    INDIVIDUAL_FUNDS = ["G Fund", "C Fund"]
    LIFECYCLE_FUNDS = []
    ALL_FUNDS = INDIVIDUAL_FUNDS + LIFECYCLE_FUNDS

    def __init__(self, required_funds=None) -> None:
        self.required_funds = required_funds
        self.dataframe = None
        self.current = None
        self.latest = None

    def check(self) -> None:
        return None


def test_normalize_dataframe_trims_and_deduplicates() -> None:
    normalizer = DummyNormalizer()
    raw = pd.DataFrame(
        {
            " date ": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "g_fund": [100.0, 101.0, 102.0],
            "C-Fund": [200.0, 201.0, 202.0],
        }
    )

    normalized = normalizer._normalize_dataframe(raw)

    assert list(normalized.columns) == ["Date", "G Fund", "C Fund"]
    assert normalized["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert normalized.loc[0, "G Fund"] == 102.0
    assert normalized.loc[0, "C Fund"] == 202.0


def test_normalize_dataframe_coerces_date_index() -> None:
    normalizer = DummyNormalizer()
    index = pd.to_datetime(["2024-01-03", "2024-01-04"])
    raw = pd.DataFrame({"G Fund": [100.0, 101.0]}, index=index)

    normalized = normalizer._normalize_dataframe(raw)

    assert "Date" in normalized.columns
    assert normalized["Date"].tolist() == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]


def test_normalize_dataframe_rejects_negative_values() -> None:
    normalizer = DummyNormalizer()
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "G Fund": [-1.0],
            "C Fund": [1.0],
        }
    )

    with pytest.raises(ValueError, match="negative values"):
        normalizer._normalize_dataframe(raw)


def test_normalize_dataframe_requires_fund_presence() -> None:
    normalizer = DummyNormalizer(required_funds=["C Fund"])
    raw = pd.DataFrame({"Date": ["2024-01-02"], "G Fund": [100.0]})

    with pytest.raises(ValueError, match="missing required funds"):
        normalizer._normalize_dataframe(raw)


def test_validate_chart_dataframe_raises_for_empty_or_missing_funds() -> None:
    with pytest.raises(ValueError, match="no data available"):
        DummyNormalizer._validate_chart_dataframe(pd.DataFrame(), "example")

    with pytest.raises(ValueError, match="no fund data available"):
        DummyNormalizer._validate_chart_dataframe(
            pd.DataFrame({"Date": [pd.Timestamp("2024-01-02")]}),
            "example",
        )


def test_filter_by_date_range_uses_boundaries() -> None:
    normalizer = DummyNormalizer()
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "G Fund": [100.0, 101.0, 102.0],
            "C Fund": [200.0, 201.0, 202.0],
        }
    )
    normalized = normalizer._normalize_dataframe(raw)
    normalizer._assign_dataframe(normalized)

    subset = normalizer._filter_by_date_range(
        start_date=pd.Timestamp("2024-01-02").date(),
        end_date=pd.Timestamp("2024-01-03").date(),
    )

    assert subset["Date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]

    with pytest.raises(ValueError, match="start_date must be"):
        normalizer._filter_by_date_range(
            start_date=pd.Timestamp("2024-01-04").date(),
            end_date=pd.Timestamp("2024-01-03").date(),
        )


def test_coerce_date_index_to_column_ignores_non_date_index() -> None:
    normalizer = DummyNormalizer()
    raw = pd.DataFrame({"G Fund": [100.0, 101.0]}, index=[1, 2])

    coerced = normalizer._coerce_date_index_to_column(raw)

    assert coerced is raw
