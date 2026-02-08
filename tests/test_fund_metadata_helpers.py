import pandas as pd

from tsp.fund_metadata import FundMetadataMixin


class DummyMetadata(FundMetadataMixin):
    INDIVIDUAL_FUNDS = ["G Fund"]
    LIFECYCLE_FUNDS = ["L 2050"]
    ALL_FUNDS = INDIVIDUAL_FUNDS + LIFECYCLE_FUNDS


def test_normalize_fund_name_cleans_tokens() -> None:
    assert DummyMetadata._normalize_fund_name(" G-Fund ") == "g fund"
    assert DummyMetadata._normalize_fund_name("L2050") == "l 2050"
    assert DummyMetadata._normalize_fund_name("l__2050") == "l 2050"


def test_get_fund_name_map_includes_aliases() -> None:
    fund_map = DummyMetadata._get_fund_name_map()

    assert fund_map["g"] == "G Fund"
    assert fund_map["gfund"] == "G Fund"
    assert fund_map["l 2050"] == "L 2050"
    assert fund_map["lifecycle 2050"] == "L 2050"
    assert fund_map["lifecycle2050"] == "L 2050"
    assert fund_map["life cycle 2050"] == "L 2050"


def test_resolve_available_funds_from_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "G Fund": [100.0, 101.0],
            "L 2050": [None, None],
        }
    )

    available = DummyMetadata._resolve_available_funds_from_dataframe(dataframe)

    assert available == ["G Fund"]
