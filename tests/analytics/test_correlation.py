import pandas as pd
import pytest

from tsp import TspIndividualFund, TspAnalytics


def test_correlation_matrix_with_custom_data() -> None:
    prices = TspAnalytics(auto_update=False)
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            TspIndividualFund.G_FUND.value: [100.0, 110.0, 115.5],
            TspIndividualFund.C_FUND.value: [50.0, 55.0, 57.75],
        }
    )
    prices.load_dataframe(dataframe)
    correlation = prices.get_correlation_matrix()
    assert correlation.loc[
        TspIndividualFund.G_FUND.value, TspIndividualFund.C_FUND.value
    ] == pytest.approx(1.0)
    assert correlation.loc[
        TspIndividualFund.G_FUND.value, TspIndividualFund.G_FUND.value
    ] == pytest.approx(1.0)


def test_correlation_pairs_summary(tsp_price: TspAnalytics) -> None:
    pairs = tsp_price.get_correlation_pairs(top_n=3)
    assert len(pairs) == 3
    assert {"fund_a", "fund_b", "correlation", "abs_correlation"}.issubset(
        pairs.columns
    )
    assert (pairs["abs_correlation"] >= 0).all()

    payload = tsp_price.get_correlation_pairs_dict(top_n=2)
    assert payload["start_date"] == "2024-01-02"
    assert payload["end_date"] == "2024-01-04"
    assert len(payload["pairs"]) == 2
