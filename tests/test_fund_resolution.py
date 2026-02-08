import pandas as pd
import pytest


def test_resolve_periods_validates_inputs(tsp_price) -> None:
    assert tsp_price._resolve_periods([1, 3, 1]) == [1, 3]

    with pytest.raises(ValueError, match="positive integer"):
        tsp_price._resolve_periods(True)

    with pytest.raises(ValueError, match="positive integer"):
        tsp_price._resolve_periods([])


def test_resolve_funds_defaults_to_available(tsp_price) -> None:
    available = tsp_price.get_available_funds()
    assert tsp_price._resolve_funds(None) == available

    with pytest.raises(ValueError, match="must contain at least one fund"):
        tsp_price._resolve_funds([])


def test_resolve_weights_normalizes_and_checks_total(tsp_price) -> None:
    weights = {
        "G Fund": 0.7,
        "C Fund": 0.3,
        "F Fund": 0.0,
    }
    resolved = tsp_price._resolve_weights(weights)
    assert resolved == {"G Fund": pytest.approx(0.7), "C Fund": pytest.approx(0.3)}

    with pytest.raises(ValueError, match="sum to 1"):
        tsp_price._resolve_weights(
            {"G Fund": 0.8, "C Fund": 0.3}, normalize_weights=False
        )


def test_coerce_date_index_to_column(tsp_price) -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    dataframe = pd.DataFrame({"G Fund": [1.0, 2.0]}, index=dates)

    coerced = tsp_price._coerce_date_index_to_column(dataframe)

    assert list(coerced.columns)[0] == "Date"
    assert coerced["Date"].tolist() == list(dates)

    untouched = tsp_price._coerce_date_index_to_column(pd.DataFrame({"G Fund": [1.0]}))
    assert "Date" not in untouched.columns


def test_ensure_fund_available_validates_availability(tsp_price) -> None:
    assert tsp_price._ensure_fund_available("G Fund") == "G Fund"

    tsp_price.dataframe = tsp_price.dataframe.copy()
    tsp_price.dataframe["C Fund"] = None

    with pytest.raises(ValueError, match="fund not available"):
        tsp_price._ensure_fund_available("C Fund")


def test_ensure_funds_available_validates_multiple(tsp_price) -> None:
    tsp_price._ensure_funds_available(["G Fund", "C Fund"])

    with pytest.raises(ValueError, match="funds not available"):
        tsp_price._ensure_funds_available(["G Fund", "Unknown Fund"])
