from datetime import time
from decimal import Decimal

import pytest


def test_validation_helpers_accept_valid_values(tsp_price) -> None:
    tsp_price._validate_positive_int(1, "periods")
    tsp_price._validate_positive_float(Decimal("1.5"), "rate")
    tsp_price._validate_time_hour(time(9, 0))
    tsp_price._validate_confidence(0.5)


def test_validation_helpers_reject_invalid_values(tsp_price) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tsp_price._validate_positive_int(0, "periods")

    with pytest.raises(ValueError, match="positive value"):
        tsp_price._validate_positive_float(-1.0, "rate")

    with pytest.raises(ValueError, match="time_hour"):
        tsp_price._validate_time_hour("09:00")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="between 0 and 1"):
        tsp_price._validate_confidence(1.0)
