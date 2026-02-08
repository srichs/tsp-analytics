import datetime as dt
import pytest

from tsp.validation import ValidationMixin


class DummyValidation(ValidationMixin):
    pass


def test_validate_date_range_rejects_inverted() -> None:
    with pytest.raises(ValueError, match="start_date must be"):
        DummyValidation._validate_date_range(dt.date(2024, 1, 2), dt.date(2024, 1, 1))


def test_validate_time_hour_requires_time() -> None:
    with pytest.raises(ValueError, match="time_hour must be"):
        DummyValidation._validate_time_hour("nope")  # type: ignore[arg-type]


def test_validate_confidence_bounds() -> None:
    for value in (0, 1, -0.1, 1.1):
        with pytest.raises(ValueError, match="float between 0 and 1"):
            DummyValidation._validate_confidence(value)

    DummyValidation._validate_confidence(0.5)


def test_validate_numeric_rejects_bool() -> None:
    with pytest.raises(ValueError, match="numeric value"):
        DummyValidation._validate_numeric(True, "value")
