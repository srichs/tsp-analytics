"""Validation helpers for common user and input parameter types."""

from datetime import date, time
from decimal import Decimal
from numbers import Real


class ValidationMixin:
    """Centralized validation utilities for dates, numbers, and ranges."""

    @staticmethod
    def _validate_date_range(start_date: date, end_date: date) -> None:
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_time_hour(value: time) -> None:
        if not isinstance(value, time):
            raise ValueError("time_hour must be a datetime.time value")

    @staticmethod
    def _validate_positive_float(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise ValueError(f"{name} must be a positive value")
        if value <= 0:
            raise ValueError(f"{name} must be a positive value")

    @staticmethod
    def _validate_year(value: int, name: str = "year") -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_month(value: int, name: str = "month") -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 12
        ):
            raise ValueError(f"{name} must be an integer between 1 and 12")

    @staticmethod
    def _validate_non_negative_float(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise ValueError(f"{name} must be a non-negative value")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative value")

    @staticmethod
    def _validate_numeric(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise ValueError(f"{name} must be a numeric value")

    @staticmethod
    def _validate_confidence(value: float, name: str = "confidence") -> None:
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise ValueError(f"{name} must be a float between 0 and 1")
        if not 0 < float(value) < 1:
            raise ValueError(f"{name} must be a float between 0 and 1")
