"""Shared analytics helper utilities and output formatting helpers."""

from datetime import date, datetime
from numbers import Real

from pandas import DataFrame, Series, isna


class AnalyticsHelpersMixin:
    """Provide reusable calculations and formatting for analytics mixins."""

    @staticmethod
    def _resolve_first_valid_prices(price_df: DataFrame) -> Series:
        first_valid = price_df.apply(
            lambda series: (
                series.dropna().iloc[0] if not series.dropna().empty else float("nan")
            )
        )
        return first_valid

    def _calculate_max_drawdown(self, prices: Series) -> float:
        cumulative_max = prices.cummax()
        drawdown = (prices / cumulative_max) - 1
        return float(drawdown.min())

    def _calculate_performance_summary(self, prices: Series, trading_days: int) -> dict:
        returns = prices.pct_change(fill_method=None).dropna(how="all")
        if returns.empty:
            return {
                "total_return": float("nan"),
                "annualized_return": float("nan"),
                "annualized_volatility": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
            }
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        annualized_volatility = returns.std() * (trading_days**0.5)
        sharpe_ratio = (
            annualized_return / annualized_volatility
            if annualized_volatility
            else float("nan")
        )
        max_drawdown = self._calculate_max_drawdown(prices)

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
        }

    @staticmethod
    def _to_daily_rate(annual_rate: float, trading_days: int) -> float:
        return (1 + float(annual_rate)) ** (1 / trading_days) - 1

    @staticmethod
    def _format_date_for_output(value: date, date_format: str | None) -> str | date:
        if date_format is None:
            return value
        if not isinstance(date_format, str) or not date_format.strip():
            raise ValueError("date_format must be a non-empty string or None")
        if date_format.lower() == "iso":
            return value.isoformat()
        return value.strftime(date_format)

    @staticmethod
    def _format_datetime_for_output(
        value: datetime, datetime_format: str | None
    ) -> str | datetime:
        if datetime_format is None:
            return value
        if not isinstance(datetime_format, str) or not datetime_format.strip():
            raise ValueError("datetime_format must be a non-empty string or None")
        if datetime_format.lower() == "iso":
            return value.isoformat()
        return value.strftime(datetime_format)

    @staticmethod
    def _format_numeric_for_output(value: float | int | None) -> float | None:
        if isna(value):
            return None
        return float(value)

    def _format_value_for_output(
        self, value: object, date_format: str | None = None
    ) -> object:
        if isinstance(value, datetime):
            value = value.date()
        if isinstance(value, date):
            return self._format_date_for_output(value, date_format)
        if isinstance(value, Real) or isna(value):
            return self._format_numeric_for_output(value)
        return value

    def _format_dataframe_for_output(
        self, dataframe: DataFrame, date_format: str | None = None
    ) -> dict[str, dict]:
        payload: dict[str, dict] = {}
        for index_value, row in dataframe.iterrows():
            payload[str(index_value)] = {
                column: self._format_value_for_output(row[column], date_format)
                for column in dataframe.columns
            }
        return payload

    def _format_long_dataframe_for_output(
        self, dataframe: DataFrame, date_format: str | None = None
    ) -> list[dict]:
        records: list[dict] = []
        for _, row in dataframe.iterrows():
            records.append(
                {
                    column: self._format_value_for_output(row[column], date_format)
                    for column in dataframe.columns
                }
            )
        return records
