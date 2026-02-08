"""Helpers for resolving fund inputs into canonical fund names."""

from collections.abc import Iterable, Mapping

from tsp.funds import FundInput, TspIndividualFund, TspLifecycleFund


class FundResolutionMixin:
    """Normalize and validate fund inputs for analytics and lookups."""

    def _resolve_funds(self, funds: Iterable[FundInput] | None) -> list[str]:
        if funds is None:
            available = self.get_available_funds()
            if not available:
                raise ValueError("no available funds in data")
            return available
        if isinstance(funds, (TspIndividualFund, TspLifecycleFund, str)):
            funds = [funds]
        elif not isinstance(funds, Iterable):
            raise ValueError(
                "funds must be an iterable of fund enums or fund name strings"
            )
        fund_list = []
        for fund in funds:
            fund_name = self._resolve_fund(fund)
            fund_list.append(fund_name)
        if not fund_list:
            raise ValueError("funds must contain at least one fund")
        return list(dict.fromkeys(fund_list))

    def _resolve_periods(self, periods: Iterable[int] | int) -> list[int]:
        if isinstance(periods, bool):
            raise ValueError(
                "periods must be a positive integer or an iterable of positive integers"
            )
        if isinstance(periods, int):
            periods_list = [periods]
        else:
            if isinstance(periods, (str, bytes)) or not isinstance(periods, Iterable):
                raise ValueError(
                    "periods must be a positive integer or an iterable of positive integers"
                )
            periods_list = list(periods)
        if not periods_list:
            raise ValueError("periods must contain at least one positive integer")
        unique_periods: list[int] = []
        for period in periods_list:
            self._validate_positive_int(period, "periods")
            if period not in unique_periods:
                unique_periods.append(period)
        return unique_periods

    def _resolve_weights(
        self, weights: Mapping[FundInput, float], normalize_weights: bool = True
    ) -> dict[str, float]:
        self._ensure_dataframe()
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("weights must be a non-empty mapping of funds to weights")
        resolved: dict[str, float] = {}
        for fund, weight in weights.items():
            if not isinstance(fund, (TspIndividualFund, TspLifecycleFund, str)):
                raise ValueError(
                    "weights must map fund enums or fund name strings to weight values"
                )
            fund_name = self._resolve_fund(fund)
            if fund_name not in self.ALL_FUNDS:
                raise ValueError(f"unknown fund: {fund_name}")
            self._validate_non_negative_float(weight, f"{fund_name} weight")
            if float(weight) > 0:
                resolved[fund_name] = float(weight)
        if not resolved:
            raise ValueError("weights must include at least one positive value")
        self._ensure_funds_available(resolved.keys())
        total = sum(resolved.values())
        if normalize_weights:
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            resolved = {fund: value / total for fund, value in resolved.items()}
        else:
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    "weights must sum to 1 when normalize_weights is False"
                )
        return resolved

    def _ensure_fund_available(self, fund: FundInput) -> str:
        fund_name = self._resolve_fund(fund)
        available_funds = self.get_available_funds()
        if fund_name not in available_funds:
            raise ValueError(f"fund not available in data: {fund_name}")
        return fund_name

    def _ensure_funds_available(self, funds: Iterable[str]) -> None:
        available_funds = set(self.get_available_funds())
        missing = [fund for fund in funds if fund not in available_funds]
        if missing:
            raise ValueError(f'funds not available in data: {", ".join(missing)}')
