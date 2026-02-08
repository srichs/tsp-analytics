"""Allocation helpers for translating fund shares into totals and percentages."""

from collections.abc import Mapping
from decimal import Decimal

from tsp.funds import FundInput, TspIndividualFund, TspLifecycleFund


class AllocationMixin:
    """Build allocation summaries from fund share counts."""

    def _add_fund_allocation(
        self, shares: Decimal, price: Decimal | None, subtotal: Decimal
    ) -> dict:
        return {
            "shares": str(shares),
            "price": str(price) if price is not None else None,
            "subtotal": f"{subtotal:,.2f}",
        }

    def _build_allocation(self, shares_by_fund: dict[str, Decimal]) -> dict:
        self._ensure_dataframe()
        latest_row = self._get_latest_row()
        available_funds = set(self.get_available_funds())
        total = Decimal("0")
        allocation = {
            "date": f"{self.latest}",
            "allocation_percent": {},
            "total": str(total),
        }
        subtotals: dict[str, Decimal] = {}

        for fund_name in self.ALL_FUNDS:
            shares = shares_by_fund.get(fund_name, Decimal("0"))
            if shares > 0 and fund_name not in available_funds:
                raise ValueError(f"fund not available in data: {fund_name}")
            price = (
                Decimal(str(latest_row[fund_name]))
                if fund_name in available_funds
                else None
            )
            subtotal = Decimal("0") if price is None else shares * price
            allocation[fund_name] = self._add_fund_allocation(shares, price, subtotal)
            subtotals[fund_name] = subtotal
            total = total + subtotal

        if total > Decimal("0"):
            for fund_name in self.ALL_FUNDS:
                info = allocation[fund_name]
                percent = subtotals[fund_name] / total * Decimal(100)
                info["percent"] = f"{percent:.2f}"

                if percent > Decimal("0"):
                    allocation["allocation_percent"][fund_name] = f"{percent:.2f}"
        else:
            for fund_name in self.ALL_FUNDS:
                allocation[fund_name]["percent"] = "0.00"

        allocation["total"] = f"{total:,.2f}"
        return allocation

    def create_allocation(
        self,
        g_shares: float = 0.0,
        f_shares: float = 0.0,
        c_shares: float = 0.0,
        s_shares: float = 0.0,
        i_shares: float = 0.0,
        l_income_shares: float = 0.0,
        l_2030_shares: float = 0.0,
        l_2035_shares: float = 0.0,
        l_2040_shares: float = 0.0,
        l_2045_shares: float = 0.0,
        l_2050_shares: float = 0.0,
        l_2055_shares: float = 0.0,
        l_2060_shares: float = 0.0,
        l_2065_shares: float = 0.0,
        l_2070_shares: float = 0.0,
        l_2075_shares: float = 0.0,
    ) -> dict:
        """
        Creates a dictionary that represents the allocation given the provided shares.

        Args:
            g_shares (float): the number of shares in the G fund.
            f_shares (float): the number of shares in the F fund.
            c_shares (float): the number of shares in the C fund.
            s_shares (float): the number of shares in the S fund.
            i_shares (float): the number of shares in the I fund.
            l_income_shares (float): the number of shares in the L Income fund.
            l_2030_shares (float): the number of shares in the L 2030 fund.
            l_2035_shares (float): the number of shares in the L 2035 fund.
            l_2040_shares (float): the number of shares in the L 2040 fund.
            l_2045_shares (float): the number of shares in the L 2045 fund.
            l_2050_shares (float): the number of shares in the L 2050 fund.
            l_2055_shares (float): the number of shares in the L 2055 fund.
            l_2060_shares (float): the number of shares in the L 2060 fund.
            l_2065_shares (float): the number of shares in the L 2065 fund.
            l_2070_shares (float): the number of shares in the L 2070 fund.
            l_2075_shares (float): the number of shares in the L 2075 fund.

        Returns:
            dict: the allocation of the user's shares and prices.

        Notes:
            If a fund is missing from the cached CSV and the share count is zero, the allocation
            includes the fund with a `None` price and a zero subtotal. Non-zero share counts for
            missing funds raise a `ValueError`.
        """
        self.logger.debug("create_allocation()")
        self._validate_non_negative_float(g_shares, "g_shares")
        self._validate_non_negative_float(f_shares, "f_shares")
        self._validate_non_negative_float(c_shares, "c_shares")
        self._validate_non_negative_float(s_shares, "s_shares")
        self._validate_non_negative_float(i_shares, "i_shares")
        self._validate_non_negative_float(l_income_shares, "l_income_shares")
        self._validate_non_negative_float(l_2030_shares, "l_2030_shares")
        self._validate_non_negative_float(l_2035_shares, "l_2035_shares")
        self._validate_non_negative_float(l_2040_shares, "l_2040_shares")
        self._validate_non_negative_float(l_2045_shares, "l_2045_shares")
        self._validate_non_negative_float(l_2050_shares, "l_2050_shares")
        self._validate_non_negative_float(l_2055_shares, "l_2055_shares")
        self._validate_non_negative_float(l_2060_shares, "l_2060_shares")
        self._validate_non_negative_float(l_2065_shares, "l_2065_shares")
        self._validate_non_negative_float(l_2070_shares, "l_2070_shares")
        self._validate_non_negative_float(l_2075_shares, "l_2075_shares")
        shares_by_fund = {
            TspIndividualFund.G_FUND.value: Decimal(str(g_shares)),
            TspIndividualFund.F_FUND.value: Decimal(str(f_shares)),
            TspIndividualFund.C_FUND.value: Decimal(str(c_shares)),
            TspIndividualFund.S_FUND.value: Decimal(str(s_shares)),
            TspIndividualFund.I_FUND.value: Decimal(str(i_shares)),
            TspLifecycleFund.L_INCOME.value: Decimal(str(l_income_shares)),
            TspLifecycleFund.L_2030.value: Decimal(str(l_2030_shares)),
            TspLifecycleFund.L_2035.value: Decimal(str(l_2035_shares)),
            TspLifecycleFund.L_2040.value: Decimal(str(l_2040_shares)),
            TspLifecycleFund.L_2045.value: Decimal(str(l_2045_shares)),
            TspLifecycleFund.L_2050.value: Decimal(str(l_2050_shares)),
            TspLifecycleFund.L_2055.value: Decimal(str(l_2055_shares)),
            TspLifecycleFund.L_2060.value: Decimal(str(l_2060_shares)),
            TspLifecycleFund.L_2065.value: Decimal(str(l_2065_shares)),
            TspLifecycleFund.L_2070.value: Decimal(str(l_2070_shares)),
            TspLifecycleFund.L_2075.value: Decimal(str(l_2075_shares)),
        }
        return self._build_allocation(shares_by_fund)

    def create_allocation_from_shares(self, shares: Mapping[FundInput, float]) -> dict:
        """
        Creates a dictionary that represents the allocation given a mapping of shares.

        Args:
            shares (Mapping): mapping of fund enums or fund name strings to share counts.

        Returns:
            dict: the allocation of the user's shares and prices.

        Notes:
            Missing funds with zero shares return `None` prices and zero subtotals. Non-zero share
            counts for missing funds raise a `ValueError`.
        """
        self.logger.debug("create_allocation_from_shares()")
        if not isinstance(shares, Mapping) or not shares:
            raise ValueError(
                "shares must be a non-empty mapping of funds to share counts"
            )
        shares_by_fund: dict[str, Decimal] = {}
        for fund, share in shares.items():
            if not isinstance(fund, (TspIndividualFund, TspLifecycleFund, str)):
                raise ValueError(
                    "shares must map fund enums or fund name strings to share values"
                )
            fund_name = self._resolve_fund(fund)
            self._validate_non_negative_float(share, f"{fund_name} shares")
            shares_by_fund[fund_name] = Decimal(str(share))
        return self._build_allocation(shares_by_fund)
