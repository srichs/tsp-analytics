"""Fund enums and shared fund input types for the TSP library."""

from enum import Enum


class TspIndividualFund(Enum):
    """Enumeration of TSP individual funds with canonical display names."""

    G_FUND = "G Fund"
    F_FUND = "F Fund"
    C_FUND = "C Fund"
    S_FUND = "S Fund"
    I_FUND = "I Fund"


class TspLifecycleFund(Enum):
    """Enumeration of TSP lifecycle funds and their target dates."""

    L_INCOME = "L Income"
    L_2030 = "L 2030"
    L_2035 = "L 2035"
    L_2040 = "L 2040"
    L_2045 = "L 2045"
    L_2050 = "L 2050"
    L_2055 = "L 2055"
    L_2060 = "L 2060"
    L_2065 = "L 2065"
    L_2070 = "L 2070"
    L_2075 = "L 2075"


FundInput = TspIndividualFund | TspLifecycleFund | str
