"""Fund metadata lookups for names, aliases, and availability."""

from functools import lru_cache
import re

from pandas import DataFrame

from tsp.funds import FundInput, TspIndividualFund, TspLifecycleFund


class FundMetadataMixin:
    """Expose fund metadata utilities for name resolution and listing."""

    @staticmethod
    def _normalize_fund_name(name: str) -> str:
        cleaned = re.sub(r"[_-]+", " ", name)
        cleaned = re.sub(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", " ", cleaned)
        return " ".join(cleaned.split()).strip().lower()

    @classmethod
    @lru_cache(maxsize=1)
    def _get_fund_name_map(cls) -> dict[str, str]:
        fund_map: dict[str, str] = {}
        for name in cls.ALL_FUNDS:
            normalized = cls._normalize_fund_name(name)
            fund_map[normalized] = name
            condensed = normalized.replace(" ", "")
            fund_map[condensed] = name
            if name in cls.INDIVIDUAL_FUNDS:
                short_code = normalized.split()[0]
                fund_map[short_code] = name
            if name in cls.LIFECYCLE_FUNDS and normalized.startswith("l "):
                lifecycle_suffix = normalized[2:]
                for prefix in ("lifecycle", "life cycle"):
                    lifecycle_alias = f"{prefix} {lifecycle_suffix}"
                    fund_map[lifecycle_alias] = name
                    fund_map[lifecycle_alias.replace(" ", "")] = name
            if not normalized.endswith(" fund"):
                with_fund = f"{normalized} fund"
                fund_map[with_fund] = name
                fund_map[with_fund.replace(" ", "")] = name
            if normalized.endswith(" fund"):
                no_fund = normalized[:-5]
                fund_map[no_fund] = name
                fund_map[no_fund.replace(" ", "")] = name
        return fund_map

    def _resolve_fund(self, fund: FundInput) -> str:
        fund_map = self._get_fund_name_map()
        if isinstance(fund, (TspIndividualFund, TspLifecycleFund)):
            fund_name = fund.value
        elif isinstance(fund, str):
            normalized = self._normalize_fund_name(fund)
            fund_name = fund_map.get(normalized, fund.strip())
        else:
            raise ValueError("fund must be a fund enum or fund name string")
        if fund_name not in self.ALL_FUNDS:
            raise ValueError(f"unknown fund: {fund_name}")
        return fund_name

    def get_fund_aliases(self) -> dict[str, list[str]]:
        """
        Gets the normalized fund name aliases supported by the library.

        Returns:
            dict[str, list[str]]: mapping of canonical fund names to alias strings.
                Alias strings are returned in normalized lowercase form.
        """
        fund_map = self._get_fund_name_map()
        aliases: dict[str, set[str]] = {fund: set() for fund in self.ALL_FUNDS}
        for alias, fund_name in fund_map.items():
            aliases[fund_name].add(alias)
        for fund_name in self.ALL_FUNDS:
            aliases[fund_name].add(self._normalize_fund_name(fund_name))
        return {fund: sorted(aliases[fund]) for fund in self.ALL_FUNDS}

    def get_fund_metadata(
        self, *, include_aliases: bool = True, include_availability: bool = True
    ) -> DataFrame:
        """
        Gets metadata about available funds, including fund category and aliases.

        Args:
            include_aliases (bool): whether to include normalized fund aliases.
            include_availability (bool): whether to include availability based on loaded data.

        Returns:
            DataFrame: dataframe indexed by fund with category, aliases, and availability.
        """
        aliases = self.get_fund_aliases() if include_aliases else {}
        available_funds: set[str] = set()
        if include_availability and self.dataframe is not None:
            available_funds = set(
                self._resolve_available_funds_from_dataframe(self.dataframe)
            )
        records: list[dict] = []
        for fund_name in self.ALL_FUNDS:
            record = {
                "fund": fund_name,
                "category": (
                    "individual" if fund_name in self.INDIVIDUAL_FUNDS else "lifecycle"
                ),
            }
            if include_aliases:
                record["aliases"] = aliases.get(fund_name, [])
            if include_availability:
                record["available"] = fund_name in available_funds
            records.append(record)
        return DataFrame(records).set_index("fund")

    def get_fund_metadata_dict(
        self, *, include_aliases: bool = True, include_availability: bool = True
    ) -> dict:
        """
        Gets metadata about funds as a JSON-friendly dictionary.

        Args:
            include_aliases (bool): whether to include normalized fund aliases.
            include_availability (bool): whether to include availability based on loaded data.

        Returns:
            dict: dictionary keyed by fund name with metadata payloads.
        """
        metadata = self.get_fund_metadata(
            include_aliases=include_aliases, include_availability=include_availability
        )
        payload: dict[str, dict] = {}
        for fund_name, row in metadata.iterrows():
            record = {"category": row["category"]}
            if include_aliases:
                record["aliases"] = list(row["aliases"])
            if include_availability:
                record["available"] = bool(row["available"])
            payload[fund_name] = record
        return {"funds": payload}

    @classmethod
    def _resolve_available_funds_from_dataframe(cls, dataframe: DataFrame) -> list[str]:
        available: list[str] = []
        for column in dataframe.columns:
            if column in cls.ALL_FUNDS:
                series = dataframe[column]
                if series.notna().any():
                    available.append(column)
        return available
