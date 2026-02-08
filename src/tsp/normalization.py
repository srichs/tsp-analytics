"""Normalization and validation helpers for raw TSP data."""

from datetime import date, datetime
from pandas import DataFrame, Series, to_datetime, to_numeric
from pandas.api.types import is_datetime64_any_dtype


class NormalizationMixin:
    """Normalize CSV data into standardized columns and validated values."""

    @classmethod
    def _normalize_columns(cls, dataframe: DataFrame) -> DataFrame:
        fund_map = cls._get_fund_name_map()
        column_map: dict = {}
        for column in dataframe.columns:
            if isinstance(column, str):
                trimmed = column.strip().lstrip("\ufeff").strip()
                if trimmed.lower() == "date":
                    column_map[column] = "Date"
                else:
                    normalized = cls._normalize_fund_name(trimmed)
                    column_map[column] = fund_map.get(normalized, trimmed)
            else:
                column_map[column] = column
        return dataframe.rename(columns=column_map)

    def _normalize_dataframe(self, dataframe: DataFrame) -> DataFrame:
        dataframe = dataframe.dropna(how="all").copy()
        dataframe = self._normalize_columns(dataframe)
        if "Date" not in dataframe.columns:
            dataframe = self._coerce_date_index_to_column(dataframe)
        if dataframe.empty:
            raise ValueError("TSP data is empty")
        self._validate_dataframe(dataframe)
        self._validate_required_funds(dataframe)
        fund_columns = [fund for fund in self.ALL_FUNDS if fund in dataframe.columns]
        dataframe = dataframe[["Date", *fund_columns]]
        date_values = dataframe["Date"].astype(str).str.strip()
        try:
            parsed_dates = to_datetime(date_values, format="mixed", errors="coerce")
        except TypeError:
            parsed_dates = to_datetime(date_values, errors="coerce")
        if getattr(parsed_dates.dt, "tz", None) is not None:
            parsed_dates = parsed_dates.dt.tz_localize(None)
        parsed_dates = parsed_dates.dt.normalize()
        dataframe["Date"] = parsed_dates
        dataframe.loc[:, fund_columns] = dataframe[fund_columns].apply(
            to_numeric, errors="coerce"
        )
        dataframe = dataframe.dropna(subset=["Date"])
        dataframe = dataframe.dropna(subset=fund_columns, how="all")
        if (dataframe[fund_columns] < 0).any().any():
            raise ValueError("TSP price data contains negative values")
        if dataframe.empty:
            raise ValueError("TSP data contains no usable rows after normalization")
        dataframe = dataframe.sort_values("Date", ascending=True)
        dataframe = dataframe.drop_duplicates(subset=["Date"], keep="last")
        return dataframe.reset_index(drop=True)

    def _validate_required_funds(self, dataframe: DataFrame) -> None:
        required_funds = getattr(self, "required_funds", None)
        if not required_funds:
            return
        missing = [fund for fund in required_funds if fund not in dataframe.columns]
        if missing:
            raise ValueError(f'TSP data missing required funds: {", ".join(missing)}')
        empty_funds = [fund for fund in required_funds if dataframe[fund].isna().all()]
        if empty_funds:
            raise ValueError(
                f'TSP data has no values for required funds: {", ".join(empty_funds)}'
            )

    @staticmethod
    def _coerce_date_index_to_column(dataframe: DataFrame) -> DataFrame:
        index = dataframe.index
        index_name = str(index.name).strip().lower() if index.name is not None else ""
        if (
            is_datetime64_any_dtype(index)
            or index_name == "date"
            or (
                len(index) > 0
                and all(isinstance(value, (datetime, date)) for value in index[:5])
            )
        ):
            dataframe = dataframe.copy()
            dataframe.insert(0, "Date", index)
            return dataframe.reset_index(drop=True)
        return dataframe

    def _assign_dataframe(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe
        self.current = self.dataframe.loc[self.dataframe["Date"].idxmax()]
        self.latest = self.current["Date"].date()

    def _get_latest_row(self) -> Series:
        self._ensure_dataframe()
        latest_timestamp = self.dataframe["Date"].max()
        if self.current is None or self.current.get("Date") != latest_timestamp:
            self.current = self.dataframe.loc[self.dataframe["Date"].idxmax()]
        self.latest = latest_timestamp.date()
        return self.current

    def _ensure_dataframe(self) -> None:
        if self.dataframe is None or self.current is None or self.latest is None:
            self.check()
        if self.dataframe is None or self.current is None or self.latest is None:
            raise RuntimeError("TSP data is not available")

    def _validate_dataframe(self, dataframe: DataFrame) -> None:
        if "Date" not in dataframe.columns:
            raise ValueError("TSP data must include a Date column")
        fund_columns = [
            column for column in dataframe.columns if column in self.ALL_FUNDS
        ]
        if not fund_columns:
            raise ValueError("TSP data must include at least one fund column")
        if dataframe["Date"].isna().all():
            raise ValueError("TSP data contains no valid Date values")

    @staticmethod
    def _validate_chart_dataframe(dataframe: DataFrame, chart_name: str) -> None:
        if dataframe.empty:
            raise ValueError(f"no data available to plot {chart_name}")
        fund_columns = [column for column in dataframe.columns if column != "Date"]
        if not fund_columns:
            raise ValueError(f"no fund data available to plot {chart_name}")

    def _filter_by_date_range(
        self, start_date: date, end_date: date, dataframe: DataFrame | None = None
    ) -> DataFrame:
        self._ensure_dataframe()
        self._validate_date_range(start_date, end_date)
        df = dataframe if dataframe is not None else self.dataframe
        start = to_datetime(start_date).normalize()
        end = to_datetime(end_date).normalize()
        return df[(df["Date"] >= start) & (df["Date"] <= end)]
