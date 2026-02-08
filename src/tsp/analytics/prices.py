"""Price lookup and price-series accessors for TSP funds."""

from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pandas import DataFrame, Series, bdate_range, concat, isna, to_datetime

from tsp.funds import FundInput


class AnalyticsPricesMixin:
    """Expose price lookup and date-range helpers for fund price data."""

    def get_price(self, fund: FundInput) -> Decimal:
        """
        Gets the latest price of the provided fund.

        Args:
            fund (FundInput): enum for the type of Fund to retrieve the price for.

        Returns:
            Decimal: price of the fund.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        latest_row = self._get_latest_row()
        latest_value = latest_row[fund_name]
        if not isna(latest_value):
            return Decimal(str(latest_value))
        series = (
            self.dataframe[["Date", fund_name]]
            .dropna(subset=[fund_name])
            .sort_values("Date")
        )
        if series.empty:
            raise ValueError(f"no price data available for fund: {fund_name}")
        return Decimal(str(series.iloc[-1][fund_name]))

    def get_prices_by_date(self, date: date) -> DataFrame:
        """
        Gets the prices for each fund on the specified date.

        Args:
            date (date): the date to retrieve the fund prices for.

        Returns:
            DataFrame: dataframe with data for the specified date.
        """
        self._ensure_dataframe()
        date_value = to_datetime(date).normalize()
        prices = self.dataframe.loc[self.dataframe["Date"] == date_value]
        return prices

    def get_prices_by_date_range(self, start_date: date, end_date: date) -> DataFrame:
        """
        Gets the prices for each fund between the specified dates (inclusive).

        Args:
            start_date (date): the start date to retrieve fund prices for.
            end_date (date): the end date to retrieve the fund prices for.

        Returns:
            DataFrame: dataframe with data for the specified date range.
        """
        return self._filter_by_date_range(start_date, end_date)

    def get_fund_price_by_date(self, fund: FundInput, date: date) -> Decimal | None:
        """
        Gets the prices for the specified fund on the specified date.

        Args:
            fund (FundInput): the fund to retrieve the price for.
            date (date): the date to retrieve the fund price for.

        Returns:
            Decimal | None: price of the fund on the specified date, if available.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        date_value = to_datetime(date).normalize()
        prices = self.dataframe.loc[self.dataframe["Date"] == date_value, fund_name]
        if prices.empty:
            return None
        return Decimal(str(prices.iloc[0]))

    def get_fund_prices_by_date_range(
        self, fund: FundInput, start_date: date, end_date: date
    ) -> DataFrame:
        """
        Gets the prices for the specified fund between the specified dates (inclusive).

        Args:
            fund (FundInput): the fund to retrieve the data for.
            start_date (date): the start date to retrieve the fund prices for.
            end_date (date): the end date to retrieve the fund prices for.

        Returns:
            DataFrame: dataframe with data for the specified date range.
        """
        self._ensure_dataframe()
        fund_name = self._ensure_fund_available(fund)
        df = self.dataframe[["Date", fund_name]].dropna(how="all")
        return self._filter_by_date_range(start_date, end_date, dataframe=df)

    def get_prices_by_month(self, year: int, month: int) -> DataFrame:
        """
        Gets the prices for each fund for each day in the specified month.

        Args:
            year (int): the year to retrieve the fund prices for.
            month (int): the month to retrieve the fund prices for.

        Returns:
            DataFrame: dataframe with data for the specified month.
        """
        self._ensure_dataframe()
        self._validate_year(year)
        self._validate_month(month)
        prices = self.dataframe.loc[
            (self.dataframe["Date"].dt.year == year)
            & (self.dataframe["Date"].dt.month == month)
        ]
        return prices

    def get_fund_prices_by_month(
        self, fund: FundInput, year: int, month: int
    ) -> Series:
        """
        Gets the prices for the specified fund for each day in the specified month.

        Args:
            fund (FundInput): the fund to retrieve the data for.
            year (int): the year to retrieve the fund prices for.
            month (int): the month to retrieve the fund prices for.

        Returns:
            Series: series with data for the specified month.
        """
        self._ensure_dataframe()
        self._validate_year(year)
        self._validate_month(month)
        fund_name = self._ensure_fund_available(fund)
        price = self.dataframe.loc[
            (self.dataframe["Date"].dt.year == year)
            & (self.dataframe["Date"].dt.month == month),
            fund_name,
        ]
        return price

    def get_prices_by_year(self, year: int) -> DataFrame:
        """
        Gets the prices for each fund for each day in the specified year.

        Args:
            year (int): the year to retrieve the fund prices for.

        Returns:
            DataFrame: dataframe with data for the specified year.
        """
        self._ensure_dataframe()
        self._validate_year(year)
        prices = self.dataframe.loc[self.dataframe["Date"].dt.year == year]
        return prices

    def get_fund_prices_by_year(self, fund: FundInput, year: int) -> Series:
        """
        Gets the prices for the specified fund for each day in the specified year.

        Args:
            fund (FundInput): the fund to retrieve the data for.
            year (int): the year to retrieve the fund prices for.

        Returns:
            Series: series with data for the specified year.
        """
        self._ensure_dataframe()
        self._validate_year(year)
        fund_name = self._ensure_fund_available(fund)
        prices = self.dataframe.loc[self.dataframe["Date"].dt.year == year, fund_name]
        return prices

    def get_latest_prices(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest available prices for all funds or a specific fund.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe containing the latest prices indexed by date. When per_fund=True,
                returns a dataframe indexed by fund with as_of and price columns.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        latest_row = self._get_latest_row()
        latest = DataFrame([latest_row]).set_index("Date")
        if fund is None and funds is None:
            available = self.get_available_funds()
            if not available:
                raise ValueError("no available funds in data")
            latest = latest[available]
        if funds is not None:
            fund_list = self._resolve_funds(funds)
            self._ensure_funds_available(fund_list)
            latest = latest[fund_list]
            return latest
        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            latest = latest[[fund_name]]
        return latest

    def get_latest_prices_per_fund(
        self, funds: Iterable[FundInput] | None = None, allow_missing: bool = False
    ) -> DataFrame:
        """
        Gets the latest available price for each fund, using each fund's last valid date.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            allow_missing (bool): when True, skip funds that have no available prices instead
                of raising an error.

        Returns:
            DataFrame: dataframe indexed by fund with as_of date and price columns.
        """
        self._ensure_dataframe()
        fund_list = self._resolve_funds(funds)
        if not allow_missing:
            self._ensure_funds_available(fund_list)
        records: list[dict] = []
        missing: list[str] = []
        for fund_name in fund_list:
            if fund_name not in self.dataframe.columns:
                missing.append(fund_name)
                continue
            series = self.dataframe[["Date", fund_name]].dropna(subset=[fund_name])
            if series.empty:
                missing.append(fund_name)
                continue
            latest_row = series.sort_values("Date").iloc[-1]
            records.append(
                {
                    "fund": fund_name,
                    "as_of": latest_row["Date"].date(),
                    "price": latest_row[fund_name],
                }
            )
        if missing and not allow_missing:
            raise ValueError(
                "no price data available for fund(s): " f'{", ".join(missing)}'
            )
        if not records:
            raise ValueError("no price data available for latest per-fund prices")
        return DataFrame(records).set_index("fund")

    def get_latest_prices_per_fund_long(
        self, funds: Iterable[FundInput] | None = None, allow_missing: bool = False
    ) -> DataFrame:
        """
        Gets the latest available price per fund in long (tidy) format.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            allow_missing (bool): when True, skip funds that have no available prices instead
                of raising an error.

        Returns:
            DataFrame: dataframe with fund, as_of, and price columns.
        """
        return self.get_latest_prices_per_fund(
            funds=funds, allow_missing=allow_missing
        ).reset_index()

    def get_latest_prices_per_fund_dict(
        self,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        allow_missing: bool = False,
    ) -> dict:
        """
        Gets the latest available price per fund as a JSON-friendly dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            allow_missing (bool): when True, skip funds that have no available prices and
                include a list of missing funds in the payload.

        Returns:
            dict: dictionary with per-fund latest prices and as-of dates.
        """
        fund_list = self._resolve_funds(funds)
        latest = self.get_latest_prices_per_fund(
            funds=funds, allow_missing=allow_missing
        )
        payload: dict[str, dict] = {}
        for fund_name, row in latest.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "price": self._format_numeric_for_output(row["price"]),
            }
        response: dict[str, object] = {"funds": payload}
        if allow_missing:
            response["missing_funds"] = [
                fund for fund in fund_list if fund not in payload
            ]
        return response

    def get_current_prices_per_fund(
        self,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        allow_missing: bool = False,
    ) -> DataFrame:
        """
        Gets the current (latest available) price for each fund.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            allow_missing (bool): when True, skip funds that have no available prices instead
                of raising an error.

        Returns:
            DataFrame: dataframe indexed by fund with as_of date and price columns.
        """
        if as_of is not None:
            return self.get_prices_as_of_per_fund(
                as_of, funds=funds, allow_missing=allow_missing
            )
        return self.get_latest_prices_per_fund(funds=funds, allow_missing=allow_missing)

    def get_current_prices_per_fund_long(
        self,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        allow_missing: bool = False,
    ) -> DataFrame:
        """
        Gets the current (latest available) price per fund in long (tidy) format.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            allow_missing (bool): when True, skip funds that have no available prices instead
                of raising an error.

        Returns:
            DataFrame: dataframe with fund, as_of, and price columns.
        """
        if as_of is not None:
            return self.get_prices_as_of_per_fund_long(
                as_of, funds=funds, allow_missing=allow_missing
            )
        return self.get_latest_prices_per_fund_long(
            funds=funds, allow_missing=allow_missing
        )

    def get_current_prices_per_fund_dict(
        self,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        as_of: date | None = None,
        allow_missing: bool = False,
    ) -> dict:
        """
        Gets the current (latest available) price per fund as a JSON-friendly dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            as_of (date | None):
                optional historical anchor date. When provided, returns the most recent available
                price per fund on or before the requested date.
            allow_missing (bool): when True, skip funds that have no available prices and
                include a list of missing funds in the payload.

        Returns:
            dict: dictionary with per-fund latest prices and as-of dates.
        """
        if as_of is not None:
            return self.get_prices_as_of_per_fund_dict(
                as_of=as_of,
                funds=funds,
                date_format=date_format,
                allow_missing=allow_missing,
            )
        return self.get_latest_prices_per_fund_dict(
            funds=funds, date_format=date_format, allow_missing=allow_missing
        )

    def get_current_prices(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        *,
        per_fund: bool = False,
        allow_missing: bool = False,
        require_all_funds: bool = False,
    ) -> DataFrame:
        """
        Gets the current (latest available) prices for all funds or a specific fund.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent available prices on or before the requested date.
            per_fund (bool): when True, return per-fund latest prices (each fund uses its
                own most recent valid date). This mirrors `get_current_prices_per_fund`.
            allow_missing (bool): when True and per_fund is enabled, skip funds that have
                no available prices instead of raising an error.
            require_all_funds (bool): when True and per_fund is False, require all requested
                funds to have prices on the selected as-of date. When False, returns the most
                recent row with any requested fund data (default behavior).

        Returns:
            DataFrame: dataframe containing the latest prices indexed by date.
        """
        if per_fund and require_all_funds:
            raise ValueError("require_all_funds cannot be used when per_fund is True")
        if allow_missing and not per_fund:
            raise ValueError("allow_missing can only be used when per_fund is True")
        if per_fund:
            if fund is not None and funds is not None:
                raise ValueError("fund and funds cannot both be provided")
            if fund is not None:
                funds = [fund]
            return self.get_current_prices_per_fund(
                funds=funds, as_of=as_of, allow_missing=allow_missing
            )
        if as_of is not None:
            return self.get_prices_as_of(
                as_of, fund=fund, funds=funds, require_all_funds=require_all_funds
            )
        return self.get_latest_prices(fund=fund, funds=funds)

    def get_prices_as_of(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        require_all_funds: bool = False,
    ) -> DataFrame:
        """
        Gets the most recent available prices on or before a specific date.

        Args:
            as_of (date): date to find the most recent available prices for.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            require_all_funds (bool): when True, requires all requested fund columns to
                have values on the selected as-of date. When False, returns the most recent
                row with any requested fund data.

        Returns:
            DataFrame: dataframe containing prices indexed by date.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        target_date = to_datetime(as_of).normalize()
        price_df = self.dataframe.sort_values("Date")
        filtered = price_df[price_df["Date"] <= target_date]
        if filtered.empty:
            raise ValueError("no price data available on or before the requested date")

        if fund is not None:
            fund_name = self._ensure_fund_available(fund)
            filtered = filtered.dropna(subset=[fund_name])
            if filtered.empty:
                raise ValueError(
                    "no price data available on or before the requested date"
                )
            latest = filtered.tail(1).set_index("Date")
            return latest[[fund_name]]

        if funds is not None:
            fund_list = self._resolve_funds(funds)
            self._ensure_funds_available(fund_list)
            drop_mode = "any" if require_all_funds else "all"
            filtered = filtered.dropna(subset=fund_list, how=drop_mode)
            if filtered.empty:
                raise ValueError(
                    "no price data available on or before the requested date"
                )
            latest = filtered.tail(1).set_index("Date")
            return latest[fund_list]

        fund_columns = self.get_available_funds()
        if not fund_columns:
            raise ValueError("no available funds in data")
        drop_mode = "any" if require_all_funds else "all"
        filtered = filtered.dropna(subset=fund_columns, how=drop_mode)
        if filtered.empty:
            raise ValueError("no price data available on or before the requested date")
        latest = filtered.tail(1).set_index("Date")
        return latest[fund_columns]

    def get_price_as_of(self, fund: FundInput, as_of: date) -> Decimal:
        """
        Gets the most recent available price for a single fund on or before a specific date.

        Args:
            fund (FundInput): fund to retrieve the price for.
            as_of (date): date to find the most recent available price for.

        Returns:
            Decimal: price of the fund as of the requested date.
        """
        fund_name = self._ensure_fund_available(fund)
        latest = self.get_prices_as_of(as_of, fund=fund_name)
        return Decimal(str(latest.iloc[0][fund_name]))

    def get_prices_as_of_dict(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        require_all_funds: bool = False,
    ) -> dict:
        """
        Gets the most recent available prices on or before a date as a JSON-friendly dictionary.

        Args:
            as_of (date): date to find the most recent available prices for.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return a date object.
            require_all_funds (bool): when True, requires all requested fund columns to
                have values on the selected as-of date. When False, returns the most recent
                row with any requested fund data.

        Returns:
            dict: dictionary with requested_as_of, as_of date, and prices mapping.
        """
        latest = self.get_prices_as_of(
            as_of, fund=fund, funds=funds, require_all_funds=require_all_funds
        )
        if latest.empty:
            raise ValueError("no price data available for as-of prices")
        as_of_date = self._format_date_for_output(latest.index[0].date(), date_format)
        requested_as_of = self._format_date_for_output(as_of, date_format)
        prices = {
            fund_name: self._format_numeric_for_output(value)
            for fund_name, value in latest.iloc[0].items()
        }
        return {
            "requested_as_of": requested_as_of,
            "as_of": as_of_date,
            "prices": prices,
        }

    def get_prices_as_of_per_fund(
        self,
        as_of: date,
        funds: Iterable[FundInput] | None = None,
        allow_missing: bool = False,
    ) -> DataFrame:
        """
        Gets the most recent available price for each fund on or before a specific date.

        Args:
            as_of (date): date to find the most recent available prices for.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            allow_missing (bool): when True, skip funds that have no available prices instead
                of raising an error.

        Returns:
            DataFrame: dataframe indexed by fund with as_of date and price columns.
        """
        self._ensure_dataframe()
        fund_list = self._resolve_funds(funds)
        if not allow_missing:
            self._ensure_funds_available(fund_list)
        target_date = to_datetime(as_of).normalize()
        price_df = self.dataframe[self.dataframe["Date"] <= target_date]
        if price_df.empty:
            raise ValueError("no price data available on or before the requested date")
        records: list[dict] = []
        missing: list[str] = []
        for fund_name in fund_list:
            if fund_name not in price_df.columns:
                missing.append(fund_name)
                continue
            series = (
                price_df[["Date", fund_name]]
                .dropna(subset=[fund_name])
                .sort_values("Date")
            )
            if series.empty:
                missing.append(fund_name)
                continue
            latest_row = series.iloc[-1]
            records.append(
                {
                    "fund": fund_name,
                    "as_of": latest_row["Date"].date(),
                    "price": latest_row[fund_name],
                }
            )
        if missing and not allow_missing:
            raise ValueError(
                "no price data available on or before the requested date for: "
                f'{", ".join(missing)}'
            )
        if not records:
            raise ValueError("no price data available on or before the requested date")
        return DataFrame(records).set_index("fund")

    def get_prices_as_of_per_fund_long(
        self,
        as_of: date,
        funds: Iterable[FundInput] | None = None,
        allow_missing: bool = False,
    ) -> DataFrame:
        """
        Gets the most recent available price per fund on or before a specific date in long format.

        Args:
            as_of (date): date to find the most recent available prices for.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            allow_missing (bool): when True, skip funds that have no available prices instead
                of raising an error.

        Returns:
            DataFrame: dataframe with fund, as_of, and price columns.
        """
        return self.get_prices_as_of_per_fund(
            as_of, funds=funds, allow_missing=allow_missing
        ).reset_index()

    def get_prices_as_of_per_fund_dict(
        self,
        as_of: date,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        allow_missing: bool = False,
    ) -> dict:
        """
        Gets the most recent available price per fund on or before a date as a JSON-friendly dict.

        Args:
            as_of (date): date to find the most recent available prices for.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for the as-of dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            allow_missing (bool): when True, skip funds that have no available prices and
                include a list of missing funds in the payload.

        Returns:
            dict: dictionary with requested_as_of date and per-fund as_of/price mappings.
        """
        fund_list = self._resolve_funds(funds)
        latest = self.get_prices_as_of_per_fund(
            as_of, funds=funds, allow_missing=allow_missing
        )
        payload: dict[str, dict] = {}
        for fund_name, row in latest.iterrows():
            payload[fund_name] = {
                "as_of": self._format_date_for_output(row["as_of"], date_format),
                "price": self._format_numeric_for_output(row["price"]),
            }
        response: dict[str, object] = {
            "requested_as_of": self._format_date_for_output(as_of, date_format),
            "funds": payload,
        }
        if allow_missing:
            response["missing_funds"] = [
                fund for fund in fund_list if fund not in payload
            ]
        return response

    def get_latest_prices_long(
        self, fund: FundInput | None = None, funds: Iterable[FundInput] | None = None
    ) -> DataFrame:
        """
        Gets the latest prices in long (tidy) format for visualization or export.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.

        Returns:
            DataFrame: dataframe with Date, fund, and price columns.
        """
        latest = self.get_latest_prices(fund=fund, funds=funds).reset_index()
        return latest.melt(id_vars="Date", var_name="fund", value_name="price").dropna(
            subset=["price"]
        )

    def get_current_prices_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        *,
        per_fund: bool = False,
        allow_missing: bool = False,
        require_all_funds: bool = False,
    ) -> DataFrame:
        """
        Gets the current (latest available) prices in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent available prices on or before the requested date.
            per_fund (bool): when True, return per-fund latest prices (each fund uses its
                own most recent valid date). This mirrors `get_current_prices_per_fund_long`.
            allow_missing (bool): when True and per_fund is enabled, skip funds that have
                no available prices instead of raising an error.
            require_all_funds (bool): when True and per_fund is False, require all requested
                funds to have prices on the selected as-of date. When False, returns the most
                recent row with any requested fund data (default behavior).

        Returns:
            DataFrame: dataframe with Date, fund, and price columns. When per_fund=True,
                returns a dataframe with fund, as_of, and price columns.
        """
        if per_fund and require_all_funds:
            raise ValueError("require_all_funds cannot be used when per_fund is True")
        if allow_missing and not per_fund:
            raise ValueError("allow_missing can only be used when per_fund is True")
        if per_fund:
            if fund is not None and funds is not None:
                raise ValueError("fund and funds cannot both be provided")
            if fund is not None:
                funds = [fund]
            return self.get_current_prices_per_fund_long(
                funds=funds, as_of=as_of, allow_missing=allow_missing
            )
        if as_of is not None:
            latest = self.get_prices_as_of(
                as_of, fund=fund, funds=funds, require_all_funds=require_all_funds
            ).reset_index()
            return latest.melt(
                id_vars="Date", var_name="fund", value_name="price"
            ).dropna(subset=["price"])
        return self.get_latest_prices_long(fund=fund, funds=funds)

    def get_latest_prices_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets the latest prices as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return a date object.

        Returns:
            dict: dictionary with as_of date and prices mapping. When per_fund=True,
                returns a dictionary keyed by fund with per-fund as_of dates.
        """
        latest = self.get_latest_prices(fund=fund, funds=funds)
        if latest.empty:
            raise ValueError("no price data available for latest prices")
        as_of = self._format_date_for_output(latest.index[0].date(), date_format)
        prices = {
            fund_name: self._format_numeric_for_output(value)
            for fund_name, value in latest.iloc[0].items()
        }
        return {"as_of": as_of, "prices": prices}

    def get_current_prices_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        date_format: str | None = "iso",
        as_of: date | None = None,
        *,
        per_fund: bool = False,
        allow_missing: bool = False,
        require_all_funds: bool = False,
    ) -> dict:
        """
        Gets the current (latest available) prices as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            date_format (str | None): format for the as-of date. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return a date object.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent available prices on or before the requested date.
            per_fund (bool): when True, return per-fund latest prices (each fund uses its
                own most recent valid date). This mirrors `get_current_prices_per_fund_dict`.
            allow_missing (bool): when True and per_fund is enabled, skip funds that have
                no available prices and include a list of missing funds in the payload.
            require_all_funds (bool): when True and per_fund is False, require all requested
                funds to have prices on the selected as-of date. When False, returns the most
                recent row with any requested fund data (default behavior).

        Returns:
            dict: dictionary with as_of date and prices mapping.
        """
        if per_fund and require_all_funds:
            raise ValueError("require_all_funds cannot be used when per_fund is True")
        if allow_missing and not per_fund:
            raise ValueError("allow_missing can only be used when per_fund is True")
        if per_fund:
            if fund is not None and funds is not None:
                raise ValueError("fund and funds cannot both be provided")
            if fund is not None:
                funds = [fund]
            return self.get_current_prices_per_fund_dict(
                funds=funds,
                date_format=date_format,
                as_of=as_of,
                allow_missing=allow_missing,
            )
        if as_of is not None:
            return self.get_prices_as_of_dict(
                as_of=as_of,
                fund=fund,
                funds=funds,
                date_format=date_format,
                require_all_funds=require_all_funds,
            )
        return self.get_latest_prices_dict(
            fund=fund, funds=funds, date_format=date_format
        )

    def get_prices_as_of_long(
        self,
        as_of: date,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        require_all_funds: bool = False,
    ) -> DataFrame:
        """
        Gets the most recent available prices on or before a specific date in long format.

        Args:
            as_of (date): date to find the most recent available prices for.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                optional collection of funds to include.
            require_all_funds (bool): when True, requires all requested fund columns to
                have values on the selected as-of date. When False, returns the most recent
                row with any requested fund data.

        Returns:
            DataFrame: dataframe with Date, fund, and price columns.
        """
        latest = self.get_prices_as_of(
            as_of, fund=fund, funds=funds, require_all_funds=require_all_funds
        ).reset_index()
        return latest.melt(id_vars="Date", var_name="fund", value_name="price").dropna(
            subset=["price"]
        )

    def get_price_history(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets price history for selected funds, optionally filtered to a date range.

        Rows where all selected fund prices are missing are dropped to avoid empty
        records in downstream analytics and charts.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
                If only one bound is provided, the other defaults to the earliest/latest date
                in the dataset.

        Returns:
            DataFrame: dataframe with Date plus selected fund columns.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        fund_list = self._resolve_funds(funds if funds is not None else fund)
        available = set(self.get_available_funds())
        missing = [fund for fund in fund_list if fund not in available]
        if missing:
            raise ValueError(f'funds not available in data: {", ".join(missing)}')
        price_df = (
            self.dataframe[["Date", *fund_list]]
            .dropna(subset=fund_list, how="all")
            .sort_values("Date")
        )
        if start_date is not None or end_date is not None:
            min_date = price_df["Date"].min().date()
            max_date = price_df["Date"].max().date()
            start = start_date or min_date
            end = end_date or max_date
            price_df = self._filter_by_date_range(start, end, dataframe=price_df)
        if price_df.empty:
            raise ValueError(
                "no price data available for requested funds or date range"
            )
        return price_df.reset_index(drop=True)

    def get_recent_prices(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets the most recent trading-day prices for the selected funds.

        Args:
            days (int): number of most recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before the requested date.

        Returns:
            DataFrame: dataframe with Date plus selected fund columns.
        """
        self._ensure_dataframe()
        if fund is not None and funds is not None:
            raise ValueError("fund and funds cannot both be provided")
        self._validate_positive_int(days, "days")
        fund_list = self._resolve_funds(funds if funds is not None else fund)
        price_df = (
            self.dataframe[["Date", *fund_list]]
            .dropna(subset=fund_list, how="all")
            .sort_values("Date")
        )
        if as_of is not None:
            target = to_datetime(as_of).normalize()
            price_df = price_df[price_df["Date"] <= target]
        if price_df.empty:
            raise ValueError("no price data available for the requested period")
        recent = price_df.tail(days)
        if recent.empty:
            raise ValueError("no price data available for the requested period")
        return recent.reset_index(drop=True)

    def get_recent_prices_long(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
    ) -> DataFrame:
        """
        Gets recent trading-day prices in long (tidy) format.

        Args:
            days (int): number of most recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before the requested date.

        Returns:
            DataFrame: dataframe with Date, fund, and price columns.
        """
        recent = self.get_recent_prices(days=days, fund=fund, funds=funds, as_of=as_of)
        return recent.melt(id_vars="Date", var_name="fund", value_name="price").dropna(
            subset=["price"]
        )

    def get_recent_prices_dict(
        self,
        days: int = 5,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        as_of: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets recent trading-day prices as a JSON-friendly dictionary.

        Args:
            days (int): number of most recent trading days to include.
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            as_of (date | None): optional historical anchor date. When provided, returns
                the most recent trading days on or before the requested date.
            date_format (str | None): format for the Date column. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with recent price records and date boundaries.
        """
        recent = self.get_recent_prices(days=days, fund=fund, funds=funds, as_of=as_of)
        start_date = recent["Date"].min().date()
        end_date = recent["Date"].max().date()
        payload = {
            "start_date": self._format_date_for_output(start_date, date_format),
            "end_date": self._format_date_for_output(end_date, date_format),
            "days": int(len(recent)),
            "prices": self._format_long_dataframe_for_output(
                recent.melt(id_vars="Date", var_name="fund", value_name="price").dropna(
                    subset=["price"]
                ),
                date_format,
            ),
        }
        if as_of is not None:
            payload["requested_as_of"] = self._format_date_for_output(
                as_of, date_format
            )
        return payload

    def get_moving_average(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        window: int = 20,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets rolling moving averages for the selected funds.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            window (int): rolling window size (number of trading days).
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with Date plus moving-average columns per fund.
        """
        self._ensure_dataframe()
        self._validate_positive_int(window, "window")
        price_df = self.get_price_history(
            fund=fund, funds=funds, start_date=start_date, end_date=end_date
        )
        price_indexed = price_df.set_index("Date")
        moving = price_indexed.rolling(window=window, min_periods=1).mean()
        return moving.reset_index()

    def get_moving_average_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        window: int = 20,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets rolling moving averages in long (tidy) format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            window (int): rolling window size (number of trading days).
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with Date, fund, and moving_average columns.
        """
        moving = self.get_moving_average(
            fund=fund,
            funds=funds,
            window=window,
            start_date=start_date,
            end_date=end_date,
        )
        return moving.melt(
            id_vars="Date", var_name="fund", value_name="moving_average"
        ).dropna(subset=["moving_average"])

    def get_price_history_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataFrame:
        """
        Gets price history in long (tidy) format for visualization or export.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.

        Returns:
            DataFrame: dataframe with Date, fund, and price columns.
        """
        self._ensure_dataframe()
        price_df = self.get_price_history(
            fund=fund, funds=funds, start_date=start_date, end_date=end_date
        )
        return price_df.melt(
            id_vars="Date", var_name="fund", value_name="price"
        ).dropna(subset=["price"])

    def get_price_history_long_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets price history in long format as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            date_format (str | None): format for the Date column. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with a list of price history records.
        """
        price_long = self.get_price_history_long(
            fund=fund, funds=funds, start_date=start_date, end_date=end_date
        )
        return {
            "prices": self._format_long_dataframe_for_output(price_long, date_format)
        }

    def get_price_history_with_metrics_long(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        base_value: float = 100.0,
    ) -> DataFrame:
        """
        Gets price history with daily returns, cumulative returns, and normalized prices in long format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            base_value (float): base value used to normalize prices.

        Returns:
            DataFrame: dataframe with Date, fund, price, return, cumulative_return, normalized_price columns.
        """
        self._ensure_dataframe()
        self._validate_positive_float(base_value, "base_value")
        price_df = self.get_price_history(
            fund=fund, funds=funds, start_date=start_date, end_date=end_date
        )
        price_df = price_df.sort_values("Date")
        price_indexed = price_df.set_index("Date")
        returns = price_indexed.pct_change(fill_method=None)
        cumulative_returns = (1 + returns).cumprod() - 1
        first_valid = self._resolve_first_valid_prices(price_indexed)
        normalized_prices = price_indexed.div(first_valid).mul(base_value)

        price_long = price_df.melt(
            id_vars="Date", var_name="fund", value_name="price"
        ).dropna(subset=["price"])
        returns_long = returns.reset_index().melt(
            id_vars="Date", var_name="fund", value_name="return"
        )
        cumulative_long = cumulative_returns.reset_index().melt(
            id_vars="Date", var_name="fund", value_name="cumulative_return"
        )
        normalized_long = normalized_prices.reset_index().melt(
            id_vars="Date", var_name="fund", value_name="normalized_price"
        )

        combined = price_long.merge(returns_long, on=["Date", "fund"], how="left")
        combined = combined.merge(cumulative_long, on=["Date", "fund"], how="left")
        combined = combined.merge(normalized_long, on=["Date", "fund"], how="left")
        return combined.sort_values(["Date", "fund"]).reset_index(drop=True)

    def get_price_history_with_metrics_dict(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        base_value: float = 100.0,
        date_format: str | None = "iso",
    ) -> dict:
        """
        Gets price history with metrics as a JSON-friendly dictionary.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            base_value (float): base value used to normalize prices.
            date_format (str | None): format for the Date column. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with a list of metrics records.
        """
        metrics_long = self.get_price_history_with_metrics_long(
            fund=fund,
            funds=funds,
            start_date=start_date,
            end_date=end_date,
            base_value=base_value,
        )
        return {
            "metrics": self._format_long_dataframe_for_output(metrics_long, date_format)
        }

    def get_price_summary(self, funds: Iterable[FundInput] | None = None) -> DataFrame:
        """
        Gets summary statistics for the available price history.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.

        Returns:
            DataFrame: dataframe indexed by fund with price summary statistics.
        """
        self._ensure_dataframe()
        fund_list = self._resolve_funds(funds)
        self._ensure_funds_available(fund_list)
        price_df = self.dataframe[["Date", *fund_list]].dropna(how="all")
        records: list[dict] = []
        for fund_name in fund_list:
            series = (
                price_df[["Date", fund_name]]
                .dropna(subset=[fund_name])
                .sort_values("Date")
            )
            if series.empty:
                raise ValueError(f"no price data available for fund: {fund_name}")
            values = series[fund_name]
            start_price = values.iloc[0]
            end_price = values.iloc[-1]
            total_return = (
                float("nan") if start_price == 0 else (end_price / start_price) - 1
            )
            records.append(
                {
                    "fund": fund_name,
                    "first_date": series["Date"].iloc[0].date(),
                    "last_date": series["Date"].iloc[-1].date(),
                    "start_price": start_price,
                    "end_price": end_price,
                    "min_price": values.min(),
                    "max_price": values.max(),
                    "mean_price": values.mean(),
                    "median_price": values.median(),
                    "std_price": values.std(),
                    "total_return": total_return,
                }
            )
        return DataFrame(records).set_index("fund")

    def get_price_summary_dict(
        self, funds: Iterable[FundInput] | None = None, date_format: str | None = "iso"
    ) -> dict:
        """
        Gets price summary statistics as a JSON-friendly dictionary.

        Args:
            funds (Iterable[FundInput] | None):
                optional collection of funds to include. When None, uses all available funds.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: dictionary with per-fund price summary statistics.
        """
        summary = self.get_price_summary(funds=funds)
        if summary.empty:
            raise ValueError("no price data available for price summary")
        payload: dict[str, dict] = {}
        for fund_name, row in summary.iterrows():
            payload[fund_name] = {
                "first_date": self._format_date_for_output(
                    row["first_date"], date_format
                ),
                "last_date": self._format_date_for_output(
                    row["last_date"], date_format
                ),
                "start_price": self._format_numeric_for_output(row["start_price"]),
                "end_price": self._format_numeric_for_output(row["end_price"]),
                "min_price": self._format_numeric_for_output(row["min_price"]),
                "max_price": self._format_numeric_for_output(row["max_price"]),
                "mean_price": self._format_numeric_for_output(row["mean_price"]),
                "median_price": self._format_numeric_for_output(row["median_price"]),
                "std_price": self._format_numeric_for_output(row["std_price"]),
                "total_return": self._format_numeric_for_output(row["total_return"]),
            }
        return {"funds": payload}

    def get_price_history_with_metrics(
        self,
        fund: FundInput | None = None,
        funds: Iterable[FundInput] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        base_value: float = 100.0,
    ) -> DataFrame:
        """
        Gets price history with returns, cumulative returns, and normalized prices in wide format.

        Args:
            fund (FundInput | None): optional fund to limit the output.
            funds (Iterable[FundInput] | None):
                funds to include. When None, returns all available funds.
            start_date (date | None): optional start date for filtering.
            end_date (date | None): optional end date for filtering.
            base_value (float): base value used to normalize prices.

        Returns:
            DataFrame: dataframe indexed by Date with MultiIndex columns of (fund, metric).
            Metrics include price, return, cumulative_return, and normalized_price.
        """
        self._ensure_dataframe()
        self._validate_positive_float(base_value, "base_value")
        price_df = self.get_price_history(
            fund=fund, funds=funds, start_date=start_date, end_date=end_date
        )
        price_df = price_df.sort_values("Date")
        price_indexed = price_df.set_index("Date")
        returns = price_indexed.pct_change(fill_method=None)
        cumulative_returns = (1 + returns).cumprod() - 1
        first_valid = self._resolve_first_valid_prices(price_indexed)
        normalized_prices = price_indexed.div(first_valid).mul(base_value)

        combined = concat(
            {
                "price": price_indexed,
                "return": returns,
                "cumulative_return": cumulative_returns,
                "normalized_price": normalized_prices,
            },
            axis=1,
        )
        combined = combined.swaplevel(0, 1, axis=1)
        metric_order = ("price", "return", "cumulative_return", "normalized_price")
        ordered_columns = [
            (fund, metric) for fund in price_indexed.columns for metric in metric_order
        ]
        combined = combined[ordered_columns]
        combined.index.name = "Date"
        return combined

    def get_available_funds(self) -> list[str]:
        """
        Gets the list of available fund columns in the dataset.

        Returns:
            list[str]: available fund names present in the CSV data.
        """
        self._ensure_dataframe()
        return self._resolve_available_funds_from_dataframe(self.dataframe)

    def get_data_summary(self) -> dict:
        """
        Gets a summary of the available price data.

        Returns:
            dict: summary of start date, end date, total rows, and fund coverage.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.dropna(how="all")
        funds = self.get_available_funds()
        missing_funds = [fund for fund in self.ALL_FUNDS if fund not in funds]
        date_series = price_df["Date"].dropna().dt.normalize()
        start_date = date_series.min().date() if not date_series.empty else None
        end_date = date_series.max().date() if not date_series.empty else None
        expected_business_days = 0
        missing_business_days = 0
        business_day_coverage = None
        if start_date is not None and end_date is not None:
            expected_business_days = len(bdate_range(start=start_date, end=end_date))
            available_business_days = int(date_series.drop_duplicates().shape[0])
            missing_business_days = max(
                0, expected_business_days - available_business_days
            )
            business_day_coverage = (
                available_business_days / expected_business_days
                if expected_business_days
                else None
            )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_rows": int(len(price_df)),
            "available_funds": funds,
            "missing_funds": missing_funds,
            "expected_business_days": expected_business_days,
            "missing_business_days": missing_business_days,
            "business_day_coverage": business_day_coverage,
        }

    def get_data_summary_dict(self, date_format: str | None = "iso") -> dict:
        """
        Gets a JSON-friendly summary of the available price data.

        Args:
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            dict: summary of start date, end date, total rows, and fund coverage.
        """
        summary = self.get_data_summary()
        return {
            "start_date": self._format_date_for_output(
                summary["start_date"], date_format
            ),
            "end_date": self._format_date_for_output(summary["end_date"], date_format),
            "total_rows": summary["total_rows"],
            "available_funds": summary["available_funds"],
            "missing_funds": summary["missing_funds"],
            "expected_business_days": summary["expected_business_days"],
            "missing_business_days": summary["missing_business_days"],
            "business_day_coverage": self._format_numeric_for_output(
                summary["business_day_coverage"]
            ),
        }

    def get_data_quality_report(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        include_cache_status: bool = True,
    ) -> dict:
        """
        Builds a consolidated data-quality report for the cached dataset.

        Args:
            start_date (date | None): optional start date for missing business day checks.
            end_date (date | None): optional end date for missing business day checks.
            include_cache_status (bool): include cache metadata in the report.

        Returns:
            dict: summary, fund coverage, missing business days, and optional cache status.
        """
        self._ensure_dataframe()
        report = {
            "summary": self.get_data_summary(),
            "fund_coverage": self.get_fund_coverage_summary(),
            "missing_business_days": self.get_missing_business_days(
                start_date=start_date, end_date=end_date
            ),
        }
        if include_cache_status:
            report["cache_status"] = self.get_cache_status()
        return report

    def get_data_quality_report_dict(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        include_cache_status: bool = True,
        date_format: str | None = "iso",
        datetime_format: str | None = "iso",
    ) -> dict:
        """
        Builds a JSON-friendly data-quality report for the cached dataset.

        Args:
            start_date (date | None): optional start date for missing business day checks.
            end_date (date | None): optional end date for missing business day checks.
            include_cache_status (bool): include cache metadata in the report.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            datetime_format (str | None): format for datetimes. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return datetime objects.

        Returns:
            dict: summary, fund coverage, missing business days, and optional cache status.
        """
        report = {
            "summary": self.get_data_summary_dict(date_format=date_format),
            "fund_coverage": self.get_fund_coverage_summary_dict(
                date_format=date_format
            ),
            "missing_business_days": self.get_missing_business_days_dict(
                start_date=start_date, end_date=end_date, date_format=date_format
            ),
        }
        if include_cache_status:
            report["cache_status"] = self.get_cache_status_dict(
                date_format=date_format, datetime_format=datetime_format
            )
        return report

    def get_fund_coverage_summary(self) -> DataFrame:
        """
        Gets coverage statistics for each fund in the dataset.

        Returns:
            DataFrame: dataframe with available rows, missing rows, coverage percent,
            and the first/last date with data for each fund.
        """
        self._ensure_dataframe()
        price_df = self.dataframe.set_index("Date").dropna(how="all")
        total_rows = int(len(price_df))
        if total_rows == 0:
            raise ValueError("no price data available for fund coverage summary")

        summary: list[dict] = []
        for fund_name in self.ALL_FUNDS:
            if fund_name in price_df.columns:
                series = price_df[fund_name].dropna()
                available_rows = int(len(series))
                first_date = series.index[0].date() if available_rows else None
                last_date = series.index[-1].date() if available_rows else None
            else:
                available_rows = 0
                first_date = None
                last_date = None
            missing_rows = total_rows - available_rows
            coverage_percent = (available_rows / total_rows) if total_rows else 0.0
            summary.append(
                {
                    "fund": fund_name,
                    "available_rows": available_rows,
                    "missing_rows": missing_rows,
                    "coverage_percent": coverage_percent,
                    "first_available_date": first_date,
                    "last_available_date": last_date,
                }
            )
        return DataFrame(summary).set_index("fund")

    def get_fund_coverage_summary_dict(
        self, date_format: str | None = "iso"
    ) -> list[dict]:
        """
        Gets a JSON-friendly fund coverage summary for the dataset.

        Args:
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            list[dict]: list of coverage summaries per fund.
        """
        coverage = self.get_fund_coverage_summary()
        payload: list[dict] = []
        for fund_name, row in coverage.iterrows():
            payload.append(
                {
                    "fund": fund_name,
                    "available_rows": int(row["available_rows"]),
                    "missing_rows": int(row["missing_rows"]),
                    "coverage_percent": float(row["coverage_percent"]),
                    "first_available_date": (
                        self._format_date_for_output(
                            row["first_available_date"], date_format
                        )
                        if row["first_available_date"] is not None
                        else None
                    ),
                    "last_available_date": (
                        self._format_date_for_output(
                            row["last_available_date"], date_format
                        )
                        if row["last_available_date"] is not None
                        else None
                    ),
                }
            )
        return payload

    def get_missing_business_days(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> DataFrame:
        """
        Gets a list of missing business days in the dataset.

        Args:
            start_date (date | None): optional start date to check.
            end_date (date | None): optional end date to check.

        Returns:
            DataFrame: dataframe with a Date column for missing business days.
        """
        self._ensure_dataframe()
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")

        date_series = self.dataframe["Date"].dropna().dt.normalize()
        if start_date is None and end_date is None:
            if date_series.empty:
                return DataFrame(columns=["Date"])
            start_date = date_series.min().date()
            end_date = date_series.max().date()

        self._validate_date_range(start_date, end_date)
        expected = bdate_range(start=start_date, end=end_date)
        if expected.empty:
            return DataFrame(columns=["Date"])
        available = to_datetime(date_series, errors="coerce").dropna()
        missing = expected.difference(available)
        return DataFrame({"Date": missing})

    def get_missing_business_days_dict(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        date_format: str | None = "iso",
    ) -> list[dict]:
        """
        Gets missing business days in a JSON-friendly list.

        Args:
            start_date (date | None): optional start date to check.
            end_date (date | None): optional end date to check.
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.

        Returns:
            list[dict]: list of missing business days with formatted dates.
        """
        missing = self.get_missing_business_days(
            start_date=start_date, end_date=end_date
        )
        payload: list[dict] = []
        if missing.empty:
            return payload
        for value in missing["Date"].dt.date:
            payload.append({"date": self._format_date_for_output(value, date_format)})
        return payload

    def get_cache_status(self) -> dict:
        """
        Gets metadata about the cached CSV file without performing a network update.

        Returns:
            dict: cache file path, existence, file update time, staleness indicators,
            and dataset coverage.
        """
        csv_path = Path(self.csv_filepath)
        exists = csv_path.is_file()
        last_updated = None
        file_size_bytes = None
        if exists:
            last_updated = datetime.fromtimestamp(csv_path.stat().st_mtime)
            file_size_bytes = csv_path.stat().st_size

        dataframe_valid: bool | None = self.dataframe is not None
        validation_error = None
        if self.dataframe is None and exists:
            try:
                self._set_values()
                dataframe_valid = True
            except Exception as exc:
                dataframe_valid = False
                validation_error = str(exc)

        latest_data_date = self.latest if self.latest is not None else None
        total_rows = int(len(self.dataframe)) if self.dataframe is not None else 0
        available_funds: list[str] = []
        missing_funds: list[str] = []
        data_start_date = None
        data_end_date = None
        if self.dataframe is not None:
            available_funds = self._resolve_available_funds_from_dataframe(
                self.dataframe
            )
            missing_funds = [
                fund for fund in self.ALL_FUNDS if fund not in available_funds
            ]
            date_series = self.dataframe["Date"].dropna()
            if not date_series.empty:
                data_start_date = date_series.min().date()
                data_end_date = date_series.max().date()

        from tsp import analytics as analytics_module

        today = analytics_module.date.today()
        metadata = self._load_cache_metadata()
        etag = metadata.get("etag")
        last_modified_header = metadata.get("last_modified")
        last_checked = None
        last_metadata_updated = None
        for key, target in (
            ("last_checked", "checked"),
            ("last_updated", "updated"),
        ):
            raw_value = metadata.get(key)
            if not raw_value:
                continue
            try:
                parsed = datetime.fromisoformat(raw_value)
            except ValueError:
                parsed = None
            if target == "checked":
                last_checked = parsed
            else:
                last_metadata_updated = parsed
        cache_age_days = (today - last_updated.date()).days if last_updated else None
        data_age_days = (today - latest_data_date).days if latest_data_date else None
        last_checked_age_days = (
            (today - last_checked.date()).days if last_checked is not None else None
        )
        is_stale = None
        last_business_day = self._get_last_business_day(today)
        stale_by_days = None
        if latest_data_date is not None:
            is_stale = latest_data_date < last_business_day
            stale_by_days = max(0, (last_business_day - latest_data_date).days)

        return {
            "csv_filepath": str(csv_path),
            "exists": exists,
            "last_updated": last_updated,
            "file_size_bytes": file_size_bytes,
            "dataframe_valid": dataframe_valid,
            "validation_error": validation_error,
            "latest_data_date": latest_data_date,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "data_span_days": (
                (data_end_date - data_start_date).days
                if data_start_date is not None and data_end_date is not None
                else None
            ),
            "etag": etag,
            "last_modified": last_modified_header,
            "last_checked": last_checked,
            "last_checked_age_days": last_checked_age_days,
            "metadata_last_updated": last_metadata_updated,
            "cache_age_days": cache_age_days,
            "data_age_days": data_age_days,
            "is_stale": is_stale,
            "stale_by_days": stale_by_days,
            "last_business_day": last_business_day,
            "total_rows": total_rows,
            "available_funds": available_funds,
            "missing_funds": missing_funds,
        }

    def get_cache_status_dict(
        self, date_format: str | None = "iso", datetime_format: str | None = "iso"
    ) -> dict:
        """
        Gets a JSON-friendly cache status payload.

        Args:
            date_format (str | None): format for dates. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return date objects.
            datetime_format (str | None): format for datetimes. Use 'iso' for ISO 8601,
                a strftime-compatible format string, or None to return datetime objects.

        Returns:
            dict: cache status metadata with formatted dates.
        """
        status = self.get_cache_status()
        last_updated = status["last_updated"]
        return {
            "csv_filepath": status["csv_filepath"],
            "exists": status["exists"],
            "last_updated": (
                self._format_datetime_for_output(last_updated, datetime_format)
                if last_updated is not None
                else None
            ),
            "file_size_bytes": status["file_size_bytes"],
            "dataframe_valid": status["dataframe_valid"],
            "validation_error": status["validation_error"],
            "latest_data_date": (
                self._format_date_for_output(status["latest_data_date"], date_format)
                if status["latest_data_date"] is not None
                else None
            ),
            "data_start_date": (
                self._format_date_for_output(status["data_start_date"], date_format)
                if status["data_start_date"] is not None
                else None
            ),
            "data_end_date": (
                self._format_date_for_output(status["data_end_date"], date_format)
                if status["data_end_date"] is not None
                else None
            ),
            "data_span_days": status["data_span_days"],
            "etag": status["etag"],
            "last_modified": status["last_modified"],
            "last_checked": (
                self._format_datetime_for_output(
                    status["last_checked"], datetime_format
                )
                if status["last_checked"] is not None
                else None
            ),
            "last_checked_age_days": status["last_checked_age_days"],
            "metadata_last_updated": (
                self._format_datetime_for_output(
                    status["metadata_last_updated"],
                    datetime_format,
                )
                if status["metadata_last_updated"] is not None
                else None
            ),
            "cache_age_days": status["cache_age_days"],
            "data_age_days": status["data_age_days"],
            "is_stale": status["is_stale"],
            "stale_by_days": status["stale_by_days"],
            "last_business_day": self._format_date_for_output(
                status["last_business_day"], date_format
            ),
            "total_rows": status["total_rows"],
            "available_funds": status["available_funds"],
            "missing_funds": status["missing_funds"],
        }
