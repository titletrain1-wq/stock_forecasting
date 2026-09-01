from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """Exception raised for errors in data provider requests."""


@dataclass
class Bar:
    """OHLCV bar."""

    ts: str  # ISO-8601 UTC
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None
    volume: float


@dataclass
class Derivative:
    """One day of crypto derivative metrics for a perpetual market."""

    ts: str  # ISO-8601 UTC, day-aligned ("YYYY-MM-DDT00:00:00Z")
    funding_rate: float | None  # daily funding cost (sum of the day's hourly rates)
    open_interest: float | None  # base-unit OI; snapshot-only, so newest day only


@runtime_checkable
class DataProvider(Protocol):
    """Data source protocol."""

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Fetch daily bars."""
        ...

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        """Fetch last N bars."""
        ...


@runtime_checkable
class DerivativesProvider(Protocol):
    """Crypto derivatives (funding rate + open interest) source protocol."""

    def get_derivatives(self, symbol: str, start: date, end: date) -> list[Derivative]:
        """Fetch day-aggregated derivative metrics for [start, end]."""
        ...
