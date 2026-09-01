from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class DataProvider(Protocol):
    """Data source protocol."""

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Fetch daily bars."""
        ...

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        """Fetch last N bars."""
        ...
