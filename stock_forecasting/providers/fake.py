from datetime import UTC, date, datetime, timedelta

from stock_forecasting.providers.base import Bar, DataProvider


class FakeProvider(DataProvider):
    """Fake provider for testing."""

    def __init__(
        self, return_malformed: bool = False, return_429: bool = False
    ) -> None:
        self.return_malformed = return_malformed
        self.return_429 = return_429

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        if self.return_429:
            raise RuntimeError("HTTP 429: Rate limited")

        bars = []
        current = start
        price = 100.0
        while current <= end:
            if self.return_malformed:
                # Return invalid structure (will be caught by validation)
                return [{"ts": "bad", "close": "not_a_number"}]  # type: ignore

            bars.append(
                Bar(
                    ts=current.isoformat() + "T00:00:00Z",
                    open=price,
                    high=price * 1.02,
                    low=price * 0.98,
                    close=price * 1.01,
                    adj_close=price * 1.01,
                    volume=1000000.0,
                )
            )
            price *= 1.001  # Gradual increase
            current += timedelta(days=1)

        return bars

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        today = datetime.now(UTC).date()
        return self.get_daily_history(symbol, today - timedelta(days=lookback), today)
