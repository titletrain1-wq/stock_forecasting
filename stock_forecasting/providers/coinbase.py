"""Coinbase public market data provider implementation."""

import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx

from stock_forecasting.providers.base import Bar, DataProvider

logger = logging.getLogger(__name__)

DEFAULT_COINBASE_MAP: dict[str, str] = {
    "BTC": "BTC-USD",
    "BTC-USD": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "bitcoin": "BTC-USD",
    "ETH": "ETH-USD",
    "ETH-USD": "ETH-USD",
    "ETHUSD": "ETH-USD",
    "ethereum": "ETH-USD",
    "SOL": "SOL-USD",
    "SOL-USD": "SOL-USD",
    "SOLUSD": "SOL-USD",
    "solana": "SOL-USD",
    "DOGE": "DOGE-USD",
    "DOGE-USD": "DOGE-USD",
    "dogecoin": "DOGE-USD",
    "ADA": "ADA-USD",
    "ADA-USD": "ADA-USD",
    "cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "XRP-USD": "XRP-USD",
    "ripple": "XRP-USD",
    "AVAX": "AVAX-USD",
    "AVAX-USD": "AVAX-USD",
    "DOT": "DOT-USD",
    "DOT-USD": "DOT-USD",
    "LINK": "LINK-USD",
    "LINK-USD": "LINK-USD",
    "BNB": "BNB-USD",
    "BNB-USD": "BNB-USD",
}


class CoinbaseProvider(DataProvider):
    """Coinbase Exchange public candles data provider."""

    def __init__(
        self,
        base_url: str = "https://api.exchange.coinbase.com",
        symbol_map: dict[str, str] | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize CoinbaseProvider.

        Args:
            base_url: Base API URL for Coinbase Exchange public API.
            symbol_map: Optional mapping overriding default symbol to Coinbase product ID.
            timeout: HTTP request timeout in seconds.
            client: Optional pre-configured httpx.Client instance.
        """
        self.base_url = base_url.rstrip("/")
        self.symbol_map = {**DEFAULT_COINBASE_MAP, **(symbol_map or {})}
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.Client:
        """Return an active HTTP client."""
        if self._client is not None:
            return self._client
        headers = {
            "Accept": "application/json",
            "User-Agent": "stock-forecasting/0.1.0",
        }
        return httpx.Client(timeout=self.timeout, headers=headers)

    def resolve_product_id(self, symbol: str) -> str:
        """Resolve a ticker symbol to a Coinbase product ID.

        Args:
            symbol: Ticker or coin identifier (e.g. 'BTC-USD', 'bitcoin', 'ETH').

        Returns:
            Coinbase product ID string (e.g. 'BTC-USD').
        """
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]

        norm = symbol.strip().upper()
        if norm in self.symbol_map:
            return self.symbol_map[norm]

        norm_lower = symbol.strip().lower()
        if norm_lower in self.symbol_map:
            return self.symbol_map[norm_lower]

        if "-" in norm:
            return norm

        return f"{norm}-USD"

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Fetch daily historical OHLCV bars from Coinbase public candles endpoint.

        Args:
            symbol: Product symbol or coin id (e.g. 'BTC-USD', 'bitcoin').
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of Bar dataclass instances sorted chronologically.
        """
        if start > end:
            return []

        product_id = self.resolve_product_id(symbol)

        close_client = False
        client = self._client
        if client is None:
            client = self._get_client()
            close_client = True

        try:
            all_bars_map: dict[str, Bar] = {}
            current_start = start

            while current_start <= end:
                # Chunk size up to 250 days to be comfortably within Coinbase 300 candle limit
                current_end = min(end, current_start + timedelta(days=250))

                start_iso = (
                    datetime.combine(current_start, time.min, tzinfo=UTC).isoformat()
                )
                end_iso = (
                    datetime.combine(
                        current_end, time(23, 59, 59), tzinfo=UTC
                    ).isoformat()
                )

                url = f"{self.base_url}/products/{product_id}/candles"
                params = {
                    "granularity": 86400,
                    "start": start_iso,
                    "end": end_iso,
                }

                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                chunk_bars = self._parse_candles(data, start, end)
                for bar in chunk_bars:
                    all_bars_map[bar.ts] = bar

                current_start = current_end + timedelta(days=1)

            sorted_bars = sorted(all_bars_map.values(), key=lambda b: b.ts)
            return sorted_bars
        finally:
            if close_client and self._client is None:
                client.close()

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        """Fetch the most recent N closed daily bars from Coinbase.

        Args:
            symbol: Product symbol or coin id (e.g. 'BTC-USD').
            lookback: Number of recent daily bars to return.

        Returns:
            List of Bar dataclass instances of length up to `lookback`.
        """
        if lookback <= 0:
            return []
        today = datetime.now(UTC).date()
        start_date = today - timedelta(days=lookback + 5)
        bars = self.get_daily_history(symbol, start=start_date, end=today)
        return bars[-lookback:]

    def _parse_candles(self, data: Any, start: date, end: date) -> list[Bar]:
        """Parse raw Coinbase candles JSON into clean Bar objects."""
        if not isinstance(data, list):
            return []

        bars: list[Bar] = []
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            # Coinbase candle schema: [time, low, high, open, close, volume]
            epoch_sec = item[0]
            dt = datetime.fromtimestamp(epoch_sec, tz=UTC)
            d = dt.date()
            if not (start <= d <= end):
                continue

            low_val = float(item[1])
            high_val = float(item[2])
            open_val = float(item[3])
            close_val = float(item[4])
            vol_val = float(item[5])

            bars.append(
                Bar(
                    ts=f"{d.isoformat()}T00:00:00Z",
                    open=open_val,
                    high=high_val,
                    low=low_val,
                    close=close_val,
                    adj_close=None,
                    volume=vol_val,
                )
            )

        return bars
