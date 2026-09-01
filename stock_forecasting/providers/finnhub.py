"""Finnhub stock candle market data provider implementation."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx

from stock_forecasting.providers.base import Bar, DataProvider

logger = logging.getLogger(__name__)


class FinnhubProvider(DataProvider):
    """Finnhub stock candles API data provider for equities (fallback source)."""

    def __init__(
        self,
        base_url: str = "https://finnhub.io/api/v1",
        api_key: str | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize FinnhubProvider.

        Args:
            base_url: Base API URL for the Finnhub API.
            api_key: Optional Finnhub API token (caller passes get_settings().finnhub_api_key).
            timeout: HTTP request timeout in seconds.
            client: Optional pre-configured httpx.Client instance.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
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

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Fetch daily historical OHLCV bars for an equity between start and end dates.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL').
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of Bar dataclass instances sorted chronologically.
        """
        if start > end:
            return []

        from_epoch = int(datetime.combine(start, time.min, tzinfo=UTC).timestamp())
        to_epoch = int(datetime.combine(end, time(23, 59, 59), tzinfo=UTC).timestamp())

        url = f"{self.base_url}/stock/candle"
        params: dict[str, Any] = {
            "symbol": symbol,
            "resolution": "D",
            "from": from_epoch,
            "to": to_epoch,
        }
        if self.api_key:
            params["token"] = self.api_key

        close_client = False
        client = self._client
        if client is None:
            client = self._get_client()
            close_client = True

        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        finally:
            if close_client and self._client is None:
                client.close()

        return self._parse_candles(data, start, end)

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        """Fetch the most recent N closed daily bars for an equity symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL').
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
        """Parse a raw Finnhub stock-candle JSON object into clean Bar objects.

        Finnhub returns HTTP 200 with ``{"s": "no_data"}`` for an invalid symbol or an
        empty window, so anything other than ``s == "ok"`` yields an empty list.
        """
        if not isinstance(data, dict) or data.get("s") != "ok":
            return []

        times = data.get("t") or []
        opens = data.get("o") or []
        highs = data.get("h") or []
        lows = data.get("l") or []
        closes = data.get("c") or []
        volumes = data.get("v") or []

        count = min(len(times), len(opens), len(highs), len(lows), len(closes))

        bars: list[Bar] = []
        for i in range(count):
            d = datetime.fromtimestamp(times[i], tz=UTC).date()
            if not (start <= d <= end):
                continue
            bars.append(
                Bar(
                    ts=f"{d.isoformat()}T00:00:00Z",
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    adj_close=None,
                    volume=float(volumes[i]) if i < len(volumes) else 0.0,
                )
            )

        bars.sort(key=lambda b: b.ts)
        return bars
