"""Tiingo end-of-day equity market data provider implementation."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from stock_forecasting.providers.base import Bar, DataProvider

logger = logging.getLogger(__name__)


class TiingoProvider(DataProvider):
    """Tiingo daily prices API data provider for equities (fallback source)."""

    def __init__(
        self,
        base_url: str = "https://api.tiingo.com/tiingo",
        api_key: str | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize TiingoProvider.

        Args:
            base_url: Base API URL for the Tiingo API.
            api_key: Optional Tiingo API token (caller passes get_settings().tiingo_api_key).
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
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
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

        url = f"{self.base_url}/daily/{quote(symbol, safe='')}/prices"
        params: dict[str, Any] = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
        }

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

        return self._parse_prices(data, start, end)

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

    def _parse_prices(self, data: Any, start: date, end: date) -> list[Bar]:
        """Parse a raw Tiingo daily-prices JSON list into clean Bar objects."""
        if not isinstance(data, list):
            return []

        bars: list[Bar] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            raw_date = item.get("date")
            if not isinstance(raw_date, str) or len(raw_date) < 10:
                continue
            day_str = raw_date[:10]
            try:
                d = date.fromisoformat(day_str)
            except ValueError:
                continue
            if not (start <= d <= end):
                continue

            bars.append(
                Bar(
                    ts=f"{day_str}T00:00:00Z",
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    adj_close=(
                        float(item["adjClose"])
                        if item.get("adjClose") is not None
                        else None
                    ),
                    volume=float(item.get("volume", 0.0) or 0.0),
                )
            )

        bars.sort(key=lambda b: b.ts)
        return bars
