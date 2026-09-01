"""CoinGecko market data provider implementation for crypto assets."""

import logging
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx

from stock_forecasting.providers.base import Bar, DataProvider

logger = logging.getLogger(__name__)

DEFAULT_COINGECKO_MAP: dict[str, str] = {
    "BTC": "bitcoin",
    "BTC-USD": "bitcoin",
    "BTCUSD": "bitcoin",
    "bitcoin": "bitcoin",
    "ETH": "ethereum",
    "ETH-USD": "ethereum",
    "ETHUSD": "ethereum",
    "ethereum": "ethereum",
    "SOL": "solana",
    "SOL-USD": "solana",
    "SOLUSD": "solana",
    "solana": "solana",
    "DOGE": "dogecoin",
    "DOGE-USD": "dogecoin",
    "dogecoin": "dogecoin",
    "ADA": "cardano",
    "ADA-USD": "cardano",
    "cardano": "cardano",
    "XRP": "ripple",
    "XRP-USD": "ripple",
    "ripple": "ripple",
    "AVAX": "avalanche-2",
    "AVAX-USD": "avalanche-2",
    "DOT": "polkadot",
    "DOT-USD": "polkadot",
    "LINK": "chainlink",
    "LINK-USD": "chainlink",
    "BNB": "binancecoin",
    "BNB-USD": "binancecoin",
}


class CoinGeckoProvider(DataProvider):
    """CoinGecko public API market data provider."""

    def __init__(
        self,
        base_url: str = "https://api.coingecko.com/api/v3",
        api_key: str | None = None,
        symbol_map: dict[str, str] | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize CoinGeckoProvider.

        Args:
            base_url: Base API URL for CoinGecko.
            api_key: Optional CoinGecko Demo/Pro API key.
            symbol_map: Optional mapping overriding default symbol to CoinGecko coin id.
            timeout: HTTP request timeout in seconds.
            client: Optional pre-configured httpx.Client instance.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.symbol_map = {**DEFAULT_COINGECKO_MAP, **(symbol_map or {})}
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
            headers["x-cg-demo-api-key"] = self.api_key
        return httpx.Client(timeout=self.timeout, headers=headers)

    def resolve_coin_id(self, symbol: str) -> str:
        """Resolve a ticker symbol to a CoinGecko coin ID.

        Args:
            symbol: Ticker or coin identifier (e.g. 'BTC-USD', 'bitcoin', 'SOL').

        Returns:
            CoinGecko coin ID string (e.g. 'bitcoin').
        """
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]

        norm = symbol.strip().upper()
        if norm in self.symbol_map:
            return self.symbol_map[norm]

        if norm.endswith("-USD"):
            norm_no_usd = norm[:-4]
            if norm_no_usd in self.symbol_map:
                return self.symbol_map[norm_no_usd]
            return norm_no_usd.lower()
        elif norm.endswith("USD"):
            norm_no_usd = norm[:-3]
            if norm_no_usd in self.symbol_map:
                return self.symbol_map[norm_no_usd]
            return norm_no_usd.lower()

        return symbol.strip().lower()

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Fetch daily historical OHLCV bars for a crypto asset between start and end dates.

        Args:
            symbol: Ticker symbol or coin id (e.g. 'BTC-USD', 'bitcoin').
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of Bar dataclass instances sorted chronologically.
        """
        if start > end:
            return []

        coin_id = self.resolve_coin_id(symbol)
        from_ts = int(datetime.combine(start, time.min, tzinfo=UTC).timestamp())
        to_ts = int(
            datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC).timestamp()
        )

        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params: dict[str, Any] = {
            "vs_currency": "usd",
            "from": str(from_ts),
            "to": str(to_ts),
        }
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key

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

        return self._parse_market_chart_data(data, start, end)

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        """Fetch the most recent N closed daily bars for a crypto symbol.

        Args:
            symbol: Ticker symbol or coin id (e.g. 'BTC-USD').
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

    def _parse_market_chart_data(self, data: Any, start: date, end: date) -> list[Bar]:
        """Parse raw CoinGecko market chart JSON or OHLC list into clean Bar objects."""
        if not data:
            return []

        if isinstance(data, list):
            bars: list[Bar] = []
            for item in data:
                if not isinstance(item, (list, tuple)) or len(item) < 5:
                    continue
                ts_ms = item[0]
                dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
                d = dt.date()
                if start <= d <= end:
                    bars.append(
                        Bar(
                            ts=f"{d.isoformat()}T00:00:00Z",
                            open=float(item[1]),
                            high=float(item[2]),
                            low=float(item[3]),
                            close=float(item[4]),
                            adj_close=None,
                            volume=float(item[5]) if len(item) > 5 else 0.0,
                        )
                    )
            bars.sort(key=lambda b: b.ts)
            return bars

        if not isinstance(data, dict):
            return []

        prices: list[list[float]] = data.get("prices", [])
        total_volumes: list[list[float]] = data.get("total_volumes", [])

        if not prices:
            return []

        daily_prices: dict[date, list[float]] = defaultdict(list)
        for entry in prices:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            ts_ms, price = entry[0], entry[1]
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
            daily_prices[dt.date()].append(float(price))

        daily_volumes: dict[date, list[float]] = defaultdict(list)
        for entry in total_volumes:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            ts_ms, vol = entry[0], entry[1]
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
            daily_volumes[dt.date()].append(float(vol))

        bars = []
        for d in sorted(daily_prices.keys()):
            if not (start <= d <= end):
                continue
            p_list = daily_prices[d]
            if not p_list:
                continue

            v_list = daily_volumes.get(d, [0.0])
            vol = float(v_list[-1]) if v_list else 0.0

            open_p = p_list[0]
            high_p = max(p_list)
            low_p = min(p_list)
            close_p = p_list[-1]

            bars.append(
                Bar(
                    ts=f"{d.isoformat()}T00:00:00Z",
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    adj_close=None,
                    volume=vol,
                )
            )

        return bars
