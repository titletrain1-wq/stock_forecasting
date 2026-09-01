"""dYdX v4 Indexer derivatives provider (funding rate + open interest).

Public API, no auth: ``https://indexer.dydx.trade/v4/``.

- ``GET /perpetualMarkets`` → ``{"markets": {"BTC-USD": {"openInterest": "...", ...}}}``
  (current snapshot only — there is no historical open-interest endpoint).
- ``GET /historicalFunding/{ticker}?limit=1000&effectiveBeforeOrAt=<iso>`` →
  ``{"historicalFunding": [{"rate": "<hourly>", "effectiveAt": "<iso>"}, ...]}``,
  newest-first, ~1000 hourly rows per page.

``get_derivatives`` returns one ``Derivative`` per UTC day in range: ``funding_rate``
is the sum of that day's hourly rates (the daily funding cost); ``open_interest`` is
attached to the newest day only (snapshot), ``None`` elsewhere.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx

from stock_forecasting.providers.base import Derivative

logger = logging.getLogger(__name__)

DEFAULT_DYDX_MAP: dict[str, str] = {
    "BTC": "BTC-USD",
    "BTC-USD": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "bitcoin": "BTC-USD",
    "ETH": "ETH-USD",
    "ETH-USD": "ETH-USD",
    "ETHUSD": "ETH-USD",
    "ethereum": "ETH-USD",
}

_MAX_PAGES = (
    60  # ~7y of hourly funding; dYdX v4 history is shorter than this in practice
)


class DydxDerivativesProvider:
    """dYdX v4 Indexer funding-rate + open-interest provider."""

    def __init__(
        self,
        base_url: str = "https://indexer.dydx.trade/v4",
        symbol_map: dict[str, str] | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
        page_limit: int = 1000,
    ) -> None:
        """Initialize DydxDerivativesProvider.

        Args:
            base_url: Base URL for the dYdX v4 Indexer.
            symbol_map: Optional overrides for symbol -> dYdX market ticker.
            timeout: HTTP request timeout in seconds.
            client: Optional pre-configured httpx.Client.
            page_limit: Rows per ``/historicalFunding`` page (dYdX max 1000).
        """
        self.base_url = base_url.rstrip("/")
        self.symbol_map = {**DEFAULT_DYDX_MAP, **(symbol_map or {})}
        self.timeout = timeout
        self._client = client
        self.page_limit = page_limit

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            timeout=self.timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "stock-forecasting/0.1.0",
            },
        )

    def resolve_market(self, symbol: str) -> str:
        """Resolve a ticker to a dYdX market id (e.g. 'BTC-USD')."""
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        norm = symbol.strip().upper()
        if norm in self.symbol_map:
            return self.symbol_map[norm]
        if norm.endswith("-USD"):
            return norm
        if norm.endswith("USD"):
            return f"{norm[:-3]}-USD"
        return f"{norm}-USD"

    def _fetch_open_interest(self, client: httpx.Client, market: str) -> float | None:
        try:
            resp = client.get(
                f"{self.base_url}/perpetualMarkets", params={"ticker": market}
            )
            resp.raise_for_status()
            markets = resp.json().get("markets", {})
        except (httpx.HTTPError, ValueError):
            logger.warning("dYdX perpetualMarkets fetch failed for %s", market)
            return None
        entry = markets.get(market)
        if not isinstance(entry, dict):
            return None
        raw = entry.get("openInterest")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _fetch_hourly_funding(
        self, client: httpx.Client, market: str, start: date, end: date
    ) -> list[tuple[datetime, float]]:
        start_dt = datetime.combine(start, time.min, tzinfo=UTC)
        end_dt = datetime.combine(end, time.max, tzinfo=UTC)
        cursor = end_dt
        out: list[tuple[datetime, float]] = []

        for _ in range(_MAX_PAGES):
            params: dict[str, Any] = {
                "limit": self.page_limit,
                "effectiveBeforeOrAt": cursor.isoformat().replace("+00:00", "Z"),
            }
            resp = client.get(
                f"{self.base_url}/historicalFunding/{market}", params=params
            )
            resp.raise_for_status()
            rows = resp.json().get("historicalFunding", [])
            if not rows:
                break

            oldest_in_page: datetime | None = None
            for row in rows:
                eff = row.get("effectiveAt")
                rate = row.get("rate")
                if eff is None or rate is None:
                    continue
                dt = datetime.fromisoformat(eff).astimezone(UTC)
                oldest_in_page = (
                    dt if oldest_in_page is None else min(oldest_in_page, dt)
                )
                if start_dt <= dt <= end_dt:
                    try:
                        out.append((dt, float(rate)))
                    except (TypeError, ValueError):
                        continue

            if oldest_in_page is None or oldest_in_page <= start_dt:
                break
            cursor = oldest_in_page - timedelta(seconds=1)

        return out

    def get_derivatives(self, symbol: str, start: date, end: date) -> list[Derivative]:
        """Day-aggregated funding rate + (newest-day) open interest for [start, end]."""
        if start > end:
            return []

        market = self.resolve_market(symbol)
        client = self._client or self._get_client()
        close_client = self._client is None
        try:
            hourly = self._fetch_hourly_funding(client, market, start, end)
            open_interest = self._fetch_open_interest(client, market)
        finally:
            if close_client:
                client.close()

        daily: dict[date, float] = defaultdict(float)
        for dt, rate in hourly:
            daily[dt.date()] += rate

        if not daily:
            return []

        days = sorted(d for d in daily if start <= d <= end)
        newest = days[-1] if days else None
        return [
            Derivative(
                ts=f"{d.isoformat()}T00:00:00Z",
                funding_rate=daily[d],
                open_interest=open_interest if d == newest else None,
            )
            for d in days
        ]

    def get_latest_derivatives(
        self, symbol: str, lookback: int = 30
    ) -> list[Derivative]:
        """Convenience: the last ``lookback`` days of derivatives."""
        if lookback <= 0:
            return []
        today = datetime.now(UTC).date()
        return self.get_derivatives(symbol, today - timedelta(days=lookback), today)
