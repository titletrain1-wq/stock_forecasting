"""Coinbase real-time price feed for the v2 live display path.

Primary source for crypto last-price. Owned by ``worker.py`` (one connection,
regardless of how many Streamlit tabs are open - see docs/2026-09-01-realtime-v2-design.md).

URL + channel confirmed by a live keyless probe in
docs/spikes/2026-09-01-M1-coinbase-ws.md:
  wss://advanced-trade-ws.coinbase.com  ·  channel "ticker_batch" (+ "heartbeats")

This module has no Streamlit, no SQLModel and no APScheduler imports: the worker
wires the callback to a repository writer in M2.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
REST_CANDLES_URL = "https://api.exchange.coinbase.com"
_BACKOFF_CAP_SEC = 60


@dataclass
class Tick:
    """A single last-price observation for one product."""

    product_id: str
    price: float
    event_ts: str  # provider event time, ISO-8601 UTC (+00:00)
    received_at: str  # wall-clock at receipt, ISO-8601 UTC (+00:00)
    pct_24h: float | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_ts(ts: str) -> str:
    """Coerce a provider timestamp to ISO-8601 with a ``+00:00`` offset."""
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).astimezone(UTC).isoformat()
    except ValueError:
        # Coinbase sends nanosecond precision; trim to microseconds and retry.
        if "." in raw:
            head, rest = raw.split(".", 1)
            frac = rest
            offset = ""
            for marker in ("+", "-"):
                if marker in frac:
                    frac, off = frac.split(marker, 1)
                    offset = marker + off
                    break
            return (
                datetime.fromisoformat(f"{head}.{frac[:6]}{offset or '+00:00'}")
                .astimezone(UTC)
                .isoformat()
            )
        raise


def _parse_message(raw: dict[str, Any]) -> list[Tick]:
    """Extract ``Tick``s from one WebSocket frame. Non-price frames -> ``[]``."""
    if raw.get("channel") != "ticker_batch":
        return []
    event_ts = _normalize_ts(raw.get("timestamp", _now_iso()))
    received_at = _now_iso()
    ticks: list[Tick] = []
    for event in raw.get("events", []):
        for t in event.get("tickers", []):
            try:
                price = float(t["price"])
            except (KeyError, TypeError, ValueError):
                continue
            pct_raw = t.get("price_percent_chg_24_h")
            ticks.append(
                Tick(
                    product_id=t["product_id"],
                    price=price,
                    event_ts=event_ts,
                    received_at=received_at,
                    pct_24h=float(pct_raw) if pct_raw is not None else None,
                )
            )
    return ticks


def _backoff_delay(attempt: int, cap: int = _BACKOFF_CAP_SEC) -> int:
    """Exponential backoff: 1, 2, 4, 8, ... capped at ``cap`` seconds."""
    return min(2**attempt, cap)


def coinbase_rest_candles(
    product_id: str,
    granularity: int,
    start: str | None = None,
    end: str | None = None,
    *,
    now_epoch: float | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """REST fallback when the WebSocket is idle/down.

    Coinbase exchange candles return ``[time, low, high, open, close, volume]``
    rows, newest first. The most recent bucket is flagged provisional while
    ``now < bucket_start + granularity``.
    """
    owns_client = client is None
    client = client or httpx.Client(
        base_url=REST_CANDLES_URL,
        timeout=10.0,
        headers={"User-Agent": "stock-forecasting/0.2.0", "Accept": "application/json"},
    )
    params: dict[str, Any] = {"granularity": granularity}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    try:
        resp = client.get(f"/products/{product_id}/candles", params=params)
        resp.raise_for_status()
        rows = resp.json()
    finally:
        if owns_client:
            client.close()

    now = now_epoch if now_epoch is not None else datetime.now(UTC).timestamp()
    out: list[dict[str, Any]] = []
    for row in rows:
        bucket_start = int(row[0])
        out.append(
            {
                "ts": datetime.fromtimestamp(bucket_start, UTC).isoformat(),
                "open": float(row[3]),
                "high": float(row[2]),
                "low": float(row[1]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "is_provisional": now < bucket_start + granularity,
            }
        )
    return out


class CoinbaseWSClient:
    """Blocking WebSocket client. Run ``run_forever`` inside a daemon thread."""

    def __init__(
        self,
        url: str,
        product_ids: list[str],
        on_tick: Callable[[Tick], None],
        *,
        idle_timeout_sec: int = 90,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        self.url = url
        self.product_ids = product_ids
        self.on_tick = on_tick
        self.idle_timeout_sec = idle_timeout_sec
        self._connect = connect or websockets.connect
        self._stopping = False
        self._status = "stopped"
        self._last_message_at: datetime | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def seconds_since_last_message(self) -> float:
        if self._last_message_at is None:
            return float("inf")
        return (datetime.now(UTC) - self._last_message_at).total_seconds()

    def stop(self) -> None:
        self._stopping = True
        self._status = "stopped"

    def run_forever(self) -> None:
        """Entry point for the worker's daemon thread."""
        asyncio.run(self._run())

    def _run_sync(self) -> None:
        """Test seam: drive the async loop on the current thread."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()

    async def _run(self) -> None:
        attempt = 0
        while not self._stopping:
            try:
                self._status = "connecting"
                async with self._connect(self.url) as ws:
                    await self._subscribe(ws)
                    self._status = "connected"
                    attempt = 0
                    await self._read_loop(ws)
            except ConnectionClosed:
                if self._stopping:
                    break
                self._status = "reconnecting"
                await asyncio.sleep(_backoff_delay(attempt))
                attempt += 1
            except Exception:
                if self._stopping:
                    break
                logger.exception("Coinbase WS loop error; backing off")
                self._status = "reconnecting"
                await asyncio.sleep(_backoff_delay(attempt))
                attempt += 1

    async def _subscribe(self, ws: Any) -> None:
        for channel in ("ticker_batch", "heartbeats"):
            await ws.send(
                json.dumps(
                    {
                        "type": "subscribe",
                        "product_ids": self.product_ids,
                        "channel": channel,
                    }
                )
            )

    async def _read_loop(self, ws: Any) -> None:
        while not self._stopping:
            raw = await ws.recv()
            self._last_message_at = datetime.now(UTC)
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            for tick in _parse_message(msg):
                self.on_tick(tick)
