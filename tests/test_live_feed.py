"""Unit tests for the Coinbase real-time price feed (v2 M1).

Channel + payload shapes are from docs/spikes/2026-09-01-M1-coinbase-ws.md
(live keyless probe): advanced-trade-ws.coinbase.com, channel ``ticker_batch``.
"""

from __future__ import annotations

import httpx
import pytest

from stock_forecasting.live_feed import (
    Tick,
    _backoff_delay,
    _normalize_ts,
    _parse_message,
    coinbase_rest_candles,
)

# --- a real ticker_batch envelope captured in the M1 spike ---
_BATCH = {
    "channel": "ticker_batch",
    "timestamp": "2026-09-01T10:16:18.822710088Z",
    "sequence_num": 2,
    "events": [
        {
            "type": "update",
            "tickers": [
                {
                    "type": "ticker",
                    "product_id": "BTC-USD",
                    "price": "77863.33",
                    "price_percent_chg_24_h": "-0.70998379242057",
                }
            ],
        }
    ],
}


def test_normalize_ts_z_to_offset() -> None:
    assert _normalize_ts("2026-09-01T10:16:18.822710088Z").endswith("+00:00")
    assert _normalize_ts("2026-09-01T10:16:18+00:00").endswith("+00:00")


def test_parse_ticker_batch_message_to_ticks() -> None:
    ticks = _parse_message(_BATCH)
    assert len(ticks) == 1
    t = ticks[0]
    assert isinstance(t, Tick)
    assert t.product_id == "BTC-USD"
    assert t.price == pytest.approx(77863.33)
    assert t.event_ts.endswith("+00:00")
    assert t.received_at.endswith("+00:00")


def test_parse_ignores_non_ticker_batch() -> None:
    assert _parse_message({"channel": "subscriptions", "events": []}) == []
    assert _parse_message({"channel": "heartbeats"}) == []


def test_backoff_delay_sequence_caps_at_60() -> None:
    seq = [_backoff_delay(i) for i in range(8)]
    assert seq[:4] == [1, 2, 4, 8]
    assert seq[-1] == 60  # 2**7 == 128 -> capped


def test_coinbase_rest_candles_maps_rows_and_flags_provisional() -> None:
    # Coinbase exchange candles: [time, low, high, open, close, volume], newest first
    now = 1_756_720_800  # fixed epoch seconds
    rows = [
        [now, 100.0, 110.0, 105.0, 108.0, 3.0],  # current bucket, still forming
        [now - 300, 95.0, 101.0, 96.0, 100.0, 5.0],  # closed bucket
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/products/BTC-USD/candles" in request.url.path
        assert request.url.params["granularity"] == "300"
        return httpx.Response(200, json=rows)

    client = httpx.Client(
        base_url="https://api.exchange.coinbase.com",
        transport=httpx.MockTransport(handler),
    )
    out = coinbase_rest_candles(
        "BTC-USD", granularity=300, now_epoch=now + 60, client=client
    )

    assert out[0]["ts"].endswith("+00:00")
    assert out[0]["open"] == 105.0 and out[0]["close"] == 108.0
    assert out[0]["is_provisional"] is True  # now+60 < bucket_start+300
    assert out[1]["is_provisional"] is False


def test_ws_client_dispatches_ticks_and_reconnects(monkeypatch) -> None:
    """The loop connects, forwards parsed ticks, and retries after a drop."""
    import stock_forecasting.live_feed as lf

    sleeps: list[float] = []

    async def fake_sleep(secs: float) -> None:
        sleeps.append(secs)

    monkeypatch.setattr(lf.asyncio, "sleep", fake_sleep)

    class FakeWS:
        def __init__(self, messages: list, *, then_raise: bool) -> None:
            self._messages = list(messages)
            self._then_raise = then_raise

        async def send(self, _payload: str) -> None:
            return None

        async def recv(self) -> str:
            import json

            if self._messages:
                return json.dumps(self._messages.pop(0))
            if self._then_raise:
                self._then_raise = False
                raise lf.ConnectionClosed(None, None)
            raise _StopLoop()

    attempts = {"n": 0}

    class FakeConnect:
        def __init__(self, *_a, **_kw) -> None:
            pass

        async def __aenter__(self):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return FakeWS([_BATCH], then_raise=True)
            return FakeWS([_BATCH], then_raise=False)

        async def __aexit__(self, *_exc) -> None:
            return None

    got: list[Tick] = []
    client = lf.CoinbaseWSClient(
        url="wss://advanced-trade-ws.coinbase.com",
        product_ids=["BTC-USD"],
        on_tick=got.append,
        connect=FakeConnect,
    )
    with pytest.raises(_StopLoop):
        client._run_sync()

    assert len(got) == 2  # one tick per connection
    assert attempts["n"] == 2  # reconnected after the ConnectionClosed
    assert sleeps and sleeps[0] == 1  # backoff slept 1s before the retry


class _StopLoop(BaseException):
    """Test-only sentinel to break the client loop deterministically.

    Subclasses BaseException so the client's ``except Exception`` backoff guard
    does not swallow it.
    """
