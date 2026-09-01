"""Chaos and fault tolerance test suite for stock_forecasting."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.circuit_breaker import CircuitBreaker
from stock_forecasting.database import get_session
from stock_forecasting.health_checks import HealthChecker
from stock_forecasting.ingestion import IngestionService
from stock_forecasting.providers.base import Bar
from stock_forecasting.providers.fake import FakeProvider
from stock_forecasting.schema import QuarantineBar, Ticker


@pytest.fixture
def chaos_db(temp_db):
    """Database fixture seeded with active tickers for chaos testing."""
    with get_session(temp_db) as session:
        t1 = Ticker(
            symbol="AAPL",
            asset_class="equity",
            display_name="Apple Inc",
            provider="yfinance",
            provider_symbol="AAPL",
            price_basis="adjusted",
            added_at=datetime.now(UTC).isoformat(),
            active=1,
        )
        t2 = Ticker(
            symbol="BTC-USD",
            asset_class="crypto",
            display_name="Bitcoin",
            provider="coingecko",
            provider_symbol="BTC-USD",
            price_basis="raw",
            added_at=datetime.now(UTC).isoformat(),
            active=1,
        )
        session.add(t1)
        session.add(t2)
        session.commit()
    return temp_db


def test_chaos_provider_429_trips_breaker_and_fails_over(chaos_db) -> None:
    """Chaos: Provider returns 429 rate limit mid-poll -> trips breaker, uses fallback, doesn't crash."""
    failing_primary = FakeProvider(return_429=True)
    working_fallback = FakeProvider()

    providers = {
        "yfinance": failing_primary,
        "tiingo": working_fallback,
    }

    with get_session(chaos_db) as session:
        breaker = CircuitBreaker(session, failure_threshold=1)
        service = IngestionService(
            session, providers=providers, circuit_breaker=breaker
        )

        # Poll AAPL (primary yfinance -> fails -> falls back to tiingo)
        res = service.poll_ticker("AAPL")
        assert res["symbol"] == "AAPL"
        assert res["failover_from"] == "yfinance"
        assert res["provider"] == "tiingo"
        assert res["inserted"] > 0

        # Verify breaker for yfinance is now open
        assert breaker.check_state("yfinance") == "open"


def test_chaos_provider_malformed_quarantines_rows(chaos_db) -> None:
    """Chaos: Provider returns malformed bars -> quarantined, no crash."""
    malformed_provider = FakeProvider(return_malformed=True)

    with get_session(chaos_db) as session:
        repo = BarRepository(session)
        # Attempt to upsert malformed bars
        inserted = repo.upsert_bars(
            "AAPL", malformed_provider.get_latest_bars("AAPL"), source="yfinance"
        )
        assert inserted == 0

        # Check quarantine_bars table
        quarantined = session.exec(select(QuarantineBar)).all()
        assert len(quarantined) > 0


def test_chaos_out_of_order_timestamps(chaos_db) -> None:
    """Chaos: Ingest out-of-order bars -> FeatureBuilder and BarRepository sort cleanly."""
    dates = [
        "2023-01-10T00:00:00Z",
        "2023-01-01T00:00:00Z",
        "2023-01-05T00:00:00Z",
    ]
    bars = [
        Bar(
            ts=d,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            adj_close=102.0,
            volume=1e6,
        )
        for d in dates
    ]

    with get_session(chaos_db) as session:
        repo = BarRepository(session)
        repo.upsert_bars("AAPL", bars, source="fake")

        fetched = repo.get_range("AAPL", "2023-01-01T00:00:00Z", "2023-01-30T00:00:00Z")
        ts_list = [b.ts for b in fetched]
        assert ts_list == sorted(ts_list)


def test_chaos_frozen_price_detection(chaos_db) -> None:
    """Chaos: 6+ identical close prices -> HealthChecker detects frozen price."""
    today = datetime.now(UTC)
    bars = []
    for i in range(10):
        dt_str = (today - timedelta(days=9 - i)).strftime("%Y-%m-%dT00:00:00Z")
        bars.append(
            Bar(
                ts=dt_str,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                adj_close=100.0,
                volume=1000.0,
            )
        )

    with get_session(chaos_db) as session:
        repo = BarRepository(session)
        repo.upsert_bars("AAPL", bars, source="yfinance")

        checker = HealthChecker(session)
        res = checker.check_gaps()
        assert "Frozen price" in res.message or res.status != "NOMINAL"


def test_chaos_future_dated_bars_detected(chaos_db) -> None:
    """Chaos: Future-dated bar -> HealthChecker detects clock skew."""
    future_ts = (datetime.now(UTC) + timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z")
    bar = Bar(
        ts=future_ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        adj_close=102.0,
        volume=1e6,
    )

    with get_session(chaos_db) as session:
        repo = BarRepository(session)
        repo.upsert_bars("AAPL", [bar], source="yfinance")

        checker = HealthChecker(session)
        res = checker.check_clock()
        assert "Future-dated" in res.message or res.status != "NOMINAL"


def test_chaos_ws_drop_mid_session_reconnects(monkeypatch) -> None:
    """Chaos: WS drops mid-session -> client reconnects, ticks resume, thread doesn't crash."""
    import json

    import stock_forecasting.live_feed as lf
    from stock_forecasting.live_feed import CoinbaseWSClient, Tick

    batch_1 = {
        "channel": "ticker_batch",
        "timestamp": "2026-09-01T10:16:18.822710088Z",
        "sequence_num": 1,
        "events": [
            {
                "type": "update",
                "tickers": [
                    {
                        "type": "ticker",
                        "product_id": "BTC-USD",
                        "price": "77860.00",
                    }
                ],
            }
        ],
    }
    batch_2 = {
        "channel": "ticker_batch",
        "timestamp": "2026-09-01T10:16:19.822710088Z",
        "sequence_num": 2,
        "events": [
            {
                "type": "update",
                "tickers": [
                    {
                        "type": "ticker",
                        "product_id": "BTC-USD",
                        "price": "77870.00",
                    }
                ],
            }
        ],
    }

    async def fake_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(lf.asyncio, "sleep", fake_sleep)

    class _StopLoop(BaseException):
        pass

    class FakeWS:
        def __init__(self, msgs: list, *, drop_after: bool = False) -> None:
            self._msgs = list(msgs)
            self._drop_after = drop_after

        async def send(self, _payload: str) -> None:
            return None

        async def recv(self) -> str:
            if self._msgs:
                return json.dumps(self._msgs.pop(0))
            if self._drop_after:
                self._drop_after = False
                raise lf.ConnectionClosed(None, None)
            raise _StopLoop()

    conn_count = {"n": 0}

    class FakeConnect:
        def __init__(self, *_a, **_kw) -> None:
            pass

        async def __aenter__(self):
            conn_count["n"] += 1
            if conn_count["n"] == 1:
                return FakeWS([batch_1], drop_after=True)
            return FakeWS([batch_2], drop_after=False)

        async def __aexit__(self, *_exc) -> None:
            return None

    received_ticks: list[Tick] = []
    client = CoinbaseWSClient(
        url="wss://advanced-trade-ws.coinbase.com",
        product_ids=["BTC-USD"],
        on_tick=received_ticks.append,
        connect=FakeConnect,
    )

    with pytest.raises(_StopLoop):
        client._run_sync()

    assert conn_count["n"] == 2
    assert len(received_ticks) == 2
    assert received_ticks[0].price == 77860.00
    assert received_ticks[1].price == 77870.00


def test_chaos_ws_silent_heartbeat_stops_triggers_rest(chaos_db, monkeypatch) -> None:
    """Chaos: WS heartbeat stops / idle timeout exceeded -> pulls REST candles, marks crypto feed degraded."""
    from stock_forecasting.config import Settings
    from stock_forecasting.intraday_store import IntradayRepository
    from stock_forecasting.worker import WorkerScheduler

    worker = WorkerScheduler(
        engine=chaos_db,
        settings=Settings(db_path=":memory:", live_ws_idle_timeout_sec=90),
    )

    class DummyWS:
        def __init__(self) -> None:
            self.seconds_since_last_message = 120.0
            self.status = "connected"
            self.product_ids = ["BTC-USD"]

    worker.ws_client = DummyWS()

    monkeypatch.setattr(
        "stock_forecasting.worker.coinbase_rest_candles",
        lambda pid, granularity: [
            {
                "ts": "2026-09-01T12:00:00+00:00",
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 10.0,
                "is_provisional": False,
            }
        ],
    )

    worker.job_check_ws_idle()

    with get_session(chaos_db) as session:
        bars = IntradayRepository(session).get_recent("BTC-USD", "1m")
        assert len(bars) >= 1
        assert bars[-1].source == "coinbase_rest"

        checker = HealthChecker(session)
        res = checker.check_live_feed_crypto()
        assert res.status in ("DEGRADED", "CRITICAL")


def test_chaos_mac_sleep_wake_time_jump(chaos_db) -> None:
    """Chaos: Mac sleep/wake creates 2h jump -> deduplication works cleanly, prune remains bounded."""
    from stock_forecasting.config import Settings
    from stock_forecasting.intraday_store import IntradayRepository
    from stock_forecasting.worker import WorkerScheduler

    worker = WorkerScheduler(
        engine=chaos_db,
        settings=Settings(db_path=":memory:", intraday_retention_days=1),
    )

    base_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    with get_session(chaos_db) as session:
        repo = IntradayRepository(session)
        repo.upsert_forming(
            "AAPL", "5m", (base_time - timedelta(days=3)).isoformat(), price=150.0
        )
        repo.upsert_forming("AAPL", "5m", base_time.isoformat(), price=151.0)
        repo.upsert_forming("AAPL", "5m", base_time.isoformat(), price=152.0)

    wake_time = base_time + timedelta(hours=2)
    with get_session(chaos_db) as session:
        repo = IntradayRepository(session)
        repo.upsert_forming("AAPL", "5m", wake_time.isoformat(), price=155.0)

    worker.job_prune_intraday()

    with get_session(chaos_db) as session:
        repo = IntradayRepository(session)
        bars = repo.get_recent("AAPL", "5m")
        timestamps = [b.ts for b in bars]
        assert len(timestamps) == len(set(timestamps))
        assert not any("2026-08-29" in ts for ts in timestamps)


def test_chaos_equity_429_storm_degraded_not_critical(chaos_db, monkeypatch) -> None:
    """Chaos: yfinance 429 storm during intraday ingest -> status is DEGRADED not CRITICAL."""
    from stock_forecasting.config import Settings
    from stock_forecasting.intraday_store import IntradayRepository
    from stock_forecasting.providers.base import ProviderError
    from stock_forecasting.worker import WorkerScheduler

    with get_session(chaos_db) as session:
        repo = IntradayRepository(session)
        repo.upsert_forming("AAPL", "5m", "2026-09-01T14:00:00+00:00", price=150.0)

    worker = WorkerScheduler(engine=chaos_db, settings=Settings(db_path=":memory:"))

    class FlakyYFinance:
        def get_intraday_bars(self, *args, **kwargs):
            raise ProviderError("HTTP 429: Too Many Requests")

    worker.providers["yfinance"] = FlakyYFinance()
    worker.job_ingest_equity_intraday()

    with get_session(chaos_db) as session:
        bars = IntradayRepository(session).get_recent("AAPL", "5m")
        assert len(bars) == 1
        assert bars[0].close == 150.0

        checker = HealthChecker(session)
        status, _warnings = checker.compute_system_status()
        assert status == "DEGRADED"


def test_chaos_eod_reconcile_and_crypto_jump_tolerance() -> None:
    """Chaos: EOD reconcile / crypto step difference -> figure builds cleanly, CI band unchanged."""
    from types import SimpleNamespace

    from stock_forecasting.viz import add_live_price_line, build_price_figure

    bars = [
        SimpleNamespace(
            ts=f"2026-08-{d:02d}T00:00:00Z",
            open=100.0 + d,
            high=102.0 + d,
            low=98.0 + d,
            close=100.0 + d,
        )
        for d in range(1, 30)
    ]
    snapshot = SimpleNamespace(
        horizon="5d",
        made_at="2026-08-29T00:00:00Z",
        made_from_ts="2026-08-29T00:00:00Z",
        target_ts="2026-09-03T00:00:00Z",
        anchor_price=129.0,
        predicted_price=132.0,
        lower_bound=120.0,
        upper_bound=144.0,
        model_type="ridge",
        model_version="2.0.0",
        realized_price=None,
        evaluated_at=None,
        is_direction_hit=None,
    )

    intraday = [
        SimpleNamespace(
            ts="2026-08-29T01:00:00Z",
            open=129.5,
            high=131.0,
            low=129.0,
            close=130.5,
            is_provisional=0,
        ),
        SimpleNamespace(
            ts="2026-08-29T02:00:00Z",
            open=130.5,
            high=136.0,
            low=130.0,
            close=135.0,
            is_provisional=1,
        ),
    ]
    quotes = [
        SimpleNamespace(price=135.0, ts="2026-08-29T02:05:00Z", source="coinbase_ws")
    ]

    fig = build_price_figure(
        bars,
        [snapshot],
        ribbon_horizon="5d",
        latest_horizons=("5d",),
        title="BTC-USD",
    )
    add_live_price_line(fig, quotes, intraday)

    names = [t.name for t in fig.data]
    assert "Actual" in names
    assert "live" in names
    assert "forming" in names

    ci_lower = next(t for t in fig.data if (t.meta or {}).get("kind") == "ci_lower")
    ci_upper = next(t for t in fig.data if (t.meta or {}).get("kind") == "ci_upper")
    assert ci_lower.y[-1] == 120.0
    assert ci_upper.y[-1] == 144.0
