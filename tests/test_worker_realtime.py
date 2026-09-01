"""Tests for the v2 live-feed integration in WorkerScheduler (M2.2)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine
from sqlmodel import Session

from stock_forecasting.config import Settings
from stock_forecasting.intraday_store import IntradayRepository
from stock_forecasting.live_feed import Tick
from stock_forecasting.schema import LiveQuote, SystemHeartbeat, Ticker
from stock_forecasting.worker import WorkerScheduler


def _seed_crypto(session: Session, symbol: str = "BTC-USD") -> None:
    session.add(
        Ticker(
            symbol=symbol,
            asset_class="crypto",
            display_name=symbol,
            provider="coinbase",
            provider_symbol=symbol,
            price_basis="raw",
            added_at="2026-01-01T00:00:00+00:00",
            active=1,
        )
    )
    session.commit()


class _FakeWSClient:
    """Stands in for CoinbaseWSClient: run_forever blocks until stop()."""

    def __init__(self, url, product_ids, on_tick, idle_timeout_sec=90) -> None:
        self.product_ids = product_ids
        self.on_tick = on_tick
        self._release = threading.Event()
        self.seconds_since_last_message = 0.0
        self.status = "connected"

    def run_forever(self) -> None:
        self._release.wait(timeout=5)

    def stop(self) -> None:
        self._release.set()


def test_start_stop_live_feed_thread_lifecycle(
    temp_db: Engine, db_session: Session, monkeypatch
) -> None:
    _seed_crypto(db_session)
    monkeypatch.setattr("stock_forecasting.worker.CoinbaseWSClient", _FakeWSClient)
    worker = WorkerScheduler(engine=temp_db, settings=Settings(db_path=":memory:"))

    worker.start()
    try:
        assert worker._ws_thread is not None and worker._ws_thread.is_alive()
        assert worker.ws_client.product_ids == ["BTC-USD"]
    finally:
        worker.stop(wait=False)

    assert worker.ws_client is None
    assert worker._ws_thread is None


def test_live_feed_not_started_when_disabled(
    temp_db: Engine, db_session: Session, monkeypatch
) -> None:
    _seed_crypto(db_session)
    monkeypatch.setattr("stock_forecasting.worker.CoinbaseWSClient", _FakeWSClient)
    worker = WorkerScheduler(
        engine=temp_db, settings=Settings(db_path=":memory:", live_ws_enabled=False)
    )
    worker.start()
    try:
        assert worker.ws_client is None
    finally:
        worker.stop(wait=False)


def test_on_tick_writes_live_quote_and_forming_bar(temp_db: Engine) -> None:
    worker = WorkerScheduler(engine=temp_db, settings=Settings(db_path=":memory:"))
    tick = Tick(
        product_id="BTC-USD",
        price=77850.25,
        event_ts="2026-09-01T10:16:18+00:00",
        received_at="2026-09-01T10:16:18.900000+00:00",
    )
    worker._on_tick(tick)

    with Session(temp_db) as session:
        quote = session.get(LiveQuote, "BTC-USD")
        assert quote is not None and quote.price == 77850.25
        bars = IntradayRepository(session).get_recent("BTC-USD", "1m")
        assert len(bars) == 1
        assert bars[0].ts == "2026-09-01T10:16:00+00:00"
        assert bars[0].close == 77850.25
        assert bars[0].is_provisional == 1


def test_job_prune_intraday_deletes_and_heartbeats(temp_db: Engine) -> None:
    worker = WorkerScheduler(
        engine=temp_db, settings=Settings(db_path=":memory:", intraday_retention_days=7)
    )
    now = datetime.now(UTC)
    with Session(temp_db) as session:
        repo = IntradayRepository(session)
        repo.upsert_forming(
            "BTC-USD", "1m", (now - timedelta(days=10)).isoformat(), price=1.0
        )
        repo.upsert_forming(
            "BTC-USD", "1m", (now - timedelta(hours=1)).isoformat(), price=2.0
        )

    worker.job_prune_intraday()

    with Session(temp_db) as session:
        assert len(IntradayRepository(session).get_recent("BTC-USD", "1m")) == 1
        hb = session.get(SystemHeartbeat, "job_prune_intraday")
        assert hb is not None and hb.last_success_ts is not None


def test_job_check_ws_idle_triggers_rest_fallback(
    temp_db: Engine, db_session: Session, monkeypatch
) -> None:
    _seed_crypto(db_session)
    worker = WorkerScheduler(engine=temp_db, settings=Settings(db_path=":memory:"))

    stub = _FakeWSClient("u", ["BTC-USD"], worker._on_tick)
    stub.seconds_since_last_message = 999.0
    worker.ws_client = stub

    monkeypatch.setattr(
        "stock_forecasting.worker.coinbase_rest_candles",
        lambda pid, granularity: [
            {
                "ts": "2026-09-01T10:15:00+00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 3.0,
                "is_provisional": False,
            }
        ],
    )

    worker.job_check_ws_idle()

    with Session(temp_db) as session:
        bars = IntradayRepository(session).get_recent("BTC-USD", "1m")
        assert bars and bars[-1].source == "coinbase_rest"
        assert bars[-1].is_provisional == 0
        hb = session.get(SystemHeartbeat, "live_feed_crypto")
        assert hb is not None and hb.consecutive_failures >= 1
        assert "REST fallback" in (hb.last_error or "")


def test_job_check_ws_idle_healthy_when_recent(
    temp_db: Engine, db_session: Session
) -> None:
    _seed_crypto(db_session)
    worker = WorkerScheduler(engine=temp_db, settings=Settings(db_path=":memory:"))
    stub = _FakeWSClient("u", ["BTC-USD"], worker._on_tick)
    stub.seconds_since_last_message = 2.0
    worker.ws_client = stub

    worker.job_check_ws_idle()

    with Session(temp_db) as session:
        hb = session.get(SystemHeartbeat, "live_feed_crypto")
        assert hb is not None and hb.last_success_ts is not None
        assert hb.consecutive_failures == 0
