"""Tests for v2 real-time display-path health checks and system status cap."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from stock_forecasting.database import create_tables
from stock_forecasting.health_checks import HealthChecker
from stock_forecasting.health_view import LiveFeedRow, build_health_view
from stock_forecasting.schema import (
    IntradayBar,
    LiveQuote,
    OhlcvBar,
    SystemHeartbeat,
    Ticker,
)


@pytest.fixture
def health_db(temp_db):
    """Database engine with tables created and seeded tickers."""
    create_tables(temp_db)
    now = datetime.now(UTC).isoformat()
    with Session(temp_db) as session:
        session.add(
            Ticker(
                symbol="AAPL",
                asset_class="equity",
                display_name="Apple",
                provider="yfinance",
                provider_symbol="AAPL",
                price_basis="adjusted",
                added_at=now,
                active=1,
            )
        )
        session.add(
            Ticker(
                symbol="BTC-USD",
                asset_class="crypto",
                display_name="Bitcoin",
                provider="coinbase",
                provider_symbol="BTC-USD",
                price_basis="raw",
                added_at=now,
                active=1,
            )
        )
        session.commit()
    return temp_db


def test_display_down_is_degraded_not_critical(health_db) -> None:
    """Verify display feed down (no live quotes / intraday bars) caps system status at DEGRADED when training data is nominal."""
    now = datetime.now(UTC)
    # Seed current daily bars (training path nominal)
    with Session(health_db) as session:
        today_ts = now.strftime("%Y-%m-%dT00:00:00Z")
        session.add(
            OhlcvBar(
                ticker="AAPL",
                interval="1d",
                ts=today_ts,
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                adj_close=151.0,
                volume=1e6,
                source="yfinance",
                ingested_at=now.isoformat(),
            )
        )
        session.add(
            OhlcvBar(
                ticker="BTC-USD",
                interval="1d",
                ts=today_ts,
                open=60000.0,
                high=61000.0,
                low=59000.0,
                close=60500.0,
                adj_close=60500.0,
                volume=100.0,
                source="coinbase",
                ingested_at=now.isoformat(),
            )
        )

        # Seed stale live quote (120s old -> display check CRITICAL)
        session.add(
            LiveQuote(
                ticker="BTC-USD",
                price=60500.0,
                ts=(now - timedelta(seconds=120)).isoformat(),
                received_at=(now - timedelta(seconds=120)).isoformat(),
                source="coinbase_ws",
            )
        )
        # Seed fresh worker heartbeats
        session.add(
            SystemHeartbeat(
                job_type="job_heartbeat",
                worker_pid=1234,
                last_pulse_ts=now.isoformat(),
                consecutive_failures=0,
            )
        )
        session.commit()

        checker = HealthChecker(session)
        status, warnings = checker.compute_system_status(now=now)

        # Live feed is down / missing, so display path warning is logged but system is DEGRADED, NOT CRITICAL
        assert status == "DEGRADED"
        assert any(
            "LIVE_FEED" in w or "DISPLAY" in w or "intraday" in w.lower()
            for w in warnings
        )


def test_training_stale_is_still_critical(health_db) -> None:
    """Verify training data stale (daily bar 4 days behind for crypto) still triggers CRITICAL system status regardless of live feeds."""
    now = datetime.now(UTC)
    with Session(health_db) as session:
        old_ts = (now - timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z")
        session.add(
            OhlcvBar(
                ticker="BTC-USD",
                interval="1d",
                ts=old_ts,
                open=60000.0,
                high=61000.0,
                low=59000.0,
                close=60500.0,
                adj_close=60500.0,
                volume=100.0,
                source="coinbase",
                ingested_at=now.isoformat(),
            )
        )

        # Seed fresh live quotes (live feed green)
        session.add(
            LiveQuote(
                ticker="BTC-USD",
                price=60500.0,
                ts=now.isoformat(),
                received_at=now.isoformat(),
                source="coinbase_ws",
            )
        )
        session.add(
            SystemHeartbeat(
                job_type="job_heartbeat",
                worker_pid=1234,
                last_pulse_ts=now.isoformat(),
                consecutive_failures=0,
            )
        )
        session.commit()

        checker = HealthChecker(session)
        status, warnings = checker.compute_system_status(now=now)

        # Training data stale -> system must be CRITICAL
        assert status == "CRITICAL"
        assert any("FRESHNESS" in w for w in warnings)


def test_check_live_feed_crypto_thresholds(health_db) -> None:
    """Verify check_live_feed_crypto status thresholds (<10s NOMINAL, 10-90s DEGRADED, >90s CRITICAL)."""
    now = datetime.now(UTC)
    with Session(health_db) as session:
        checker = HealthChecker(session)

        # 1. Fresh quote (5s old) -> NOMINAL
        session.add(
            LiveQuote(
                ticker="BTC-USD",
                price=60000.0,
                ts=(now - timedelta(seconds=5)).isoformat(),
                received_at=(now - timedelta(seconds=5)).isoformat(),
                source="coinbase_ws",
            )
        )
        session.commit()
        res_fresh = checker.check_live_feed_crypto(now=now)
        assert res_fresh.status == "NOMINAL"

        # 2. Aged quote (45s old) -> DEGRADED
        quote = session.get(LiveQuote, "BTC-USD")
        quote.received_at = (now - timedelta(seconds=45)).isoformat()
        session.add(quote)
        session.commit()
        res_degraded = checker.check_live_feed_crypto(now=now)
        assert res_degraded.status == "DEGRADED"

        # 3. Very old quote (120s old) -> CRITICAL
        quote.received_at = (now - timedelta(seconds=120)).isoformat()
        session.add(quote)
        session.commit()
        res_critical = checker.check_live_feed_crypto(now=now)
        assert res_critical.status == "CRITICAL"


def test_check_live_feed_equity_thresholds(health_db) -> None:
    """Verify check_live_feed_equity status thresholds (<25m NOMINAL, 25-45m DEGRADED, >45m CRITICAL)."""
    now = datetime.now(UTC)
    with Session(health_db) as session:
        checker = HealthChecker(session)

        # 1. Fresh bar (10m old) -> NOMINAL
        session.add(
            IntradayBar(
                ticker="AAPL",
                interval="5m",
                ts=(now - timedelta(minutes=10)).isoformat(),
                open=150.0,
                high=151.0,
                low=149.0,
                close=150.5,
                volume=1000.0,
                is_provisional=1,
                source="yfinance_intraday",
                ingested_at=now.isoformat(),
            )
        )
        session.commit()
        res_fresh = checker.check_live_feed_equity(now=now)
        assert res_fresh.status == "NOMINAL"

        # 2. Aged bar (30m old) -> DEGRADED
        bar = session.exec(
            select(IntradayBar).where(IntradayBar.ticker == "AAPL")
        ).first()
        bar.ts = (now - timedelta(minutes=30)).isoformat()
        session.add(bar)
        session.commit()
        res_degraded = checker.check_live_feed_equity(now=now)
        assert res_degraded.status == "DEGRADED"

        # 3. Very old bar (60m old) -> CRITICAL
        bar.ts = (now - timedelta(minutes=60)).isoformat()
        session.add(bar)
        session.commit()
        res_critical = checker.check_live_feed_equity(now=now)
        assert res_critical.status == "CRITICAL"


def test_build_health_view_returns_live_feed_rows(health_db) -> None:
    """Verify build_health_view produces live_feed_rows."""
    with Session(health_db) as session:
        view = build_health_view(session)
        assert hasattr(view, "live_feed_rows")
        assert len(view.live_feed_rows) > 0
        assert all(isinstance(r, LiveFeedRow) for r in view.live_feed_rows)
