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
