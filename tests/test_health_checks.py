"""Tests for HealthChecker data-feed and system health monitoring."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from sqlmodel import Session

from stock_forecasting.health_checks import HealthChecker
from stock_forecasting.schema import (
    LinkMetrics,
    OhlcvBar,
    QuarantineBar,
    SystemHeartbeat,
    Ticker,
)


def _seed_healthy_state(
    session: Session, now: datetime
) -> tuple[Ticker, Ticker, LinkMetrics, SystemHeartbeat]:
    """Helper to populate database with fully healthy system components."""
    now_iso = now.isoformat()

    crypto_ticker = Ticker(
        symbol="BTC",
        asset_class="crypto",
        display_name="Bitcoin",
        provider="fake",
        provider_symbol="BTC",
        price_basis="raw",
        added_at=now_iso,
        active=1,
    )
    equity_ticker = Ticker(
        symbol="AAPL",
        asset_class="equity",
        display_name="Apple Inc.",
        provider="fake",
        provider_symbol="AAPL",
        price_basis="adjusted",
        added_at=now_iso,
        active=1,
    )
    session.add(crypto_ticker)
    session.add(equity_ticker)

    # Bars: BTC 2m ago (<5m), AAPL 10m ago (<20m)
    crypto_bar = OhlcvBar(
        ticker="BTC",
        interval="1d",
        ts=(now - timedelta(minutes=2)).isoformat(),
        open=50000.0,
        high=51000.0,
        low=49500.0,
        close=50500.0,
        volume=100.0,
        source="fake",
        ingested_at=now_iso,
    )
    equity_bar = OhlcvBar(
        ticker="AAPL",
        interval="1d",
        ts=(now - timedelta(minutes=10)).isoformat(),
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
        volume=10000.0,
        source="fake",
        ingested_at=now_iso,
    )
    session.add(crypto_bar)
    session.add(equity_bar)

    # LinkMetrics: Low latency, 0 failures, 20% quota
    metric = LinkMetrics(
        provider="fake",
        rtt_p50_ms=250.0,
        rtt_p95_ms=600.0,
        rtt_jitter_ms=100.0,
        error_rate=0.0,
        consecutive_failures=0,
        breaker_state="closed",
        calls_today=200,
        daily_limit=1000,
        quota_pct=0.20,
        updated_at=now_iso,
    )
    session.add(metric)

    # Watchdog pulse: 30s ago (<5m)
    heartbeat = SystemHeartbeat(
        job_type="job_heartbeat",
        worker_pid=1234,
        last_pulse_ts=(now - timedelta(seconds=30)).isoformat(),
        last_success_ts=(now - timedelta(seconds=30)).isoformat(),
        consecutive_failures=0,
    )
    session.add(heartbeat)
    session.commit()
    return crypto_ticker, equity_ticker, metric, heartbeat


def test_health_checker_nominal(db_session: Session) -> None:
    """Verify NOMINAL system status when all providers, worker, and data are healthy."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    _seed_healthy_state(db_session, now)

    checker = HealthChecker(db_session)
    status, reasons = checker.compute_system_status(now=now)
    assert status == "NOMINAL"
    assert reasons == []


def test_health_checker_degraded(db_session: Session) -> None:
    """Verify DEGRADED status when provider quota > 80% or worker lag is 5-15m."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    _, _, metric, heartbeat = _seed_healthy_state(db_session, now)
    checker = HealthChecker(db_session)

    # 1. Test degraded due to quota > 80%
    metric.quota_pct = 0.85
    metric.calls_today = 850
    db_session.add(metric)
    db_session.commit()
    status, reasons = checker.compute_system_status(now=now)
    assert status == "DEGRADED"
    assert any("quota" in r.lower() for r in reasons)

    # Reset quota and test degraded due to worker heartbeat lag 5-15m
    metric.quota_pct = 0.20
    metric.calls_today = 200
    heartbeat.last_pulse_ts = (now - timedelta(minutes=8)).isoformat()
    db_session.add(metric)
    db_session.add(heartbeat)
    db_session.commit()

    status, reasons = checker.compute_system_status(now=now)
    assert status == "DEGRADED"
    assert any("worker heartbeat lag" in r.lower() for r in reasons)


def test_health_checker_critical(db_session: Session) -> None:
    """Verify CRITICAL status when worker lag > 15m or all providers are down."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    _, _, metric, heartbeat = _seed_healthy_state(db_session, now)
    checker = HealthChecker(db_session)

    # 1. Test critical due to worker lag > 15m (20 min)
    heartbeat.last_pulse_ts = (now - timedelta(minutes=20)).isoformat()
    db_session.add(heartbeat)
    db_session.commit()
    status, reasons = checker.compute_system_status(now=now)
    assert status == "CRITICAL"
    assert any("worker" in r.lower() and "15m" in r.lower() for r in reasons)

    # 2. Test critical due to all providers down (consecutive_failures >= 5)
    heartbeat.last_pulse_ts = (now - timedelta(seconds=30)).isoformat()
    metric.consecutive_failures = 5
    metric.breaker_state = "open"
    db_session.add(heartbeat)
    db_session.add(metric)
    db_session.commit()

    status, reasons = checker.compute_system_status(now=now)
    assert status == "CRITICAL"
    assert any("all primary providers down" in r.lower() for r in reasons)


def test_check_freshness(db_session: Session) -> None:
    """Test freshness evaluation for crypto and equity bar age boundaries."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    checker = HealthChecker(db_session)
    assert checker.check_freshness(now=now).status == "NOMINAL"

    ticker_crypto = Ticker(
        symbol="ETH",
        asset_class="crypto",
        display_name="Ethereum",
        provider="fake",
        provider_symbol="ETH",
        price_basis="raw",
        added_at=now.isoformat(),
        active=1,
    )
    db_session.add(ticker_crypto)
    db_session.commit()
    assert checker.check_freshness(now=now).status == "DEGRADED"

    bar = OhlcvBar(
        ticker="ETH",
        interval="1d",
        ts=(now - timedelta(minutes=4)).isoformat(),
        open=3000.0,
        high=3100.0,
        low=2950.0,
        close=3050.0,
        volume=50.0,
        source="fake",
        ingested_at=now.isoformat(),
    )
    db_session.add(bar)
    db_session.commit()
    assert checker.check_freshness(now=now).status == "NOMINAL"

    bar.ts = (now - timedelta(minutes=15)).isoformat()
    db_session.add(bar)
    db_session.commit()
    assert checker.check_freshness(now=now).status == "DEGRADED"

    bar.ts = (now - timedelta(minutes=90)).isoformat()
    db_session.add(bar)
    db_session.commit()
    assert checker.check_freshness(now=now).status == "CRITICAL"


def test_check_latency(db_session: Session) -> None:
    """Test RTT latency and jitter threshold evaluation."""
    checker = HealthChecker(db_session)
    assert checker.check_latency().status == "NOMINAL"

    metric = LinkMetrics(
        provider="yfinance",
        rtt_p50_ms=400.0,
        rtt_p95_ms=1200.0,
        rtt_jitter_ms=300.0,
        error_rate=0.0,
        consecutive_failures=0,
        breaker_state="closed",
        calls_today=50,
        daily_limit=2000,
        quota_pct=0.025,
        updated_at="2026-09-01T00:00:00Z",
    )
    db_session.add(metric)
    db_session.commit()
    assert checker.check_latency().status == "NOMINAL"

    metric.rtt_p50_ms = 850.0
    db_session.add(metric)
    db_session.commit()
    assert checker.check_latency().status == "DEGRADED"

    metric.rtt_p50_ms = 400.0
    metric.rtt_p95_ms = 2600.0
    db_session.add(metric)
    db_session.commit()
    assert checker.check_latency().status == "DEGRADED"

    metric.rtt_p95_ms = 1200.0
    metric.rtt_jitter_ms = 1600.0
    db_session.add(metric)
    db_session.commit()
    assert checker.check_latency().status == "DEGRADED"


def test_check_error_rate(db_session: Session) -> None:
    """Test consecutive provider failures (>=3 degraded, >=5 down)."""
    checker = HealthChecker(db_session)
    assert checker.check_error_rate().status == "NOMINAL"

    p1 = LinkMetrics(
        provider="p1",
        consecutive_failures=0,
        breaker_state="closed",
        daily_limit=1000,
        updated_at="2026-09-01T00:00:00Z",
    )
    p2 = LinkMetrics(
        provider="p2",
        consecutive_failures=0,
        breaker_state="closed",
        daily_limit=1000,
        updated_at="2026-09-01T00:00:00Z",
    )
    db_session.add(p1)
    db_session.add(p2)
    db_session.commit()
    assert checker.check_error_rate().status == "NOMINAL"

    p1.consecutive_failures = 3
    db_session.add(p1)
    db_session.commit()
    assert checker.check_error_rate().status == "DEGRADED"

    p1.consecutive_failures = 5
    p1.breaker_state = "open"
    db_session.add(p1)
    db_session.commit()
    assert checker.check_error_rate().status == "DEGRADED"

    p2.consecutive_failures = 5
    p2.breaker_state = "open"
    db_session.add(p2)
    db_session.commit()
    assert checker.check_error_rate().status == "CRITICAL"


def test_check_gaps_and_frozen_price(db_session: Session) -> None:
    """Test gap detection in OHLCV bar series and frozen prices."""
    ticker = Ticker(
        symbol="SOL",
        asset_class="crypto",
        display_name="Solana",
        provider="fake",
        provider_symbol="SOL",
        price_basis="raw",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    db_session.add(ticker)

    b1 = OhlcvBar(
        ticker="SOL",
        interval="1d",
        ts="2026-01-01T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10.0,
        source="fake",
        ingested_at="2026-01-01T00:00:00Z",
    )
    b2 = OhlcvBar(
        ticker="SOL",
        interval="1d",
        ts="2026-01-02T00:00:00Z",
        open=102.0,
        high=108.0,
        low=100.0,
        close=107.0,
        volume=12.0,
        source="fake",
        ingested_at="2026-01-02T00:00:00Z",
    )
    db_session.add(b1)
    db_session.add(b2)
    db_session.commit()

    checker = HealthChecker(db_session)
    assert checker.check_gaps().status == "NOMINAL"

    b3 = OhlcvBar(
        ticker="SOL",
        interval="1d",
        ts="2026-01-07T00:00:00Z",
        open=107.0,
        high=110.0,
        low=104.0,
        close=109.0,
        volume=15.0,
        source="fake",
        ingested_at="2026-01-07T00:00:00Z",
    )
    db_session.add(b3)
    db_session.commit()
    res_gap = checker.check_gaps()
    assert res_gap.status == "DEGRADED"
    assert "gap of 5.0d" in res_gap.message


def test_check_data_sanity(db_session: Session) -> None:
    """Test QuarantineBar row count in the last 24h."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    checker = HealthChecker(db_session)
    assert checker.check_data_sanity(now=now).status == "NOMINAL"

    q1 = QuarantineBar(
        ticker="AAPL",
        raw_json='{"price": -5}',
        reason="price_le_0",
        provider="yfinance",
        detected_at=(now - timedelta(hours=2)).isoformat(),
    )
    q2 = QuarantineBar(
        ticker="AAPL",
        raw_json='{"price": -10}',
        reason="price_le_0",
        provider="yfinance",
        detected_at=(now - timedelta(hours=30)).isoformat(),
    )
    db_session.add(q1)
    db_session.add(q2)
    db_session.commit()

    res_dirty = checker.check_data_sanity(now=now)
    assert res_dirty.status == "DEGRADED"
    assert res_dirty.details["count_24h"] == 1


def test_check_clock_future_skew(db_session: Session) -> None:
    """Test clock check detects bars with future timestamps."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    checker = HealthChecker(db_session)

    b1 = OhlcvBar(
        ticker="MSFT",
        interval="1d",
        ts="2026-09-01T11:00:00Z",
        open=400.0,
        high=405.0,
        low=398.0,
        close=402.0,
        volume=1000.0,
        source="fake",
        ingested_at="2026-09-01T11:00:00Z",
    )
    db_session.add(b1)
    db_session.commit()
    assert checker.check_clock(now=now).status == "NOMINAL"

    b2 = OhlcvBar(
        ticker="MSFT",
        interval="1d",
        ts="2026-09-01T12:10:00Z",
        open=402.0,
        high=406.0,
        low=401.0,
        close=405.0,
        volume=500.0,
        source="fake",
        ingested_at="2026-09-01T12:00:00Z",
    )
    db_session.add(b2)
    db_session.commit()

    res = checker.check_clock(now=now)
    assert res.status == "DEGRADED"
    assert res.details["count"] == 1


def test_check_quota_thresholds(db_session: Session) -> None:
    """Test daily quota consumption thresholds (>=80% degraded, >=95% critical)."""
    checker = HealthChecker(db_session)
    assert checker.check_quota().status == "NOMINAL"

    metric = LinkMetrics(
        provider="finnhub",
        calls_today=750,
        daily_limit=1000,
        quota_pct=0.75,
        updated_at="2026-09-01T00:00:00Z",
    )
    db_session.add(metric)
    db_session.commit()
    assert checker.check_quota().status == "NOMINAL"

    metric.calls_today = 850
    metric.quota_pct = 0.85
    db_session.add(metric)
    db_session.commit()
    assert checker.check_quota().status == "DEGRADED"

    metric.calls_today = 980
    metric.quota_pct = 0.98
    db_session.add(metric)
    db_session.commit()
    assert checker.check_quota().status == "CRITICAL"


def test_compute_system_status_db_failure() -> None:
    """Test compute_system_status catches DB exceptions and reports CRITICAL."""
    mock_session = MagicMock()
    mock_session.exec.side_effect = Exception("SQLite disk I/O error")

    checker = HealthChecker(mock_session)
    status, reasons = checker.compute_system_status()
    assert status == "CRITICAL"
    assert any("SQLite disk I/O error" in r for r in reasons)
