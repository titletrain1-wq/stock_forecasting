"""Unit tests for LinkMonitor provider health, RTT metrics, and quota tracking."""

import time

import pytest
from sqlmodel import Session, select

from stock_forecasting.link_monitor import LinkMonitor
from stock_forecasting.schema import LinkMetrics


def test_link_monitor_instrument_success(db_session: Session) -> None:
    """Verifies timing RTT, updating calls_today, quota_pct, and RTT stats in LinkMetrics DB table."""
    monitor = LinkMonitor(session=db_session, default_limits={"fake": 100})

    # Execute an instrumented call block with a small sleep
    with monitor.instrument("fake"):
        time.sleep(0.01)

    # Verify LinkMetrics row in database
    metric = db_session.exec(
        select(LinkMetrics).where(LinkMetrics.provider == "fake")
    ).first()
    assert metric is not None
    assert metric.provider == "fake"
    assert metric.calls_today == 1
    assert metric.daily_limit == 100
    assert metric.quota_pct == 0.01
    assert metric.error_rate == 0.0
    assert metric.consecutive_failures == 0
    assert metric.rtt_p50_ms is not None and metric.rtt_p50_ms >= 5.0
    assert metric.rtt_p95_ms is not None and metric.rtt_p95_ms >= 5.0
    assert metric.rtt_jitter_ms == 0.0  # 1 sample -> jitter 0.0
    assert metric.updated_at is not None

    # Perform a second successful call
    with monitor.instrument("fake"):
        time.sleep(0.02)

    db_session.refresh(metric)
    assert metric.calls_today == 2
    assert metric.quota_pct == 0.02
    assert metric.error_rate == 0.0
    assert metric.consecutive_failures == 0
    assert metric.rtt_p50_ms is not None
    assert metric.rtt_p95_ms >= metric.rtt_p50_ms
    assert metric.rtt_jitter_ms >= 0.0


def test_link_monitor_instrument_error(db_session: Session) -> None:
    """Verifies tracking exception, updating error_rate and consecutive_failures."""
    monitor = LinkMonitor(session=db_session, default_limits={"yfinance": 2000})

    # Call should raise exception and record error
    with (
        pytest.raises(RuntimeError, match="API timeout or 500 error"),
        monitor.instrument("yfinance"),
    ):
        raise RuntimeError("API timeout or 500 error")

    metric = monitor.get_metrics("yfinance")
    assert metric is not None
    assert metric.calls_today == 1
    assert metric.consecutive_failures == 1
    assert metric.error_rate == 1.0

    # Second failure increments consecutive_failures
    with pytest.raises(ValueError, match="Bad payload"), monitor.instrument("yfinance"):
        raise ValueError("Bad payload")

    db_session.refresh(metric)
    assert metric.calls_today == 2
    assert metric.consecutive_failures == 2
    assert metric.error_rate == 1.0


def test_link_monitor_recovery_resets_consecutive_failures(
    db_session: Session,
) -> None:
    """Verifies a successful call following errors resets consecutive_failures to 0."""
    monitor = LinkMonitor(session=db_session, default_limits={"fake": 500})

    # 2 failures
    for _ in range(2):
        with (
            pytest.raises(ConnectionError, match="connection lost"),
            monitor.instrument("fake"),
        ):
            raise ConnectionError("connection lost")

    metric = monitor.get_metrics("fake")
    assert metric is not None
    assert metric.consecutive_failures == 2
    assert metric.error_rate == 1.0

    # 1 success
    with monitor.instrument("fake"):
        pass

    db_session.refresh(metric)
    assert metric.consecutive_failures == 0
    # 2 failures out of 3 calls -> error_rate ~ 0.6667
    assert metric.error_rate == round(2 / 3, 4)
    assert metric.calls_today == 3


def test_link_monitor_daily_rollover(db_session: Session) -> None:
    """Verifies that calls_today resets to 1 when a new calendar day is encountered."""
    # Pre-populate metric with yesterday's timestamp and 50 calls
    yesterday_metric = LinkMetrics(
        provider="tiingo",
        daily_limit=1000,
        breaker_state="closed",
        calls_today=50,
        quota_pct=0.05,
        error_rate=0.0,
        consecutive_failures=0,
        updated_at="2020-01-01T00:00:00Z",
    )
    db_session.add(yesterday_metric)
    db_session.commit()

    monitor = LinkMonitor(session=db_session)
    with monitor.instrument("tiingo"):
        pass

    metric = monitor.get_metrics("tiingo")
    assert metric is not None
    assert metric.calls_today == 1
    assert metric.quota_pct == 0.001
