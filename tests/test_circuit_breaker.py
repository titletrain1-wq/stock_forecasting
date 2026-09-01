"""Unit tests for CircuitBreaker state transitions and guard context manager."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from stock_forecasting.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenException,
)
from stock_forecasting.schema import LinkMetrics


def test_circuit_breaker_transitions(db_session: Session) -> None:
    """Test closed -> open on 5 failures -> half_open after cooldown -> closed on trial success."""
    cb = CircuitBreaker(session=db_session, failure_threshold=5, cooldown_minutes=15)
    provider = "tiingo"

    # Initial state should be closed
    assert cb.check_state(provider) == "closed"

    # Record 4 failures -> still closed
    for _ in range(4):
        cb.record_failure(provider)
        assert cb.check_state(provider) == "closed"

    # 5th failure -> transitions to open
    cb.record_failure(provider)
    assert cb.check_state(provider) == "open"

    metric = db_session.exec(
        select(LinkMetrics).where(LinkMetrics.provider == provider)
    ).first()
    assert metric is not None
    assert metric.breaker_state == "open"
    assert metric.consecutive_failures == 5
    assert metric.breaker_opened_at is not None

    # Before cooldown expires -> still open
    assert cb.check_state(provider) == "open"

    # Simulate cooldown expiration by setting breaker_opened_at to 20 minutes ago
    metric.breaker_opened_at = (datetime.now(UTC) - timedelta(minutes=20)).isoformat()
    db_session.add(metric)
    db_session.commit()

    # check_state should now auto-transition to half_open
    assert cb.check_state(provider) == "half_open"
    db_session.refresh(metric)
    assert metric.breaker_state == "half_open"

    # Successful trial call -> transitions back to closed
    cb.record_success(provider)
    assert cb.check_state(provider) == "closed"
    db_session.refresh(metric)
    assert metric.breaker_state == "closed"
    assert metric.consecutive_failures == 0
    assert metric.breaker_opened_at is None


def test_circuit_breaker_half_open_failure_reopens(db_session: Session) -> None:
    """Test failure during half_open transitions back to open."""
    cb = CircuitBreaker(session=db_session, failure_threshold=5, cooldown_minutes=15)
    provider = "finnhub"

    # Trigger open
    for _ in range(5):
        cb.record_failure(provider)
    assert cb.check_state(provider) == "open"

    # Set opened_at to past
    metric = db_session.exec(
        select(LinkMetrics).where(LinkMetrics.provider == provider)
    ).first()
    assert metric is not None
    metric.breaker_opened_at = (datetime.now(UTC) - timedelta(minutes=16)).isoformat()
    db_session.add(metric)
    db_session.commit()

    # check_state -> half_open
    assert cb.check_state(provider) == "half_open"

    # Failed trial call in half_open -> reopens
    cb.record_failure(provider)
    assert cb.check_state(provider) == "open"
    db_session.refresh(metric)
    assert metric.breaker_state == "open"


def test_circuit_breaker_guard_open_raises(db_session: Session) -> None:
    """Test guard() raising CircuitBreakerOpenException when open."""
    cb = CircuitBreaker(session=db_session, failure_threshold=3, cooldown_minutes=10)
    provider = "coinbase"

    # Closed initially: guard allows execution and records success
    executed = False
    with cb.guard(provider):
        executed = True
    assert executed

    # Guard handles failure and re-raises original exception
    for _ in range(3):
        with (
            pytest.raises(ConnectionError, match="Network down"),
            cb.guard(provider),
        ):
            raise ConnectionError("Network down")

    assert cb.check_state(provider) == "open"

    # Once open, guard immediately raises CircuitBreakerOpenException without executing body
    body_executed = False
    with (
        pytest.raises(
            CircuitBreakerOpenException,
            match=r"Circuit breaker for coinbase is OPEN",
        ),
        cb.guard(provider),
    ):
        body_executed = True

    assert not body_executed
