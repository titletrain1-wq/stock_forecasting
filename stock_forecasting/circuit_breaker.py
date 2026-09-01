"""Circuit breaker state machine for external data provider fault tolerance."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from sqlmodel import Session

from stock_forecasting.schema import LinkMetrics

DEFAULT_DAILY_LIMIT: int = 2000


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the provider circuit breaker is OPEN."""


class CircuitBreaker:
    """Circuit breaker state machine (closed -> open -> half_open -> closed)."""

    def __init__(
        self,
        session: Session,
        failure_threshold: int = 5,
        cooldown_minutes: int = 15,
    ) -> None:
        """Initialize CircuitBreaker.

        Args:
            session: Active SQLModel session for querying and persisting LinkMetrics.
            failure_threshold: Number of consecutive failures before opening circuit (default: 5).
            cooldown_minutes: Minutes circuit stays open before transitioning to half_open (default: 15).
        """
        self.session = session
        self.failure_threshold = failure_threshold
        self.cooldown_minutes = cooldown_minutes

    def _get_or_create_metric(self, provider: str) -> LinkMetrics:
        """Fetch LinkMetrics record for provider, creating default record if missing."""
        metric = self.session.get(LinkMetrics, provider)
        if metric is None:
            now_iso = datetime.now(UTC).isoformat()
            metric = LinkMetrics(
                provider=provider,
                daily_limit=DEFAULT_DAILY_LIMIT,
                breaker_state="closed",
                breaker_opened_at=None,
                calls_today=0,
                quota_pct=0.0,
                error_rate=0.0,
                consecutive_failures=0,
                updated_at=now_iso,
            )
            self.session.add(metric)
            self.session.commit()
            self.session.refresh(metric)
        return metric

    def check_state(self, provider: str) -> str:
        """Check current state from LinkMetrics.

        If state is 'open' and opened_at >= cooldown_minutes ago, auto-transitions to 'half_open'.
        Returns state ('closed', 'open', 'half_open').
        """
        metric = self._get_or_create_metric(provider)
        if metric.breaker_state == "open" and metric.breaker_opened_at:
            opened_at_dt = datetime.fromisoformat(metric.breaker_opened_at)
            if opened_at_dt.tzinfo is None:
                opened_at_dt = opened_at_dt.replace(tzinfo=UTC)
            now_dt = datetime.now(UTC)
            if now_dt - opened_at_dt >= timedelta(minutes=self.cooldown_minutes):
                metric.breaker_state = "half_open"
                metric.updated_at = now_dt.isoformat()
                self.session.add(metric)
                self.session.commit()
                self.session.refresh(metric)

        return metric.breaker_state

    def record_failure(self, provider: str) -> None:
        """Record a failure for the provider.

        Increments consecutive_failures.
        If consecutive_failures >= failure_threshold and state is 'closed',
        sets state to 'open' and sets breaker_opened_at to current UTC timestamp.
        If state was 'half_open', transitions back to 'open' and updates breaker_opened_at.
        """
        metric = self._get_or_create_metric(provider)
        metric.consecutive_failures = (metric.consecutive_failures or 0) + 1
        now_iso = datetime.now(UTC).isoformat()
        metric.updated_at = now_iso

        if metric.breaker_state == "half_open":
            metric.breaker_state = "open"
            metric.breaker_opened_at = now_iso
        elif metric.breaker_state == "closed":
            if metric.consecutive_failures >= self.failure_threshold:
                metric.breaker_state = "open"
                metric.breaker_opened_at = now_iso
        elif metric.breaker_state == "open":
            if not metric.breaker_opened_at:
                metric.breaker_opened_at = now_iso

        self.session.add(metric)
        self.session.commit()
        self.session.refresh(metric)

    def record_success(self, provider: str) -> None:
        """Record a success for the provider.

        Resets consecutive_failures = 0.
        If state was 'half_open' or 'open', sets state to 'closed' and clears breaker_opened_at.
        """
        metric = self._get_or_create_metric(provider)
        metric.consecutive_failures = 0
        now_iso = datetime.now(UTC).isoformat()
        metric.updated_at = now_iso

        if metric.breaker_state in ("half_open", "open"):
            metric.breaker_state = "closed"
            metric.breaker_opened_at = None

        self.session.add(metric)
        self.session.commit()
        self.session.refresh(metric)

    @contextmanager
    def guard(self, provider: str) -> Generator[None, None, None]:
        """Context manager guarding provider calls.

        Calls check_state(provider).
        If 'open', raises CircuitBreakerOpenException(f"Circuit breaker for {provider} is OPEN").
        On exception during execution, calls record_failure(provider).
        On success, calls record_success(provider).
        """
        state = self.check_state(provider)
        if state == "open":
            raise CircuitBreakerOpenException(f"Circuit breaker for {provider} is OPEN")

        try:
            yield
        except Exception:
            self.record_failure(provider)
            raise
        else:
            self.record_success(provider)
