"""LinkMonitor for measuring API provider latency, health, and quota usage."""

import time
from collections import defaultdict, deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
from sqlmodel import Session

from stock_forecasting.schema import LinkMetrics

DEFAULT_DAILY_LIMITS: dict[str, int] = {
    "yfinance": 2000,
    "tiingo": 1000,
    "finnhub": 1000,
    "coingecko": 10000,
    "coinbase": 10000,
    "dydx": 10000,
    "fake": 10000,
}
DEFAULT_DAILY_LIMIT: int = 2000
DEFAULT_WINDOW_SIZE: int = 20


class LinkMonitor:
    """Monitors external API provider health, timing RTT, errors, and quota."""

    def __init__(
        self,
        session: Session,
        window_size: int = DEFAULT_WINDOW_SIZE,
        default_limits: dict[str, int] | None = None,
    ) -> None:
        """Initialize LinkMonitor with a DB session and rolling window size.

        Args:
            session: Active SQLModel session for persisting LinkMetrics.
            window_size: Rolling window sample size (default 20).
            default_limits: Optional map of provider name to daily call limit.
        """
        self.session = session
        self.window_size = window_size
        self.default_limits = default_limits or DEFAULT_DAILY_LIMITS
        self._rtt_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._call_history: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    @contextmanager
    def instrument(self, provider: str) -> Generator[None, None, None]:
        """Context manager measuring RTT and recording call outcome for a provider.

        Args:
            provider: Identifier of the API provider (e.g. "yfinance", "tiingo").

        Yields:
            None
        """
        start = time.perf_counter()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record_call(provider=provider, rtt_ms=elapsed_ms, success=success)

    def _record_call(self, provider: str, rtt_ms: float, success: bool) -> LinkMetrics:
        """Record an API call measurement and update the LinkMetrics table.

        Args:
            provider: Name of the external data provider.
            rtt_ms: Round-trip time in milliseconds.
            success: Whether the call succeeded or raised an exception.

        Returns:
            The updated LinkMetrics model instance.
        """
        now_iso = datetime.now(UTC).isoformat()
        today_str = now_iso[:10]

        metric = self.session.get(LinkMetrics, provider)
        if metric is None:
            daily_limit = self.default_limits.get(provider.lower(), DEFAULT_DAILY_LIMIT)
            metric = LinkMetrics(
                provider=provider,
                daily_limit=daily_limit,
                breaker_state="closed",
                breaker_opened_at=None,
                calls_today=0,
                quota_pct=0.0,
                error_rate=0.0,
                consecutive_failures=0,
                updated_at=now_iso,
            )

        # Check if calendar day rolled over to reset calls_today
        if (
            metric.updated_at
            and len(metric.updated_at) >= 10
            and metric.updated_at[:10] != today_str
        ):
            metric.calls_today = 1
        else:
            metric.calls_today = (metric.calls_today or 0) + 1

        if metric.daily_limit and metric.daily_limit > 0:
            metric.quota_pct = round(metric.calls_today / metric.daily_limit, 4)
        else:
            metric.quota_pct = 0.0

        # Update rolling histories
        self._rtt_history[provider].append(rtt_ms)
        self._call_history[provider].append(success)

        samples = list(self._rtt_history[provider])
        metric.rtt_p50_ms = round(float(np.percentile(samples, 50)), 2)
        metric.rtt_p95_ms = round(float(np.percentile(samples, 95)), 2)
        metric.rtt_jitter_ms = (
            round(float(np.std(samples)), 2) if len(samples) > 1 else 0.0
        )

        history = list(self._call_history[provider])
        errors = sum(1 for s in history if not s)
        metric.error_rate = round(errors / len(history), 4) if history else 0.0

        if success:
            metric.consecutive_failures = 0
        else:
            metric.consecutive_failures = (metric.consecutive_failures or 0) + 1

        metric.updated_at = now_iso

        self.session.add(metric)
        self.session.commit()
        self.session.refresh(metric)
        return metric

    def get_metrics(self, provider: str) -> LinkMetrics | None:
        """Fetch current LinkMetrics for a provider from the database."""
        return self.session.get(LinkMetrics, provider)
