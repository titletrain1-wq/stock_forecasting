"""Data-feed and system health monitoring service.

Implements the 8 health checks defined in Spec §6:
1. Freshness (bar age vs expected)
2. Latency / Jitter (RTT metrics)
3. Error Rate (consecutive provider failures)
4. Gap / Frame Sync (missing / duplicate / frozen bars)
5. Data Sanity / SNR (quarantined bars in last 24h)
6. Watchdog (worker heartbeat lag)
7. Clock / Skew (future-dated bars)
8. Quota Budget (daily API call consumption)
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from stock_forecasting.schema import (
    LinkMetrics,
    OhlcvBar,
    QuarantineBar,
    SystemHeartbeat,
    Ticker,
)


def _parse_utc(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class HealthCheckResult:
    """Outcome and diagnostics of a single system health check."""

    check_name: str
    status: str  # "NOMINAL", "DEGRADED", or "CRITICAL"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Evaluates data-feed health, provider metrics, and scheduler watchdog pulse."""

    def __init__(self, session: Session) -> None:
        """Initialize HealthChecker with a database session."""
        self.session = session

    def check_freshness(self, now: datetime | None = None) -> HealthCheckResult:
        """Check market bar age against expected thresholds (crypto <5m, equity <20m)."""
        current_time = now or datetime.now(UTC)
        tickers = self.session.exec(select(Ticker).where(Ticker.active == 1)).all()

        if not tickers:
            return HealthCheckResult(
                check_name="freshness",
                status="NOMINAL",
                message="No active tickers configured.",
            )

        details: dict[str, Any] = {}
        warnings: list[str] = []
        has_critical = False
        has_degraded = False

        for ticker in tickers:
            latest_bar = self.session.exec(
                select(OhlcvBar)
                .where(OhlcvBar.ticker == ticker.symbol)
                .order_by(OhlcvBar.ts.desc())
                .limit(1)
            ).first()

            if latest_bar is None:
                has_degraded = True
                warnings.append(f"{ticker.symbol}: no bar data ingested")
                details[ticker.symbol] = {"status": "DEGRADED", "reason": "no_data"}
                continue

            bar_dt = _parse_utc(latest_bar.ts)
            age_sec = (current_time - bar_dt).total_seconds()
            age_min = age_sec / 60.0
            is_crypto = ticker.asset_class.lower() == "crypto"
            nominal_limit_sec = 300.0 if is_crypto else 1200.0

            if age_sec <= nominal_limit_sec:
                ticker_status = "NOMINAL"
            elif age_sec <= 3600.0:
                ticker_status = "DEGRADED"
                has_degraded = True
                warnings.append(
                    f"{ticker.symbol} ({ticker.asset_class}) stale by {age_min:.1f}m"
                )
            else:
                ticker_status = "CRITICAL"
                has_critical = True
                warnings.append(
                    f"{ticker.symbol} ({ticker.asset_class}) severely stale by {age_min:.1f}m"
                )

            details[ticker.symbol] = {
                "status": ticker_status,
                "asset_class": ticker.asset_class,
                "latest_ts": latest_bar.ts,
                "age_seconds": age_sec,
            }

        if has_critical:
            status = "CRITICAL"
            msg = f"Stale market bars detected: {'; '.join(warnings)}"
        elif has_degraded:
            status = "DEGRADED"
            msg = f"Data freshness degraded: {'; '.join(warnings)}"
        else:
            status = "NOMINAL"
            msg = "All active ticker market bars are fresh."

        return HealthCheckResult("freshness", status, msg, details)

    def check_latency(self) -> HealthCheckResult:
        """Check RTT latency metrics (p50 <800ms, p95 <2.5s, jitter <1.5s)."""
        metrics = self.session.exec(select(LinkMetrics)).all()
        if not metrics:
            return HealthCheckResult(
                "latency", "NOMINAL", "No link latency metrics recorded."
            )

        details: dict[str, Any] = {}
        warnings: list[str] = []
        is_degraded = False

        for metric in metrics:
            p50_high = metric.rtt_p50_ms is not None and metric.rtt_p50_ms >= 800.0
            p95_high = metric.rtt_p95_ms is not None and metric.rtt_p95_ms >= 2500.0
            jitter_high = (
                metric.rtt_jitter_ms is not None and metric.rtt_jitter_ms >= 1500.0
            )

            provider_status = "NOMINAL"
            if p50_high or p95_high or jitter_high:
                provider_status = "DEGRADED"
                is_degraded = True
                warnings.append(
                    f"{metric.provider} high latency: p50={metric.rtt_p50_ms}ms, "
                    f"p95={metric.rtt_p95_ms}ms, jitter={metric.rtt_jitter_ms}ms"
                )

            details[metric.provider] = {
                "status": provider_status,
                "p50_ms": metric.rtt_p50_ms,
                "p95_ms": metric.rtt_p95_ms,
                "jitter_ms": metric.rtt_jitter_ms,
            }

        status = "DEGRADED" if is_degraded else "NOMINAL"
        msg = (
            f"Latency thresholds exceeded: {'; '.join(warnings)}"
            if is_degraded
            else "Provider latency and jitter within nominal limits."
        )
        return HealthCheckResult("latency", status, msg, details)

    def check_error_rate(self) -> HealthCheckResult:
        """Check provider consecutive failures (>=3 degraded, >=5 down)."""
        metrics = self.session.exec(select(LinkMetrics)).all()
        if not metrics:
            return HealthCheckResult(
                "error_rate", "NOMINAL", "No provider link metrics found."
            )

        details: dict[str, Any] = {}
        down_providers: list[str] = []
        degraded_providers: list[str] = []

        for metric in metrics:
            failures = metric.consecutive_failures or 0
            is_down = failures >= 5 or metric.breaker_state == "open"
            is_degraded = failures >= 3 or (metric.error_rate or 0.0) >= 0.2

            if is_down:
                p_status = "CRITICAL"
                down_providers.append(
                    f"{metric.provider} (failures={failures}, breaker={metric.breaker_state})"
                )
            elif is_degraded:
                p_status = "DEGRADED"
                degraded_providers.append(
                    f"{metric.provider} (failures={failures}, error_rate={metric.error_rate:.1%})"
                )
            else:
                p_status = "NOMINAL"

            details[metric.provider] = {
                "status": p_status,
                "consecutive_failures": failures,
                "error_rate": metric.error_rate,
                "breaker_state": metric.breaker_state,
            }

        if len(down_providers) == len(metrics) and len(metrics) > 0:
            status = "CRITICAL"
            msg = f"All primary providers down: {'; '.join(down_providers)}"
        elif down_providers or degraded_providers:
            status = "DEGRADED"
            msg = f"Provider errors detected: {'; '.join(down_providers + degraded_providers)}"
        else:
            status = "NOMINAL"
            msg = "All external providers responding normally."

        return HealthCheckResult("error_rate", status, msg, details)

    def check_gaps(self) -> HealthCheckResult:
        """Check for missing consecutive bars, duplicates, or frozen prices."""
        tickers = self.session.exec(select(Ticker).where(Ticker.active == 1)).all()
        details: dict[str, Any] = {}
        warnings: list[str] = []
        is_degraded = False

        for ticker in tickers:
            bars = self.session.exec(
                select(OhlcvBar)
                .where(OhlcvBar.ticker == ticker.symbol)
                .order_by(OhlcvBar.ts.asc())
            ).all()

            if len(bars) < 2:
                continue

            max_allowed_gap_days = 2 if ticker.asset_class.lower() == "crypto" else 4
            consecutive_equal_close = 1
            ticker_gaps: list[str] = []

            for i in range(1, len(bars)):
                prev_bar, curr_bar = bars[i - 1], bars[i]
                prev_dt = _parse_utc(prev_bar.ts)
                curr_dt = _parse_utc(curr_bar.ts)
                delta_days = (curr_dt - prev_dt).total_seconds() / 86400.0

                if delta_days > max_allowed_gap_days:
                    ticker_gaps.append(
                        f"gap of {delta_days:.1f}d between {prev_bar.ts} and {curr_bar.ts}"
                    )

                if curr_bar.close == prev_bar.close:
                    consecutive_equal_close += 1
                else:
                    consecutive_equal_close = 1

                if consecutive_equal_close >= 6:
                    ticker_gaps.append(
                        f"frozen price detected (>= 6 bars identical at {curr_bar.close})"
                    )

            if ticker_gaps:
                is_degraded = True
                warnings.append(f"{ticker.symbol}: {'; '.join(ticker_gaps)}")
                details[ticker.symbol] = {"status": "DEGRADED", "gaps": ticker_gaps}
            else:
                details[ticker.symbol] = {"status": "NOMINAL", "bar_count": len(bars)}

        status = "DEGRADED" if is_degraded else "NOMINAL"
        msg = (
            f"Bar gaps or frozen prices detected: {'; '.join(warnings)}"
            if is_degraded
            else "No bar gaps or frozen prices detected."
        )
        return HealthCheckResult("gaps", status, msg, details)

    def check_data_sanity(self, now: datetime | None = None) -> HealthCheckResult:
        """Check count of QuarantineBar records inserted in the last 24 hours."""
        current_time = now or datetime.now(UTC)
        cutoff_dt = current_time - timedelta(hours=24)
        quarantined = self.session.exec(select(QuarantineBar)).all()

        recent_quarantined = [
            q for q in quarantined if _parse_utc(q.detected_at) >= cutoff_dt
        ]
        count = len(recent_quarantined)

        if count > 0:
            reasons = {q.reason for q in recent_quarantined}
            return HealthCheckResult(
                check_name="data_sanity",
                status="DEGRADED",
                message=f"{count} quarantined bar(s) detected in last 24h (reasons: {', '.join(reasons)}).",
                details={"count_24h": count, "reasons": list(reasons)},
            )

        return HealthCheckResult(
            check_name="data_sanity",
            status="NOMINAL",
            message="0 quarantined rows in last 24h.",
            details={"count_24h": 0},
        )

    def check_watchdog(self, now: datetime | None = None) -> HealthCheckResult:
        """Check SystemHeartbeat worker pulse lag (>2x job interval, 5m/15m thresholds)."""
        current_time = now or datetime.now(UTC)
        heartbeats = self.session.exec(select(SystemHeartbeat)).all()

        if not heartbeats:
            return HealthCheckResult(
                check_name="watchdog",
                status="CRITICAL",
                message="No worker heartbeat records found; background worker is down.",
            )

        valid_pulses = [
            _parse_utc(hb.last_pulse_ts)
            for hb in heartbeats
            if hb.last_pulse_ts is not None
        ]
        if not valid_pulses:
            return HealthCheckResult(
                check_name="watchdog",
                status="CRITICAL",
                message="Worker heartbeat timestamps missing.",
            )

        latest_pulse = max(valid_pulses)
        lag_sec = (current_time - latest_pulse).total_seconds()
        lag_min = lag_sec / 60.0

        details: dict[str, Any] = {
            "latest_pulse_ts": latest_pulse.isoformat(),
            "lag_seconds": lag_sec,
            "lag_minutes": lag_min,
        }

        if lag_sec > 900.0:
            status = "CRITICAL"
            msg = f"Worker heartbeat lag is {lag_min:.1f}m (> 15m); worker dead."
        elif lag_sec > 300.0:
            status = "DEGRADED"
            msg = f"Worker heartbeat lag is {lag_min:.1f}m (5-15m); worker degraded."
        else:
            status = "NOMINAL"
            msg = f"Worker heartbeat healthy (lag {lag_sec:.0f}s < 5m)."

        return HealthCheckResult("watchdog", status, msg, details)

    def check_clock(self, now: datetime | None = None) -> HealthCheckResult:
        """Check for future-dated bars (clock drift > 2 minutes)."""
        current_time = now or datetime.now(UTC)
        future_cutoff = current_time + timedelta(minutes=2)
        bars = self.session.exec(select(OhlcvBar)).all()

        future_bars = [
            (bar.ticker, bar.ts)
            for bar in bars
            if _parse_utc(bar.ts) > future_cutoff
        ]

        if future_bars:
            return HealthCheckResult(
                check_name="clock",
                status="DEGRADED",
                message=f"Detected {len(future_bars)} future-dated bar(s) with clock skew > 2m.",
                details={"future_bars": future_bars, "count": len(future_bars)},
            )

        return HealthCheckResult(
            check_name="clock",
            status="NOMINAL",
            message="No future-dated bars detected.",
        )

    def check_quota(self) -> HealthCheckResult:
        """Check API provider daily quota usage (>=0.8 degraded, >=0.95 critical)."""
        metrics = self.session.exec(select(LinkMetrics)).all()
        if not metrics:
            return HealthCheckResult(
                "quota", "NOMINAL", "No provider quota metrics recorded."
            )

        details: dict[str, Any] = {}
        critical_warnings: list[str] = []
        degraded_warnings: list[str] = []

        for metric in metrics:
            quota_pct = metric.quota_pct or 0.0
            if metric.daily_limit and metric.daily_limit > 0:
                quota_pct = max(quota_pct, metric.calls_today / metric.daily_limit)

            details[metric.provider] = {
                "calls_today": metric.calls_today,
                "daily_limit": metric.daily_limit,
                "quota_pct": quota_pct,
            }

            if quota_pct >= 0.95:
                critical_warnings.append(
                    f"{metric.provider} quota critical ({quota_pct:.1%} >= 95%)"
                )
            elif quota_pct >= 0.80:
                degraded_warnings.append(
                    f"{metric.provider} quota degraded ({quota_pct:.1%} >= 80%)"
                )

        if critical_warnings:
            status, msg = "CRITICAL", f"Daily API quota critical: {'; '.join(critical_warnings)}"
        elif degraded_warnings:
            status, msg = "DEGRADED", f"Daily API quota degraded: {'; '.join(degraded_warnings)}"
        else:
            status, msg = "NOMINAL", "All provider daily quotas within normal limits (< 80%)."

        return HealthCheckResult("quota", status, msg, details)

    def compute_system_status(
        self, now: datetime | None = None
    ) -> tuple[str, list[str]]:
        """Compute the overarching system health status from all 8 health checks.

        Status levels according to specification:
        - NOMINAL: All providers responding, worker heartbeat < 5m, 0 quarantined rows in 24h.
        - DEGRADED: 1 provider degraded/down, worker lag 5-15m, or quota > 80%.
        - CRITICAL: All primary providers down, worker lag > 15m, or DB failure.

        Args:
            now: Optional UTC datetime override (defaults to current time).

        Returns:
            Tuple of (status_string, list_of_warning_messages).
        """
        try:
            checks = [
                self.check_watchdog(now=now),
                self.check_error_rate(),
                self.check_quota(),
                self.check_data_sanity(now=now),
                self.check_freshness(now=now),
                self.check_latency(),
                self.check_gaps(),
                self.check_clock(now=now),
            ]
        except Exception as exc:  # noqa: BLE001
            return ("CRITICAL", [f"Database or system failure: {exc}"])

        warnings: list[str] = []
        is_critical = False
        is_degraded = False

        for res in checks:
            if res.status == "CRITICAL":
                is_critical = True
                warnings.append(f"[{res.check_name.upper()}] {res.message}")
            elif res.status == "DEGRADED":
                is_degraded = True
                warnings.append(f"[{res.check_name.upper()}] {res.message}")

        if is_critical:
            return ("CRITICAL", warnings)
        if is_degraded:
            return ("DEGRADED", warnings)
        return ("NOMINAL", [])
