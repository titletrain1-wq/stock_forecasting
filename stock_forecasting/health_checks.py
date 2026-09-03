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

from stock_forecasting.market_calendar import classify_bar_freshness
from stock_forecasting.schema import (
    IntradayBar,
    LinkMetrics,
    LiveQuote,
    OhlcvBar,
    QuarantineBar,
    SystemHeartbeat,
    Ticker,
)

# Map job types to their associated providers
# Based on worker.py job definitions
JOB_TYPE_TO_PROVIDERS = {
    "job_ingest_crypto": {"CoinGecko", "Coinbase", "dYdX", "fake"},
    "job_ingest_equities": {"yfinance", "Tiingo", "Finnhub", "fake"},
    "job_ingest_equity_intraday": {"Coinbase"},
    "job_ingest_derivatives": {"dYdX"},
}


def _parse_utc(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _get_active_providers(session: Session, active_threshold_min: int = 60) -> set[str]:
    """Get the set of currently-active providers based on recent job heartbeats.

    A provider is considered active if it's used by a job that has pulsed within
    the last `active_threshold_min` minutes. This filters out stale providers
    that are no longer being polled.

    Args:
        session: Database session
        active_threshold_min: Threshold in minutes for considering a job "active"

    Returns:
        Set of provider names that are actively used by recent jobs
    """
    now = datetime.now(UTC)
    threshold = now - timedelta(minutes=active_threshold_min)

    # Get all job heartbeats
    heartbeats = session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.last_pulse_ts.isnot(None))
    ).all()

    active_providers: set[str] = set()
    for heartbeat in heartbeats:
        if heartbeat.last_pulse_ts is None:
            continue
        pulse_time = _parse_utc(heartbeat.last_pulse_ts)
        if pulse_time >= threshold:
            job_type = heartbeat.job_type
            if job_type in JOB_TYPE_TO_PROVIDERS:
                active_providers.update(JOB_TYPE_TO_PROVIDERS[job_type])

    return active_providers


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
        """Check each ticker's latest bar against its expected daily schedule.

        This is an end-of-day system (daily bars), so freshness is judged against
        the trading calendar -- NYSE sessions for equities, one bar per UTC day
        for crypto -- not against wall-clock minutes. A bar is CRITICAL only when
        it is genuinely overdue: >=2 missed NYSE sessions, or >=3 days behind for
        crypto. See ``market_calendar.classify_bar_freshness``.
        """
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
                .where(
                    OhlcvBar.ticker == ticker.symbol,
                    OhlcvBar.interval == "1d",
                )
                .order_by(OhlcvBar.ts.desc())
                .limit(1)
            ).first()

            latest_ts = latest_bar.ts if latest_bar is not None else None
            ticker_status, fresh_details = classify_bar_freshness(
                ticker.asset_class, latest_ts, current_time
            )

            if ticker_status == "CRITICAL":
                has_critical = True
                warnings.append(
                    f"{ticker.symbol} ({ticker.asset_class}) bar overdue: {fresh_details}"
                )
            elif ticker_status == "DEGRADED":
                has_degraded = True
                reason = fresh_details.get("reason", "behind schedule")
                warnings.append(
                    f"{ticker.symbol} ({ticker.asset_class}) {reason}: {fresh_details}"
                )

            details[ticker.symbol] = {
                "status": ticker_status,
                "asset_class": ticker.asset_class,
                "latest_ts": latest_ts,
                **fresh_details,
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
        all_metrics = self.session.exec(select(LinkMetrics)).all()
        active_providers = _get_active_providers(self.session)

        # If no active providers detected, use all metrics (fallback for setup/testing)
        if active_providers:
            metrics = [m for m in all_metrics if m.provider in active_providers]
        else:
            metrics = all_metrics

        if not metrics:
            return HealthCheckResult(
                "latency", "NOMINAL", "No active provider latency metrics."
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
        all_metrics = self.session.exec(select(LinkMetrics)).all()
        active_providers = _get_active_providers(self.session)

        # If no active providers detected, use all metrics (fallback for setup/testing)
        if active_providers:
            metrics = [m for m in all_metrics if m.provider in active_providers]
        else:
            metrics = all_metrics

        if not metrics:
            return HealthCheckResult(
                "error_rate", "NOMINAL", "No active provider link metrics found."
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
            (bar.ticker, bar.ts) for bar in bars if _parse_utc(bar.ts) > future_cutoff
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
        all_metrics = self.session.exec(select(LinkMetrics)).all()
        active_providers = _get_active_providers(self.session)

        # If no active providers detected, use all metrics (fallback for setup/testing)
        if active_providers:
            metrics = [m for m in all_metrics if m.provider in active_providers]
        else:
            metrics = all_metrics

        if not metrics:
            return HealthCheckResult(
                "quota", "NOMINAL", "No active provider quota metrics recorded."
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
            status, msg = (
                "CRITICAL",
                f"Daily API quota critical: {'; '.join(critical_warnings)}",
            )
        elif degraded_warnings:
            status, msg = (
                "DEGRADED",
                f"Daily API quota degraded: {'; '.join(degraded_warnings)}",
            )
        else:
            status, msg = (
                "NOMINAL",
                "All provider daily quotas within normal limits (< 80%).",
            )

        return HealthCheckResult("quota", status, msg, details)

    def check_live_feed_crypto(self, now: datetime | None = None) -> HealthCheckResult:
        """Check status of crypto live quotes feed (<10s NOMINAL, 10-90s DEGRADED, >90s CRITICAL)."""
        current_time = now or datetime.now(UTC)
        quotes = self.session.exec(
            select(LiveQuote).where(LiveQuote.source.like("coinbase%"))
        ).all()
        hb = self.session.get(SystemHeartbeat, "live_feed_crypto")

        if not quotes and hb is None:
            return HealthCheckResult(
                check_name="live_feed_crypto",
                status="NOMINAL",
                message="Crypto live feed: no quotes recorded yet.",
            )

        newest_received = None
        for q in quotes:
            try:
                dt = _parse_utc(q.received_at)
                if newest_received is None or dt > newest_received:
                    newest_received = dt
            except (ValueError, TypeError):
                continue

        if newest_received is None:
            age_sec = 999999.0
        else:
            age_sec = (current_time - newest_received).total_seconds()

        hb_fresh = True
        if hb is not None:
            try:
                hb_dt = _parse_utc(hb.last_pulse_ts)
                if (
                    current_time - hb_dt
                ).total_seconds() > 120.0 or hb.consecutive_failures > 0:
                    hb_fresh = False
            except (ValueError, TypeError):
                hb_fresh = False

        if age_sec < 10.0 and hb_fresh:
            status = "NOMINAL"
            msg = f"Crypto live feed nominal ({age_sec:.1f}s ago)."
        elif age_sec <= 90.0 and hb_fresh:
            status = "DEGRADED"
            msg = f"Crypto live feed degraded (age {age_sec:.1f}s)."
        else:
            status = "CRITICAL"
            msg = f"Crypto live feed critical: age {age_sec:.1f}s, WebSocket and REST failing."

        return HealthCheckResult(
            check_name="live_feed_crypto",
            status=status,
            message=msg,
            details={"age_sec": age_sec, "hb_fresh": hb_fresh},
        )

    def check_live_feed_equity(self, now: datetime | None = None) -> HealthCheckResult:
        """Check status of 15m-delayed equity intraday feed (<25m NOMINAL, 25-45m DEGRADED, >45m CRITICAL)."""
        current_time = now or datetime.now(UTC)
        bars = self.session.exec(
            select(IntradayBar)
            .where(IntradayBar.source == "yfinance_intraday")
            .order_by(IntradayBar.ts.desc())
        ).all()

        if not bars:
            return HealthCheckResult(
                check_name="live_feed_equity",
                status="NOMINAL",
                message="Equity intraday feed: no intraday bars recorded yet.",
            )

        newest_dt = None
        for b in bars:
            try:
                dt = _parse_utc(b.ts)
                if newest_dt is None or dt > newest_dt:
                    newest_dt = dt
            except (ValueError, TypeError):
                continue

        if newest_dt is None:
            age_min = 999999.0
        else:
            age_min = (current_time - newest_dt).total_seconds() / 60.0

        if age_min < 25.0:
            status = "NOMINAL"
            msg = f"Equity intraday feed nominal ({age_min:.1f}m ago)."
        elif age_min <= 45.0:
            status = "DEGRADED"
            msg = f"Equity intraday feed degraded (age {age_min:.1f}m)."
        else:
            status = "CRITICAL"
            msg = f"Equity intraday feed critical (age {age_min:.1f}m >2 missed poll windows)."

        return HealthCheckResult(
            check_name="live_feed_equity",
            status=status,
            message=msg,
            details={"age_min": age_min},
        )

    def check_ws_connection(self, now: datetime | None = None) -> HealthCheckResult:
        """Check status of WebSocket client from system heartbeat."""
        hb = self.session.get(SystemHeartbeat, "live_feed_crypto")
        if hb is None:
            return HealthCheckResult(
                check_name="ws_connection",
                status="NOMINAL",
                message="WebSocket connection status: no heartbeat recorded yet.",
            )

        if hb.consecutive_failures > 0 or hb.last_error is not None:
            status = "DEGRADED"
            msg = f"WebSocket reconnecting/degraded: {hb.last_error or 'failures reported'}"
        else:
            status = "NOMINAL"
            msg = "WebSocket connected."

        return HealthCheckResult(
            check_name="ws_connection",
            status=status,
            message=msg,
        )

    def check_intraday_prune(self, now: datetime | None = None) -> HealthCheckResult:
        """Check that intraday retention prune job ran within the last 26 hours."""
        current_time = now or datetime.now(UTC)
        hb = self.session.get(SystemHeartbeat, "job_prune_intraday")
        if hb is None:
            return HealthCheckResult(
                check_name="intraday_prune",
                status="NOMINAL",
                message="Intraday prune job: no run record yet.",
            )

        try:
            pulse_dt = _parse_utc(hb.last_pulse_ts)
            age_hours = (current_time - pulse_dt).total_seconds() / 3600.0
            if age_hours > 26.0 or hb.consecutive_failures > 0:
                status = "DEGRADED"
                msg = f"Intraday prune job stale or failed (age {age_hours:.1f}h)."
            else:
                status = "NOMINAL"
                msg = f"Intraday prune job healthy ({age_hours:.1f}h ago)."
        except (ValueError, TypeError):
            status = "DEGRADED"
            msg = "Intraday prune job timestamp invalid."

        return HealthCheckResult(
            check_name="intraday_prune",
            status=status,
            message=msg,
        )

    def compute_system_status(
        self,
        now: datetime | None = None,
        display_only_checks: set[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Compute overarching system health status from all checks.

        Display-path health checks (live feeds, ws_connection, prune job) can
        contribute at most DEGRADED status to system status, ensuring a live feed
        outage never marks the training/prediction core as CRITICAL.
        """
        if display_only_checks is None:
            display_only_checks = {
                "live_feed_crypto",
                "live_feed_equity",
                "ws_connection",
                "intraday_prune",
            }

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
                self.check_live_feed_crypto(now=now),
                self.check_live_feed_equity(now=now),
                self.check_ws_connection(now=now),
                self.check_intraday_prune(now=now),
            ]
        except Exception as exc:  # noqa: BLE001
            return ("CRITICAL", [f"Database or system failure: {exc}"])

        warnings: list[str] = []
        is_critical = False
        is_degraded = False

        for res in checks:
            is_display_only = res.check_name in display_only_checks
            if res.status == "CRITICAL":
                if is_display_only:
                    is_degraded = True
                    warnings.append(f"[{res.check_name.upper()}] {res.message}")
                else:
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
