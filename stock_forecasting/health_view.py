"""Assemble the health-panel view model from the DB (no Streamlit).

Spec §6 "Health status model":

    SYSTEM: [🟢 NOMINAL]   Worker: [🟢 ALIVE lag 4s]   Data Quality: [100%]
    [yfinance]  RTT 340ms | err 0% | quota 22%  🟢 ACTIVE
    ...
    Watchdog: ingest 2m ago · forecast 1h ago · eval 4h ago    Pending evals: 4

One entry point, ``build_health_view(session)``, so the whole panel is testable
against an in-memory DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlmodel import Session, select

from stock_forecasting.health_checks import HealthChecker
from stock_forecasting.schema import (
    LinkMetrics,
    PredictionSnapshot,
    QuarantineBar,
    SystemHeartbeat,
)

STATUS_BADGE: dict[str, str] = {
    "NOMINAL": "🟢 NOMINAL",
    "DEGRADED": "🟡 DEGRADED",
    "CRITICAL": "🔴 CRITICAL",
}


def _parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _age_label(seconds: float) -> str:
    seconds = max(seconds, 0)
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f}h ago"
    return f"{seconds / 86400:.0f}d ago"


@dataclass
class ProviderCard:
    """One row of the per-provider health strip."""

    provider: str
    rtt_p50_ms: float | None
    error_rate: float
    quota_pct: float
    breaker_state: str
    badge: str


@dataclass
class WatchdogRow:
    """Freshness of one background job."""

    job_type: str
    age_seconds: float | None
    age_label: str
    ok: bool


@dataclass
class HealthView:
    """Everything the health panel renders."""

    status: str
    badge: str
    warnings: list[str] = field(default_factory=list)
    worker_label: str = "🔴 no heartbeat"
    data_quality_pct: int = 100
    providers: list[ProviderCard] = field(default_factory=list)
    watchdog: list[WatchdogRow] = field(default_factory=list)
    pending_evals: int = 0


def _provider_badge(m: LinkMetrics) -> str:
    if m.breaker_state == "open":
        return "🔴 DOWN"
    if m.breaker_state == "half_open" or (m.consecutive_failures or 0) > 0:
        return "🟡 RECOVERING"
    if (m.calls_today or 0) == 0:
        return "🟢 STANDBY"
    return "🟢 ACTIVE"


def _provider_cards(session: Session) -> list[ProviderCard]:
    metrics = session.exec(select(LinkMetrics).order_by(LinkMetrics.provider)).all()
    cards: list[ProviderCard] = []
    for m in metrics:
        quota = m.quota_pct or 0.0
        if m.daily_limit and m.daily_limit > 0:
            quota = max(quota, (m.calls_today or 0) / m.daily_limit)
        cards.append(
            ProviderCard(
                provider=m.provider,
                rtt_p50_ms=m.rtt_p50_ms,
                error_rate=m.error_rate or 0.0,
                quota_pct=quota,
                breaker_state=m.breaker_state,
                badge=_provider_badge(m),
            )
        )
    return cards


def _watchdog_rows(session: Session, now: datetime) -> tuple[list[WatchdogRow], str]:
    heartbeats = session.exec(select(SystemHeartbeat)).all()
    rows: list[WatchdogRow] = []
    newest_age: float | None = None
    for hb in sorted(heartbeats, key=lambda h: h.job_type):
        ref = hb.last_success_ts or hb.last_pulse_ts
        if ref is None:
            rows.append(WatchdogRow(hb.job_type, None, "never", False))
            continue
        age = (now - _parse_utc(ref)).total_seconds()
        rows.append(WatchdogRow(hb.job_type, age, _age_label(age), age <= 900))
        newest_age = age if newest_age is None else min(newest_age, age)

    if newest_age is None:
        worker_label = "🔴 no heartbeat"
    elif newest_age <= 300:
        worker_label = f"🟢 ALIVE lag {_age_label(newest_age)}"
    elif newest_age <= 900:
        worker_label = f"🟡 LAGGING {_age_label(newest_age)}"
    else:
        worker_label = f"🔴 DOWN {_age_label(newest_age)}"
    return rows, worker_label


def _pending_evals(session: Session, now: datetime) -> int:
    snaps = session.exec(
        select(PredictionSnapshot).where(PredictionSnapshot.evaluated_at.is_(None))
    ).all()
    now_iso = now.isoformat()
    return sum(1 for s in snaps if str(s.target_ts) <= now_iso)


def _data_quality_pct(session: Session, now: datetime) -> int:
    cutoff = now.timestamp() - 86400
    recent = [
        q
        for q in session.exec(select(QuarantineBar)).all()
        if _parse_utc(q.detected_at).timestamp() >= cutoff
    ]
    return max(0, 100 - 5 * len(recent))


def build_health_view(session: Session, now: datetime | None = None) -> HealthView:
    """Compute the full health-panel view model."""
    current = now or datetime.now(UTC)
    checker = HealthChecker(session)
    status, warnings = checker.compute_system_status(now=current)
    watchdog, worker_label = _watchdog_rows(session, current)

    return HealthView(
        status=status,
        badge=STATUS_BADGE.get(status, status),
        warnings=list(warnings),
        worker_label=worker_label,
        data_quality_pct=_data_quality_pct(session, current),
        providers=_provider_cards(session),
        watchdog=watchdog,
        pending_evals=_pending_evals(session, current),
    )
