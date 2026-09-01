"""Tests for the health-panel view model (stock_forecasting.health_view)."""

from datetime import UTC, datetime, timedelta

from stock_forecasting.health_view import (
    STATUS_BADGE,
    _age_label,
    build_health_view,
)
from stock_forecasting.schema import (
    LinkMetrics,
    PredictionSnapshot,
    SystemHeartbeat,
)

NOW = datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC)


def _iso(**delta) -> str:
    return (NOW - timedelta(**delta)).isoformat()


def test_age_label_buckets() -> None:
    assert _age_label(4) == "4s ago"
    assert _age_label(120) == "2m ago"
    assert _age_label(3600) == "60m ago"
    assert _age_label(7200) == "2h ago"
    assert _age_label(3 * 86400) == "3d ago"


def test_empty_db_is_critical_no_heartbeat(db_session) -> None:
    view = build_health_view(db_session, now=NOW)
    # watchdog check returns CRITICAL with no heartbeats
    assert view.status == "CRITICAL"
    assert view.badge == STATUS_BADGE["CRITICAL"]
    assert view.worker_label == "🔴 no heartbeat"
    assert view.providers == []
    assert view.pending_evals == 0
    assert view.data_quality_pct == 100


def test_nominal_with_fresh_heartbeat_and_healthy_provider(db_session) -> None:
    db_session.add(
        SystemHeartbeat(
            job_type="job_heartbeat",
            last_pulse_ts=_iso(seconds=30),
            last_success_ts=_iso(seconds=30),
            consecutive_failures=0,
        )
    )
    db_session.add(
        LinkMetrics(
            provider="yfinance",
            rtt_p50_ms=340.0,
            rtt_p95_ms=800.0,
            rtt_jitter_ms=50.0,
            error_rate=0.0,
            consecutive_failures=0,
            breaker_state="closed",
            calls_today=220,
            daily_limit=1000,
            quota_pct=0.22,
            updated_at=_iso(seconds=30),
        )
    )
    db_session.commit()

    view = build_health_view(db_session, now=NOW)
    assert view.status == "NOMINAL"
    assert view.worker_label.startswith("🟢 ALIVE")
    card = view.providers[0]
    assert card.provider == "yfinance"
    assert card.badge == "🟢 ACTIVE"
    assert round(card.quota_pct, 2) == 0.22


def test_open_breaker_provider_marked_down(db_session) -> None:
    db_session.add(SystemHeartbeat(job_type="hb", last_success_ts=_iso(seconds=10)))
    db_session.add(
        LinkMetrics(
            provider="coingecko",
            breaker_state="open",
            consecutive_failures=6,
            error_rate=1.0,
            calls_today=3,
            daily_limit=10000,
            updated_at=_iso(seconds=10),
        )
    )
    db_session.commit()
    view = build_health_view(db_session, now=NOW)
    assert view.providers[0].badge == "🔴 DOWN"
    assert view.status in ("DEGRADED", "CRITICAL")


def test_provider_badge_standby_and_recovering(db_session) -> None:
    db_session.add(SystemHeartbeat(job_type="hb", last_pulse_ts=_iso(seconds=10)))
    db_session.add(
        LinkMetrics(
            provider="coinbase",
            breaker_state="closed",
            consecutive_failures=0,
            calls_today=0,
            daily_limit=10000,
            updated_at=_iso(seconds=10),
        )
    )
    db_session.add(
        LinkMetrics(
            provider="finnhub",
            breaker_state="half_open",
            consecutive_failures=2,
            calls_today=4,
            daily_limit=1000,
            updated_at=_iso(seconds=10),
        )
    )
    db_session.commit()
    badges = {
        c.provider: c.badge for c in build_health_view(db_session, now=NOW).providers
    }
    assert badges["coinbase"] == "🟢 STANDBY"
    assert badges["finnhub"] == "🟡 RECOVERING"


def test_closed_breaker_with_stale_failure_count_is_not_recovering(db_session) -> None:
    """A long-closed breaker whose consecutive_failures never got reset (dead
    provider, never re-polled) must not be stuck on RECOVERING forever."""
    db_session.add(SystemHeartbeat(job_type="hb", last_pulse_ts=_iso(seconds=10)))
    db_session.add(
        LinkMetrics(
            provider="coingecko",
            breaker_state="closed",
            consecutive_failures=2,
            error_rate=0.0,
            calls_today=0,
            daily_limit=10000,
            updated_at=_iso(seconds=10),
        )
    )
    db_session.commit()
    badges = {
        c.provider: c.badge for c in build_health_view(db_session, now=NOW).providers
    }
    assert badges["coingecko"] == "🟢 STANDBY"


def test_worker_label_lagging_then_down(db_session) -> None:
    db_session.add(SystemHeartbeat(job_type="a", last_pulse_ts=_iso(minutes=8)))
    db_session.commit()
    assert build_health_view(db_session, now=NOW).worker_label.startswith("🟡 LAGGING")

    hb = db_session.get(SystemHeartbeat, "a")
    hb.last_pulse_ts = _iso(minutes=40)
    db_session.add(hb)
    db_session.commit()
    assert build_health_view(db_session, now=NOW).worker_label.startswith("🔴 DOWN")


def test_pending_evals_counts_only_matured_ungraded(db_session) -> None:
    base = {
        "ticker": "AAPL",
        "made_at": _iso(days=10),
        "made_from_ts": _iso(days=10),
        "anchor_price": 100.0,
        "predicted_return": 0.01,
        "predicted_price": 101.0,
        "lower_bound": 98.0,
        "upper_bound": 104.0,
        "model_type": "ridge",
        "model_version": "1.0.0",
        "model_run_id": 1,
        "explain_json": "{}",
        "input_is_stale": 0,
    }
    db_session.add(SystemHeartbeat(job_type="hb", last_success_ts=_iso(seconds=5)))
    # matured, ungraded -> counts
    db_session.add(
        PredictionSnapshot(
            prediction_id="a", horizon="1d", target_ts=_iso(days=5), **base
        )
    )
    # not matured -> excluded
    db_session.add(
        PredictionSnapshot(
            prediction_id="b",
            horizon="30d",
            target_ts=(NOW + timedelta(days=5)).isoformat(),
            **base,
        )
    )
    # matured but graded -> excluded
    db_session.add(
        PredictionSnapshot(
            prediction_id="c",
            horizon="5d",
            target_ts=_iso(days=3),
            realized_price=101.2,
            evaluated_at=_iso(days=2),
            **base,
        )
    )
    db_session.commit()
    view = build_health_view(db_session, now=NOW)
    assert view.pending_evals == 1
