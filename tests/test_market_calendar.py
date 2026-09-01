"""Tests for the end-of-day expected-bar-schedule helpers."""

from datetime import UTC, datetime

from stock_forecasting.market_calendar import (
    bar_is_stale,
    classify_bar_freshness,
    last_completed_equity_session,
)

# 2026-09-01 is a Tuesday, a normal NYSE trading day (Labor Day 2026 is Sep 7).
# NYSE regular hours are 13:30-20:00 UTC.
PRE_OPEN_TUE = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
AFTER_CLOSE_TUE = datetime(2026, 9, 1, 21, 0, tzinfo=UTC)
FRI_MIDDAY = datetime(2026, 9, 4, 16, 0, tzinfo=UTC)
SAT = datetime(2026, 9, 5, 16, 0, tzinfo=UTC)


def test_last_completed_session_is_prev_day_pre_open() -> None:
    assert last_completed_equity_session(PRE_OPEN_TUE).isoformat() == "2026-08-31"


def test_last_completed_session_is_today_after_close() -> None:
    assert last_completed_equity_session(AFTER_CLOSE_TUE).isoformat() == "2026-09-01"


def test_last_completed_session_on_weekend_is_friday() -> None:
    assert last_completed_equity_session(SAT).isoformat() == "2026-09-04"


def test_equity_premarket_with_prev_session_bar_is_nominal() -> None:
    # Monday's close bar, checked Tuesday pre-open: nothing is overdue.
    status, _ = classify_bar_freshness("equity", "2026-08-31T00:00:00Z", PRE_OPEN_TUE)
    assert status == "NOMINAL"


def test_equity_one_missed_session_is_degraded() -> None:
    # Bar is Monday's; it is now Friday midday -> Tue/Wed/Thu closed already.
    status, details = classify_bar_freshness(
        "equity", "2026-09-02T00:00:00Z", FRI_MIDDAY
    )
    assert status == "DEGRADED"
    assert details["sessions_behind"] == 1


def test_equity_multiple_missed_sessions_is_critical() -> None:
    status, details = classify_bar_freshness(
        "equity", "2026-08-31T00:00:00Z", FRI_MIDDAY
    )
    assert status == "CRITICAL"
    assert details["sessions_behind"] >= 2


def test_equity_weekend_with_friday_bar_is_nominal() -> None:
    status, _ = classify_bar_freshness("equity", "2026-09-04T00:00:00Z", SAT)
    assert status == "NOMINAL"


def test_crypto_same_day_bar_is_nominal() -> None:
    status, _ = classify_bar_freshness(
        "crypto", "2026-09-01T00:00:00Z", datetime(2026, 9, 1, 8, 40, tzinfo=UTC)
    )
    assert status == "NOMINAL"


def test_crypto_two_days_behind_is_degraded() -> None:
    status, _ = classify_bar_freshness(
        "crypto", "2026-08-30T00:00:00Z", datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    )
    assert status == "DEGRADED"


def test_crypto_four_days_behind_is_critical() -> None:
    status, _ = classify_bar_freshness(
        "crypto", "2026-08-28T00:00:00Z", datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    )
    assert status == "CRITICAL"


def test_no_bar_is_degraded_no_data() -> None:
    status, details = classify_bar_freshness("crypto", None, PRE_OPEN_TUE)
    assert status == "DEGRADED"
    assert details["reason"] == "no_data"


def test_bar_is_stale_matches_non_nominal() -> None:
    assert bar_is_stale("equity", "2026-08-31T00:00:00Z", FRI_MIDDAY) is True
    assert bar_is_stale("crypto", "2026-09-01T00:00:00Z", PRE_OPEN_TUE) is False
