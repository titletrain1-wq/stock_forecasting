"""Expected-bar-schedule helpers shared by freshness checks and staleness flags.

The app is an **end-of-day** system: every provider returns daily OHLCV bars, so
"is this bar stale?" is a question about the trading calendar, not wall-clock
minutes. Equities follow the NYSE calendar (``pandas-market-calendars``); crypto
trades 24/7 so its schedule is simply one bar per calendar day (UTC).

Spec ref: design doc "Equities: pandas-market-calendars for market hours +
holidays. Crypto: 24/7 calendar."
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


@lru_cache(maxsize=64)
def _sessions_and_closes(
    start_iso: str, end_iso: str
) -> tuple[tuple[date, ...], tuple[datetime, ...]]:
    """Cached NYSE (session_date, market_close_utc) pairs for a date range.

    ``check_freshness`` builds this per equity ticker on every Streamlit rerun;
    the range args are stable within a render so the cache collapses it to one
    calendar build.
    """
    sched = _NYSE.schedule(start_date=start_iso, end_date=end_iso)
    dates = tuple(ts.date() for ts in sched.index)
    closes = tuple(c.to_pydatetime() for c in sched["market_close"])
    return dates, closes


# Crypto: a same-day or one-day-old bar is fine (a provider may not have published
# today's forming candle yet early in the UTC day); 2 days behind is degraded, more
# is critical.
_CRYPTO_DEGRADED_DAYS = 2
_CRYPTO_CRITICAL_DAYS = 3


def _as_utc(ts: str | datetime) -> datetime:
    """Coerce an ISO string or datetime to a tz-aware UTC datetime."""
    dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def last_completed_equity_session(now: datetime) -> date | None:
    """Most recent NYSE session whose regular close is at or before ``now``.

    Pre-market on a trading day this is the *previous* session; on a weekend or
    holiday it is the last weekday session.
    """
    now = _as_utc(now)
    dates, closes = _sessions_and_closes(
        (now - timedelta(days=14)).date().isoformat(), now.date().isoformat()
    )
    completed = [d for d, c in zip(dates, closes, strict=True) if c <= now]
    return completed[-1] if completed else None


def equity_market_is_open(now: datetime | None = None) -> bool:
    """True if the NYSE regular session is open at ``now`` (weekends/holidays/
    after-hours are all False).

    Used by the live equity intraday feed check: a stale 15-min feed is only a
    problem while the market is actually trading.
    """
    current = _as_utc(now) if now is not None else datetime.now(UTC)
    sched = _NYSE.schedule(
        start_date=(current - timedelta(days=7)).date().isoformat(),
        end_date=current.date().isoformat(),
    )
    try:
        return bool(_NYSE.open_at_time(sched, current))
    except ValueError:
        # open_at_time raises if `current` is outside the schedule's span
        # (e.g. after the last session's close) -> market is closed.
        return False


def _equity_sessions_after(d: date, upto: date) -> list[date]:
    """NYSE session dates strictly after ``d`` and up to (inclusive) ``upto``."""
    if upto <= d:
        return []
    dates, _ = _sessions_and_closes(
        (d + timedelta(days=1)).isoformat(), upto.isoformat()
    )
    return list(dates)


def classify_bar_freshness(
    asset_class: str, last_bar_ts: str | datetime | None, now: datetime | None = None
) -> tuple[str, dict]:
    """Classify a ticker's latest bar against its expected daily schedule.

    Returns ``(status, details)`` where status is ``"NOMINAL"``, ``"DEGRADED"``
    or ``"CRITICAL"``. ``None`` (no bar at all) is ``"DEGRADED"`` / ``no_data``.
    """
    current = _as_utc(now) if now is not None else datetime.now(UTC)

    if last_bar_ts is None:
        return "DEGRADED", {"reason": "no_data"}

    bar_date = _as_utc(last_bar_ts).date()
    is_crypto = asset_class.lower() == "crypto"

    if is_crypto:
        days_behind = (current.date() - bar_date).days
        details = {"asset_class": "crypto", "days_behind": days_behind}
        if days_behind >= _CRYPTO_CRITICAL_DAYS:
            return "CRITICAL", details
        if days_behind >= _CRYPTO_DEGRADED_DAYS:
            return "DEGRADED", details
        return "NOMINAL", details

    expected = last_completed_equity_session(current)
    details = {
        "asset_class": "equity",
        "expected_session": expected.isoformat() if expected else None,
        "latest_bar_date": bar_date.isoformat(),
    }
    if expected is None or bar_date >= expected:
        return "NOMINAL", details

    missed = _equity_sessions_after(bar_date, expected)
    details["sessions_behind"] = len(missed)
    if len(missed) >= 2:
        return "CRITICAL", details
    if len(missed) == 1:
        return "DEGRADED", details
    return "NOMINAL", details


def bar_is_stale(
    asset_class: str, last_bar_ts: str | datetime | None, now: datetime | None = None
) -> bool:
    """True when the latest bar is behind its expected daily schedule at all.

    Used for the prediction ``input_is_stale`` flag: any non-NOMINAL freshness
    means the model ran on data older than the market has since produced.
    """
    status, _ = classify_bar_freshness(asset_class, last_bar_ts, now)
    return status != "NOMINAL"
