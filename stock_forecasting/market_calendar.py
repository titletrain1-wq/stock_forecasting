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

import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")

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
    sched = _NYSE.schedule(
        start_date=(now - timedelta(days=14)).date().isoformat(),
        end_date=now.date().isoformat(),
    )
    if sched.empty:
        return None
    completed = sched[sched["market_close"] <= now]
    if completed.empty:
        return None
    return completed.index[-1].date()


def _equity_sessions_after(d: date, upto: date) -> list[date]:
    """NYSE session dates strictly after ``d`` and up to (inclusive) ``upto``."""
    if upto <= d:
        return []
    sched = _NYSE.schedule(
        start_date=(d + timedelta(days=1)).isoformat(), end_date=upto.isoformat()
    )
    return [ts.date() for ts in sched.index]


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
