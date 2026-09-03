"""M5: intraday forecast evaluator.

Runs hourly from the worker. For every ``intraday_prediction_snapshots`` row
whose ``target_ts`` has passed, it finds the realized closed bar in
``intraday_bars_history`` and writes / updates one ``intraday_accuracy_records``
row (signed error, directional hit, CI coverage). A forecast whose target bar
has not landed yet just gets its ``grading_attempts`` bumped and is retried
next hour.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, select

from stock_forecasting.schema import (
    IntradayAccuracyRecord,
    IntradayBarsHistory,
    IntradayPredictionSnapshot,
)

logger = logging.getLogger(__name__)

_MATCH_TOLERANCE = timedelta(minutes=15)  # accept the first closed bar at/after target


def _realized_close(
    session: Session, ticker: str, target_ts: str
) -> tuple[float, str] | None:
    """Return (close, ts) of the first closed 5m bar at/after ``target_ts``.

    Anchors + horizons land exactly on a 5m boundary, so an exact match is the
    norm; the small tolerance covers a missing bar (exchange gap).
    """
    target = datetime.fromisoformat(target_ts)
    upper = (target + _MATCH_TOLERANCE).isoformat().replace("+00:00", "Z")
    row = session.exec(
        select(IntradayBarsHistory)
        .where(
            (IntradayBarsHistory.ticker == ticker)
            & (IntradayBarsHistory.interval == "5m")
            & (IntradayBarsHistory.ts >= target_ts)
            & (IntradayBarsHistory.ts <= upper)
        )
        .order_by(IntradayBarsHistory.ts.asc())
    ).first()
    return (row.close, row.ts) if row else None


def grade_intraday_forecasts(session: Session, now: datetime | None = None) -> dict:
    """Grade every matured, ungraded intraday forecast. Returns a small summary."""
    now = now or datetime.now(UTC)
    now_iso = now.isoformat().replace("+00:00", "Z")
    graded = deferred = 0

    snapshots = session.exec(
        select(IntradayPredictionSnapshot).where(
            IntradayPredictionSnapshot.target_ts <= now_iso
        )
    ).all()

    for snap in snapshots:
        record = session.exec(
            select(IntradayAccuracyRecord).where(
                IntradayAccuracyRecord.prediction_id == snap.id
            )
        ).first()
        if record is not None and record.realized_return is not None:
            continue  # already fully graded

        if record is None:
            record = IntradayAccuracyRecord(
                prediction_id=snap.id,
                ticker=snap.ticker,
                horizon=snap.horizon,
                graded_at=now_iso,
                grading_attempts=0,
            )
            session.add(record)

        hit = _realized_close(session, snap.ticker, snap.target_ts)
        record.graded_at = now_iso
        if hit is None:
            record.grading_attempts += 1
            deferred += 1
            continue

        realized_price, _ = hit
        realized_return = math.log(realized_price / snap.anchor_price)
        record.realized_price = realized_price
        record.realized_return = realized_return
        record.signed_error = realized_return - snap.predicted_return
        record.abs_error_pct = abs(record.signed_error) * 100
        record.direction_hit = int(
            (realized_return >= 0) == (snap.predicted_return >= 0)
        )
        record.ci_cover = int(
            snap.ci_lower_return <= realized_return <= snap.ci_upper_return
        )
        record.grading_attempts += 1
        graded += 1

    session.commit()
    logger.info("intraday evaluator: %d graded, %d deferred", graded, deferred)
    return {"graded": graded, "deferred": deferred, "checked": len(snapshots)}
