"""Evaluator service for scoring matured prediction snapshots against realized prices."""

import logging
from datetime import UTC, datetime

import numpy as np
from sqlmodel import Session, select

from stock_forecasting.accuracy import AccuracyService
from stock_forecasting.schema import OhlcvBar, PredictionSnapshot

logger = logging.getLogger(__name__)


class EvaluatorService:
    """Service evaluating matured forecasts against actual market prices."""

    def __init__(self, session: Session | None = None) -> None:
        """Initialize EvaluatorService with an optional database session.

        Args:
            session: Optional SQLModel database session.
        """
        self.session = session

    def run(
        self, session: Session | None = None, as_of: str | None = None
    ) -> list[PredictionSnapshot]:
        """Evaluate all matured prediction snapshots up to as_of.

        Finds all PredictionSnapshot rows where target_ts <= as_of and
        evaluated_at is NULL. For each snapshot:
        - Finds the nearest OhlcvBar for ticker where ts >= target_ts.
        - If found: computes realized_price, realized_return, error_abs, error_signed,
          is_direction_hit, is_within_ci, and sets evaluated_at.
        - If bar is missing: increments eval_attempts.
        - Commits session and calls AccuracyService.rebuild_aggregates(session).

        Args:
            session: Active database session (falls back to self.session).
            as_of: Cutoff ISO-8601 UTC timestamp (defaults to now).

        Returns:
            List of evaluated PredictionSnapshot instances.
        """
        sess = session or self.session
        if sess is None:
            raise ValueError("A database session is required for EvaluatorService.")

        if as_of is None:
            as_of = datetime.now(UTC).isoformat()

        # Query all matured but unevaluated prediction snapshots
        stmt = (
            select(PredictionSnapshot)
            .where(
                PredictionSnapshot.target_ts <= as_of,
                PredictionSnapshot.evaluated_at.is_(None),
            )
            .order_by(PredictionSnapshot.target_ts.asc())
        )
        candidates = list(sess.exec(stmt).all())

        if not candidates:
            logger.debug(
                "No matured unevaluated prediction snapshots found as of %s.", as_of
            )
            return []

        evaluated_snapshots: list[PredictionSnapshot] = []
        now_iso = datetime.now(UTC).isoformat()

        for snap in candidates:
            # Find nearest bar at or after target_ts
            bar_stmt = (
                select(OhlcvBar)
                .where(
                    OhlcvBar.ticker == snap.ticker,
                    OhlcvBar.ts >= snap.target_ts,
                )
                .order_by(OhlcvBar.ts.asc())
                .limit(1)
            )
            bar = sess.exec(bar_stmt).first()

            if bar is not None:
                if bar.adj_close is not None and not np.isnan(bar.adj_close):
                    realized_price = float(bar.adj_close)
                else:
                    realized_price = float(bar.close)

                anchor_price = float(snap.anchor_price)
                if anchor_price <= 0 or realized_price <= 0:
                    realized_return = 0.0
                else:
                    realized_return = float(np.log(realized_price / anchor_price))

                predicted_return = float(snap.predicted_return)
                error_abs = float(abs(predicted_return - realized_return))
                error_signed = float(predicted_return - realized_return)

                is_direction_hit = (
                    1 if np.sign(predicted_return) == np.sign(realized_return) else 0
                )
                is_within_ci = (
                    1 if (snap.lower_bound <= realized_price <= snap.upper_bound) else 0
                )

                snap.realized_price = realized_price
                snap.realized_return = realized_return
                snap.error_abs = error_abs
                snap.error_signed = error_signed
                snap.is_direction_hit = is_direction_hit
                snap.is_within_ci = is_within_ci
                snap.evaluated_at = now_iso
                sess.add(snap)
                evaluated_snapshots.append(snap)
            else:
                snap.eval_attempts = (snap.eval_attempts or 0) + 1
                sess.add(snap)

        sess.commit()

        # Rebuild accuracy aggregates after committing evaluated snapshots
        AccuracyService.rebuild_aggregates(sess)

        logger.info(
            "Evaluated %d/%d matured prediction snapshots as of %s.",
            len(evaluated_snapshots),
            len(candidates),
            as_of,
        )
        return evaluated_snapshots

    def evaluate_matured(self, as_of: str | None = None) -> list[PredictionSnapshot]:
        """Convenience alias for run() to evaluate matured snapshots.

        Args:
            as_of: Optional cutoff timestamp.

        Returns:
            List of evaluated PredictionSnapshot instances.
        """
        return self.run(session=self.session, as_of=as_of)
