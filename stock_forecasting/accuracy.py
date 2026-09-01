"""Accuracy calculation and aggregate rebuilding service."""

import logging
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
from sqlmodel import Session, select

from stock_forecasting.schema import AccuracyRecord, PredictionSnapshot

logger = logging.getLogger(__name__)


class AccuracyService:
    """Service for computing and caching model accuracy metrics."""

    def __init__(self, session: Session | None = None) -> None:
        """Initialize AccuracyService with an optional database session.

        Args:
            session: Optional SQLModel database session.
        """
        self.session = session

    @classmethod
    def _compute_metrics(
        cls, snapshots: list[PredictionSnapshot]
    ) -> dict[str, float | int]:
        """Compute aggregate metrics for a list of evaluated prediction snapshots.

        Args:
            snapshots: List of evaluated PredictionSnapshot objects.

        Returns:
            Dictionary containing n, mae, rmse, dir_acc, ci_coverage, mae_price_pct, is_trustworthy.
        """
        n = len(snapshots)
        if n == 0:
            return {
                "n": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "dir_acc": 0.0,
                "ci_coverage": 0.0,
                "mae_price_pct": 0.0,
                "is_trustworthy": 0,
            }

        abs_errors = [s.error_abs for s in snapshots if s.error_abs is not None]
        mae = float(np.mean(abs_errors)) if abs_errors else 0.0

        signed_errors = [
            s.error_signed for s in snapshots if s.error_signed is not None
        ]
        rmse = (
            float(np.sqrt(np.mean(np.square(signed_errors)))) if signed_errors else 0.0
        )

        dir_hits = [
            s.is_direction_hit for s in snapshots if s.is_direction_hit is not None
        ]
        dir_acc = float(np.mean(dir_hits)) if dir_hits else 0.0

        ci_hits = [s.is_within_ci for s in snapshots if s.is_within_ci is not None]
        ci_coverage = float(np.mean(ci_hits)) if ci_hits else 0.0

        price_pct_errors = [
            abs(s.predicted_price - s.realized_price) / s.realized_price
            for s in snapshots
            if s.predicted_price is not None
            and s.realized_price is not None
            and s.realized_price != 0
        ]
        mae_price_pct = float(np.mean(price_pct_errors)) if price_pct_errors else 0.0

        is_trustworthy = 1 if (dir_acc >= 0.55 and n >= 30) else 0

        return {
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "dir_acc": dir_acc,
            "ci_coverage": ci_coverage,
            "mae_price_pct": mae_price_pct,
            "is_trustworthy": is_trustworthy,
        }

    @classmethod
    def _upsert_record(
        cls,
        session: Session,
        scope: str,
        ticker: str | None,
        horizon: str,
        model_type: str,
        metrics: dict[str, float | int],
        window: str = "all",
        now_iso: str | None = None,
    ) -> AccuracyRecord:
        """Upsert a single AccuracyRecord row into the database.

        Args:
            session: Active database session.
            scope: Record scope ('ticker' or 'global').
            ticker: Ticker symbol (or None for global scope).
            horizon: Forecast horizon (e.g. '1d', '5d', '30d').
            model_type: Model identifier (e.g. 'ridge', 'random_forest').
            metrics: Computed metric dictionary.
            window: Aggregation window label (default 'all').
            now_iso: Current UTC timestamp ISO string.

        Returns:
            The created or updated AccuracyRecord instance.
        """
        if now_iso is None:
            now_iso = datetime.now(UTC).isoformat()

        ticker_cond = (
            AccuracyRecord.ticker.is_(None)
            if ticker is None
            else (AccuracyRecord.ticker == ticker)
        )
        statement = select(AccuracyRecord).where(
            AccuracyRecord.scope == scope,
            ticker_cond,
            AccuracyRecord.horizon == horizon,
            AccuracyRecord.model_type == model_type,
            AccuracyRecord.window == window,
        )
        record = session.exec(statement).first()

        if record is None:
            record = AccuracyRecord(
                scope=scope,
                ticker=ticker,
                horizon=horizon,
                model_type=model_type,
                window=window,
                n=int(metrics["n"]),
                mae=float(metrics["mae"]),
                rmse=float(metrics["rmse"]),
                dir_acc=float(metrics["dir_acc"]),
                ci_coverage=float(metrics["ci_coverage"]),
                mae_price_pct=float(metrics["mae_price_pct"]),
                is_trustworthy=int(metrics["is_trustworthy"]),
                updated_at=now_iso,
            )
            session.add(record)
        else:
            record.n = int(metrics["n"])
            record.mae = float(metrics["mae"])
            record.rmse = float(metrics["rmse"])
            record.dir_acc = float(metrics["dir_acc"])
            record.ci_coverage = float(metrics["ci_coverage"])
            record.mae_price_pct = float(metrics["mae_price_pct"])
            record.is_trustworthy = int(metrics["is_trustworthy"])
            record.updated_at = now_iso
            session.add(record)

        return record

    @classmethod
    def rebuild_aggregates(
        cls,
        session: Session,
        window: str = "all",
    ) -> list[AccuracyRecord]:
        """Rebuild accuracy aggregate records from evaluated prediction snapshots.

        Queries all evaluated PredictionSnapshot rows, calculates metrics per
        (ticker, horizon, model_type) and globally per (horizon, model_type),
        upserts into accuracy_records, and commits the session.

        Args:
            session: Active database session.
            window: Aggregation window label (default 'all').

        Returns:
            List of created/updated AccuracyRecord rows.
        """
        now_iso = datetime.now(UTC).isoformat()
        stmt = select(PredictionSnapshot).where(
            PredictionSnapshot.evaluated_at.is_not(None)
        )
        evaluated_snapshots = list(session.exec(stmt).all())

        if not evaluated_snapshots:
            logger.info(
                "No evaluated prediction snapshots found; skipping aggregate rebuild."
            )
            session.commit()
            return []

        # 1. Group by (ticker, horizon, model_type)
        ticker_groups: dict[tuple[str, str, str], list[PredictionSnapshot]] = (
            defaultdict(list)
        )
        # 2. Group by (horizon, model_type) for global scope
        global_groups: dict[tuple[str, str], list[PredictionSnapshot]] = defaultdict(
            list
        )

        for snap in evaluated_snapshots:
            ticker_groups[(snap.ticker, snap.horizon, snap.model_type)].append(snap)
            global_groups[(snap.horizon, snap.model_type)].append(snap)

        records: list[AccuracyRecord] = []

        # Rebuild ticker-scoped aggregates
        for (ticker, horizon, model_type), snaps in ticker_groups.items():
            metrics = cls._compute_metrics(snaps)
            rec = cls._upsert_record(
                session=session,
                scope="ticker",
                ticker=ticker,
                horizon=horizon,
                model_type=model_type,
                metrics=metrics,
                window=window,
                now_iso=now_iso,
            )
            records.append(rec)

        # Rebuild global-scoped aggregates
        for (horizon, model_type), snaps in global_groups.items():
            metrics = cls._compute_metrics(snaps)
            rec = cls._upsert_record(
                session=session,
                scope="global",
                ticker=None,
                horizon=horizon,
                model_type=model_type,
                metrics=metrics,
                window=window,
                now_iso=now_iso,
            )
            records.append(rec)

        session.commit()
        logger.info(
            "Rebuilt %d accuracy aggregate records (ticker & global).",
            len(records),
        )
        return records

    def run(
        self, session: Session | None = None, window: str = "all"
    ) -> list[AccuracyRecord]:
        """Instance method to trigger aggregate rebuild.

        Args:
            session: Optional database session (falls back to self.session).
            window: Aggregation window label (default 'all').

        Returns:
            List of created/updated AccuracyRecord rows.
        """
        sess = session or self.session
        if sess is None:
            raise ValueError("A database session is required to rebuild aggregates.")
        return self.rebuild_aggregates(sess, window=window)
