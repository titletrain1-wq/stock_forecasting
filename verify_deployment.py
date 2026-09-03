#!/usr/bin/env python3
"""Verify end-to-end deployment: data, forecasting, accuracy."""

import json
from datetime import UTC, datetime

from sqlmodel import Session, select

from stock_forecasting.database import get_engine
from stock_forecasting.schema import (
    AccuracyRecord,
    ModelRun,
    OhlcvBar,
    PredictionSnapshot,
)


def verify_real_data():
    """Verify OHLCV bars are fresh, all tickers present, growing."""
    engine = get_engine()
    with Session(engine) as session:
        # Get bar stats
        bars = session.exec(select(OhlcvBar)).all()
        if not bars:
            return "FAIL", "No bars in database"

        # Parse ts strings to datetime for comparison
        from dateutil.parser import isoparse

        max_ts = max(isoparse(b.ts) for b in bars)
        now = datetime.now(UTC)
        days_old = (now - max_ts).days

        # Get unique tickers
        tickers = {b.ticker for b in bars}
        expected_tickers = {
            "AAPL",
            "NVDA",
            "SPY",
            "BTC-USD",
            "ETH-USD",
        }  # daily forecaster watchlist

        result = {
            "latest_bar_ts": max_ts.isoformat(),
            "days_old": days_old,
            "bar_count": len(bars),
            "tickers_present": sorted(tickers),
            "expected_tickers": sorted(expected_tickers),
        }

        if days_old > 2:
            return "FAIL", f"Bars too old ({days_old} days)", result
        if not tickers >= expected_tickers:
            return "FAIL", f"Missing tickers: {expected_tickers - tickers}", result

        return "PASS", "Fresh data, all tickers", result


def verify_forecasting():
    """Verify prediction_snapshots exist with horizons, models trained."""
    engine = get_engine()
    with Session(engine) as session:
        # Get configured horizons from viz module
        from stock_forecasting.viz import DEFAULT_LATEST_HORIZONS

        horizons = DEFAULT_LATEST_HORIZONS

        # Check prediction snapshots
        snapshots = session.exec(select(PredictionSnapshot)).all()
        models = session.exec(select(ModelRun)).all()

        result = {
            "prediction_snapshots_count": len(snapshots),
            "model_runs_count": len(models),
            "configured_horizons": horizons,
        }

        if snapshots:
            snapshot_horizons = {s.horizon for s in snapshots}
            result["horizons_in_db"] = sorted(snapshot_horizons)
            has_bounds = all(
                s.predicted_price is not None
                and s.lower_bound is not None
                and s.upper_bound is not None
                for s in snapshots[:5]  # Check first 5
            )
            result["has_bounds"] = has_bounds

        if not snapshots:
            return "WARN", "No prediction snapshots yet", result
        if not models:
            return "WARN", "No model runs recorded", result

        return "PASS", f"{len(snapshots)} snapshots, {len(models)} models", result


def verify_accuracy_pipeline():
    """Verify accuracy pipeline is wired and works."""
    engine = get_engine()
    with Session(engine) as session:
        accuracy_records = session.exec(select(AccuracyRecord)).all()

        result = {
            "accuracy_records_count": len(accuracy_records),
        }

        if accuracy_records:
            result["sample_record"] = {
                "ticker": accuracy_records[0].ticker,
                "horizon": accuracy_records[0].horizon,
                "evaluated_at": accuracy_records[0].evaluated_at.isoformat()
                if accuracy_records[0].evaluated_at
                else None,
            }

        # Check if evaluator module exists and has test
        try:
            import stock_forecasting.evaluator  # noqa: F401

            result["evaluator_module_exists"] = True
        except ImportError:
            result["evaluator_module_exists"] = False

        if accuracy_records:
            return "PASS", f"{len(accuracy_records)} accuracy records", result
        else:
            return (
                "WARN",
                "No accuracy records yet (may be OK if no forecasts matured)",
                result,
            )


if __name__ == "__main__":
    print("=" * 70)
    print("END-TO-END DEPLOYMENT VERIFICATION")
    print("=" * 70)

    print("\n1. REAL DATA")
    status, msg, data = verify_real_data()
    print(f"   Status: {status}")
    print(f"   Message: {msg}")
    print(f"   Data: {json.dumps(data, indent=2, default=str)}")

    print("\n2. FORECASTING / HORIZON")
    status, msg, data = verify_forecasting()
    print(f"   Status: {status}")
    print(f"   Message: {msg}")
    print(f"   Data: {json.dumps(data, indent=2, default=str)}")

    print("\n3. ACCURACY PIPELINE")
    status, msg, data = verify_accuracy_pipeline()
    print(f"   Status: {status}")
    print(f"   Message: {msg}")
    print(f"   Data: {json.dumps(data, indent=2, default=str)}")

    print("\n" + "=" * 70)
