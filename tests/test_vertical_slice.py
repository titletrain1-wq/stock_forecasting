import pytest
from stock_forecasting.database import create_tables
from stock_forecasting.schema import Ticker, PredictionSnapshot, AccuracyRecord, OhlcvBar
from stock_forecasting.providers.fake import FakeProvider
from stock_forecasting.ingestion import IngestionService
from stock_forecasting.trainer import Trainer
from stock_forecasting.forecaster import ForecastService
from stock_forecasting.evaluator import EvaluatorService
from sqlmodel import Session, select
from datetime import date, datetime, timedelta, timezone

def test_full_vertical_slice(temp_db):
    """End-to-end integration test of the vertical slice (M0-M4)."""
    with Session(temp_db) as session:
        # 1. Setup active ticker
        ticker = Ticker(
            symbol="AAPL",
            asset_class="equity",
            display_name="Apple Inc.",
            provider="fake",
            provider_symbol="AAPL",
            price_basis="raw",
            added_at="2024-01-01T00:00:00Z",
            active=1
        )
        session.add(ticker)
        session.commit()

        # 2. Ingest bars via IngestionService
        provider = FakeProvider()
        ingestion = IngestionService(session, {"fake": provider})
        backfill_res = ingestion.backfill("AAPL", years=1)
        assert backfill_res["inserted"] > 0

        # 3. Train models via Trainer
        trainer = Trainer(session)
        artifact_1d = trainer.train("AAPL", "1d", "ridge")
        assert artifact_1d.model_type == "ridge"
        assert artifact_1d.wf_mae > 0

        # 4. Forecast + write immutable ledger via ForecastService
        forecaster = ForecastService(session)
        forecast_res = forecaster.generate_and_persist("AAPL", horizons=["1d"], model_types=["ridge"])
        assert "1d_ridge" in forecast_res or "1d" in forecast_res

        # Verify snapshot written to DB
        snapshots = session.exec(select(PredictionSnapshot).where(PredictionSnapshot.ticker == "AAPL")).all()
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap.predicted_price > 0
        assert snap.evaluated_at is None

        # 5. Add a realized bar at target_ts and grade via EvaluatorService
        realized_bar = OhlcvBar(
            ticker="AAPL",
            ts=snap.target_ts,
            open=snap.anchor_price * 1.01,
            high=snap.anchor_price * 1.03,
            low=snap.anchor_price * 0.99,
            close=snap.anchor_price * 1.02,
            adj_close=snap.anchor_price * 1.02,
            volume=1000000,
            source="fake",
            ingested_at="2026-09-01T00:00:00Z"
        )
        session.add(realized_bar)
        session.commit()

        # Run evaluator
        evaluator = EvaluatorService()
        evaluator.run(session, as_of="2030-01-01T00:00:00Z")

        # Verify snapshot graded
        session.refresh(snap)
        assert snap.evaluated_at is not None
        assert snap.realized_price == snap.anchor_price * 1.02
        assert snap.is_direction_hit in (0, 1)

        # Verify accuracy records generated
        acc_records = session.exec(select(AccuracyRecord)).all()
        assert len(acc_records) > 0
