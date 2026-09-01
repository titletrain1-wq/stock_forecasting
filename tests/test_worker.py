"""Tests for WorkerScheduler background jobs, scheduling, and watchdog heartbeats."""

from unittest.mock import patch

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import Engine
from sqlmodel import Session, select

from stock_forecasting.config import Settings
from stock_forecasting.providers.fake import FakeProvider
from stock_forecasting.schema import SystemHeartbeat, Ticker
from stock_forecasting.worker import WorkerScheduler, _update_heartbeat


def _create_sample_ticker(
    session: Session,
    symbol: str,
    asset_class: str,
    provider: str = "fake",
) -> Ticker:
    """Helper to insert a sample ticker for testing."""
    ticker = Ticker(
        symbol=symbol,
        asset_class=asset_class,
        display_name=f"Asset {symbol}",
        provider=provider,
        provider_symbol=symbol,
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    session.add(ticker)
    session.commit()
    return ticker


def test_worker_scheduler_init(temp_db: Engine) -> None:
    """Verify WorkerScheduler initializes, registers expected jobs, and starts/stops cleanly."""
    settings = Settings(
        poll_interval_crypto_sec=30,
        poll_interval_equity_min=10,
        retrain_hour_utc=3,
        db_path=":memory:",
    )
    worker = WorkerScheduler(engine=temp_db, settings=settings)

    assert worker.engine is temp_db
    assert worker.settings.poll_interval_crypto_sec == 30
    assert "fake" in worker.providers
    assert "yfinance" in worker.providers

    worker.start()
    try:
        jobs = {job.id: job for job in worker.scheduler.get_jobs()}
        expected_job_ids = {
            "job_ingest_crypto",
            "job_ingest_equities",
            "job_retrain_nightly",
            "job_evaluate_hourly",
            "job_heartbeat",
        }
        assert expected_job_ids.issubset(jobs.keys())

        # Verify trigger configurations
        crypto_trigger = jobs["job_ingest_crypto"].trigger
        assert isinstance(crypto_trigger, IntervalTrigger)
        assert crypto_trigger.interval.total_seconds() == 30

        equity_trigger = jobs["job_ingest_equities"].trigger
        assert isinstance(equity_trigger, IntervalTrigger)
        assert equity_trigger.interval.total_seconds() == 600

        retrain_trigger = jobs["job_retrain_nightly"].trigger
        assert isinstance(retrain_trigger, CronTrigger)
        assert str(retrain_trigger.fields[5]) == "3"  # hour field in cron trigger

        eval_trigger = jobs["job_evaluate_hourly"].trigger
        assert isinstance(eval_trigger, IntervalTrigger)
        assert eval_trigger.interval.total_seconds() == 3600

        heartbeat_trigger = jobs["job_heartbeat"].trigger
        assert isinstance(heartbeat_trigger, IntervalTrigger)
        assert heartbeat_trigger.interval.total_seconds() == 60
    finally:
        worker.stop(wait=False)


def test_worker_heartbeat(db_session: Session) -> None:
    """Verify _update_heartbeat updates and persists SystemHeartbeat table in DB."""
    # Initial success pulse
    hb = _update_heartbeat(db_session, "job_test", success=True)
    assert hb.job_type == "job_test"
    assert hb.worker_pid is not None
    assert hb.last_pulse_ts is not None
    assert hb.last_success_ts is not None
    assert hb.consecutive_failures == 0
    assert hb.last_error is None

    # Query directly from DB
    persisted = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_test")
    ).first()
    assert persisted is not None
    assert persisted.job_type == "job_test"
    assert persisted.consecutive_failures == 0

    # Failure pulse
    hb_fail = _update_heartbeat(
        db_session, "job_test", success=False, error_msg="Simulated failure"
    )
    assert hb_fail.consecutive_failures == 1
    assert hb_fail.last_error == "Simulated failure"

    # Second failure increments failure count
    hb_fail2 = _update_heartbeat(
        db_session, "job_test", success=False, error_msg="Second failure"
    )
    assert hb_fail2.consecutive_failures == 2
    assert hb_fail2.last_error == "Second failure"

    # Subsequent success resets failure count and clears error
    hb_recovered = _update_heartbeat(db_session, "job_test", success=True)
    assert hb_recovered.consecutive_failures == 0
    assert hb_recovered.last_error is None


def test_worker_jobs_execution(temp_db: Engine, db_session: Session) -> None:
    """Verify individual job execution methods run and log heartbeat successfully."""
    _create_sample_ticker(db_session, "BTC-USD", asset_class="crypto")
    _create_sample_ticker(db_session, "AAPL", asset_class="equity")

    worker = WorkerScheduler(
        engine=temp_db,
        providers={"fake": FakeProvider()},
    )

    # 1. job_heartbeat
    worker.job_heartbeat()
    hb = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_heartbeat")
    ).first()
    assert hb is not None
    assert hb.consecutive_failures == 0

    # 2. job_ingest_crypto
    worker.job_ingest_crypto()
    hb_crypto = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_ingest_crypto")
    ).first()
    assert hb_crypto is not None
    assert hb_crypto.consecutive_failures == 0

    # 3. job_ingest_equities
    worker.job_ingest_equities()
    hb_equities = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_ingest_equities")
    ).first()
    assert hb_equities is not None
    assert hb_equities.consecutive_failures == 0

    # 4. job_retrain_nightly (no historical bars inserted yet, should gracefully skip/handle)
    worker.job_retrain_nightly()
    hb_retrain = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_retrain_nightly")
    ).first()
    assert hb_retrain is not None
    assert hb_retrain.consecutive_failures == 0

    # 5. job_evaluate_hourly
    worker.job_evaluate_hourly()
    hb_eval = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_evaluate_hourly")
    ).first()
    assert hb_eval is not None
    assert hb_eval.consecutive_failures == 0


def test_poll_cadence_defaults_are_end_of_day() -> None:
    """Defaults poll hourly, not sub-minute -- daily bars don't change faster."""
    s = Settings()
    assert s.poll_interval_crypto_sec == 3600
    assert s.poll_interval_equity_min == 60


def test_default_providers_include_keyless_set_and_gate_on_keys(
    temp_db: Engine,
) -> None:
    """yfinance + coinbase (keyless) + fake always register; keyed providers gate on their key."""
    no_keys = Settings(db_path=":memory:")
    w1 = WorkerScheduler(engine=temp_db, settings=no_keys)
    assert {"yfinance", "coinbase", "fake"}.issubset(w1.providers)
    assert "tiingo" not in w1.providers
    assert "finnhub" not in w1.providers
    assert "coingecko" not in w1.providers

    with_keys = Settings(
        db_path=":memory:",
        tiingo_api_key="tk",
        finnhub_api_key="fk",
        coingecko_api_key="ck",
    )
    w2 = WorkerScheduler(engine=temp_db, settings=with_keys)
    assert {"tiingo", "finnhub", "coingecko"}.issubset(w2.providers)


def test_worker_registers_derivatives_ingest_job(temp_db: Engine) -> None:
    """A job_ingest_derivatives must be scheduled so M9 crypto derivatives actually ingest."""
    worker = WorkerScheduler(engine=temp_db, settings=Settings(db_path=":memory:"))
    worker.start()
    try:
        job_ids = {job.id for job in worker.scheduler.get_jobs()}
        assert "job_ingest_derivatives" in job_ids
    finally:
        worker.stop(wait=False)


def test_worker_reconciles_crypto_ticker_to_keyless_primary(
    temp_db: Engine, db_session: Session
) -> None:
    """A crypto ticker left on coingecko (needs a key) is moved to coinbase (keyless)."""
    _create_sample_ticker(
        db_session, "BTC-USD", asset_class="crypto", provider="coingecko"
    )

    WorkerScheduler(engine=temp_db, settings=Settings(db_path=":memory:"))

    row = db_session.exec(select(Ticker).where(Ticker.symbol == "BTC-USD")).first()
    assert row is not None and row.provider == "coinbase"


def test_worker_ingest_records_failure_when_all_polls_error(
    temp_db: Engine, db_session: Session
) -> None:
    """A total ingest outage must surface as a heartbeat FAILURE, not success.

    poll_ticker returns an error dict (it never raises) when no provider is
    registered for the ticker. If the job ignores those dicts it records
    success=True and the watchdog stays green while data goes stale.
    """
    _create_sample_ticker(
        db_session, "BTC-USD", asset_class="crypto", provider="coingecko"
    )

    worker = WorkerScheduler(engine=temp_db, providers={"fake": FakeProvider()})
    worker.job_ingest_crypto()

    hb = db_session.exec(
        select(SystemHeartbeat).where(SystemHeartbeat.job_type == "job_ingest_crypto")
    ).first()
    assert hb is not None
    assert hb.consecutive_failures >= 1
    assert hb.last_error is not None and "BTC-USD" in hb.last_error


def test_worker_job_error_handling(temp_db: Engine, db_session: Session) -> None:
    """Verify worker jobs properly record failure status when exceptions occur."""
    worker = WorkerScheduler(engine=temp_db)

    with patch.object(
        worker,
        "_init_default_providers",
        side_effect=RuntimeError("Provider fault"),
    ):
        pass

    with patch(
        "stock_forecasting.worker.IngestionService",
        side_effect=RuntimeError("Ingestion crashed"),
    ):
        _create_sample_ticker(db_session, "ETH-USD", asset_class="crypto")
        worker.job_ingest_crypto()

        hb = db_session.exec(
            select(SystemHeartbeat).where(
                SystemHeartbeat.job_type == "job_ingest_crypto"
            )
        ).first()
        assert hb is not None
        assert hb.consecutive_failures >= 1
        assert "Ingestion crashed" in (hb.last_error or "")
