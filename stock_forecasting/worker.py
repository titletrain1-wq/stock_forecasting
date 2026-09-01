"""Background worker scheduler for scheduled ingestion, model retraining, and health monitoring."""

import logging
import os
from datetime import UTC, datetime

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import Engine
from sqlmodel import Session, select

from stock_forecasting.config import Settings, get_settings
from stock_forecasting.database import create_tables, get_engine
from stock_forecasting.forecaster import ForecastService
from stock_forecasting.ingestion import IngestionService
from stock_forecasting.providers.base import DataProvider, DerivativesProvider
from stock_forecasting.providers.coinbase import CoinbaseProvider
from stock_forecasting.providers.coingecko import CoinGeckoProvider
from stock_forecasting.providers.dydx import DydxDerivativesProvider
from stock_forecasting.providers.fake import FakeProvider
from stock_forecasting.providers.finnhub import FinnhubProvider
from stock_forecasting.providers.tiingo import TiingoProvider
from stock_forecasting.providers.yfinance import YFinanceProvider
from stock_forecasting.schema import SystemHeartbeat, Ticker
from stock_forecasting.trainer import Trainer

logger = logging.getLogger(__name__)

# Crypto tickers must default to a keyless provider so ingestion works out of
# the box. CoinGecko needs a Demo key (HTTP 401 without one); Coinbase is keyless.
_KEYLESS_CRYPTO_PRIMARY = "coinbase"


def _update_heartbeat(
    session: Session,
    job_type: str,
    success: bool = True,
    error_msg: str | None = None,
) -> SystemHeartbeat:
    """Upsert a SystemHeartbeat record for the given job_type.

    Args:
        session: Active SQLModel session.
        job_type: Identifier for the job or process.
        success: True if the execution was successful, False otherwise.
        error_msg: Optional error description if run failed.

    Returns:
        The created or updated SystemHeartbeat model instance.
    """
    heartbeat = session.get(SystemHeartbeat, job_type)
    now_iso = datetime.now(UTC).isoformat()
    pid = os.getpid()

    if heartbeat is None:
        heartbeat = SystemHeartbeat(
            job_type=job_type,
            worker_pid=pid,
            consecutive_failures=0,
        )

    heartbeat.worker_pid = pid
    heartbeat.last_pulse_ts = now_iso

    if success:
        heartbeat.last_success_ts = now_iso
        heartbeat.consecutive_failures = 0
        heartbeat.last_error = None
    else:
        heartbeat.consecutive_failures = (heartbeat.consecutive_failures or 0) + 1
        heartbeat.last_error = error_msg

    session.add(heartbeat)
    session.commit()
    session.refresh(heartbeat)
    return heartbeat


def _reconcile_ticker_providers(
    session: Session, providers: dict[str, DataProvider]
) -> None:
    """Point crypto tickers at a working keyless primary.

    A crypto ticker whose primary provider is not currently registered (the
    common case being ``coingecko`` with no Demo key) is moved to
    ``coinbase``, which is keyless and always available. Idempotent.
    """
    crypto = session.exec(select(Ticker).where(Ticker.asset_class == "crypto")).all()
    changed = False
    for ticker in crypto:
        if ticker.provider not in providers and _KEYLESS_CRYPTO_PRIMARY in providers:
            logger.info(
                "Reconciling ticker %s: provider %s -> %s",
                ticker.symbol,
                ticker.provider,
                _KEYLESS_CRYPTO_PRIMARY,
            )
            ticker.provider = _KEYLESS_CRYPTO_PRIMARY
            session.add(ticker)
            changed = True
    if changed:
        session.commit()


class WorkerScheduler:
    """Orchestrates scheduled tasks for stock_forecasting background processing."""

    def __init__(
        self,
        engine: Engine | None = None,
        settings: Settings | None = None,
        providers: dict[str, DataProvider] | None = None,
        scheduler: BackgroundScheduler | None = None,
    ) -> None:
        """Initialize the WorkerScheduler with database engine, settings, and providers.

        Args:
            engine: Optional SQLAlchemy Engine. If omitted, uses default engine from settings.
            settings: Optional Settings instance. If omitted, loads via get_settings().
            providers: Optional dictionary of DataProvider instances keyed by name.
            scheduler: Optional BackgroundScheduler instance.
        """
        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings.db_path)
        create_tables(self.engine)
        self.providers = providers or self._init_default_providers()
        self.derivatives_provider: DerivativesProvider = DydxDerivativesProvider()
        self.scheduler = scheduler or BackgroundScheduler()
        with Session(self.engine) as session:
            _reconcile_ticker_providers(session, self.providers)

    def _init_default_providers(self) -> dict[str, DataProvider]:
        """Build the provider map: keyless providers always, keyed providers gated on key.

        yfinance + coinbase are keyless and always registered. tiingo / finnhub /
        coingecko register only when their API key is configured. ``fake`` is kept
        for tests and offline demos.
        """
        providers: dict[str, DataProvider] = {
            "yfinance": YFinanceProvider(),
            "coinbase": CoinbaseProvider(),
            "fake": FakeProvider(),
        }
        if self.settings.tiingo_api_key:
            providers["tiingo"] = TiingoProvider(api_key=self.settings.tiingo_api_key)
        if self.settings.finnhub_api_key:
            providers["finnhub"] = FinnhubProvider(
                api_key=self.settings.finnhub_api_key
            )
        if self.settings.coingecko_api_key:
            providers["coingecko"] = CoinGeckoProvider(
                api_key=self.settings.coingecko_api_key
            )
        logger.info("Registered data providers: %s", ", ".join(sorted(providers)))
        return providers

    def _update_heartbeat(
        self,
        session: Session,
        job_type: str,
        success: bool = True,
        error_msg: str | None = None,
    ) -> SystemHeartbeat:
        """Update system heartbeat record."""
        return _update_heartbeat(
            session=session,
            job_type=job_type,
            success=success,
            error_msg=error_msg,
        )

    def _run_ingest_job(self, asset_class: str, job_type: str) -> None:
        """Poll every active ticker of ``asset_class`` and record an honest heartbeat.

        ``poll_ticker`` returns an error dict rather than raising, so a total
        outage would otherwise be logged as success. The heartbeat is recorded
        as a failure only when *every* ticker errored (a partial failure still
        counts as a live feed); the aggregated per-ticker errors become the
        heartbeat ``last_error``.
        """
        logger.info("Executing %s...", job_type)
        with Session(self.engine) as session:
            try:
                active = session.exec(
                    select(Ticker).where(
                        Ticker.active == 1,
                        Ticker.asset_class == asset_class,
                    )
                ).all()
                if not active:
                    _update_heartbeat(session, job_type, success=True)
                    return

                ingestion_service = IngestionService(session, self.providers)
                results = [
                    ingestion_service.poll_ticker(ticker.symbol) for ticker in active
                ]
                errored = [r for r in results if r.get("error")]

                if errored and len(errored) == len(results):
                    detail = "; ".join(
                        f"{r.get('symbol', '?')}: {r['error']}" for r in errored
                    )
                    _update_heartbeat(
                        session,
                        job_type,
                        success=False,
                        error_msg=f"all {len(results)} {asset_class} polls failed: {detail}",
                    )
                else:
                    _update_heartbeat(session, job_type, success=True)
            except Exception as exc:
                logger.exception("Error during %s", job_type)
                _update_heartbeat(
                    session,
                    job_type,
                    success=False,
                    error_msg=str(exc),
                )

    def job_ingest_crypto(self) -> None:
        """Poll and ingest latest market bars for all active crypto tickers."""
        self._run_ingest_job("crypto", "job_ingest_crypto")

    def job_ingest_equities(self) -> None:
        """Poll and ingest latest market bars for all active equity tickers."""
        self._run_ingest_job("equity", "job_ingest_equities")

    def job_ingest_derivatives(self) -> None:
        """Refresh crypto funding-rate + open-interest for all active crypto tickers."""
        job_type = "job_ingest_derivatives"
        logger.info("Executing %s...", job_type)
        with Session(self.engine) as session:
            try:
                svc = IngestionService(
                    session,
                    self.providers,
                    derivatives_provider=self.derivatives_provider,
                )
                results = svc.poll_all_derivatives()
                errored = [r for r in results.values() if r.get("error")]
                if results and len(errored) == len(results):
                    detail = "; ".join(
                        f"{r.get('symbol', '?')}: {r['error']}" for r in errored
                    )
                    _update_heartbeat(
                        session,
                        job_type,
                        success=False,
                        error_msg=f"all {len(results)} derivatives polls failed: {detail}",
                    )
                else:
                    _update_heartbeat(session, job_type, success=True)
            except Exception as exc:
                logger.exception("Error during %s", job_type)
                _update_heartbeat(session, job_type, success=False, error_msg=str(exc))

    def job_retrain_nightly(self) -> None:
        """Retrain predictive models for active tickers and generate updated forecasts."""
        logger.info("Executing job_retrain_nightly...")
        with Session(self.engine) as session:
            try:
                active_tickers = session.exec(
                    select(Ticker).where(Ticker.active == 1)
                ).all()
                trainer = Trainer(session)
                forecaster = ForecastService(session)
                horizons = ["1d", "5d", "30d"]
                model_types = ["ridge", "random_forest"]

                for ticker in active_tickers:
                    for horizon in horizons:
                        for model_type in model_types:
                            try:
                                trainer.train(
                                    ticker=ticker.symbol,
                                    horizon=horizon,
                                    model_type=model_type,
                                )
                            except Exception as train_exc:  # noqa: BLE001
                                logger.warning(
                                    "Nightly train skipped/failed for %s %s %s: %s",
                                    ticker.symbol,
                                    horizon,
                                    model_type,
                                    train_exc,
                                )
                    for horizon in horizons:
                        try:
                            forecaster.predict(ticker=ticker.symbol, horizon=horizon)
                        except Exception as pred_exc:  # noqa: BLE001
                            logger.warning(
                                "Nightly forecast skipped/failed for %s %s: %s",
                                ticker.symbol,
                                horizon,
                                pred_exc,
                            )
                _update_heartbeat(session, "job_retrain_nightly", success=True)
            except Exception as exc:
                logger.exception("Error during job_retrain_nightly")
                _update_heartbeat(
                    session,
                    "job_retrain_nightly",
                    success=False,
                    error_msg=str(exc),
                )

    def job_evaluate_hourly(self) -> None:
        """Evaluate matured prediction snapshots against realized prices."""
        logger.info("Executing job_evaluate_hourly...")
        with Session(self.engine) as session:
            try:
                try:
                    # Dynamically import EvaluatorService if available (Task 4.2)
                    from stock_forecasting.evaluator import (  # type: ignore[import-not-found]
                        EvaluatorService,
                    )

                    evaluator = EvaluatorService(session)
                    if hasattr(evaluator, "evaluate_matured"):
                        evaluator.evaluate_matured()
                    elif hasattr(evaluator, "run"):
                        evaluator.run()
                except (ImportError, AttributeError):
                    logger.debug(
                        "EvaluatorService not available or missing evaluation method; skipping."
                    )
                _update_heartbeat(session, "job_evaluate_hourly", success=True)
            except Exception as exc:
                logger.exception("Error during job_evaluate_hourly")
                _update_heartbeat(
                    session,
                    "job_evaluate_hourly",
                    success=False,
                    error_msg=str(exc),
                )

    def job_heartbeat(self) -> None:
        """Record a periodic watchdog pulse to indicate the worker scheduler is healthy."""
        with Session(self.engine) as session:
            try:
                _update_heartbeat(session, "job_heartbeat", success=True)
            except Exception as exc:
                logger.exception("Error during job_heartbeat")
                _update_heartbeat(
                    session,
                    "job_heartbeat",
                    success=False,
                    error_msg=str(exc),
                )

    def start(self) -> None:
        """Register all periodic jobs and start the BackgroundScheduler."""
        self.scheduler.add_job(
            self.job_ingest_crypto,
            "interval",
            seconds=self.settings.poll_interval_crypto_sec,
            id="job_ingest_crypto",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.job_ingest_equities,
            "interval",
            minutes=self.settings.poll_interval_equity_min,
            id="job_ingest_equities",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.job_ingest_derivatives,
            "interval",
            seconds=self.settings.poll_interval_crypto_sec,
            id="job_ingest_derivatives",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.job_retrain_nightly,
            "cron",
            hour=self.settings.retrain_hour_utc,
            id="job_retrain_nightly",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.job_evaluate_hourly,
            "interval",
            hours=1,
            id="job_evaluate_hourly",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.job_heartbeat,
            "interval",
            minutes=1,
            id="job_heartbeat",
            replace_existing=True,
        )

        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("WorkerScheduler started successfully.")

    def stop(self, wait: bool = False) -> None:
        """Shutdown the background scheduler if currently running."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("WorkerScheduler stopped.")


def main() -> None:
    """Run worker loop indefinitely until interrupted."""
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting stock_forecasting worker process...")
    worker = WorkerScheduler()
    worker.start()
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping worker process...")
        worker.stop()


if __name__ == "__main__":
    main()
