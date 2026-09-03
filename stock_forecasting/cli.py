"""Command-line interface for stock_forecasting background jobs.

Provides entry points for both scheduled (default) and once-only (serverless)
job execution.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def run_once() -> None:
    """Execute all background jobs once (no scheduler), then exit.

    Used by GitHub Actions cron job for serverless deployments
    (Streamlit Cloud, etc.).

    Job execution order:
    1. Ingest crypto OHLCV bars
    2. Ingest equity OHLCV bars
    3. Ingest equity intraday (5m bars)
    4. Ingest crypto derivatives (funding rates, OI)
    5. Retrain models nightly
    6. Evaluate matured predictions hourly
    7. Prune old intraday bars
    8. Check WebSocket idle and fallback

    Exits after completion (process should not restart indefinitely).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting one-time job execution (--once mode)")

    from stock_forecasting.worker import WorkerScheduler

    try:
        scheduler = WorkerScheduler()

        # Run each job synchronously (no BackgroundScheduler)
        logger.info("Running job_ingest_crypto...")
        scheduler.job_ingest_crypto()

        logger.info("Running job_ingest_equities...")
        scheduler.job_ingest_equities()

        logger.info("Running job_ingest_equity_intraday...")
        scheduler.job_ingest_equity_intraday()

        logger.info("Running job_ingest_derivatives...")
        scheduler.job_ingest_derivatives()

        logger.info("Running job_retrain_nightly...")
        scheduler.job_retrain_nightly()

        logger.info("Running job_evaluate_hourly...")
        scheduler.job_evaluate_hourly()

        logger.info("Running job_prune_intraday...")
        scheduler.job_prune_intraday()

        logger.info("Running job_check_ws_idle...")
        scheduler.job_check_ws_idle()

        logger.info("All jobs completed successfully.")
        sys.exit(0)

    except Exception:
        logger.exception("Fatal error during job execution")
        sys.exit(1)


def run_scheduler() -> None:
    """Run the background job scheduler indefinitely (default mode).

    Starts BackgroundScheduler with all periodic jobs. Runs until
    interrupted (Ctrl+C).
    """
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from stock_forecasting.worker import WorkerScheduler

    logger.info("Starting background scheduler (scheduled mode)")

    worker = WorkerScheduler()
    worker.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
        worker.stop(wait=True)
        sys.exit(0)


def main() -> None:
    """CLI entry point: route to run_once or run_scheduler."""
    if "--once" in sys.argv or "run-once" in sys.argv:
        run_once()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
