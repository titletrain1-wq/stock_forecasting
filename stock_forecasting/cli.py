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
    except Exception:
        logger.exception("Could not construct WorkerScheduler - aborting")
        sys.exit(1)

    try:
        from stock_forecasting.database import seed_watchlist

        n = seed_watchlist(scheduler.engine)
        logger.info("Watchlist seed: %d ticker(s) added", n)
    except Exception:
        logger.exception("Watchlist seed failed (continuing)")

    # Run each job independently: a single provider/data failure must not
    # abort the whole pass or block the heartbeat. Serverless-friendly.
    jobs = [
        ("job_ingest_crypto", scheduler.job_ingest_crypto),
        ("job_ingest_equities", scheduler.job_ingest_equities),
        ("job_ingest_equity_intraday", scheduler.job_ingest_equity_intraday),
        ("job_ingest_derivatives", scheduler.job_ingest_derivatives),
        ("job_retrain_nightly", scheduler.job_retrain_nightly),
        ("job_evaluate_hourly", scheduler.job_evaluate_hourly),
        ("job_prune_intraday", scheduler.job_prune_intraday),
        ("job_check_ws_idle", scheduler.job_check_ws_idle),
        ("job_heartbeat", scheduler.job_heartbeat),
    ]
    failures: list[str] = []
    for name, fn in jobs:
        logger.info("Running %s...", name)
        try:
            fn()
        except Exception:
            logger.exception("%s failed", name)
            failures.append(name)

    if failures:
        logger.warning("Completed with %d job failure(s): %s", len(failures), failures)
    else:
        logger.info("All jobs completed successfully.")
    # Exit non-zero only if every job failed (total outage worth alerting on).
    sys.exit(1 if len(failures) == len(jobs) else 0)


def run_backfill(years: int = 2) -> None:
    """One-off: fetch `years` of daily history for every active ticker, then
    retrain + forecast so the chart has a forecast line and the accuracy panel
    has something to grade. The hourly `--once` pass only polls recent bars.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting historical backfill (%d years)", years)

    from sqlmodel import Session, select

    from stock_forecasting.database import seed_watchlist
    from stock_forecasting.ingestion import IngestionService
    from stock_forecasting.schema import Ticker
    from stock_forecasting.worker import WorkerScheduler

    try:
        scheduler = WorkerScheduler()
        seed_watchlist(scheduler.engine)
    except Exception:
        logger.exception("Could not construct WorkerScheduler - aborting")
        sys.exit(1)

    with Session(scheduler.engine) as session:
        symbols = [
            t.symbol
            for t in session.exec(select(Ticker).where(Ticker.active == 1)).all()
        ]
        for sym in symbols:
            # Fresh session per ticker: a failure on one must not poison the
            # rest (SQLAlchemy leaves the session in a rolled-back state).
            with Session(scheduler.engine) as tsession:
                svc = IngestionService(tsession, scheduler.providers)
                try:
                    res = svc.backfill(sym, years=years)
                    tsession.commit()
                    logger.info("backfill %s -> %s", sym, res)
                except Exception:
                    tsession.rollback()
                    logger.exception("backfill %s failed", sym)

    for name, fn in (
        ("job_retrain_nightly", scheduler.job_retrain_nightly),
        ("job_evaluate_hourly", scheduler.job_evaluate_hourly),
        ("job_heartbeat", scheduler.job_heartbeat),
    ):
        logger.info("Running %s...", name)
        try:
            fn()
        except Exception:
            logger.exception("%s failed", name)

    logger.info("Backfill complete.")
    sys.exit(0)


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


def run_backtest(ticker: str | None = None, days: int = 180) -> None:
    """Run walk-forward backtest for a ticker and print results.

    Args:
        ticker: Ticker symbol (e.g. 'BTC-USD'). If None, use first active ticker.
        days: Lookback window in days (default 180).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting backtest (ticker=%s, days=%d)", ticker, days)

    from sqlmodel import Session, select

    from stock_forecasting.backtest import BacktestService
    from stock_forecasting.database import seed_watchlist
    from stock_forecasting.schema import Ticker
    from stock_forecasting.worker import WorkerScheduler

    try:
        scheduler = WorkerScheduler()
        seed_watchlist(scheduler.engine)
    except Exception:
        logger.exception("Could not construct WorkerScheduler - aborting")
        sys.exit(1)

    with Session(scheduler.engine) as session:
        # Determine target ticker
        if ticker is None:
            # Use first active ticker
            t = session.exec(select(Ticker).where(Ticker.active == 1)).first()
            if not t:
                logger.error("No active tickers found")
                sys.exit(1)
            ticker = t.symbol

        # Run backtest
        service = BacktestService(session)
        results = service.run_backtest(
            ticker=ticker,
            horizons=("1d", "5d", "30d"),
            lookback_days=days,
            model_type="ridge",
        )

    # Print results table
    print(f"\nBacktest Results: {ticker} (lookback={days}d)")
    print("=" * 80)
    print(f"{'Horizon':<10} {'n':<5} {'MAE':<10} {'RMSE':<10} {'Dir %':<8} {'CI %':<8}")
    print("-" * 80)
    for h in ("1d", "5d", "30d"):
        if h in results:
            r = results[h]
            print(
                f"{h:<10} {r.n:<5} {r.mae:<10.6f} {r.rmse:<10.6f} "
                f"{r.dir_acc * 100:<8.1f} {r.ci_coverage * 100:<8.1f}"
            )
    print("=" * 80)
    sys.exit(0)


def main() -> None:
    """CLI entry point: route to backfill, backtest, run_once, or run_scheduler."""
    if "--backtest" in sys.argv:
        ticker = None
        days = 180
        if "--ticker" in sys.argv:
            try:
                ticker = sys.argv[sys.argv.index("--ticker") + 1]
            except IndexError:
                pass
        if "--days" in sys.argv:
            try:
                days = int(sys.argv[sys.argv.index("--days") + 1])
            except (IndexError, ValueError):
                logger.warning("Bad --days value, defaulting to %d", days)
        run_backtest(ticker, days)
    elif "--backfill" in sys.argv:
        years = 2
        if "--years" in sys.argv:
            try:
                years = int(sys.argv[sys.argv.index("--years") + 1])
            except (IndexError, ValueError):
                logger.warning("Bad --years value, defaulting to %d", years)
        run_backfill(years)
    elif "--once" in sys.argv or "run-once" in sys.argv:
        run_once()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
