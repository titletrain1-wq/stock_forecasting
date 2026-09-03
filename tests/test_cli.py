"""Tests for stock_forecasting CLI module."""

from unittest.mock import MagicMock, patch


def test_run_once_executes_all_jobs() -> None:
    """Verify run_once() calls each job method in order."""
    with patch("stock_forecasting.worker.WorkerScheduler") as mock_scheduler_class:
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler

        with patch("stock_forecasting.cli.sys.exit") as mock_exit:
            from stock_forecasting.cli import run_once

            run_once()

            # Verify all job methods were called
            mock_scheduler.job_ingest_crypto.assert_called_once()
            mock_scheduler.job_ingest_equities.assert_called_once()
            mock_scheduler.job_ingest_equity_intraday.assert_called_once()
            mock_scheduler.job_ingest_derivatives.assert_called_once()
            mock_scheduler.job_retrain_nightly.assert_called_once()
            mock_scheduler.job_evaluate_hourly.assert_called_once()
            mock_scheduler.job_prune_intraday.assert_called_once()
            mock_scheduler.job_check_ws_idle.assert_called_once()

            # Verify clean exit
            mock_exit.assert_called_once_with(0)


def test_run_once_exits_on_error() -> None:
    """Verify run_once() exits with code 1 on exception."""
    with patch("stock_forecasting.worker.WorkerScheduler") as mock_scheduler_class:
        mock_scheduler = MagicMock()
        mock_scheduler.job_ingest_crypto.side_effect = RuntimeError("Test error")
        mock_scheduler_class.return_value = mock_scheduler

        with patch("stock_forecasting.cli.sys.exit") as mock_exit:
            from stock_forecasting.cli import run_once

            run_once()

            # Verify exit with error code
            mock_exit.assert_called_once_with(1)


def test_run_scheduler_starts_and_handles_interrupt() -> None:
    """Verify run_scheduler() starts scheduler and stops on KeyboardInterrupt."""
    with patch("stock_forecasting.worker.WorkerScheduler") as mock_scheduler_class:
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler

        with (
            patch("time.sleep") as mock_sleep,
            patch("stock_forecasting.cli.sys.exit") as mock_exit,
        ):
            # Raise KeyboardInterrupt on second call to sleep
            mock_sleep.side_effect = [None, KeyboardInterrupt()]

            from stock_forecasting.cli import run_scheduler

            run_scheduler()

            # Verify scheduler started and stopped
            mock_scheduler.start.assert_called_once()
            mock_scheduler.stop.assert_called_once_with(wait=True)
            mock_exit.assert_called_once_with(0)


def test_main_routes_to_run_once_with_flag() -> None:
    """Verify main() routes to run_once() when --once flag is present."""
    with (
        patch("stock_forecasting.cli.run_once") as mock_run_once,
        patch("sys.argv", ["cli.py", "--once"]),
    ):
        from stock_forecasting.cli import main

        main()

        mock_run_once.assert_called_once()


def test_main_routes_to_run_scheduler_without_flag() -> None:
    """Verify main() routes to run_scheduler() when no --once flag."""
    with (
        patch("stock_forecasting.cli.run_scheduler") as mock_run_scheduler,
        patch("sys.argv", ["cli.py"]),
    ):
        from stock_forecasting.cli import main

        main()

        mock_run_scheduler.assert_called_once()
