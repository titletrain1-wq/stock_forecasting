"""Walk-forward backtest for evaluating forecasts on historical data.

Iterates through historical as-of dates, generates forecasts using the active model,
and grades against actual realized prices. Reuses EvaluatorService grading logic.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.forecaster import ForecastService
from stock_forecasting.schema import OhlcvBar

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest metrics aggregated per horizon."""

    ticker: str
    horizon: str
    model_type: str
    n: int  # number of graded forecasts
    mae: float  # mean absolute error (log-return)
    rmse: float  # root mean square error
    dir_acc: float  # directional accuracy
    ci_coverage: float  # fraction of realized within confidence interval
    mae_price_pct: float  # mean absolute error as % of realized price


def _grade_forecast(
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    predicted_return: float,
    anchor_price: float,
    realized_price: float,
) -> dict[str, float | int]:
    """Grade a single forecast against the realized price.

    Same math as ``EvaluatorService.run``: realized log-return is measured from
    the forecast's own anchor (the close at the as-of date) to the realized bar,
    then compared to the predicted log-return.

    Args:
        predicted_price: Model point prediction.
        lower_bound: 95% CI lower bound (price).
        upper_bound: 95% CI upper bound (price).
        predicted_return: Predicted log-return over the horizon.
        anchor_price: Close at the as-of date the forecast was made from.
        realized_price: Actual bar close at or after target_ts.

    Returns:
        Dict with error_abs, error_signed, is_direction_hit, is_within_ci, price_error_pct.
    """
    if anchor_price <= 0 or realized_price <= 0:
        realized_return = 0.0
    else:
        realized_return = float(np.log(realized_price / anchor_price))

    error_abs = float(abs(predicted_return - realized_return))
    error_signed = float(predicted_return - realized_return)

    is_direction_hit = 1 if np.sign(predicted_return) == np.sign(realized_return) else 0
    is_within_ci = 1 if (lower_bound <= realized_price <= upper_bound) else 0

    price_error_pct = (
        abs(predicted_price - realized_price) / realized_price
        if realized_price != 0
        else 0.0
    )

    return {
        "error_abs": error_abs,
        "error_signed": error_signed,
        "is_direction_hit": is_direction_hit,
        "is_within_ci": is_within_ci,
        "price_error_pct": price_error_pct,
    }


class BacktestService:
    """Walk-forward backtest for evaluating forecasts on historical data."""

    def __init__(self, session: Session, model_dir: str = "./model_store") -> None:
        """Initialize with database session and model artifact directory."""
        self.session = session
        self.bar_repo = BarRepository(session)
        self.forecaster = ForecastService(session, model_dir=model_dir)

    def run_backtest(
        self,
        ticker: str,
        horizons: tuple[str, ...] = ("1d", "5d", "30d"),
        lookback_days: int = 180,
        model_type: str = "ridge",
    ) -> dict[str, BacktestResult]:
        """Run walk-forward backtest for a ticker.

        For each historical as-of date T (last lookback_days):
        1. Load bars with ts <= T
        2. Generate forecast using active model
        3. Grade forecast against actual realized bar (ts >= target_ts)
        4. Aggregate metrics per horizon

        Args:
            ticker: Ticker symbol.
            horizons: Forecast horizons ('1d', '5d', '30d', etc).
            lookback_days: How many days back to test (~180 for 6 months).
            model_type: Model type to use ('ridge', 'random_forest', etc).

        Returns:
            Dict mapping horizon -> BacktestResult with n, mae, rmse, dir_acc, ci_coverage, mae_price_pct.

        Note:
            v1 uses the current active model (trained on full history).
            Mild train/test leakage is acceptable for a labelled backtest.
            Backtest results are isolated (not persisted to prediction_snapshots
            or accuracy_records).
        """
        results: dict[str, BacktestResult] = {}

        # Initialize graded forecasts per horizon
        graded_per_horizon: dict[str, list[dict]] = {h: [] for h in horizons}

        # Get all bars for this ticker (we'll iterate backwards)
        all_bars = self.bar_repo.get_latest(ticker, limit=999999)  # get all
        if not all_bars:
            logger.warning("No bars found for ticker %s", ticker)
            for h in horizons:
                results[h] = BacktestResult(
                    ticker=ticker,
                    horizon=h,
                    model_type=model_type,
                    n=0,
                    mae=0.0,
                    rmse=0.0,
                    dir_acc=0.0,
                    ci_coverage=0.0,
                    mae_price_pct=0.0,
                )
            return results

        all_bars_sorted = sorted(all_bars, key=lambda b: str(b.ts))

        # Determine backtest window: last lookback_days
        if len(all_bars_sorted) == 0:
            logger.warning("No bars available for backtest on %s", ticker)
            for h in horizons:
                results[h] = BacktestResult(
                    ticker=ticker,
                    horizon=h,
                    model_type=model_type,
                    n=0,
                    mae=0.0,
                    rmse=0.0,
                    dir_acc=0.0,
                    ci_coverage=0.0,
                    mae_price_pct=0.0,
                )
            return results

        def _parse_iso(ts_str: str) -> datetime:
            """Parse ISO string handling both Z and +00:00 formats."""
            if ts_str.endswith("Z"):
                return datetime.fromisoformat(ts_str[:-1] + "+00:00")  # noqa: FURB162
            return datetime.fromisoformat(ts_str)

        latest_ts = _parse_iso(all_bars_sorted[-1].ts)
        cutoff_ts = latest_ts - timedelta(days=lookback_days)

        # Filter bars in backtest window
        backtest_bars = [b for b in all_bars_sorted if _parse_iso(b.ts) >= cutoff_ts]

        if not backtest_bars:
            logger.warning("No bars in backtest window for %s", ticker)
            for h in horizons:
                results[h] = BacktestResult(
                    ticker=ticker,
                    horizon=h,
                    model_type=model_type,
                    n=0,
                    mae=0.0,
                    rmse=0.0,
                    dir_acc=0.0,
                    ci_coverage=0.0,
                    mae_price_pct=0.0,
                )
            return results

        # Walk-forward: for each bar in backtest window, generate forecast using
        # ONLY the bars at or before that date (as_of_ts cutoff, persist=False so
        # nothing touches prediction_snapshots / accuracy_records).
        for as_of_bar in backtest_bars[:-1]:  # don't forecast from the last bar
            as_of_ts = as_of_bar.ts

            # Step 1: Generate forecast with data up to as_of_ts (no lookahead)
            try:
                snapshot_results = self.forecaster.generate_and_persist(
                    ticker,
                    horizons=horizons,
                    model_types=(model_type,),
                    now=_parse_iso(as_of_ts),
                    as_of_ts=as_of_ts,
                    persist=False,
                )
            except (ValueError, FileNotFoundError) as e:
                logger.debug(
                    "Could not generate forecast as-of %s for %s: %s",
                    as_of_ts,
                    ticker,
                    e,
                )
                continue

            # Step 2: Grade each forecast against realized bar
            for horizon in horizons:
                key = f"{horizon}_{model_type}"
                if key not in snapshot_results:
                    continue

                result = snapshot_results[key]
                predicted_price = result.predicted_price
                lower_bound = result.lower_bound
                upper_bound = result.upper_bound
                predicted_return = result.predicted_return
                anchor_price = result.anchor_price
                target_ts = result.target_ts

                # Find realized bar at or after target_ts
                bar_stmt = (
                    select(OhlcvBar)
                    .where(
                        OhlcvBar.ticker == ticker,
                        OhlcvBar.ts >= target_ts,
                    )
                    .order_by(OhlcvBar.ts.asc())
                    .limit(1)
                )
                realized_bar = self.session.exec(bar_stmt).first()

                if realized_bar is None:
                    # No realized data yet; skip this forecast
                    continue

                if realized_bar.adj_close is not None and not np.isnan(
                    realized_bar.adj_close
                ):
                    realized_price = float(realized_bar.adj_close)
                else:
                    realized_price = float(realized_bar.close)

                # Skip if the "realized" bar is the anchor bar itself (no gap yet)
                if realized_bar.ts <= as_of_ts:
                    continue

                # Grade this forecast
                grade = _grade_forecast(
                    predicted_price=predicted_price,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    predicted_return=predicted_return,
                    anchor_price=anchor_price,
                    realized_price=realized_price,
                )
                graded_per_horizon[horizon].append(grade)

        # Step 3: Aggregate metrics per horizon
        for horizon in horizons:
            graded = graded_per_horizon[horizon]
            n = len(graded)

            if n == 0:
                results[horizon] = BacktestResult(
                    ticker=ticker,
                    horizon=horizon,
                    model_type=model_type,
                    n=0,
                    mae=0.0,
                    rmse=0.0,
                    dir_acc=0.0,
                    ci_coverage=0.0,
                    mae_price_pct=0.0,
                )
                continue

            abs_errors = [g["error_abs"] for g in graded]
            mae = float(np.mean(abs_errors))

            signed_errors = [g["error_signed"] for g in graded]
            rmse = float(np.sqrt(np.mean(np.square(signed_errors))))

            dir_hits = [g["is_direction_hit"] for g in graded]
            dir_acc = float(np.mean(dir_hits))

            ci_hits = [g["is_within_ci"] for g in graded]
            ci_coverage = float(np.mean(ci_hits))

            price_pct_errors = [g["price_error_pct"] for g in graded]
            mae_price_pct = float(np.mean(price_pct_errors))

            results[horizon] = BacktestResult(
                ticker=ticker,
                horizon=horizon,
                model_type=model_type,
                n=n,
                mae=mae,
                rmse=rmse,
                dir_acc=dir_acc,
                ci_coverage=ci_coverage,
                mae_price_pct=mae_price_pct,
            )

        logger.info(
            "Backtest complete for %s: %d horizons evaluated",
            ticker,
            len([r for r in results.values() if r.n > 0]),
        )
        return results
