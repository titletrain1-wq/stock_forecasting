"""Data ingestion service for polling active watchlist tickers and historical backfills.

`poll_ticker` / `backfill` fetch through a per-provider circuit breaker (M5) and
fail over to a fallback provider when the primary's breaker is open or its call
raises. `source` on each stored bar records the provider that actually served it.

`poll_derivatives` / `backfill_derivatives` ingest crypto funding rate + open
interest (dYdX) into `crypto_derivatives` for active crypto tickers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenException,
)
from stock_forecasting.crypto_store import CryptoDerivativeStore
from stock_forecasting.providers.base import Bar, DataProvider, DerivativesProvider
from stock_forecasting.schema import Ticker

logger = logging.getLogger(__name__)

# Primary provider id -> ordered fallback provider ids (spec §8).
FALLBACK_PROVIDERS: dict[str, list[str]] = {
    "yfinance": ["tiingo", "finnhub"],
    "yahoo": ["tiingo", "finnhub"],
    "tiingo": ["finnhub"],
    "coingecko": ["coinbase"],
    "coinbase": ["coingecko"],
}


class IngestionService:
    """Service orchestrating market data ingestion across configured providers."""

    def __init__(
        self,
        session: Session,
        providers: dict[str, DataProvider],
        bar_repo: BarRepository | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        derivatives_provider: DerivativesProvider | None = None,
        crypto_store: CryptoDerivativeStore | None = None,
    ) -> None:
        """Initialize IngestionService with DB session and providers.

        Args:
            session: SQLModel database session.
            providers: Mapping of provider identifier strings to DataProvider instances.
            bar_repo: Optional BarRepository (defaults to one built from ``session``).
            circuit_breaker: Optional CircuitBreaker (defaults to one built from ``session``).
            derivatives_provider: Optional crypto derivatives source (e.g. dYdX).
            crypto_store: Optional CryptoDerivativeStore (defaults to one from ``session``).
        """
        self.session = session
        self.providers = providers
        self.bar_repo = bar_repo if bar_repo is not None else BarRepository(session)
        self.circuit_breaker = (
            circuit_breaker if circuit_breaker is not None else CircuitBreaker(session)
        )
        self.derivatives_provider = derivatives_provider
        self.crypto_store = (
            crypto_store if crypto_store is not None else CryptoDerivativeStore(session)
        )

    def poll_watchlist(self) -> dict[str, dict[str, Any]]:
        """Poll the latest bars for all active tickers in the watchlist."""
        statement = select(Ticker).where(Ticker.active == 1)
        active_tickers = self.session.exec(statement).all()
        return {t.symbol: self.poll_ticker(t.symbol) for t in active_tickers}

    def _provider_chain(self, primary: str) -> list[str]:
        """Primary provider id followed by its registered fallbacks."""
        chain = [primary, *FALLBACK_PROVIDERS.get(primary, [])]
        seen: set[str] = set()
        ordered: list[str] = []
        for pid in chain:
            if pid not in seen and pid in self.providers:
                ordered.append(pid)
                seen.add(pid)
        return ordered

    def _fetch_with_failover(
        self,
        ticker: Ticker,
        fetch: Callable[[DataProvider, str], list[Bar]],
    ) -> dict[str, Any]:
        """Run ``fetch`` against the primary provider, failing over on an open
        breaker or a raised error. Records success/failure in the breaker.
        """
        chain = self._provider_chain(ticker.provider)
        if not chain:
            logger.warning(
                "No registered provider for ticker '%s' (primary '%s')",
                ticker.symbol,
                ticker.provider,
            )
            return {
                "symbol": ticker.symbol,
                "inserted": 0,
                "error": f"Provider '{ticker.provider}' not found",
            }

        provider_symbol = ticker.provider_symbol or ticker.symbol
        last_error: str | None = None

        for pid in chain:
            try:
                with self.circuit_breaker.guard(pid):
                    bars = fetch(self.providers[pid], provider_symbol)
                    inserted = self.bar_repo.upsert_bars(
                        ticker=ticker.symbol, bars=bars, source=pid
                    )
            except CircuitBreakerOpenException as exc:
                logger.warning("Skipping provider '%s': %s", pid, exc)
                last_error = str(exc)
                continue
            except Exception as exc:  # any provider failure triggers failover
                logger.exception(
                    "Provider '%s' failed for ticker '%s'", pid, ticker.symbol
                )
                last_error = str(exc)
                continue

            result: dict[str, Any] = {
                "symbol": ticker.symbol,
                "inserted": inserted,
                "provider": pid,
            }
            if pid != ticker.provider:
                result["failover_from"] = ticker.provider
            return result

        return {
            "symbol": ticker.symbol,
            "inserted": 0,
            "error": last_error or "all providers unavailable",
        }

    def _get_ticker(self, symbol: str) -> Ticker | None:
        return self.session.exec(select(Ticker).where(Ticker.symbol == symbol)).first()

    def poll_ticker(self, symbol: str, lookback: int = 5) -> dict[str, Any]:
        """Fetch and upsert recent bars for a single ticker, with failover.

        Returns a dict with ``symbol`` and ``inserted`` on success (plus ``provider``
        and, on failover, ``failover_from``), or ``error`` on failure.
        """
        ticker = self._get_ticker(symbol)
        if ticker is None:
            logger.warning("Cannot poll ticker '%s': not found in database", symbol)
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Ticker '{symbol}' not found",
            }

        return self._fetch_with_failover(
            ticker,
            lambda provider, psym: provider.get_latest_bars(psym, lookback=lookback),
        )

    def backfill(self, symbol: str, years: int = 5) -> dict[str, Any]:
        """Fetch and upsert historical daily bars for a single ticker, with failover."""
        ticker = self._get_ticker(symbol)
        if ticker is None:
            logger.warning("Cannot backfill ticker '%s': not found in database", symbol)
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Ticker '{symbol}' not found",
            }

        end_date = datetime.now(UTC).date()
        try:
            start_date = end_date.replace(year=end_date.year - years)
        except ValueError:  # Feb 29 in a non-leap target year
            start_date = end_date.replace(year=end_date.year - years, day=28)

        return self._fetch_with_failover(
            ticker,
            lambda provider, psym: provider.get_daily_history(
                psym, start=start_date, end=end_date
            ),
        )

    # ---- crypto derivatives (funding rate + open interest) -----------------

    def _ingest_derivatives(
        self, ticker: Ticker, start: date, end: date
    ) -> dict[str, Any]:
        if self.derivatives_provider is None:
            return {
                "symbol": ticker.symbol,
                "inserted": 0,
                "error": "no derivatives provider configured",
            }
        if ticker.asset_class != "crypto":
            return {
                "symbol": ticker.symbol,
                "inserted": 0,
                "error": "not a crypto ticker",
            }
        provider_symbol = ticker.provider_symbol or ticker.symbol
        try:
            rows = self.derivatives_provider.get_derivatives(
                provider_symbol, start=start, end=end
            )
        except Exception as exc:  # derivatives are non-critical; never break bar ingest
            logger.exception("Derivatives fetch failed for '%s'", ticker.symbol)
            return {"symbol": ticker.symbol, "inserted": 0, "error": str(exc)}

        inserted = self.crypto_store.upsert(ticker.symbol, rows, source="dydx")
        return {"symbol": ticker.symbol, "inserted": inserted, "rows": len(rows)}

    def poll_derivatives(self, symbol: str, lookback_days: int = 45) -> dict[str, Any]:
        """Refresh recent crypto derivatives for one ticker."""
        ticker = self._get_ticker(symbol)
        if ticker is None:
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Ticker '{symbol}' not found",
            }
        end = datetime.now(UTC).date()
        return self._ingest_derivatives(
            ticker, end - timedelta(days=lookback_days), end
        )

    def backfill_derivatives(self, symbol: str, years: int = 3) -> dict[str, Any]:
        """Backfill crypto derivatives history for one ticker."""
        ticker = self._get_ticker(symbol)
        if ticker is None:
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Ticker '{symbol}' not found",
            }
        end = datetime.now(UTC).date()
        try:
            start = end.replace(year=end.year - years)
        except ValueError:
            start = end.replace(year=end.year - years, day=28)
        return self._ingest_derivatives(ticker, start, end)

    def poll_all_derivatives(
        self, lookback_days: int = 45
    ) -> dict[str, dict[str, Any]]:
        """Refresh recent derivatives for every active crypto ticker."""
        if self.derivatives_provider is None:
            return {}
        crypto = self.session.exec(
            select(Ticker).where(Ticker.active == 1, Ticker.asset_class == "crypto")
        ).all()
        end = datetime.now(UTC).date()
        start = end - timedelta(days=lookback_days)
        return {t.symbol: self._ingest_derivatives(t, start, end) for t in crypto}
