"""Data ingestion service for polling active watchlist tickers and historical backfills.

`poll_ticker` / `backfill` fetch through a per-provider circuit breaker (M5) and
fail over to a fallback provider when the primary's breaker is open or its call
raises. `source` on each stored bar records the provider that actually served it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenException,
)
from stock_forecasting.providers.base import Bar, DataProvider
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
    ) -> None:
        """Initialize IngestionService with DB session and providers.

        Args:
            session: SQLModel database session.
            providers: Mapping of provider identifier strings to DataProvider instances.
            bar_repo: Optional BarRepository (defaults to one built from ``session``).
            circuit_breaker: Optional CircuitBreaker (defaults to one built from ``session``).
        """
        self.session = session
        self.providers = providers
        self.bar_repo = bar_repo if bar_repo is not None else BarRepository(session)
        self.circuit_breaker = (
            circuit_breaker if circuit_breaker is not None else CircuitBreaker(session)
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
