"""Data ingestion service for polling active watchlist tickers and historical backfills."""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.providers.base import DataProvider
from stock_forecasting.schema import Ticker

logger = logging.getLogger(__name__)


class IngestionService:
    """Service orchestrating market data ingestion across configured providers."""

    def __init__(
        self,
        session: Session,
        providers: dict[str, DataProvider],
        bar_repo: BarRepository | None = None,
    ) -> None:
        """Initialize IngestionService with DB session and providers.

        Args:
            session: SQLModel database session.
            providers: Mapping of provider identifier strings to DataProvider instances.
            bar_repo: Optional BarRepository instance (defaults to creating one from session).
        """
        self.session = session
        self.providers = providers
        self.bar_repo = bar_repo if bar_repo is not None else BarRepository(session)

    def poll_watchlist(self) -> dict[str, dict[str, Any]]:
        """Poll the latest bars for all active tickers in the watchlist.

        Returns:
            Dictionary mapping symbol to result dictionary from poll_ticker.
        """
        statement = select(Ticker).where(Ticker.active == 1)
        active_tickers = self.session.exec(statement).all()
        results: dict[str, dict[str, Any]] = {}
        for ticker in active_tickers:
            results[ticker.symbol] = self.poll_ticker(ticker.symbol)
        return results

    def poll_ticker(self, symbol: str, lookback: int = 5) -> dict[str, Any]:
        """Fetch and upsert recent bars for a single ticker.

        Args:
            symbol: Ticker symbol to poll.
            lookback: Number of recent days/bars to fetch (default: 5).

        Returns:
            Dictionary with status, e.g. {"symbol": symbol, "inserted": inserted} or error details.
        """
        ticker = self.session.exec(
            select(Ticker).where(Ticker.symbol == symbol)
        ).first()
        if ticker is None:
            logger.warning("Cannot poll ticker '%s': not found in database", symbol)
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Ticker '{symbol}' not found",
            }

        provider = self.providers.get(ticker.provider)
        if provider is None:
            logger.warning(
                "Cannot poll ticker '%s': provider '%s' not registered",
                symbol,
                ticker.provider,
            )
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Provider '{ticker.provider}' not found",
            }

        provider_symbol = ticker.provider_symbol or symbol
        try:
            bars = provider.get_latest_bars(provider_symbol, lookback=lookback)
            inserted = self.bar_repo.upsert_bars(
                ticker=ticker.symbol,
                bars=bars,
                source=ticker.provider,
            )
            return {"symbol": symbol, "inserted": inserted}
        except Exception as exc:
            logger.exception("Failed to poll ticker '%s'", symbol)
            return {"symbol": symbol, "inserted": 0, "error": str(exc)}

    def backfill(self, symbol: str, years: int = 5) -> dict[str, Any]:
        """Fetch and upsert historical daily bars for a single ticker.

        Args:
            symbol: Ticker symbol to backfill.
            years: Number of years of history to retrieve (default: 5).

        Returns:
            Dictionary with status, e.g. {"symbol": symbol, "inserted": inserted} or error details.
        """
        ticker = self.session.exec(
            select(Ticker).where(Ticker.symbol == symbol)
        ).first()
        if ticker is None:
            logger.warning("Cannot backfill ticker '%s': not found in database", symbol)
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Ticker '{symbol}' not found",
            }

        provider = self.providers.get(ticker.provider)
        if provider is None:
            logger.warning(
                "Cannot backfill ticker '%s': provider '%s' not registered",
                symbol,
                ticker.provider,
            )
            return {
                "symbol": symbol,
                "inserted": 0,
                "error": f"Provider '{ticker.provider}' not found",
            }

        end_date = datetime.now(UTC).date()
        try:
            start_date = end_date.replace(year=end_date.year - years)
        except ValueError:
            # Handle leap year (Feb 29)
            start_date = end_date.replace(year=end_date.year - years, day=28)

        provider_symbol = ticker.provider_symbol or symbol
        try:
            bars = provider.get_daily_history(
                provider_symbol,
                start=start_date,
                end=end_date,
            )
            inserted = self.bar_repo.upsert_bars(
                ticker=ticker.symbol,
                bars=bars,
                source=ticker.provider,
            )
            return {"symbol": symbol, "inserted": inserted}
        except Exception as exc:
            logger.exception("Failed to backfill ticker '%s'", symbol)
            return {"symbol": symbol, "inserted": 0, "error": str(exc)}
