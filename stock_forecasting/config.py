"""Application configuration and environment settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    watchlist: str = "AAPL,NVDA,SPY,BTC-USD,ETH-USD"
    # This is an end-of-day system: providers serve daily OHLCV bars, so polling
    # is hourly, not sub-minute. Hourly is frequent enough to pick up a new daily
    # bar (and an intraday forming crypto candle) shortly after it appears without
    # burning the free-tier request budget re-fetching an unchanged bar.
    poll_interval_crypto_sec: int = 3600
    poll_interval_equity_min: int = 60
    backfill_years: int = 5
    db_path: str = "./data/app.db"
    retrain_hour_utc: int = 22
    tiingo_api_key: str = ""
    finnhub_api_key: str = ""
    coingecko_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application Settings."""
    return Settings()
