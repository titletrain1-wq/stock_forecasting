"""Application configuration and environment settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    watchlist: str = "AAPL,NVDA,SPY,BTC-USD,ETH-USD"
    poll_interval_crypto_sec: int = 60
    poll_interval_equity_min: int = 5
    backfill_years: int = 5
    db_path: str = "./data/app.db"
    retrain_hour_utc: int = 22
    tiingo_api_key: str = ""
    finnhub_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application Settings."""
    return Settings()
