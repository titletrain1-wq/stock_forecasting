"""Unit tests for application configuration settings."""

from stock_forecasting.config import Settings


def test_realtime_settings_defaults() -> None:
    """v2: real-time display layer settings have sane keyless-friendly defaults."""
    s = Settings(_env_file=None)
    assert s.live_ws_enabled is True
    assert s.coinbase_ws_url.startswith("wss://")
    assert s.ws_idle_timeout_sec == 90
    assert s.intraday_equity_interval == "5m"
    assert s.intraday_poll_equity_min == 5
    assert s.intraday_retention_days == 7
    assert s.live_fragment_refresh_crypto_sec == 2
    assert s.live_fragment_refresh_equity_sec == 15


def test_realtime_settings_env_override(monkeypatch) -> None:
    """v2: settings load from the environment like the rest of the config."""
    monkeypatch.setenv("LIVE_WS_ENABLED", "false")
    monkeypatch.setenv("INTRADAY_RETENTION_DAYS", "14")
    s = Settings(_env_file=None)
    assert s.live_ws_enabled is False
    assert s.intraday_retention_days == 14
