"""Tests for yfinance 5m equity intraday poller and WorkerScheduler integration."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlmodel import Session, select

from stock_forecasting.providers.base import Bar, ProviderError
from stock_forecasting.providers.yfinance import YFinanceProvider
from stock_forecasting.schema import IntradayBar, SystemHeartbeat, Ticker
from stock_forecasting.worker import WorkerScheduler


@pytest.fixture
def sample_5m_dataframe():
    """Create a sample 5m yfinance DataFrame."""
    now = datetime.now(UTC)
    times = [now - timedelta(minutes=5 * i) for i in range(12, 0, -1)]
    data = {
        "Open": [150.0 + i for i in range(12)],
        "High": [152.0 + i for i in range(12)],
        "Low": [149.0 + i for i in range(12)],
        "Close": [151.0 + i for i in range(12)],
        "Volume": [1000.0 * (i + 1) for i in range(12)],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(times, tz="UTC"))


def test_yfinance_get_intraday_bars_parsed_and_sorted(sample_5m_dataframe) -> None:
    """Verify YFinanceProvider.get_intraday_bars parses and sorts closed 5m bars."""
    provider = YFinanceProvider()
    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_5m_dataframe
        mock_ticker_cls.return_value = mock_instance

        bars = provider.get_intraday_bars("AAPL", interval="5m", lookback_bars=12)

        assert len(bars) == 12
        assert all(isinstance(b, Bar) for b in bars)
        # Oldest first, newest last
        assert bars[0].ts < bars[-1].ts
        assert bars[-1].close == 162.0


def test_job_ingest_equity_intraday_provisional_window(temp_db) -> None:
    """Verify provisional window logic: bar within 20m (5m + 15m delay) is marked provisional=1."""
    now = datetime.now(UTC)
    bar_old_ts = (now - timedelta(minutes=30)).isoformat()
    bar_recent_ts = (now - timedelta(minutes=3)).isoformat()

    mock_bars = [
        Bar(
            ts=bar_old_ts,
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            adj_close=150.5,
            volume=1000.0,
        ),
        Bar(
            ts=bar_recent_ts,
            open=150.5,
            high=152.0,
            low=150.0,
            close=151.5,
            adj_close=151.5,
            volume=1200.0,
        ),
    ]

    mock_provider = MagicMock()
    mock_provider.get_intraday_bars.return_value = mock_bars

    scheduler = WorkerScheduler(engine=temp_db, providers={"yfinance": mock_provider})

    # Add active equity ticker
    with Session(temp_db) as session:
        t = Ticker(
            symbol="AAPL",
            asset_class="equity",
            display_name="Apple",
            provider="yfinance",
            provider_symbol="AAPL",
            price_basis="adjusted",
            added_at=now.isoformat(),
            active=1,
        )

        session.add(t)
        session.commit()

    scheduler.job_ingest_equity_intraday()

    with Session(temp_db) as session:
        rows = session.exec(
            select(IntradayBar)
            .where(IntradayBar.ticker == "AAPL")
            .order_by(IntradayBar.ts)
        ).all()
        assert len(rows) == 2
        # Older bar (>20m ago) should be provisional=0
        assert rows[0].is_provisional == 0
        # Recent bar (<20m ago) should be provisional=1
        assert rows[1].is_provisional == 1

        # Check heartbeat
        hb = session.exec(
            select(SystemHeartbeat).where(
                SystemHeartbeat.job_type == "job_ingest_equity_intraday"
            )
        ).first()
        assert hb is not None
        assert hb.consecutive_failures == 0


def test_job_ingest_equity_intraday_429_backoff_and_error_handling(temp_db) -> None:
    """Verify ProviderError(429) rate limit handles retries/errors gracefully without crashing."""
    mock_provider = MagicMock()
    mock_provider.get_intraday_bars.side_effect = ProviderError(
        "HTTP 429: Rate limited"
    )

    scheduler = WorkerScheduler(engine=temp_db, providers={"yfinance": mock_provider})

    now = datetime.now(UTC).isoformat()
    with Session(temp_db) as session:
        t = Ticker(
            symbol="AAPL",
            asset_class="equity",
            display_name="Apple",
            provider="yfinance",
            provider_symbol="AAPL",
            price_basis="adjusted",
            added_at=now,
            active=1,
        )

        session.add(t)
        session.commit()

    # Job should handle error without crashing
    scheduler.job_ingest_equity_intraday()

    with Session(temp_db) as session:
        hb = session.exec(
            select(SystemHeartbeat).where(
                SystemHeartbeat.job_type == "job_ingest_equity_intraday"
            )
        ).first()
        assert hb is not None
        assert hb.consecutive_failures > 0
        assert "429" in (hb.last_error or "")
