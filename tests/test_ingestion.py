"""Tests for BarRepository and YFinanceProvider ingestion workflows."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.providers.base import Bar
from stock_forecasting.providers.yfinance import YFinanceProvider
from stock_forecasting.schema import OhlcvBar, QuarantineBar, Ticker


def _create_sample_ticker(session: Session, symbol: str = "AAPL") -> Ticker:
    """Helper to insert a sample ticker for FK constraints."""
    ticker = Ticker(
        symbol=symbol,
        asset_class="equity",
        display_name="Apple Inc.",
        provider="yfinance",
        provider_symbol=symbol,
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    session.add(ticker)
    session.commit()
    return ticker


def test_bar_repository_upsert(db_session: Session) -> None:
    """Verify BarRepository inserts valid bars and handles idempotent upserts."""
    _create_sample_ticker(db_session, "AAPL")
    repo = BarRepository(db_session)

    bars = [
        Bar(
            ts="2026-01-01T00:00:00Z",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            adj_close=103.5,
            volume=1000.0,
        ),
        Bar(
            ts="2026-01-02T00:00:00Z",
            open=104.0,
            high=108.0,
            low=103.0,
            close=107.0,
            adj_close=106.5,
            volume=1500.0,
        ),
        Bar(
            ts="2026-01-03T00:00:00Z",
            open=107.0,
            high=110.0,
            low=106.0,
            close=109.0,
            adj_close=108.5,
            volume=1200.0,
        ),
    ]

    # First insert: all 3 should be newly inserted
    inserted = repo.upsert_bars("AAPL", bars, source="yfinance")
    assert inserted == 3

    # Query range
    range_bars = repo.get_range(
        "AAPL", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"
    )
    assert len(range_bars) == 3
    assert range_bars[0].ts == "2026-01-01T00:00:00Z"
    assert range_bars[2].ts == "2026-01-03T00:00:00Z"
    assert range_bars[0].close == 104.0
    assert range_bars[0].adj_close == 103.5
    assert range_bars[0].source == "yfinance"

    # Query latest
    latest_bars = repo.get_latest("AAPL", limit=2)
    assert len(latest_bars) == 2
    assert latest_bars[0].ts == "2026-01-03T00:00:00Z"
    assert latest_bars[1].ts == "2026-01-02T00:00:00Z"

    # Query latest timestamp
    assert repo.latest_ts("AAPL") == "2026-01-03T00:00:00Z"

    # Re-upsert identical bars: 0 new rows
    re_inserted = repo.upsert_bars("AAPL", bars, source="yfinance")
    assert re_inserted == 0

    # Re-upsert modified bar: update values without creating duplicates
    updated_bars = [
        Bar(
            ts="2026-01-03T00:00:00Z",
            open=107.0,
            high=115.0,
            low=106.0,
            close=114.0,
            adj_close=113.5,
            volume=2000.0,
        )
    ]
    upsert_count = repo.upsert_bars("AAPL", updated_bars, source="yfinance")
    assert upsert_count == 0

    all_bars = repo.get_range(
        "AAPL", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"
    )
    assert len(all_bars) == 3
    assert all_bars[2].close == 114.0
    assert all_bars[2].high == 115.0


def test_bar_repository_quarantine_invalid(db_session: Session) -> None:
    """Verify BarRepository filters malformed/anomaly bars to quarantine_bars."""
    _create_sample_ticker(db_session, "AAPL")
    repo = BarRepository(db_session)

    valid_bar = Bar(
        ts="2026-01-01T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        adj_close=102.0,
        volume=1000.0,
    )
    price_le_0_bar = Bar(
        ts="2026-01-02T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=-1.0,
        adj_close=None,
        volume=1000.0,
    )
    ohlc_inconsistent_bar = Bar(
        ts="2026-01-03T00:00:00Z",
        open=100.0,
        high=90.0,
        low=110.0,  # low > high
        close=95.0,
        adj_close=None,
        volume=1000.0,
    )
    neg_vol_bar = Bar(
        ts="2026-01-04T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        adj_close=None,
        volume=-50.0,
    )
    multi_fail_bar = Bar(
        ts="2026-01-05T00:00:00Z",
        open=100.0,
        high=80.0,
        low=120.0,
        close=0.0,  # price <= 0 and low > high and negative volume
        adj_close=None,
        volume=-10.0,
    )

    test_bars = [
        valid_bar,
        price_le_0_bar,
        ohlc_inconsistent_bar,
        neg_vol_bar,
        multi_fail_bar,
    ]

    inserted = repo.upsert_bars("AAPL", test_bars, source="test_source")
    assert inserted == 1

    # Valid bars check
    ohlcv_rows = db_session.exec(select(OhlcvBar)).all()
    assert len(ohlcv_rows) == 1
    assert ohlcv_rows[0].ts == "2026-01-01T00:00:00Z"

    # Quarantine bars check
    quarantine_rows = db_session.exec(
        select(QuarantineBar).order_by(QuarantineBar.id.asc())
    ).all()
    assert len(quarantine_rows) == 4

    reasons = [q.reason for q in quarantine_rows]
    assert "price_le_0" in reasons[0]
    assert "ohlc_inconsistent" in reasons[1]
    assert "negative_volume" in reasons[2]
    assert "price_le_0" in reasons[3]
    assert "ohlc_inconsistent" in reasons[3]
    assert "negative_volume" in reasons[3]

    for q in quarantine_rows:
        assert q.ticker == "AAPL"
        assert q.provider == "test_source"
        assert len(q.raw_json) > 0


def test_yfinance_provider_real_call() -> None:
    """Test YFinanceProvider against live yfinance or verify clean error fallback."""
    provider = YFinanceProvider()
    try:
        bars = provider.get_daily_history(
            "AAPL",
            start=date(2024, 1, 2),
            end=date(2024, 1, 5),
        )
        if bars:
            assert len(bars) > 0
            assert all(isinstance(b, Bar) for b in bars)
            assert bars[0].ts.endswith("T00:00:00Z")
            assert bars[0].close > 0
            assert bars[0].high >= bars[0].low

        latest = provider.get_latest_bars("AAPL", lookback=3)
        if latest:
            assert len(latest) <= 3
            assert all(isinstance(b, Bar) for b in latest)
    except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as exc:
        # If offline or blocked by remote host, test should not crash
        pytest.skip(f"Live yfinance network call skipped due to: {exc}")


def test_yfinance_provider_mocked() -> None:
    """Test YFinanceProvider parsing with mocked DataFrame structures."""
    provider = YFinanceProvider()

    # Mock MultiIndex DataFrame (standard yfinance format)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Adj Close", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )
    data = [
        [180.0, 185.0, 179.0, 184.0, 183.0, 50000.0],
        [184.0, 188.0, 183.0, 187.0, 186.0, 60000.0],
    ]
    mock_df = pd.DataFrame(data, index=dates, columns=columns)

    with patch("yfinance.download", return_value=mock_df):
        bars = provider.get_daily_history(
            "AAPL", date(2024, 1, 2), date(2024, 1, 3)
        )
        assert len(bars) == 2
        assert bars[0].ts == "2024-01-02T00:00:00Z"
        assert bars[0].open == 180.0
        assert bars[0].high == 185.0
        assert bars[0].low == 179.0
        assert bars[0].close == 184.0
        assert bars[0].adj_close == 183.0
        assert bars[0].volume == 50000.0

    # Mock SingleIndex DataFrame
    flat_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [98.0],
            "Close": [103.0],
            "Adj Close": [102.5],
            "Volume": [10000.0],
        },
        index=pd.to_datetime(["2024-01-05"]),
    )

    with patch("yfinance.download", return_value=flat_df):
        bars = provider.get_daily_history(
            "AAPL", date(2024, 1, 5), date(2024, 1, 5)
        )
        assert len(bars) == 1
        assert bars[0].ts == "2024-01-05T00:00:00Z"
        assert bars[0].close == 103.0

    # Mock empty DataFrame
    with patch("yfinance.download", return_value=pd.DataFrame()):
        bars = provider.get_daily_history(
            "AAPL", date(2024, 1, 1), date(2024, 1, 5)
        )
        assert bars == []

    # Mock get_latest_bars with 0 lookback
    assert provider.get_latest_bars("AAPL", lookback=0) == []
