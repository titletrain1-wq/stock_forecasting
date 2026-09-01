from datetime import date

import pytest

from stock_forecasting.providers.base import Bar
from stock_forecasting.providers.fake import FakeProvider


def test_fake_provider_returns_bars() -> None:
    """FakeProvider returns canned bars."""
    provider = FakeProvider()
    start = date(2024, 1, 1)
    end = date(2024, 1, 31)
    bars = provider.get_daily_history("AAPL", start, end)
    assert len(bars) == 31
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[0].ts <= bars[-1].ts
    assert bars[0].open == 100.0
    assert bars[0].volume == 1000000.0


def test_fake_provider_malformed() -> None:
    """FakeProvider can return malformed data."""
    provider = FakeProvider(return_malformed=True)
    bars = provider.get_daily_history("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert len(bars) == 1
    assert isinstance(bars[0], dict)
    assert bars[0]["ts"] == "bad"


def test_fake_provider_429() -> None:
    """FakeProvider can simulate rate limit."""
    provider = FakeProvider(return_429=True)
    with pytest.raises(Exception, match="429") as exc_info:
        provider.get_daily_history("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert "429" in str(exc_info.value)


def test_fake_provider_latest_bars() -> None:
    """FakeProvider returns latest bars for lookback window."""
    provider = FakeProvider()
    bars = provider.get_latest_bars("AAPL", lookback=5)
    assert len(bars) == 6  # today - 5 days to today inclusive
    assert all(isinstance(b, Bar) for b in bars)


def test_bar_dataclass() -> None:
    """Verify Bar dataclass attributes and optional adj_close."""
    bar = Bar(
        ts="2026-09-01T00:00:00Z",
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        adj_close=None,
        volume=500000.0,
    )
    assert bar.adj_close is None
    assert bar.close == 154.0
