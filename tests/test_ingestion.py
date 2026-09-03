"""Tests for BarRepository and YFinanceProvider ingestion workflows."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.ingestion import IngestionService
from stock_forecasting.providers.base import Bar
from stock_forecasting.providers.fake import FakeProvider
from stock_forecasting.providers.yfinance import YFinanceProvider
from stock_forecasting.schema import LinkMetrics, OhlcvBar, QuarantineBar, Ticker


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
    range_bars = repo.get_range("AAPL", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z")
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

    all_bars = repo.get_range("AAPL", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z")
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
        bars = provider.get_daily_history("AAPL", date(2024, 1, 2), date(2024, 1, 3))
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
        bars = provider.get_daily_history("AAPL", date(2024, 1, 5), date(2024, 1, 5))
        assert len(bars) == 1
        assert bars[0].ts == "2024-01-05T00:00:00Z"
        assert bars[0].close == 103.0

    # Mock empty DataFrame
    with patch("yfinance.download", return_value=pd.DataFrame()):
        bars = provider.get_daily_history("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert bars == []

    # Mock get_latest_bars with 0 lookback
    assert provider.get_latest_bars("AAPL", lookback=0) == []


def test_ingestion_service_poll(db_session: Session) -> None:
    """Verify IngestionService polls active watchlist tickers and stores bars."""
    t1 = Ticker(
        symbol="AAPL",
        asset_class="equity",
        display_name="Apple Inc.",
        provider="fake",
        provider_symbol="AAPL",
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    t2 = Ticker(
        symbol="MSFT",
        asset_class="equity",
        display_name="Microsoft Corp.",
        provider="fake",
        provider_symbol="MSFT",
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    t3 = Ticker(
        symbol="GOOG",
        asset_class="equity",
        display_name="Alphabet Inc.",
        provider="fake",
        provider_symbol="GOOG",
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=0,
    )
    db_session.add_all([t1, t2, t3])
    db_session.commit()

    service = IngestionService(
        session=db_session,
        providers={"fake": FakeProvider()},
    )

    results = service.poll_watchlist()
    assert "AAPL" in results
    assert "MSFT" in results
    assert "GOOG" not in results

    assert results["AAPL"]["inserted"] > 0
    assert results["MSFT"]["inserted"] > 0

    # Verify bars written to database
    aapl_bars = db_session.exec(select(OhlcvBar).where(OhlcvBar.ticker == "AAPL")).all()
    assert len(aapl_bars) == results["AAPL"]["inserted"]

    # Poll single ticker directly
    single_res = service.poll_ticker("AAPL", lookback=2)
    assert single_res["symbol"] == "AAPL"
    # Idempotent: re-polling same recent bars inserts 0 new rows
    assert single_res["inserted"] == 0


def test_ingestion_service_backfill(db_session: Session) -> None:
    """Verify IngestionService historical backfill for a ticker."""
    ticker = Ticker(
        symbol="BTC",
        asset_class="crypto",
        display_name="Bitcoin",
        provider="fake",
        provider_symbol="BTC-USD",
        price_basis="raw",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    db_session.add(ticker)
    db_session.commit()

    service = IngestionService(
        session=db_session,
        providers={"fake": FakeProvider()},
    )

    res = service.backfill("BTC", years=2)
    assert res["symbol"] == "BTC"
    assert res["inserted"] > 700  # 2 years of daily bars

    btc_bars = db_session.exec(select(OhlcvBar).where(OhlcvBar.ticker == "BTC")).all()
    assert len(btc_bars) == res["inserted"]
    assert all(b.source == "fake" for b in btc_bars)


def test_ingestion_service_missing_ticker_or_provider(
    db_session: Session,
) -> None:
    """Verify IngestionService handles missing tickers, providers, and provider errors gracefully."""
    t_unsupported = Ticker(
        symbol="UNK",
        asset_class="equity",
        display_name="Unknown Provider Co.",
        provider="unregistered_provider",
        provider_symbol="UNK",
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    t_failing = Ticker(
        symbol="FAIL",
        asset_class="equity",
        display_name="Failing Provider Co.",
        provider="error_provider",
        provider_symbol="FAIL",
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    db_session.add_all([t_unsupported, t_failing])
    db_session.commit()

    service = IngestionService(
        session=db_session,
        providers={"error_provider": FakeProvider(return_429=True)},
    )

    # 1. Missing ticker in database
    poll_missing = service.poll_ticker("NONEXISTENT")
    assert poll_missing["inserted"] == 0
    assert "not found" in poll_missing.get("error", "").lower()

    backfill_missing = service.backfill("NONEXISTENT")
    assert backfill_missing["inserted"] == 0
    assert "not found" in backfill_missing.get("error", "").lower()

    # 2. Unregistered provider
    poll_unk = service.poll_ticker("UNK")
    assert poll_unk["inserted"] == 0
    assert "not found" in poll_unk.get("error", "").lower()

    # ...and the outage is surfaced to LinkMetrics so check_error_rate sees it,
    # rather than being silently swallowed on the empty-provider-chain path.

    unk_metric = db_session.get(LinkMetrics, "unregistered_provider")
    assert unk_metric is not None
    assert unk_metric.consecutive_failures >= 1

    backfill_unk = service.backfill("UNK")
    assert backfill_unk["inserted"] == 0
    assert "not found" in backfill_unk.get("error", "").lower()

    # 3. Provider throwing exception (e.g. rate limit 429)
    poll_fail = service.poll_ticker("FAIL")
    assert poll_fail["inserted"] == 0
    assert "429" in poll_fail.get("error", "")

    backfill_fail = service.backfill("FAIL")
    assert backfill_fail["inserted"] == 0
    assert "429" in backfill_fail.get("error", "")


class _AlwaysFailProvider:
    """Provider that raises on every call (simulates a downed upstream)."""

    def get_daily_history(self, symbol, start, end):
        raise RuntimeError("HTTP 503: upstream down")

    def get_latest_bars(self, symbol, lookback=5):
        raise RuntimeError("HTTP 503: upstream down")


def test_poll_ticker_fails_over_to_fallback_when_primary_breaker_open(
    db_session: Session,
) -> None:
    """yfinance breaker trips after 5 failures -> poll_ticker serves via tiingo."""
    _create_sample_ticker(db_session, "AAPL")  # provider="yfinance"

    service = IngestionService(
        session=db_session,
        providers={"yfinance": _AlwaysFailProvider(), "tiingo": FakeProvider()},
    )

    # First poll: yfinance raises + records a failure, then fails over to tiingo.
    first = service.poll_ticker("AAPL")
    assert first["provider"] == "tiingo"
    assert first["failover_from"] == "yfinance"
    assert first["inserted"] > 0
    bars = db_session.exec(select(OhlcvBar).where(OhlcvBar.ticker == "AAPL")).all()
    assert bars and all(b.source == "tiingo" for b in bars)

    # 4 more failing yfinance attempts trip its breaker (threshold 5).
    for _ in range(4):
        service.poll_ticker("AAPL")
    yf = db_session.get(LinkMetrics, "yfinance")
    assert yf.breaker_state == "open"

    # Breaker now open: yfinance skipped outright, still served by tiingo.
    after = service.poll_ticker("AAPL")
    assert after["provider"] == "tiingo"
    assert after["failover_from"] == "yfinance"


def test_poll_ticker_records_success_keeps_breaker_closed(db_session: Session) -> None:
    _create_sample_ticker(db_session, "AAPL")
    service = IngestionService(
        session=db_session,
        providers={"yfinance": FakeProvider()},
    )
    res = service.poll_ticker("AAPL")
    assert res["provider"] == "yfinance"
    assert "failover_from" not in res

    yf = db_session.get(LinkMetrics, "yfinance")
    assert yf.breaker_state == "closed"
    assert yf.consecutive_failures == 0


def test_poll_ticker_all_providers_fail_returns_error(db_session: Session) -> None:
    _create_sample_ticker(db_session, "AAPL")
    service = IngestionService(
        session=db_session,
        providers={
            "yfinance": _AlwaysFailProvider(),
            "tiingo": _AlwaysFailProvider(),
            "finnhub": _AlwaysFailProvider(),
        },
    )
    res = service.poll_ticker("AAPL")
    assert res["inserted"] == 0
    assert "503" in res["error"]


def test_backfill_fails_over_to_fallback(db_session: Session) -> None:
    _create_sample_ticker(db_session, "AAPL")
    service = IngestionService(
        session=db_session,
        providers={"yfinance": _AlwaysFailProvider(), "tiingo": FakeProvider()},
    )
    res = service.backfill("AAPL", years=1)
    assert res["provider"] == "tiingo"
    assert res["failover_from"] == "yfinance"


def test_bar_repository_filters_forming_candles(db_session: Session) -> None:
    """Verify BarRepository skips today's forming/provisional candles (Issue c)."""
    from datetime import datetime, UTC

    _create_sample_ticker(db_session, "BTC")
    repo = BarRepository(db_session)

    from datetime import timedelta

    today = datetime.now(UTC).date()
    today_ts = f"{today.isoformat()}T00:00:00Z"
    yesterday = today - timedelta(days=1)
    yesterday_ts = f"{yesterday.isoformat()}T00:00:00Z"

    bars = [
        Bar(
            ts=yesterday_ts,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            adj_close=None,
            volume=100.0,
        ),
        Bar(
            ts=today_ts,  # This should be filtered out (forming candle)
            open=50500.0,
            high=51200.0,
            low=50400.0,
            close=50800.0,  # Partial/forming
            adj_close=None,
            volume=50.0,
        ),
    ]

    inserted = repo.upsert_bars("BTC", bars, source="coinbase")
    assert inserted == 1  # Only yesterday's bar should be inserted

    # Verify only yesterday's bar exists
    rows = db_session.exec(select(OhlcvBar).where(OhlcvBar.ticker == "BTC")).all()
    assert len(rows) == 1
    assert rows[0].ts == yesterday_ts
    assert rows[0].close == 50500.0
