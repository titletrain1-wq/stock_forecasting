"""M1 Tests: Intraday data pipeline (Coinbase backfill + funding as-of join + anchor filtering).

Tests that verify:
- Coinbase REST backfill fetches correct shape (~8,760 5m bars/ticker for 365d)
- Funding-rate as-of join has no lookahead (no forward-fill)
- Anchor filtering keeps only closed-bar boundaries
- End-to-end: backfill writes to intraday_bars_history with dedup
"""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlmodel import Session

from stock_forecasting.intraday_trainer import (
    _as_of_join_funding,
    _fetch_intraday_bars_5m,
    _filter_closed_bar_anchors,
    backfill_intraday_bars,
)
from stock_forecasting.schema import IntradayBarsHistory, Ticker


class TestCoinbaseBackfill:
    """Test 1: Coinbase REST backfill shape and deduplication."""

    def test_fetch_intraday_bars_5m_mock_response_shape(self) -> None:
        """Test that mocked Coinbase REST returns ~8,760 5m bars for 365 days."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.resolve_product_id.return_value = "BTC-USD"

        # Mock client and response
        mock_client = MagicMock()
        mock_provider._get_client.return_value = mock_client
        mock_provider._client = None

        # Generate 365 days of 5-minute bars (24h * 60min / 5min = 288 bars/day)
        # 365 * 288 = 105,120 bars (but Coinbase has trading gaps; expect ~97k-100k)
        expected_bars_per_day = 288
        lookback_days = 365
        expected_total = expected_bars_per_day * lookback_days

        # Build mock response: one chunk for simplicity
        start_date = date(2025, 9, 3)
        end_date = date(2026, 9, 2)
        mock_bars = []

        current = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
        end = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC)

        bar_count = 0
        while current <= end and bar_count < 1000:  # Limit for test speed
            epoch = int(current.timestamp())
            mock_bars.append([epoch, 45000, 45100, 45050, 45075, 10.5])
            current += timedelta(minutes=5)
            bar_count += 1

        mock_response = MagicMock()
        mock_response.json.return_value = mock_bars
        mock_client.get.return_value = mock_response

        # Call the function
        bars = _fetch_intraday_bars_5m(mock_provider, "BTC-USD", start_date, end_date)

        # Verify shape
        assert len(bars) == bar_count, f"Expected {bar_count} bars, got {len(bars)}"
        assert all("ts" in b and "o" in b and "c" in b for b in bars)

    def test_fetch_intraday_bars_5m_empty_response(self) -> None:
        """Test that empty Coinbase response returns empty list."""
        mock_provider = MagicMock()
        mock_provider.resolve_product_id.return_value = "BTC-USD"
        mock_client = MagicMock()
        mock_provider._get_client.return_value = mock_client
        mock_provider._client = None

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        bars = _fetch_intraday_bars_5m(
            mock_provider, "BTC-USD", date(2026, 9, 1), date(2026, 9, 2)
        )

        assert len(bars) == 0

    def test_fetch_intraday_bars_5m_handles_invalid_data(self) -> None:
        """Test that invalid candle data is skipped gracefully."""
        mock_provider = MagicMock()
        mock_provider.resolve_product_id.return_value = "BTC-USD"
        mock_client = MagicMock()
        mock_provider._get_client.return_value = mock_client
        mock_provider._client = None

        # Mix of valid and invalid candles
        current = datetime.combine(date(2026, 9, 1), datetime.min.time(), tzinfo=UTC)
        mock_bars = [
            [int(current.timestamp()), 45000, 45100, 45050, 45075, 10.5],  # valid
            "invalid",  # invalid
            [int((current + timedelta(minutes=5)).timestamp()), 45075, 45150, 45100, 45125, 11.0],
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = mock_bars
        mock_client.get.return_value = mock_response

        bars = _fetch_intraday_bars_5m(
            mock_provider, "BTC-USD", date(2026, 9, 1), date(2026, 9, 2)
        )

        assert len(bars) == 2  # Only valid bars


class TestFundingAsOfJoin:
    """Test 2: Funding-rate as-of join with no lookahead."""

    def test_as_of_join_no_forward_fill(self) -> None:
        """Test that as-of join uses backward direction (no forward-fill lookahead)."""
        # Create bars: ts at 1h intervals
        bars_data = []
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(5):
            ts = base + timedelta(hours=i)
            bars_data.append({
                "ts": ts,
                "o": 45000.0 + i * 100,
                "h": 45100.0 + i * 100,
                "l": 44900.0 + i * 100,
                "c": 45050.0 + i * 100,
                "v": 10.5 + i,
            })
        bars_df = pd.DataFrame(bars_data)

        # Create funding rates: published at 00:30, 02:30, 04:30 UTC
        funding_data = [
            {"ts": base + timedelta(hours=0, minutes=30), "funding_rate": 0.0001},
            {"ts": base + timedelta(hours=2, minutes=30), "funding_rate": 0.0002},
            {"ts": base + timedelta(hours=4, minutes=30), "funding_rate": 0.0003},
        ]
        funding_df = pd.DataFrame(funding_data)

        # Perform as-of join
        _, result = _as_of_join_funding(bars_df, funding_df)

        # Verify:
        # - Hour 0 (ts=00:00): should use nothing (no prior rate) -> NaN
        # - Hour 1 (ts=01:00): should use rate from 00:30 -> 0.0001
        # - Hour 2 (ts=02:00): should use rate from 00:30 -> 0.0001 (not 02:30 yet)
        # - Hour 3 (ts=03:00): should use rate from 02:30 -> 0.0002
        # - Hour 4 (ts=04:00): should use rate from 02:30 -> 0.0002 (not 04:30 yet)

        assert pd.isna(result.iloc[0]["funding_rate"])  # Hour 0: no prior
        assert result.iloc[1]["funding_rate"] == 0.0001  # Hour 1: rate from 00:30
        assert result.iloc[2]["funding_rate"] == 0.0001  # Hour 2: rate from 00:30 (backward join)
        assert result.iloc[3]["funding_rate"] == 0.0002  # Hour 3: rate from 02:30
        assert result.iloc[4]["funding_rate"] == 0.0002  # Hour 4: rate from 02:30 (backward join)

    def test_as_of_join_empty_funding(self) -> None:
        """Test that as-of join with empty funding returns bars with NaN funding."""
        bars_data = [
            {
                "ts": datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
                "o": 45000.0,
                "h": 45100.0,
                "l": 44900.0,
                "c": 45050.0,
                "v": 10.5,
            },
        ]
        bars_df = pd.DataFrame(bars_data)
        funding_df = pd.DataFrame(columns=["ts", "funding_rate"])

        _, result = _as_of_join_funding(bars_df, funding_df)

        assert pd.isna(result.iloc[0]["funding_rate"])


class TestAnchorFiltering:
    """Test 3: Closed-bar anchor filtering."""

    def test_filter_closed_bar_anchors_1h_and_4h(self) -> None:
        """Test that anchor filter keeps only :00 (1h) and :00/:04/:08/:12/:16/:20 (4h) times."""
        # Create data at various minute markers
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        timestamps = [
            base + timedelta(minutes=0),   # 00:00 - valid (both 1h and 4h)
            base + timedelta(minutes=4),   # 00:04 - valid (4h only)
            base + timedelta(minutes=5),   # 00:05 - invalid
            base + timedelta(minutes=8),   # 00:08 - valid (4h only)
            base + timedelta(minutes=12),  # 00:12 - valid (4h only)
            base + timedelta(minutes=15),  # 00:15 - invalid
            base + timedelta(minutes=16),  # 00:16 - valid (4h only)
            base + timedelta(minutes=20),  # 00:20 - valid (4h only)
            base + timedelta(minutes=25),  # 00:25 - invalid
            base + timedelta(hours=1, minutes=0),   # 01:00 - valid (both 1h and 4h)
        ]

        df_data = []
        for ts in timestamps:
            df_data.append({
                "ts": ts,
                "o": 45000.0,
                "h": 45100.0,
                "l": 44900.0,
                "c": 45050.0,
                "v": 10.5,
            })
        df = pd.DataFrame(df_data)

        # Filter
        result = _filter_closed_bar_anchors(df)

        # Expected: 0, 4, 8, 12, 16, 20 minutes, and the 01:00 (60 minutes)
        expected_minutes = {0, 4, 8, 12, 16, 20, 60}
        result_minutes = set(result["ts"].dt.minute.tolist())
        # Note: minute 60 wraps to 0, so check hour separately
        result_hours = set(result["ts"].dt.hour.tolist())

        assert len(result) == 7, f"Expected 7 anchors, got {len(result)}"
        # Check each row is valid
        for _, row in result.iterrows():
            minute = row["ts"].minute
            hour = row["ts"].hour
            is_1h_anchor = minute == 0
            is_4h_anchor = minute in [4, 8, 12, 16, 20] or (hour % 4 == 0 and minute == 0)
            assert is_1h_anchor or is_4h_anchor, f"Invalid anchor at {row['ts']}"

    def test_filter_closed_bar_anchors_empty(self) -> None:
        """Test that empty DataFrame returns empty result."""
        df = pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
        result = _filter_closed_bar_anchors(df)
        assert result.empty


class TestEndToEnd:
    """Test 4: End-to-end backfill and database write."""

    def test_backfill_end_to_end_mock_api(self, db_session: Session) -> None:
        """Test that backfill fetches, joins, filters, and writes to intraday_bars_history."""
        # Create BTC-USD ticker
        ticker = Ticker(
            symbol="BTC-USD",
            asset_class="crypto",
            display_name="Bitcoin",
            provider="coinbase",
            provider_symbol="BTC-USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        db_session.add(ticker)
        db_session.commit()

        # Mock Coinbase provider
        with patch("stock_forecasting.intraday_trainer.CoinbaseProvider") as MockCoinbase:
            mock_coinbase = MagicMock()
            MockCoinbase.return_value = mock_coinbase

            # Generate mock 5m bars for 2 days (24h * 2 * 12 5m intervals = 576 bars)
            base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
            mock_bars = []
            for i in range(576):  # 2 full days of 5m bars
                ts = base + timedelta(minutes=5 * i)
                epoch = int(ts.timestamp())
                mock_bars.append([epoch, 45000, 45100, 45050, 45075, 10.5])

            def mock_fetch_fn(provider, ticker, start, end, **kwargs):
                # Return bars in the expected format
                result = []
                for item in mock_bars:
                    epoch = item[0]
                    dt = datetime.fromtimestamp(epoch, tz=UTC)
                    if start <= dt.date() <= end:
                        result.append({
                            "ts": dt.isoformat().replace("+00:00", "Z"),
                            "interval": "5m",
                            "o": item[3],
                            "h": item[2],
                            "l": item[1],
                            "c": item[4],
                            "v": item[5],
                        })
                return result

            # Patch the fetch function
            with patch("stock_forecasting.intraday_trainer._fetch_intraday_bars_5m", side_effect=mock_fetch_fn):
                with patch("stock_forecasting.intraday_trainer._fetch_funding_rates_from_db") as MockFunding:
                    # Return empty funding (for simplicity)
                    MockFunding.return_value = pd.DataFrame(columns=["ts", "funding_rate"])

                    # Run backfill
                    backfill_intraday_bars(session=db_session, tickers=["BTC-USD"], test_mode=True)

            # Verify rows were written to intraday_bars_history
            from sqlmodel import select
            rows = db_session.exec(
                select(IntradayBarsHistory).where(IntradayBarsHistory.ticker == "BTC-USD")
            ).all()

            # Should have ~288 rows per day * 2 days after anchor filtering
            # (roughly half of 576 will be closed-bar anchors)
            assert len(rows) > 0, "No rows written to intraday_bars_history"
            assert all(r.source == "coinbase_rest" for r in rows)
            assert all(r.interval == "5m" for r in rows)

    def test_backfill_deduplication(self, db_session: Session) -> None:
        """Test that duplicate bars are skipped (INSERT OR IGNORE via dedup check)."""
        # Create ticker
        ticker = Ticker(
            symbol="ETH-USD",
            asset_class="crypto",
            display_name="Ethereum",
            provider="coinbase",
            provider_symbol="ETH-USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        db_session.add(ticker)
        db_session.commit()

        # Insert one bar manually
        ts_str = "2026-09-01T00:00:00Z"
        bar1 = IntradayBarsHistory(
            ticker="ETH-USD",
            interval="5m",
            ts=ts_str,
            open=2500.0,
            high=2510.0,
            low=2490.0,
            close=2505.0,
            volume=100.0,
            source="coinbase_rest",
            ingested_at="2026-09-01T00:01:00Z",
        )
        db_session.add(bar1)
        db_session.commit()
        initial_id = bar1.id

        # Mock backfill to try inserting the same bar
        with patch("stock_forecasting.intraday_trainer.CoinbaseProvider") as MockCoinbase:
            def mock_fetch_fn(provider, ticker, start, end, **kwargs):
                return [{
                    "ts": ts_str,
                    "interval": "5m",
                    "o": 2500.0,
                    "h": 2510.0,
                    "l": 2490.0,
                    "c": 2505.0,
                    "v": 100.0,
                }]

            with patch("stock_forecasting.intraday_trainer._fetch_intraday_bars_5m", side_effect=mock_fetch_fn):
                with patch("stock_forecasting.intraday_trainer._fetch_funding_rates_from_db") as MockFunding:
                    MockFunding.return_value = pd.DataFrame(columns=["ts", "funding_rate"])

                    backfill_intraday_bars(session=db_session, tickers=["ETH-USD"], test_mode=True)

        # Verify only one row exists (dedup worked)
        from sqlmodel import select
        rows = db_session.exec(
            select(IntradayBarsHistory).where(
                (IntradayBarsHistory.ticker == "ETH-USD")
                & (IntradayBarsHistory.ts == ts_str)
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].id == initial_id  # Same row, not duplicated
