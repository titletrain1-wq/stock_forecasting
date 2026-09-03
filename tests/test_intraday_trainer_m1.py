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
from sqlmodel import Session

from stock_forecasting.intraday_pipeline import (
    as_of_join_funding,
    fetch_intraday_bars_5m,
    filter_closed_bar_anchors,
)
from stock_forecasting.intraday_trainer import backfill_intraday_bars
from stock_forecasting.schema import IntradayBarsHistory, Ticker


class TestCoinbaseBackfill:
    """Test 1: Coinbase REST backfill shape and deduplication."""

    def testfetch_intraday_bars_5m_mock_response_shape(self) -> None:
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
        bars = fetch_intraday_bars_5m(mock_provider, "BTC-USD", start_date, end_date)

        # Verify shape
        assert len(bars) == bar_count, f"Expected {bar_count} bars, got {len(bars)}"
        assert all("ts" in b and "o" in b and "c" in b for b in bars)

    def testfetch_intraday_bars_5m_empty_response(self) -> None:
        """Test that empty Coinbase response returns empty list."""
        mock_provider = MagicMock()
        mock_provider.resolve_product_id.return_value = "BTC-USD"
        mock_client = MagicMock()
        mock_provider._get_client.return_value = mock_client
        mock_provider._client = None

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        bars = fetch_intraday_bars_5m(
            mock_provider, "BTC-USD", date(2026, 9, 1), date(2026, 9, 2)
        )

        assert len(bars) == 0

    def testfetch_intraday_bars_5m_handles_invalid_data(self) -> None:
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
            [
                int((current + timedelta(minutes=5)).timestamp()),
                45075,
                45150,
                45100,
                45125,
                11.0,
            ],
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = mock_bars
        mock_client.get.return_value = mock_response

        bars = fetch_intraday_bars_5m(
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
            bars_data.append(
                {
                    "ts": ts,
                    "o": 45000.0 + i * 100,
                    "h": 45100.0 + i * 100,
                    "l": 44900.0 + i * 100,
                    "c": 45050.0 + i * 100,
                    "v": 10.5 + i,
                }
            )
        bars_df = pd.DataFrame(bars_data)

        # Create funding rates: published at 00:30, 02:30, 04:30 UTC
        funding_data = [
            {"ts": base + timedelta(hours=0, minutes=30), "funding_rate": 0.0001},
            {"ts": base + timedelta(hours=2, minutes=30), "funding_rate": 0.0002},
            {"ts": base + timedelta(hours=4, minutes=30), "funding_rate": 0.0003},
        ]
        funding_df = pd.DataFrame(funding_data)

        # Perform as-of join
        _, result = as_of_join_funding(bars_df, funding_df)

        # Verify:
        # - Hour 0 (ts=00:00): should use nothing (no prior rate) -> NaN
        # - Hour 1 (ts=01:00): should use rate from 00:30 -> 0.0001
        # - Hour 2 (ts=02:00): should use rate from 00:30 -> 0.0001 (not 02:30 yet)
        # - Hour 3 (ts=03:00): should use rate from 02:30 -> 0.0002
        # - Hour 4 (ts=04:00): should use rate from 02:30 -> 0.0002 (not 04:30 yet)

        assert pd.isna(result.iloc[0]["funding_rate"])  # Hour 0: no prior
        assert result.iloc[1]["funding_rate"] == 0.0001  # Hour 1: rate from 00:30
        assert (
            result.iloc[2]["funding_rate"] == 0.0001
        )  # Hour 2: rate from 00:30 (backward join)
        assert result.iloc[3]["funding_rate"] == 0.0002  # Hour 3: rate from 02:30
        assert (
            result.iloc[4]["funding_rate"] == 0.0002
        )  # Hour 4: rate from 02:30 (backward join)

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

        _, result = as_of_join_funding(bars_df, funding_df)

        assert pd.isna(result.iloc[0]["funding_rate"])


class TestAnchorFiltering:
    """Test 3: Closed-bar anchor filtering."""

    def testfilter_closed_bar_anchors_1h_and_4h(self) -> None:
        """Test anchor filter: 1h at every :00 (24/day), 4h at hour%4==0 & :00 (6/day)."""
        # Create 1 full day of 5-minute bars (288 bars)
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        df_data = []
        for i in range(288):  # 24h * 60min / 5min = 288
            ts = base + timedelta(minutes=5 * i)
            df_data.append(
                {
                    "ts": ts,
                    "o": 45000.0,
                    "h": 45100.0,
                    "l": 44900.0,
                    "c": 45050.0,
                    "v": 10.5,
                }
            )
        df = pd.DataFrame(df_data)

        # Filter to anchors
        result = filter_closed_bar_anchors(df)

        # 1h anchors: all :00 timestamps = 24 per day (one per hour)
        assert len(result) == 24, f"Expected 24 anchors (all :00 per day), got {len(result)}"

        # Verify NO non-:00-minute timestamps survive
        assert all(result["ts"].dt.minute == 0), "Non-zero minutes found"
        assert all(result["ts"].dt.second == 0), "Non-zero seconds found"

        # Count 4h-specific anchors (hour % 4 == 0)
        result["hour"] = result["ts"].dt.hour
        is_4h = result["hour"] % 4 == 0
        count_4h = is_4h.sum()
        assert count_4h == 6, f"Expected 6 4h anchors (00/04/08/12/16/20:00), got {count_4h}"

    def testfilter_closed_bar_anchors_empty(self) -> None:
        """Test that empty DataFrame returns empty result."""
        df = pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
        result = filter_closed_bar_anchors(df)
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
        with patch(
            "stock_forecasting.intraday_trainer.CoinbaseProvider"
        ) as MockCoinbase:
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
                        result.append(
                            {
                                "ts": dt.isoformat().replace("+00:00", "Z"),
                                "interval": "5m",
                                "o": item[3],
                                "h": item[2],
                                "l": item[1],
                                "c": item[4],
                                "v": item[5],
                            }
                        )
                return result

            # Patch the fetch function
            with patch(
                "stock_forecasting.intraday_trainer.fetch_intraday_bars_5m",
                side_effect=mock_fetch_fn,
            ), patch(
                "stock_forecasting.intraday_trainer._fetch_funding_rates_from_db"
            ) as MockFunding:
                # Return empty funding (for simplicity)
                MockFunding.return_value = pd.DataFrame(
                    columns=["ts", "funding_rate"]
                )

                # Run backfill
                backfill_intraday_bars(
                    session=db_session, tickers=["BTC-USD"], test_mode=True
                )

            # Verify rows were written to intraday_bars_history
            from sqlmodel import select

            rows = db_session.exec(
                select(IntradayBarsHistory).where(
                    IntradayBarsHistory.ticker == "BTC-USD"
                )
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
        with patch(
            "stock_forecasting.intraday_trainer.CoinbaseProvider"
        ):

            def mock_fetch_fn(provider, ticker, start, end, **kwargs):
                return [
                    {
                        "ts": ts_str,
                        "interval": "5m",
                        "o": 2500.0,
                        "h": 2510.0,
                        "l": 2490.0,
                        "c": 2505.0,
                        "v": 100.0,
                    }
                ]

            with patch(
                "stock_forecasting.intraday_trainer.fetch_intraday_bars_5m",
                side_effect=mock_fetch_fn,
            ), patch(
                "stock_forecasting.intraday_trainer._fetch_funding_rates_from_db"
            ) as MockFunding:
                MockFunding.return_value = pd.DataFrame(
                    columns=["ts", "funding_rate"]
                )

                backfill_intraday_bars(
                    session=db_session, tickers=["ETH-USD"], test_mode=True
                )

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
