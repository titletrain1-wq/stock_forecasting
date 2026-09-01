"""Unit tests for CoinGeckoProvider and CoinbaseProvider crypto data sources."""

from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest
from sqlmodel import Session

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.ingestion import IngestionService
from stock_forecasting.providers.base import Bar, DataProvider
from stock_forecasting.providers.coinbase import CoinbaseProvider
from stock_forecasting.providers.coingecko import CoinGeckoProvider
from stock_forecasting.schema import Ticker

# ---------------------------------------------------------------------------
# CoinGeckoProvider Tests
# ---------------------------------------------------------------------------


def test_coingecko_symbol_resolution() -> None:
    """CoinGeckoProvider maps standard crypto symbols to coin IDs."""
    provider = CoinGeckoProvider()

    assert provider.resolve_coin_id("BTC-USD") == "bitcoin"
    assert provider.resolve_coin_id("BTC") == "bitcoin"
    assert provider.resolve_coin_id("bitcoin") == "bitcoin"
    assert provider.resolve_coin_id("ETH-USD") == "ethereum"
    assert provider.resolve_coin_id("ETH") == "ethereum"
    assert provider.resolve_coin_id("SOL-USD") == "solana"
    assert provider.resolve_coin_id("SOL") == "solana"
    assert provider.resolve_coin_id("DOGE-USD") == "dogecoin"
    assert provider.resolve_coin_id("UNKNOWN-USD") == "unknown"
    assert provider.resolve_coin_id("custom_coin") == "custom_coin"

    # Custom symbol map override
    custom_provider = CoinGeckoProvider(symbol_map={"CUSTOM": "custom-id"})
    assert custom_provider.resolve_coin_id("CUSTOM") == "custom-id"


def test_coingecko_protocol_compliance() -> None:
    """CoinGeckoProvider implements DataProvider protocol."""
    provider = CoinGeckoProvider()
    assert isinstance(provider, DataProvider)


def test_coingecko_get_daily_history_success() -> None:
    """CoinGeckoProvider parses market_chart/range response into sorted Bar list."""
    # Timestamps in ms for 2026-01-01 and 2026-01-02
    ts_day1 = 1767225600000  # 2026-01-01T00:00:00Z
    ts_day2 = 1767312000000  # 2026-01-02T00:00:00Z

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "prices": [
            [ts_day1, 50000.0],
            [ts_day2, 52000.0],
        ],
        "total_volumes": [
            [ts_day1, 100000000.0],
            [ts_day2, 120000000.0],
        ],
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinGeckoProvider(client=mock_client)
    start = date(2026, 1, 1)
    end = date(2026, 1, 2)

    bars = provider.get_daily_history("BTC-USD", start=start, end=end)

    assert len(bars) == 2
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[0].ts == "2026-01-01T00:00:00Z"
    assert bars[0].open == 50000.0
    assert bars[0].high == 50000.0
    assert bars[0].low == 50000.0
    assert bars[0].close == 50000.0
    assert bars[0].volume == 100000000.0
    assert bars[0].adj_close is None

    assert bars[1].ts == "2026-01-02T00:00:00Z"
    assert bars[1].close == 52000.0
    assert bars[1].volume == 120000000.0


def test_coingecko_intraday_aggregation() -> None:
    """CoinGeckoProvider aggregates multiple intraday points into single daily OHLC."""
    # Two points on same day 2026-01-01 (00:00 and 12:00 UTC)
    ts1 = 1767225600000  # 2026-01-01T00:00:00Z
    ts2 = 1767268800000  # 2026-01-01T12:00:00Z

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "prices": [
            [ts1, 50000.0],
            [ts2, 53000.0],
        ],
        "total_volumes": [
            [ts1, 50000000.0],
            [ts2, 80000000.0],
        ],
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinGeckoProvider(client=mock_client)
    bars = provider.get_daily_history(
        "BTC-USD", start=date(2026, 1, 1), end=date(2026, 1, 1)
    )

    assert len(bars) == 1
    assert bars[0].ts == "2026-01-01T00:00:00Z"
    assert bars[0].open == 50000.0
    assert bars[0].high == 53000.0
    assert bars[0].low == 50000.0
    assert bars[0].close == 53000.0
    assert bars[0].volume == 80000000.0


def test_coingecko_ohlc_list_format() -> None:
    """CoinGeckoProvider handles direct OHLC list format [[ts_ms, o, h, l, c], ...]."""
    ts_day1 = 1767225600000  # 2026-01-01

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [
        [ts_day1, 50000.0, 55000.0, 49000.0, 54000.0],
    ]

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinGeckoProvider(client=mock_client)
    bars = provider.get_daily_history(
        "ETH-USD", start=date(2026, 1, 1), end=date(2026, 1, 1)
    )

    assert len(bars) == 1
    assert bars[0].ts == "2026-01-01T00:00:00Z"
    assert bars[0].open == 50000.0
    assert bars[0].high == 55000.0
    assert bars[0].low == 49000.0
    assert bars[0].close == 54000.0
    assert bars[0].volume == 0.0


def test_coingecko_empty_or_invalid_responses() -> None:
    """CoinGeckoProvider gracefully handles empty, invalid, or inverted date ranges."""
    provider = CoinGeckoProvider()

    # Inverted date range
    assert (
        provider.get_daily_history(
            "BTC", start=date(2026, 1, 10), end=date(2026, 1, 1)
        )
        == []
    )

    # Empty data parsing
    assert (
        provider._parse_market_chart_data(
            {}, start=date(2026, 1, 1), end=date(2026, 1, 2)
        )
        == []
    )
    assert (
        provider._parse_market_chart_data(
            {"prices": []}, start=date(2026, 1, 1), end=date(2026, 1, 2)
        )
        == []
    )
    assert (
        provider._parse_market_chart_data(
            "not json", start=date(2026, 1, 1), end=date(2026, 1, 2)
        )
        == []
    )


def test_coingecko_error_handling_429() -> None:
    """CoinGeckoProvider raises HTTPStatusError on rate limit (429)."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Rate limit exceeded", request=MagicMock(), response=mock_response
    )

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinGeckoProvider(client=mock_client)
    with pytest.raises(httpx.HTTPStatusError):
        provider.get_daily_history(
            "BTC-USD", start=date(2026, 1, 1), end=date(2026, 1, 2)
        )


def test_coingecko_api_key_and_headers() -> None:
    """CoinGeckoProvider sends API key in query params / headers when configured."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"prices": [], "total_volumes": []}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinGeckoProvider(api_key="test-api-key", client=mock_client)
    provider.get_daily_history("BTC-USD", start=date(2026, 1, 1), end=date(2026, 1, 2))

    call_args = mock_client.get.call_args
    assert call_args is not None
    params = call_args[1].get("params", {})
    assert params.get("x_cg_demo_api_key") == "test-api-key"


def test_coingecko_get_latest_bars() -> None:
    """CoinGeckoProvider returns requested lookback slice of bars."""
    provider = CoinGeckoProvider()

    # Lookback <= 0 returns empty
    assert provider.get_latest_bars("BTC-USD", lookback=0) == []
    assert provider.get_latest_bars("BTC-USD", lookback=-5) == []

    # Mock get_daily_history
    mock_bars = [
        Bar(f"2026-01-0{i}T00:00:00Z", 100.0, 105.0, 99.0, 102.0, None, 1000.0)
        for i in range(1, 8)
    ]
    provider.get_daily_history = MagicMock(return_value=mock_bars)  # type: ignore[assignment]

    latest = provider.get_latest_bars("BTC-USD", lookback=3)
    assert len(latest) == 3
    assert latest[-1].ts == "2026-01-07T00:00:00Z"


# ---------------------------------------------------------------------------
# CoinbaseProvider Tests
# ---------------------------------------------------------------------------


def test_coinbase_symbol_resolution() -> None:
    """CoinbaseProvider maps standard crypto symbols to product IDs."""
    provider = CoinbaseProvider()

    assert provider.resolve_product_id("BTC") == "BTC-USD"
    assert provider.resolve_product_id("BTC-USD") == "BTC-USD"
    assert provider.resolve_product_id("bitcoin") == "BTC-USD"
    assert provider.resolve_product_id("ETH") == "ETH-USD"
    assert provider.resolve_product_id("ETH-USD") == "ETH-USD"
    assert provider.resolve_product_id("ethereum") == "ETH-USD"
    assert provider.resolve_product_id("SOL") == "SOL-USD"
    assert provider.resolve_product_id("SOL-USD") == "SOL-USD"
    assert provider.resolve_product_id("UNKNOWN") == "UNKNOWN-USD"
    assert provider.resolve_product_id("SOL-EUR") == "SOL-EUR"

    # Custom symbol map override
    custom_provider = CoinbaseProvider(symbol_map={"CUSTOM": "CUSTOM-PAIR"})
    assert custom_provider.resolve_product_id("CUSTOM") == "CUSTOM-PAIR"


def test_coinbase_protocol_compliance() -> None:
    """CoinbaseProvider implements DataProvider protocol."""
    provider = CoinbaseProvider()
    assert isinstance(provider, DataProvider)


def test_coinbase_get_daily_history_success() -> None:
    """CoinbaseProvider parses candles response and sorts ascending chronologically."""
    # Coinbase returns [ [time, low, high, open, close, volume], ... ] in descending order
    ts_day1 = 1767225600  # 2026-01-01T00:00:00Z (seconds)
    ts_day2 = 1767312000  # 2026-01-02T00:00:00Z (seconds)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [
        [ts_day2, 51000.0, 53000.0, 51500.0, 52800.0, 1500.0],
        [ts_day1, 49000.0, 51000.0, 49500.0, 50800.0, 1200.0],
    ]

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinbaseProvider(client=mock_client)
    start = date(2026, 1, 1)
    end = date(2026, 1, 2)

    bars = provider.get_daily_history("BTC-USD", start=start, end=end)

    assert len(bars) == 2
    assert all(isinstance(b, Bar) for b in bars)
    # Sorted ascending
    assert bars[0].ts == "2026-01-01T00:00:00Z"
    assert bars[0].open == 49500.0
    assert bars[0].high == 51000.0
    assert bars[0].low == 49000.0
    assert bars[0].close == 50800.0
    assert bars[0].volume == 1200.0
    assert bars[0].adj_close is None

    assert bars[1].ts == "2026-01-02T00:00:00Z"
    assert bars[1].open == 51500.0
    assert bars[1].close == 52800.0


def test_coinbase_pagination_chunking() -> None:
    """CoinbaseProvider splits large date ranges (>250 days) into chunked requests."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = []

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinbaseProvider(client=mock_client)
    # 600 days range: should trigger at least 3 chunks
    start = date(2024, 1, 1)
    end = date(2025, 8, 23)

    bars = provider.get_daily_history("BTC-USD", start=start, end=end)
    assert bars == []
    assert mock_client.get.call_count >= 3


def test_coinbase_empty_or_invalid_responses() -> None:
    """CoinbaseProvider gracefully handles empty, invalid, or inverted date ranges."""
    provider = CoinbaseProvider()

    # Inverted date range
    assert (
        provider.get_daily_history(
            "BTC", start=date(2026, 1, 10), end=date(2026, 1, 1)
        )
        == []
    )

    # Empty data parsing
    assert (
        provider._parse_candles([], start=date(2026, 1, 1), end=date(2026, 1, 2)) == []
    )
    assert (
        provider._parse_candles(
            "not a list", start=date(2026, 1, 1), end=date(2026, 1, 2)
        )
        == []
    )


def test_coinbase_error_handling_429() -> None:
    """CoinbaseProvider raises HTTPStatusError on rate limit or 5xx."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests", request=MagicMock(), response=mock_response
    )

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    provider = CoinbaseProvider(client=mock_client)
    with pytest.raises(httpx.HTTPStatusError):
        provider.get_daily_history(
            "BTC-USD", start=date(2026, 1, 1), end=date(2026, 1, 2)
        )


def test_coinbase_get_latest_bars() -> None:
    """CoinbaseProvider returns requested lookback slice of bars."""
    provider = CoinbaseProvider()

    assert provider.get_latest_bars("BTC-USD", lookback=0) == []
    assert provider.get_latest_bars("BTC-USD", lookback=-3) == []

    mock_bars = [
        Bar(f"2026-01-0{i}T00:00:00Z", 200.0, 210.0, 195.0, 205.0, None, 500.0)
        for i in range(1, 8)
    ]
    provider.get_daily_history = MagicMock(return_value=mock_bars)  # type: ignore[assignment]

    latest = provider.get_latest_bars("BTC-USD", lookback=4)
    assert len(latest) == 4
    assert latest[-1].ts == "2026-01-07T00:00:00Z"


# ---------------------------------------------------------------------------
# Ingestion & Fallback Integration Tests
# ---------------------------------------------------------------------------


def test_crypto_ingestion_with_providers(db_session: Session) -> None:
    """IngestionService correctly polls and backfills crypto assets with CoinGecko and Coinbase."""
    # Setup tickers
    btc_ticker = Ticker(
        symbol="BTC-USD",
        asset_class="crypto",
        display_name="Bitcoin",
        provider="coingecko",
        provider_symbol="bitcoin",
        price_basis="raw",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    eth_ticker = Ticker(
        symbol="ETH-USD",
        asset_class="crypto",
        display_name="Ethereum",
        provider="coinbase",
        provider_symbol="ETH-USD",
        price_basis="raw",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    db_session.add(btc_ticker)
    db_session.add(eth_ticker)
    db_session.commit()

    # Mock providers
    coingecko_mock = MagicMock(spec=CoinGeckoProvider)
    coingecko_mock.get_latest_bars.return_value = [
        Bar("2026-01-01T00:00:00Z", 50000.0, 51000.0, 49000.0, 50500.0, None, 10000.0)
    ]

    coinbase_mock = MagicMock(spec=CoinbaseProvider)
    coinbase_mock.get_latest_bars.return_value = [
        Bar("2026-01-01T00:00:00Z", 3000.0, 3100.0, 2950.0, 3050.0, None, 5000.0)
    ]

    repo = BarRepository(db_session)
    service = IngestionService(
        session=db_session,
        providers={
            "coingecko": coingecko_mock,
            "coinbase": coinbase_mock,
        },
        bar_repo=repo,
    )

    # Poll watchlist
    results = service.poll_watchlist()
    assert "BTC-USD" in results
    assert results["BTC-USD"]["inserted"] == 1
    assert "ETH-USD" in results
    assert results["ETH-USD"]["inserted"] == 1

    # Verify bars in repo
    btc_bars = repo.get_latest("BTC-USD", limit=5)
    assert len(btc_bars) == 1
    assert btc_bars[0].close == 50500.0
    assert btc_bars[0].source == "coingecko"

    eth_bars = repo.get_latest("ETH-USD", limit=5)
    assert len(eth_bars) == 1
    assert eth_bars[0].close == 3050.0
    assert eth_bars[0].source == "coinbase"
