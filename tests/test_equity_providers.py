"""Unit tests for TiingoProvider and FinnhubProvider equity fallback data sources."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest

from stock_forecasting.providers.base import Bar, DataProvider
from stock_forecasting.providers.finnhub import FinnhubProvider
from stock_forecasting.providers.tiingo import TiingoProvider

# ---------------------------------------------------------------------------
# Shared canned payloads
# ---------------------------------------------------------------------------

# Tiingo returns a JSON list of daily price objects (here deliberately out of
# chronological order to exercise the ascending sort).
TIINGO_PAYLOAD = [
    {
        "date": "2026-01-02T00:00:00.000Z",
        "open": 102.0,
        "high": 106.0,
        "low": 101.0,
        "close": 105.0,
        "volume": 2000,
        "adjClose": 104.5,
        "adjOpen": 101.5,
        "adjHigh": 105.5,
        "adjLow": 100.5,
        "adjVolume": 2000,
    },
    {
        "date": "2026-01-01T00:00:00.000Z",
        "open": 100.0,
        "high": 104.0,
        "low": 99.0,
        "close": 103.0,
        "volume": 1500,
        "adjClose": 102.5,
        "adjOpen": 99.5,
        "adjHigh": 103.5,
        "adjLow": 98.5,
        "adjVolume": 1500,
    },
]

# Finnhub epoch seconds (UTC midnight) for 2026-01-01 / 02 / 03, out of order.
_EPOCH_D1 = 1767225600  # 2026-01-01T00:00:00Z
_EPOCH_D2 = 1767312000  # 2026-01-02T00:00:00Z
_EPOCH_D3 = 1767398400  # 2026-01-03T00:00:00Z

FINNHUB_PAYLOAD = {
    "s": "ok",
    "t": [_EPOCH_D2, _EPOCH_D1, _EPOCH_D3],
    "o": [102.0, 100.0, 105.0],
    "h": [106.0, 104.0, 108.0],
    "l": [101.0, 99.0, 104.0],
    "c": [105.0, 103.0, 107.0],
    "v": [2000, 1500, 2500],
}


def _mock_client(payload: object) -> MagicMock:
    """Build a MagicMock httpx.Client whose .get(...) returns `payload` as JSON."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = payload

    client = MagicMock(spec=httpx.Client)
    client.get.return_value = response
    return client


# ---------------------------------------------------------------------------
# TiingoProvider
# ---------------------------------------------------------------------------


def test_tiingo_protocol_compliance() -> None:
    """TiingoProvider implements the DataProvider protocol."""
    assert isinstance(TiingoProvider(), DataProvider)


def test_tiingo_get_daily_history_parses_and_sorts() -> None:
    """A normal multi-row payload parses into sorted Bars with correct mapping."""
    client = _mock_client(TIINGO_PAYLOAD)
    provider = TiingoProvider(client=client)

    bars = provider.get_daily_history(
        "AAPL", start=date(2026, 1, 1), end=date(2026, 1, 2)
    )

    assert [b.ts for b in bars] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]
    assert all(isinstance(b, Bar) for b in bars)

    first = bars[0]
    assert first.open == 100.0
    assert first.high == 104.0
    assert first.low == 99.0
    assert first.close == 103.0
    assert first.volume == 1500.0
    assert first.adj_close == 102.5

    assert bars[1].close == 105.0
    assert bars[1].adj_close == 104.5


def test_tiingo_request_shape() -> None:
    """The request hits /daily/<symbol>/prices with the documented query params."""
    client = _mock_client(TIINGO_PAYLOAD)
    provider = TiingoProvider(client=client)

    provider.get_daily_history("AAPL", start=date(2026, 1, 1), end=date(2026, 1, 3))

    args, kwargs = client.get.call_args
    assert args[0] == "https://api.tiingo.com/tiingo/daily/AAPL/prices"
    assert kwargs["params"] == {
        "startDate": "2026-01-01",
        "endDate": "2026-01-03",
        "format": "json",
    }


def test_tiingo_symbol_is_path_encoded() -> None:
    """Symbols with URL-unsafe characters are percent-encoded in the path."""
    client = _mock_client([])
    provider = TiingoProvider(client=client)

    provider.get_daily_history("BRK/B", start=date(2026, 1, 1), end=date(2026, 1, 3))

    args, _ = client.get.call_args
    assert args[0] == "https://api.tiingo.com/tiingo/daily/BRK%2FB/prices"


def test_tiingo_date_range_filtering() -> None:
    """Rows outside [start, end] are dropped."""
    client = _mock_client(TIINGO_PAYLOAD)
    provider = TiingoProvider(client=client)

    bars = provider.get_daily_history(
        "AAPL", start=date(2026, 1, 2), end=date(2026, 1, 2)
    )

    assert [b.ts for b in bars] == ["2026-01-02T00:00:00Z"]


def test_tiingo_start_after_end_returns_empty() -> None:
    """start > end short-circuits to []."""
    client = _mock_client(TIINGO_PAYLOAD)
    provider = TiingoProvider(client=client)

    assert (
        provider.get_daily_history(
            "AAPL", start=date(2026, 1, 10), end=date(2026, 1, 1)
        )
        == []
    )
    client.get.assert_not_called()


def test_tiingo_get_latest_bars_slice() -> None:
    """get_latest_bars returns the last `lookback` bars; lookback<=0 returns []."""
    provider = TiingoProvider()
    assert provider.get_latest_bars("AAPL", lookback=0) == []
    assert provider.get_latest_bars("AAPL", lookback=-2) == []

    mock_bars = [
        Bar(f"2026-01-0{i}T00:00:00Z", 100.0, 105.0, 99.0, 102.0, 101.0, 1000.0)
        for i in range(1, 8)
    ]
    provider.get_daily_history = MagicMock(return_value=mock_bars)  # type: ignore[method-assign]

    latest = provider.get_latest_bars("AAPL", lookback=3)
    assert [b.ts for b in latest] == [
        "2026-01-05T00:00:00Z",
        "2026-01-06T00:00:00Z",
        "2026-01-07T00:00:00Z",
    ]


def test_tiingo_authorization_header_sent_when_api_key_set() -> None:
    """A configured api_key produces an `Authorization: Token <key>` header."""
    client = TiingoProvider(api_key="secret-key")._get_client()
    try:
        assert client.headers["Authorization"] == "Token secret-key"
    finally:
        client.close()


def test_tiingo_no_authorization_header_without_api_key() -> None:
    """Without an api_key no Authorization header is attached."""
    client = TiingoProvider()._get_client()
    try:
        assert "Authorization" not in client.headers
    finally:
        client.close()


def test_tiingo_raise_for_status_propagates() -> None:
    """An HTTP error status raises rather than being swallowed."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=MagicMock()
    )
    client = MagicMock(spec=httpx.Client)
    client.get.return_value = response

    provider = TiingoProvider(client=client)
    with pytest.raises(httpx.HTTPStatusError):
        provider.get_daily_history("AAPL", start=date(2026, 1, 1), end=date(2026, 1, 3))


# ---------------------------------------------------------------------------
# FinnhubProvider
# ---------------------------------------------------------------------------


def test_finnhub_protocol_compliance() -> None:
    """FinnhubProvider implements the DataProvider protocol."""
    assert isinstance(FinnhubProvider(), DataProvider)


def test_finnhub_get_daily_history_parses_and_sorts() -> None:
    """The columnar candle payload parses into sorted Bars with correct mapping."""
    client = _mock_client(FINNHUB_PAYLOAD)
    provider = FinnhubProvider(client=client)

    bars = provider.get_daily_history(
        "AAPL", start=date(2026, 1, 1), end=date(2026, 1, 3)
    )

    assert [b.ts for b in bars] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
    ]

    first = bars[0]
    assert first.open == 100.0
    assert first.high == 104.0
    assert first.low == 99.0
    assert first.close == 103.0
    assert first.volume == 1500.0
    assert first.adj_close is None

    assert bars[2].close == 107.0


def test_finnhub_request_shape() -> None:
    """The request hits /stock/candle with resolution=D and epoch-second bounds."""
    client = _mock_client(FINNHUB_PAYLOAD)
    provider = FinnhubProvider(api_key="k", client=client)

    provider.get_daily_history("AAPL", start=date(2026, 1, 1), end=date(2026, 1, 2))

    args, kwargs = client.get.call_args
    assert args[0] == "https://finnhub.io/api/v1/stock/candle"
    params = kwargs["params"]
    assert params["symbol"] == "AAPL"
    assert params["resolution"] == "D"
    assert params["token"] == "k"
    assert params["from"] == _EPOCH_D1
    assert isinstance(params["from"], int)
    # end-of-day of `end` (2026-01-02T23:59:59Z)
    assert params["to"] == _EPOCH_D2 + 86399
    assert isinstance(params["to"], int)


def test_finnhub_date_range_filtering() -> None:
    """Candles outside [start, end] are dropped."""
    client = _mock_client(FINNHUB_PAYLOAD)
    provider = FinnhubProvider(client=client)

    bars = provider.get_daily_history(
        "AAPL", start=date(2026, 1, 1), end=date(2026, 1, 2)
    )

    assert [b.ts for b in bars] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]


def test_finnhub_start_after_end_returns_empty() -> None:
    """start > end short-circuits to []."""
    client = _mock_client(FINNHUB_PAYLOAD)
    provider = FinnhubProvider(client=client)

    assert (
        provider.get_daily_history(
            "AAPL", start=date(2026, 1, 10), end=date(2026, 1, 1)
        )
        == []
    )
    client.get.assert_not_called()


def test_finnhub_no_data_status_returns_empty() -> None:
    """The HTTP-200 `{"s": "no_data"}` quirk returns [] without raising or indexing."""
    client = _mock_client({"s": "no_data"})
    provider = FinnhubProvider(client=client)

    assert (
        provider.get_daily_history(
            "BOGUS", start=date(2026, 1, 1), end=date(2026, 1, 3)
        )
        == []
    )


def test_finnhub_get_latest_bars_slice() -> None:
    """get_latest_bars returns the last `lookback` bars; lookback<=0 returns []."""
    provider = FinnhubProvider()
    assert provider.get_latest_bars("AAPL", lookback=0) == []
    assert provider.get_latest_bars("AAPL", lookback=-2) == []

    mock_bars = [
        Bar(f"2026-01-0{i}T00:00:00Z", 100.0, 105.0, 99.0, 102.0, None, 1000.0)
        for i in range(1, 8)
    ]
    provider.get_daily_history = MagicMock(return_value=mock_bars)  # type: ignore[method-assign]

    latest = provider.get_latest_bars("AAPL", lookback=3)
    assert [b.ts for b in latest] == [
        "2026-01-05T00:00:00Z",
        "2026-01-06T00:00:00Z",
        "2026-01-07T00:00:00Z",
    ]


def test_finnhub_raise_for_status_propagates() -> None:
    """A genuine HTTP error status still raises."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=MagicMock()
    )
    client = MagicMock(spec=httpx.Client)
    client.get.return_value = response

    provider = FinnhubProvider(client=client)
    with pytest.raises(httpx.HTTPStatusError):
        provider.get_daily_history("AAPL", start=date(2026, 1, 1), end=date(2026, 1, 3))
