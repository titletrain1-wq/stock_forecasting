"""Tests for DydxDerivativesProvider (mocked HTTP — no live calls)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlmodel import Session, select

from stock_forecasting.crypto_store import CryptoDerivativeStore
from stock_forecasting.ingestion import IngestionService
from stock_forecasting.providers.base import Derivative
from stock_forecasting.providers.dydx import DydxDerivativesProvider
from stock_forecasting.schema import CryptoDerivative, Ticker


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Serves canned funding pages + a markets snapshot; records requests."""

    def __init__(
        self, funding_pages: list[list[dict]], open_interest: str | None
    ) -> None:
        self._funding_pages = funding_pages
        self._page_idx = 0
        self._open_interest = open_interest
        self.requests: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None):
        params = params or {}
        self.requests.append((url, params))
        if "perpetualMarkets" in url:
            market = params.get("ticker", "BTC-USD")
            if self._open_interest is None:
                return _FakeResponse({"markets": {}})
            return _FakeResponse(
                {"markets": {market: {"openInterest": self._open_interest}}}
            )
        # historicalFunding
        if self._page_idx < len(self._funding_pages):
            page = self._funding_pages[self._page_idx]
        else:
            page = []
        self._page_idx += 1
        return _FakeResponse({"historicalFunding": page})

    def close(self) -> None:
        return None


def _hour(day: str, hh: int, rate: str) -> dict:
    return {
        "ticker": "BTC-USD",
        "rate": rate,
        "price": "70000",
        "effectiveAt": f"{day}T{hh:02d}:00:00.000Z",
    }


def test_resolve_market() -> None:
    p = DydxDerivativesProvider()
    assert p.resolve_market("BTC-USD") == "BTC-USD"
    assert p.resolve_market("btc") == "BTC-USD"
    assert p.resolve_market("ETHUSD") == "ETH-USD"
    assert p.resolve_market("SOL-USD") == "SOL-USD"


def test_get_derivatives_aggregates_hourly_to_daily() -> None:
    page = [
        _hour("2026-02-02", 0, "0.00001"),
        _hour("2026-02-02", 1, "0.00002"),
        _hour("2026-02-02", 2, "-0.00001"),
        _hour("2026-02-01", 23, "0.00005"),
        _hour("2026-02-01", 22, "0.00005"),
    ]
    client = _FakeClient([page, []], open_interest="201.99")
    p = DydxDerivativesProvider(client=client)

    out = p.get_derivatives("BTC-USD", date(2026, 2, 1), date(2026, 2, 2))
    assert [d.ts for d in out] == [
        "2026-02-01T00:00:00Z",
        "2026-02-02T00:00:00Z",
    ]
    assert out[0].funding_rate == pytest.approx(0.0001)
    assert out[1].funding_rate == pytest.approx(0.00002)
    # open interest only on the newest day
    assert out[0].open_interest is None
    assert out[1].open_interest == pytest.approx(201.99)
    assert all(isinstance(d, Derivative) for d in out)


def test_get_derivatives_filters_out_of_range_days() -> None:
    page = [
        _hour("2026-02-05", 0, "0.001"),  # out of range
        _hour("2026-02-02", 0, "0.001"),
        _hour("2026-01-20", 0, "0.001"),  # out of range
    ]
    client = _FakeClient([page, []], open_interest="10")
    p = DydxDerivativesProvider(client=client)
    out = p.get_derivatives("BTC-USD", date(2026, 2, 1), date(2026, 2, 3))
    assert [d.ts for d in out] == ["2026-02-02T00:00:00Z"]


def test_get_derivatives_start_after_end() -> None:
    p = DydxDerivativesProvider(client=_FakeClient([], None))
    assert p.get_derivatives("BTC-USD", date(2026, 2, 5), date(2026, 2, 1)) == []


def test_get_derivatives_empty_history() -> None:
    client = _FakeClient([[]], open_interest="10")
    p = DydxDerivativesProvider(client=client)
    assert p.get_derivatives("BTC-USD", date(2026, 2, 1), date(2026, 2, 3)) == []


def test_get_derivatives_survives_missing_open_interest() -> None:
    page = [_hour("2026-02-02", 0, "0.001")]
    client = _FakeClient([page, []], open_interest=None)
    p = DydxDerivativesProvider(client=client)
    out = p.get_derivatives("BTC-USD", date(2026, 2, 1), date(2026, 2, 3))
    assert out[0].open_interest is None
    assert out[0].funding_rate == pytest.approx(0.001)


def test_pagination_walks_backwards_until_start_covered() -> None:
    page1 = [_hour("2026-02-10", h, "0.0001") for h in range(3)]
    page2 = [_hour("2026-02-09", h, "0.0001") for h in range(3)]
    page3 = [_hour("2026-02-08", h, "0.0001") for h in range(3)]
    client = _FakeClient([page1, page2, page3, []], open_interest="5")
    p = DydxDerivativesProvider(client=client, page_limit=3)
    out = p.get_derivatives("BTC-USD", date(2026, 2, 8), date(2026, 2, 10))
    assert [d.ts for d in out] == [
        "2026-02-08T00:00:00Z",
        "2026-02-09T00:00:00Z",
        "2026-02-10T00:00:00Z",
    ]
    funding_reqs = [r for r in client.requests if "historicalFunding" in r[0]]
    assert len(funding_reqs) >= 3


# ---- CryptoDerivativeStore + IngestionService.poll_derivatives ----


def _crypto_ticker(session, symbol="BTC-USD"):
    t = Ticker(
        symbol=symbol,
        asset_class="crypto",
        display_name=symbol,
        provider="coingecko",
        provider_symbol=symbol,
        price_basis="raw",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    session.add(t)
    session.commit()
    return t


def test_crypto_store_upsert_idempotent_and_null_preserving(
    db_session: Session,
) -> None:
    _crypto_ticker(db_session)
    store = CryptoDerivativeStore(db_session)

    rows = [
        Derivative(ts="2026-02-01T00:00:00Z", funding_rate=0.001, open_interest=None),
        Derivative(ts="2026-02-02T00:00:00Z", funding_rate=0.002, open_interest=555.0),
    ]
    assert store.upsert("BTC-USD", rows) == 2
    # re-upsert same ts: 0 new; a None open_interest must not clobber the stored 555
    again = [
        Derivative(ts="2026-02-02T00:00:00Z", funding_rate=0.009, open_interest=None),
    ]
    assert store.upsert("BTC-USD", again) == 0
    stored = db_session.exec(
        select(CryptoDerivative).where(CryptoDerivative.ts == "2026-02-02T00:00:00Z")
    ).first()
    assert stored.funding_rate == pytest.approx(0.009)
    assert stored.open_interest == pytest.approx(555.0)


def test_ingestion_poll_derivatives_writes_crypto_derivatives(
    db_session: Session,
) -> None:
    _crypto_ticker(db_session)
    page = [_hour("2026-02-02", h, "0.0001") for h in range(3)]
    provider = DydxDerivativesProvider(
        client=_FakeClient([page, []], open_interest="42")
    )
    svc = IngestionService(
        session=db_session,
        providers={},
        derivatives_provider=provider,
    )
    res = svc.poll_derivatives("BTC-USD", lookback_days=3650)
    assert res["inserted"] >= 1
    rows = db_session.exec(select(CryptoDerivative)).all()
    assert rows and all(r.ticker == "BTC-USD" and r.source == "dydx" for r in rows)


def test_ingestion_poll_derivatives_rejects_equity(db_session: Session) -> None:
    t = Ticker(
        symbol="AAPL",
        asset_class="equity",
        display_name="Apple",
        provider="yfinance",
        provider_symbol="AAPL",
        price_basis="adjusted",
        added_at="2026-01-01T00:00:00Z",
        active=1,
    )
    db_session.add(t)
    db_session.commit()
    svc = IngestionService(
        session=db_session,
        providers={},
        derivatives_provider=DydxDerivativesProvider(client=_FakeClient([], None)),
    )
    res = svc.poll_derivatives("AAPL")
    assert res["inserted"] == 0
    assert "not a crypto ticker" in res["error"]


def test_ingestion_poll_derivatives_no_provider(db_session: Session) -> None:
    _crypto_ticker(db_session)
    svc = IngestionService(session=db_session, providers={})
    res = svc.poll_derivatives("BTC-USD")
    assert res["inserted"] == 0
    assert "no derivatives provider" in res["error"]
