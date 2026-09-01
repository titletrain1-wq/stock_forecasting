"""Unit tests for the intraday + live-quote repositories (v2 M2.1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stock_forecasting.intraday_store import IntradayRepository, LiveQuoteRepository


def test_bucket_start_floors_to_interval() -> None:
    r = IntradayRepository.__new__(IntradayRepository)
    assert (
        r.bucket_start("2026-09-01T10:16:18.822710Z", "1m")
        == "2026-09-01T10:16:00+00:00"
    )
    assert (
        r.bucket_start("2026-09-01T10:16:18+00:00", "5m") == "2026-09-01T10:15:00+00:00"
    )


def test_upsert_forming_tracks_ohlc(db_session) -> None:
    r = IntradayRepository(db_session)
    b = "2026-09-01T00:00:00+00:00"
    r.upsert_forming("BTC-USD", "1m", b, price=100.0)
    r.upsert_forming("BTC-USD", "1m", b, price=105.0)
    r.upsert_forming("BTC-USD", "1m", b, price=98.0)
    rows = r.get_recent("BTC-USD", "1m")
    assert len(rows) == 1
    row = rows[-1]
    assert (row.open, row.high, row.low, row.close) == (100.0, 105.0, 98.0, 98.0)
    assert row.is_provisional == 1


def test_close_bucket_flips_provisional(db_session) -> None:
    r = IntradayRepository(db_session)
    b = "2026-09-01T00:00:00+00:00"
    r.upsert_forming("BTC-USD", "1m", b, price=100.0)
    r.close_bucket("BTC-USD", "1m", b)
    assert r.get_recent("BTC-USD", "1m")[-1].is_provisional == 0


def test_get_recent_orders_oldest_last_and_limits(db_session) -> None:
    r = IntradayRepository(db_session)
    for i in range(5):
        r.upsert_forming(
            "BTC-USD", "1m", f"2026-09-01T00:0{i}:00+00:00", price=float(i)
        )
    rows = r.get_recent("BTC-USD", "1m", limit=3)
    assert [row.ts for row in rows] == [
        "2026-09-01T00:02:00+00:00",
        "2026-09-01T00:03:00+00:00",
        "2026-09-01T00:04:00+00:00",
    ]


def test_prune_deletes_only_rows_older_than_retention(db_session) -> None:
    r = IntradayRepository(db_session)
    now = datetime(2026, 9, 15, tzinfo=UTC)
    old = (now - timedelta(days=8)).isoformat()
    fresh = (now - timedelta(days=2)).isoformat()
    r.upsert_forming("BTC-USD", "1m", old, price=1.0)
    r.upsert_forming("BTC-USD", "1m", fresh, price=2.0)
    deleted = r.prune(older_than_days=7, now=now)
    assert deleted == 1
    remaining = r.get_recent("BTC-USD", "1m")
    assert [row.ts for row in remaining] == [fresh]


def test_live_quote_repository_upsert_keeps_one_row(db_session) -> None:
    r = LiveQuoteRepository(db_session)
    r.upsert(
        "BTC-USD", price=100.0, ts="2026-09-01T00:00:00+00:00", source="coinbase_ws"
    )
    r.upsert(
        "BTC-USD", price=101.5, ts="2026-09-01T00:00:05+00:00", source="coinbase_ws"
    )
    r.upsert(
        "ETH-USD", price=50.0, ts="2026-09-01T00:00:05+00:00", source="coinbase_ws"
    )
    assert r.get("BTC-USD").price == 101.5
    assert {q.ticker for q in r.get_all()} == {"BTC-USD", "ETH-USD"}
