# M2 spike — SQLite-WAL per-tick write vs 2s fragment read

**Date**: 2026-09-01 · **God condition 2**: spike the contention BEFORE building
`IntradayRepository` on the per-tick write path. If it forces an in-memory flush
loop or any structural change to the write model, STOP and escalate.

## Setup

Real on-disk SQLite via `stock_forecasting.database.get_engine` +
`create_tables` (`PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`). Fresh DB,
2 crypto tickers seeded. A `database is locked` / `database is busy` on either
side is counted, not raised.

## Run A — the exact scenario in the condition

Writer: one `live_quotes` upsert per ticker every 200ms (~10 writes/s).
Reader: `SELECT * FROM live_quotes` every 2s (the `@st.fragment` read). 45s.

```
writes=434  write_lock_errors=0  write_p50=1.65ms  write_p95=3.85ms  write_max=10.99ms
reads=23    read_lock_errors=0   read_p50=0.85ms   read_p95=2.23ms   read_max=2.28ms
journal_mode=wal  busy_timeout=5000
```

## Run B — stress variant (harder than production)

Writer: **both** `live_quotes` upsert **and** `intraday_bars` forming-bucket
upsert (read-modify-write of high/low/close) per ticker every **100ms** (~20 writes/s).
Readers: **3 concurrent** threads (3 browser tabs), each reading `live_quotes`
**and** the last 200 `intraday_bars` rows every 2s. 30s.

```
writes=550  write_lock=0  w_p50=2.41ms  w_p95=5.29ms  w_max=53.96ms
reads=45    read_lock=0   r_p50=0.78ms  r_p95=2.54ms  r_max=2.71ms
```

The single 54ms write outlier is a WAL checkpoint / GC pause, absorbed well within
`busy_timeout`. Zero lock errors across 550 writes and 45 multi-table reads.

## Verdict — **CLEAN. Proceed with per-tick DB writes as designed. No escalation.**

- Coinbase `ticker_batch` batches every **~5s** (M1 spike), so real write load is
  ~25× lighter than Run A and ~50× lighter than Run B.
- WAL lets readers and the single writer proceed concurrently; `busy_timeout=5000`
  covers the rare checkpoint pause. Even 3 tabs + dual-table writes never blocked.
- No in-memory shared-state flush loop needed. `IntradayRepository` and
  `LiveQuoteRepository` (M2.1) are built directly on `Session` + upsert, and
  `worker._on_tick` (M2.2) writes straight through per tick.

## Note for M2.2 implementation

Keep each tick's write in its own short `Session`/`commit` (as the spike did) so a
tick write never holds a transaction open across the next `recv()`. The forming
`intraday_bars` upsert is a read-modify-write; a single writer thread (the WS
daemon) means no writer-writer race, but still use one transaction per tick.
