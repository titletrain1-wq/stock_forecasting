# stock_forecasting v2 — Real-Time Design Spec

> **Status**: DESIGN — Phase 1, pre-implementation. Stops at GATE 0 (god review, then user approval).
> **Date**: 2026-09-01 · **Build lead**: Toby · **Branch**: `feat/realtime-v2` off `main@46a9515`
> **Evidence base** (`hive/shared/stock_forecasting/`): `v2-research-apis.md` (Creed, API surfaces),
> `v2-research-arch.md` (Meredith, architecture + ML soundness). v1 spec:
> `docs/2026-09-01-stock-forecasting-design.md` (still the source of truth for ML, ledger, evaluator,
> accuracy, indicators; v2 changes only the data + chart + schema + health layer).

---

## 0. Why v2

v1.0.1 is an honest **end-of-day** system: every provider serves daily bars, the worker polls
hourly, and freshness is judged against the NYSE trading calendar. That is correct for the ML
ledger but it never met the user's actual ask: a **live-moving chart** like TradingView / Binance /
Yahoo. Issue 9 = option A ("declare the app not-real-time") made the gap worse.

v2 adds a real-time-capable **display layer** on top of the unchanged daily ML core:

- **Crypto**: genuine real-time via Coinbase's keyless WebSocket.
- **Equities**: ~15-minute-delayed via a yfinance intraday poller (the honest free ceiling; no free
  real-time equity feed exists).
- **ML**: daily models, daily ledger, daily evaluation — **unchanged**. The live price only moves
  the on-screen price line and the current-price header. Forecast ribbons and confidence bands stay
  anchored to the last completed daily close.

### Guiding principle

> The training-data path and the display path are **separate concerns with separate freshness
> rules**. Live data feeds the eyes; daily closes feed the models and the ledger. They meet once a
> day, at the close.

---

## 1. Data layer

### 1.1 Crypto — Coinbase WebSocket (real-time, keyless)

> **M1 spike update (2026-09-01)**: the live keyless probe
> (`docs/spikes/2026-09-01-M1-coinbase-ws.md`) moved the endpoint from the classic
> exchange feed to the Advanced Trade feed, because `ticker_batch` is an
> Advanced-Trade channel and the classic `wss://ws-feed.exchange.coinbase.com`
> does not serve it. The table below reflects the spike's decision.

| Item | Value |
|---|---|
| Endpoint | `wss://advanced-trade-ws.coinbase.com` (Advanced Trade feed) |
| Auth | none — keyless for public channels, confirmed 2026 |
| Channels | `ticker_batch` (batched price ticks, ~5s on change) + `heartbeats` (keepalive for quiet products) |
| Subscribe | one message per channel (`channel` singular): `{"type":"subscribe","product_ids":["BTC-USD","ETH-USD"],"channel":"ticker_batch"}` then `...,"channel":"heartbeats"}` |
| Limits | 100 subscriptions per IP connection |
| Client | `websockets` (already locked; async loop on a single background daemon thread) |

**Ownership**: `worker.py` owns exactly one WebSocket connection (Meredith Q2, option A). Browser
tabs never open sockets. The thread pushes the newest tick per product into a thread-safe in-memory
dict, and a flush loop writes:

- `live_quotes` — one row per ticker, overwritten each tick (the current-price anchor).
- `intraday_bars` — the **forming** bucket for the active interval, updated in place until the
  bucket closes, then left immutable.

### 1.2 Equities — yfinance intraday poller (~15 min delayed)

| Item | Value |
|---|---|
| Interval | `5m` (60-day lookback ceiling; `1m` only reaches 7 days) |
| Delay | ~15 minutes on the free Yahoo web endpoint |
| Poll cadence | every 5 minutes per active equity ticker, fetch only the latest N bars |
| Anti-throttle | real `User-Agent` header; never poll faster than 1–5 min or risk `HTTP 429` / UA block |
| Fallback | none worth naming — document a **"15-min delayed"** badge on every equity chart |

Runs as an APScheduler `interval` job inside `worker.py`, writing closed `5m` bars into
`intraday_bars`. The most recent `5m` bar is treated as **provisional** until
`now >= bar_start + 5m + 15m delay`.

### 1.3 Crypto derivatives — dYdX v4 Indexer (unchanged from v1)

Still keyless, base `https://indexer.dydx.trade/v4`. `GET /historicalFunding/{ticker}` (hourly),
`GET /perpetualMarkets?ticker=` (current OI). Funding is hourly and OI moves slowly, so **stay REST
poll every 5–15 min** via the existing `DydxDerivativesProvider`. No WebSocket.

### 1.4 Failover story

| Feed | Primary | Degraded | Critical |
|---|---|---|---|
| Crypto price | Coinbase WS `ticker_batch` | WS idle > `WS_IDLE_TIMEOUT_SEC` or closed → Coinbase **REST candles** (`GET api.exchange.coinbase.com/products/{id}/candles`, `granularity=60`/`300`; last bucket provisional) | both WS + REST failing → serve last-good `intraday_bars` row + red "live feed down" badge; **daily ledger unaffected** |
| Equity price | yfinance `5m` poll | poll returns `429`/empty → exponential backoff, serve last-good `intraday_bars` + amber badge | > 2 missed poll windows → "delayed feed stale" badge; fall back to the daily `ohlcv_bars` close for the header |
| Derivatives | dYdX REST | existing circuit breaker | last-good `crypto_derivatives` row |

- Per-feed circuit breaker reuses `circuit_breaker.py` (`closed → open` after 5 consecutive
  failures in 10 min; `open` 15 min; `half_open` trial). `link_monitor.py` records RTT / error rate
  into `link_metrics` for the WS and REST fallback as pseudo-providers `coinbase_ws`,
  `coinbase_rest`, `yfinance_intraday`.
- A crypto or equity **display** feed being down is **DEGRADED**, never CRITICAL — the ML core and
  ledger keep working off daily bars.

---

## 2. Schema changes

`ohlcv_bars` (daily), `model_runs`, `prediction_snapshots`, `accuracy_records`, `quarantine_bars`,
`system_heartbeat`, `link_metrics`, `job_queue`, `crypto_derivatives`: **unchanged**. `ohlcv_bars`
stays the single, immutable training + evaluation ledger.

### 2.1 New table — `intraday_bars`

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK → `tickers.symbol` | |
| `interval` | TEXT | `1m` \| `5m` (crypto forming bucket / equity poll) |
| `ts` | TEXT | bucket start, ISO-8601 UTC |
| `open` `high` `low` `close` | REAL | |
| `volume` | REAL | |
| `is_provisional` | INTEGER | 1 = bucket still forming or inside the delay window |
| `source` | TEXT | `coinbase_ws` \| `coinbase_rest` \| `yfinance_intraday` |
| `ingested_at` | TEXT | wall-clock upsert |

**Unique index** `(ticker, interval, ts)`. Ingest = upsert; a provisional row is overwritten each
tick and flipped to `is_provisional = 0` when the bucket closes.
**Retention**: `worker.py` daily job `DELETE FROM intraday_bars WHERE ts < now - INTRADAY_RETENTION_DAYS`
(default 7). Daily bars are never pruned.

### 2.2 New table — `live_quotes`

| col | type | notes |
|---|---|---|
| `ticker` | TEXT PK → `tickers.symbol` | |
| `price` | REAL | last trade / last tick |
| `ts` | TEXT | provider event time, UTC |
| `received_at` | TEXT | wall-clock (drives display-freshness) |
| `source` | TEXT | `coinbase_ws` \| `coinbase_rest` \| `yfinance_intraday` |

One row per ticker, overwritten in place. This is the cheapest possible read for the
`@st.fragment(run_every=...)` current-price widget.

### 2.3 Migration mechanics

`database.create_tables()` already runs `SQLModel.metadata.create_all`. Adding the two models to
`schema.py` creates the tables on next worker/app start. No data backfill: `intraday_bars` fills
forward from first poll; `live_quotes` fills on first tick. A one-line idempotent guard drops
pre-7-day rows on startup.

---

## 3. Streaming chart

### 3.1 Rendering pattern (Streamlit 1.62.0)

- Wrap the live price/quote region in `@st.fragment(run_every=<cadence>)`. Only the decorated
  function re-executes; surrounding controls, range picker, accuracy panel, and scroll position are
  untouched (Meredith Q1). `streamlit-autorefresh` and manual `st.rerun()` are rejected (full-page
  rerun, flicker).
- `st.plotly_chart(fig, key="live_price_chart")` — a **stable key** reuses the existing canvas
  instead of remounting.
- `fig.update_layout(uirevision=True)` — preserves zoom / pan / hover state across tick updates.
- The fragment reads only `live_quotes` + the last ~N `intraday_bars` rows (indexed, WAL reads).
  It never calls a provider.

### 3.2 Poll cadence per asset class

| Asset | Fragment `run_every` | Rationale |
|---|---|---|
| Crypto | `2s` (`LIVE_FRAGMENT_REFRESH_CRYPTO_SEC`) | WS delivers ~5s batches; 2s read keeps the line smooth |
| Equity | `15s` (`LIVE_FRAGMENT_REFRESH_EQUITY_SEC`) | data is 15-min delayed and 5-min-bucketed; faster is pointless |

Worker-side write cadence: crypto WS continuous; equity poll every 5 min; dYdX every 5–15 min;
prune once daily.

### 3.3 Candle-forming behaviour

- The active interval's forming bar renders as a **live candle** built from `live_quotes` (or the
  latest provisional `intraday_bars` row): `open` fixed at bucket start, `high`/`low` tracked,
  `close` = latest tick.
- Rendered with a distinct style (dashed border / lower opacity) and an `is_provisional` marker in
  the hover text until the bucket closes.
- Crypto: bucket = `1m` (or `5m`). Equity: bucket = `5m`, provisional through the 15-min delay
  window.

### 3.4 EOD reconciliation (Meredith Q5)

```
Session close (crypto 00:00 UTC · equity 16:00 ET)
  1. worker finalizes the last intraday bucket (is_provisional -> 0)
  2. worker writes the official daily bar into ohlcv_bars (interval='1d')   [once]
  3. Trainer (nightly) + Forecaster run -> new prediction_snapshots anchored at P_close
  4. Evaluator grades newly-matured snapshots against realized P_close
  5. next fragment refresh: ribbon origin shifts to P_close
```

**Zero visible jump**: at the close `P_live ≈ P_close`, so the new ribbon starts where the intraday
line ended. `ohlcv_bars` is written exactly once per close; `intraday_bars` is read-only to the UI.

---

## 4. Health rework

Two freshness paths, judged by different clocks, surfaced side by side.

### 4.1 DISPLAY path (new) — judged against intraday cadence

| Feed | 🟢 NOMINAL | 🟡 DEGRADED | 🔴 DOWN (display only) |
|---|---|---|---|
| Crypto WS | last tick < 10s AND heartbeat < 3s | tick 10–90s, or on REST fallback | WS + REST both failing |
| Equity intraday | last `5m` bar age < 25 min (5m bucket + 15m delay + margin) | 25–45 min | > 2 missed poll windows |

Backed by `live_quotes.received_at` and `intraday_bars.ts`. New checks in `health_checks.py`:
`check_live_feed_crypto`, `check_live_feed_equity`, `check_ws_connection` (connected / reconnecting /
down), `check_intraday_prune` (retention job heartbeat).

### 4.2 TRAINING-DATA path (unchanged from v1.0.1) — judged against the trading calendar

`market_calendar.classify_bar_freshness` / `bar_is_stale` on `ohlcv_bars` **daily** bars, exactly as
Issue 1 built them. Still drives: `input_is_stale`, retrain deferral, and the ledger's own
NOMINAL / DEGRADED / CRITICAL. `check_freshness` in `health_checks.py` is **scoped to `interval='1d'`
only** so intraday rows never enter its window.

### 4.3 Reconciliation — one system status

```
SYSTEM: [ status ]  =  worst_of( training_data_health , worker_heartbeat )
                       ( display-feed health is reported but CANNOT raise CRITICAL )

Live feed:     🟢 crypto WS 3s   ·  🟡 equity delayed 18m
Training data: 🟢 daily bars current (last NYSE session graded)
Worker:        🟢 heartbeat 4s
```

- Display feed down + training data current → **DEGRADED** ("live chart stale, forecasts unaffected").
- Training data ≥ 2 missed sessions → **CRITICAL** (as today), regardless of live feed.
- `health_view.build_health_view` returns both a `LiveFeedRow` list and the existing
  training/worker rows; `compute_system_status` gains a `display_only` flag that caps a live-feed
  fault at DEGRADED.

---

## 5. ML integration (daily core unchanged)

### 5.1 What the live price may and may not touch

| Element | Source | Updates intraday? |
|---|---|---|
| Current-price header (`$227.41 +0.8%`) | `live_quotes` | **yes** |
| On-chart live price line / forming candle | `live_quotes` + `intraday_bars` | **yes** |
| Forecast ribbon (per-horizon predicted line) | `prediction_snapshots` | **no** — once/day post-close |
| Confidence band (`lower_bound`/`upper_bound`) | `prediction_snapshots`, anchored `P_close` | **no** |
| Historical forecast markers, accuracy panel, explain panel | ledger + `accuracy_records` | **no** |

No intraday features, no intraday retraining, no intraday grading. `forecaster.py`, `trainer.py`,
`features.py`, `evaluator.py`, `accuracy.py` are untouched.

### 5.2 Mandatory disclaimer (verbatim, from Meredith)

> *"Statistical confidence intervals (±1.96 σ_h) and horizon accuracy evaluations are strictly
> calibrated to forecasts anchored at completed daily market closes (P_close). Plotted CI bands
> anchored to live intraday prices (P_live) represent informal visual projections; using P_live as a
> dynamic band origin invalidates the calibrated 95% walk-forward coverage guarantee."*

This text appears in the chart caption, `KNOWN_LIMITATIONS.md`, and `ARCHITECTURE.md`. The band is
**always** anchored to `P_close`; the live line simply moves across the chart toward the static band.

### 5.3 Why not anchor the band to P_live

`σ_h` is the full h-day error variance. Mid-session, remaining uncertainty is `σ_{h-Δ} < σ_h`, so a
`P_live`-anchored band over-states remaining error, and shifting by an intraday move bakes in noise
as trend (anchor-drift bias). Confirmed statistically invalid by Meredith.

---

## 6. Migration — what is reused, changed, deleted

### 6.1 Reused unchanged
`forecaster.py` · `trainer.py` · `features.py` · `evaluator.py` · `accuracy.py` · `panels.py` ·
`circuit_breaker.py` · `link_monitor.py` · `market_calendar.py` (now training-data path only) ·
`providers/{base,yfinance,coinbase,tiingo,finnhub,coingecko,dydx,fake}.py` · all v1 tables.

### 6.2 Changed
| File | Change |
|---|---|
| `schema.py` | + `IntradayBar`, + `LiveQuote` models |
| `config.py` | + settings (§7) |
| `worker.py` | + WS daemon thread lifecycle; + equity intraday poll job; + intraday prune job; + WS-idle → REST fallback; heartbeat rows `live_feed_crypto`, `job_ingest_equity_intraday`, `job_prune_intraday` |
| `ingestion.py` | + `ingest_intraday_bar()`, + `upsert_live_quote()` |
| `bar_store.py` | new sibling `intraday_store.py` (`IntradayRepository`: `upsert_forming`, `close_bucket`, `get_recent`, `prune`) — keeps `BarRepository` daily-only |
| `health_checks.py` | + 4 display-path checks; `check_freshness` scoped to `interval='1d'`; `compute_system_status` gains `display_only` cap |
| `health_view.py` | + `LiveFeedRow`; two-path panel |
| `viz.py` | + live price line + forming candle series; `uirevision=True`; provisional styling |
| `app.py` | wrap live region in `@st.fragment(run_every=...)`; stable chart `key`; "15-min delayed" badge for equities |
| `.env.example` | + new keys |

### 6.3 New modules
- `stock_forecasting/live_feed.py` — `CoinbaseWSClient` (connect, subscribe, parse `ticker_batch`,
  mandatory `heartbeats`, reconnect with exponential backoff, idle-timeout detection) +
  `coinbase_rest_candles()` fallback.
- `stock_forecasting/intraday_store.py` — `IntradayRepository` (see above).

### 6.4 Deleted / retired
Nothing structural. `poll_interval_equity_min` (hourly daily-bar poll) is **retained** for the daily
`ohlcv_bars` top-up but the intraday poller becomes the display source. No files removed.

---

## 7. Config / `.env` additions

```
LIVE_WS_ENABLED=true
COINBASE_WS_URL=wss://advanced-trade-ws.coinbase.com
WS_IDLE_TIMEOUT_SEC=90                 # no tick/heartbeat this long -> REST fallback
INTRADAY_EQUITY_INTERVAL=5m
INTRADAY_POLL_EQUITY_MIN=5
INTRADAY_RETENTION_DAYS=7
LIVE_FRAGMENT_REFRESH_CRYPTO_SEC=2
LIVE_FRAGMENT_REFRESH_EQUITY_SEC=15
```

No new secrets. Coinbase WS + REST and dYdX are keyless; yfinance is keyless. Existing
`TIINGO_API_KEY` / `FINNHUB_API_KEY` / `COINGECKO_API_KEY` stay optional daily-bar fallbacks.

Run is unchanged: `uv run python worker.py` + `uv run streamlit run app.py`.

---

## 8. Explicitly OUT OF SCOPE

- Paid real-time equity feeds (Polygon, IEX paid, Alpaca, Databento). Equities stay ~15-min delayed.
- Intraday forecasting models or intraday horizons. Models remain daily 1d / 5d / 30d.
- Trade signals, buy/sell advice, position sizing, price alerts, notifications.
- Full order book / level-2 / per-trade tick storage. Only `ticker_batch` last-price and OHLC buckets.
- Historical intraday backfill beyond the provider lookback window (1m: 7d, 5m: 60d).
- WebSocket for equities or dYdX.
- Multi-user, auth, hosting, Postgres. Still one machine, one worker, one Streamlit, SQLite.

---

## 9. Milestones (per-milestone DoD: a test that fails on `main@46a9515`, `ruff check`/`format` clean, full `pytest` green)

| M | Deliverable | Definition of done |
|---|---|---|
| **M0** | Schema + config foundation | `IntradayBar` + `LiveQuote` in `schema.py`; new settings in `config.py`; `.env.example` updated. Test: tables exist after `create_tables`, settings parse. Fails on old code (models absent). |
| **M1** | `live_feed.py` — Coinbase WS client | Connect, subscribe `ticker_batch`+`heartbeats`, parse ticks, reconnect w/ exp backoff, idle-timeout flag, `coinbase_rest_candles()` fallback. Tests: fake WS server (tick parse, heartbeat, reconnect), REST fallback shape, last-bucket-provisional. Fails on old code (module absent). |
| **M2** | `intraday_store.py` + worker WS integration | `IntradayRepository` (`upsert_forming`/`close_bucket`/`get_recent`/`prune`); worker starts/stops the WS daemon thread cleanly, flushes ticks to `live_quotes` + forming `intraday_bars`, idle → REST fallback, daily prune job. Tests: thread lifecycle, tick→row, prune deletes > retention, heartbeat rows written. Fails on old worker. |
| **M3** | Equity intraday poller | yfinance `5m` latest-N-bars job in worker, real UA, `429` backoff, provisional-window logic. Tests: fake yfinance intraday fixture → `intraday_bars` rows, delay-window marks provisional, backoff on 429. Fails on old code. |
| **M4** | Health rework | 4 new display-path checks; `check_freshness` scoped to `1d`; `compute_system_status` `display_only` cap; `health_view` two-path rows. Tests: each path independently, reconciliation (display down + training current = DEGRADED not CRITICAL; training stale = CRITICAL regardless). Old combined-path test updated. Fails on old code. |
| **M5** | Streaming chart | `viz.py` live price line + forming candle + `uirevision`; `app.py` `@st.fragment(run_every=...)` per asset class + stable key + delayed badge; EOD ribbon-origin swap. Tests: viz unit (live-line series, forming-candle OHLC, provisional style), fragment smoke, EOD continuity (P_live≈P_close → no gap). Fails on old viz. |
| **M6** | ML overlay integrity guard | Regression test: mutating `live_quotes.price` does **not** move `lower_bound`/`upper_bound` or ribbon points; disclaimer string present in caption + `KNOWN_LIMITATIONS.md`. Fails on old code (no guard, no disclaimer). |
| **M7** | Docs + chaos + `v2.0.0` | `KNOWN_LIMITATIONS.md` (delayed equity, provisional last candle, display-only live, P_close-only calibration), `ARCHITECTURE.md` + `API.md` + `CHANGELOG` updated. Chaos suite: WS drop mid-session, WS silent (heartbeat stops), Mac sleep/wake, equity `429` storm, close-time reconcile jump check. DoD: chaos green, full `pytest`+`ruff`, annotated tag `v2.0.0` local (no push). |

Dependency order: M0 → M1 → M2 → (M3 ∥ M4) → M5 → M6 → M7.

---

## 10. Open items (resolve during implementation, non-blocking)

1. **Forming-bucket interval for crypto** — `1m` (finer live candle) vs `5m` (matches equity, fewer
   rows). Lean `1m`; confirm chart readability in M5.
2. **`live_quotes` vs a column on `intraday_bars`** — separate table chosen for read cheapness;
   revisit if the extra write hurts.
3. **WS thread + APScheduler coexistence** — the WS runs as a raw daemon thread, not an APScheduler
   job (APScheduler is for interval work). Confirm clean shutdown ordering in `worker.stop()`.
4. **Streamlit `@st.fragment` + SQLite WAL under 2s reads** — verify no `database is locked` at the
   fastest cadence with the worker writing every tick; add a read retry if observed (v1 §13.5 noted
   the same risk).
5. **Multiple browser tabs** — each tab runs its own fragment timer reading the DB; harmless (no
   extra sockets), but confirm SQLite read contention is negligible at N=3 tabs.

---

## 11. Spec self-review

- **Placeholders**: none. §10 items are scoped and non-blocking with a lean.
- **Consistency**: v1 tables listed as unchanged match §2; migration file list (§6.2) matches the
  modules named in §1/§3/§4; two-path health (§4) matches the `display_only` cap referenced in §5.1
  and the milestone M4 DoD.
- **Scope**: single implementation plan feasible via M0–M7. Real-time is display-only; ML core is
  explicitly frozen, which bounds the blast radius.
- **Ambiguity**: the one real design fork (band anchor `P_close` vs `P_live`) is resolved with a
  statistical rationale and a verbatim disclaimer, not left open.
- **Boundary**: this document is Phase 1 only. No code, no migration run. Next step after GATE 0
  approval: the `writing-plans` skill turns §9 into a task-level implementation plan.
