# stock_forecasting — Design Spec

> **Status**: DESIGN — approved brainstorm, pre-implementation.
> **Date**: 2026-09-01 · **Owner**: god (synthesis) · **Build lead**: Toby
> **Sources** (`hive/shared/stock_forecasting/`): `BRAINSTORM_SUMMARY.md`, `research-market.md`,
> `research-ml.md`, `research-indicators.md`, `research-reliability.md`, `architecture-draft.md` (rev 2).
> This spec is the single source of truth; the research docs are its evidence base.

---

## Recorded deviations — v1.0.1 (2026-09-01)

The post-release health analysis
(`docs/reports/2026-09-01-postrelease-health-analysis.md`) found that the
"expected next bar" model below was written for an intraday/streaming feed, but
every provider only serves **daily** bars. v1.0.1 adopts an explicit
**end-of-day** operating model (Issue 9, option A). The following clauses are
superseded:

- **§2 / §11 poll cadence** ("crypto ~60s, equities ~5min", "2,880 calls/day"):
  the worker now polls **hourly** (`poll_interval_crypto_sec=3600`,
  `poll_interval_equity_min=60`). Daily bars do not change faster; sub-minute
  polling only re-fetched an unchanged bar and burned the free-tier budget.
- **§6 check #1 Freshness** ("🟢 <20m stock / <5m crypto · 🔴 >1h in-hours"):
  freshness is judged against the **trading calendar**
  (`stock_forecasting/market_calendar.py`) — NYSE sessions via
  `pandas-market-calendars` for equities, one bar per UTC day for crypto — not a
  wall-clock age. CRITICAL only when a bar is genuinely overdue (equity: ≥2
  missed sessions; crypto: ≥3 days behind).
- **`input_is_stale`** ("data > 1h old at prediction time"): computed from the
  anchor bar's calendar freshness (any non-NOMINAL state), not a 1-hour age.

Everything else in the spec stands.

---

## 1. Overview

A **personal, single-user, no-auth** web app that forecasts **US stocks and crypto** at fixed
horizons (**+1d / +5d / +30d**), **persists every forecast immutably**, grades each past forecast
against realized price, and shows **predicted-vs-actual on one chart** plus a **per-horizon
accuracy panel** so the user knows which horizon/model to trust. Includes a **telecom-style
data-feed health monitor**. ML is **simple and explainable first** (Ridge + Random Forest).

### Goals
1. **Correct, immutable prediction ledger** — the product. Every forecast recorded with full
   provenance; graded automatically after its target date.
2. **Honest accuracy reporting** — walk-forward validation, no lookahead, directional hit-rate as
   the headline metric.
3. **Explainability** — every forecast shows which features drove it.
4. **Reliability & observability** — the app degrades gracefully, self-heals, and surfaces its own
   health; a single user leaving it running for weeks should be able to trust it isn't silently rotting.
5. **Low ops** — one machine, `streamlit run` + one worker process, SQLite, `.env`.

### Non-goals
Real-time/tick data · order execution / brokerage · portfolio / P&L / position sizing ·
multi-user / auth / hosting · deep learning · intraday bars (MVP) · mobile app.

### Locked decisions (from `BRAINSTORM_SUMMARY.md` §7 + §11)
| Topic | Decision |
|---|---|
| Stack | Streamlit monolith (Streamlit + `streamlit-lightweight-charts` + `worker.py` + SQLite + shared service layer). React+FastAPI = documented fallback only. |
| Model scope | Per-ticker models. |
| Horizons | Fixed 1d / 5d / 30d. |
| Target | Log return `r_{t+h} = ln(P_{t+h}/P_t)`, reconstruct price `P̂ = P_t·exp(r̂)`. |
| Models | `ridge` (baseline) + `random_forest` (upgrade). LightGBM later. |
| Validation | Walk-forward rolling window. |
| Retrain cadence | Auto nightly after the daily bar closes + manual "Run forecast now" button. |
| Backfill | 5 years of daily history per ticker. |
| Trust threshold | Verdict = trustworthy when `dir_acc ≥ 0.55 AND n ≥ 30`. |
| Data — equities | yfinance primary; Tiingo / Finnhub cached fallback. |
| Data — crypto | CoinGecko primary; Coinbase public fallback. |
| Data — crypto derivatives | dYdX v4 Indexer API (funding rate + open interest, BTC/ETH). |
| Runtime | Mac, this machine, leave a terminal open. APScheduler in-process, no `launchd`. |
| Circuit breaker + auto-failover | In MVP. |
| Health panel | Full panel in MVP. |
| TA library | `pandas-ta` (pure Python). |

### Project location
New standalone git repo at **`~/Desktop/stock_forecasting/`** (not inside `ACG_package`). This
spec stays in `ACG_package/docs/superpowers/specs/` for the hive's reference; copy it into the new
repo's `docs/` on init.

---

## 2. Architecture

```
                 ┌───────────────────────── worker.py (one process) ─────────────────────────┐
 free data APIs  │  APScheduler jobs:                                                        │
 (yfinance,      │   • ingest_poll      — crypto ~60s, equities ~5min                        │
  CoinGecko,     │   • retrain_nightly  — after daily bar close, per ticker×horizon×model    │
  Coinbase,      │   • evaluate_hourly  — grade matured prediction_snapshots                 │
  dYdX)          │   • heartbeat        — write system_heartbeat every tick                  │
                 │                              │ all via ↓                                  │
                 └──────────────────────────────┼───────────────────────────────────────────┘
                                                │
          ┌─────────────────────── shared service layer (plain Python) ───────────────────────┐
          │  providers/  ·  ingestion  ·  bar_store  ·  features  ·  trainer  ·  forecaster    │
          │  evaluator  ·  link_monitor  ·  circuit_breaker  ·  accuracy                       │
          │  (each = plain functions/classes taking a DB session — no Streamlit, no HTTP)      │
          └──────────────────────────────────┬──────────────────────────────────────────────┘
                                             │
                                    SQLite (WAL, busy_timeout=5000)
                                             │
          ┌──────────────────────────────────┴──────────────────────────────────────────────┐
          │  app.py — Streamlit UI (reads the service layer directly, never the providers)   │
          │   watchlist · price+overlay chart · accuracy panel · explain panel · health panel │
          └─────────────────────────────────────────────────────────────────────────────────┘
```

- **`worker.py`** owns all scheduled work and all outbound API calls. It is the *only* process
  that hits providers on a schedule.
- **`app.py`** (Streamlit) is read-mostly: it reads the DB via the service layer, and can enqueue
  an on-demand job (writes a row to a `job_queue` table the worker polls, or calls the service
  function directly in-process for the "Run forecast now" button — see §4.6).
- **Service layer** has no Streamlit or HTTP imports, so the React+FastAPI fallback is a drop-in
  (wrap the same functions in routes).
- If `worker.py` is not running, `app.py` still renders last-known-good data with a **CRITICAL**
  health banner and a "Start worker" hint.

### Tech
Python 3.12 · Streamlit · `streamlit-lightweight-charts` · `pandas`, `numpy`, `pandas-ta` ·
`scikit-learn` (Ridge, RandomForest) · `joblib` · `APScheduler` · `SQLModel` (SQLAlchemy 2 +
Pydantic) · SQLite · `httpx` · `tenacity` · `pydantic-settings` · `pandas-market-calendars` ·
`pytest` · `ruff` · `uv`.

---

## 3. Data model (SQLite, times = ISO-8601 UTC text)

### 3.1 `tickers`
| col | type | notes |
|---|---|---|
| `symbol` | TEXT PK | `AAPL`, `BTC-USD` |
| `asset_class` | TEXT | `equity` \| `crypto` — picks provider + calendar |
| `display_name` | TEXT | |
| `provider` | TEXT | primary provider id |
| `provider_symbol` | TEXT | provider's own id (CoinGecko `bitcoin`, dYdX `BTC-USD`) |
| `price_basis` | TEXT | `adjusted` \| `raw` |
| `added_at` | TEXT | |
| `active` | INTEGER | 1 = polled |

### 3.2 `ohlcv_bars`
| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK | |
| `interval` | TEXT | `1d` (MVP) |
| `ts` | TEXT | bar timestamp from provider (UTC) |
| `open` `high` `low` `close` | REAL | |
| `adj_close` | REAL NULL | if provider supplies |
| `volume` | REAL | |
| `source` | TEXT | provider id (audit) |
| `ingested_at` | TEXT | wall-clock insert |

**Unique index** `(ticker, interval, ts)`. Ingest = `INSERT ... ON CONFLICT DO UPDATE` (upsert).

### 3.3 `crypto_derivatives`
| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK | crypto only |
| `ts` | TEXT | UTC |
| `funding_rate` | REAL NULL | dYdX |
| `open_interest` | REAL NULL | dYdX |
| `source` | TEXT | `dydx` |

**Unique index** `(ticker, ts)`.

### 3.4 `model_runs` — one row per training fit (reproducibility anchor)
`id` PK · `ticker` · `horizon` (`1d`/`5d`/`30d`) · `model_type` (`ridge`/`random_forest`) ·
`model_version` (semver) · `code_git_sha` · `trained_at` · `train_start` `train_end` ·
`hyperparams_json` · `feature_list_json` · `random_seed` · `wf_mae` `wf_rmse` `wf_dir_acc`
`wf_ci_cov` (walk-forward OOS scores) · `residual_std` (feeds CI band) · `artifact_path` (`.joblib`)
· `is_active` (1 = the artifact `/forecast` serves).

### 3.5 `prediction_snapshots` — THE LEDGER (append-only)
`prediction_id` PK (UUID) · `ticker` · `made_at` · `made_from_ts` (last bar the features used =
`P_t` anchor) · `anchor_price` · `horizon` · `target_ts` (calendar-aware for equities) ·
`predicted_return` · `predicted_price` · `lower_bound` `upper_bound` (~95% CI) · `model_type` ·
`model_version` · `model_run_id` FK · `explain_json` (per-feature contributions) ·
`input_is_stale` (INTEGER — features built from data > 1h old at prediction time)
**— evaluator writes ONCE, never updated after —**
`realized_price` · `realized_return` · `evaluated_at` · `error_abs` (`|r̂−r|`) ·
`error_signed` (`r̂−r`) · `is_direction_hit` · `is_within_ci` · `eval_attempts` (INT DEFAULT 0).

Indexes: `(ticker, horizon, model_type, made_at)`; partial `(target_ts) WHERE evaluated_at IS NULL`.
**Guardrail**: repository exposes no `update` for prediction columns; a SQLite trigger rejects
UPDATE of any prediction column (realized columns exempted).

### 3.6 `accuracy_records` — derived cache (rebuildable from the ledger)
`id` PK · `scope` (`ticker`/`global`) · `ticker` NULL · `horizon` · `model_type` · `n` ·
`mae` `rmse` (on returns) · `dir_acc` · `ci_coverage` · `mae_price_pct` · `window` (`all` MVP;
`90d`/`30d` later) · `is_trustworthy` (INT — `dir_acc ≥ 0.55 AND n ≥ 30`) · `updated_at`.
Unique `(scope, ticker, horizon, model_type, window)`.

### 3.7 `quarantine_bars` — rejected rows for review
`id` PK · `ticker` · `raw_json` · `reason` (`price_le_0`/`ohlc_inconsistent`/`spike`/`schema`/
`future_dated`) · `provider` · `detected_at`.

### 3.8 `system_heartbeat` — watchdog
`job_type` PK (`ingest`/`retrain`/`evaluate`/`heartbeat`) · `worker_pid` · `last_pulse_ts` ·
`last_success_ts` · `last_error` · `consecutive_failures`.

### 3.9 `link_metrics` — per-provider rolling health (folds Meredith's api_call_budget + provider_health)
`provider` PK · `rtt_p50_ms` `rtt_p95_ms` `rtt_jitter_ms` (rolling 20-sample) ·
`error_rate` · `consecutive_failures` · `breaker_state` (`closed`/`open`/`half_open`) ·
`breaker_opened_at` · `calls_today` · `daily_limit` · `quota_pct` · `updated_at`.

### 3.10 `job_queue` — on-demand work from the UI
`id` PK · `job_type` (`forecast_now`/`backfill`) · `payload_json` · `status`
(`pending`/`running`/`done`/`error`) · `requested_at` · `finished_at` · `error`.

---

## 4. Components (service layer)

### 4.1 `providers/` — data source adapters
- `DataProvider` protocol: `get_daily_history(symbol, start, end) -> list[Bar]`,
  `get_latest_bars(symbol, lookback=5) -> list[Bar]`.
- Impls: `YFinanceProvider`, `TiingoProvider`, `FinnhubProvider` (equities);
  `CoinGeckoProvider`, `CoinbaseProvider` (crypto); `DydxDerivativesProvider` (funding/OI).
- Every provider call goes through `link_monitor.instrument(provider)` (times RTT, counts errors,
  increments quota) and `circuit_breaker.guard(provider)` (fail-fast if open).
- **Validation at the boundary**: raw payload → strict Pydantic model → `Bar`. Malformed →
  `quarantine_bars`, not an exception. Catch the "HTTP 200 + error body / empty `[]`" trap
  explicitly (validate shape, not just status).

### 4.2 `ingestion`
- `IngestionService.poll_watchlist()` — APScheduler target. For each active ticker: fetch latest
  bars from its primary provider (or fallback if breaker open), upsert into `ohlcv_bars`; for
  crypto also pull `crypto_derivatives`.
- `IngestionService.backfill(ticker, years=5)` — one-shot on first sight of a ticker, throttled hard.
- Dedupe on `(ticker, interval, ts)`. Store the provider's bar ts, never wall-clock.
- Equities: `pandas-market-calendars` for market hours + holidays. Crypto: 24/7 calendar.
- Gap / frozen-price / sanity checks run here (see §6).

### 4.3 `bar_store`
`BarRepository`: `upsert_bars`, `get_range(ticker, start, end, interval)`, `get_latest(ticker)`,
`latest_ts(ticker)`. Store both `close` and `adj_close` when available (models train `adj_close`,
chart shows `close`).

### 4.4 `features` — `FeatureBuilder`
- `build(bars_df) -> DataFrame` — pure, **no lookahead** (every rolling window uses only past rows;
  `center=False`; scaler fit on train window only).
- ~12–15 features (§5.3). Computed with `pandas-ta` where applicable.
- Property test: for any row `t`, `build()` output for `t` is unchanged when rows `> t` are removed.

### 4.5 `trainer` — `Trainer.train(ticker, horizon, model_type) -> ModelArtifact`
Fits on 5y history, walk-forward rolling window for OOS metrics, computes `residual_std`, persists
`.joblib` to `./model_store/{ticker}/{horizon}/{model}/{version}.joblib` + a `model_runs` row,
flips `is_active`.

### 4.6 `forecaster`
- `Forecaster.predict(ticker, horizon, model_type) -> ForecastResult` — loads active artifact,
  builds features from newest **closed** bars, returns `predicted_return`, `predicted_price`,
  `lower/upper_bound` (`residual_std · √h`, ~95%), and `explain` (ridge: `coef·value` per feature;
  RF: per-prediction contributions or global importances for MVP).
- `ForecastService.generate_and_persist(ticker, horizons, model_types)` — produces the
  `ForecastResult`s **and writes `prediction_snapshots` in ONE `BEGIN IMMEDIATE` transaction**.
- On-demand "Run forecast now": `app.py` calls `ForecastService.generate_and_persist` directly
  in-process (fast, no worker round-trip). Backfill of a new ticker goes via `job_queue`.

### 4.7 `evaluator` — `EvaluatorService.run(as_of=now)`
APScheduler hourly. `SELECT ... WHERE target_ts <= now AND evaluated_at IS NULL`. For each: get
realized price at/after `target_ts` (nearest trading bar equities / nearest bar crypto), compute
`realized_return`, `error_abs`, `error_signed`, `is_direction_hit`, `is_within_ci`; `attach_realized`
(the one permitted post-insert write). If the realized bar doesn't exist yet, bump `eval_attempts`,
retry next run. Then `rebuild_aggregates()` → `accuracy_records`.

### 4.8 `link_monitor` + `circuit_breaker`
- `link_monitor`: per-provider rolling RTT (p50/p95/jitter over 20 samples), error rate, quota
  counter; writes `link_metrics`.
- `circuit_breaker`: per provider. `closed → open` after **5 consecutive failures within 10 min**;
  `open` for **15 min** (fail fast); `open → half_open` (one trial call) → `closed` on success.
  When `open`, ingestion switches that asset class to its fallback provider automatically.

### 4.9 `accuracy`
`rebuild_aggregates()` — recompute MAE/RMSE/dir_acc/ci_coverage/n and `is_trustworthy` per
`(ticker, horizon, model_type)` and per `(horizon, model_type)` global; write `accuracy_records`.
Always fully rebuildable from `prediction_snapshots`.

---

## 5. ML methodology (from `research-ml.md`)

### 5.1 Target
Log return `r_{t+h} = ln(P_{t+h}/P_t)`. Never predict raw price. Reconstruct `P̂_{t+h} = P_t·exp(r̂_{t+h})`.

### 5.2 Multi-horizon
**Direct** — a separate model per `(ticker, horizon, model_type)`. No recursive forecasting
(error compounding).

### 5.3 Features (~12–15, all normalized, no-lookahead)
`log_return_1d/5d/20d` · `sma20_stretch` (`P/SMA20 − 1`) · `sma_crossover` (`SMA20/SMA50 − 1`) ·
`rsi_14` · `stoch_k_14` `stoch_d_3` · `macd_hist_norm` (`MACD_hist / P`) · `bollinger_pct_b` ·
`bollinger_bandwidth` · `norm_atr_14` (`ATR14 / P`) · `volume_ratio` (`V / SMA20(V)`) ·
`obv_10d_change`.
**Crypto only**: `funding_rate` · `open_interest_norm` (z-scored on trailing window) ·
`is_weekend` · `weekend_vol_ratio`.
Calendar: `day_of_week`, `is_month_end`, `is_quarter_end`.

### 5.4 Validation
Walk-forward **rolling window** (expanding also acceptable). No random k-fold. Store OOS
MAE/RMSE/dir_acc/ci_coverage per `model_run` — this is the *a priori* accuracy estimate;
the evaluator produces the *realized* one.

### 5.5 Confidence band
`residual_std` from walk-forward errors, `√h`-scaled → `lower/upper_bound` prices at ~95%.

### 5.6 Model versioning
`model_version` = hand-bumped semver on methodology change. `model_run_id` = FK to the exact fit
(data window, hyperparams, seed, git SHA). Every snapshot references `model_run_id`.

---

## 6. Data-feed health monitoring (from `research-reliability.md`, telecom lens)

| # | Check | Metric | Threshold | UI | App action |
|---|---|---|---|---|---|
| 1 | Freshness | bar age vs expected (market-hours + 15m delay aware) | 🟢 <20m stock / <5m crypto · 🟡 <1h · 🔴 >1h in-hours | `🟡 Stale (35m)` | serve last-good + badge; defer retrain if >2h |
| 2 | Latency/jitter | RTT p50/p95, σ | p50<800ms · p95<2.5s · jitter<1.5s | sparkline | widen timeout on p95 spike |
| 3 | Error rate | 4xx/5xx, 429, body-error trap, consecutive fails | ≥3 consec = degraded · ≥5 = down | provider card | trip circuit breaker → failover |
| 4 | Gap / frame sync | missing bars, out-of-order ts, dupes, frozen price (≥6 identical in-hours) | >2 consecutive missing | `⚠️ Gap: 3 bars` | targeted backfill; drop dupes |
| 5 | Data sanity / SNR | price≤0, \|r\|>30% stock / >50% crypto, high<low | any | `🚫 Bad bar (quarantined)` | quarantine row, don't crash |
| 6 | Watchdog | `system_heartbeat` lag | >2× job interval | header banner | warn + "restart worker" hint |
| 7 | Clock / tz | provider ts vs system; future-dated | drift >2m future | `⚠️ clock skew +120s` | reject future bars; normalize UTC at boundary |
| 8 | Quota budget | daily calls vs free tier | 🟡 80% · 🔴 95% | progress bar | auto-relax poll interval |

### Health status model (Streamlit panel)
```
SYSTEM: [🟢 NOMINAL]   Worker: [🟢 ALIVE lag 4s]   Data Quality: [100%]
[yfinance]  RTT 340ms | err 0% | quota 22%  🟢 ACTIVE
[CoinGecko] RTT 620ms | err 0% | quota 12%  🟢 ACTIVE
[Coinbase]  RTT 180ms | err 0% | quota  2%  🟢 STANDBY
[dYdX]      RTT 210ms | err 0% | quota  5%  🟢 ACTIVE
Watchdog: ingest 2m ago · forecast 1h ago · eval 4h ago    Pending evals: 4
```
- **🟢 NOMINAL** — all providers responding, worker heartbeat <5m, 0 quarantined rows in 24h.
- **🟡 DEGRADED** — 1 provider down (fallback active), worker lag 5–15m, or any quota >80%.
- **🔴 CRITICAL** — all primary providers down, worker dead >15m, or SQLite read error.

### Reliability patterns
**Essential (MVP)**: retry w/ exp backoff + jitter (3×, only 5xx/timeout, never 4xx) · idempotent
upsert ingest · SQLite `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000` · Pydantic boundary
validation + `quarantine_bars` · graceful degradation + staleness badge · per-provider circuit
breaker + auto-failover · full health panel · transactional forecast+snapshot write · `.env` via
`pydantic-settings` · `uv.lock`.
**Nice (v1.1)**: Mac-sleep catch-up (detect missed window on wake, one catch-up poll) · structured
JSON logging · nightly self-healing evaluator re-scan (catches any snapshot the hourly job missed).
**Overkill (skip)**: K8s/Docker · OpenTelemetry/Jaeger · Redis/Celery · DB replication ·
`launchd` auto-start.

---

## 7. Technical indicators (from `research-indicators.md`)

### Default chart (uncluttered)
| Pane | Content |
|---|---|
| Price (main) | candles/line + SMA 20 + SMA 50 + Bollinger Bands (20, 2σ) |
| Sub — momentum | RSI 14 (70/30 guide lines) |
| Sub — trend-momentum | MACD (12,26,9) line + signal + histogram |
| Sub — volume | volume bars + Volume SMA 20 |
Toggle-only: EMA 12/26, Stochastic, VWAP.

### Library
`pandas-ta` (pure Python, no C build). TA-Lib rejected (install friction).

### Lookahead / repainting bans (grading integrity)
1. No ZigZag / unconfirmed pivots / Supertrend-on-open-bar in features or eval.
2. Features on **fully closed bars only** — never a live intraday bar.
3. `center=False` on every rolling op.
4. Z-score / scaler fit on **training window only**, never global.
5. VWAP anchored to session start (00:00 UTC crypto / 09:30 EST equity).

---

## 8. Providers & rate limits

| Asset | Primary | Fallback | Derivatives |
|---|---|---|---|
| US equities | yfinance (no key, ~15m delay, decades daily) | Tiingo (50/hr, 1000/day, batch) → Finnhub (60/min) | — |
| Crypto | CoinGecko (10–30/min, no key) | Coinbase public (no geo-block) | dYdX v4 Indexer (`indexer.dydx.trade/v4/`, no key, historical funding + OI, BTC/ETH) |

- **One poller** (`worker.py`). `app.py` never calls a provider.
- Token-bucket per provider sized to its free tier. Projected daily call count shown in the health
  panel; auto-throttle at 80%.
- 10 equities polled every 5 min ≈ 2,880 yfinance calls/day — within limits.
- Backfill (5y, one-shot per ticker) throttled to a low steady rate.

---

## 9. UX (from `research-market.md` + `architecture-draft.md` §4)

### Single screen
```
┌──────────────────────────────────────────────┬──────────────────┐
│ AAPL  $227.41  +0.8%   🟡 delayed 15m         │  Watchlist       │
│ ┌──────────────────────────────────────────┐ │  AAPL  227.41 ▲  │
│ │  actual price (candles/line) + SMA/BB     │ │  BTC-USD 63.1k ▼ │
│ │  ribbon: continuous +5d predicted line    │ │  [+ add ticker]  │
│ │  latest forecast: dashed + CI band        │ ├──────────────────┤
│ └──────────────────────────────────────────┘ │  Accuracy        │
│ [RSI] [MACD] [Volume] sub-panes              │  h │MAE%│dir%│ n  │
│ Range [1M][3M][6M][1Y]  Horizons [✓1d][✓5d][30d]│ 1d│0.9│58% │140 │
│ Model (ridge▾)  Overlay [✓past] [ribbon▾]    │ 5d│2.1│55% │120 │
│ [ Run forecast now ]                         │ 30d│6.4│51% │ 90 │
│                                              │ verdict: 1d ✅ 30d ❌│
├──────────────────────────────────────────────┴──────────────────┤
│ ▸ Why this forecast?  (feature contribution bar chart)          │
├────────────────────────────────────────────────────────────────┤
│ SYSTEM [🟢 NOMINAL]  Worker [🟢 4s]  providers…  pending evals 4 │
└────────────────────────────────────────────────────────────────┘
```

### The overlay
1. **Actual** — `close` (or `adj_close` toggle), line or candles.
2. **Ribbon view (default, the money view)** — a continuous predicted line for one horizon: each
   point = "what the +Nd model, run on day d, predicted for d+N", plotted at `target_ts`, overlaid
   on actual. Tracking vs drift is visible directly.
3. **Historical forecast markers** (toggle) — each past prediction a faint segment
   `(made_from_ts, anchor_price) → (target_ts, predicted_price)`; colored once evaluated (green =
   direction hit, red = miss, grey = not matured). Hover → made_at, horizon, model, r̂, predicted
   vs realized, error, top-3 feature contributions.
4. **Latest forecast** — actual→forecast dashed line + widening CI band, one per selected horizon.

### Accuracy panel
Rows = horizons; cols = MAE (price %), RMSE, directional %, CI coverage, n. Model + scope
(this ticker / global) selectors. One-line verdict per horizon from `is_trustworthy`
(`dir_acc ≥ 0.55 AND n ≥ 30`), e.g. *"AAPL 1d ridge: 58% directional (n=140) — trust. 30d:
coin-flip — don't."*

### Explain panel
Collapsible "Why this forecast?" — horizontal bar chart of signed feature contributions for the
latest forecast.

### Health panel
§6 layout, sidebar or bottom strip, expandable.

---

## 10. Config / local run

```
# repo: ~/Desktop/stock_forecasting/
uv sync

# terminal 1 — background worker (poll / retrain / evaluate / heartbeat)
uv run python worker.py

# terminal 2 — UI
uv run streamlit run app.py            # opens http://localhost:8501
```

- `.env` (gitignored, via `pydantic-settings`): `TIINGO_API_KEY`, `FINNHUB_API_KEY` (optional
  fallbacks), `POLL_INTERVAL_CRYPTO_SEC=60`, `POLL_INTERVAL_EQUITY_MIN=5`, `BACKFILL_YEARS=5`,
  `DB_PATH=./data/app.db`, `WATCHLIST=AAPL,NVDA,SPY,BTC-USD,ETH-USD`, `RETRAIN_HOUR_UTC`.
- `./data/app.db` (SQLite WAL) + `./model_store/` — both gitignored; back up by copying the folder.
- No `launchd`. If the Mac sleeps, the evaluator's "matured but ungraded" re-scan self-heals the gap
  (v1.1 adds an explicit catch-up poll on wake).

---

## 11. Testing

- **Unit**: `FeatureBuilder` (incl. the no-lookahead property test), CI-band math, target/price
  reconstruction, calendar-aware `target_ts`, circuit-breaker state machine, walk-forward splitter.
- **Provider tests**: a `FakeProvider` returning canned / malformed / empty / 429 / slow responses;
  assert quarantine, breaker trip, failover.
- **Ledger tests**: append-only enforcement (update rejected), transactional forecast+snapshot
  (partial write rolls back), idempotent ingest (double poll = no dupes).
- **Evaluator tests**: matured/unmatured, realized-bar-missing retry, aggregate rebuild determinism.
- **Chaos cases**: provider down mid-poll, partial series, clock skew, out-of-order bars, frozen price.
- `pytest`, `ruff check` / `ruff format` clean, `uv.lock` committed.

---

## 12. MVP scope vs later

### MVP
US equities + major crypto · daily bars only · yfinance + CoinGecko→Coinbase + dYdX ·
poll crypto 60s / equities 5min · 5y backfill · watchlist add/remove (~10) · ridge +
random_forest, per-ticker · horizons 1d/5d/30d · ~15 features incl. crypto derivatives ·
auto nightly retrain + "Run forecast now" · append-only ledger + `model_runs` · hourly evaluator +
`accuracy_records` · chart: actual + latest forecast + CI band + ribbon + markers + hover · accuracy
panel (all-window) + verdict line · ridge explain + RF global importances · **full health panel +
8 checks + circuit breaker + auto-failover** · local run.

### Later (not now)
Intraday / hourly bars + horizons · LightGBM / ensembling / deep models · per-prediction SHAP for
trees · rolling 30d/90d accuracy windows + sparklines · backtest replay UI · multi-provider health
dashboard beyond the panel · alerts / notifications · Mac-sleep catch-up poll · structured JSON
logging · auth / hosting / Postgres · WebSockets · on-chain / exchange-netflow signals ·
portfolio / P&L.

---

## 13. Open items (non-blocking, resolve during implementation)

1. **On-demand forecast path** — direct in-process call vs `job_queue` round-trip. Spec assumes
   direct for "Run forecast now", queue for "backfill new ticker". Confirm during build.
2. **dYdX symbol coverage** — verify BTC-USD, ETH-USD market names and historical funding depth on
   the v4 Indexer before wiring; if depth < 5y, note it (funding is a shorter-history feature).
3. **RF explainability** — MVP ships global importances; per-prediction contributions
   (`treeinterpreter`) is a fast follow if the explain panel feels weak for RF.
4. **`adj_close` for crypto** — crypto has no corporate actions; `price_basis='raw'` always.
5. **Streamlit rerun vs worker writes** — confirm WAL + `busy_timeout` fully eliminates
   `database is locked` under a 5s Streamlit auto-refresh; add a read retry if not.
6. **Ghost-Fan overlay feasibility in Streamlit** — prototype the ribbon + markers overlay in
   `streamlit-lightweight-charts` in week 1. If it can't do togglable multi-series overlays
   cleanly, trigger the React+FastAPI fallback (service layer already isolates this).

---

## 14. Milestones (rough, for the implementation plan)

| M | Deliverable |
|---|---|
| M0 | Repo init, `uv`, SQLite schema + migrations, config, `FakeProvider`, CI (`ruff` + `pytest`). |
| M1 | Providers (yfinance + CoinGecko) + ingestion + bar_store + backfill; idempotent upsert; boundary validation + quarantine. |
| M2 | `FeatureBuilder` (+ no-lookahead property test) + `pandas-ta` indicators. |
| M3 | `Trainer` (walk-forward) + `Forecaster` + `model_runs` + `.joblib` store + CI band. |
| M4 | `prediction_snapshots` ledger (append-only + trigger) + transactional generate_and_persist + APScheduler nightly retrain. |
| M5 | `evaluator` + `accuracy_records` + hourly job + verdict logic. |
| M6 | `link_monitor` + `circuit_breaker` + auto-failover (Coinbase, Tiingo/Finnhub) + `system_heartbeat`. |
| M7 | Streamlit UI — watchlist + price chart + **ribbon overlay** (feasibility gate) + latest forecast + CI band. |
| M8 | Accuracy panel + verdict + explain panel + **full health panel**. |
| M9 | dYdX derivatives provider + crypto features + `crypto_derivatives` table. |
| M10 | Chaos/reliability test pass, docs, README, first real multi-week run. |

---

## Spec self-review

- Placeholders: none (§13 open items are explicit, scoped, non-blocking).
- Consistency: table set in §3 (10 tables) matches components in §4 and health model in §6.
  `link_metrics` (§3.9) intentionally merges Meredith's two proposed tables — noted.
- Scope: single implementation plan feasible via the M0–M10 milestone spine.
- Ambiguity: on-demand forecast path and Streamlit-overlay feasibility are the two real unknowns —
  both flagged in §13 with a decision rule, not left open-ended.
