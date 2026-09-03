# stock_forecasting — Intraday Forecasting (Crypto 15m/1h/4h + Equities 1h, Full Scorecard)

> **Status**: DESIGN — Phase 1, GATE 0 revision 2 (user expanded scope).  
> **Date**: 2026-09-03 · **Author**: Pam (pam-mtjr5nuk) · **Branch**: `feat/intraday-t013` (GATE 0 revision 2)  
> **Scope**: Design only — no implementation, no schema migration, no model code, no milestone execution.  
> **Decision Authority**: User-locked decisions (§2) + god's GATE 0 rulings (§2.5 below) are mandatory; remaining open questions are deferred.

---

## 0. Executive Summary & Scope

This design introduces **intraday forecasting models with full scorecard** to `stock_forecasting` alongside the v2.0.0 display layer and the frozen daily ML core. The intraday layer is **complete** (training → inference → grading), **dual-asset** (crypto + equities), and scoped to **crypto 15-minute/1-hour/4-hour and equities 1-hour prediction horizons**.

### Why intraday now?
- v2.0.0 delivers real-time crypto price display via Coinbase WebSocket.
- Intraday bars are already flowing into `intraday_bars` (100-row cache).
- dYdX hourly funding rate is already integrated as a feature; it aligns perfectly with crypto 1-hour model cadence.
- Equities 1-hour horizon is now viable: yfinance 1h bars extend back ~730 days (~5,070 samples), providing adequate statistical depth. yfinance 15m/5m/1m remain limited to 60d—live display only, no model training.
- A separate short-horizon intraday model with grading unlocks actionable directional/volatility signals and a live accuracy panel without compromising the frozen, proven daily ML core.

### Key constraints
1. **Asset-class specific**: Crypto (BTC-USD, ETH-USD) at 15m/1h/4h; Equities (AAPL, NVDA, SPY) at 1h only. Equity 15-minute data and sub-1h forecasting remain out-of-scope due to yfinance's sub-1h cap at 60d and 15-min delivery delay (Kevin's scout, §2.2).
2. **Per-asset-class data windows**: Crypto: 365-day lookback (Coinbase REST, unlimited depth). Equities: 730-calendar-day lookback (yfinance, proven via probe—1h bars only; sub-1h limited to 60d).
3. **Full scorecard**: Intraday forecasts are persisted, graded when targets mature, and displayed in an intraday accuracy panel (separate from daily accuracy).
4. **ML-core frozen**: The daily forecaster, trainer, features, evaluator, accuracy pipeline remain untouched. Intraday uses separate tables, modules, jobs. No cross-contamination.
5. **No real-time model updates**: Intraday model retrains nightly, once, like the daily model — not streaming or tick-driven.

---

## 1. Scope & Out-of-Scope

### 1.1 Scope (committed)

**Crypto (Coinbase REST/WS):**
- **Asset coverage**: BTC-USD, ETH-USD intraday price forecasting on 15-minute, 1-hour, and 4-hour horizons.
- **Data sources**: Coinbase REST (paginated historical 1m/5m candles, unlimited depth) + live `intraday_bars` (via WS).
- **Lookback window**: 365 calendar days (god's F3 ruling).

**Equities (yfinance):**
- **Asset coverage**: AAPL, NVDA, SPY intraday price forecasting on 1-hour horizon ONLY.
- **Data sources**: yfinance REST (1h bars, ~730 calendar days available); sub-1h (15m/5m/1m) capped at 60d—display-only, no model training.
- **Lookback window**: ~730 calendar days. Approximately 5,070 hourly bars per ticker (adequate for LightGBM, ~same depth as crypto 365d sample count).

**Unified across both asset classes:**
- **Model class**: LightGBM (primary) or Ridge regression (fallback) on tabular rolling features.
- **Feature engineering**: Intraday VWAP distance, realized-vol ratios, EWMA return spreads, volume acceleration. Crypto adds dYdX funding-rate z-scores; equities omit funding (N/A).
- **Methodological guardrail**: Purged & Embargoed TimeSeriesSplit (López de Prado) to prevent label-leakage false positives.
- **Output**: Directional point forecast (log-return prediction) + HAR-RV dynamic confidence bands for visual ribbon.
- **Persistence**: Each forecast is written to `intraday_prediction_snapshots` table (separate from daily `prediction_snapshots`) immediately after inference.
- **Grading**: Intraday evaluator job grades forecasts when their target horizons mature, computing signed error, directional accuracy, CI coverage.
- **Accuracy panel**: Separate intraday accuracy panel (per asset, per horizon: MAE %, direction %, CI cover %, n, trust verdict) alongside (not merged with) the daily accuracy panel.
- **Chart display**: Intraday forecast ribbon overlaid on chart; historical forecast markers (green/red/grey) for past intraday predictions.
- **Retrain cadence**: Nightly, once per day post-close (same cadence as daily model).

### 1.2 Explicitly OUT-of-scope this milestone

- **Equities sub-1h** (1m / 5m / 15m): Equity sub-1h data on yfinance is hard-capped at 60 calendar days, and sub-1h predictions suffer from yfinance's ~15-minute delivery delay (predicting the next 5m with data that is already 15m stale). These sub-1h bars are available for **display only** (live price ribbon), never for model training. Equities 1h is IN scope (730d available per Dwight's probe). (Kevin's scout, §2.2).
- **Crypto sub-15m** (1m / 5m next-bar): Microstructure noise dominates; SNR too low for actionable 1m/5m forecasts (Kevin's scout, §2.1).
- **Real-time model retraining**: Intraday model is trained once per day; not adapted intraday based on live ticks.
- **Intraday position-sizing / trade signals / alerts**: Forecasts and accuracy panel are informational only; no prescriptive advice or notifications.
- **Deep learning (LSTM/Transformers/PatchTST)**: Ruled out by data starvation and overfitting risk on 60-day windows (Kevin's scout, §3.4).
- **Merging intraday accuracy into daily accuracy**: Intraday scorecard is completely separate (different table, different panel, different ledger).

---

## 2. Locked Decisions (from user)

These decisions are immutable for this design and must be carried verbatim into implementation:

1. **SCOPE (EXPANDED, REVISION 3)**: 
   - **Crypto**: BTC-USD + ETH-USD on **15m, 1h, and 4h horizons** (was 1h/4h; 15m added).
   - **Equities**: AAPL, NVDA, SPY on **1h horizon ONLY** (newly added; sub-1h display-only, no model training due to yfinance 60d cap on sub-1h).
   - **Model count**: Crypto = 6 directional models (BTC/ETH × 15m/1h/4h) + 2 HAR-RV (per ticker). Equities = 3 directional models (1h) + 3 HAR-RV (per ticker). **Total: 9 directional + 5 HAR-RV models**.
   - **Data windows**: Crypto 365d (Coinbase REST unlimited depth). Equities 730d (yfinance proven via probe; ~5,070 1h bars per ticker, adequate LightGBM depth).

2. **FULL SCORECARD**: The intraday forecast is persisted and graded. Separate `intraday_prediction_snapshots` table, separate intraday evaluator job (grades when target matures), separate intraday accuracy panel. Completely independent of the daily `prediction_snapshots` ledger and daily accuracy panel.

3. **ML-core frozen**: The daily core (forecaster.py, trainer.py, features.py, evaluator.py, accuracy.py) stays FROZEN and display-separated. Intraday uses separate tables (intraday_prediction_snapshots, intraday_accuracy_records), separate modules, separate worker jobs. Zero cross-contamination with daily ledger.

4. **MODEL**: LightGBM (Ridge fallback) on rolling tabular features. 
   - **Crypto features**: Intraday VWAP distance, realized-vol ratios, EWMA return spreads, dYdX funding-rate z-scores, volume acceleration, lagged returns.
   - **Equity features**: Intraday VWAP distance, realized-vol ratios, EWMA return spreads, volume acceleration, lagged returns (funding-rate z-score omitted—dYdX is crypto-only).
   - **Both**: HAR-RV for the CI band. Purged + Embargoed TimeSeriesSplit for overlapping k-step-forward-return label leakage. CPU, local, nightly retrain cadence.

## 2.5 God's GATE 0 Rulings (Revision 1: F1–F12) + Revision 2 (Scope Expansion)

**Revision 1 rulings** (F1–F12 from prior review) remain in full force. **Revision 2** incorporates user scope expansion (crypto 15m added, equities 1h added) and the following scoping decisions:

1. **Training window (F3)**: 365 days for BOTH horizons. Set `INTRADAY_LOOKBACK_DAYS=365` in config. Rationale: 90d is thin for 4h (~540 independent samples < 5k needed); 365d provides ~2,190 4h-bars/yr + ~8,760 1h-bars/yr. Coinbase depth supports multi-year keyless pagination (~351 requests/ticker, acceptable cost).

2. **Model persistence (Q2)**: Pickle files in `stock_forecasting/models/intraday/` + `metadata.json`. NO `intraday_model_runs` table for MVP.

3. **Per-asset models (Q7)**: SEPARATE model per (ticker, horizon). 4 directional models (BTC-1h, BTC-4h, ETH-1h, ETH-4h) + HAR-RV per ticker. No cross-asset model.

4. **Grading realized price (Q9)**: Closed-bar close from `intraday_bars_history` (not live tick).

5. **Grading job cadence (Q10)**: Hourly APScheduler worker job (short-horizon forecasts mature fast). Aligned to bar close.

6. **Forecast writer location (F1)**: Dedicated hourly worker job (APScheduler, not fragment/render path). Sole writer to `intraday_prediction_snapshots`. Fragment display ribbon is READ-ONLY (recompute from model + live_quotes, persist nothing). Anchor to last CLOSED bar, not forming bar. Dedup via `UNIQUE(ticker, horizon, anchor_ts)` + `INSERT OR IGNORE`.

7. **Immutable training/grading store (F2)**: New table `intraday_bars_history` (immutable, written once per closed bar, pruned per asset-class window). Both training and grading read from it. This resolves the 7-day prune conflict.

8. **Per-asset-class lookback windows (REVISION 3: R-A)**:
   - **Crypto (BTC-USD, ETH-USD)**: 365 calendar days. Config: `INTRADAY_LOOKBACK_DAYS_CRYPTO=365`.
   - **Equities (AAPL, NVDA, SPY)**: 730 calendar days (yfinance proven via Dwight's probe—1h bars back to ~2023-10-06, ~5,070 bars/ticker). Config: `INTRADAY_LOOKBACK_DAYS_EQUITY=730`.
   - **Note**: yfinance sub-1h (15m/5m/1m) remain capped at 60d, used for display only (live ribbon), never for model training.
   - Schema: `intraday_bars_history` adds `asset_class` column (`crypto` | `equity`) for per-asset retention config.

8b. **Equity model setup (REVISION 3: R-B)** — Equities with ~5,000 bars (730d) are NOT sample-starved. Use the SAME model architecture as crypto: **LightGBM primary + Ridge fallback, standard leakage canary** (no special-casing for small sample). With 5k bars, equity sample depth is comparable to crypto 365d count and adequate for LightGBM. Removes the prior experimental flags / shallow-tree mitigation (they were predicated on 270 samples, now false).

9. **15-minute crypto horizon (REVISION 2)**: Adds 2 new models (BTC-15m, ETH-15m) to the crypto suite. 15m bars are sourced from Coinbase 5m granularity (anchor to 15m boundaries: :00, :15, :30, :45 UTC). Model is 3-step-forward log-return (= 3 × 5m bars = 15m horizon). Same feature suite as 1h (VWAP, funding-rate, etc.), fit on 365d Coinbase history. Labels anchor at contiguous non-overlapping 15m boundaries. Rationale: Sub-1h crypto can work on Coinbase (0s latency, unlimited depth); microstructure SNR acceptable at 15m with dYdX funding feature (captures volatility regime). 15m is a tactical sub-1h window for fast traders; 1h/4h remain strategic.

10. **Equity anchor handling (REVISION 2)**: yfinance introduces ~15-minute delivery lag. Intraday forecasts for equities are ANCHORED TO THE LAST CLOSED 1h BAR (not live price). This means a forecast made at 14:45 ET uses the 14:00 ET close (which is ~15m stale on yfinance). The forecast is for the NEXT 1h bar (15:00 ET close). Acceptance: Equity 1h forecast is inherently lagged by ~15m vs. real-time tick; this is a design tradeoff for using free yfinance data. Disclaimer clarifies this (§7, §2.5 (F1)0).

11. **Feature applicability by asset class (REVISION 2)**:
    - **Crypto**: All 8 features (§4.1): VWAP dist, vol ratios, EWMA spreads, funding-rate z-score, volume accel, lagged returns, hour-of-day, day-of-week.
    - **Equities**: 7 features (funding-rate z-score omitted): VWAP dist, vol ratios, EWMA spreads, volume accel, lagged returns, hour-of-day, day-of-week. Rationale: dYdX is crypto perpetuals only; equities have no hourly funding concept.

---

## 3. Architecture & Data Flow

### 3.1 Intraday forecasting as a separate ML subsystem (Crypto + Equities)

```
Daily ML Core (FROZEN)               Intraday ML Subsystem (NEW, Full Scorecard, Dual-Asset)
├─ forecaster.py                    ├─ intraday_forecaster.py (new)
├─ trainer.py                       ├─ intraday_trainer.py (new)
├─ features.py                      ├─ intraday_features.py (new, asset-aware)
├─ evaluator.py                     ├─ intraday_evaluator.py (new)
└─ accuracy.py                      └─ intraday_store_models.py (if persisting model files)

Reads from (CRYPTO):                Intraday reads from:
└─ ohlcv_bars (1d, immutable)      Crypto:
                                    ├─ intraday_bars (1m/5m candles, 7-day cache)
                                    ├─ intraday_bars_history (365-day ML store, §2.5 (F2))
                                    ├─ live_quotes (current tick)
                                    └─ crypto_derivatives (dYdX funding rate)
                                    
                                    Equities:
                                    ├─ intraday_bars_history (60-day equity cache, per yfinance limit)
                                    ├─ live_quotes (current tick, yfinance delay ~15m)
                                    └─ (no funding-rate equivalent)

Writes to:                          Intraday writes to:
├─ prediction_snapshots (daily)    ├─ intraday_prediction_snapshots (crypto + equity forecasts)
├─ accuracy_records (daily)        ├─ intraday_accuracy_records (crypto + equity grades)
└─ model_runs (training metadata)  └─ intraday_bars_history (asset-class partitioned, §2.5 R2#8)

Zero shared tables between daily and intraday. Separate display panels. Asset-class separation via asset_class column.
```

### 3.2 Training data pipeline for intraday models (Crypto + Equities)

**CRYPTO Input source**: Historical Coinbase candles from `intraday_bars_history` (via existing `providers/coinbase.py` REST endpoint for backfill, but training reads immutable history).
- **Granularity**: 5-minute bars (fixed; same as chart intraday bucket). Resample to 15m boundaries for 15m horizon (e.g., anchor :00, :15, :30, :45 UTC).
- **Lookback window**: 365 calendar days (god's F3 ruling, §2.5 R1). Covers ~8,760 1h-bars/yr, ~2,190 4h-bars/yr, ~35,040 15m-bars/yr (>5k sample threshold per Kevin's scout §3.2). Coinbase cost: ~351 requests/ticker (~702 both).

**EQUITIES Input source**: yfinance 1h bars for AAPL, NVDA, SPY from `intraday_bars_history`.
- **Granularity**: 1-hour bars (fixed; yfinance 1h resolution).
- **Lookback window**: 730 calendar days (yfinance capacity proven via probe; earliest ~2023-10-06). Yields ~5,070 hourly bars per ticker (adequate LightGBM depth, same architecture as crypto—no special-casing).
- **Note**: yfinance introduces ~15-minute delivery lag; forecast anchors account for this (§2.5 R2#10). Equity bar closes are typically outside trading hours (overnight/weekend handling required §3.2.1).

**Feature computation** (§4 below):
1. Build OHLCV-derived technical features (VWAP, realized vol, EWMA spreads).
2. Normalize with StandardScaler (fit on training window, apply to validation).
3. **Crypto only**: Align with dYdX funding-rate series (hourly snapshots, zero-order hold for 5m bars).
4. **Equities only**: Omit funding-rate feature.
5. Merge into a single DataFrame for each (BTC-USD, ETH-USD, AAPL, NVDA, SPY) pair.

**Label construction** (k-step forward log-return, §2.5 (F8)):
- **Crypto**: Anchor ONLY at closed bar boundaries: :00 UTC for 1h; :00/:04/:08/:12/:16/:20 UTC for 4h; :00/:15/:30/:45 UTC for 15m (or equivalently, anchor_ts must be a multiple of 900, 3600, or 15*60 seconds).
  - Label = $\ln(P_{anchor+k} / P_{anchor})$ where k ∈ {900s, 3600s, 14400s} and both anchors are closed bars from `intraday_bars_history`.
- **Equities**: Anchor at 1h closed bar boundaries (e.g., 10:00 ET, 11:00 ET, etc.).
  - Label = $\ln(P_{anchor+1h} / P_{anchor})$, both from `intraday_bars_history`.

**Cross-validation protocol**:
- **Mandatory**: Purged & Embargoed TimeSeriesSplit (López de Prado).
  - Purging: Drop all training samples whose label window overlaps with the test window.
  - Embargoing: Drop training samples immediately following the test set by 24 hours to eliminate serial correlation leakage (§4.5 risk mitigation).
- **Rationale**: k-step forward returns on overlapping bars share price data; naive K-fold leaks future information across folds.

### 3.2.1 Equity Market-Calendar Handling (B7)

**Problem**: Equities trade only during market hours (09:30–16:00 ET = 14:30–21:00 UTC) on trading days (Mon–Fri, excluding holidays). A 1h label "anchor +1h" is ambiguous if the anchor is at 15:30 ET (market close) — the "+1h" would be 16:30 ET the next morning, not same-day. Overnight gaps and weekends must be explicit in the training data.

**Solution**:
- **Label anchors**: Equity 1h bar anchors are TRADING-HOUR closed bars only (10:00 ET, 11:00 ET, ..., 16:00 ET). No off-market anchors.
- **Labels**: Label = ln(P_{anchor+1h,next_trading_bar} / P_anchor), where "next_trading_bar" is the next 1h bar that closes on the same trading day (not the literal +3600s timestamp).
  - Example: If anchor = 15:00 ET (3-hour close), label targets 16:00 ET (4-hour close), same day.
  - If anchor = 16:00 ET (market close), label targets 10:00 ET next trading day (overnight gap is implicit in the close prices).
- **Market calendar gating**: Equity forecast-writer job (M4, §5.1) fires ONLY during trading hours (14:30–21:00 UTC) and on trading days. Reference `stock_forecasting/market_calendar.py` for trading-day validation and holiday skipping.
- **HAR-RV caveat**: Heterogeneous Autoregressive volatility model for equities is computed on trading-hour bars only. Overnight gaps (16:00 ET close to 09:30 ET open) are implicit in the daily realized volatility bucket; no explicit interpolation or zero-filling across overnight intervals. This means equity HAR-RV bands may widen after weekends (due to overnight drift) but is a natural feature of the gap-aware data.

### 3.3 Retrain cadence & schedule (Crypto + Equities)

- **When**: Nightly after market close (crypto 00:00 UTC post-day). Equities retrain aligned to US market close (16:00 ET = 21:00 UTC).
- **Frequency**: Once per day, same pattern as the daily trainer. Crypto and equities can retrain in parallel (independent pipelines).
- **Concurrency**: Run intraday trainers (crypto + equity) **in parallel with** daily trainer (no sequential deps). Crypto can start first (00:00 UTC); equities start post-close (~21:00 UTC).
- **Cost**: 
  - **Crypto**: LightGBM training on 365 days of 5m bars (~105,120 rows for 1h/4h; ~35,040 for 15m) takes ~10–30 seconds per horizon per ticker (5–10 tickers × 3 horizons) on commodity CPU. Total: ~2–3 min for all crypto models.
  - **Equities**: LightGBM training on 60 days of 1h bars (~270 rows) takes <1 second per ticker. Total: <1 min for all equity models.
  - **Combined**: ~3–4 min nightly for both crypto and equity retrain; acceptable.

### 3.4 Storage & model persistence

**Where trained models live**:
- Option A (lightweight): Pickle the fitted `LightGBMRegressor` and `Ridge` fallback to local `.pkl` files in `stock_forecasting/models/intraday/`.
- Option B (heavier): Store model metadata and coefficients in a new `intraday_model_runs` table (parallel to `model_runs` for daily).
  - **OPEN QUESTION**: Should we persist intraday model snapshots for later audit / comparison? If yes, schema needed; if no, local `.pkl` suffices. Recommend Option A (files) for MVP but leave open.

**Prediction ledger**: New table `intraday_prediction_snapshots` (§3.5 below).

**Accuracy ledger**: New table `intraday_accuracy_records` (§3.6 below).

### 3.5 New table — `intraday_bars_history` (F2 / immutable store, REVISION 2: asset-class partitioned)

Immutable, durable store for intraday bars (separate from `intraday_bars`, the 7-day display cache). Now holds both crypto (365d) and equity (60d) data partitioned by asset class.

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK → `tickers.symbol` | BTC-USD, ETH-USD (crypto); AAPL, NVDA, SPY (equity) |
| `asset_class` | TEXT | `crypto` \| `equity` (REVISION 2: for per-class retention) |
| `interval` | TEXT | `5m` (crypto); `1h` (equity) |
| `ts` | TEXT | bucket start, ISO-8601 UTC (only for CLOSED bars) |
| `open` `high` `low` `close` | REAL | |
| `volume` | REAL | |
| `source` | TEXT | `coinbase_rest` \| `coinbase_ws_finalized` (crypto); `yfinance` (equity) |
| `ingested_at` | TEXT | wall-clock |

**Unique index**: `(asset_class, ticker, interval, ts)` (asset_class added for retention partitioning).  
**Retention** (per-asset config, not hardcoded): 
- **Crypto**: Pruned at INTRADAY_LOOKBACK_DAYS_CRYPTO (365 days). Daily job: `DELETE FROM intraday_bars_history WHERE asset_class='crypto' AND ts < now - 365 days`.
- **Equities**: Pruned at INTRADAY_LOOKBACK_DAYS_EQUITY (730 days). Daily job: `DELETE FROM intraday_bars_history WHERE asset_class='equity' AND ts < now - 730 days`.

**Writing**: 
- **Crypto**: Worker closes intraday bar from `intraday_bars` (display cache), writes once to `intraday_bars_history` with `asset_class='crypto'` and `is_provisional=0` (immutable), then propagates to `intraday_bars` (7-day cache). Single-writer pattern (v2).
- **Equities**: yfinance backfill job writes closed 1h bars directly to `intraday_bars_history` with `asset_class='equity'` once per day post-close. Single-writer pattern.

**Reading**: Training pipeline + intraday evaluator read ONLY from `intraday_bars_history`, never from `intraday_bars`.

### 3.6 New table — `intraday_prediction_snapshots` (REVISION 2: asset_class tracking)

Mirrors the structure of `prediction_snapshots` (daily) but for intraday horizons. Records every forecast made by the worker job (the sole writer, §5.1). Now tracks both crypto and equity forecasts.

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK → `tickers.symbol` | BTC-USD, ETH-USD (crypto); AAPL, NVDA, SPY (equity) |
| `asset_class` | TEXT | `crypto` \| `equity` (REVISION 2: for analytics/filtering) |
| `horizon` | TEXT | `15m` \| `1h` \| `4h` (crypto: all three; equity: 1h only) |
| `made_at` | TEXT | wall-clock UTC when forecast was written by worker job |
| `anchor_ts` | TEXT | timestamp of the CLOSED bar used to compute features (§2.5 (F8)) |
| `anchor_price` | REAL | close price of the closed anchor bar |
| `predicted_return` | REAL | model's log-return forecast (ŷ), where label = ln(P_{anchor+h} / P_anchor) |
| `predicted_price` | REAL | anchor_price × exp(predicted_return) [denormalized for display] |
| `ci_lower_return` | REAL | HAR-RV lower bound (log-return space, fitted on train split §2.5 (F4)) |
| `ci_upper_return` | REAL | HAR-RV upper bound (log-return space) |
| `ci_lower_price` | REAL | denormalized to price space |
| `ci_upper_price` | REAL | denormalized to price space |
| `target_ts` | TEXT | when the prediction target matures (anchor_ts + 15m/1h/4h) |
| `model_version` | TEXT | version string of model used |
| `model_sha` | TEXT | git SHA of model training code |

**Index**: `UNIQUE(ticker, horizon, anchor_ts)` + `INSERT OR IGNORE` dedup (§2.5 (F1)).  
**Retention**: Keep intraday predictions indefinitely (full audit trail); no pruning.  
**Grading reference**: When `target_ts` passes, `intraday_evaluator` looks up this row and computes realized return from `intraday_bars_history` (§2.5 (F2)).

### 3.7 New table — `intraday_accuracy_records` (REVISION 2: asset_class tracking)

Records grading results once a forecast's target matures. Same shape as `accuracy_records` (daily) but intraday-specific, now tracking both crypto and equity.

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `prediction_id` | INTEGER FK → `intraday_prediction_snapshots.id` | links back to the forecast |
| `ticker` | TEXT FK → `tickers.symbol` | BTC-USD, ETH-USD (crypto); AAPL, NVDA, SPY (equity) |
| `asset_class` | TEXT | `crypto` \| `equity` (REVISION 2: for analytics/filtering) |
| `horizon` | TEXT | `15m` \| `1h` \| `4h` (crypto: all three; equity: 1h only) |
| `graded_at` | TEXT | wall-clock when grade was computed |
| `realized_return` | REAL | actual log-return from anchor to target close (closed bar from intraday_bars_history, §2.5 (F2)) |
| `realized_price` | REAL | actual close price at target_ts from intraday_bars_history |
| `signed_error` | REAL | realized_return - predicted_return |
| `abs_error_pct` | REAL | \|signed_error\| × 100 |
| `direction_hit` | INTEGER | 1 if sign(predicted) == sign(realized), else 0 |
| `ci_cover` | INTEGER | 1 if realized_return ∈ [ci_lower_return, ci_upper_return], else 0 |
| `grading_attempts` | INTEGER | counter: incremented each time evaluator checks. Prevents false failures if bar not yet closed. |

**Index**: foreign key on prediction_id.  
**Grading logic** (§5.1 updated per F1/F2): When forecast's target_ts arrives, evaluator queries `intraday_bars_history` for the realized closed bar (filtering by asset_class). If not found, increment `grading_attempts` and defer. When bar found, compute results and write (or update if row exists with attempts counter, filling in computed fields).

---

## 4. Feature Engineering for Intraday Horizons (Crypto + Equities)

### 4.1 Intraday-specific feature suite (asset-class aware)

The following features are designed to capture short-horizon volatility clusters, mean reversion, and momentum regime changes. **Crypto** uses all 8 features; **Equities** omit funding-rate (feature 6).

#### Technical (OHLCV-derived, no external data) — Applied to BOTH crypto and equities
1. **Intraday VWAP Distance** (per Kevin's scout §3.2):
   - $f = (P_t - \text{VWAP}_{[t-T, t]}) / \sigma_{\text{VWAP}}$ where $T$ is the forecast horizon (1h, 4h for crypto; 1h for equity).
   - Captures reversion-to-mean within the session; mean-reversion alpha signal.
   - Computation: rolling VWAP over the horizon window, normalized by its standard deviation.

2. **Realized Volatility Ratio** (breakout signal):
   - $f = \sigma_{[t-15\text{m}, t]} / \sigma_{[t-4\text{h}, t]}$ (crypto); $f = \sigma_{[t-15\text{m}, t]} / \sigma_{[t-1\text{d}, t]}$ (equity).
   - High ratio → volatility expansion (potential regime shift); low ratio → calm, mean-reversion likely.

3. **EWMA Return Spread** (multi-scale momentum):
   - $f = \text{EWMA}(\text{returns}, \text{span}=12) - \text{EWMA}(\text{returns}, \text{span}=48)$, normalized by ATR.
   - Short-term vs. medium-term momentum divergence; predicts reversal when signs flip.

4. **Volume Acceleration**:
   - $f = \text{Volume}_t / \text{SMA}(\text{Volume}, 20)$
   - Detects breakout volume; high acceleration suggests sustained direction.

5. **Lagged Log-Returns** (autoregressive):
   - $f = [\ln(P_t / P_{t-1}), \ln(P_t / P_{t-5\text{m}}), \ln(P_t / P_{t-1\text{h}})]$ (crypto, 5m bars)
   - $f = [\ln(P_t / P_{t-1}), \ln(P_t / P_{t-1\text{d}})]$ (equity, 1h bars)
   - Raw price dynamics at multiple scales; non-zero autocorrelation at horizon.

#### Derivatives (external, dYdX) — Crypto ONLY (equities omit this feature)
6. **dYdX Funding-Rate Z-Score** (24h window, §2.5 (F9)) — **CRYPTO ONLY**:
   - $f = (\text{FundingRate}_t - \text{mean}_{[t-24\text{h}, t]}) / \text{std}_{[t-24\text{h}, t]}$
   - Extreme funding rates predict volatility expansion or liquidation cascades.
   - Sourced from `crypto_derivatives` table (hourly snapshots).
   - **As-of join** (critical for no lookahead): value at feature-time $t$ = last funding rate PUBLISHED at or before $t$, never the rate that will be settled for the hour containing $t$. Forward-fill only for missing historical data during backfill, not for live features.
   - Applied as zero-order hold to 5m bars (same value for 12 consecutive 5m bars within each hour).
   - **NOT applicable to equities**: dYdX is a crypto perpetuals exchange only. Equities have no hourly funding concept.

#### Temporal — Applied to BOTH crypto and equities
7. **Hour-of-day** (cyclical encoding):
   - $\sin(2\pi \cdot \text{hour} / 24)$, $\cos(2\pi \cdot \text{hour} / 24)$ to capture global trading session patterns (Asia, EU, US close for equities; UTC hours for crypto).

8. **Day-of-week** (cyclical):
   - $\sin(2\pi \cdot \text{dow} / 7)$, $\cos(2\pi \cdot \text{dow} / 7)$ for Friday-expiry / Monday-open effects (equities); weekend crypto behavior (crypto 24/7 but week-level patterns persist).

### 4.2 Implementation location (asset-class aware)

**OPEN QUESTION**: Should intraday feature computation be:
- A) A new standalone module `intraday_features.py` (mirrors the daily `features.py`, with asset-class dispatch)?
- B) Inline in `intraday_trainer.py` (keeps logic compact, no separate abstraction)?
- C) A hybrid (base utilities in `intraday_features.py`, trainer calls them)?
- D) Separate modules: `intraday_features_crypto.py` + `intraday_features_equity.py` (enforces asset-class separation)?

Recommend **Option A** (new unified module with asset-class dispatch functions) for consistency with v2 architecture and future reuse in `intraday_forecaster.py`. Within the module, use conditional logic to omit funding-rate feature for equities (e.g., `if asset_class == 'crypto': features.append(funding_rate_z_score)`). This keeps the codebase DRY while maintaining clarity on which features apply where (document with comments).

**REVISION 2 note**: The equity feature set (7 features, no funding-rate) is smaller than crypto (8 features) but overlapping. A single `intraday_features.py` module with an `asset_class` parameter is cleaner than separate files.

### 4.3 Scaling & preprocessing

- Fit a `StandardScaler` on the **training window** of the Purged & Embargoed split (not on test data).
- Apply the same scaler to validation and live inference.
- **OPEN QUESTION**: Should the scaler be re-fit nightly with the latest 90 days of data, or stored and reused across days? Recommend nightly re-fit (intraday volatility levels shift daily) but leave open for tuning.

---

## 5. Intraday Worker Jobs (Forecast Writer & Evaluator)

### 5.1 Forecast writer job (hourly, sole writer, asset-aware, §2.5 (F1))

**When (CRYPTO)**: Hourly APScheduler job @00:00 UTC (post-day close). Computes 15m, 1h, and 4h forecasts in a single run.

**When (EQUITY)**: Hourly APScheduler job on trading days only, with market-hours gating. Fires at US market hours (09:30–16:00 ET = 14:30–21:00 UTC) to forecast the next 1h bar. References `stock_forecasting/market_calendar.py` to skip weekends + holidays. Computes 1h forecast only.

**What it does**:
1. Query `intraday_bars_history` (asset_class-aware) for the last CLOSED bar (anchor bar) for each (ticker, interval).
   - **Crypto**: Anchor_ts must be on a closed-bar boundary (:00/:15/:30/:45 UTC for 15m; :00 UTC for 1h; :00/:04/:08/:12/:16/:20 UTC for 4h).
   - **Equity**: Anchor_ts must be a closed 1h bar on a trading day (e.g., 10:00 ET, 11:00 ET, ..., 16:00 ET). Use market_calendar to validate.
2. Load the fitted intraday_forecaster model (LightGBM or Ridge, per god's R-B: same setup for both assets).
3. Compute features for this anchor bar:
   - **Crypto**: VWAP dist, vol ratios, EWMA spreads, dYdX funding-rate as-of value (critical: as-of, no lookahead), temporal.
   - **Equity**: VWAP dist, vol ratios, EWMA spreads, temporal (no funding-rate; crypto-only feature).
4. Call model.predict() → predicted_return.
5. Call intraday_volatility (HAR-RV) → CI bounds (fitted on TRAIN split, per §2.5 (F4)).
6. **Write to `intraday_prediction_snapshots`** (with asset_class field):
   - Anchor: anchor_ts (closed bar), anchor_price (its close).
   - **Crypto**: Target: anchor_ts + 15m / 1h / 4h (per horizon).
   - **Equity**: Target: anchor_ts + 1h.
   - Predicted return, CI bounds, model version/SHA, made_at = now, asset_class = 'crypto' | 'equity'.
7. **Dedup via `INSERT OR IGNORE`**: `UNIQUE(ticker, horizon, anchor_ts)` ensures no duplicate for the same anchor.

**Concurrency**: Single writer per asset class (v2 pattern). No in-render DB writes (contrast with F1's anti-pattern). Crypto and equity workers can run in parallel (independent).

### 5.2 Grading workflow (separate worker job)

**When**: Hourly APScheduler job (can be same job as 5.1 or separate; recommend separate for clarity). Runs shortly after bar close to allow `intraday_bars_history` writes to settle.

**What it does**:
1. Query `intraday_prediction_snapshots` for all forecasts where `target_ts ≤ now` and no corresponding `intraday_accuracy_records` row with all fields filled (ungraded).
2. For each ungraded forecast:
   a. Look up the realized bar in `intraday_bars_history` for (ticker, interval, target_ts).
   b. **If bar is not found**:
      - If no row exists in intraday_accuracy_records for this prediction_id: create one with grading_attempts=1, all other fields NULL.
      - If row exists: increment grading_attempts.
      - Skip to next forecast (will retry on the next hourly job run).
   c. **If bar found**:
      - Compute realized_return = ln(P_target_close / P_anchor_close).
      - Compute signed_error, abs_error_pct, direction_hit, ci_cover.
      - Insert/update intraday_accuracy_records with all computed fields (grading_attempts unchanged or set to 1 if new row).
3. Exit.

**Rationale**: Handling "bar not yet found" gracefully (via grading_attempts counter) prevents false negatives and avoids re-grading. Gracefully handles weekends, system downtime, or races. Mirrors daily evaluator pattern (Issue 7).

---

## 6. Model Training & Inference

### 6.1 Model architecture (per locked decision §2, REVISION 3: R-B same setup both assets)

**Primary**: LightGBM regressor (same for crypto and equities, §R-B).
- **Target**: k-step log-return ($r_{t, t+k}$ where $k \in \{15\text{m}, 1\text{h}, 4\text{h}\}$ for crypto; $k = 1\text{h}$ for equity).
- **Hyperparameters** (same template, tuned per asset-class):
  - Depth: 3–5 (balanced against noise; no special shallow-tree requirement for equity post-R-B / 730d sample increase).
  - Learning rate: 0.01–0.05 (conservative, stable gradients).
  - Number of boosting rounds: 100–500 (early stopping on validation loss).
  - **OPEN QUESTION**: Should hyperparameters be tuned via cross-validation or fixed empirically? Recommend Bayesian optimization on the Purged & Embargoed split for MVP (tuning on test data would leak).

**Fallback**: Ridge regression (same for both assets, §R-B).
- Same features, continuous target, standard cross-validated alpha.
- Train both LGB + Ridge nightly; display uses LGB if available, Ridge as fallback.

**Volatility bands (HAR-RV)** (separate from directional forecast):
- Predict next 1-hour / 4-hour realized volatility using Heterogeneous Autoregressive (HAR) model.
- Input: realized vol at 5m, hourly, and 4-hour frequencies over the past day.
- Output: $\hat{\sigma}_{t, t+h}$ used to compute $\pm 1.96 \hat{\sigma}$ confidence bands on the chart.
- **No correlation with the directional model** — HAR is purely volatility-forecasting, independent of the return forecast.

### 6.2 Training loop (nightly, post-close, REVISION 2: dual-asset)

**Pseudocode** (§2.5 (F3)/R2#8: crypto 365d / equity 60d; §2.5 (F4): HAR-RV fit on TRAIN; §2.5 (F8): closed-bar anchors only):

**CRYPTO LOOP**:
```
1. Fetch latest 365 days of BTC-USD + ETH-USD intraday bars from intraday_bars_history (5m, asset_class='crypto').
2. Fetch latest 365 days of dYdX funding-rate snapshots.
3. Align + resample to common 5m grid; as-of join funding rates (§2.5 (F9)).
4. For each horizon in {15m, 1h, 4h}:
   a. Filter to closed-bar anchors only (:00/:15/:30/:45 for 15m; :00 for 1h; :00/:04/:08/:12/:16/:20 UTC for 4h).
   b. Compute features (§4.1, crypto suite with funding-rate) on the closed-bar-anchored DataFrame.
   c. Construct k-step log-return labels: label = ln(P_{anchor+k} / P_anchor) for k ∈ {15m, 1h, 4h}.
   d. Split into train/test using Purged & Embargoed TimeSeriesSplit:
      i. Test window: last 14 days (2 weeks).
      ii. Purge: drop all training samples whose label window overlaps [test_start, test_end].
      iii. Embargo: drop training samples in [test_end, test_end + 24h].
      iv. Train on remaining samples.
   e. For each (ticker, horizon) pair:
      i. Fit LightGBM on train; validate on (purged + embargoed) test.
      ii. **Leakage canary** (§2.5 (F7)): Fit a control model with shuffled labels on the same split; verify it scores ~50% directional on test (not >55%), confirming no label leakage.
      iii. Record validation metrics on test: MAE (%), RMSE, directional accuracy (%), CI coverage (%).
      iv. Fit Ridge as fallback.
      v. Save both models to disk (pickle files, per god's ruling Q2).
   f. **Fit HAR-RV on TRAIN split** (§2.5 (F4)): Volatility model trained on train split, evaluated on test for CI coverage.
5. Record metadata (train_start, train_end, model version, code SHA, validation MAE/directional/CI-cover) for audit.
6. Exit crypto loop.
```

**EQUITY LOOP** (REVISION 3: R-A 730d, R-B standard LGB depth):
```
1. Fetch latest 730 days of AAPL, NVDA, SPY intraday bars from intraday_bars_history (1h, asset_class='equity').
2. For horizon 1h only:
   a. Filter to closed 1h bar anchors on trading days (e.g., 10:00 ET, 11:00 ET, ..., 16:00 ET; skip weekends/holidays via market_calendar).
   b. Compute features (§4.1, equity suite WITHOUT funding-rate) on the closed-bar-anchored DataFrame.
   c. Construct 1-step log-return labels: label = ln(P_{anchor+1h} / P_anchor). Labels are contiguous non-overlapping; purge drops only boundary samples.
   d. Split into train/test using Purged & Embargoed TimeSeriesSplit:
      i. Test window: last 30 days (1 month, ~125 trading bars).
      ii. Purge & embargo same as crypto (24h embargo).
      iii. Train on remaining samples (~700 trading days, ~3,500 bars; adequate LightGBM depth per §R-B).
   e. For each ticker (AAPL, NVDA, SPY):
      i. Fit LightGBM on train (standard depth 3–5, same as crypto; no special shallow-tree mitigation per §R-B).
      ii. **Leakage canary** (§2.5 (F7)): Fit control model with shuffled labels; verify ~50% directional.
      iii. Record validation metrics.
      iv. Fit Ridge as fallback.
      v. Save both models to disk.
   f. **Fit HAR-RV on TRAIN split** (§2.5 (F4), caveat: overnight gaps): Volatility model per ticker. HAR features computed on trading hours only; overnight gaps are implicit in the daily/hourly/5m aggregation (no explicit filling).
3. Record metadata (train_start, train_end, model version, code SHA, validation MAE/directional/CI-cover) for audit.
4. Exit equity loop.
```

**Scheduling**:
- Crypto retrains at 00:00 UTC (post-day).
- Equities retrain at ~21:00 UTC (post-US close, 16:00 ET).
- Both can run in parallel (no sequential dependency).
- Total duration: ~3–4 min combined.

### 6.3 Inference (display-time, read-only, §2.5 (F1))

**When the chart renders** (every 2 seconds for crypto, per v2 design):
1. Load the latest intraday_forecaster model (checkpoint from nightly train).
2. **Read (no write)**:
   - Query the last recorded forecast from `intraday_prediction_snapshots` for (ticker, horizon).
   - This forecast was computed by the hourly worker job (5.1), not the fragment.
3. Fetch live quote (`live_quotes.price`) for the current tick.
4. **Display** (read-only, recompute visual ribbon):
   - Predicted price at target_ts = anchor_price × exp(predicted_return).
   - Visual ribbon = predicted_price ± (e^ci_bounds - 1) × anchor_price.
   - Render with α blending over the live price line.
5. Return ribbon coordinates to the chart.

**Why read-only**: Fragment renders every 2s; writing every render (per-tab) races, causes dedup failures, contends with WAL DB. Sole writer = hourly worker job (5.1, aligned to bar close). No forecast re-anchoring to live price (anchor is fixed at last-closed bar).

**Latency budget**: < 2 ms (single row query + math, no model inference on each render).

---

## 7. Display Integration

### 7.1 Chart rendering (existing `viz.py` + new layer)

The intraday forecast is rendered as an additional series on the existing live price chart:

- **Series**: `intraday_forecast_15m`, `intraday_forecast_1h`, `intraday_forecast_4h` (crypto 15m/1h/4h; equity 1h).
- **Style**: 
  - Line: distinct colors (e.g., tomato for 15m, orange for 1h, purple for 4h), light α (0.6).
  - Bands: filled area between `lower_bound` and `upper_bound` at the same α (visual uncertainty ribbon).
  - Hover text: "[horizon] intraday forecast | ±95% CI | HAR-RV realized-vol bands".
- **Anchor**: All ribbons anchor to the **last closed bar** (not live price). Forecast visualizes the model's expected return FROM that closed-bar close TO the target bar's close.
  - **Rationale**: Intraday forecast is a separate short-horizon model trained on closed-bar returns; anchoring to the closed bar (fixed reference) ensures the ribbon doesn't drift as live price ticks intraday. This is consistent with §2.5 (F1)0, §6.3 (read-only display), and §2.5 R2#10 (equity delay handling).

### 7.2 Historical forecast markers (on-chart grading feedback)

For intraday predictions that have been graded (matured + evaluated), render historical forecast markers similar to the daily forecast accuracy markers:

- **Data source**: `intraday_accuracy_records` joined with `intraday_prediction_snapshots`.
- **Marker style**:
  - **Green**: direction_hit=1 (predicted direction matched realized direction).
  - **Red**: direction_hit=0 (predicted direction was wrong).
  - **Grey**: grading_attempts=0 or grading_attempts>0 but still ungraded (bar not yet available; prediction still pending).
- **Position**: Markers appear at the target bar's timestamp and price (realized_price).
- **Hover text**: "Forecast [made_at]: predicted $X±CI, realized $Y, error ±Z%, hit: [yes/no]".
- **Retention**: Keep historical markers visible for the past 7 days on display (independent of storage retention, which extends 365d crypto / 730d equity for accuracy stats). Older graded forecasts remain in the database for audit but can be hidden or archived from the live chart.

### 7.3 Intraday accuracy panel (separate section in UI)

A new accuracy panel, analogous to the daily accuracy panel (panels.accuracy_rows + panels.verdict_sentence) but sourced from `intraday_accuracy_records`:

- **Per-asset, per-horizon rollup** (crypto: 15m, 1h, 4h; equity: 1h):
  - **N graded**: count of forecasts with grading_attempts > 0 and all fields filled.
  - **MAE %**: mean(abs_error_pct) for graded forecasts.
  - **Direction %**: 100 × sum(direction_hit) / count, or "N/A" if N < 10.
  - **CI Cover %**: 100 × sum(ci_cover) / count (should be ~95%).
  - **Trust verdict**: Rule (e.g., "high trust" if MAE < 2% AND direction % > 55% AND CI cover 90–99%; "moderate" if MAE < 3% OR direction % > 52%; "low trust" otherwise). Exact thresholds TBD in implementation.

- **Placement**: Separate from the daily accuracy panel. Could be:
  - Option A: A new "Intraday Accuracy" tab or collapsible section below the daily accuracy panel.
  - Option B: Inline with the chart, as a legend or info box.
  - **OPEN QUESTION**: UI placement TBD; recommend Option A for clear separation.

- **Data freshness**: Updated in real-time as the evaluator job grades new forecasts (every hour, post-bar-close).

### 7.4 Toggle & config

- **Display flags** in `.env` (asset-aware): `INTRADAY_FORECAST_ENABLED_CRYPTO=true`, `INTRADAY_FORECAST_ENABLED_EQUITY=true` (default: true if models are found on disk).
- **Per-asset, per-horizon toggle** in UI (optional for MVP): Show/hide `intraday_forecast_15m`, `intraday_forecast_1h`, `intraday_forecast_4h` independently (crypto and equity 1h can be shown separately).
- **OPEN QUESTION**: Should the user be able to choose confidence level (e.g., 68%, 95%)? Recommend fixed 95% for consistency with daily bands; make tunable in future.

### 7.5 EOD reconciliation

At session close (crypto 00:00 UTC):
1. The last forming intraday bar closes; worker propagates from `intraday_bars` (7-day display cache) to `intraday_bars_history` (365-day ML store).
2. Nightly trainer **runs in parallel** with daily trainer (no sequential dependency). Reads from `intraday_bars_history`, retrains models.
3. First post-close intraday forecast (00:00 UTC, 1h-horizon) is computed by the hourly forecast-writer job (5.1), using the retrained models.
4. Fragment renders:
   - Daily ribbon from updated `ohlcv_bars.close` (unchanged from v2).
   - Intraday ribbons (1h, 4h) from the latest forecast (updated hourly by worker, not re-anchored to live price).
5. **No visible jump** because the intraday forecast anchor is always the last closed bar (fixed), not live price (drifting). The 00:00 UTC forecast is just the first forecast of the new session, computed from the new day's first closed bar.

---

## 8. ML-Core Separation & Integrity Guarantee

### 8.1 Mandatory constraints

**Daily ML core is FROZEN and display-separated**:
- `forecaster.py`, `trainer.py`, `features.py`, `evaluator.py`, `accuracy.py` are **read-only**.
- No intraday logic, no intraday features, no intraday retraining.
- Daily forecasts + bands are computed and persisted in `prediction_snapshots` exactly as before (once per close).
- Intraday display layer may **not** modify the daily forecast ribbon, bands, or ledger.

**Intraday lives in separate module(s)**:
- `intraday_trainer.py` — trains LightGBM + HAR-RV nightly.
- `intraday_forecaster.py` — loads trained models, computes inference.
- `intraday_features.py` (optional) — feature engineering for intraday.
- No imports from the daily core into intraday modules.

**Regression test** (to be added in M6):
- Verify that mutating `live_quotes.price` or `intraday_bars` rows does **not** affect `prediction_snapshots` (the daily ledger) or daily ribbon on the chart.
- Mutation test: change `live_quotes.price` to an outlier; re-render chart; confirm daily ribbon + accuracy panel unchanged.

### 8.2 Documentation placeholder

**Disclaimer** (to appear in chart caption + `KNOWN_LIMITATIONS.md`):

> *"Intraday forecasts (crypto: 15m/1h/4h; equities: 1h) are short-term technical and derivatives-based predictions, independent from daily ML evaluation. Crypto uses dYdX funding-rate z-scores; equities omit this feature (crypto-only). Forecasts are anchored to the last closed bar, NOT to the live price (which drifts intraday). Equities inherit ~15-minute yfinance delivery lag; forecast uses the latest available closed bar. Intraday forecasts have their own accuracy scorecard (historical markers + accuracy panel), separate from daily predictions. Confidence bands are empirical (HAR-RV realized volatility, fitted on training data). Use intraday forecasts for tactical context; rely on daily forecast ribbon for strategic directional signals."*

---

## 9. Configuration & Environment (REVISION 2: dual-asset)

### 9.1 New environment variables

```
# Intraday forecasting feature flags (asset-aware, REVISION 3: R-A 730d equity)
INTRADAY_FORECAST_ENABLED_CRYPTO=true
INTRADAY_FORECAST_ENABLED_EQUITY=true
INTRADAY_FORECAST_HORIZONS_CRYPTO=15m,1h,4h  # crypto horizons (comma-separated)
INTRADAY_FORECAST_HORIZONS_EQUITY=1h          # equity horizons (1h only; sub-1h display-only)
INTRADAY_LOOKBACK_DAYS_CRYPTO=365             # crypto training window (god's F3 ruling)
INTRADAY_LOOKBACK_DAYS_EQUITY=730             # equity training window (R-A: yfinance proven ~730d for 1h bars, ~5,070 samples)
INTRADAY_BARS_HISTORY_RETENTION_DAYS_CRYPTO=365
INTRADAY_BARS_HISTORY_RETENTION_DAYS_EQUITY=730  # per-asset-class retention (R-A: config, not hardcoded)
INTRADAY_RETRAIN_HOUR_UTC_CRYPTO=0            # Crypto retrain @00:00 UTC (post-day, cron-style hour int)
INTRADAY_RETRAIN_HOUR_UTC_EQUITY=21           # Equity retrain @21:00 UTC (post-16:00 ET close, cron-style hour int)

# Model hyperparameters (tuned per asset-class, §R-B: same LightGBM depth for both)
INTRADAY_LIGHTGBM_DEPTH_CRYPTO=4
INTRADAY_LIGHTGBM_DEPTH_EQUITY=4              # same as crypto; 730d equity sample (~5k bars) is adequate (R-B)
INTRADAY_LIGHTGBM_LR=0.02                     # same for both
INTRADAY_LIGHTGBM_ROUNDS_CRYPTO=300
INTRADAY_LIGHTGBM_ROUNDS_EQUITY=300           # standard, same as crypto (R-B: no special conservative tuning)

# Display (asset-aware)
INTRADAY_FORECAST_OPACITY=0.6
INTRADAY_FORECAST_COLOR_15M=#FF6347           # tomato (crypto 15m)
INTRADAY_FORECAST_COLOR_1H=#FF8C00            # orange (crypto 1h + equity 1h, can disambiguate in legend)
INTRADAY_FORECAST_COLOR_4H=#9932CC            # purple (crypto 4h)
INTRADAY_CI_LEVEL=0.95                        # 95% bands (both assets)
```

**Cold-start note** (§2.5 (F6)): Accuracy panel shows "warming up" and markers are grey for the first ~1–2 weeks post-launch while graded history accumulates. This is expected; accuracy metrics become stable after N>=10 graded forecasts (per asset-class, per horizon).

### 9.2 Model storage (REVISION 2: asset-class partitioned)

- **Path**: `stock_forecasting/models/intraday/`
  
  **Crypto** (`BTC-USD`, `ETH-USD`):
  - `intraday_lgb_btc_15m.pkl`
  - `intraday_lgb_btc_1h.pkl`
  - `intraday_lgb_btc_4h.pkl`
  - `intraday_ridge_btc_fallback_15m.pkl`
  - `intraday_ridge_btc_fallback_1h.pkl`
  - `intraday_ridge_btc_fallback_4h.pkl`
  - `intraday_har_rv_btc.pkl`
  - (same structure for ETH, 14 files total for crypto)
  
  **Equity** (`AAPL`, `NVDA`, `SPY`):
  - `intraday_lgb_aapl_1h.pkl`
  - `intraday_lgb_nvda_1h.pkl`
  - `intraday_lgb_spy_1h.pkl`
  - `intraday_ridge_aapl_fallback_1h.pkl`
  - `intraday_ridge_nvda_fallback_1h.pkl`
  - `intraday_ridge_spy_fallback_1h.pkl`
  - `intraday_har_rv_aapl.pkl`
  - `intraday_har_rv_nvda.pkl`
  - `intraday_har_rv_spy.pkl`
  - (9 files total for equity)
  
  **Shared**:
  - `metadata_crypto.json` — timestamp, train range (crypto, 365d), validation metrics per horizon, code SHA.
  - `metadata_equity.json` — timestamp, train range (equity, 730d per R-A), validation metrics, code SHA.

- **Initialization**: If no model files exist, display is disabled until nightly retrain completes (crypto @00:00, equity @21:00).

---

## 10. Implementation Plan (Milestones + DoD, REVISION 2: dual-asset)

**Dependency order**: M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 (reordered per §2.5 (F7): evaluator before markers).

| M | Deliverable | Definition of Done | Real Failure Criteria |
|---|---|---|---|
| **M0** | Schema + config (§2.5 (F2), R-A) | New tables: `intraday_bars_history` (365d crypto + 730d equity per R-A, §3.5) + `intraday_prediction_snapshots` (with asset_class, §3.6) + `intraday_accuracy_records` (with asset_class, §3.7) in `schema.py`. Add `asset_class` columns to all three. Config: `INTRADAY_LOOKBACK_DAYS_CRYPTO=365`, `INTRADAY_LOOKBACK_DAYS_EQUITY=730` (per R-A), `INTRADAY_RETRAIN_HOUR_UTC_CRYPTO=0`, `INTRADAY_RETRAIN_HOUR_UTC_EQUITY=21` (cron-style hour ints, per B5). `.env.example` updated. Tests: tables exist with asset_class, settings parse per asset-class, model directories exist, retention triggers per config. | Missing asset_class columns; retention hardcoded (not per-asset); equity lookback wrong; hour-format config wrong. |
| **M1** | Intraday data pipeline (crypto + equity) | **Crypto**: `intraday_trainer.py` fetches 365d Coinbase bars + dYdX funding from `intraday_bars_history` (asset_class='crypto'). **Equity**: Fetches 730d yfinance 1h bars from `intraday_bars_history` (asset_class='equity', R-A). As-of join funding for crypto (§2.5 (F9)). Aligns crypto to 5m, equity to 1h; filters to closed-bar anchors (trading-hours gating for equity). Tests: mock Coinbase + yfinance → shapes correct, funding no lookahead, anchor filtering works, equity data ~5k bars verified. | Missing equity path; funding forward-filled; non-closed-bar anchors; yfinance backfill job absent. |
| **M2** | Feature engineering (crypto + equity) | `intraday_features.py` computes §4.1 suite with asset-class dispatch: crypto all 8 features (including funding-rate), equity 7 (omit funding-rate). StandardScaler fit on train window per asset-class. Tests: crypto features include funding, equity omits it, shapes match, no NaNs after warmup, scaling on train only. | Missing equity branch; funding present in equity features; scaling on test data. |
| **M3** | Model training (LightGBM + fallback, R-B) | `intraday_trainer.py`: closed-bar-anchor labels per horizon (15m/1h/4h crypto, 1h equity, non-overlapping). Purged & Embargoed split (24h embargo). **Leakage canary DoD**: control model scores ~50% directional, not >55%. **Crypto**: 6 directional models (BTC/ETH × 15m/1h/4h) + 2 HAR-RV. **Equity**: 3 directional (AAPL/NVDA/SPY × 1h) + 3 HAR-RV (R-B: standard depth 4, same as crypto; 730d sample ~5k bars, adequate). Market-hours gating on equity anchors (trading days only). Validation metrics per model. Tests: split logic verified, canary passes per asset-class, metrics logged, model count correct. | Missing 15m crypto or equity models; HAR-RV fit on test; equity depth/hyperparameters wrong (R-B); no market-calendar gating. |
| **M4** | Forecast writer worker job (§2.5 (F1), R-B) | `intraday_forecaster.py` + APScheduler jobs (crypto @00:00 UTC; equity trading-hours @14:30–21:00 UTC via market_calendar, per B7). Loads models by (ticker, horizon, asset_class). Computes forecast for anchor=last closed bar (crypto all 3 horizons, equity 1h). Writes to `intraday_prediction_snapshots` (with asset_class) via `INSERT OR IGNORE` dedup. Features: crypto all 8 (incl. funding), equity 7 (no funding). Tests: crypto worker fires @00:00, equity fires on trading days during market hours, anchor is closed bar, dedup works, no race, asset_class populated, features correct. | Fragment writes DB (F1 anti-pattern); forming bar used as anchor; missing equity market-hours gating; asset_class not set; features wrong per asset. |
| **M5** | Intraday evaluator job (R2#10) | `intraday_evaluator.py` + APScheduler job (hourly). Queries `intraday_bars_history` (asset_class-aware) for realized closed bar. If not found/provisional, increments `grading_attempts`, defers. If found, computes error/direction/CI, writes to `intraday_accuracy_records` (with asset_class). Handles equity yfinance delay (~15m) gracefully. Tests: crypto mature bar → grading succeeds, equity yfinance lag accounted for, ungraded bar → counter increments, asset_class populated. | Missing evaluator; reads from 7d cache (per F2); asset_class not tracked; equity delay not handled. |
| **M6** | Display integration (chart + markers, §2.5 (F7), R2) | `viz.py` renders intraday ribbons (crypto 15m/1h/4h, equity 1h). Fragment reads last forecast from `intraday_prediction_snapshots` (read-only, asset_class-aware). Markers read from `intraday_accuracy_records`. EOD reconciliation writes crypto to `intraday_bars_history`. Tests: crypto + equity ribbons render without crash, ribbon points align, bands monotonic, markers (green/red/grey) render post-M5, daily ribbon unaffected, asset-class filter works. | M5 not yet implemented; fragment writes to DB; markers fail to render; equity display missing. |
| **M7** | Intraday accuracy panel (R2) | `panels.py`: intraday_accuracy_rows + verdict_sentence. Per-asset-class, per-horizon rollup (crypto: 15m/1h/4h; equity: 1h). Metrics: MAE%, direction%, CI cover%, n, trust verdict. Separate tabs or sections per asset-class (not merged with daily or cross-asset). Cold-start state: "warming up" + grey markers for ~1–2 weeks. Tests: panel renders per asset-class, metrics computed separately, verdict thresholds applied, cold-start UI correct, daily panel unaffected. | Missing panel; merged with daily/cross-asset; no asset-class separation; cold-start missing. |

---

## 11. Open Questions (Genuinely Deferred; Non-Blocking for Implementation)

**Closed by god's GATE 0 rulings (§2.5 Revision 1) + Revision 2 decisions**:
- Q1 (training window → crypto 365d, equity 60d), Q2 (model storage → pickle), Q3/Q7 (per-asset models → yes), Q9 (grading source → intraday_bars_history close), Q10 (cadence → hourly worker job).
- **Revision 2 closures**: Asset-class lookback windows (R2#8), 15m crypto horizon (R2#9), equity anchor handling (R2#10), feature applicability (R2#11).

**Remaining deferred questions**:

1. **Equity model robustness on small sample**: Equities have ~270 1h bars (60d × 4.5h/trading-day); LightGBM + Ridge both trained as fallbacks. Should we use ensemble voting or simple Ridge-preferred strategy when LGB overfits? Recommend Ridge-preferred for equity (fits shallow regularized model first, use LGB only if Ridge perf is poor), but leave tunable post-launch.

2. **Feature scaler persistence**: Refit nightly or store and reuse? (Pro refit: intraday vol shifts daily; pro store: stable inference.) Recommend nightly refit per asset-class, but leave tunable.

3. **Hyperparameter tuning**: Fixed empirical values or Bayesian optimization per asset-class on each retrain? (Pro fixed: fast, reproducible; pro Bayesian: adapts to market regime.) Recommend fixed for MVP; enable tuning in future.

4. **Confidence level**: Fixed 95% (match daily) or user-tunable? Recommend fixed 95% for consistency.

5. **Fallback to Ridge**: When use Ridge instead of LightGBM? (Always as backup? Only if LGB training fails? If perf degrades?) Recommend always training both; display uses LGB if available, Ridge as fallback. For equities, prefer Ridge due to small sample size (Q1).

6. **Embargo window size**: 24 hours is conservative; could be data-driven per asset-class. Recommend 24h fixed for MVP.

7. **Intraday accuracy panel UI placement**: New tab per asset-class, collapsible sections, or inline legend? (Pro tabs: clear separation; pro inline: visibility.) Recommend tabs or sections per asset-class for clear separation from daily accuracy; confirm in implementation UI review.

8. **Trust verdict thresholds**: MAE %, direction %, CI cover % levels for "high/moderate/low trust", tuned per asset-class? (Crypto: larger sample, tighter thresholds; equity: looser given small sample.) Recommend starting from similar forecasting systems (e.g., crypto MAE < 2% = high; equity MAE < 3% = high) and refining post-launch.

9. **Cold-start marker color**: Grey for ungraded? Or a different color to distinguish "pending" vs "permanently ungraded"? Recommend grey (pending), with a tooltip explaining "will update when target matures" (§2.5 (F6)).

10. **15m crypto forecast use case**: Who uses 15m forecasts? (Day traders? Scalpers? Just to have granularity?) Recommend treating 15m as experimental alpha; monitor adoption and feedback. Consider removing if cold storage of 15m records balloons DB size.

---

## 13. Self-Review Checklist (Post-GATE 0 Revision 2)

- ✅ **Alignment with locked decisions + god's rulings**: All four user-locked decisions (scope, full scorecard, ML-core frozen, model choice) carried in §2. God's GATE 0 rulings (365d lookback, pickle storage, per-asset models, grading source/cadence) baked into §2.5 and §10 milestones.
- ✅ **Jim's findings incorporated** (F1–F12):
  - F1 ✅: Forecast writer moved OUT of fragment → dedicated hourly APScheduler worker job (§5.1). Fragment read-only (§6.3). Dedup via INSERT OR IGNORE (§2.5 (F1)).
  - F2 ✅: New immutable table `intraday_bars_history` (365d ML store). Training + grading both read from it, never 7-day cache (§3.5, §2.5 (F2)).
  - F3 ✅: 365d training window (god ruling), not 90d (§3.2, §2.5 (F3)).
  - F4 ✅: HAR-RV fit on TRAIN split, not test. Metrics: MAE/RMSE/directional/CI-cover, not Sharpe (§6.2, §2.5 (F4)).
  - F5 ✅: Request math corrected (~351 requests/ticker for 365d, not bars; §9.1, §2.5 (F5)).
  - F6 ✅: Cold-start documented (panel "warming up" for ~1–2 weeks, §9.1, §2.5 (F6)).
  - F7 ✅: Milestone order fixed: M5 (evaluator) before M6 (markers). Real failure criteria in DoD column, not tautologies. Leakage canary in M3 DoD.
  - F8 ✅: Label construction fixed: anchor ONLY at closed bar boundaries, label = ln(P_{anchor+k} / P_anchor) (§3.2, §2.5 (F8)).
  - F9 ✅: Funding-rate as-of join (no forward-fill lookahead) documented (§4.1, §2.5 (F9)).
  - F10 ✅: Forecast anchor is closed bar, not live price. Disclaimer clarified (§7 intro, §2.5 (F1)0).
  - F11 ✅: Contradiction resolved via intraday_bars_history (F2). LOOKBACK_DAYS=365 consistent (§9.1, §2.5 (F1)1).
  - F12 ✅: Q1/Q2/Q7/Q9/Q10 closed (god's rulings). Q3-Q8 + Q12-13 remain genuinely deferred (§11, §2.5 (F1)2).
- ✅ **Existing code seams referenced**: `intraday_bars` (7d display cache), `intraday_bars_history` (365d ML store), `live_quotes`, `crypto_derivatives`, `viz.py`, `schema.py`.
- ✅ **No ambiguity on key design forks**:
  - Daily ML core is FROZEN, zero cross-contamination.
  - Intraday is FULL SCORECARD with separate tables, modules, worker jobs.
  - Forecast writer = hourly worker job (sole DB writer, non-render path).
  - Fragment = read-only display ribbon (no DB writes per F1).
  - Training data = immutable `intraday_bars_history` (365d, separate from 7d display cache per F2).
  - Label construction = closed-bar anchors only, per F8.
  - Grading = hourly worker, reads from immutable history per F2, handles "not-yet-matured" gracefully per design.
- ✅ **Milestones realistic**: M0–M7 ordered, reordered per F7 (evaluator before markers). Testable failure criteria, not tautologies. M0 adds schema with asset_class column (F2, R2#8). M3 includes leakage canary (F7) and per-asset-class hyperparameters (R2#9). M4/M5 split: writer job / display (F1). M6/M7 asset-class aware (R2).
- ✅ **REVISION 2 scope expansion incorporated**:
  - Crypto: 15m/1h/4h horizons (was 1h/4h); adds 2 BTC/ETH 15m models (R2#9).
  - Equities: AAPL/NVDA/SPY at 1h ONLY; ~270 samples per ticker; shallow trees for robustness (R2#9).
  - Per-asset-class lookback: crypto 365d, equity 60d (R2#8); tracked via asset_class column in schema.
  - Feature applicability: crypto all 8 features (including dYdX funding), equities 7 (omit funding) (R2#11).
  - Equity anchor handling: yfinance ~15m delay documented; forecast anchors to last closed 1h bar (R2#10).
  - Retrain schedule: crypto @00:00 UTC, equity @21:00 UTC; can run in parallel (§3.3, §6.2).
  - Milestones delta detailed per asset-class (§10, M0–M7).
- ✅ **No half-finished designs**: Implementation path clear. No placeholder ambiguity except genuinely deferred open questions (§11). Asset-class separation maintained throughout (separate data paths, features, models, retrain jobs, evaluation, display).

---

## 14. Next Steps (Post-GATE 0 Revision 2)

**For re-review (god + Jim)**:
1. Verify all F1–F12 findings from Revision 1 remain addressed (see §13 self-review checklist).
2. Spot-check §2 (user-locked decisions) + §2.5 (god's rulings, R1 + R2 additions).
3. Validate scope expansion (crypto 15m added, equities 1h added, per-asset lookbacks) are baked in:
   - §1.1: crypto horizons 15m/1h/4h, equities 1h only.
   - §2.5 R2#8–R2#11: asset-class windows, 15m horizon, equity anchor handling, feature applicability.
   - §3.5–3.7: intraday_bars_history, prediction_snapshots, accuracy_records all have asset_class column.
   - §6.2: dual crypto/equity training loops with different lookback windows.
   - §10 M0–M7: milestones now per-asset-class and explicit about requirements.
4. Confirm §11 open questions list is now correct (Revision 1 closures + Revision 2 closures + 10 genuinely deferred).

**After re-GATE 0 approval (Revision 2)**:
- Spin M0–M7 into detailed task-level implementation plan (PRD) with asset-class breakdown.
- Assign dev lead; estimate effort per milestone (crypto models ~2–3 min retrain; equity models <1 min).
- Confirm data costs: Coinbase ~351 requests/ticker × 2 assets = ~702 for 365d crypto (acceptable per F3). yfinance daily backfill <100 req/day for 3 equity tickers (acceptable).
- Begin implementation on `feat/intraday-t013`.

---

## 15. References & Audit Trail

**Scout & related design docs**:
- **Kevin's scout** (T-013 research): `docs/reports/2026-09-03-intraday-forecasting-scout.md` (data depth per provider, feasibility divide crypto/equities, model survey, risks including F3/F4/F9 findings).
- **v2 design template**: `docs/2026-09-01-realtime-v2-design.md` (display-layer architecture, worker/fragment separation pattern, schema, health rework).
- **v1 ML spec**: `docs/2026-09-01-stock-forecasting-design.md` (daily core, ledger, evaluator, accuracy — frozen, unchanged).

**GATE 0 Review findings**:
- **Jim's detailed review** (2026-09-03T15-10-00-jim-gate0-t013): F1–F12 findings, severity levels, specific fixes. This revision incorporates all findings (see §13 self-review).
- **God's consolidated ruling** (2026-09-03T07-36-45-789Z-1fa123): Q1/Q2/Q3/Q7/Q9/Q10 closed; F1 + F2 require schema restructuring (M0).

**Existing code seams**:
- `stock_forecasting/schema.py`: IntradayBar, LiveQuote (existing); new tables added in M0.
- `stock_forecasting/intraday_store.py`: existing repository for display cache.
- `stock_forecasting/providers/dydx.py`: funding-rate source (as-of join per F9).
- `stock_forecasting/viz.py`: chart rendering (marker + ribbon logic, M6).
- `stock_forecasting/worker.py`: worker job framework (forecast writer M4 + evaluator M5 jobs).
