# stock_forecasting — Intraday Forecasting (Crypto 1h/4h, Full Scorecard)

> **Status**: DESIGN — Phase 1, pre-GATE 0 (BOUNCE revision after Jim's review).  
> **Date**: 2026-09-03 · **Author**: Pam (pam-mtjr5nuk) · **Branch**: `feat/realtime-v2` (GATE 0 revision)  
> **Scope**: Design only — no implementation, no schema migration, no model code, no milestone execution.  
> **Decision Authority**: User-locked decisions (§2) + god's GATE 0 rulings (§2.5 below) are mandatory; remaining open questions are deferred.

---

## 0. Executive Summary & Scope

This design introduces **intraday forecasting models with full scorecard** to `stock_forecasting` alongside the v2.0.0 display layer and the frozen daily ML core. The intraday layer is **complete** (training → inference → grading), **crypto-exclusive** (BTC-USD, ETH-USD), and scoped to **1-hour and 4-hour prediction horizons**.

### Why intraday now?
- v2.0.0 delivers real-time crypto price display via Coinbase WebSocket.
- Intraday bars are already flowing into `intraday_bars` (100-row cache).
- dYdX hourly funding rate is already integrated as a feature; it aligns perfectly with 1-hour model cadence.
- A separate short-horizon intraday model with grading unlocks actionable directional/volatility signals and a live accuracy panel without compromising the frozen, proven daily ML core.

### Key constraints
1. **Crypto-only**: yfinance's 15-minute delay and 60-day cap make equity sub-hour forecasting fundamentally broken (Kevin's scout, §2.2).
2. **Full scorecard**: Intraday forecasts are persisted, graded when targets mature, and displayed in an intraday accuracy panel (separate from daily accuracy).
3. **ML-core frozen**: The daily forecaster, trainer, features, evaluator, accuracy pipeline remain untouched. Intraday uses separate tables, modules, jobs. No cross-contamination.
4. **No real-time model updates**: Intraday model retrains nightly, once, like the daily model — not streaming or tick-driven.

---

## 1. Scope & Out-of-Scope

### 1.1 Scope (committed)

- **Asset coverage**: BTC-USD, ETH-USD intraday price forecasting on 1-hour and 4-hour horizons.
- **Data sources**: Coinbase REST (paginated historical 1m/5m candles) + live `intraday_bars` (via WS).
- **Model class**: LightGBM (primary) or Ridge regression (fallback) on tabular rolling features.
- **Feature engineering**: Intraday VWAP distance, realized-vol ratios, EWMA return spreads, dYdX funding-rate z-scores, volume acceleration.
- **Methodological guardrail**: Purged & Embargoed TimeSeriesSplit (López de Prado) to prevent label-leakage false positives.
- **Output**: Directional point forecast (log-return prediction) + HAR-RV dynamic confidence bands for visual ribbon.
- **Persistence**: Each forecast is written to `intraday_prediction_snapshots` table (separate from daily `prediction_snapshots`) immediately after inference.
- **Grading**: Intraday evaluator job grades forecasts when their target horizons mature, computing signed error, directional accuracy, CI coverage.
- **Accuracy panel**: Separate intraday accuracy panel (per horizon: MAE %, direction %, CI cover %, n, trust verdict) alongside (not merged with) the daily accuracy panel.
- **Chart display**: Intraday forecast ribbon overlaid on chart; historical forecast markers (green/red/grey) for past intraday predictions.
- **Retrain cadence**: Nightly, once per day post-close (same cadence as daily model).

### 1.2 Explicitly OUT-of-scope this milestone

- **Equities**: Not covered due to the 15-minute delay and 60-calendar-day lookback ceiling (Kevin's scout, §2.1, §4.2).
- **Sub-1-hour horizons** (1m / 5m next-bar): Microstructure noise dominates; SNR too low (Kevin's scout, §2.1).
- **Real-time model retraining**: Intraday model is trained once per day; not adapted intraday based on live ticks.
- **Intraday position-sizing / trade signals / alerts**: Forecasts and accuracy panel are informational only; no prescriptive advice or notifications.
- **Deep learning (LSTM/Transformers/PatchTST)**: Ruled out by data starvation and overfitting risk on 60-day windows (Kevin's scout, §3.4).
- **Merging intraday accuracy into daily accuracy**: Intraday scorecard is completely separate (different table, different panel, different ledger).

---

## 2. Locked Decisions (from user)

These decisions are immutable for this design and must be carried verbatim into implementation:

1. **SCOPE**: Crypto-only, BTC-USD + ETH-USD. Horizons 1h and 4h. Equities explicitly OUT.
2. **FULL SCORECARD**: The intraday forecast is persisted and graded. Separate `intraday_prediction_snapshots` table, separate intraday evaluator job (grades when target matures), separate intraday accuracy panel. Completely independent of the daily `prediction_snapshots` ledger and daily accuracy panel.
3. **ML-core frozen**: The daily core (forecaster.py, trainer.py, features.py, evaluator.py, accuracy.py) stays FROZEN and display-separated. Intraday uses separate tables (intraday_prediction_snapshots, intraday_accuracy_records), separate modules, separate worker jobs. Zero cross-contamination with daily ledger.
4. **MODEL**: LightGBM (Ridge fallback) on rolling tabular features (intraday VWAP distance, realized-vol ratios, EWMA return spreads, dYdX funding-rate z-scores). HAR-RV for the CI band. Purged + Embargoed TimeSeriesSplit for overlapping k-step-forward-return label leakage. CPU, local, nightly retrain cadence.

## 2.5 God's GATE 0 Rulings (Consolidated from Jim's F1–F12 Review)

These decisions close the open questions and resolve load-bearing findings:

1. **Training window (F3)**: 365 days for BOTH horizons. Set `INTRADAY_LOOKBACK_DAYS=365` in config. Rationale: 90d is thin for 4h (~540 independent samples < 5k needed); 365d provides ~2,190 4h-bars/yr + ~8,760 1h-bars/yr. Coinbase depth supports multi-year keyless pagination (~351 requests/ticker, acceptable cost).

2. **Model persistence (Q2)**: Pickle files in `stock_forecasting/models/intraday/` + `metadata.json`. NO `intraday_model_runs` table for MVP.

3. **Per-asset models (Q7)**: SEPARATE model per (ticker, horizon). 4 directional models (BTC-1h, BTC-4h, ETH-1h, ETH-4h) + HAR-RV per ticker. No cross-asset model.

4. **Grading realized price (Q9)**: Closed-bar close from `intraday_bars_history` (not live tick).

5. **Grading job cadence (Q10)**: Hourly APScheduler worker job (short-horizon forecasts mature fast). Aligned to bar close.

6. **Forecast writer location (F1)**: Dedicated hourly worker job (APScheduler, not fragment/render path). Sole writer to `intraday_prediction_snapshots`. Fragment display ribbon is READ-ONLY (recompute from model + live_quotes, persist nothing). Anchor to last CLOSED bar, not forming bar. Dedup via `UNIQUE(ticker, horizon, anchor_ts)` + `INSERT OR IGNORE`.

7. **Immutable training/grading store (F2)**: New table `intraday_bars_history` (immutable, written once per closed bar, pruned only at training-window horizon = 365d). Both training and grading read from it. This resolves the 7-day prune conflict.

---

## 3. Architecture & Data Flow

### 3.1 Intraday forecasting as a separate ML subsystem

```
Daily ML Core (FROZEN)               Intraday ML Subsystem (NEW, Full Scorecard)
├─ forecaster.py                    ├─ intraday_forecaster.py (new)
├─ trainer.py                       ├─ intraday_trainer.py (new)
├─ features.py                      ├─ intraday_features.py (new, or inline)
├─ evaluator.py                     ├─ intraday_evaluator.py (new)
└─ accuracy.py                      └─ intraday_store_models.py (if persisting model files)

Reads from:                         Intraday reads from:
└─ ohlcv_bars (1d, immutable)      ├─ intraday_bars (1m/5m candles, 7-day cache)
                                    ├─ live_quotes (current tick)
                                    ├─ crypto_derivatives (dYdX funding rate)
                                    └─ (for grading) intraday_bars again when target matures

Writes to:                          Intraday writes to:
├─ prediction_snapshots (daily)    ├─ intraday_prediction_snapshots (daily inference records)
├─ accuracy_records (daily)        └─ intraday_accuracy_records (daily grading results)
└─ model_runs (training metadata)  (Model runs: separate table or files, TBD in §3.4)

Zero shared tables between daily and intraday. Separate display panels.
```

### 3.2 Training data pipeline for intraday models

**Input source**: Historical Coinbase candles from `intraday_bars_history` (via existing `providers/coinbase.py` REST endpoint for backfill, but training reads immutable history).
- **Granularity**: 5-minute bars (fixed; same as chart intraday bucket, per scope).
- **Lookback window**: 365 calendar days (god's F3 ruling). Covers ~8,760 1h-bars/yr and ~2,190 4h-bars/yr (>5k sample threshold per Kevin's scout §3.2). Coinbase cost: ~351 requests/ticker (~702 both).

**Feature computation** (§4 below):
1. Build OHLCV-derived technical features (VWAP, realized vol, EWMA spreads).
2. Normalize with StandardScaler (fit on training window, apply to validation).
3. Align with dYdX funding-rate series (hourly snapshots, zero-order hold for 5m bars).
4. Merge into a single DataFrame for each (BTC-USD, ETH-USD) pair.

**Label construction** (k-step forward log-return, §F8):
- **Anchor ONLY at closed bar boundaries**: :00 UTC for 1h; :00/:04/:08/:12/:16/:20 UTC for 4h (or equivalently, only use bars whose anchor_ts is a multiple of 1h/4h in seconds).
- Label = $\ln(P_{anchor+k} / P_{anchor})$ where $P_{anchor}$ is the anchor bar's close, $P_{anchor+k}$ is the target bar's close (k ∈ {1h, 4h}), and both anchors are closed bars from `intraday_bars_history`.

**Cross-validation protocol**:
- **Mandatory**: Purged & Embargoed TimeSeriesSplit (López de Prado).
  - Purging: Drop all training samples whose label window overlaps with the test window.
  - Embargoing: Drop training samples immediately following the test set by 24 hours to eliminate serial correlation leakage (§4.5 risk mitigation).
- **Rationale**: k-step forward returns on 5m/1m bars share (k-1) bars of identical price data; naive K-fold leaks future information across folds.

### 3.3 Retrain cadence & schedule

- **When**: Nightly after market close (crypto 00:00 UTC, no equity session boundary).
- **Frequency**: Once per day, same pattern as the daily trainer.
- **Concurrency**: Run intraday trainer **in parallel with** daily trainer (no sequential deps).
- **Cost**: LightGBM training on 90 days of bars @ 5m granularity (~26,000 rows) takes ~5–10 seconds on commodity CPU (per Kevin's scout, §3.2); acceptable for nightly.

### 3.4 Storage & model persistence

**Where trained models live**:
- Option A (lightweight): Pickle the fitted `LightGBMRegressor` and `Ridge` fallback to local `.pkl` files in `stock_forecasting/models/intraday/`.
- Option B (heavier): Store model metadata and coefficients in a new `intraday_model_runs` table (parallel to `model_runs` for daily).
  - **OPEN QUESTION**: Should we persist intraday model snapshots for later audit / comparison? If yes, schema needed; if no, local `.pkl` suffices. Recommend Option A (files) for MVP but leave open.

**Prediction ledger**: New table `intraday_prediction_snapshots` (§3.5 below).

**Accuracy ledger**: New table `intraday_accuracy_records` (§3.6 below).

### 3.5 New table — `intraday_bars_history` (F2 / immutable store)

Immutable, durable store for intraday bars (separate from `intraday_bars`, the 7-day display cache).

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK → `tickers.symbol` | |
| `interval` | TEXT | `1m` \| `5m` (fixed per scope, likely 5m) |
| `ts` | TEXT | bucket start, ISO-8601 UTC (only for CLOSED bars) |
| `open` `high` `low` `close` | REAL | |
| `volume` | REAL | |
| `source` | TEXT | `coinbase_rest` \| `coinbase_ws_finalized` |
| `ingested_at` | TEXT | wall-clock |

**Unique index**: `(ticker, interval, ts)`.  
**Retention**: Pruned only at training-window horizon = 365 days. Daily job: `DELETE FROM intraday_bars_history WHERE ts < now - 365 days`.  
**Writing**: Worker closes intraday bar from `intraday_bars` (display cache), writes once to `intraday_bars_history` with `is_provisional=0` (immutable), then propagates to `intraday_bars` (7-day cache). Single-writer pattern (v2).  
**Reading**: Training pipeline + intraday evaluator read ONLY from `intraday_bars_history`, never from `intraday_bars`.

### 3.6 New table — `intraday_prediction_snapshots`

Mirrors the structure of `prediction_snapshots` (daily) but for intraday horizons. Records every forecast made by the worker job (the sole writer, §5.1).

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK → `tickers.symbol` | |
| `horizon` | TEXT | `1h` \| `4h` |
| `made_at` | TEXT | wall-clock UTC when forecast was written by worker job |
| `anchor_ts` | TEXT | timestamp of the CLOSED bar used to compute features (§F8) |
| `anchor_price` | REAL | close price of the closed anchor bar |
| `predicted_return` | REAL | model's log-return forecast (ŷ), where label = ln(P_{anchor+h} / P_anchor) |
| `predicted_price` | REAL | anchor_price × exp(predicted_return) [denormalized for display] |
| `ci_lower_return` | REAL | HAR-RV lower bound (log-return space, fitted on train split §F4) |
| `ci_upper_return` | REAL | HAR-RV upper bound (log-return space) |
| `ci_lower_price` | REAL | denormalized to price space |
| `ci_upper_price` | REAL | denormalized to price space |
| `target_ts` | TEXT | when the prediction target matures (anchor_ts + 1h or +4h) |
| `model_version` | TEXT | version string of model used |
| `model_sha` | TEXT | git SHA of model training code |

**Index**: `UNIQUE(ticker, horizon, anchor_ts)` + `INSERT OR IGNORE` dedup (§F1).  
**Retention**: Keep intraday predictions indefinitely (full audit trail); no pruning.  
**Grading reference**: When `target_ts` passes, `intraday_evaluator` looks up this row and computes realized return from `intraday_bars_history` (§F2).

### 3.7 New table — `intraday_accuracy_records`

Records grading results once a forecast's target matures. Same shape as `accuracy_records` (daily) but intraday-specific.

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `prediction_id` | INTEGER FK → `intraday_prediction_snapshots.id` | links back to the forecast |
| `ticker` | TEXT FK → `tickers.symbol` | |
| `horizon` | TEXT | `1h` \| `4h` |
| `graded_at` | TEXT | wall-clock when grade was computed |
| `realized_return` | REAL | actual log-return from anchor to target close (closed bar from intraday_bars_history, §F2) |
| `realized_price` | REAL | actual close price at target_ts from intraday_bars_history |
| `signed_error` | REAL | realized_return - predicted_return |
| `abs_error_pct` | REAL | \|signed_error\| × 100 |
| `direction_hit` | INTEGER | 1 if sign(predicted) == sign(realized), else 0 |
| `ci_cover` | INTEGER | 1 if realized_return ∈ [ci_lower_return, ci_upper_return], else 0 |
| `grading_attempts` | INTEGER | counter: incremented each time evaluator checks. Prevents false failures if bar not yet closed. |

**Index**: foreign key on prediction_id.  
**Grading logic** (§5.1 updated per F1/F2): When forecast's target_ts arrives, evaluator queries `intraday_bars_history` for the realized closed bar. If not found, increment `grading_attempts` and defer. When bar found, compute results and write (or update if row exists with attempts counter, filling in computed fields).

---

## 4. Feature Engineering for Intraday Horizons

### 4.1 Intraday-specific feature suite

The following features are designed to capture short-horizon volatility clusters, mean reversion, and momentum regime changes. All features are computed from `intraday_bars` (5m granularity, no lookahead).

#### Technical (OHLCV-derived, no external data)
1. **Intraday VWAP Distance** (per Kevin's scout §3.2):
   - $f = (P_t - \text{VWAP}_{[t-T, t]}) / \sigma_{\text{VWAP}}$ where $T = 1\text{h}$ or $4\text{h}$
   - Captures reversion-to-mean within the session; mean-reversion alpha signal.
   - Computation: rolling VWAP over the last hour (12 × 5m bars) or 4 hours (48 bars), normalized by its standard deviation.

2. **Realized Volatility Ratio** (breakout signal):
   - $f = \sigma_{[t-15\text{m}, t]} / \sigma_{[t-4\text{h}, t]}$
   - High ratio → volatility expansion (potential regime shift); low ratio → calm, mean-reversion likely.

3. **EWMA Return Spread** (multi-scale momentum):
   - $f = \text{EWMA}(\text{returns}, \text{span}=12) - \text{EWMA}(\text{returns}, \text{span}=48)$, normalized by ATR.
   - Short-term vs. medium-term momentum divergence; predicts reversal when signs flip.

4. **Volume Acceleration**:
   - $f = \text{Volume}_t / \text{SMA}(\text{Volume}, 20)$
   - Detects breakout volume; high acceleration suggests sustained direction.

5. **Lagged Log-Returns** (autoregressive):
   - $f = [\ln(P_t / P_{t-1}), \ln(P_t / P_{t-5\text{m}}), \ln(P_t / P_{t-1\text{h}})]$
   - Raw price dynamics at multiple scales; non-zero autocorrelation at 1h/4h horizons.

#### Derivatives (external, dYdX)
6. **dYdX Funding-Rate Z-Score** (24h window, §F9):
   - $f = (\text{FundingRate}_t - \text{mean}_{[t-24\text{h}, t]}) / \text{std}_{[t-24\text{h}, t]}$
   - Extreme funding rates predict volatility expansion or liquidation cascades.
   - Sourced from `crypto_derivatives` table (hourly snapshots).
   - **As-of join** (critical for no lookahead): value at feature-time $t$ = last funding rate PUBLISHED at or before $t$, never the rate that will be settled for the hour containing $t$. Forward-fill only for missing historical data during backfill, not for live features.
   - Applied as zero-order hold to 5m bars (same value for 12 consecutive 5m bars within each hour).

#### Temporal
7. **Hour-of-day** (cyclical encoding):
   - $\sin(2\pi \cdot \text{hour} / 24)$, $\cos(2\pi \cdot \text{hour} / 24)$ to capture global trading session patterns (Asia, EU, US close).

8. **Day-of-week** (cyclical):
   - $\sin(2\pi \cdot \text{dow} / 7)$, $\cos(2\pi \cdot \text{dow} / 7)$ for Friday-expiry / Monday-open effects.

### 4.2 Implementation location

**OPEN QUESTION**: Should intraday feature computation be:
- A) A new standalone module `intraday_features.py` (mirrors the daily `features.py`)?
- B) Inline in `intraday_trainer.py` (keeps logic compact, no separate abstraction)?
- C) A hybrid (base utilities in `intraday_features.py`, trainer calls them)?

Recommend **Option A** (new module) for consistency with the v2 architecture and future reuse in `intraday_forecaster.py`.

### 4.3 Scaling & preprocessing

- Fit a `StandardScaler` on the **training window** of the Purged & Embargoed split (not on test data).
- Apply the same scaler to validation and live inference.
- **OPEN QUESTION**: Should the scaler be re-fit nightly with the latest 90 days of data, or stored and reused across days? Recommend nightly re-fit (intraday volatility levels shift daily) but leave open for tuning.

---

## 5. Intraday Worker Jobs (Forecast Writer & Evaluator)

### 5.1 Forecast writer job (hourly, sole writer, §F1)

**When**: Hourly APScheduler job, aligned to bar close (e.g., top of each hour UTC). Computes both 1h and 4h forecasts in a single run.

**What it does**:
1. Query `intraday_bars_history` for the last CLOSED bar (anchor bar) for each (ticker, interval).
   - Anchor_ts must be on a closed-bar boundary (:00 for 1h; :00/:04/:08/:12/:16/:20 UTC for 4h).
2. Load the fitted intraday_forecaster model (LightGBM or Ridge, per god's Q2).
3. Compute features for this anchor bar (VWAP, vol ratios, EWMA spreads, dYdX funding-rate as-of value, temporal).
4. Call model.predict() → predicted_return.
5. Call intraday_volatility (HAR-RV) → CI bounds (fitted on TRAIN split, per §F4).
6. **Write to `intraday_prediction_snapshots`**:
   - Anchor: anchor_ts (closed bar), anchor_price (its close).
   - Target: anchor_ts + 1h (or +4h for 4h horizon).
   - Predicted return, CI bounds, model version/SHA, made_at = now.
7. **Dedup via `INSERT OR IGNORE`**: `UNIQUE(ticker, horizon, anchor_ts)` ensures no duplicate for the same anchor.

**Concurrency**: Single writer (v2 pattern). No in-render DB writes (contrast with F1's anti-pattern).

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

### 6.1 Model architecture (per locked decision §2)

**Primary**: LightGBM regressor.
- **Target**: k-step log-return ($r_{t, t+k}$ where $k \in \{1, 4\}$ hours).
- **Hyperparameters** (to be proposed in implementation):
  - Depth: shallow trees (3–5) to avoid overfitting on 5-min bar noise.
  - Learning rate: 0.01–0.05 (conservative, stable gradients).
  - Number of boosting rounds: 100–500 (early stopping on validation loss).
  - **OPEN QUESTION**: Should hyperparameters be tuned via cross-validation or fixed empirically? Recommend Bayesian optimization on the Purged & Embargoed split for MVP (tuning on test data would leak).

**Fallback**: Ridge regression.
- Same features, continuous target.
- Alpha (regularization): determined via cross-validation on training set.
- Use Ridge if LightGBM overfits or retrains become too slow.

**Volatility bands (HAR-RV)** (separate from directional forecast):
- Predict next 1-hour / 4-hour realized volatility using Heterogeneous Autoregressive (HAR) model.
- Input: realized vol at 5m, hourly, and 4-hour frequencies over the past day.
- Output: $\hat{\sigma}_{t, t+h}$ used to compute $\pm 1.96 \hat{\sigma}$ confidence bands on the chart.
- **No correlation with the directional model** — HAR is purely volatility-forecasting, independent of the return forecast.

### 6.2 Training loop (nightly, post-close)

**Pseudocode** (§F3: 365d window; §F4: HAR-RV fit on TRAIN; §F8: closed-bar anchors only):
```
1. Fetch latest 365 days of BTC-USD + ETH-USD intraday bars from intraday_bars_history (5m).
2. Fetch latest 365 days of dYdX funding-rate snapshots.
3. Align + resample to common 5m grid; as-of join funding rates (§F9).
4. Filter to closed-bar anchors only (:00 for 1h; :00/:04/:08/:12/:16/:20 UTC for 4h).
5. Compute features (§4.1) on the closed-bar-anchored DataFrame.
6. Construct k-step log-return labels: label = ln(P_{anchor+k} / P_anchor) for k ∈ {1h, 4h}.
7. Split into train/test using Purged & Embargoed TimeSeriesSplit:
   a. Test window: last 14 days (2 weeks).
   b. Purge: drop all training samples whose label window overlaps [test_start, test_end].
   c. Embargo: drop training samples in [test_end, test_end + 24h].
   d. Train on remaining samples.
8. For each (ticker, horizon):
   a. Fit LightGBM on train; validate on (purged + embargoed) test.
   b. **Leakage canary** (§F7): Fit a control model with shuffled labels on the same split; verify it scores ~50% directional on test (not >55%), confirming no label leakage.
   c. Record validation metrics on test: MAE (%), RMSE, directional accuracy (%), CI coverage (%).
   d. Fit Ridge as fallback.
   e. Save both models to disk (pickle files, per god's ruling Q2).
9. **Fit HAR-RV on TRAIN split** (§F4): Volatility model trained on train split, evaluated on test for CI coverage.
10. Record metadata (train_start, train_end, model version, code SHA, validation MAE/directional/CI-cover) for audit.
11. Exit; next day, repeat.
```

### 6.3 Inference (display-time, read-only, §F1)

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

- **Series**: `intraday_forecast_1h` and `intraday_forecast_4h` (one per horizon).
- **Style**: 
  - Line: distinct color (e.g., orange for 1h, purple for 4h), light α (0.6).
  - Bands: filled area between `lower_bound` and `upper_bound` at the same α (visual uncertainty ribbon).
  - Hover text: "1h intraday forecast | ±95% CI | HAR-RV realized-vol bands".
- **Anchor**: All ribbons start from the current live price (`live_quotes.price`), not from `ohlcv_bars.close` (unlike the daily ribbon, which anchors to P_close).
  - **Rationale**: Intraday is a separate short-horizon model; anchoring to the live price allows the user to see the model's next-1h expectation relative to "now".

### 7.2 Historical forecast markers (on-chart grading feedback)

For intraday predictions that have been graded (matured + evaluated), render historical forecast markers similar to the daily forecast accuracy markers:

- **Data source**: `intraday_accuracy_records` joined with `intraday_prediction_snapshots`.
- **Marker style**:
  - **Green**: direction_hit=1 (predicted direction matched realized direction).
  - **Red**: direction_hit=0 (predicted direction was wrong).
  - **Grey**: grading_attempts=0 or grading_attempts>0 but still ungraded (bar not yet available; prediction still pending).
- **Position**: Markers appear at the target bar's timestamp and price (realized_price).
- **Hover text**: "Forecast [made_at]: predicted $X±CI, realized $Y, error ±Z%, hit: [yes/no]".
- **Retention**: Keep historical markers for the past 7 days (matching intraday_bars retention). Older graded forecasts can be archived or hidden.

### 7.3 Intraday accuracy panel (separate section in UI)

A new accuracy panel, analogous to the daily accuracy panel (panels.accuracy_rows + panels.verdict_sentence) but sourced from `intraday_accuracy_records`:

- **Per-horizon rollup** (1h and 4h separately):
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

- **Display flag** in `.env`: `INTRADAY_FORECAST_DISPLAY=true` (default: true if LightGBM models are found on disk).
- **Per-asset toggle** in UI (optional for MVP): Show/hide `intraday_forecast_1h` and `intraday_forecast_4h` independently.
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

**Disclaimer** (to appear in chart caption + `KNOWN_LIMITATIONS.md`, §F10):

> *"Intraday forecasts (1-hour and 4-hour horizons) are short-term technical and derivatives-based predictions, independent from daily ML evaluation. Forecasts are anchored to the last closed bar, NOT to the live price (which drifts intraday). Intraday forecasts have their own accuracy scorecard (historical markers + accuracy panel), separate from daily predictions. Confidence bands are empirical (HAR-RV realized volatility, fitted on training data). Use intraday forecasts for tactical context; rely on daily forecast ribbon for strategic directional signals."*

---

## 9. Configuration & Environment

### 9.1 New environment variables

```
# Intraday forecasting feature flags
INTRADAY_FORECAST_ENABLED=true
INTRADAY_FORECAST_HORIZONS=1h,4h             # comma-separated; subset of {1h, 4h}
INTRADAY_LOOKBACK_DAYS=365                   # training data window (god's F3 ruling)
INTRADAY_BARS_HISTORY_RETENTION_DAYS=365     # immutable ML store retention
INTRADAY_RETRAIN_SCHEDULE=nightly             # "nightly" | "none" (disabled)
INTRADAY_FORECAST_WRITER_INTERVAL_SECONDS=3600  # hourly worker job (aligned to bar close)

# Model hyperparameters (to be tuned per F3 recommendation)
INTRADAY_LIGHTGBM_DEPTH=4
INTRADAY_LIGHTGBM_LR=0.02
INTRADAY_LIGHTGBM_ROUNDS=300

# Display
INTRADAY_FORECAST_OPACITY=0.6
INTRADAY_FORECAST_COLOR_1H=#FF8C00            # orange
INTRADAY_FORECAST_COLOR_4H=#9932CC            # purple
INTRADAY_CI_LEVEL=0.95                        # 95% bands
```

**Cold-start note** (§F6): Accuracy panel shows "warming up" and markers are grey for the first ~1–2 weeks post-launch while graded history accumulates. This is expected; accuracy metrics become stable after N>=10 graded forecasts (per horizon).

### 9.2 Model storage

- **Path**: `stock_forecasting/models/intraday/`
  - `intraday_lgb_btc_1h.pkl`
  - `intraday_lgb_btc_4h.pkl`
  - `intraday_ridge_btc_fallback_1h.pkl`
  - `intraday_ridge_btc_fallback_4h.pkl`
  - `intraday_har_rv_btc.pkl`
  - (same structure for ETH)
  - `metadata.json` — timestamp, train range, validation metrics, code SHA.

- **Initialization**: If no model files exist, display is disabled until nightly retrain completes.

---

## 10. Implementation Plan (Milestones + DoD)

**Dependency order**: M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 (reordered per §F7: evaluator before markers).

| M | Deliverable | Definition of Done | Real Failure Criteria |
|---|---|---|---|
| **M0** | Schema + config (§F2, F3) | New tables: `intraday_bars_history` (365d immutable ML store) + `intraday_prediction_snapshots` + `intraday_accuracy_records` in `schema.py`. Config: `INTRADAY_LOOKBACK_DAYS=365`, worker + evaluator job intervals. `.env.example` updated. Tests: tables exist, settings parse, model directory exists, retention triggers. | Missing tables; retention=90d instead of 365d (§F2); settings absent. |
| **M1** | Intraday data pipeline | `intraday_trainer.py` fetches 365d Coinbase bars + dYdX funding from `intraday_bars_history`. As-of join funding (§F9). Aligns to 5m grid; filters to closed-bar anchors (§F8). Tests: mock Coinbase REST → shape correct, funding no lookahead, anchor filtering works, feature matrix non-null. | Missing module; funding forward-filled (lookahead leak per F9); non-closed-bar anchors included. |
| **M2** | Feature engineering | `intraday_features.py` computes §4.1 suite (§F9: as-of funding). StandardScaler fit on train window. Tests: shape matches, no NaNs after warmup, as-of funding validated, scaling applied. | Missing module; forward-fill funding; scaling on test data. |
| **M3** | Model training (LightGBM + fallback, §F4, F7, F8) | `intraday_trainer.py`: closed-bar-anchor labels (label = ln(P_anchor+k / P_anchor), §F8). Purged & Embargoed split. **Leakage canary DoD**: shuffled-label control model scores ~50% directional on test, not >55%. HAR-RV fit on TRAIN split (§F4). Validation metrics: MAE(%), RMSE, directional(%), CI-cover(%). Tests: split logic verified, canary passes, metrics logged. | Missing training logic; labels on non-closed bars; HAR-RV fit on test data; no leakage canary. |
| **M4** | Forecast writer worker job (§F1) | `intraday_forecaster.py` + APScheduler job (hourly, aligned to bar close). Loads model, computes forecast for anchor=last closed bar, writes to `intraday_prediction_snapshots` via `INSERT OR IGNORE` dedup. Tests: worker fires hourly, anchor is closed bar, dedup works, no race with tabs. | Fragment writes DB (F1 anti-pattern); forming bar used as anchor; dedup via SELECT-then-INSERT (race per F1). |
| **M5** | Intraday evaluator job | `intraday_evaluator.py` + APScheduler job (hourly). Queries `intraday_bars_history` for realized closed bar. If not found/provisional, increments `grading_attempts`, defers. If found, computes error/direction/CI, writes to `intraday_accuracy_records`. Tests: mature bar → grading succeeds, ungraded bar → counter increments, all fields correct. | Missing evaluator; reads from `intraday_bars` (7d cache, can miss historic forecasts per F2). |
| **M6** | Display integration (chart + markers, §F7) | `viz.py` renders intraday ribbons (1h, 4h). Fragment reads last forecast from `intraday_prediction_snapshots` (read-only, per §F1). Markers read from `intraday_accuracy_records` (populated by M5). EOD reconciliation writes to `intraday_bars_history`. Tests: render without crash, ribbon points align, bands monotonic, markers (green/red/grey) render post-M5, daily ribbon unaffected. | M5 not yet implemented (no accuracy records to display markers); fragment writes to DB (F1); markers fail to render. |
| **M7** | Intraday accuracy panel | `panels.py`: intraday_accuracy_rows + verdict_sentence. Per-horizon rollup (MAE%, direction%, CI cover%, n, trust verdict). Separate from daily accuracy panel. Cold-start state: "warming up" + grey markers for ~1-2 weeks (§F6). Tests: panel renders without crash, metrics computed, verdict thresholds applied, cold-start UI correct. | Missing panel; merged with daily accuracy; no cold-start logic. |

---

## 11. Open Questions (Genuinely Deferred; Non-Blocking for Implementation)

**Closed by god's GATE 0 rulings (§2.5)**:
- Q1 (training window → 365d), Q2 (model storage → pickle), Q3/Q7 (per-asset models → yes), Q9 (grading source → intraday_bars_history close), Q10 (cadence → hourly worker job).

**Remaining deferred questions**:

1. **Feature scaler persistence**: Refit nightly or store and reuse? (Pro refit: intraday vol shifts daily; pro store: stable inference.) Recommend nightly refit, but leave tunable.

2. **Hyperparameter tuning**: Fixed empirical values or Bayesian optimization on each retrain? (Pro fixed: fast, reproducible; pro Bayesian: adapts to market regime.) Recommend fixed for MVP; enable tuning in future.

3. **Confidence level**: Fixed 95% (match daily) or user-tunable? Recommend fixed 95% for consistency.

4. **Fallback to Ridge**: When use Ridge instead of LightGBM? (Always as backup? Only if LGB training fails? If perf degrades?) Recommend always training both; display uses LGB if available, Ridge as fallback.

5. **Embargo window size**: 24 hours is conservative; could be data-driven. Recommend 24h fixed for MVP.

6. **Historical marker retention & archival**: Keep markers for 7 days, then archive? Or infinite? (Pro 7d: faster renders; pro infinite: audit trail.) Recommend 7d, with optional archival (TBD post-MVP).

7. **Intraday accuracy panel UI placement**: New tab, collapsible section, or inline legend? (Pro tab: clear separation; pro inline: visibility.) Recommend tab for clear separation from daily accuracy; confirm in implementation UI review.

8. **Trust verdict thresholds**: MAE %, direction %, CI cover % levels for "high/moderate/low trust"? (Tuning, impacts label calibration.) Recommend starting from similar forecasting systems (e.g., MAE < 2% = high) and refining post-launch.

9. **Cold-start marker color**: Grey for ungraded? Or a different color to distinguish "pending" vs "permanently ungraded"? Recommend grey (pending), with a tooltip explaining "will update when target matures" (§F6).

---

## 13. Self-Review Checklist (Post-GATE 0 Revision)

- ✅ **Alignment with locked decisions + god's rulings**: All four user-locked decisions (scope, full scorecard, ML-core frozen, model choice) carried in §2. God's GATE 0 rulings (365d lookback, pickle storage, per-asset models, grading source/cadence) baked into §2.5 and §10 milestones.
- ✅ **Jim's findings incorporated** (F1–F12):
  - F1 ✅: Forecast writer moved OUT of fragment → dedicated hourly APScheduler worker job (§5.1). Fragment read-only (§6.3). Dedup via INSERT OR IGNORE (§F1).
  - F2 ✅: New immutable table `intraday_bars_history` (365d ML store). Training + grading both read from it, never 7-day cache (§3.5, §F2).
  - F3 ✅: 365d training window (god ruling), not 90d (§3.2, §F3).
  - F4 ✅: HAR-RV fit on TRAIN split, not test. Metrics: MAE/RMSE/directional/CI-cover, not Sharpe (§6.2, §F4).
  - F5 ✅: Request math corrected (~351 requests/ticker for 365d, not bars; §9.1, §F5).
  - F6 ✅: Cold-start documented (panel "warming up" for ~1–2 weeks, §9.1, §F6).
  - F7 ✅: Milestone order fixed: M5 (evaluator) before M6 (markers). Real failure criteria in DoD column, not tautologies. Leakage canary in M3 DoD.
  - F8 ✅: Label construction fixed: anchor ONLY at closed bar boundaries, label = ln(P_{anchor+k} / P_anchor) (§3.2, §F8).
  - F9 ✅: Funding-rate as-of join (no forward-fill lookahead) documented (§4.1, §F9).
  - F10 ✅: Forecast anchor is closed bar, not live price. Disclaimer clarified (§7 intro, §F10).
  - F11 ✅: Contradiction resolved via intraday_bars_history (F2). LOOKBACK_DAYS=365 consistent (§9.1, §F11).
  - F12 ✅: Q1/Q2/Q7/Q9/Q10 closed (god's rulings). Q3-Q8 + Q12-13 remain genuinely deferred (§11, §F12).
- ✅ **Existing code seams referenced**: `intraday_bars` (7d display cache), `intraday_bars_history` (365d ML store), `live_quotes`, `crypto_derivatives`, `viz.py`, `schema.py`.
- ✅ **No ambiguity on key design forks**:
  - Daily ML core is FROZEN, zero cross-contamination.
  - Intraday is FULL SCORECARD with separate tables, modules, worker jobs.
  - Forecast writer = hourly worker job (sole DB writer, non-render path).
  - Fragment = read-only display ribbon (no DB writes per F1).
  - Training data = immutable `intraday_bars_history` (365d, separate from 7d display cache per F2).
  - Label construction = closed-bar anchors only, per F8.
  - Grading = hourly worker, reads from immutable history per F2, handles "not-yet-matured" gracefully per design.
- ✅ **Milestones realistic**: M0–M7 ordered, reordered per F7 (evaluator before markers). Testable failure criteria, not tautologies. M0 adds schema (F2). M3 includes leakage canary (F7). M4/M5 split: writer job / display (F1).
- ✅ **No half-finished designs**: Implementation path clear. No placeholder ambiguity except genuinely deferred open questions (§11).

---

## 14. Next Steps (Post-GATE 0 Revision)

**For re-review (god)**:
1. Verify all F1–F12 findings are addressed (see §13 self-review checklist).
2. Spot-check §2.5 (god's rulings) are baked in correctly.
3. Validate §10 milestones are now correctly ordered (M5 before M6 per F7) and have real failure criteria (not tautologies).
4. Confirm §11 open questions list is now correct (5 closed by god's ruling + 9 genuinely deferred).

**After re-GATE 0 approval**:
- Spin M0–M7 into detailed task-level implementation plan (PRD).
- Assign dev lead; estimate effort per milestone.
- Confirm Coinbase pagination cost (~351 requests/ticker × 2 = ~702 total for 365d; acceptable per F3).
- Begin implementation on `feat/realtime-v2`.

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
