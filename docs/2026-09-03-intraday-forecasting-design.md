# stock_forecasting — Intraday Forecasting (Crypto 1h/4h, Full Scorecard)

> **Status**: DESIGN — Phase 1, pre-implementation. Stops at GATE 0 (god + Jim review).  
> **Date**: 2026-09-03 · **Author**: Pam (pam-mtjr5nuk) · **Branch**: `feat/realtime-v2` (revised 2026-09-03 07:28)  
> **Scope**: Design only — no implementation, no schema migration, no model code, no milestone execution.  
> **Decision Authority**: User-locked decisions (§2) are mandatory; all other design choices are open to review.

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

**Input source**: Historical Coinbase candles (via existing `providers/coinbase.py` REST endpoint).
- **Granularity**: 1-minute and/or 5-minute bars, depending on feature needs and chart bucket size.
- **Lookback window**: 90 calendar days of historical data (covers ~2,160 one-hour bars and ~540 four-hour bars).
  - **OPEN QUESTION**: Should training data extend further (e.g., 365 days) if Coinbase pagination cost is acceptable? Pros: more robust seasonal patterns, better volatility-regime diversity. Cons: DB storage, longer training time. Recommend 90d for MVP; revisit if models underfit.

**Feature computation** (§4 below):
1. Build OHLCV-derived technical features (VWAP, realized vol, EWMA spreads).
2. Normalize with StandardScaler (fit on training window, apply to validation).
3. Align with dYdX funding-rate series (hourly snapshots, zero-order hold for 5m bars).
4. Merge into a single DataFrame for each (BTC-USD, ETH-USD) pair.

**Label construction** (k-step forward log-return):
- 1-hour horizon: $r_{t, t+1\text{h}}$
- 4-hour horizon: $r_{t, t+4\text{h}}$
- Constructed from the **close** of the next complete hour/4-hour candle.

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

### 3.5 New table — `intraday_prediction_snapshots`

Mirrors the structure of `prediction_snapshots` (daily) but for intraday horizons. Records every forecast the instant it is made.

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `ticker` | TEXT FK → `tickers.symbol` | |
| `horizon` | TEXT | `1h` \| `4h` |
| `made_at` | TEXT | wall-clock UTC when forecast was generated |
| `anchor_ts` | TEXT | timestamp of the bar used to compute features |
| `anchor_price` | REAL | close price at anchor time (reference point) |
| `predicted_return` | REAL | model's log-return forecast (ŷ) |
| `predicted_price` | REAL | anchor_price × (1 + predicted_return) [denormalized for display] |
| `ci_lower_return` | REAL | HAR-RV lower bound (log-return space) |
| `ci_upper_return` | REAL | HAR-RV upper bound (log-return space) |
| `ci_lower_price` | REAL | denormalized to price space |
| `ci_upper_price` | REAL | denormalized to price space |
| `target_ts` | TEXT | when the prediction target matures (anchor_ts + horizon) |
| `model_version` | TEXT | version string of model used |
| `model_sha` | TEXT | git SHA of model training code |

**Index**: unique on (ticker, horizon, anchor_ts) to prevent duplicate forecasts.  
**Retention**: Keep intraday predictions indefinitely (full audit trail); no pruning.
**Grading reference**: When `target_ts` passes, `intraday_evaluator` looks up this row and computes realized return from `intraday_bars`.

### 3.6 New table — `intraday_accuracy_records`

Records grading results once a forecast's target matures. Same shape as `accuracy_records` (daily) but intraday-specific.

| col | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `prediction_id` | INTEGER FK → `intraday_prediction_snapshots.id` | links back to the forecast |
| `ticker` | TEXT FK → `tickers.symbol` | |
| `horizon` | TEXT | `1h` \| `4h` |
| `graded_at` | TEXT | wall-clock when grade was computed |
| `realized_return` | REAL | actual log-return from anchor to target close |
| `realized_price` | REAL | actual price at target_ts |
| `signed_error` | REAL | realized_return - predicted_return |
| `abs_error_pct` | REAL | |abs(signed_error)| × 100 (as % of predicted move) |
| `direction_hit` | INTEGER | 1 if sign(predicted) == sign(realized), else 0 |
| `ci_cover` | INTEGER | 1 if realized_return ∈ [ci_lower_return, ci_upper_return], else 0 |
| `grading_attempts` | INTEGER | counter: incremented each time evaluator checks this prediction (0 at creation, → 1 when graded). Prevents false failures if bar not yet available. |

**Index**: foreign key on prediction_id.  
**Grading logic**: When a forecast's target_ts arrives, evaluator queries `intraday_bars` for the realized bar. If bar is not yet closed (still provisional or missing), increment `grading_attempts` and defer. When bar is finalized (is_provisional=0), compute results and write.

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
6. **dYdX Funding-Rate Z-Score** (24h window):
   - $f = (\text{FundingRate}_t - \text{mean}_{[t-24\text{h}, t]}) / \text{std}_{[t-24\text{h}, t]}$
   - Extreme funding rates predict volatility expansion or liquidation cascades.
   - Sourced from `crypto_derivatives` table (hourly snapshots).
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

## 5. Intraday Evaluator & Grading Job

### 5.1 Grading workflow (separate worker job)

**When**: Every hour, after the top-of-hour bar closes (intraday_bars becomes is_provisional=0).

**What it does**:
1. Query `intraday_prediction_snapshots` for all forecasts where `target_ts ≤ now` and no corresponding `intraday_accuracy_records.id` exists (ungraded).
2. For each ungraded forecast:
   a. Look up the realized bar in `intraday_bars` for (ticker, interval, target_ts).
   b. **If bar is not found or still provisional** (is_provisional=1):
      - Increment `grading_attempts` counter (if a row exists in intraday_accuracy_records for this prediction_id; if not, create one with grading_attempts=1, all other fields NULL).
      - Skip to next forecast (will retry on the next hourly job run).
   c. **If bar is closed** (is_provisional=0):
      - Compute realized_return from anchor_price to target bar close.
      - Compute signed_error, abs_error_pct, direction_hit, ci_cover.
      - Write record to intraday_accuracy_records (or update if grading_attempts row exists, filling in the computed fields).
3. Exit.

**Rationale**: Handling "bar not ready yet" gracefully (via grading_attempts counter) prevents false negatives and avoids re-grading the same forecast multiple times. Mirrors the approach discussed in daily evaluator (Issue 7 context).

### 5.2 Intraday forecaster & predictions writer (post-inference)

**When**: Each time the fragment renders (every 2s for crypto, §6.2 below).

**What it does**:
1. Load intraday_forecaster model.
2. Compute features from the current bar + live quote.
3. Call model.predict() → predicted_return.
4. Call intraday_volatility.predict() → CI bounds.
5. **Write to `intraday_prediction_snapshots`**:
   - Anchor: the current bar's ts and close price.
   - Target: ts + 1h (or 4h for 4-hour model).
   - Predicted return, CI, model version/SHA.
   - made_at = now.
6. Return (predicted_price, ci_lower, ci_upper) to the display layer (§6.2).

**Duplicate prevention**: Before writing, check if a row already exists for (ticker, horizon, anchor_ts). If yes, skip the write (don't re-record the same forecast). This prevents duplicate rows if the same bar is re-rendered.

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

**Pseudocode**:
```
1. Fetch latest 90 days of BTC-USD + ETH-USD intraday bars (1m or 5m).
2. Fetch latest 90 days of dYdX funding-rate snapshots.
3. Align + resample to common 5m grid; forward-fill missing funding rates.
4. Compute features (§4.1).
5. Construct k-step forward log-return labels (k=1h, k=4h).
6. Split into train/test using Purged & Embargoed TimeSeriesSplit:
   a. Test window: last 14 days (2 weeks; ~336 5m bars per horizon).
   b. Purge: drop all training samples whose label window overlaps [test_start, test_end].
   c. Embargo: drop training samples in [test_end, test_end + 24h].
   d. Train on remaining samples.
7. For each (ticker, horizon):
   a. Fit LightGBM on train; validate on (purged + embargoed) test.
   b. Record validation Sharpe, Calmar, directional accuracy on test.
   c. Fit Ridge as fallback.
   d. Save both models to disk (pickle, or DB depending on §3.4 decision).
8. Fit HAR-RV volatility model on same test window; save.
9. Record metadata (train_start, train_end, validation_sharpe, code_sha) for audit.
10. Exit; next day, repeat.
```

### 6.3 Inference (display-time, low-latency)

**When the chart renders** (every 2 seconds for crypto, per v2 design):
1. Load the latest intraday_forecaster model (checkpoint from nightly train).
2. Fetch the most recent intraday bars + live quote for BTC-USD / ETH-USD.
3. Compute features for the **current bar** (the forming one).
4. Call `intraday_forecaster.predict()` → point forecast (log-return).
5. Call `intraday_volatility.predict()` → confidence band width.
6. Transform to price-space: predicted_return × last_close + last_close, ± bands.
7. **Write to `intraday_prediction_snapshots`**: Record the forecast (anchor_ts, anchor_price, predicted_return, CI bounds, target_ts, model_version/SHA, made_at=now).
   - Check for duplicate (same ticker, horizon, anchor_ts); skip write if already exists.
8. Return the ribbon series for the next 1h / 4h (visual projection to the display layer).

**Latency budget**: < 10 ms for inference; write to DB is non-blocking (can be async or on a separate thread).

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

### 7.5 EOD reconciliation & ribbon reset

At session close (crypto 00:00 UTC):
1. The forming intraday bar closes; `is_provisional` flips to 0.
2. The nightly intraday trainer **runs in parallel** with the daily trainer (no sequential dependency).
3. Fragment renders:
   - New daily ribbon from the updated `ohlcv_bars.close` (unchanged from v2).
   - New intraday ribbons (1h, 4h) from the retrained models, anchored to the new live price (which ≈ P_close at 00:00 UTC).
4. **No visible jump** because the intraday forecast is updated hourly (not tied to the close); the 00:00 UTC ribbon is just the first forecast of the new session.

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

> *"Intraday forecasts (1-hour and 4-hour horizons) are short-term technical and derivatives-based predictions, independent from the daily ML evaluation. Intraday forecasts have their own accuracy scorecard (historical forecast markers + accuracy panel), separate from daily predictions. Confidence bands are empirical (HAR-RV realized volatility). Use intraday forecasts for tactical context; rely on the daily forecast ribbon for strategic directional signals."*

---

## 9. Configuration & Environment

### 9.1 New environment variables

```
# Intraday forecasting feature flags
INTRADAY_FORECAST_ENABLED=true
INTRADAY_FORECAST_HORIZONS=1h,4h             # comma-separated; subset of {1h, 4h}
INTRADAY_LOOKBACK_DAYS=90                    # training data window
INTRADAY_RETRAIN_SCHEDULE=nightly             # "nightly" | "none" (disabled)

# Model hyperparameters (to be tuned)
INTRADAY_LIGHTGBM_DEPTH=4
INTRADAY_LIGHTGBM_LR=0.02
INTRADAY_LIGHTGBM_ROUNDS=300

# Display
INTRADAY_FORECAST_OPACITY=0.6
INTRADAY_FORECAST_COLOR_1H=#FF8C00            # orange
INTRADAY_FORECAST_COLOR_4H=#9932CC            # purple
INTRADAY_CI_LEVEL=0.95                        # 95% bands
```

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

**Dependency order**: M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7.

| M | Deliverable | Definition of Done | Fails On |
|---|---|---|---|
| **M0** | Schema + config + prediction/accuracy tables | `INTRADAY_FORECAST_*` settings in `config.py`; `.env.example` updated. `intraday_prediction_snapshots` and `intraday_accuracy_records` tables created in `schema.py`. Test: tables exist after `create_tables`, settings parse, model directory exists. | Old code without intraday config/schema. |
| **M1** | Intraday data pipeline | `intraday_trainer.py` fetches 90d Coinbase bars + dYdX funding. Aligns to 5m grid; no lookahead in feature computation. Tests: mock Coinbase REST → DataFrame shape correct, funding aligned by timestamp, feature matrix non-null. | Old code (module absent). |
| **M2** | Feature engineering | `intraday_features.py` computes §4.1 suite (VWAP distance, vol ratios, EWMA spreads, dYdX z-score, temporal). StandardScaler fit on train window. Tests: feature matrix shape matches expected, no NaNs after warmup, scaling applied. | Missing module. |
| **M3** | Model training (LightGBM + fallback) | `intraday_trainer.py` trains LightGBM + Ridge on Purged & Embargoed TimeSeriesSplit. HAR-RV volatility model. Saves checkpoints to disk. Tests: Purged & Embargoed split logic (no label-leakage), model saves, metadata recorded, validation metrics computed. | Missing training logic. |
| **M4** | Inference pipeline + prediction writing | `intraday_forecaster.py` loads model, computes live forecast in <10ms, writes to `intraday_prediction_snapshots` (with duplicate prevention). Tests: mock live bar → forecast shape correct, DB write succeeds, duplicate check works, bands positive. | Missing inference or write logic. |
| **M5** | Display integration (chart + markers) | `viz.py` renders intraday ribbons (1h, 4h) on the chart. Fragment reads models + live data + historical markers from `intraday_accuracy_records`. EOD reconciliation updates daily ribbon. Tests: render without crash, ribbon points align with prices, bands monotonic, markers render (green/red/grey). Regression: daily ribbon unmoved by intraday logic. | Missing display logic. |
| **M6** | Intraday evaluator job | `intraday_evaluator.py` grades forecasts nightly/hourly when targets mature. Queries `intraday_bars` for realized bar; if not closed, increments `grading_attempts` and defers; if closed, computes error/direction/CI and writes to `intraday_accuracy_records`. Tests: mock mature bar → grading succeeds, ungraded bar → counter increments, all fields populated correctly. | Missing evaluator logic. |
| **M7** | Intraday accuracy panel | New accuracy panel in `panels.py` (intraday_accuracy_rows + verdict_sentence). Per-horizon rollup (MAE%, direction%, CI cover%, n, trust). Separate from daily accuracy panel. UI placement confirmed. Tests: panel renders without crash, metrics computed correctly, verdict logic applied. | Missing panel or UI logic. |

---

## 11. Open Questions (Non-Blocking for Design)

1. **Training data window**: Should lookback extend beyond 90 days if Coinbase pagination cost is acceptable? (Pro: volatility-regime diversity; con: DB storage, train time.) Recommend 90d MVP, revisit post-launch.

2. **Model checkpoint storage**: Pickle files in `stock_forecasting/models/intraday/`, or persist in a new `intraday_model_runs` table? (Pro pickle: simplicity, versioning via file mtime; pro table: audit trail, easy queries.) Recommend pickle for MVP.

3. **Feature scaler persistence**: Refit nightly or store and reuse? (Pro refit: intraday vol shifts daily, adapt; pro store: stable inference.) Recommend nightly refit.

4. **Hyperparameter tuning**: Fixed empirical values or Bayesian optimization on each retrain? (Pro fixed: fast, reproducible; pro Bayesian: adapts to market regime.) Recommend fixed for MVP; enable tuning in future milestone.

5. **Confidence level**: Fixed 95% (match daily) or user-tunable? Recommend fixed 95% for consistency.

6. **Fallback to Ridge**: When should Ridge be used instead of LightGBM? (Always as backup? Only if LGB training fails? Only if perf degrades?) Recommend always training both; display uses LGB if available, Ridge as fallback.

7. **Per-asset vs. cross-asset model**: Separate model per (ticker, horizon) or one cross-asset model? (Pro separate: ticker-specific regime; pro cross: more data, shared patterns.) Recommend separate (BTC 1h, BTC 4h, ETH 1h, ETH 4h = 4 models) for initial scope.

8. **Embargo window size**: 24 hours chosen conservatively; could be data-driven (e.g., autocorrelation decay). Recommend 24h fixed for MVP.

9. **Intraday realized-price source for grading**: Should grading use the bar's close price from `intraday_bars`, or should it aggregate from higher-frequency ticks (e.g., best-bid/ask from WS)? (Pro close: simplicity, matches chart; pro ticks: more granular ground truth.) Recommend bar close for MVP, consistent with prediction anchor.

10. **Grading job frequency**: Should evaluator run hourly (every hour post-bar-close) or nightly (post-session)? (Pro hourly: live accuracy feedback; con hourly: more DB load, more complex scheduling.) Recommend hourly for MVP (short-horizon forecasts mature frequently), with nightly batching as fallback.

11. **Historical marker retention & archival**: Keep markers for 7 days, then archive? Or infinite retention? (Pro 7d: faster chart renders, cleaner UI; pro infinite: audit trail, offline replay.) Recommend 7d retention, with optional archival table for deep analysis (TBD post-MVP).

12. **Intraday accuracy panel UI placement**: New tab, collapsible section, or inline legend? (Pro tab: clear separation; pro inline: immediate visibility.) Recommend tab or section for clear separation from daily accuracy; confirm in implementation UI review.

13. **Trust verdict thresholds**: What MAE %, direction %, CI cover % levels = "high trust" vs. "moderate" vs. "low"? (This is tuning, not design, but impacts label calibration.) Recommend starting with thresholds from similar short-term forecasting systems (e.g., MAE < 2% = high, < 3% = moderate) and refining post-launch.

---

## 12. Self-Review Checklist

- ✅ **Alignment with locked decisions**: All four user-locked decisions (scope, full scorecard, ML-core frozen, model choice) carried verbatim in §2.
- ✅ **Kevin's scout incorporated**: Data depth analysis (§0, §3.2), model survey (§6.1), risks & mitigation (§4, label-leakage via Purged & Embargoed), EOD reconciliation (§7.5).
- ✅ **Existing code seams referenced**: `intraday_bars`, `live_quotes`, `crypto_derivatives` (§3.2), `viz.py` (§7.1), `schema.py` (§3.4/3.5/3.6).
- ✅ **Explicit open questions**: 13 questions flagged in §11 with rationale; none block design → implementation handoff.
- ✅ **No ambiguity on key design forks**:
  - Daily ML core is FROZEN (§8.1, integrity guarantee).
  - Intraday is FULL SCORECARD (persistence, grading, accuracy panel, separate tables).
  - Model is LightGBM + Ridge, not deep learning.
  - Retrain is nightly, not streaming or per-tick.
  - Confidence bands use HAR-RV, not daily-model error rescaling.
  - Grading is per-horizon, with graceful handling of "not-yet-mature" forecasts via grading_attempts counter.
- ✅ **Scope clear**: Crypto only (BTC, ETH), 1h / 4h horizons, no equities, no sub-hour, full persistence/grading, no real-time retraining.
- ✅ **Milestones realistic**: M0–M7 ordered, each has testable DoD, no circular deps. M0 adds schema; M6-M7 add evaluator and accuracy panel.
- ✅ **No half-finished designs**: Placeholder text appears only for open questions; implementation path is clear. Evaluator logic (§5.1) handles edge cases (grading_attempts, deferred grading).

---

## 13. Next Steps

**For GATE 0 (god + Jim review)**:
1. Read §1–§8 for scope, architecture, locked decisions, display integration, ML-core separation.
2. Read §5–§7 for new scorecard components (prediction table, evaluator job, accuracy panel, historical markers).
3. Spot-check §4 feature suite against Kevin's scout (§3.2).
4. Validate §10 milestones (M0–M7) against estimated dev effort; confirm M6–M7 are feasible for MVP.
5. Approve open questions in §11 or offer guidance (especially §11.9–11.13 for scorecard specifics).

**After GATE 0**:
- Spin M0–M7 into a detailed task-level implementation plan (PRD).
- Assign dev lead; estimate effort per milestone. (Scorecard adds 2 milestones; budget ~30% more total effort than display-only design.)
- Confirm Coinbase pagination costs (90d × 2 tickers × 5m bars = ~26k requests; cost TBD).
- Confirm evaluator job scheduling (hourly vs. nightly).
- Begin implementation on `feat/realtime-v2`.

---

## 14. References

- **Kevin's scout**: `docs/reports/2026-09-03-intraday-forecasting-scout.md` (data depth, feasibility, risks, methodological guardrails).
- **v2 design template**: `docs/2026-09-01-realtime-v2-design.md` (display-layer architecture, health rework, schema, milestones structure).
- **v1 ML spec**: `docs/2026-09-01-stock-forecasting-design.md` (unchanged daily core, ledger, evaluator, accuracy — referenced but not quoted in this doc).
- **Existing intraday seams**: `stock_forecasting/intraday_store.py`, `stock_forecasting/schema.py` (IntradayBar, LiveQuote), `stock_forecasting/providers/dydx.py`, `stock_forecasting/viz.py`, `stock_forecasting/worker.py`.
