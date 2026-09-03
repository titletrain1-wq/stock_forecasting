# stock_forecasting — Intraday Forecasting (Crypto 1h/4h, Display-Only)

> **Status**: DESIGN — Phase 1, pre-implementation. Stops at GATE 0 (god + Jim review).  
> **Date**: 2026-09-03 · **Author**: Pam (pam-mtjr5nuk) · **Branch**: `feat/realtime-v2`  
> **Scope**: Design only — no implementation, no schema migration, no model code, no milestone execution.  
> **Decision Authority**: User-locked decisions (§1.5) are mandatory; all other design choices are open to review.

---

## 0. Executive Summary & Scope

This design introduces **intraday forecasting models** to `stock_forecasting` alongside the v2.0.0 display layer and the frozen daily ML core. The intraday layer is **display-only**, **crypto-exclusive** (BTC-USD, ETH-USD), and scoped to **1-hour and 4-hour prediction horizons**.

### Why intraday now?
- v2.0.0 delivers real-time crypto price display via Coinbase WebSocket.
- Intraday bars are already flowing into `intraday_bars` (100-row cache, display-only).
- dYdX hourly funding rate is already integrated as a feature; it aligns perfectly with 1-hour model cadence.
- A separate short-horizon intraday model unlocks actionable directional/volatility signals without compromising the frozen, proven daily ML core.

### Key constraints
1. **Crypto-only**: yfinance's 15-minute delay and 60-day cap make equity sub-hour forecasting fundamentally broken (Kevin's scout, §2.2).
2. **Display-only**: No persistence to the ledger, no grading, no intraday predictions table, no accuracy evaluator/panel this milestone.
3. **ML-core frozen**: The daily forecaster, trainer, features, evaluator, accuracy pipeline remain untouched. Intraday lives in its own module(s).
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
- **Display**: Overlaid on the existing live price chart as an intraday forecast ribbon (α-blended visual layer, no persistence).
- **Retrain cadence**: Nightly, once per day post-close (same cadence as daily model).

### 1.2 Explicitly OUT-of-scope this milestone

- **Equities**: Not covered due to the 15-minute delay and 60-calendar-day lookback ceiling (Kevin's scout, §2.1, §4.2).
- **Sub-1-hour horizons** (1m / 5m next-bar): Microstructure noise dominates; SNR too low (Kevin's scout, §2.1).
- **Persistence to the ledger**: No `intraday_predictions` table, no grading, no scorecard, no backward-in-time accuracy panel.
- **Real-time model retraining**: Intraday model is trained once per day; not adapted intraday based on live ticks.
- **Intraday position-sizing / trade signals / alerts**: Display-only, no prescriptive advice.
- **Multi-user state / persistence of intraday forecasts across sessions**: Forecasts live only in the display fragment; no archival this milestone.
- **Deep learning (LSTM/Transformers/PatchTST)**: Ruled out by data starvation and overfitting risk on 60-day windows (Kevin's scout, §3.4).

---

## 2. Locked Decisions (from user)

These decisions are immutable for this design and must be carried verbatim into implementation:

1. **SCOPE**: Crypto-only, BTC-USD + ETH-USD. Horizons 1h and 4h. Equities explicitly OUT.
2. **DISPLAY-ONLY**: The intraday forecast is drawn on the chart only. NO persistence, NO grading, NO intraday predictions table / evaluator / accuracy panel this milestone. A scorecard is a deferred later milestone.
3. **ML-core frozen**: The daily core (forecaster.py, trainer.py, features.py, evaluator.py, accuracy.py) stays FROZEN and display-separated — identical rule to the v2 live-price line. Intraday model lives in its own module(s).
4. **MODEL**: LightGBM (Ridge fallback) on rolling tabular features (intraday VWAP distance, realized-vol ratios, EWMA return spreads, dYdX funding-rate z-scores). HAR-RV for the CI band. Purged + Embargoed TimeSeriesSplit for overlapping k-step-forward-return label leakage. CPU, local, retrain cadence to be proposed in the doc.

---

## 3. Architecture & Data Flow

### 3.1 Intraday forecasting as a separate ML subsystem

```
Daily ML Core (FROZEN)               Intraday ML Subsystem (NEW, Display-Only)
├─ forecaster.py                    ├─ intraday_forecaster.py (new)
├─ trainer.py                       ├─ intraday_trainer.py (new)
├─ features.py                      ├─ intraday_features.py (new, or inline)
├─ evaluator.py                     └─ intraday_store_models.py (if persisting model files)
└─ accuracy.py

Both read from:                     Intraday reads only from:
└─ ohlcv_bars (1d, immutable)      ├─ intraday_bars (1m/5m candles, 7-day cache)
                                    ├─ live_quotes (current tick)
                                    └─ crypto_derivatives (dYdX funding rate)
                                    
                                    Writes to:
                                    └─ (display fragment only; no DB ledger)
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
  - **OPEN QUESTION**: Should we persist intraday model snapshots for later audit / comparison? If yes, schema needed in §2; if no, local `.pkl` suffices. Recommend Option A (files) for MVP.

**No ledger table**: No `intraday_predictions` table. Forecasts are computed at render time by the display fragment (§5.1).

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

## 5. Model Training & Inference

### 5.1 Model architecture (per locked decision §2)

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

### 5.2 Training loop (nightly, post-close)

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

### 5.3 Inference (display-time, low-latency)

**When the chart renders** (every 2 seconds for crypto, per v2 design):
1. Load the latest intraday_forecaster model (checkpoint from nightly train).
2. Fetch the most recent intraday bars + live quote for BTC-USD / ETH-USD.
3. Compute features for the **current bar** (the forming one).
4. Call `intraday_forecaster.predict()` → point forecast (log-return).
5. Call `intraday_volatility.predict()` → confidence band width.
6. Transform to price-space: predicted_return × last_close + last_close, ± bands.
7. Return the ribbon series for the next 1h / 4h (visual projection).

**Latency budget**: < 10 ms (already on the fragment's 2s poll cycle).

---

## 6. Display Integration

### 6.1 Chart rendering (existing `viz.py` + new layer)

The intraday forecast is rendered as an additional series on the existing live price chart:

- **Series**: `intraday_forecast_1h` and `intraday_forecast_4h` (one per horizon).
- **Style**: 
  - Line: distinct color (e.g., orange for 1h, purple for 4h), light α (0.6).
  - Bands: filled area between `lower_bound` and `upper_bound` at the same α (visual uncertainty ribbon).
  - Hover text: "1h intraday forecast | ±95% CI | HAR-RV realized-vol bands".
- **Anchor**: All ribbons start from the current live price (`live_quotes.price`), not from `ohlcv_bars.close` (unlike the daily ribbon, which anchors to P_close).
  - **Rationale**: Intraday is a separate short-horizon model; anchoring to the live price allows the user to see the model's next-1h expectation relative to "now".

### 6.2 Toggle & config

- **Display flag** in `.env`: `INTRADAY_FORECAST_DISPLAY=true` (default: true if LightGBM models are found on disk).
- **Per-asset toggle** in UI (optional for MVP): Show/hide `intraday_forecast_1h` and `intraday_forecast_4h` independently.
- **OPEN QUESTION**: Should the user be able to choose confidence level (e.g., 68%, 95%)? Recommend fixed 95% for consistency with daily bands; make tunable in future.

### 6.3 EOD reconciliation & ribbon reset

At session close (crypto 00:00 UTC):
1. The forming intraday bar closes; `is_provisional` flips to 0.
2. The nightly intraday trainer **runs in parallel** with the daily trainer (no sequential dependency).
3. Fragment renders:
   - New daily ribbon from the updated `ohlcv_bars.close` (unchanged from v2).
   - New intraday ribbons (1h, 4h) from the retrained models, anchored to the new live price (which ≈ P_close at 00:00 UTC).
4. **No visible jump** because the intraday forecast is updated hourly (not tied to the close); the 00:00 UTC ribbon is just the first forecast of the new session.

---

## 7. ML-Core Separation & Integrity Guarantee

### 7.1 Mandatory constraints

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

### 7.2 Documentation placeholder

**Disclaimer** (to appear in chart caption + `KNOWN_LIMITATIONS.md`):

> *"Intraday forecasts (1-hour and 4-hour horizons) are display-only projections based on short-term technical and derivatives features. They are **not** graded, **not** persisted to the ledger, and **not** part of the daily ML evaluation. Use for visual context only; rely on the daily forecast ribbon for directional signals. Confidence bands are empirical (HAR-RV realized volatility), not calibrated to walk-forward accuracy like the daily bands."*

---

## 8. Configuration & Environment

### 8.1 New environment variables

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

### 8.2 Model storage

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

## 9. Implementation Plan (Milestones + DoD)

**Dependency order**: M0 → M1 → M2 → M3 → M4 → M5.

| M | Deliverable | Definition of Done | Fails On |
|---|---|---|---|
| **M0** | Schema + config | `INTRADAY_FORECAST_*` settings in `config.py`; `.env.example` updated. Test: settings parse, model directory exists. | Old code without intraday config. |
| **M1** | Intraday data pipeline | `intraday_trainer.py` fetches 90d Coinbase bars + dYdX funding. Aligns to 5m grid; no lookahead in feature computation. Tests: mock Coinbase REST → DataFrame shape correct, funding aligned by timestamp, feature matrix non-null. | Old code (module absent). |
| **M2** | Feature engineering | `intraday_features.py` computes §4.1 suite (VWAP distance, vol ratios, EWMA spreads, dYdX z-score, temporal). StandardScaler fit on train window. Tests: feature matrix shape matches expected, no NaNs after warmup, scaling applied. | Missing module. |
| **M3** | Model training (LightGBM + fallback) | `intraday_trainer.py` trains LightGBM + Ridge on Purged & Embargoed TimeSeriesSplit. HAR-RV volatility model. Saves checkpoints to disk. Tests: Purged & Embargoed split logic (no label-leakage), model saves, metadata recorded, validation metrics computed. | Missing training logic. |
| **M4** | Inference pipeline | `intraday_forecaster.py` loads model, computes live forecast in <10ms. Tests: mock live bar → forecast shape correct, confidence band width positive, handles missing funding gracefully. | Missing inference module. |
| **M5** | Display integration | `viz.py` renders intraday ribbons (1h, 4h) on the chart. Fragment reads models + live data. EOD reconciliation swaps daily ribbon, preserves intraday ribbon transparency. Tests: render without crash, ribbon points align with prices, bands are monotonic upper ≥ point ≥ lower. Regression: daily ribbon unmoved by intraday logic. | Missing display logic. |

---

## 10. Open Questions (Non-Blocking for Design)

1. **Training data window**: Should lookback extend beyond 90 days if Coinbase pagination cost is acceptable? (Pro: volatility-regime diversity; con: DB storage, train time.) Recommend 90d MVP, revisit post-launch.

2. **Model checkpoint storage**: Pickle files in `stock_forecasting/models/intraday/`, or persist in a new `intraday_model_runs` table? (Pro pickle: simplicity, versioning via file mtime; pro table: audit trail, easy queries.) Recommend pickle for MVP.

3. **Feature scaler persistence**: Refit nightly or store and reuse? (Pro refit: intraday vol shifts daily, adapt; pro store: stable inference.) Recommend nightly refit.

4. **Hyperparameter tuning**: Fixed empirical values or Bayesian optimization on each retrain? (Pro fixed: fast, reproducible; pro Bayesian: adapts to market regime.) Recommend fixed for MVP; enable tuning in future milestone.

5. **Confidence level**: Fixed 95% (match daily) or user-tunable? Recommend fixed 95% for consistency.

6. **Fallback to Ridge**: When should Ridge be used instead of LightGBM? (Always as backup? Only if LGB training fails? Only if perf degrades?) Recommend always training both; display uses LGB if available, Ridge as fallback.

7. **Per-asset vs. cross-asset model**: Separate model per (ticker, horizon) or one cross-asset model? (Pro separate: ticker-specific regime; pro cross: more data, shared patterns.) Recommend separate (BTC 1h, BTC 4h, ETH 1h, ETH 4h = 4 models) for initial scope.

8. **Embargo window size**: 24 hours chosen conservatively; could be data-driven (e.g., autocorrelation decay). Recommend 24h fixed for MVP.

---

## 11. Self-Review Checklist

- ✅ **Alignment with locked decisions**: All four user-locked decisions (scope, display-only, ML-core frozen, model choice) carried verbatim in §2.
- ✅ **Kevin's scout incorporated**: Data depth analysis (§2, §3.2), model survey (§5.1), risks & mitigation (§4.5, label-leakage via Purged & Embargoed), EOD reconciliation (§6.3).
- ✅ **Existing code seams referenced**: `intraday_bars`, `live_quotes`, `crypto_derivatives` (§3.2), `viz.py` (§6.1), `schema.py` (§3.4).
- ✅ **Explicit open questions**: Eight questions flagged in §10 with rationale; none block design → implementation handoff.
- ✅ **No ambiguity on key design forks**:
  - Daily ML core is FROZEN (§7.1, integrity guarantee).
  - Intraday is display-only (no ledger table, no grading).
  - Model is LightGBM + Ridge, not deep learning.
  - Retrain is nightly, not streaming or per-tick.
  - Confidence bands use HAR-RV, not daily-model error rescaling.
- ✅ **Scope clear**: Crypto only (BTC, ETH), 1h / 4h horizons, no equities, no sub-hour, no persistence, no real-time retraining.
- ✅ **Milestones realistic**: M0–M5 ordered, each has testable DoD, no circular deps.
- ✅ **No half-finished designs**: Placeholder text appears only for open questions; implementation path is clear.

---

## 12. Next Steps

**For GATE 0 (god + Jim review)**:
1. Read §1–§7 for scope, architecture, locked decisions, display integration.
2. Spot-check §4 feature suite against Kevin's scout (§3.2).
3. Validate §9 milestones against estimated dev effort.
4. Approve open questions in §10 or offer guidance.

**After GATE 0**:
- Spin M0–M5 into a detailed task-level implementation plan (PRD).
- Assign dev lead; estimate effort per milestone.
- Confirm Coinbase pagination costs (90d × 2 tickers × 5m bars = ~26k requests; cost TBD).
- Begin implementation on `feat/realtime-v2`.

---

## 13. References

- **Kevin's scout**: `docs/reports/2026-09-03-intraday-forecasting-scout.md` (data depth, feasibility, risks, methodological guardrails).
- **v2 design template**: `docs/2026-09-01-realtime-v2-design.md` (display-layer architecture, health rework, schema, milestones structure).
- **v1 ML spec**: `docs/2026-09-01-stock-forecasting-design.md` (unchanged daily core, ledger, evaluator, accuracy — referenced but not quoted in this doc).
- **Existing intraday seams**: `stock_forecasting/intraday_store.py`, `stock_forecasting/schema.py` (IntradayBar, LiveQuote), `stock_forecasting/providers/dydx.py`, `stock_forecasting/viz.py`, `stock_forecasting/worker.py`.
