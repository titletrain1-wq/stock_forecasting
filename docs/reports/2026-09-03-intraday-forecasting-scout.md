# Intraday Forecasting Scout: Data Depth, Plausible Horizons, Model Survey & Risks

**Date:** 2026-09-03  
**Author:** Kevin (Hive Agent `kevin-mtjr8a6q`)  
**Task ID:** T-013 (`stockfc-intraday-t013`)  
**Target Repository:** `stock_forecasting`  
**Branch:** `feat/realtime-v2`  
**Scope:** Research & findings scout only. Zero implementation / zero code changes.

---

## Executive Summary

This report evaluates the feasibility of introducing short-horizon intraday forecasting models to `stock_forecasting` alongside the existing v2.0.0 real-time display layer and the frozen daily ML core. 

### Key Takeaways:
1. **Asymmetric Data Realities:** Keyless Coinbase REST/WS provides unlimited historical crypto candle depth (paginated) with 0-second latency, while free `yfinance` equity data is hard-capped at a **60-calendar-day lookback window for 5m bars** and incurs an unavoidable **~15-minute delay**.
2. **Horizon Feasibility Divide:**
   - **Crypto:** **1-hour** and **4-hour** horizons are statistically viable and practically meaningful (aligning with dYdX hourly funding cycles and multi-hour volatility clusters). Sub-5-minute horizons are dominated by microstructure noise.
   - **Equities:** Sub-hour intraday forecasting on free feeds is **fundamentally broken** due to the 15-minute delivery delay (predicting $t+5\text{m}$ using $t-15\text{m}$ data is a 20-minute lookahead gap) and the tiny sample size (~3,276 5-minute bars over 60 days).
3. **Recommended Model Class:** **LightGBM / Ridge Regressors** on tabular rolling features (intraday VWAP deviations, EWMA return spreads, realized volatility, dYdX hourly funding rate z-scores). Deep sequential neural networks (LSTM/TCN) and Transformers (PatchTST/Chronos) are severe overkill, data-starved on 60-day windows, and introduce unacceptable dependency and compute bloat for a single-user local SQLite application.
4. **Primary Methodological Guardrail:** Overlapping intraday return labels create extreme serial correlation and false walk-forward validation optimism unless **Purged & Embargoed Time-Series Cross-Validation** (López de Prado) is strictly enforced.
5. **Verdict:** **Conditional GO for Crypto (1h and 4h horizons)**; **NO-GO for Free Equities (sub-1h horizons)** without a real-time, paid data provider.

---

## 1. Intraday Data Depth & Quality Per Provider

Real numbers and operational parameters obtained from provider API specifications and the `stock_forecasting/providers/` codebase.

```
+----------------------------------------------------------------------------------------------------+
|                                    INTRADAY DATA PROVIDER PROFILES                                  |
+--------------------+---------------------+--------------------+--------------------+---------------+
| Provider           | Granularities       | Retention / Depth  | Latency / Delay    | Key Limits    |
+--------------------+---------------------+--------------------+--------------------+---------------+
| Coinbase REST      | 60s (1m), 300s (5m),| Product inception  | ~0s (live candle   | 300 candles/  |
| (Exchange / Adv)   | 900s, 3600s (1h),   | (~2015 for BTC;    | head is            | req; ~10-30   |
|                    | 21600s (6h), 86400s | multi-year history)| provisional)       | req/s public  |
+--------------------+---------------------+--------------------+--------------------+---------------+
| Coinbase WS        | Tick-by-tick batch  | Live stream only   | Real-time (<5s     | 100 subs/IP;  |
| (Advanced Trade)   | (~5s ticks)         | (in-memory buffer) | tick batches)      | keyless public|
+--------------------+---------------------+--------------------+--------------------+---------------+
| yfinance           | 1m                  | 7 calendar days    | ~15-min delay      | Unofficial    |
| (Yahoo Finance)    | 2m, 5m, 15m, 30m    | 60 calendar days   | during US hours    | scraper; 429  |
|                    | 60m / 1h            | 730 calendar days  |                    | throttle risk |
+--------------------+---------------------+--------------------+--------------------+---------------+
| dYdX v4 Indexer    | Hourly funding rate | Genesis (Oct 2023, | Point-in-time snapshot for OI;    |
| (Derivatives)      | (Snapshot only OI)  | ~25,000+ rows/mkt) | hourly funding ~1h cadence         |
+--------------------+---------------------+--------------------+--------------------+---------------+
```

### 1.1 Coinbase REST & WebSocket (Crypto)
- **Granularities:** Supported bucket sizes are fixed by the exchange engine: `60` (1m), `300` (5m), `900` (15m), `3600` (1h), `21600` (6h), and `86400` (1d) seconds.
- **Request Limits & Pagination:** Exactly **300 candles per single HTTP request**. A requested `(start, end)` range spanning more than 300 intervals is rejected by the server. Paginating historical data requires deterministic chunking:
  - 1 day of 1m candles = 1,440 bars -> 5 HTTP requests.
  - 90 days of 1m candles = 129,600 bars -> 432 HTTP requests.
  - 1 year of 5m candles = 105,120 bars -> 351 HTTP requests.
- **Historical Depth:** Inception-to-date history is accessible via REST for major pairs (e.g., `BTC-USD` history spans back to 2015+).
- **Data Quality & Edge Cases:**
  - *Zero-volume intervals:* Coinbase omits buckets where zero trades occurred. Time-series indexing must explicitly forward-fill or zero-fill missing timestamps.
  - *Provisional Head:* The current open bucket is returned by REST as an active candle. As codified in `stock_forecasting/live_feed.py:153`, `is_provisional = (now < bucket_start + granularity)`.

### 1.2 yfinance / Yahoo Finance (Equities)
- **Hard Lookback Ceilings:** Imposed by the Yahoo Finance chart backend:
  - `1m` interval: **7 calendar days** maximum.
  - `2m`, `5m`, `15m`, `30m`, `90m` intervals: **60 calendar days** maximum.
  - `60m` / `1h` interval: **730 calendar days** (~2 years).
- **Effective Sample Size for 5m Equity Bars:**
  - US regular trading hours (09:30-16:00 ET) span 6.5 hours = **78 bars per session** (at 5-minute intervals).
  - Over a 60-calendar-day lookback, there are approximately 42 trading sessions:
    $$\text{Total 5m Bars} \approx 42 \times 78 = 3,276 \text{ bars per ticker}$$
  - This is an extraordinarily narrow dataset for machine learning (comparable to ~13 trading days of continuous 24/7 crypto data).
- **Delay & Feed Instability:**
  - Free web endpoints have an unadvertised, uncontracted **~15-minute delay**.
  - Polling more frequently than once every 1-5 minutes triggers `HTTP 429 Too Many Requests` or persistent IP blacklisting.
  - Splits, dividend adjustments, and overnight session boundaries require careful index handling.

### 1.3 dYdX v4 Derivatives (Perpetual Funding & Open Interest)
- **Granularity & Depth:**
  - Hourly funding rates (`/historicalFunding/{market}`) are available from chain genesis (October/November 2023 to present), yielding **~25,000+ hourly observations per active market** (`BTC-USD`, `ETH-USD`, `SOL-USD`).
  - Pagination limit: Max `1000` items per page via `effectiveBeforeOrAt` cursor.
- **Open Interest (OI) Constraint:**
  - The dYdX v4 indexer (`/perpetualMarkets`) provides only a **single live snapshot** of open interest. There is **no historical OI endpoint** on the public indexer.
  - Historical OI is only accessible if recorded forward by a continuous local poller or ingested via external third-party data archives.
- **Alignment:** Funding is updated hourly. When merged with sub-hourly (1m/5m) price bars, the funding feature operates as a zero-order hold (step function), changing at the top of each hour.

---

## 2. Horizon Feasibility Analysis

When moving from daily end-of-day forecasting ($h \in \{1\text{d}, 5\text{d}, 30\text{d}\SingleQuote.replace("SingleQuote", "}") to intraday horizons, model feasibility depends on the Signal-to-Noise Ratio (SNR), data retention windows, and execution latency.

```
+----------------------------------------------------------------------------------------------------+
|                                    HORIZON FEASIBILITY MATRIX                                      |
+---------------+-------------------+----------------------+--------------------+--------------------+
| Horizon       | Ticker Class      | Effective Sample N   | SNR & Dynamics     | Feasibility Status |
+---------------+-------------------+----------------------+--------------------+--------------------+
| Next-Bar      | Equities (5m)     | ~3,200 bars (60d)    | SNR ~ 0; 15m delay | ❌ INFEASIBLE       |
| (1m / 5m)     |                   |                      | creates lookahead  | (Broken latency)   |
|               +-------------------+----------------------+--------------------+--------------------+
|               | Crypto (1m / 5m)  | ~26,000 bars (90d)   | Microstructure     | ⚠️ NOT RECOMMENDED  |
|               |                   |                      | noise dominates    | (Low SNR/Overfit)  |
+---------------+-------------------+----------------------+--------------------+--------------------+
| Next-Hour     | Equities (1h)     | ~270 hourly bars     | Low sample power;  | ⚠️ MARGINAL        |
| (60m)         |                   | (60d lookback)       | 15m delay lag      | (Data starved)     |
|               +-------------------+----------------------+--------------------+--------------------+
|               | Crypto (1h)       | ~8,760 bars/yr       | Matches funding;   | ✅ HIGHLY VIABLE    |
|               |                   | (Multi-year avail.)  | filters tick noise | (Target sweetspot) |
+---------------+-------------------+----------------------+--------------------+--------------------+
| Next-4-Hour / | Equities (4h)     | ~84 sessions (60d)   | Sample size too    | ❌ INFEASIBLE       |
| Half-Session  |                   |                      | small for ML       | (Sample starvation)|
| (4h / 8h)     +-------------------+----------------------+--------------------+--------------------+
|               | Crypto (4h / 8h)  | ~2,190 bars/yr       | Strong volatility  | ✅ HIGHLY VIABLE    |
|               |                   |                      | regime clustering  | (Swing momentum)   |
+---------------+-------------------+----------------------+--------------------+--------------------+
```

### 2.1 Next-Bar (1m / 5m) Horizon
- **The Equity Latency Paradox:** Predicting bar $t+1$ (the next 5 minutes) when the latest observable price is delayed by 15 minutes ($t-3$) requires predicting 4 bars into the future to know what is happening 'now'. Real-time evaluation is meaningless on delayed equity feeds.
- **Crypto Microstructure Noise:** Bid-ask bounce, order routing delays, and discrete execution ticks dominate returns at the 1m/5m level. Without full Level-2/Level-3 order book data (order flow toxicity, bid-ask queue imbalance), OHLCV-only 1m/5m directional predictions have an $R^2 \approx 0$ and high false discovery rates.

### 2.2 Next-Hour (1h / 12 * 5m bars) Horizon - The Optimal Target
- **Crypto Alignment:** 
  - 1-hour returns filter out transient tick bounces while capturing momentum drifts and mean-reversion around intraday volume-weighted average price (VWAP).
  - Perfectly matches the 1-hour funding rate calculation window on dYdX.
  - Annual sample size of 8,760 non-overlapping bars (or 105,120 5-minute step bars) provides adequate statistical power for non-linear tree models.
- **Equity Constraints:** At 1 hour, equity data yields only ~6 bars per trading day. With a 60-day yfinance cap, training is limited to ~270 bars-insufficient for anything beyond a simple regularized linear baseline.

### 2.3 Multi-Hour (4h / 8h) Horizon
- **Crypto Volatility & Funding Regime:** The 4h and 8h horizons correspond to global derivatives funding settlement intervals (major perpetual exchanges settle funding every 8 hours at 00:00, 08:00, and 16:00 UTC). Volatility clusters and macro liquidation cascades exhibit actionable autocorrelation at the 4h scale.

---

## 3. Model Approach Survey for Short-Horizon Price/Return Forecasting

Survey of model paradigms evaluated for a single-user local application (Python 3.12, SQLite WAL, scikit-learn stack, nightly local CPU retrain).

```
+----------------------------------------------------------------------------------------------------+
|                                    MODEL APPROACH COMPARISON                                       |
+----------------------+--------------------+---------------+-----------------+----------------------+
| Model Family         | Compute Cost       | Data Hunger   | Local Footprint | Assessment & Verdict |
|                      | (Train / Predict)  | (Min Samples) | & Dependencies  |                      |
+----------------------+--------------------+---------------+-----------------+----------------------+
| 1. Regularized Linear| CPU: < 1 second    | ~500 bars     | Zero overhead   | ✅ ESSENTIAL BASELINE|
| (Ridge / ElasticNet) | Inference: < 1 ms  | (Low)         | (scikit-learn)  | Fast, robust to noise|
+----------------------+--------------------+---------------+-----------------+----------------------+
| 2. Gradient Boosted  | CPU: 2 - 15 sec    | ~5,000 bars   | Minimal         | 🏆 RECOMMENDED       |
| Trees (LightGBM)     | Inference: < 5 ms  | (Moderate)    | (lightgbm pip)  | Best tabular alpha,  |
|                      |                    |               |                 | non-linear, robust   |
+----------------------+--------------------+---------------+-----------------+----------------------+
| 3. Realized Vol      | CPU: < 1 second    | ~1,000 bars   | Zero overhead   | 💡 EXCELLENT FOR CIs |
| (HAR-RV / GARCH)     | Inference: < 1 ms  | (Low)         | (numpy/scipy)   | Dynamic bands, not   |
|                      |                    |               |                 | directional returns  |
+----------------------+--------------------+---------------+-----------------+----------------------+
| 4. Recurrent / Conv  | CPU/GPU: 2 - 10 min| > 50,000 bars | Heavy           | ❌ OVERKILL          |
| (LSTM / GRU / TCN)   | Inference: ~50 ms  | (High)        | (PyTorch/TF)    | Prone to overfitting,|
|                      |                    |               |                 | high tuning burden   |
+----------------------+--------------------+---------------+-----------------+----------------------+
| 5. Foundation / TSF  | Heavy GPU / API    | Pretrained /  | Massive         | ❌ EXCESSIVE BLOAT   |
| (PatchTST / Chronos) | Inference: >500 ms | Millions      | (PyTorch + Hug) | Slow local inference,|
|                      |                    |               |                 | black-box latency    |
+----------------------+--------------------+---------------+-----------------+----------------------+
```

### 3.1 Regularized Linear Models (Ridge / ElasticNet)
- **Mechanics:** Fit continuous multi-step or single-step log-returns against standardized rolling feature vectors (lagged returns, momentum differentials, realized volatility, funding rate).
- **Pros:** Sub-second training, zero tendency to memorize noisy microstructure spikes when $\alpha$ penalty is tuned, analytical confidence intervals via residual variance.
- **Cons:** Unable to capture non-linear threshold effects (e.g., volatility expansion triggering momentum regime changes).

### 3.2 Gradient Boosted Decision Trees (LightGBM / XGBoost) — *Recommended Standard*
- **Mechanics:** Boosted shallow trees (depth 3-5) trained on tabular technical, statistical, and derivatives features to predict $k$-step forward log return:
  $$r_{t, t+k} = \ln(P_{t+k} / P_t)$$
- **Engineered Intraday Feature Suite:**
  - *Intraday VWAP Distance:* $(P_t - \text{VWAP}_t) / \sigma_{\text{VWAP}}$
  - *Realized Volatility Ratios:* $\sigma_{15\text{m}} / \sigma_{4\text{h}}$ (volatility breakout signal)
  - *Multi-Scale Return Spreads:* EWMA(5m) - EWMA(60m) normalized by ATR
  - *Funding Rate Momentum:* dYdX hourly funding rate 24h z-score
  - *Volume Acceleration:* $\text{Volume}_t / \text{SMA}(\text{Volume}, 20)$
- **Pros:** Industry standard for quantitative intraday alpha; handles mixed feature distributions without manual scaling; native missing-value tolerance; trains in seconds on commodity CPUs.

### 3.3 Heterogeneous Autoregressive Realized Volatility (HAR-RV)
- **Mechanics:** Predicts future volatility over daily, hourly, and sub-hourly intervals using a linear cascade of past realized volatilities at matching frequencies (Corsi, 2009).
- **Application:** Ideal for generating **dynamic intraday confidence intervals** ($\pm 1.96 \hat{\sigma}_{t, t+h}$) rather than directional point estimates.

### 3.4 Deep Sequential Networks (LSTM / GRU / TCN) & Transformers (PatchTST / Chronos)
- **Why They Are Overkill for `stock_forecasting`:**
  1. *Signal-to-Noise Deficit:* Deep architectures excel at high-SNR sequential patterns (audio, NLP, machine telemetry). In low-SNR financial returns, multi-layer LSTMs and Transformers rapidly memorize noise and fail out-of-sample walk-forward tests.
  2. *Data Starvation:* 60 days of 5m equity bars (~3,200 rows) is insufficient to train millions of weights without severe degenerate collapse.
  3. *Local System Complexity:* Introducing PyTorch / Transformers / HuggingFace libraries bloats the environment by 2-4 GB, complicates Windows binary dependencies, and increases worker CPU retrain cycles from seconds to tens of minutes.

---

## 4. Key Risks, Methodological Traps & Go/No-Go Decision

```
+----------------------------------------------------------------------------------------------------+
|                                    KEY METHODOLOGICAL RISKS                                        |
+------------------------------------+---------------------------------------------------------------+
| Risk Area                          | Mechanism & Impact on Model Validity                          |
+------------------------------------+---------------------------------------------------------------+
| 1. Label Leakage & Overlapping     | k-step forward returns share (k-1) bars of identical data.    |
|    Windows                         | Naive cross-validation causes massive serial correlation leak.|
+------------------------------------+---------------------------------------------------------------+
| 2. Free Equity 15-Min Delay        | Model trains on bar t to predict t+1, but bar t is only       |
|                                    | received at t+15m. Real-time inference is fundamentally stale.|
+------------------------------------+---------------------------------------------------------------+
| 3. Crypto 24/7 Continuity vs.      | No fixed daily close; diurnal volume waves shift globally.    |
|    No True Close                   | Dynamic intraday origin causes anchor drift against daily CI. |
+------------------------------------+---------------------------------------------------------------+
| 4. SQLite WAL & Retention Bloat    | Storing 1m bars indefinitely balloons SQLite size, locks WAL, |
|                                    | and slows Streamlit fragment render cycles.                   |
+------------------------------------+---------------------------------------------------------------+
```

### 4.1 Risk 1: Label Leakage on Overlapping Intraday Windows (The Multi-Step Trap)
When constructing a multi-bar forecast (e.g., predicting 1-hour forward return $r_{t, t+12}$ using 5-minute bars):
- Observation $t$ spans time $[t, t+12]$.
- Observation $t+1$ spans time $[t+1, t+13]$.
- These two training instances share **11 out of 12 bars** of identical future price movement.
- **Consequence:** Standard $K$-fold cross-validation or unpurged walk-forward validation leaks future information across folds, yielding artificially inflated Sharpe ratios and directional accuracies that collapse in live execution.
- **Required Mitigation:** Implement **Purged and Embargoed TimeSeriesSplit** (López de Prado, *Advances in Financial Machine Learning*):
  1. *Purging:* Drop all training samples whose label window overlaps with the test evaluation window.
  2. *Embargoing:* Drop training samples immediately following the test set by an autoregressive memory buffer (e.g., 24 hours) to eliminate serial correlation leakage.

### 4.2 Risk 2: The Equity 15-Minute Delay Mismatch
In v2.0.0, the display layer explicitly handles the 15-minute delay by applying a '🟡 15-min delayed' badge. However, in ML forecasting:
- If a model is fed the latest available bar at wall-clock 10:15 (which is actually the 10:00 bar) and outputs a 'next-5-minute' prediction for 10:05, that predicted event **already occurred 10 minutes ago**.
- To predict the *actual* next 5 minutes (10:15 -> 10:20), the model must predict across a **4-bar forward gap** ($t+3 -> t+4$).
- Predicting across an unobserved 15-minute blind spot severely degrades model accuracy and renders sub-hour equity forecasting unviable on free feeds.

### 4.3 Risk 3: Crypto 24/7 Continuity & Anchor Drift
- In equities, the 16:00 ET close provides an immutable, high-liquidity session anchor with overnight auction price discovery.
- Crypto has no natural close. In v2.0.0, the daily close is chosen as 00:00 UTC. Intraday price movements throughout the 24-hour cycle cause the live price to drift significantly from the $P_{\text{close}}$ anchor.
- As established by the v2 design integrity fence, confidence intervals calibrated to daily closes ($\pm 1.96 \sigma_h$) cannot be dynamically re-anchored to $P_{\text{live}}$ without invalidating the 95% walk-forward coverage guarantee. Any intraday forecast ribbon must be explicitly modeled as a separate short-horizon process ($r_{t, t+h_{\text{intra}}}$) rather than an ad-hoc mutation of the daily forecast ribbon.

### 4.4 Risk 4: Database Storage & Retention Dynamics
- Storing high-frequency bars across multiple assets creates significant SQLite churn:
  - 10 tickers * 1,440 1m bars/day = **14,400 rows/day** (~5.25M rows/year).
  - Unbounded retention leads to database locking during worker write bursts and slows Streamlit `@st.fragment` query times.
- **Mitigation:** The 7-day retention prune job (`INTRADAY_RETENTION_DAYS = 7`) introduced in v2.0.0 is adequate for display, but ML model training requires historical training buffers. If intraday models are built, historical training data must be stored in partitioned Parquet files or a dedicated historical table separate from the high-turnover `intraday_bars` table.

---

## 5. Go / No-Go Decision & Next Steps

```
+----------------------------------------------------------------------------------------------------+
|                                      GO / NO-GO SUMMARY                                            |
+------------------------------------+-----------+---------------------------------------------------+
| Track                              | Verdict   | Rationale                                         |
+------------------------------------+-----------+---------------------------------------------------+
| Crypto Intraday (1h / 4h Horizons) | 🟢 GO      | High data depth via keyless Coinbase REST/WS,     |
|                                    |           | 0s latency, aligned with dYdX hourly funding,     |
|                                    |           | solid tabular ML feasibility (LightGBM).          |
+------------------------------------+-----------+---------------------------------------------------+
| Equity Intraday (Sub-1h Horizons)  | 🔴 NO-GO  | 15-min feed delay creates an unbridgeable look-   |
|                                    |           | ahead/latency gap; 60-day cap limits sample size. |
+------------------------------------+-----------+---------------------------------------------------+
| Deep Learning / Transformers       | 🔴 NO-GO  | Extreme compute/dependency bloat; low SNR causes  |
|                                    |           | severe overfitting on local single-user setup.    |
+------------------------------------+-----------+---------------------------------------------------+
```

### Recommendation for Architecture Roadmap:
1. **Maintain Display/ML Separation:** Keep the daily ML core (`forecaster.py`, `trainer.py`, `evaluator.py`) frozen and display-separated as designed in v2.0.0.
2. **If Intraday Forecasting is Pursued:**
   - Scope exclusively to **Crypto** (`BTC-USD`, `ETH-USD`, `SOL-USD`) on **1-hour** and **4-hour** horizons.
   - Use **LightGBM / Ridge** with rolling intraday features and dYdX funding rate z-scores.
   - Mandate **Purged & Embargoed Walk-Forward Cross-Validation** to prevent overlapping label leakage.
   - Keep intraday training artifacts strictly separate from the daily prediction ledger.
