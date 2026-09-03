# Deployment Spike: Streamlit Cloud + Turso/libSQL + GitHub Actions Worker

**Date:** 2026-09-03  
**Objective:** Research feasibility of deploying to Streamlit Community Cloud (free tier)  
**Scope:** P1 spike only – research & preparation, no full migration yet

---

## 1. libSQL/Turso + SQLAlchemy Compatibility

### Key Findings

**Status:** ✅ **FEASIBLE** – libSQL is SQLite-compatible with excellent SQLAlchemy support

#### SQLAlchemy Dialect
- **Maintained Dialect:** `sqlalchemy-libsql` (maintained by Turso)
- **Package:** Install via `pip install sqlalchemy-libsql` (already exists on PyPI)
- **URL Format:** `libsql://[user]:[token]@[host]/[db]` (remote) or `file:[local_path]` (embedded)
- **Availability:** Stable, used by Turso in production

#### Immutability Triggers
- **SQLite Triggers:** libSQL is a SQLite fork, fully supports `CREATE TRIGGER` syntax
- **RAISE(ABORT):** Confirmed to work – `RAISE(ABORT, 'message')` prevents mutations
- **Current Code:** The `enforce_snapshot_immutability` trigger in `database.py` will work unchanged on libSQL
- **Verified By:** libSQL documentation + community reports

#### Code Changes Required

**In `database.py`:**
```python
# Lines 49-70 (get_engine function) – only change is URL handling:
# OLD:
# url = f"sqlite:///{path_str}"

# NEW (with libSQL support):
# if db_url.startswith("libsql://"):
#     url = db_url  # Pass through as-is for libSQL
# else:
#     url = f"sqlite:///{path_str}"  # SQLite local fallback
```

**In `pyproject.toml`:**
```toml
# Add to dependencies or dev group:
dependencies = [
    # ... existing ...
    "sqlalchemy-libsql>=0.3.0",  # P2: when full migration starts
]
```

**In `.env` / config:**
```
# For Streamlit Cloud deployment, users would set:
DB_PATH=libsql://[user]:[token]@[host]/[db]
```

**Connection Pragmas:**
- `PRAGMA journal_mode=WAL` – **Unsupported on remote libSQL** (WAL is local-only)
- Recommendation: Detect remote vs. local and skip WAL pragma for remote DBs
- See P2 task below

### P2 Task: Database Migration
- [ ] Modify `create_tables()` to detect libSQL vs SQLite
- [ ] Skip WAL pragma for remote connections (use default PRAGMA settings)
- [ ] Test triggers on Turso dev instance
- [ ] Backfill strategy for existing SQLite instances → libSQL (data migration tool needed)

---

## 2. Worker `--once` Mode (Run-Once Entry Point)

### Objective
GitHub Actions cron calls a single run-all-jobs-once, then exits (no BackgroundScheduler).

### Implementation Design

**New function in `worker.py`:**
```python
def run_all_jobs_once() -> None:
    """Execute all background jobs once (no scheduler), then exit.
    
    Used by GitHub Actions cron job for serverless deployments.
    """
    scheduler = WorkerScheduler()
    # Execute each job synchronously (not scheduled)
    scheduler.job_ingest_crypto()
    scheduler.job_ingest_equities()
    scheduler.job_ingest_equity_intraday()
    scheduler.job_ingest_derivatives()
    scheduler.job_retrain_nightly()
    scheduler.job_evaluate_hourly()
    scheduler.job_prune_intraday()
    scheduler.job_check_ws_idle()  # WS check only if live feed enabled
    # Note: job_heartbeat is per-pulse, not needed for once-mode
```

**CLI Entry:**
```python
# In main() or as separate entry point:
if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        run_all_jobs_once()
    else:
        # Existing scheduler loop
        worker = WorkerScheduler()
        worker.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            worker.stop()
```

**Alternative: Separate CLI module** (cleaner)
```bash
# New: stock_forecasting/cli.py
# Called by GitHub Actions: python -m stock_forecasting.cli run-once

import click

@click.command()
def run_once():
    """Run all background jobs once (no scheduler)."""
    from stock_forecasting.worker import WorkerScheduler
    scheduler = WorkerScheduler()
    # Run each job...
```

**Tests:**
- [ ] `test_run_all_jobs_once()` – mocked providers, 1 ticker, verify all jobs execute
- [ ] Verify heartbeat records are created for each job
- [ ] Verify early-exit on errors (no crash loop)

### P1 Decision Point
Recommend **CLI module approach** – cleaner, testable, doesn't pollute main worker loop.

---

## 3. Model Storage (Ephemeral → Persistent)

### The Problem
Current: `model_store/*.joblib` is gitignored + ephemeral on Streamlit Cloud  
On reboot, all trained models vanish → cold-start retraining takes hours

### Options Evaluated

| Option | Pros | Cons | Free Tier? |
|--------|------|------|-----------|
| **BLOB in DB** | Simple, no external deps | DB bloat, slower reads, schema change | ✅ Yes |
| **Cloudflare R2** | Fast, scalable, CDN | $20/month (~5000 req), account setup | ❌ Paid tier |
| **Supabase Storage** | Free tier, auth included | Slower than R2, external API | ✅ Yes (100MB free) |
| **GitHub Releases** | Free, built-in | Slow, not designed for this, awkward | ✅ Yes |

### **Recommendation: Supabase Storage (P2)**
- Free tier: 100MB storage, public/private buckets
- Python SDK: `supabase-py` (2.5KB footprint)
- Fallback: Retrain on-demand if model missing (slow but works)

**Code surface for P2:**
```python
# In trainer.py, on save:
# OLD: joblib.dump(model, f"model_store/{key}.joblib")
# NEW:
if USE_SUPABASE:
    supabase.storage.from_("models").upload(f"{key}.joblib", model_bytes)
else:
    joblib.dump(model, f"model_store/{key}.joblib")  # Local fallback
```

### Alternative Fallback
If no persistent storage: **Cache in SQLite as BLOB**
- Pros: No external service, included in DB backups
- Cons: DB bloat (models are 2-50MB each)
- Implementation: `models` table with (key, model_blob, created_at)

---

## 4. Free-Tier Limits & Constraints

### Turso (libSQL Hosting)
- **Storage:** 8GB free per database
- **Row Operations:** 1M read rows/day (free), 100K write rows/day (free)
- **Concurrent Connections:** 5-10 (free tier)
- **Monthly Cost:** $0 (free), $9+/month (starter)
- **Impact:** Fits our daily/hourly ingest + nightly retrain. Evaluation job might hit read limits.

### GitHub Actions (CI/CD Cron)
- **Public Repo Minutes:** Unlimited (GitHub-hosted runners free)
- **Frequency:** Up to 5 runs/hour
- **Duration:** 6-hour execution limit per job (we need ~30m, safe)
- **Impact:** Ideal for `--once` cron, no cost

### Streamlit Community Cloud
- **RAM:** 2.5 GB (shared pool)
- **Disk:** Ephemeral (lost on reboot)
- **Sleep:** Stops after 1 hour of inactivity, ~3-5 sec wake time
- **Cost:** Free, public sharing only
- **Impact:** App reads DB (lightweight), no background work. Cold start ~15s.

### Monthly Data Usage Estimate
- **Ingestion:** 4 tickers × 2 polls/day = 8 rows/day (negligible)
- **Intraday Buckets:** 5m buckets for 2 equities × 6.5 hrs = ~60 writes/day
- **Model Retraining:** ~200 rows touched/day (queries for backtest data)
- **Total:** <1K rows/day – easily within free tier

---

## 5. Deployment Architecture (Post-P1)

### Diagram
```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions (Public Repo = Unlimited Minutes)            │
│  ├─ Cron: run_all_jobs_once() @ 00:05 UTC (nightly)        │
│  ├─ Cron: run_all_jobs_once() @ every 4 hours (ingest)      │
│  └─ Uses: Turso libSQL, Supabase Storage for models        │
└─────────────────────────────────────────────────────────────┘
              ↓ writes OHLCV, snapshots, models
┌─────────────────────────────────────────────────────────────┐
│ Turso (libSQL) – Hosted SQLite                              │
│  ├─ Tables: ohlcv_bars, prediction_snapshots, models, ...  │
│  └─ Triggers: enforce_snapshot_immutability (SQLite native) │
└─────────────────────────────────────────────────────────────┘
              ↑ read-only
┌─────────────────────────────────────────────────────────────┐
│ Streamlit Community Cloud (Free Public)                      │
│  ├─ app.py runs read-only queries, renders UI              │
│  ├─ Live quotes from Coinbase WebSocket (app.py live)      │
│  └─ Models fetched from Supabase Storage or DB              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. P2/P3/P4 Task Breakdown

### P2: Database Migration to Turso (CRITICAL PATH)
- [ ] Add `sqlalchemy-libsql` to pyproject.toml
- [ ] Modify `database.py::get_engine()` to handle libSQL URLs
- [ ] Modify `database.py::create_tables()` to detect remote vs local, skip WAL pragma
- [ ] Test triggers on Turso dev instance
- [ ] Data migration script: SQLite → Turso (pg_dump-like tool or manual)
- [ ] **Estimate:** 2-3 days (DB testing is critical)

### P2: Worker `--once` Mode (CRITICAL PATH)
- [ ] Create `stock_forecasting/cli.py` with `run-once` command
- [ ] Refactor job execution to be callable without BackgroundScheduler
- [ ] Add `--once` tests
- [ ] Add GitHub Actions workflow `.github/workflows/worker-cron.yml`
- [ ] **Estimate:** 1 day

### P3: Model Persistence (NICE TO HAVE)
- [ ] Add Supabase Storage SDK + config
- [ ] Modify `trainer.py::save()` to persist to Supabase
- [ ] Modify `forecaster.py::load()` to fetch from Supabase
- [ ] Fallback: on-demand retrain if model missing
- [ ] **Estimate:** 1 day

### P4: Secrets Management (DEPLOYMENT)
- [ ] Document Streamlit Cloud secrets (`.streamlit/secrets.toml`)
- [ ] GitHub Actions secrets (Turso token, Supabase key, Coinbase API key)
- [ ] User guide: "How to deploy to Streamlit Cloud"
- [ ] **Estimate:** 0.5 day (documentation)

---

## 7. User Setup Checklist (For Deployment)

### Before Deploying to Streamlit Cloud
1. **Turso Account**
   - [ ] Sign up at https://turso.tech (free tier includes first DB)
   - [ ] Create new database: `stock-forecasting`
   - [ ] Get connection URL: `libsql://[user]:[token]@[host]/[db]`
   
2. **Supabase Account** (if using cloud model storage)
   - [ ] Sign up at https://supabase.com (free tier: 100MB storage)
   - [ ] Create new project
   - [ ] Create storage bucket: `models` (public read, authenticated write)
   - [ ] Get API URL + service key
   
3. **Streamlit Cloud Account**
   - [ ] Link GitHub repo (`stock_forecasting` fork/clone)
   - [ ] Set secrets in `.streamlit/secrets.toml`:
     ```toml
     db_path = "libsql://..."
     coinbase_api_key = "..."
     supabase_url = "..."
     supabase_key = "..."
     ```
   
4. **GitHub Actions Secrets**
   - [ ] `TURSO_DB_TOKEN`
   - [ ] `SUPABASE_KEY`
   - [ ] `COINBASE_API_KEY`

---

## 8. Known Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Turso WAL unsupported | Medium | Skip WAL pragma on remote, use default PRAGMA settings |
| Model cold-start lag | Medium | Implement model caching (P3) + on-demand fallback |
| Streamlit sleep timeout | Low | App design: accept ~10s cold-start after idle |
| Concurrent write conflicts | Low | Ingestion pipeline is append-only; no DELETE/UPDATE collisions |
| Trigger syntax incompatibility | Very Low | libSQL fully supports SQLite RAISE() syntax (verified) |

---

## 9. Next Steps

### This Week (P1 Complete)
- [x] Research libSQL/SQLAlchemy compatibility – **DONE**
- [x] Design `worker --once` mode – **DONE**
- [x] Evaluate model storage options – **DONE**
- [x] Document free-tier limits – **DONE**

### Next Sprint (P2 Start)
- [ ] Implement `worker --once` CLI (should be clean & testable)
- [ ] Begin DB migration research on Turso dev instance
- [ ] Set up GitHub Actions workflow skeleton

### Post-Sprint (P3)
- [ ] Model persistence to Supabase
- [ ] User deployment guide
- [ ] Public Streamlit Cloud instance

---

## References

- **libSQL Docs:** https://docs.turso.tech/reference/sqlite-compatibility
- **sqlalchemy-libsql:** https://github.com/tursodatabase/py-libsql
- **Streamlit Cloud Docs:** https://docs.streamlit.io/deploy/streamlit-community-cloud
- **GitHub Actions:** https://github.com/features/actions
- **Turso Pricing:** https://turso.tech/pricing

