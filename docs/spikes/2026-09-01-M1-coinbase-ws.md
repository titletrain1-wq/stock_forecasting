# M1 spike — Coinbase WebSocket URL ↔ channel pairing

**Date**: 2026-09-01 · **Probe**: live, keyless, `websockets` 16.1.1, 6 messages per pair.
**God condition 1**: confirm which URL serves a keyless last-price channel and document the exact channel name used.

## Result — all three candidates connect keyless

| # | URL | Subscribe | Outcome |
|---|---|---|---|
| A | `wss://ws-feed.exchange.coinbase.com` (classic exchange feed) | `{"type":"subscribe","product_ids":["BTC-USD"],"channels":["ticker","heartbeat"]}` | OK. Per-trade `{"type":"ticker","price":"77872.91","time":"2026-09-01T10:16:10.623011Z",...}` ~2/s + separate `{"type":"heartbeat",...}`. Flat schema, verbose (bid/ask/sizes). |
| B | `wss://advanced-trade-ws.coinbase.com` (Advanced Trade) | `{"type":"subscribe","product_ids":["BTC-USD"],"channel":"ticker_batch"}` | OK. Batched ~every 5s. Nested: `{"channel":"ticker_batch","timestamp":"...Z","events":[{"type":"snapshot|update","tickers":[{"product_id":"BTC-USD","price":"77863.33","price_percent_chg_24_h":"-0.70998..."}]}]}`. |
| C | `wss://advanced-trade-ws.coinbase.com` | `{"type":"subscribe","product_ids":["BTC-USD"],"channel":"ticker"}` | OK. Same nested shape as B but sub-second updates + bid/ask. Higher volume. |

## Decision — **B: `wss://advanced-trade-ws.coinbase.com`, channel `ticker_batch`**

Plus a second subscribe for channel `heartbeats` (Advanced Trade keepalive; the classic feed's
per-product `heartbeat` is a different channel and not used here).

**Why `ticker_batch` over the alternatives:**
- Batched ~5s → lowest bandwidth; a 2s `@st.fragment` read never starves and never floods.
- Payload already carries `price_percent_chg_24_h` → the price header's % change needs no extra
  computation or a second REST call.
- Matches Creed's `v2-research-apis.md` recommendation (`ticker_batch` + `heartbeats`).
- The classic feed (A) is per-trade and bid/ask-heavy; channel `ticker` (C) is sub-second — both are
  more data than a last-price display needs.

**Subscribe messages** (Advanced Trade takes `channel` singular, one message per channel):
```json
{"type":"subscribe","product_ids":["BTC-USD","ETH-USD"],"channel":"ticker_batch"}
{"type":"subscribe","product_ids":["BTC-USD","ETH-USD"],"channel":"heartbeats"}
```

**Price-message parse** (`_parse_message` in `live_feed.py`):
- gate on `msg.get("channel") == "ticker_batch"`
- envelope event time: `msg["timestamp"]` (ISO-8601, `...Z`)
- for each `ev in msg["events"]`, for each `t in ev["tickers"]`:
  `product_id = t["product_id"]`, `price = float(t["price"])`, optional `pct_24h = float(t["price_percent_chg_24_h"])`
- emit one `Tick(product_id, price, event_ts=<normalize Z→+00:00>, received_at=<utcnow>)` per ticker

**Keepalive**: while subscribed to an active product the ~5s batch itself keeps the socket alive;
the `heartbeats` subscription covers quiet products / off-hours. Reconnect on `ConnectionClosed`
with exponential backoff 1→2→4…→60s.

**Client library**: reuse `websockets` (already in `uv.lock`, no new dependency). Its async loop
runs inside a `worker.py` daemon thread (M2). `websocket-client` not added.

**Reconnect behaviour observed**: forced close → clean `ConnectionClosed`; immediate re-subscribe
replays a fresh `snapshot` event, so no gap-fill logic needed for a last-price display.
