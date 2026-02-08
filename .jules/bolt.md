# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-23 - [Rate Limiting Logic]
**Learning:** Rate limiting logic (sleeps) was applied unconditionally in `fetch_multiple` loops, even when data was retrieved from the local cache. This caused unnecessary delays (N * sleep_time) for fully cached operations.
**Action:** Always check the source of data (cached vs fresh) before applying rate limits. Use flags like `result.cached` to bypass sleeps.
