# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [Cached Flag Correctness]
**Learning:** `FetchResult` cached in memory retained `cached=False` because the object was reused from the initial fetch. This prevented `YahooFinanceFetcher` from optimizing rate limits on cache hits.
**Action:** Always return a copy (e.g., via `dataclasses.replace`) with `cached=True` when serving from cache to ensure downstream logic can rely on this flag.
