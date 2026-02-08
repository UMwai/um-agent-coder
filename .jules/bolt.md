# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2025-02-23 - [In-Memory Cache State Mutation]
**Learning:** In `DataFetcher`, objects stored in `_memory_cache` retain their initial state (`cached=False`) when retrieved, leading to false negatives in cache hit detection. This caused `YahooFinanceFetcher.fetch_multiple` to sleep for rate limiting even on cache hits.
**Action:** Use `dataclasses.replace(result, cached=True)` when returning from memory cache to ensure the caller knows the source without mutating the stored object or relying on stale state.
