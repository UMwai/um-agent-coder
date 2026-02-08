# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2026-01-28 - [Cache Hit Optimization]
**Learning:** Returning objects directly from memory cache can lead to ambiguous state (e.g., `cached=False` on a cached item) and accidental mutation. Always use `replace()` to return a copy with updated metadata.
**Action:** When implementing caching layers, ensure cache hits explicitly communicate their status via the returned object, and prevent unnecessary side effects (like rate limiting) when data is local.
