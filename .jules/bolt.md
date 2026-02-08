# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-23 - [Cache Hit Identification]
**Learning:** In-memory cache hits in `DataFetcher` were returning the original result object (often with `cached=False`) instead of indicating a cache hit. This caused consumers to trigger unnecessary rate limiting or re-processing.
**Action:** Use `dataclasses.replace(result, cached=True)` when returning objects from L1 memory cache to correctly signal the source without side effects.
