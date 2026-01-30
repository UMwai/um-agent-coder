# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [DataFetcher Cache State]
**Learning:** In-memory cache hits in `DataFetcher` were returning original objects with `cached=False` (default), causing downstream logic (rate limiting bypass) to fail.
**Action:** Always use `dataclasses.replace(result, cached=True)` when returning from memory cache to ensure state consistency without mutating stored objects.
