# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2026-01-27 - [Cache Mutation & Unnecessary Sleeps]
**Learning:** In-memory objects stored in `DataFetcher` cache retained their initial state (`cached=False`) even when retrieved from cache. Mutating them in place is risky; returning a modified copy via `dataclasses.replace` is safer for consumers. Also, `YahooFinanceFetcher` was sleeping unconditionally even on cache hits.
**Action:** Always check if cached objects need status updates before returning, and use immutable copies. Ensure rate-limiting logic explicitly checks cache status.
