# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2026-01-26 - [Cache-Aware Rate Limiting]
**Learning:** Rate limiting logic was applied blindly even for cache hits, causing unnecessary delays (2.5s vs 0s) in `YahooFinanceFetcher`.
**Action:** Always check `result.cached` or equivalent before applying artificial delays or rate limits in fetch loops.
