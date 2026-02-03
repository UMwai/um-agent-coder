# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [Data Fetcher Optimization]
**Learning:** `YahooFinanceFetcher.fetch_multiple` was applying rate limiting (sleep) even when data was served from cache. This caused unnecessary delays scaling with the number of requested items.
**Action:** Always check if a result came from cache (`FetchResult.cached`) before applying rate limiting logic in batched operations.
