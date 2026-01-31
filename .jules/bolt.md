# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [Rate Limiting and Caching]
**Learning:** Rate limiting logic in loops often unnecessarily penalizes cached results. `YahooFinanceFetcher` was sleeping 0.5s for every item even if the data was retrieved from cache.
**Action:** Always check `result.cached` or similar flags before applying rate limiting sleeps. Use `dataclasses.replace` to non-destructively set flags on cached objects.
