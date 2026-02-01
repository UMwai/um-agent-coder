# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [Cache Optimization]
**Learning:** Rate-limiting sleeps (e.g., in `data_fetchers.py`) must be conditional on actual API requests. Unconditional sleeps negate the performance benefits of caching.
**Action:** When implementing rate limiting, always check if the data was retrieved from cache before sleeping.
