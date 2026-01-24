# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [Cache-Aware Rate Limiting]
**Learning:** Rate limiting logic in fetchers often penalizes cache hits unnecessarily.
**Action:** Always check if the result is cached before applying rate limit sleeps. This can turn O(seconds) operations into O(milliseconds).
