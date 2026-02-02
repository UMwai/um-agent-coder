## 2024-05-22 - CLI First-Run Experience
**Learning:** CLI tools often crash on first run due to missing configuration, creating a hostile onboarding experience. Users shouldn't be greeted with a traceback or a raw error message when they first try the tool.
**Action:** Always check for configuration existence. If missing, generate a default config, print a friendly "Welcome" banner, and guide the user on the next steps (e.g., setting API keys) before cleanly exiting. Use colors to denote success (creation) and required action (setup).

## 2024-05-23 - Compact Data Visualization in CLI
**Learning:** Large raw numbers (e.g., 1,000,000 or 128,000) in CLI output create visual noise and make it hard to scan and compare metrics like context window size or cost.
**Action:** Use a "compact number" formatter (e.g., 1M, 128k) for technical metrics in lists/tables. This reduces cognitive load and aligns numbers visually, making the interface feel more professional and cleaner.
