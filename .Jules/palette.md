## 2024-05-22 - CLI First-Run Experience
**Learning:** CLI tools often crash on first run due to missing configuration, creating a hostile onboarding experience. Users shouldn't be greeted with a traceback or a raw error message when they first try the tool.
**Action:** Always check for configuration existence. If missing, generate a default config, print a friendly "Welcome" banner, and guide the user on the next steps (e.g., setting API keys) before cleanly exiting. Use colors to denote success (creation) and required action (setup).

## 2025-05-23 - Scannable CLI Lists
**Learning:** Long lists of data in CLI output are hard to read without visual guides. Users struggle to connect labels (left) with values (right) across whitespace.
**Action:** Use dot leaders (`....`) to visually connect labels to values. Align values to a consistent column. Use compact number formatting (e.g., `128k`, `1M`) for technical metrics like token counts to reduce visual noise and improve comparison. Use color sparingly (e.g., Green/Yellow/Red) to highlight key performance metrics.
