## 2024-05-22 - CLI First-Run Experience
**Learning:** CLI tools often crash on first run due to missing configuration, creating a hostile onboarding experience. Users shouldn't be greeted with a traceback or a raw error message when they first try the tool.
**Action:** Always check for configuration existence. If missing, generate a default config, print a friendly "Welcome" banner, and guide the user on the next steps (e.g., setting API keys) before cleanly exiting. Use colors to denote success (creation) and required action (setup).

## 2024-05-23 - CLI Data Density & Readability
**Learning:** High-density data lists in CLI are hard to scan. Users struggle to compare values like performance scores or costs when they are just plain text labels stacked vertically.
**Action:** Use color coding for critical metrics (Green/Yellow/Red for performance) and dot leaders for visual alignment to create scan lines. Compact numbers (1.2M vs 1,200,000) also reduce visual noise.
