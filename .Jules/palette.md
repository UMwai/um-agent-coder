## 2024-05-22 - CLI First-Run Experience
**Learning:** CLI tools often crash on first run due to missing configuration, creating a hostile onboarding experience. Users shouldn't be greeted with a traceback or a raw error message when they first try the tool.
**Action:** Always check for configuration existence. If missing, generate a default config, print a friendly "Welcome" banner, and guide the user on the next steps (e.g., setting API keys) before cleanly exiting. Use colors to denote success (creation) and required action (setup).

## 2025-02-12 - CLI Visual Hierarchy
**Learning:** Dense text lists in CLI interfaces (like model registries) are hard to scan. Users struggle to compare metrics (like performance vs cost) when they are just numbers in a block of text.
**Action:** Use color-coding for metrics (e.g., Green/Red for performance scores) and visual markers (like ✨ or bold text) to guide the eye to important elements. Highlight key values (like cost) in a distinct color to make them pop.
