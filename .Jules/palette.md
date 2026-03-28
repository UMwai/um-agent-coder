## 2024-05-22 - CLI First-Run Experience
**Learning:** CLI tools often crash on first run due to missing configuration, creating a hostile onboarding experience. Users shouldn't be greeted with a traceback or a raw error message when they first try the tool.
**Action:** Always check for configuration existence. If missing, generate a default config, print a friendly "Welcome" banner, and guide the user on the next steps (e.g., setting API keys) before cleanly exiting. Use colors to denote success (creation) and required action (setup).

## 2025-02-18 - CLI Interactive Fallback
**Learning:** Users often run CLI tools without arguments to explore them. Showing a help message or an error ("prompt required") is functional but dry. Detect interactive TTY sessions and escalate to an interactive prompt instead of erroring out.
**Action:** Use `sys.stdin.isatty()` to detect humans. If detected and arguments are missing, launch an interactive input mode. Fall back to standard error messages for automation/pipes.
