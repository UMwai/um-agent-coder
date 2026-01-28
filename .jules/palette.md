## 2026-01-28 - Interactive Fallback for CLI Arguments
**Learning:** CLI tools often error out when required arguments are missing, which is hostile to new users. Adding an interactive fallback (using `sys.stdin.isatty()`) transforms a "User Error" into a "Guided Experience".
**Action:** Apply this pattern to other required CLI arguments where appropriate.
