# Autonomous Loop Test Roadmap

This roadmap tests the autonomous loop implementation.

## Tasks

- [ ] **auto-test-001**: Implement a simple feature with tests
  - ralph: true
  - max_iterations: 50
  - max_time: 30m
  - cli: codex,gemini
  - stuck_after: 5
  - completion_promise: FEATURE_COMPLETE
  - success: Feature works, tests pass, coverage > 80%

  Goal: Create a simple utility module with:
  1. A helper function for string manipulation
  2. Unit tests for the function
  3. Documentation

  Output `<progress>...</progress>` after each step.
  Output `<promise>FEATURE_COMPLETE</promise>` when done.
