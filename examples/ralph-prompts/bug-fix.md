# Ralph Prompt Template: Bug Fix

Use this template for fixing bugs with the Ralph Loop.

## Roadmap Task Definition

```markdown
- [ ] **fix-XXX**: [Bug description]. Output <promise>BUG_NAME_FIXED</promise> when the fix is verified with tests.
  - ralph: true
  - max_iterations: 15
  - completion_promise: BUG_NAME_FIXED
  - timeout: 30min
  - success: [Root cause identified, fix applied, regression test added]
  - cli: codex
```

## Example: Race Condition Fix

```markdown
- [ ] **fix-001**: Fix race condition causing duplicate user records during concurrent registration. Output <promise>RACE_CONDITION_FIXED</promise> when concurrent registrations are handled correctly.
  - ralph: true
  - max_iterations: 15
  - completion_promise: RACE_CONDITION_FIXED
  - timeout: 30min
  - success: Database constraint prevents duplicates, concurrent registration test passes, no duplicate users created under load
  - cli: codex
  - depends: none
```

## Example: Memory Leak Fix

```markdown
- [ ] **fix-002**: Fix memory leak in WebSocket connection handler. Output <promise>MEMORY_LEAK_FIXED</promise> when connections are properly cleaned up.
  - ralph: true
  - max_iterations: 20
  - completion_promise: MEMORY_LEAK_FIXED
  - timeout: 45min
  - success: Connections properly closed on disconnect, memory usage stable over 1000 connect/disconnect cycles, unit test verifies cleanup
  - cli: codex
  - depends: none
```

## Example: Authentication Bypass Fix

```markdown
- [ ] **fix-003**: Fix authentication bypass when JWT token is malformed. Output <promise>AUTH_BYPASS_FIXED</promise> when malformed tokens are rejected.
  - ralph: true
  - max_iterations: 10
  - completion_promise: AUTH_BYPASS_FIXED
  - timeout: 20min
  - success: Malformed JWT returns 401, empty token returns 401, expired token returns 401, security test cases added
  - cli: codex
  - depends: none
```

## Example: Data Corruption Fix

```markdown
- [ ] **fix-004**: Fix data corruption when saving UTF-8 characters in comments. Output <promise>UTF8_CORRUPTION_FIXED</promise> when all character encodings are preserved.
  - ralph: true
  - max_iterations: 12
  - completion_promise: UTF8_CORRUPTION_FIXED
  - timeout: 25min
  - success: Emojis, CJK characters, and special symbols preserved correctly, regression tests for edge cases
  - cli: codex
  - depends: none
```

## Tips for Bug Fix Tasks

1. **Describe the Bug Clearly**: Include symptoms, conditions that trigger it
2. **Lower Iteration Counts**: Bug fixes typically need 10-20 iterations
3. **Require Regression Tests**: Always add a test that would have caught the bug
4. **Include Root Cause**: Success criteria should verify the actual cause is fixed
5. **Shorter Timeouts**: Bug fixes are typically faster than features

## Common Success Criteria Patterns

```markdown
# Race condition
- success: Concurrent operations handled correctly, no data corruption, stress test passes

# Security fix
- success: Vulnerability no longer exploitable, security test cases added, audit log updated

# Performance fix
- success: Operation completes in < [N]ms, no regression under load, benchmark test added

# Data integrity fix
- success: Data preserved correctly in all cases, migration handles existing data, validation tests pass
```

## Debugging Strategy

For complex bugs, consider splitting into multiple tasks:

```markdown
# Investigation phase
- [ ] **fix-005a**: Investigate root cause of intermittent 500 errors in checkout. Document findings.
  - ralph: true
  - max_iterations: 10
  - completion_promise: ROOT_CAUSE_IDENTIFIED
  - success: Root cause documented, reproduction steps identified
  - cli: claude

# Fix implementation
- [ ] **fix-005b**: Implement fix for checkout 500 errors. Output <promise>CHECKOUT_ERROR_FIXED</promise> when stable.
  - ralph: true
  - max_iterations: 15
  - completion_promise: CHECKOUT_ERROR_FIXED
  - success: No 500 errors in checkout flow, regression tests added
  - cli: codex
  - depends: fix-005a
```
