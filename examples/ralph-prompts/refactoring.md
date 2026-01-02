# Ralph Prompt Template: Code Refactoring

Use this template for refactoring tasks with the Ralph Loop.

## Roadmap Task Definition

```markdown
- [ ] **refactor-XXX**: [Refactoring description]. Output <promise>REFACTOR_NAME_COMPLETE</promise> when all code is migrated and tests pass.
  - ralph: true
  - max_iterations: 40
  - completion_promise: REFACTOR_NAME_COMPLETE
  - timeout: 60min
  - success: [Old code removed, new structure in place, all tests pass]
  - cli: codex
```

## Example: Module Extraction

```markdown
- [ ] **refactor-001**: Extract authentication logic into separate auth/ module. Output <promise>AUTH_MODULE_EXTRACTED</promise> when extraction is complete.
  - ralph: true
  - max_iterations: 40
  - completion_promise: AUTH_MODULE_EXTRACTED
  - timeout: 60min
  - success: auth/ directory created with login.py, logout.py, tokens.py; all auth imports updated; no auth logic in main app; all tests pass
  - cli: codex
  - depends: none
```

## Example: Database Migration

```markdown
- [ ] **refactor-002**: Migrate from SQLite to PostgreSQL. Output <promise>POSTGRES_MIGRATION_COMPLETE</promise> when all data is migrated and tests pass.
  - ralph: true
  - max_iterations: 50
  - completion_promise: POSTGRES_MIGRATION_COMPLETE
  - timeout: 90min
  - success: PostgreSQL connection configured, all models work with Postgres, data migration script tested, SQLite removed from config
  - cli: codex
  - depends: none
```

## Example: API Version Upgrade

```markdown
- [ ] **refactor-003**: Upgrade REST API from v1 to v2 with breaking changes. Output <promise>API_V2_COMPLETE</promise> when v2 is fully implemented.
  - ralph: true
  - max_iterations: 45
  - completion_promise: API_V2_COMPLETE
  - timeout: 75min
  - success: /api/v2/ endpoints implemented, v1 endpoints deprecated with warnings, migration guide created, all v2 tests pass
  - cli: codex
  - depends: none
```

## Example: Dependency Upgrade

```markdown
- [ ] **refactor-004**: Upgrade React from 17 to 18 with concurrent features. Output <promise>REACT_18_UPGRADE_COMPLETE</promise> when upgrade is verified.
  - ralph: true
  - max_iterations: 35
  - completion_promise: REACT_18_UPGRADE_COMPLETE
  - timeout: 60min
  - success: React 18 installed, createRoot used, Suspense boundaries added, no console warnings, all component tests pass
  - cli: codex
  - depends: none
```

## Example: Code Quality Improvement

```markdown
- [ ] **refactor-005**: Add type hints to all functions in utils/ module. Output <promise>UTILS_TYPED</promise> when typing is complete.
  - ralph: true
  - max_iterations: 30
  - completion_promise: UTILS_TYPED
  - timeout: 45min
  - success: All functions have type hints, mypy passes with no errors, no Any types except where necessary
  - cli: codex
  - depends: none
```

## Example: Design Pattern Implementation

```markdown
- [ ] **refactor-006**: Implement repository pattern for data access layer. Output <promise>REPOSITORY_PATTERN_COMPLETE</promise> when pattern is applied.
  - ralph: true
  - max_iterations: 40
  - completion_promise: REPOSITORY_PATTERN_COMPLETE
  - timeout: 60min
  - success: Repository interfaces defined, concrete implementations for each model, direct DB access removed from services, dependency injection configured
  - cli: claude
  - depends: none
```

## Tips for Refactoring Tasks

1. **Higher Iteration Counts**: Refactoring often needs 30-50 iterations
2. **Require All Tests Pass**: Never accept partial refactoring
3. **Specify What Gets Removed**: Clarify old code/patterns to eliminate
4. **Use Claude for Complex Patterns**: Design patterns benefit from reasoning
5. **Longer Timeouts**: Large refactoring needs 60-90min per iteration

## Common Success Criteria Patterns

```markdown
# Module extraction
- success: [module]/ directory exists, all related code moved, imports updated, old location empty, tests pass

# Database migration
- success: New database configured, data migrated, old database removed, no hardcoded references, tests pass

# Pattern implementation
- success: Pattern interfaces defined, implementations complete, old approach removed, documentation updated

# Dependency upgrade
- success: New version installed, deprecated APIs replaced, no console warnings, all tests pass
```

## Staged Refactoring

For large refactorings, break into phases:

```markdown
# Phase 1: Create new structure
- [ ] **refactor-007a**: Create new microservice structure with shared types.
  - ralph: true
  - max_iterations: 25
  - completion_promise: NEW_STRUCTURE_READY
  - success: services/ directory with user/, product/, order/ subdirectories; shared types in common/

# Phase 2: Migrate logic
- [ ] **refactor-007b**: Migrate business logic to microservice structure.
  - ralph: true
  - max_iterations: 40
  - completion_promise: LOGIC_MIGRATED
  - success: All business logic in respective services, no cross-service imports except common/
  - depends: refactor-007a

# Phase 3: Remove old code
- [ ] **refactor-007c**: Remove deprecated monolith code.
  - ralph: true
  - max_iterations: 20
  - completion_promise: MONOLITH_REMOVED
  - success: Old monolith files deleted, no orphan imports, all tests pass
  - depends: refactor-007b
```

## Quality Metrics

Include measurable metrics in success criteria:

```markdown
# Code coverage
- success: Test coverage above 80%, no uncovered critical paths

# Complexity reduction
- success: Average cyclomatic complexity below 10, no function over 50 lines

# Type safety
- success: mypy passes with strict mode, no untyped functions

# Performance
- success: Response time under 100ms, no N+1 queries
```
