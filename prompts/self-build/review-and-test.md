# Review and Test

## Context

- **Priority**: Ongoing
- **Scope**: Quality assurance across all features

## Goal

Comprehensive review and testing of implemented features. Run after completing implementation to verify everything works correctly.

## Review Checklist

### Code Quality

- [ ] **Type hints**: All functions have type hints
- [ ] **Docstrings**: Public APIs have docstrings
- [ ] **No commented code**: Remove any commented-out code
- [ ] **Consistent style**: Follow existing patterns in codebase
- [ ] **Error handling**: Exceptions are caught and logged appropriately

### Architecture

- [ ] **Interface compliance**: All implementations match interfaces in `specs/architecture/interfaces.md`
- [ ] **Single responsibility**: Each class has one clear purpose
- [ ] **Dependency direction**: Lower layers don't depend on higher layers
- [ ] **Configuration**: Hardcoded values are configurable

### Security

- [ ] **No secrets in code**: API keys come from environment
- [ ] **Input validation**: User input is validated
- [ ] **Subprocess safety**: Commands are properly escaped
- [ ] **File access**: Operations stay within working directory

## Testing Checklist

### Unit Tests

Run unit tests:
```bash
pytest tests/ -v
```

- [ ] All existing tests pass
- [ ] New features have tests
- [ ] Edge cases are covered

### Integration Tests

Run integration tests:
```bash
pytest tests/integration/ -v
```

- [ ] Harness can execute simple roadmap
- [ ] Autonomous loop runs for 10+ iterations
- [ ] Meta-harness can spawn sub-harness

### Manual Testing

#### Single Harness

```bash
# Create test roadmap
cat > /tmp/test-roadmap.md << 'EOF'
## Tasks
- [ ] **test-001**: Create a simple Python hello world script
  - timeout: 5min
  - success: hello.py exists and runs
  - cli: codex
EOF

# Run harness
python -m src.um_agent_coder.harness --roadmap /tmp/test-roadmap.md --dry-run
python -m src.um_agent_coder.harness --roadmap /tmp/test-roadmap.md
```

- [ ] Dry run shows expected behavior
- [ ] Harness completes task
- [ ] State persists correctly

#### Autonomous Mode

```bash
# Run with autonomous mode
python -m src.um_agent_coder.harness \
    --roadmap /tmp/test-roadmap.md \
    --autonomous \
    --max-iterations 10
```

- [ ] Progress detection works
- [ ] Alerts are generated
- [ ] Can check status in another terminal

#### Meta-Harness (when implemented)

```bash
# Create meta-roadmap
cat > /tmp/meta-roadmap.md << 'EOF'
## Strategy
coordination: parallel

## Sub-Harnesses

### harness-a
- working_dir: ./a
- roadmap: roadmap.md
- cli: codex

### harness-b
- working_dir: ./b
- roadmap: roadmap.md
- cli: codex
EOF

# Run meta-harness
python -m src.um_agent_coder.harness --meta --roadmap /tmp/meta-roadmap.md
```

- [ ] Sub-harnesses spawn
- [ ] Progress aggregates
- [ ] Results aggregate

### Performance Testing

```bash
# Check memory usage
python -m src.um_agent_coder.harness --roadmap large-roadmap.md &
PID=$!
while kill -0 $PID 2>/dev/null; do
    ps -o rss= -p $PID
    sleep 5
done
```

- [ ] Memory stays stable over time
- [ ] No memory leaks
- [ ] Context window doesn't grow unbounded

### Error Handling

- [ ] Graceful handling of CLI failures
- [ ] Graceful handling of timeout
- [ ] Graceful handling of Ctrl+C
- [ ] Resume works after interruption

## Linting and Formatting

```bash
# Format code
black src/
isort src/

# Lint
ruff check src/

# Type check
mypy src/
```

- [ ] No black formatting changes
- [ ] No isort changes
- [ ] No ruff errors/warnings
- [ ] No mypy errors

## Documentation Check

- [ ] CLAUDE.md is up to date
- [ ] README.md reflects current capabilities
- [ ] All CLI flags are documented
- [ ] Specs match implementation

## Final Verification

Before considering complete:

1. [ ] All tests pass
2. [ ] No linting errors
3. [ ] Documentation is current
4. [ ] Manual testing successful
5. [ ] No known regressions

## Reporting Issues

If issues are found:

1. Document the issue clearly
2. Create a task in roadmap or todo list
3. Prioritize based on severity
4. Fix before marking feature complete

---

*Use this prompt after implementing features to verify quality*
