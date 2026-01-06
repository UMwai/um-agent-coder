# Implement Foundation Components

## Context

- **Reference**: `specs/features/foundation/*.md`
- **Priority**: MEDIUM (20%)
- **Scope**: Bug fixes, enhancements, polish

## Goal

Complete and polish the foundation components: CLI routing, progress detection, stuck recovery, and context management. These components support both the autonomous loop and meta-harness features.

## Spec Summary

Foundation components are building blocks used by higher-level features. Most are already implemented but may need fixes or enhancements.

### Components

1. **CLI Routing** (`specs/features/foundation/cli-routing.md`)
   - Route tasks to optimal CLI (Codex, Gemini, Claude)
   - Auto-routing based on task characteristics
   - Opus preservation (rate limiting)

2. **Progress Detection** (`specs/features/foundation/progress-detection.md`)
   - Multi-signal weighted scoring
   - Output diff, file changes, explicit markers, checklist

3. **Stuck Recovery** (`specs/features/foundation/stuck-recovery.md`)
   - Prompt mutation
   - Model escalation
   - Branch exploration

4. **Context Management** (`specs/features/foundation/context-management.md`)
   - Rolling window of raw iterations
   - LLM-based summarization

## Implementation Checklist

### CLI Routing Fixes

- [ ] Review `src/um_agent_coder/harness/autonomous/cli_router.py`
  - [ ] Verify IMPLEMENTATION_KEYWORDS are complete
  - [ ] Verify RESEARCH_KEYWORDS are complete
  - [ ] Verify COMPLEX_KEYWORDS are complete
  - [ ] Test auto-routing with various task types

- [ ] Review OpusGuard
  - [ ] Verify daily limit works correctly
  - [ ] Verify reset at midnight
  - [ ] Add logging for Opus usage

- [ ] Add tests
  - [ ] Test routing decisions
  - [ ] Test Opus rate limiting

### Progress Detection Fixes

- [ ] Review `src/um_agent_coder/harness/autonomous/progress_detector.py`
  - [ ] Verify weights are optimal (output_diff: 30%, file_changes: 30%, markers: 25%, checklist: 15%)
  - [ ] Verify output_diff algorithm (SequenceMatcher)
  - [ ] Verify file_changes detects git changes

- [ ] Improve marker detection
  - [ ] Support `<progress>...</progress>` tags
  - [ ] Support checklist format (`- [x]`)

- [ ] Add tests
  - [ ] Test progress scoring
  - [ ] Test edge cases (no previous output, etc.)

### Stuck Recovery Fixes

- [ ] Review `src/um_agent_coder/harness/autonomous/recovery/`
  - [ ] Verify stuck_detector thresholds (default: 3 no-progress iterations)
  - [ ] Verify recovery_manager stage order

- [ ] Review PromptMutator
  - [ ] Verify mutation strategies work
  - [ ] Add more mutation types if needed

- [ ] Review ModelEscalator
  - [ ] Verify escalation order is optimal
  - [ ] Handle case when already at highest model

- [ ] Add tests
  - [ ] Test stuck detection
  - [ ] Test each recovery strategy

### Context Management Fixes

- [ ] Review `src/um_agent_coder/harness/autonomous/context_manager.py`
  - [ ] Verify rolling window size (default: 5)
  - [ ] Verify truncation limits

- [ ] Implement missing summarization
  - [ ] See `implement-autonomous-loop.md` for details

- [ ] Add tests
  - [ ] Test window management
  - [ ] Test context doesn't grow unbounded

### Documentation

- [ ] Create `specs/features/foundation/cli-routing.md`
- [ ] Create `specs/features/foundation/progress-detection.md`
- [ ] Create `specs/features/foundation/stuck-recovery.md`
- [ ] Create `specs/features/foundation/context-management.md`

## Success Criteria

1. **CLI Routing**:
   - [ ] Tasks route to expected CLI 90%+ of the time
   - [ ] Opus usage stays within daily limit

2. **Progress Detection**:
   - [ ] Detects progress on 90%+ of productive iterations
   - [ ] No false positives on stuck iterations

3. **Stuck Recovery**:
   - [ ] Recovers from 80%+ of stuck states
   - [ ] Doesn't waste iterations on hopeless tasks

4. **Context Management**:
   - [ ] Context stays under 50k tokens
   - [ ] Key information is preserved in summaries

## References

- `src/um_agent_coder/harness/autonomous/` - Existing implementation
- `specs/features/autonomous-loop/spec.md` - How components integrate
- `specs/architecture/interfaces.md` - Interface contracts

---

*Use this prompt with: `cat prompts/self-build/implement-foundation.md | claude-code`*
