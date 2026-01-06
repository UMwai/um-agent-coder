# Autonomous Loop Implementation Roadmap

> **Reference**: [spec.md](spec.md)
> **Priority**: HIGH (30%)

## Overview

This roadmap tracks the implementation of autonomous loop features. Many components already exist - this roadmap focuses on gaps and enhancements.

## Current State

### Implemented

| Component | Location | Status |
|-----------|----------|--------|
| ProgressDetector | `harness/autonomous/progress_detector.py` | Complete |
| StuckDetector | `harness/autonomous/recovery/stuck_detector.py` | Complete |
| RecoveryManager | `harness/autonomous/recovery/recovery_manager.py` | Complete |
| ContextManager | `harness/autonomous/context_manager.py` | Partial |
| CLIRouter | `harness/autonomous/cli_router.py` | Complete |
| AlertManager | `harness/autonomous/alerts/alert_manager.py` | Complete |
| EnvironmentManager | `harness/autonomous/environment/environment_manager.py` | Complete |
| RunawayDetector | `harness/autonomous/runaway_detector.py` | Complete |

### Gaps

| Component | Gap | Priority |
|-----------|-----|----------|
| ContextManager | Summarization not LLM-based | Medium |
| RecoveryManager | Branch exploration incomplete | Medium |
| CLI Integration | `--autonomous` shorthand mode | High |
| Documentation | User guide | Medium |

## Implementation Phases

### Phase 1: Consolidation

**Goal**: Ensure all existing components work together smoothly.

**Tasks**:
- [ ] **1.1** Verify ProgressDetector weights are optimal
- [ ] **1.2** Test StuckDetector thresholds (3 iterations default)
- [ ] **1.3** Validate RecoveryManager escalation order
- [ ] **1.4** Test EnvironmentManager file watchers
- [ ] **1.5** Document current behavior in spec.md

**Files to Review**:
- `src/um_agent_coder/harness/autonomous/*.py`

---

### Phase 2: Context Summarization Enhancement

**Goal**: Implement LLM-based context summarization.

**Tasks**:
- [ ] **2.1** Add summarization to ContextManager
- [ ] **2.2** Use cheap model (gemini-3-flash) for summarization
- [ ] **2.3** Configure summarize_every interval
- [ ] **2.4** Preserve key decisions in summaries
- [ ] **2.5** Test context window doesn't grow unbounded

**Files to Modify**:
```
src/um_agent_coder/harness/autonomous/
├── context_manager.py  # Add summarization
└── summarizer.py       # NEW: LLM summarization
```

---

### Phase 3: Branch Exploration

**Goal**: Complete parallel branch exploration for stuck recovery.

**Tasks**:
- [ ] **3.1** Implement `BranchExplorer` class
- [ ] **3.2** Generate 2-3 alternative approaches
- [ ] **3.3** Execute branches in parallel (threads)
- [ ] **3.4** Score branches by progress
- [ ] **3.5** Merge winning branch
- [ ] **3.6** Clean up losing branches

**Files to Create**:
```
src/um_agent_coder/harness/autonomous/recovery/
└── branch_explorer.py  # NEW
```

---

### Phase 4: CLI Polish

**Goal**: Add `--autonomous` shorthand and improve UX.

**Tasks**:
- [ ] **4.1** Add `--autonomous` flag to CLI
- [ ] **4.2** Configure sensible defaults for autonomous mode
- [ ] **4.3** Add `--conservative` mode (more human oversight)
- [ ] **4.4** Improve `--status` output formatting
- [ ] **4.5** Add progress bar for long-running tasks

**Files to Modify**:
```
src/um_agent_coder/harness/
├── main.py             # Add flags
└── cli_config.py       # Shorthand configurations
```

---

### Phase 5: Documentation & Testing

**Goal**: Complete documentation and test coverage.

**Tasks**:
- [ ] **5.1** Update `docs/autonomous-loop.md` user guide
- [ ] **5.2** Add integration tests for autonomous mode
- [ ] **5.3** Document all CLI flags
- [ ] **5.4** Add troubleshooting guide
- [ ] **5.5** Update CLAUDE.md with new capabilities

**Files to Create/Modify**:
```
docs/
├── autonomous-loop.md  # User guide
└── troubleshooting.md  # NEW

tests/
└── test_autonomous_integration.py  # NEW
```

---

## Dependency Graph

```
Phase 1 (Consolidation)
    │
    ├──────────────────────┐
    ▼                      ▼
Phase 2 (Context)    Phase 3 (Branch)
    │                      │
    └─────────┬────────────┘
              ▼
        Phase 4 (CLI)
              │
              ▼
        Phase 5 (Docs)
```

## Success Criteria

### Functional

- [ ] Autonomous mode runs 24h+ without human intervention
- [ ] Progress detection catches 90%+ of actual progress
- [ ] Stuck recovery resolves 80%+ of stuck states
- [ ] Context window stays under 50k tokens

### Observability

- [ ] `--status` shows clear progress
- [ ] Alerts are actionable
- [ ] Logs are structured and searchable

### Reliability

- [ ] Resume works after any interruption
- [ ] No data loss on crash
- [ ] Graceful degradation on model failures

---

*Last Updated: January 2026*
