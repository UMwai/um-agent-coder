# Implement Autonomous Loop Enhancements

## Context

- **Reference**: `specs/features/autonomous-loop/spec.md`
- **Roadmap**: `specs/features/autonomous-loop/roadmap.md`
- **Priority**: HIGH (30%)
- **Scope**: ~5 new files, ~10 modified files

## Goal

Enhance the existing autonomous loop implementation with LLM-based context summarization, complete branch exploration for stuck recovery, and improved CLI experience.

## Current State

Most autonomous components are already implemented:
- ProgressDetector (complete)
- StuckDetector (complete)
- RecoveryManager (complete)
- CLIRouter (complete)
- AlertManager (complete)
- EnvironmentManager (complete)

## Spec Summary

### Components to Enhance

1. **ContextManager Enhancement** (`harness/autonomous/context_manager.py`)
   - Add LLM-based summarization (use gemini-3-flash)
   - Summarize every N iterations
   - Keep raw window of last 5 iterations

2. **BranchExplorer** (`harness/autonomous/recovery/branch_explorer.py`)
   - Generate 2-3 alternative approaches when stuck
   - Execute in parallel
   - Score by progress, pick winner

3. **CLI Polish** (`harness/main.py`)
   - `--autonomous` shorthand mode
   - `--conservative` mode
   - Improved `--status` output

## Implementation Checklist

### Phase 1: Context Summarization

- [ ] Create `src/um_agent_coder/harness/autonomous/summarizer.py`
  - [ ] Implement `ContextSummarizer` class
  - [ ] Use gemini-3-flash for summarization
  - [ ] Preserve key decisions in summaries
  - [ ] Limit summary to 500 words

- [ ] Modify `src/um_agent_coder/harness/autonomous/context_manager.py`
  - [ ] Add `summarize_every` config (default: 10)
  - [ ] Call summarizer when total_iterations % summarize_every == 0
  - [ ] Store summary in context

- [ ] Test context summarization
  - [ ] Verify summary captures progress
  - [ ] Verify context window doesn't grow unbounded

### Phase 2: Branch Exploration

- [ ] Create `src/um_agent_coder/harness/autonomous/recovery/branch_explorer.py`
  - [ ] Define `ExplorationBranch` dataclass
  - [ ] Implement `BranchExplorer.explore(goal, context)` → List[Branch]
  - [ ] Generate approaches: bottom-up, research-first, constraint-driven
  - [ ] Implement `execute_branches(branches)` using ThreadPoolExecutor
  - [ ] Score branches by progress_score
  - [ ] Return winning branch

- [ ] Modify `src/um_agent_coder/harness/autonomous/recovery/recovery_manager.py`
  - [ ] Add Stage 3: Branch Exploration (after Model Escalation)
  - [ ] Configure branch_count (default: 3)
  - [ ] Configure branch_budget (default: 10 iterations each)

- [ ] Test branch exploration
  - [ ] Verify branches execute in parallel
  - [ ] Verify winning branch is selected correctly

### Phase 3: CLI Polish

- [ ] Modify `src/um_agent_coder/harness/main.py`
  - [ ] Add `--autonomous` flag
  - [ ] Add `--conservative` flag
  - [ ] Configure defaults for each mode

- [ ] Create shorthand configurations:
  ```python
  AUTONOMOUS_DEFAULTS = {
      "max_time": "24h",
      "cli": "auto",
      "watch_workspace": True,
      "enable_inbox": True,
      "alert_every": 10,
  }

  CONSERVATIVE_DEFAULTS = {
      "max_iterations": 100,
      "alert_every": 5,
      "pause_on_critical": True,
  }
  ```

- [ ] Improve `--status` output
  - [ ] Show progress bar
  - [ ] Show current task
  - [ ] Show recent alerts
  - [ ] Show CLI/model in use

### Phase 4: Documentation

- [ ] Update `docs/autonomous-loop.md`
  - [ ] Add usage examples
  - [ ] Document all flags
  - [ ] Add troubleshooting section

- [ ] Update CLAUDE.md
  - [ ] Document `--autonomous` mode
  - [ ] Document new capabilities

## Success Criteria

1. **Summarization**:
   - [ ] Context stays under 50k tokens after 100+ iterations
   - [ ] Summaries capture key progress points

2. **Branch Exploration**:
   - [ ] 3 branches execute in parallel
   - [ ] Winner is selected within 10 iterations per branch
   - [ ] Improves stuck recovery rate

3. **CLI**:
   - [ ] `--autonomous` works with sensible defaults
   - [ ] `--status` shows clear, useful information

## References

- `specs/features/autonomous-loop/spec.md` - Full specification
- `specs/features/autonomous-loop/roadmap.md` - Implementation phases
- `src/um_agent_coder/harness/autonomous/` - Existing code
- `src/um_agent_coder/harness/autonomous/recovery/` - Recovery components

## Existing Code Patterns

Look at these files for patterns:

1. **ProgressDetector** (`progress_detector.py`)
   - How signals are weighted
   - How scores are calculated

2. **RecoveryManager** (`recovery_manager.py`)
   - How recovery stages are ordered
   - How results are returned

3. **CLIRouter** (`cli_router.py`)
   - How task analysis works
   - How routing decisions are made

---

*Use this prompt with: `cat prompts/self-build/implement-autonomous-loop.md | claude-code`*
