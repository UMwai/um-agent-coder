# World-Aware Agent — Implementation Roadmap

> **Reference**: [spec.md](spec.md)
> **Status**: Planning

## Phase 1: Planner Skeleton + Single Collector

**Goal**: Cloud Run planner runs OODA cycles with one event source.

### Tasks
- [ ] **wa-001**: Data models (`Event`, `Goal`, `WorldState`, `PlannedTask`, `Signal`)
  - Pydantic models in `src/um_agent_coder/world_agent/models.py`
- [ ] **wa-002**: Goal store (load YAML goals, persist to Firestore)
  - `src/um_agent_coder/world_agent/goals.py`
- [ ] **wa-003**: Event collector base class + GitHub Events collector
  - `src/um_agent_coder/world_agent/collectors/base.py`
  - `src/um_agent_coder/world_agent/collectors/github.py`
- [ ] **wa-004**: Orientation layer (LLM filters events → signals)
  - `src/um_agent_coder/world_agent/orient.py`
- [ ] **wa-005**: FastAPI planner service (`POST /cycle`, `GET /status`)
  - `src/um_agent_coder/world_agent/planner.py`
- [ ] **wa-006**: Deploy planner to Cloud Run + Cloud Scheduler heartbeat
- [ ] **wa-007**: Firestore integration (events, world state, system metadata)

### Acceptance
- Planner runs on Cloud Run, collects GitHub events, filters them against a goal
- `/status` shows current world state
- Cloud Scheduler triggers `/cycle` every 15 min

---

## Phase 2: Decision Layer + Worker Dispatch

**Goal**: Planner generates tasks and spawns Cloud Run Job workers.

### Tasks
- [ ] **wa-008**: Decision layer (LLM generates `PlannedTask` from signals)
  - `src/um_agent_coder/world_agent/decide.py`
- [ ] **wa-009**: Worker dispatcher (spawn Cloud Run Jobs via API)
  - `src/um_agent_coder/world_agent/dispatch.py`
- [ ] **wa-010**: Worker container (runs iteration loop with injected task)
  - `src/um_agent_coder/world_agent/worker/main.py`
  - Dockerfile for worker image
- [ ] **wa-011**: Worker → Firestore progress reporting
- [ ] **wa-012**: Planner monitors workers, collects results on each cycle
- [ ] **wa-013**: Task queue management (priority, dedup, dependency resolution)

### Acceptance
- Planner detects a relevant event, creates a task, dispatches a worker
- Worker runs iteration loop, reports progress to Firestore
- Planner sees completion and updates goal progress

---

## Phase 3: Full Collector Suite

**Goal**: Monitor financial, dev, and news sources.

### Tasks
- [ ] **wa-014**: SEC EDGAR collector (10-K, 10-Q, 8-K filings)
- [ ] **wa-015**: Yahoo Finance collector (price alerts, volume spikes)
- [ ] **wa-016**: Earnings calendar collector
- [ ] **wa-017**: Dependency watch collector (PyPI, npm new versions)
- [ ] **wa-018**: News/RSS collector (configurable feed list)
- [ ] **wa-019**: AI research collector (ArXiv, HuggingFace releases)
- [ ] **wa-020**: CI/CD status collector (GitHub Actions across repos)
- [ ] **wa-021**: Collector registry + dynamic enable/disable per goal

### Acceptance
- All collectors running, events flowing to Firestore
- Orientation layer handles mixed event types
- Goal-specific event routing works

---

## Phase 4: Adaptive Scheduler + Cost Control

**Goal**: Smart cycle frequency and budget enforcement.

### Tasks
- [ ] **wa-022**: Adaptive scheduler (market hours, event volume, urgency)
- [ ] **wa-023**: Cost tracking per LLM call, per component, per goal
- [ ] **wa-024**: Daily budget enforcement (pause at limit)
- [ ] **wa-025**: Alert system (Discord webhook for budget + critical events)
- [ ] **wa-026**: Goal KPI tracking (snapshots over time)
- [ ] **wa-027**: `/costs` endpoint + dashboard data

### Acceptance
- Cycle frequency adapts (faster during market hours, slower overnight)
- System pauses when daily budget exceeded
- Discord alerts for critical events and budget warnings

---

## Phase 5: Multi-Project & Cross-Repo

**Goal**: Workers handle arbitrary repos, share context.

### Tasks
- [ ] **wa-028**: Worker container supports dynamic repo cloning
- [ ] **wa-029**: Cross-project context passing via Firestore
- [ ] **wa-030**: Dependency-aware scheduling across projects
- [ ] **wa-031**: Project-level resource quotas
- [ ] **wa-032**: Coordinated multi-project operations

### Acceptance
- Planner dispatches workers to different repos
- Workers in Project A can surface context to Project B via planner
- Task dependencies across projects are respected

---

## Phase 6: Learning & Self-Improvement

**Goal**: Agent improves its own decision-making over time.

### Tasks
- [ ] **wa-033**: Outcome tracking (task result vs goal KPI impact)
- [ ] **wa-034**: Strategy learning (which signals → which actions → which outcomes)
- [ ] **wa-035**: Collector tuning (adjust frequency based on signal-to-noise ratio)
- [ ] **wa-036**: Goal auto-refinement (suggest KPI/constraint updates)
- [ ] **wa-037**: Self-evaluation reports (weekly summary of decisions + outcomes)

### Acceptance
- Agent can show "these signal types lead to good outcomes"
- Collector frequencies adjusted based on actual utility
- Weekly report summarizing decisions, outcomes, and suggested improvements
