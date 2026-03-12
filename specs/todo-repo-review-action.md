# TODO: UMClaw Repo Review & Advance Action

> Status: **Planned**
> Priority: **High** — this is the core missing piece for UMClaw to operate autonomously across repos

## Problem

UMClaw can collect events (GitHub, local repo scan, market signals) and has goals defined,
but it cannot yet **review an adjacent repo against its goal and generate executable tasks**
to move it forward. The pipeline has gaps between Orient → Decide → Act → Execute.

## What Works Today

```
[Observe]  → GitHub events, local repo scans, market/news signals     ✅
[Orient]   → Summarize events into signals                            ✅ (partial)
[Goals]    → Store/query goals with KPIs and constraints              ✅
[Cycle]    → Run OODA loop via /api/world-agent/cycle                 ✅
[PR]       → Create GitHub PRs via /api/world-agent/repos/.../pr      ✅
[Harness]  → Ralph loop executes roadmap tasks locally                ✅
```

## What's Missing

### 1. Repo Review Action (Decide Phase)
**Gap**: Orient collects events but Decide doesn't map signals → concrete tasks against a specific repo.

**Needed**: A "review repo" action that:
- Takes a repo path + goal ID as input
- Reads the repo's specs/roadmap, recent git history, test results, code structure
- Compares current state against goal KPIs
- Identifies the highest-impact gaps
- Outputs a prioritized task list

```
POST /api/world-agent/repos/{owner}/{repo}/review
{
  "goal_id": "hedge-fund-build",
  "repo_path": "/home/umwai/um_ai-hedge-fund",   // optional for local
  "depth": "standard"                              // quick | standard | deep
}

Response:
{
  "repo": "UMwai/um_ai-hedge-fund",
  "goal_id": "hedge-fund-build",
  "kpi_assessment": [...],
  "gaps": [...],
  "recommended_tasks": [
    {"id": "auto-001", "description": "...", "priority": 1, "estimated_effort": "medium"}
  ]
}
```

### 2. Task → Harness Bridge (Act Phase)
**Gap**: The daemon runs on Cloud Run but can't trigger local harness execution.

**Options** (pick one):
- **A) Webhook callback**: Daemon creates GitHub issue → local watcher picks it up → harness executes
- **B) Polling**: Local harness polls daemon for pending tasks → executes → reports back
- **C) Local daemon mode**: Run daemon locally alongside harness, share SQLite

**Recommended**: Option B (polling) — simplest, no infra changes.

```
# Local harness polls for tasks
GET /api/world-agent/tasks/pending?repo=UMwai/um_ai-hedge-fund

# Harness reports completion
POST /api/world-agent/tasks/{id}/complete
{ "output": "...", "pr_url": "..." }
```

### 3. Roadmap Generation (Act Phase)
**Gap**: No automatic conversion from review findings → harness-compatible roadmap.md

**Needed**: Generate a `specs/roadmap-{goal}.md` in the target repo that the harness can execute:

```markdown
## Tasks
- [ ] **auto-001**: Wire SignalCollector to live Yahoo Finance data
  - ralph: true
  - max_iterations: 30
  - timeout: 30min
  - success: SignalCollector returns real market data for SPY
  - cwd: /home/umwai/um_ai-hedge-fund
```

## Target Adjacent Repo: um-personal-budget

**Repo**: `/home/umwai/um-personal-budget`
**Stack**: FastAPI + React 19 + TypeScript + SQLite + Tailwind
**Current state**: Phase 1 complete (3 commits), Phase 2 planned (budgets, recurring, savings goals)
**Roadmap**: `specs/ROADMAP.md` — 4 phases defined, Phase 2 is next

### Proposed Goal

```yaml
id: personal-budget-phase2
name: Build Smart Budgeting Features
description: >
  Advance um-personal-budget from Phase 1 (core CRUD) to Phase 2
  (smart budgeting). Implement budget alerts, recurring transactions,
  savings goals, and transaction search.
priority: 2
status: planned
projects:
  - repo: UMwai/um-personal-budget
    role: Target project
kpis:
  - metric: phase2_features_complete
    target: "7/7 features from Phase 2 roadmap"
    current: "0/7"
  - metric: test_coverage
    target: "> 70%"
    current: "unknown"
  - metric: api_endpoints
    target: "all Phase 2 endpoints documented"
    current: "Phase 1 only"
constraints:
  - All changes must pass existing tests
  - Follow existing FastAPI + SQLAlchemy patterns
  - No new infrastructure dependencies
  - Privacy-first: no external API calls for core features
```

## Implementation Plan

### Step 1: Repo Review Endpoint
- Add `POST /api/world-agent/repos/{owner}/{repo}/review` to daemon
- Use Gemini to analyze repo structure, specs, git history against goal KPIs
- Return structured gap analysis with recommended tasks

### Step 2: Roadmap Generator
- Convert review output → harness-compatible `roadmap.md`
- Support Ralph loop task definitions with proper fields
- Write to target repo or return as API response

### Step 3: Task Polling Bridge
- Add `GET /api/world-agent/tasks/pending` endpoint
- Add `POST /api/world-agent/tasks/{id}/complete` endpoint
- Update local harness to optionally poll daemon for new work

### Step 4: End-to-End Flow
```
User: "UMClaw, review um-personal-budget and move it to Phase 2"
  ↓
[Daemon] POST /review → analyzes repo, gaps vs Phase 2 roadmap
  ↓
[Daemon] generates roadmap-personal-budget.md with Ralph tasks
  ↓
[Harness] picks up roadmap → Ralph loop executes each task
  ↓
[Harness] reports completion → Daemon updates goal KPIs
  ↓
[Daemon] creates PR with all changes
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `daemon/routes/world_agent/_reviewer.py` | Create | Repo review logic |
| `daemon/routes/world_agent/endpoints.py` | Modify | Add review + task polling endpoints |
| `daemon/routes/world_agent/_roadmap_gen.py` | Create | Roadmap.md generator |
| `harness/daemon_poller.py` | Create | Poll daemon for pending tasks |
| `harness/main.py` | Modify | Add `--poll-daemon` mode |
| `goals/personal-budget.yaml` | Create | Goal definition |
| `specs/roadmap-personal-budget.md` | Create | Generated harness roadmap |
