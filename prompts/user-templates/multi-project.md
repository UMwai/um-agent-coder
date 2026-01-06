# Multi-Project Orchestration Template

Use this template when you need to coordinate work across multiple repositories or sub-projects using the meta-harness.

## Usage

1. Copy this template
2. Create sub-roadmaps for each project
3. Run with `python -m src.um_agent_coder.harness --meta --roadmap meta-roadmap.md`

---

# [PROJECT NAME] Meta-Roadmap

## Strategy

coordination: [parallel | pipeline | race | voting]
max_concurrent: [e.g., 5]
fail_fast: [true | false]

## Sub-Harnesses

### [sub-harness-1-name]

- working_dir: [path to sub-project]
- roadmap: [path to sub-project roadmap.md]
- cli: [codex | gemini | claude | auto]
- depends: [none | other-harness-name]

### [sub-harness-2-name]

- working_dir: [path]
- roadmap: [path]
- cli: auto
- depends: [none | list of dependencies]

### [sub-harness-3-name]

- working_dir: [path]
- roadmap: [path]
- cli: auto
- depends: [harness-1, harness-2]

## Aggregation

on_complete: [merge_all | pick_winner | human_review]
output_dir: [where to put final output]

---

## Example: Full Stack Application

```markdown
# E-Commerce Platform Meta-Roadmap

## Strategy

coordination: parallel
max_concurrent: 3
fail_fast: false

## Sub-Harnesses

### frontend-harness

- working_dir: ./frontend
- roadmap: frontend/roadmap.md
- cli: codex
- depends: none

  Context:
  - React 18 with TypeScript
  - Use Tailwind CSS
  - Connect to backend at localhost:8000

### backend-harness

- working_dir: ./backend
- roadmap: backend/roadmap.md
- cli: codex
- depends: none

  Context:
  - FastAPI with SQLAlchemy
  - PostgreSQL database
  - JWT authentication

### infra-harness

- working_dir: ./infrastructure
- roadmap: infrastructure/roadmap.md
- cli: gemini
- depends: frontend-harness, backend-harness

  Context:
  - Terraform for AWS
  - ECS Fargate for containers
  - RDS PostgreSQL
  - CloudFront for frontend

## Aggregation

on_complete: merge_all
output_dir: ./dist
```

---

## Example: Parallel Strategy Exploration

```markdown
# Caching Implementation Meta-Roadmap

## Strategy

coordination: race
max_concurrent: 3
fail_fast: false

## Sub-Harnesses

### redis-branch

- working_dir: ./experiments/redis
- roadmap: experiments/redis/roadmap.md
- cli: codex
- depends: none

  Goal: Implement caching using Redis
  Success: Response time < 100ms, cache hit rate > 80%

### memcached-branch

- working_dir: ./experiments/memcached
- roadmap: experiments/memcached/roadmap.md
- cli: codex
- depends: none

  Goal: Implement caching using Memcached
  Success: Response time < 100ms, cache hit rate > 80%

### in-memory-branch

- working_dir: ./experiments/inmemory
- roadmap: experiments/inmemory/roadmap.md
- cli: codex
- depends: none

  Goal: Implement caching using local LRU cache
  Success: Response time < 50ms (no network)

## Aggregation

on_complete: pick_winner
selection_criteria: best_performance
output_dir: ./src/cache
```

---

## Example: Pipeline (Sequential)

```markdown
# Documentation Pipeline Meta-Roadmap

## Strategy

coordination: pipeline
fail_fast: true

## Sub-Harnesses

### analyze-harness

- working_dir: ./
- roadmap: docs/analyze-roadmap.md
- cli: gemini
- depends: none

  Goal: Analyze codebase, identify all public APIs
  Output: docs/api-inventory.md

### generate-harness

- working_dir: ./
- roadmap: docs/generate-roadmap.md
- cli: codex
- depends: analyze-harness

  Goal: Generate API documentation from inventory
  Input: docs/api-inventory.md (from analyze-harness)
  Output: docs/api-reference.md

### review-harness

- working_dir: ./
- roadmap: docs/review-roadmap.md
- cli: claude
- depends: generate-harness

  Goal: Review and improve documentation quality
  Input: docs/api-reference.md
  Output: docs/api-reference-final.md

## Aggregation

on_complete: use_final
output_dir: ./docs
```

---

## Coordination Strategies

| Strategy | Use When | Behavior |
|----------|----------|----------|
| **parallel** | Independent sub-projects | All run simultaneously |
| **pipeline** | Sequential dependencies | Output feeds next input |
| **race** | Exploring alternatives | First success wins |
| **voting** | Need consensus | Multiple complete, pick best |

## Tips

1. **Use parallel** for independent projects (frontend + backend)
2. **Use pipeline** when order matters (analyze → generate → review)
3. **Use race** when exploring alternatives (redis vs memcached)
4. **Use voting** for critical decisions needing validation

---

*Template for um-agent-coder meta-harness*
