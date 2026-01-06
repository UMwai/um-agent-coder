# um-agent-coder Prompts

Actionable prompts for AI-driven development. These prompts reference the technical specifications in `specs/` and provide structured guidance for implementation.

## Directory Structure

```
prompts/
├── README.md                      # This file
├── self-build/                    # Prompts for building THIS repo
│   ├── implement-meta-harness.md  # Build the meta-harness feature
│   ├── implement-autonomous-loop.md
│   ├── implement-foundation.md
│   └── review-and-test.md
│
└── user-templates/                # Reference prompts for users
    ├── autonomous-task.md         # Template for autonomous task execution
    ├── multi-project.md           # Template for meta-harness usage
    └── research-then-build.md     # Research → implement pattern
```

## Usage

### For AI Agents Building This Repo

Use prompts in `self-build/` when implementing features:

```bash
# Example: Start meta-harness implementation
cat prompts/self-build/implement-meta-harness.md | claude-code
```

Each prompt:
1. References the relevant spec in `specs/`
2. Provides implementation checklist
3. Defines success criteria
4. Lists files to create/modify

### For Users Running the Harness

Use templates in `user-templates/` as starting points:

```bash
# Copy template to your project
cp prompts/user-templates/autonomous-task.md my-project/roadmap.md

# Edit for your use case
vim my-project/roadmap.md

# Run harness
python -m src.um_agent_coder.harness --roadmap my-project/roadmap.md
```

## Prompt Format

All prompts follow this structure:

```markdown
# [Feature/Task Name]

## Context
- Reference: specs/features/[feature]/spec.md
- Priority: [Critical/High/Medium]
- Estimated scope: [files to create/modify]

## Goal
[1-2 sentence goal statement]

## Spec Summary
[Key points from the spec - what must be built]

## Implementation Checklist
- [ ] Step 1: ...
- [ ] Step 2: ...
- [ ] Step N: ...

## Success Criteria
[How to verify completion]

## References
- specs/features/[feature]/spec.md
- [existing code files]
```

## Quick Start

| I want to... | Use this prompt |
|--------------|-----------------|
| Implement meta-harness | `self-build/implement-meta-harness.md` |
| Implement autonomous loop | `self-build/implement-autonomous-loop.md` |
| Run autonomous task on my project | `user-templates/autonomous-task.md` |
| Orchestrate multiple projects | `user-templates/multi-project.md` |

## Priority Order

When building features, follow this priority:

1. **Meta-Harness** (50%) - `implement-meta-harness.md`
2. **Autonomous Loop** (30%) - `implement-autonomous-loop.md`
3. **Foundation** (20%) - `implement-foundation.md`

---

*Last Updated: January 2026*
