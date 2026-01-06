# um-agent-coder Specifications

Technical specifications for the um-agent-coder project, organized by feature priority.

## Priority Weighting

| Priority | Feature | Weight | Description |
|----------|---------|--------|-------------|
| **CRITICAL** | Meta-Harness | 50% | Harness managing harnesses - the core vision |
| **HIGH** | Autonomous Loop | 30% | 24/7 unattended execution with recovery |
| **MEDIUM** | Foundation | 20% | CLI routing, progress detection, etc. |

## Directory Structure

```
specs/
├── README.md                 # This file
├── architecture/             # Core system design
│   ├── overview.md           # System diagram, component map
│   └── interfaces.md         # Shared contracts (Executor, Router, etc.)
│
├── features/                 # Feature specifications
│   ├── meta-harness/         # PRIMARY (50%)
│   │   ├── spec.md           # Full technical specification
│   │   └── roadmap.md        # Phased implementation plan
│   │
│   ├── autonomous-loop/      # SECONDARY (30%)
│   │   ├── spec.md           # Autonomous execution spec
│   │   └── roadmap.md        # Implementation phases
│   │
│   └── foundation/           # TERTIARY (20%)
│       ├── cli-routing.md    # Multi-CLI routing
│       ├── progress-detection.md
│       ├── stuck-recovery.md
│       └── context-management.md
│
└── reference/                # Supporting documentation
    ├── cli-reference.md      # CLI flags and usage
    ├── config-schema.md      # Configuration file spec
    └── glossary.md           # Terms and definitions
```

## How to Read These Specs

### For Implementers

1. Start with `architecture/overview.md` to understand the system
2. Read `architecture/interfaces.md` for contracts you must implement
3. Read the feature spec you're implementing (`features/*/spec.md`)
4. Follow the roadmap for phased implementation

### For AI Agents

1. Reference specs in prompts: `See: specs/features/meta-harness/spec.md`
2. Use the prompts in `prompts/self-build/` which reference these specs
3. Each spec contains implementation details, not just requirements

### For Users

1. Read `architecture/overview.md` for system understanding
2. Check `reference/cli-reference.md` for usage
3. See `prompts/user-templates/` for example prompts

## Spec Format

Each feature spec follows this structure:

```markdown
# Feature Name

## Vision
1-2 sentence summary

## Use Cases
What problems this solves

## Core Concepts
Key abstractions and their responsibilities

## Architecture
How it fits into the system

## Implementation Details
Code structure, algorithms, data models

## Configuration
YAML/CLI configuration

## Implementation Phases
Ordered phases for incremental delivery
```

## Quick Links

| I want to... | Read this |
|--------------|-----------|
| Understand the system | [architecture/overview.md](architecture/overview.md) |
| Implement meta-harness | [features/meta-harness/spec.md](features/meta-harness/spec.md) |
| Implement autonomous loop | [features/autonomous-loop/spec.md](features/autonomous-loop/spec.md) |
| Add a new CLI backend | [architecture/interfaces.md](architecture/interfaces.md) |
| Configure the harness | [reference/config-schema.md](reference/config-schema.md) |

## Related Resources

- `prompts/` - Actionable prompts that reference these specs
- `docs/` - User documentation (how to use)
- `CLAUDE.md` - AI assistant instructions

---

*Last Updated: January 2026*
