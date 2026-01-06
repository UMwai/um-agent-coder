# Progress Detection Specification

> **Priority**: MEDIUM (Foundation)
> **Status**: Implemented
> **Location**: `src/um_agent_coder/harness/autonomous/progress_detector.py`

## Overview

Progress detection determines whether an iteration made meaningful progress toward the goal. It uses a weighted combination of multiple signals.

## Progress Signals

### Signal Weights

| Signal | Weight | Description |
|--------|--------|-------------|
| Output Diff | 30% | How different is current output from previous |
| File Changes | 30% | Did files in workspace change (git diff) |
| Explicit Markers | 25% | Did output contain `<progress>...</progress>` |
| Checklist Progress | 15% | Did subtasks get checked off |

### Signal Details

#### Output Diff (30%)

Compares current iteration output to previous iteration.

```python
def output_diff_score(prev_output: str, curr_output: str) -> float:
    """Higher score = more different = more progress."""
    if not prev_output:
        return 1.0  # First iteration always has progress

    prev_norm = normalize(prev_output)
    curr_norm = normalize(curr_output)

    similarity = SequenceMatcher(None, prev_norm, curr_norm).ratio()
    return 1.0 - similarity  # Convert to difference
```

Scoring:
- 1.0 = Completely different (good)
- 0.0 = Identical (no progress)

#### File Changes (30%)

Detects file modifications via git.

```python
def file_changes_score(workspace: Path) -> float:
    result = subprocess.run(
        ["git", "diff", "--stat", "--cached", "HEAD~1"],
        capture_output=True, cwd=workspace
    )

    stats = parse_git_diff_stat(result.stdout)
    total_changes = stats.insertions + stats.deletions

    # Cap at 100 lines = 1.0
    return min(1.0, total_changes / 100)
```

Scoring:
- 1.0 = 100+ lines changed
- 0.5 = 50 lines changed
- 0.0 = No changes

#### Explicit Markers (25%)

Detects `<progress>...</progress>` tags in output.

```python
def extract_progress_markers(output: str) -> List[str]:
    pattern = r'<progress>(.*?)</progress>'
    return re.findall(pattern, output, re.DOTALL)

# Score based on number of markers
marker_score = min(1.0, len(markers) * 0.5)
```

Scoring:
- 1.0 = 2+ progress markers
- 0.5 = 1 progress marker
- 0.0 = No markers

#### Checklist Progress (15%)

Detects checklist items getting checked off.

```python
def checklist_progress(prev_output: str, curr_output: str) -> float:
    prev_checked = count_checked(prev_output)  # Count [x]
    curr_checked = count_checked(curr_output)

    if curr_checked > prev_checked:
        return 1.0
    return 0.0
```

## Score Calculation

```python
@dataclass
class ProgressSignal:
    output_diff_score: float
    file_changes_score: float
    explicit_markers: List[str]
    checklist_progress: float

def calculate_progress_score(signal: ProgressSignal) -> float:
    weights = {
        'output_diff': 0.30,
        'file_changes': 0.30,
        'explicit_markers': 0.25,
        'checklist': 0.15
    }

    marker_score = min(1.0, len(signal.explicit_markers) * 0.5)

    return (
        weights['output_diff'] * signal.output_diff_score +
        weights['file_changes'] * signal.file_changes_score +
        weights['explicit_markers'] * marker_score +
        weights['checklist'] * signal.checklist_progress
    )
```

## Thresholds

### No-Progress Threshold

```python
PROGRESS_THRESHOLD = 0.15  # Below this = no meaningful progress
```

A score < 0.15 means:
- Output is nearly identical to previous
- No file changes
- No progress markers
- No checklist progress

### Stuck Detection

Consecutive no-progress iterations trigger stuck recovery:

```python
STUCK_AFTER = 3  # Trigger after 3 no-progress iterations
```

## Configuration

### CLI Flags

```bash
--progress-threshold 0.15  # Minimum score for "progress"
--stuck-after 3            # Iterations before stuck recovery
```

### YAML Config

```yaml
loop:
  progress_threshold: 0.15
  stuck_after: 3
```

## Best Practices for Agents

To help progress detection work:

1. **Use progress markers**: Output `<progress>Completed X</progress>`
2. **Make incremental commits**: Commit changes frequently
3. **Be explicit**: Describe what was accomplished

Example good output:
```
<progress>Implemented login endpoint</progress>
<progress>Added password hashing</progress>

Next steps:
- [ ] Add rate limiting
- [x] Create user model
- [x] Implement login
```

## Edge Cases

| Case | Handling |
|------|----------|
| First iteration | Always 1.0 (no previous to compare) |
| Empty output | 0.0 for output_diff |
| No git repo | Skip file_changes, redistribute weight |
| Malformed markers | Ignore, score 0.0 |

---

*Last Updated: January 2026*
