#!/usr/bin/env python3
"""
Test script for parallel harness execution.

This creates a simple roadmap with independent tasks to verify parallel execution works.
"""

import tempfile
from pathlib import Path

# Create a test roadmap with parallel tasks
ROADMAP_CONTENT = """# Test Roadmap for Parallel Execution

## Objective
Test parallel task execution in the harness.

## Tasks

### Phase 1: Independent Tasks
- [ ] **task-001**: First independent task
  - timeout: 1min
  - depends: none
  - success: Task should complete

- [ ] **task-002**: Second independent task
  - timeout: 1min
  - depends: none
  - success: Task should complete

- [ ] **task-003**: Third independent task
  - timeout: 1min
  - depends: none
  - success: Task should complete

### Phase 2: Dependent Tasks
- [ ] **task-004**: Depends on all phase 1 tasks
  - timeout: 1min
  - depends: task-001, task-002, task-003
  - success: Task should complete after all dependencies

- [ ] **task-005**: Another dependent task
  - timeout: 1min
  - depends: task-001, task-002, task-003
  - success: Task should complete after all dependencies

### Phase 3: Sequential Task
- [ ] **task-006**: Final task
  - timeout: 1min
  - depends: task-004, task-005
  - success: Task should complete last

## Success Criteria
- All tasks complete successfully
- Parallel tasks (task-001, task-002, task-003) run concurrently
- Dependent tasks wait for dependencies
"""

def main():
    # Create temporary roadmap file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(ROADMAP_CONTENT)
        roadmap_path = f.name

    print(f"Created test roadmap at: {roadmap_path}")
    print("\nTest the implementation with these commands:\n")
    print("# Sequential execution (baseline):")
    print(f"python -m src.um_agent_coder.harness --roadmap {roadmap_path} --dry-run\n")
    print("# Parallel execution (new feature):")
    print(f"python -m src.um_agent_coder.harness --roadmap {roadmap_path} --parallel --dry-run\n")
    print("# Parallel with max workers:")
    print(f"python -m src.um_agent_coder.harness --roadmap {roadmap_path} --parallel --max-parallel 2 --dry-run\n")

    print("\nExpected behavior:")
    print("- Sequential: Executes tasks one at a time in dependency order")
    print("- Parallel: Executes task-001, task-002, task-003 concurrently")
    print("           Then executes task-004, task-005 concurrently")
    print("           Finally executes task-006")

    # Keep the file for manual testing
    print(f"\nRoadmap file saved at: {roadmap_path}")
    print("Delete it when done with: rm " + roadmap_path)

if __name__ == "__main__":
    main()
