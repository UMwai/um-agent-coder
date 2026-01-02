"""
Roadmap parser for specs/roadmap.md format.

Parses markdown roadmap files into structured Roadmap objects.
Supports ralph loop task definitions with special fields.
"""

import re
from pathlib import Path

from .models import Phase, RalphConfig, Roadmap, Task, TaskStatus


class RoadmapParser:
    """Parse roadmap.md files into structured data."""

    def __init__(self, roadmap_path: str):
        self.roadmap_path = Path(roadmap_path)

    def parse(self) -> Roadmap:
        """Parse the roadmap file and return a Roadmap object."""
        if not self.roadmap_path.exists():
            raise FileNotFoundError(f"Roadmap not found: {self.roadmap_path}")

        content = self.roadmap_path.read_text()
        return self._parse_content(content)

    def _parse_content(self, content: str) -> Roadmap:
        """Parse markdown content into Roadmap."""
        # Extract project name from title
        name_match = re.search(r'^#\s+Roadmap:\s*(.+)$', content, re.MULTILINE)
        name = name_match.group(1).strip() if name_match else "Unnamed Project"

        # Extract objective
        objective = self._extract_section(content, "Objective")

        # Extract constraints
        constraints = self._extract_section(content, "Constraints")
        max_time, max_retries, working_dir = self._parse_constraints(constraints)

        # Extract success criteria
        success_section = self._extract_section(content, "Success Criteria")
        success_criteria = self._parse_checklist(success_section)

        # Extract phases and tasks
        phases = self._extract_phases(content)

        # Extract growth instructions
        growth_section = self._extract_section(content, "Growth Mode")
        growth_instructions = self._parse_numbered_list(growth_section)

        return Roadmap(
            name=name,
            objective=objective,
            success_criteria=success_criteria,
            phases=phases,
            growth_instructions=growth_instructions,
            max_time_per_task=max_time,
            max_retries=max_retries,
            working_directory=working_dir,
        )

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract content under a ## heading until the next ## or end."""
        # Find the section header
        header_pattern = rf'^##\s+{re.escape(section_name)}\s*$'
        header_match = re.search(header_pattern, content, re.MULTILINE)
        if not header_match:
            return ""

        # Find content after header until next ## section or end
        start = header_match.end()
        rest = content[start:]

        # Find next ## section (not ###)
        next_section = re.search(r'^## ', rest, re.MULTILINE)
        if next_section:
            return rest[:next_section.start()].strip()
        return rest.strip()

    def _parse_constraints(self, constraints: str) -> tuple[int, int, str]:
        """Parse constraints section for time, retries, working dir."""
        max_time = 30  # default
        max_retries = 3  # default
        working_dir = "./"  # default

        for line in constraints.split('\n'):
            line = line.strip().lower()
            if 'max time' in line or 'timeout' in line:
                time_match = re.search(r'(\d+)\s*min', line)
                if time_match:
                    max_time = int(time_match.group(1))
            elif 'retries' in line:
                retry_match = re.search(r'(\d+)', line)
                if retry_match:
                    max_retries = int(retry_match.group(1))
            elif 'working directory' in line or 'cwd' in line:
                dir_match = re.search(r':\s*(.+)$', line, re.IGNORECASE)
                if dir_match:
                    working_dir = dir_match.group(1).strip()

        return max_time, max_retries, working_dir

    def _parse_checklist(self, section: str) -> list[str]:
        """Parse markdown checklist items (- [ ] or - [x])."""
        items = []
        for line in section.split('\n'):
            match = re.match(r'^-\s*\[[ x]\]\s*(.+)$', line.strip())
            if match:
                items.append(match.group(1).strip())
        return items

    def _parse_numbered_list(self, section: str) -> list[str]:
        """Parse numbered list items."""
        items = []
        for line in section.split('\n'):
            match = re.match(r'^\d+\.\s*(.+)$', line.strip())
            if match:
                items.append(match.group(1).strip())
        return items

    def _extract_phases(self, content: str) -> list[Phase]:
        """Extract all phases (### Phase X: Name) and their tasks."""
        phases = []

        # Find the Tasks section first
        tasks_section = self._extract_section(content, "Tasks")
        if not tasks_section:
            return phases

        # Split by ### headers for phases
        phase_pattern = r'^###\s+(?:Phase\s+\d+[:\s]*)?(.+?)$'
        phase_splits = re.split(phase_pattern, tasks_section, flags=re.MULTILINE)

        # phase_splits will be: ['', 'Phase Name 1', 'content1', 'Phase Name 2', 'content2', ...]
        for i in range(1, len(phase_splits), 2):
            phase_name = phase_splits[i].strip()
            phase_content = phase_splits[i + 1] if i + 1 < len(phase_splits) else ""

            tasks = self._parse_tasks(phase_content, phase_name)
            phases.append(Phase(name=phase_name, tasks=tasks))

        return phases

    def _parse_tasks(self, content: str, phase_name: str) -> list[Task]:
        """Parse tasks from phase content."""
        tasks = []

        # Pattern for task lines: - [ ] **task-001**: Description
        task_pattern = r'^-\s*\[[ x]\]\s*\*\*([^*]+)\*\*:\s*(.+?)(?=^-\s*\[|\Z)'
        matches = re.finditer(task_pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            task_id = match.group(1).strip()
            task_block = match.group(2).strip()

            # First line is description, rest are properties
            lines = task_block.split('\n')
            description = lines[0].strip()

            # Parse properties (indented lines with - property: value)
            props = self._parse_task_properties('\n'.join(lines[1:]))

            # Check if task is already marked complete
            is_complete = '[x]' in match.group(0).lower()

            # Build ralph config if enabled
            ralph_config = None
            if props.get('ralph', False):
                ralph_config = RalphConfig(
                    enabled=True,
                    max_iterations=props.get('max_iterations', 30),
                    completion_promise=props.get('completion_promise', 'COMPLETE'),
                )

            task = Task(
                id=task_id,
                description=description,
                phase=phase_name,
                depends=props.get('depends', []),
                timeout_minutes=props.get('timeout', 30),
                success_criteria=props.get('success', ''),
                cwd=props.get('cwd', './'),
                cli=props.get('cli', ''),
                model=props.get('model', ''),
                status=TaskStatus.COMPLETED if is_complete else TaskStatus.PENDING,
                ralph_config=ralph_config,
            )
            tasks.append(task)

        return tasks

    def _parse_task_properties(self, content: str) -> dict:
        """Parse task properties from indented lines.

        Supports ralph-specific fields:
        - ralph: true/false - Enable ralph loop for this task
        - max_iterations: N - Override default max iterations
        - completion_promise: TEXT - Custom promise text
        """
        props = {
            'depends': [],
            'timeout': 30,
            'success': '',
            'cwd': './',
            'cli': '',
            'model': '',
            # Ralph-specific properties
            'ralph': False,
            'max_iterations': 30,
            'completion_promise': 'COMPLETE',
        }

        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith('-'):
                continue

            line = line[1:].strip()  # Remove leading -

            if line.startswith('timeout:'):
                time_match = re.search(r'(\d+)', line)
                if time_match:
                    props['timeout'] = int(time_match.group(1))
            elif line.startswith('depends:'):
                deps = line.split(':', 1)[1].strip()
                if deps.lower() != 'none':
                    props['depends'] = [d.strip() for d in deps.split(',')]
            elif line.startswith('success:'):
                props['success'] = line.split(':', 1)[1].strip()
            elif line.startswith('cwd:'):
                props['cwd'] = line.split(':', 1)[1].strip()
            elif line.startswith('cli:'):
                props['cli'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('model:'):
                props['model'] = line.split(':', 1)[1].strip()
            # Ralph-specific fields
            elif line.startswith('ralph:'):
                value = line.split(':', 1)[1].strip().lower()
                props['ralph'] = value in ('true', 'yes', '1')
            elif line.startswith('max_iterations:'):
                iter_match = re.search(r'(\d+)', line)
                if iter_match:
                    props['max_iterations'] = int(iter_match.group(1))
            elif line.startswith('completion_promise:'):
                props['completion_promise'] = line.split(':', 1)[1].strip()

        return props

    def update_task_status(self, task_id: str, completed: bool = True) -> None:
        """Update a task's checkbox status in the roadmap file."""
        content = self.roadmap_path.read_text()

        # Find and update the checkbox for this task
        if completed:
            new_content = re.sub(
                rf'\[ \](\s*\*\*{re.escape(task_id)}\*\*)',
                r'[x]\1',
                content
            )
        else:
            new_content = re.sub(
                rf'\[x\](\s*\*\*{re.escape(task_id)}\*\*)',
                r'[ ]\1',
                content,
                flags=re.IGNORECASE
            )

        if new_content != content:
            self.roadmap_path.write_text(new_content)

    def append_growth_task(self, task: Task) -> None:
        """Append a new task generated during growth mode."""
        content = self.roadmap_path.read_text()

        # Find the last task in the last phase
        new_task_md = f"""
- [ ] **{task.id}**: {task.description}
  - timeout: {task.timeout_minutes}min
  - depends: {', '.join(task.depends) if task.depends else 'none'}
  - success: {task.success_criteria}
"""

        # Append before Growth Mode section if exists, otherwise at end
        if '## Growth Mode' in content:
            content = content.replace('## Growth Mode', f'{new_task_md}\n## Growth Mode')
        else:
            content += f'\n{new_task_md}'

        self.roadmap_path.write_text(content)
