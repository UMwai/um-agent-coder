"""Instruction queue for mid-loop directives.

Allows users to drop instructions into the loop while it's running
by placing files in the inbox directory.

Reference: specs/autonomous-loop-spec.md Section 4.3
"""

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional


class InstructionPriority(IntEnum):
    """Priority levels for instructions."""

    URGENT = 0  # Process immediately
    HIGH = 1  # Process before next iteration
    NORMAL = 5  # Process in order
    LOW = 9  # Process when convenient


@dataclass
class Instruction:
    """Represents a queued instruction."""

    id: str
    content: str
    priority: InstructionPriority = InstructionPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Instruction":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            priority=InstructionPriority(data.get("priority", InstructionPriority.NORMAL)),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_file=data.get("source_file"),
        )


class InstructionQueue:
    """Process queued instructions between iterations.

    Instructions are text files dropped into the inbox directory.
    Files are sorted by name to determine processing order.
    Processed files are moved to the processed subdirectory.

    File naming conventions:
    - 000-urgent.txt: Priority 0 (urgent)
    - 001-high.txt: Priority 1 (high)
    - instruction.txt: Priority 5 (normal, default)
    - zzz-low.txt: Priority 9 (low)

    Args:
        inbox_path: Path to the inbox directory.
        auto_create: Whether to create directories if missing.
    """

    def __init__(
        self,
        inbox_path: Path,
        auto_create: bool = True,
    ):
        """Initialize instruction queue."""
        self.inbox_path = Path(inbox_path)
        self.processed_path = self.inbox_path / "processed"

        if auto_create:
            self.inbox_path.mkdir(parents=True, exist_ok=True)
            self.processed_path.mkdir(parents=True, exist_ok=True)

    def poll(self) -> list[Instruction]:
        """Get pending instructions, sorted by priority.

        Returns:
            List of instructions sorted by priority (lowest first).
        """
        instructions = []

        if not self.inbox_path.exists():
            return instructions

        # Find all instruction files
        for file in self.inbox_path.glob("*.txt"):
            if file.is_file() and file.name != "processed":
                try:
                    content = file.read_text().strip()
                    if content:  # Skip empty files
                        instructions.append(
                            Instruction(
                                id=file.stem,
                                content=content,
                                priority=self._parse_priority(file.name),
                                source_file=str(file),
                            )
                        )
                except (OSError, UnicodeDecodeError):
                    continue

        # Sort by priority (lower number = higher priority)
        instructions.sort(key=lambda i: (i.priority, i.id))
        return instructions

    def poll_urgent(self) -> list[Instruction]:
        """Get only urgent instructions.

        Returns:
            List of urgent priority instructions.
        """
        all_instructions = self.poll()
        return [i for i in all_instructions if i.priority == InstructionPriority.URGENT]

    def has_pending(self) -> bool:
        """Check if there are pending instructions."""
        return len(self.poll()) > 0

    def has_urgent(self) -> bool:
        """Check if there are urgent instructions."""
        return len(self.poll_urgent()) > 0

    def mark_processed(self, instruction: Instruction) -> bool:
        """Move instruction to processed folder.

        Args:
            instruction: The instruction to mark as processed.

        Returns:
            True if successfully moved, False otherwise.
        """
        if not instruction.source_file:
            return False

        src = Path(instruction.source_file)
        if not src.exists():
            return False

        # Ensure processed directory exists
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Add timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self.processed_path / f"{timestamp}_{src.name}"

        try:
            shutil.move(str(src), str(dst))
            return True
        except (OSError, shutil.Error):
            return False

    def add_instruction(
        self,
        content: str,
        priority: InstructionPriority = InstructionPriority.NORMAL,
        instruction_id: Optional[str] = None,
    ) -> Instruction:
        """Add a new instruction to the queue.

        Args:
            content: The instruction content.
            priority: Priority level.
            instruction_id: Optional custom ID.

        Returns:
            The created instruction.
        """
        # Ensure inbox exists
        self.inbox_path.mkdir(parents=True, exist_ok=True)

        # Generate ID if not provided
        if instruction_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            instruction_id = f"{priority:03d}-{timestamp}"

        # Write to file
        filename = f"{instruction_id}.txt"
        file_path = self.inbox_path / filename
        file_path.write_text(content)

        return Instruction(
            id=instruction_id,
            content=content,
            priority=priority,
            source_file=str(file_path),
        )

    def clear_processed(self, older_than_days: int = 7) -> int:
        """Clear old processed instructions.

        Args:
            older_than_days: Remove files older than this many days.

        Returns:
            Number of files removed.
        """
        if not self.processed_path.exists():
            return 0

        removed = 0
        cutoff = datetime.now().timestamp() - (older_than_days * 86400)

        for file in self.processed_path.glob("*.txt"):
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
                    removed += 1
            except OSError:
                continue

        return removed

    def _parse_priority(self, filename: str) -> InstructionPriority:
        """Parse priority from filename.

        Convention:
        - Files starting with 000- are urgent
        - Files starting with 001- are high priority
        - Files starting with 00X- where X < 5 are high
        - Files starting with 00X- where X >= 5 are normal
        - Files starting with zzz or 9XX are low
        - Everything else is normal priority
        """
        name = filename.lower()

        # Check for urgent prefix
        if name.startswith("000"):
            return InstructionPriority.URGENT

        # Check for numeric prefix
        if len(name) >= 3 and name[:3].isdigit():
            prefix_num = int(name[:3])
            if prefix_num <= 1:
                return InstructionPriority.HIGH
            elif prefix_num >= 900:
                return InstructionPriority.LOW
            elif prefix_num < 5:
                return InstructionPriority.HIGH

        # Check for zzz low priority marker
        if name.startswith("zzz") or name.startswith("low"):
            return InstructionPriority.LOW

        # Check for urgent keyword
        if "urgent" in name:
            return InstructionPriority.URGENT

        # Check for high keyword
        if "high" in name:
            return InstructionPriority.HIGH

        return InstructionPriority.NORMAL

    def get_queue_status(self) -> dict:
        """Get status of the instruction queue.

        Returns:
            Dictionary with queue statistics.
        """
        instructions = self.poll()

        status = {
            "total_pending": len(instructions),
            "urgent": sum(1 for i in instructions if i.priority == InstructionPriority.URGENT),
            "high": sum(1 for i in instructions if i.priority == InstructionPriority.HIGH),
            "normal": sum(1 for i in instructions if i.priority == InstructionPriority.NORMAL),
            "low": sum(1 for i in instructions if i.priority == InstructionPriority.LOW),
        }

        # Count processed
        if self.processed_path.exists():
            status["processed"] = len(list(self.processed_path.glob("*.txt")))
        else:
            status["processed"] = 0

        return status
