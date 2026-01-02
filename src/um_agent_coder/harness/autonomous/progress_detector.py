"""Multi-signal progress detection system.

Detects progress between iterations using multiple signals:
- Output diff (30%): How different is current output from previous
- File changes (30%): Git diff showing actual code changes
- Explicit markers (25%): <progress>...</progress> tags in output
- Checklist progress (15%): Subtasks completed (if defined)

Reference: specs/autonomous-loop-spec.md Section 2
"""

import subprocess
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .progress_markers import extract_progress_markers


@dataclass
class ProgressSignal:
    """Signals used to calculate progress score."""

    output_diff_score: float = 0.0  # 0.0 = identical, 1.0 = completely different
    file_changes_score: float = 0.0  # 0.0 = no changes, 1.0 = significant changes
    explicit_markers: list[str] = field(default_factory=list)
    checklist_progress: float = 0.0  # 0.0 = none complete, 1.0 = all complete

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "output_diff_score": self.output_diff_score,
            "file_changes_score": self.file_changes_score,
            "explicit_markers": self.explicit_markers,
            "checklist_progress": self.checklist_progress,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgressSignal":
        """Deserialize from dictionary."""
        return cls(
            output_diff_score=data.get("output_diff_score", 0.0),
            file_changes_score=data.get("file_changes_score", 0.0),
            explicit_markers=data.get("explicit_markers", []),
            checklist_progress=data.get("checklist_progress", 0.0),
        )


# Default weights for progress signals (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "output_diff": 0.30,
    "file_changes": 0.30,
    "explicit_markers": 0.25,
    "checklist": 0.15,
}


def calculate_progress_score(
    signal: ProgressSignal,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Calculate overall progress score from signals.

    Args:
        signal: The progress signals to evaluate.
        weights: Optional custom weights. Defaults to DEFAULT_WEIGHTS.

    Returns:
        Progress score from 0.0 (no progress) to 1.0 (significant progress).
    """
    weights = weights or DEFAULT_WEIGHTS

    # Convert explicit markers to a score (each marker worth 0.5, capped at 1.0)
    marker_score = min(1.0, len(signal.explicit_markers) * 0.5)

    score = (
        weights["output_diff"] * signal.output_diff_score
        + weights["file_changes"] * signal.file_changes_score
        + weights["explicit_markers"] * marker_score
        + weights["checklist"] * signal.checklist_progress
    )

    return min(1.0, max(0.0, score))


def _normalize_output(text: str) -> str:
    """Normalize output for comparison.

    Strips whitespace, lowercases, and removes common non-semantic elements.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove timestamp-like patterns (ISO dates, Unix timestamps)
    import re

    # After lowercase, T becomes t
    text = re.sub(r"\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}", "", text)
    text = re.sub(r"\d{10,13}", "", text)  # Unix timestamps

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def output_diff_score(prev_output: str, curr_output: str) -> float:
    """Calculate how different current output is from previous.

    Args:
        prev_output: Output from previous iteration.
        curr_output: Output from current iteration.

    Returns:
        Score from 0.0 (identical) to 1.0 (completely different).
        Higher score = more progress.
    """
    if not prev_output:
        return 1.0  # First iteration always counts as progress

    if not curr_output:
        return 0.0  # Empty current output is no progress

    # Normalize for comparison
    prev_norm = _normalize_output(prev_output)
    curr_norm = _normalize_output(curr_output)

    if not prev_norm and not curr_norm:
        return 0.0

    # Calculate similarity ratio
    similarity = SequenceMatcher(None, prev_norm, curr_norm).ratio()

    # Convert similarity to difference score
    return 1.0 - similarity


@dataclass
class GitDiffStats:
    """Statistics from git diff."""

    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


def _parse_git_diff_stat(output: str) -> GitDiffStats:
    """Parse git diff --stat output.

    Example output:
        src/foo.py | 10 +++++-----
        src/bar.py |  5 +++++
        2 files changed, 10 insertions(+), 5 deletions(-)
    """
    stats = GitDiffStats()

    if not output:
        return stats

    lines = output.strip().split("\n")
    if not lines:
        return stats

    # Look for summary line at the end
    summary_line = lines[-1]

    import re

    # Match "X file(s) changed"
    files_match = re.search(r"(\d+) files? changed", summary_line)
    if files_match:
        stats.files_changed = int(files_match.group(1))

    # Match "X insertions(+)"
    ins_match = re.search(r"(\d+) insertions?\(\+\)", summary_line)
    if ins_match:
        stats.insertions = int(ins_match.group(1))

    # Match "X deletions(-)"
    del_match = re.search(r"(\d+) deletions?\(-\)", summary_line)
    if del_match:
        stats.deletions = int(del_match.group(1))

    return stats


def file_changes_score(
    workspace: Optional[Path] = None,
    compare_to: str = "HEAD",
) -> float:
    """Calculate progress score based on file changes.

    Uses git diff to detect changes in the workspace.

    Args:
        workspace: Path to the workspace directory. Defaults to cwd.
        compare_to: Git ref to compare against. Defaults to HEAD.

    Returns:
        Score from 0.0 (no changes) to 1.0 (significant changes).
        Normalized: 100+ lines changed = 1.0
    """
    workspace = workspace or Path.cwd()

    try:
        # Get diff stats for both staged and unstaged changes
        result = subprocess.run(
            ["git", "diff", "--stat", compare_to],
            capture_output=True,
            text=True,
            cwd=workspace,
            timeout=10,
        )

        if result.returncode != 0:
            return 0.0

        stats = _parse_git_diff_stat(result.stdout)

        # Also check staged changes
        staged_result = subprocess.run(
            ["git", "diff", "--stat", "--cached", compare_to],
            capture_output=True,
            text=True,
            cwd=workspace,
            timeout=10,
        )

        if staged_result.returncode == 0:
            staged_stats = _parse_git_diff_stat(staged_result.stdout)
            stats.insertions += staged_stats.insertions
            stats.deletions += staged_stats.deletions
            stats.files_changed += staged_stats.files_changed

        # Calculate score: normalize to 0-1 (100 lines = 1.0)
        total_changes = stats.insertions + stats.deletions
        return min(1.0, total_changes / 100)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0.0


class ProgressDetector:
    """Detect progress between iterations using multiple signals.

    Example:
        detector = ProgressDetector(workspace=Path("/my/project"))
        signal = detector.detect(
            prev_output="Working on step 1...",
            curr_output="<progress>Completed step 1</progress> Working on step 2..."
        )
        score = calculate_progress_score(signal)
        print(f"Progress: {score:.2f}")  # e.g., "Progress: 0.45"
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        weights: Optional[dict[str, float]] = None,
        no_progress_threshold: float = 0.15,
    ):
        """Initialize progress detector.

        Args:
            workspace: Path to workspace for file change detection.
            weights: Custom weights for signal combination.
            no_progress_threshold: Score below which is considered no progress.
        """
        self.workspace = workspace or Path.cwd()
        self.weights = weights or DEFAULT_WEIGHTS
        self.no_progress_threshold = no_progress_threshold

    def detect(
        self,
        prev_output: str,
        curr_output: str,
        checklist_completed: int = 0,
        checklist_total: int = 0,
        git_compare_to: str = "HEAD",
    ) -> ProgressSignal:
        """Detect progress signals from iteration outputs.

        Args:
            prev_output: Output from previous iteration.
            curr_output: Output from current iteration.
            checklist_completed: Number of checklist items completed.
            checklist_total: Total number of checklist items.
            git_compare_to: Git ref to compare file changes against.

        Returns:
            ProgressSignal with all detected signals.
        """
        # Calculate output diff
        diff_score = output_diff_score(prev_output, curr_output)

        # Calculate file changes
        file_score = file_changes_score(self.workspace, git_compare_to)

        # Extract explicit markers
        markers = extract_progress_markers(curr_output)

        # Calculate checklist progress
        checklist_score = 0.0
        if checklist_total > 0:
            checklist_score = checklist_completed / checklist_total

        return ProgressSignal(
            output_diff_score=diff_score,
            file_changes_score=file_score,
            explicit_markers=markers,
            checklist_progress=checklist_score,
        )

    def calculate_score(self, signal: ProgressSignal) -> float:
        """Calculate progress score from signal.

        Args:
            signal: The progress signal to score.

        Returns:
            Progress score from 0.0 to 1.0.
        """
        return calculate_progress_score(signal, self.weights)

    def has_progress(self, signal: ProgressSignal) -> bool:
        """Check if signal indicates meaningful progress.

        Args:
            signal: The progress signal to check.

        Returns:
            True if progress score is above threshold.
        """
        score = self.calculate_score(signal)
        return score >= self.no_progress_threshold

    def detect_and_score(
        self,
        prev_output: str,
        curr_output: str,
        checklist_completed: int = 0,
        checklist_total: int = 0,
        git_compare_to: str = "HEAD",
    ) -> tuple[ProgressSignal, float, bool]:
        """Detect signals, calculate score, and check for progress.

        Convenience method that combines detect(), calculate_score(),
        and has_progress().

        Args:
            prev_output: Output from previous iteration.
            curr_output: Output from current iteration.
            checklist_completed: Number of checklist items completed.
            checklist_total: Total number of checklist items.
            git_compare_to: Git ref to compare file changes against.

        Returns:
            Tuple of (signal, score, has_progress).
        """
        signal = self.detect(
            prev_output,
            curr_output,
            checklist_completed,
            checklist_total,
            git_compare_to,
        )
        score = self.calculate_score(signal)
        has_prog = score >= self.no_progress_threshold

        return signal, score, has_prog
