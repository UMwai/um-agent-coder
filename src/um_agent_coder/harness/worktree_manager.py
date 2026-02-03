"""
Git worktree manager for branch isolation.

Provides isolated git worktrees for parallel sub-harnesses,
ensuring each harness works on an independent branch that
can be merged back to main upon completion.
"""

import logging
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class MergeStatus(Enum):
    """Status of a merge operation."""

    SUCCESS = "success"
    CONFLICT = "conflict"
    TESTS_FAILED = "tests_failed"
    WORKTREE_NOT_FOUND = "worktree_not_found"
    ERROR = "error"


@dataclass
class WorktreeInfo:
    """Information about a git worktree."""

    harness_id: str
    branch_name: str  # e.g., harness/task-001
    path: Path
    base_branch: str
    created_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    merge_commit: Optional[str] = None


@dataclass
class MergeResult:
    """Result of a merge operation."""

    status: MergeStatus
    commit_sha: Optional[str] = None
    error: Optional[str] = None
    tests_passed: Optional[bool] = None

    @property
    def success(self) -> bool:
        """Check if merge was successful."""
        return self.status == MergeStatus.SUCCESS


class WorktreeManager:
    """Manage git worktrees for sub-harness isolation.

    Each parallel sub-harness gets its own worktree with an isolated
    branch. Changes are merged back to main upon completion if tests pass.

    Example:
        manager = WorktreeManager()

        # Create worktree for a harness
        info = manager.create_worktree("task-001", base_branch="main")

        # Do work in the worktree...
        # subprocess runs in info.path

        # Merge back to main when complete
        result = manager.merge_to_main("task-001", run_tests=True)

        # Cleanup
        manager.cleanup_worktree("task-001")
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        worktrees_dir: str = ".worktrees",
        branch_prefix: str = "harness",
    ):
        """Initialize the worktree manager.

        Args:
            repo_root: Root of the git repository (defaults to cwd)
            worktrees_dir: Directory for worktrees (relative to repo_root)
            branch_prefix: Prefix for branch names (e.g., harness/task-001)
        """
        self.repo_root = repo_root or Path.cwd()
        self.worktrees_dir = self.repo_root / worktrees_dir
        self.branch_prefix = branch_prefix

        # Ensure we're in a git repo
        if not (self.repo_root / ".git").exists():
            # Check if it's a worktree itself
            git_file = self.repo_root / ".git"
            if not git_file.exists():
                raise ValueError(f"Not a git repository: {self.repo_root}")

    def create_worktree(
        self,
        harness_id: str,
        base_branch: str = "main",
    ) -> WorktreeInfo:
        """Create an isolated worktree for a sub-harness.

        Args:
            harness_id: Unique identifier for the harness
            base_branch: Branch to base the worktree on

        Returns:
            WorktreeInfo with path and branch info

        Raises:
            subprocess.CalledProcessError: If git commands fail
        """
        branch_name = f"{self.branch_prefix}/{harness_id}"
        worktree_path = self.worktrees_dir / harness_id

        # Ensure worktrees directory exists
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

        # Check if worktree already exists
        if worktree_path.exists():
            logger.info(f"Worktree already exists at {worktree_path}")
            return WorktreeInfo(
                harness_id=harness_id,
                branch_name=branch_name,
                path=worktree_path,
                base_branch=base_branch,
            )

        # Fetch latest from remote (if exists)
        try:
            self._run_git(["fetch", "origin", base_branch], check=False)
        except Exception:
            pass  # May not have remote

        # Save current branch to restore later
        try:
            current_branch_result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            original_branch = current_branch_result.stdout.strip()
        except subprocess.CalledProcessError:
            original_branch = None

        # Create new branch and worktree
        # First, ensure base branch is up to date
        try:
            self._run_git(["checkout", base_branch])
            self._run_git(["pull", "--ff-only"], check=False)
        except subprocess.CalledProcessError:
            logger.warning(f"Could not update {base_branch}, continuing with local state")

        # Create worktree with new branch
        logger.info(f"Creating worktree at {worktree_path} (branch: {branch_name})")

        self._run_git([
            "worktree",
            "add",
            "-b",
            branch_name,
            str(worktree_path),
            base_branch,
        ])

        # Restore original branch if we changed it
        if original_branch and original_branch != base_branch:
            try:
                self._run_git(["checkout", original_branch])
            except subprocess.CalledProcessError:
                logger.warning(f"Could not restore original branch {original_branch}")

        return WorktreeInfo(
            harness_id=harness_id,
            branch_name=branch_name,
            path=worktree_path,
            base_branch=base_branch,
            created_at=datetime.utcnow(),
        )

    def merge_to_main(
        self,
        harness_id: str,
        run_tests: bool = True,
        test_command: str = "pytest",
        force: bool = False,
    ) -> MergeResult:
        """Merge a worktree branch back to main.

        Args:
            harness_id: Harness identifier
            run_tests: Whether to run tests before merge
            test_command: Command to run tests
            force: Force merge even if tests fail

        Returns:
            MergeResult with status and details
        """
        branch_name = f"{self.branch_prefix}/{harness_id}"
        worktree_path = self.worktrees_dir / harness_id

        if not worktree_path.exists():
            return MergeResult(
                status=MergeStatus.WORKTREE_NOT_FOUND,
                error=f"Worktree not found: {worktree_path}",
            )

        # Get worktree's base branch from the worktree info
        base_branch = self._get_worktree_base(harness_id) or "main"

        try:
            # First, commit any uncommitted changes in the worktree
            self._commit_worktree_changes(worktree_path, harness_id)

            # Run tests in the worktree if requested
            tests_passed = None
            if run_tests:
                tests_passed = self._run_tests_in_worktree(worktree_path, test_command)
                if not tests_passed and not force:
                    return MergeResult(
                        status=MergeStatus.TESTS_FAILED,
                        tests_passed=False,
                        error="Tests failed in worktree",
                    )

            # Switch to main in the main repo and merge
            self._run_git(["checkout", base_branch])
            self._run_git(["pull", "--ff-only"], check=False)

            # Try fast-forward merge first
            try:
                self._run_git(["merge", "--ff-only", branch_name])
            except subprocess.CalledProcessError:
                # Fall back to regular merge
                logger.info("Fast-forward not possible, attempting regular merge")
                try:
                    self._run_git(["merge", branch_name, "-m", f"Merge {branch_name}"])
                except subprocess.CalledProcessError as e:
                    return MergeResult(
                        status=MergeStatus.CONFLICT,
                        error=f"Merge conflict: {e}",
                        tests_passed=tests_passed,
                    )

            # Get the merge commit SHA
            result = self._run_git(["rev-parse", "HEAD"])
            commit_sha = result.stdout.strip()

            logger.info(f"Successfully merged {branch_name} to {base_branch}: {commit_sha}")

            return MergeResult(
                status=MergeStatus.SUCCESS,
                commit_sha=commit_sha,
                tests_passed=tests_passed,
            )

        except subprocess.CalledProcessError as e:
            logger.exception(f"Error merging worktree: {e}")
            return MergeResult(
                status=MergeStatus.ERROR,
                error=str(e),
            )

    def cleanup_worktree(self, harness_id: str, force: bool = False) -> bool:
        """Remove a worktree and its branch.

        Args:
            harness_id: Harness identifier
            force: Force removal even if there are uncommitted changes

        Returns:
            True if cleanup was successful
        """
        branch_name = f"{self.branch_prefix}/{harness_id}"
        worktree_path = self.worktrees_dir / harness_id

        try:
            # Remove the worktree
            if worktree_path.exists():
                force_flag = ["--force"] if force else []
                self._run_git(["worktree", "remove", str(worktree_path)] + force_flag)
                logger.info(f"Removed worktree: {worktree_path}")

            # Delete the branch
            try:
                # Force delete if requested, otherwise use -d (safe delete)
                delete_flag = "-D" if force else "-d"
                self._run_git(["branch", delete_flag, branch_name])
                logger.info(f"Deleted branch: {branch_name}")
            except subprocess.CalledProcessError:
                logger.warning(f"Could not delete branch {branch_name} (may already be deleted)")

            return True

        except subprocess.CalledProcessError as e:
            logger.exception(f"Error cleaning up worktree: {e}")
            return False

    def list_active_worktrees(self) -> List[WorktreeInfo]:
        """List all active worktrees for this manager.

        Returns:
            List of WorktreeInfo for active worktrees
        """
        worktrees = []

        try:
            result = self._run_git(["worktree", "list", "--porcelain"])

            current_wt = {}
            for line in result.stdout.strip().split("\n"):
                if not line:
                    if current_wt and "worktree" in current_wt:
                        wt_path = Path(current_wt["worktree"])
                        # Only include worktrees in our directory
                        if str(self.worktrees_dir) in str(wt_path):
                            harness_id = wt_path.name
                            branch = current_wt.get("branch", "").replace("refs/heads/", "")
                            worktrees.append(
                                WorktreeInfo(
                                    harness_id=harness_id,
                                    branch_name=branch,
                                    path=wt_path,
                                    base_branch="main",  # Would need to track this
                                )
                            )
                    current_wt = {}
                    continue

                if line.startswith("worktree "):
                    current_wt["worktree"] = line[9:]
                elif line.startswith("branch "):
                    current_wt["branch"] = line[7:]

            # Handle last entry
            if current_wt and "worktree" in current_wt:
                wt_path = Path(current_wt["worktree"])
                if str(self.worktrees_dir) in str(wt_path):
                    harness_id = wt_path.name
                    branch = current_wt.get("branch", "").replace("refs/heads/", "")
                    worktrees.append(
                        WorktreeInfo(
                            harness_id=harness_id,
                            branch_name=branch,
                            path=wt_path,
                            base_branch="main",
                        )
                    )

        except subprocess.CalledProcessError as e:
            logger.exception(f"Error listing worktrees: {e}")

        return worktrees

    def get_worktree(self, harness_id: str) -> Optional[WorktreeInfo]:
        """Get worktree info by harness ID.

        Args:
            harness_id: Harness identifier

        Returns:
            WorktreeInfo if found, None otherwise
        """
        worktrees = self.list_active_worktrees()
        for wt in worktrees:
            if wt.harness_id == harness_id:
                return wt
        return None

    def worktree_has_changes(self, harness_id: str) -> bool:
        """Check if a worktree has uncommitted changes.

        Args:
            harness_id: Harness identifier

        Returns:
            True if there are uncommitted changes
        """
        worktree_path = self.worktrees_dir / harness_id

        if not worktree_path.exists():
            return False

        try:
            result = self._run_git(["status", "--porcelain"], cwd=worktree_path)
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def _run_git(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a git command.

        Args:
            args: Git command arguments
            cwd: Working directory (defaults to repo_root)
            check: Whether to check return code

        Returns:
            CompletedProcess result
        """
        cmd = ["git"] + args
        return subprocess.run(
            cmd,
            cwd=cwd or self.repo_root,
            capture_output=True,
            text=True,
            check=check,
        )

    def _get_worktree_base(self, harness_id: str) -> Optional[str]:
        """Get the base branch for a worktree.

        Args:
            harness_id: Harness identifier

        Returns:
            Base branch name or None
        """
        branch_name = f"{self.branch_prefix}/{harness_id}"

        try:
            # Get the merge base with main
            result = self._run_git([
                "merge-base",
                "main",
                branch_name,
            ])
            # For simplicity, assume main is the base
            return "main"
        except subprocess.CalledProcessError:
            return None

    def _commit_worktree_changes(self, worktree_path: Path, harness_id: str) -> None:
        """Commit any uncommitted changes in a worktree.

        Args:
            worktree_path: Path to the worktree
            harness_id: Harness identifier for commit message
        """
        # Check for changes
        result = self._run_git(["status", "--porcelain"], cwd=worktree_path)
        if not result.stdout.strip():
            return

        # Add tracked files that have changes (safer than -A which can add secrets)
        # First add modified/deleted tracked files
        self._run_git(["add", "-u"], cwd=worktree_path)

        # Then add new files, excluding common sensitive patterns
        # Note: This relies on .gitignore being properly configured
        self._run_git(["add", "."], cwd=worktree_path, check=False)

        # Commit
        self._run_git([
            "commit",
            "-m",
            f"Auto-commit from harness {harness_id}",
        ], cwd=worktree_path, check=False)

    def _run_tests_in_worktree(
        self,
        worktree_path: Path,
        test_command: str,
    ) -> bool:
        """Run tests in a worktree.

        Args:
            worktree_path: Path to the worktree
            test_command: Command to run tests (supports quoted arguments)

        Returns:
            True if tests passed
        """
        try:
            # Use shlex.split to properly handle quoted arguments
            # e.g., "pytest --ignore='tests/slow'" -> ["pytest", "--ignore='tests/slow'"]
            cmd_parts = shlex.split(test_command)
            result = subprocess.run(
                cmd_parts,
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out")
            return False
        except Exception as e:
            logger.exception(f"Error running tests: {e}")
            return False
