"""
Tests for WorktreeManager git isolation.

Tests:
1. Worktree creation
2. Worktree listing
3. Merge to main
4. Cleanup
5. Conflict handling
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.worktree_manager import (
    MergeResult,
    MergeStatus,
    WorktreeInfo,
    WorktreeManager,
)


class TestWorktreeInfo(unittest.TestCase):
    """Tests for WorktreeInfo dataclass."""

    def test_worktree_info_creation(self):
        """Test creating WorktreeInfo."""
        info = WorktreeInfo(
            harness_id="task-001",
            branch_name="harness/task-001",
            path=Path("/tmp/worktrees/task-001"),
            base_branch="main",
        )
        self.assertEqual(info.harness_id, "task-001")
        self.assertEqual(info.branch_name, "harness/task-001")


class TestMergeResult(unittest.TestCase):
    """Tests for MergeResult dataclass."""

    def test_success_result(self):
        """Test successful merge result."""
        result = MergeResult(
            status=MergeStatus.SUCCESS,
            commit_sha="abc123",
            tests_passed=True,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.commit_sha, "abc123")

    def test_failure_result(self):
        """Test failed merge result."""
        result = MergeResult(
            status=MergeStatus.CONFLICT,
            error="Merge conflict in file.py",
        )
        self.assertFalse(result.success)
        self.assertIn("conflict", result.error.lower())


class TestWorktreeManagerInit(unittest.TestCase):
    """Tests for WorktreeManager initialization."""

    def test_init_not_in_repo(self):
        """Test initialization outside git repo raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                WorktreeManager(repo_root=Path(tmpdir))

    @patch("pathlib.Path.exists")
    def test_init_in_repo(self, mock_exists):
        """Test initialization in git repo succeeds."""
        mock_exists.return_value = True
        manager = WorktreeManager()
        self.assertEqual(manager.branch_prefix, "harness")


class TestWorktreeManagerWithMocks(unittest.TestCase):
    """Tests for WorktreeManager using mocks."""

    def setUp(self):
        self.patcher = patch.object(Path, 'exists', return_value=True)
        self.mock_exists = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("subprocess.run")
    def test_create_worktree(self, mock_run):
        """Test creating a worktree."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        manager = WorktreeManager()
        info = manager.create_worktree("task-001", base_branch="main")

        self.assertEqual(info.harness_id, "task-001")
        self.assertEqual(info.branch_name, "harness/task-001")
        self.assertEqual(info.base_branch, "main")

    @patch("subprocess.run")
    def test_list_worktrees(self, mock_run):
        """Test listing worktrees."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="""worktree /path/to/main
HEAD abc123
branch refs/heads/main

worktree /path/to/.worktrees/task-001
HEAD def456
branch refs/heads/harness/task-001

""",
        )

        manager = WorktreeManager(worktrees_dir=".worktrees")
        manager.worktrees_dir = Path("/path/to/.worktrees")

        worktrees = manager.list_active_worktrees()

        self.assertEqual(len(worktrees), 1)
        self.assertEqual(worktrees[0].harness_id, "task-001")

    @patch("subprocess.run")
    def test_cleanup_worktree(self, mock_run):
        """Test cleaning up a worktree."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, 'exists', return_value=True):
            manager = WorktreeManager()
            manager.worktrees_dir = Path("/tmp/worktrees")
            result = manager.cleanup_worktree("task-001")

        self.assertTrue(result)

    def test_merge_worktree_not_found(self):
        """Test merging when worktree doesn't exist."""
        # Create a real temp directory for the worktrees
        with tempfile.TemporaryDirectory() as tmpdir:
            # The class patcher is active, so we need to work with it
            self.patcher.stop()  # Temporarily stop the Path.exists mock

            try:
                with patch.object(Path, 'exists', return_value=True):
                    manager = WorktreeManager()

                # Now use a real temp dir where the worktree subdir doesn't exist
                manager.worktrees_dir = Path(tmpdir)

                result = manager.merge_to_main("task-001")

                self.assertEqual(result.status, MergeStatus.WORKTREE_NOT_FOUND)
                self.assertFalse(result.success)
            finally:
                # Restart the patcher
                self.patcher.start()


class TestWorktreeManagerIntegration(unittest.TestCase):
    """Integration tests for WorktreeManager with real git."""

    @classmethod
    def setUpClass(cls):
        """Check if git is available."""
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            cls.git_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            cls.git_available = False

    def setUp(self):
        if not self.git_available:
            self.skipTest("Git not available")

        # Create a temporary git repository
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / "repo"
        self.repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo_path,
            capture_output=True,
        )

        # Create initial commit
        (self.repo_path / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=self.repo_path,
            capture_output=True,
        )

    def tearDown(self):
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_and_list_worktree(self):
        """Test creating and listing a worktree."""
        manager = WorktreeManager(repo_root=self.repo_path)

        # Create worktree
        info = manager.create_worktree("task-001")

        self.assertEqual(info.harness_id, "task-001")
        self.assertTrue(info.path.exists())

        # List worktrees
        worktrees = manager.list_active_worktrees()
        self.assertEqual(len(worktrees), 1)
        self.assertEqual(worktrees[0].harness_id, "task-001")

    def test_create_merge_cleanup_workflow(self):
        """Test full worktree workflow."""
        manager = WorktreeManager(repo_root=self.repo_path)

        # Create worktree
        info = manager.create_worktree("task-002")

        # Make changes in worktree
        (info.path / "new_file.py").write_text("# New file")
        subprocess.run(["git", "add", "."], cwd=info.path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add new file"],
            cwd=info.path,
            capture_output=True,
        )

        # Merge back to main
        result = manager.merge_to_main("task-002", run_tests=False)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.commit_sha)

        # Cleanup
        cleanup_result = manager.cleanup_worktree("task-002")
        self.assertTrue(cleanup_result)

        # Verify worktree is removed
        worktrees = manager.list_active_worktrees()
        self.assertEqual(len(worktrees), 0)

    def test_worktree_has_changes(self):
        """Test detecting uncommitted changes."""
        manager = WorktreeManager(repo_root=self.repo_path)

        # Create worktree
        info = manager.create_worktree("task-003")

        # No changes initially
        self.assertFalse(manager.worktree_has_changes("task-003"))

        # Make uncommitted change
        (info.path / "uncommitted.txt").write_text("Uncommitted")

        self.assertTrue(manager.worktree_has_changes("task-003"))


if __name__ == "__main__":
    unittest.main()
