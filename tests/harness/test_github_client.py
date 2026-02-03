"""
Tests for GitHubClient issue management.

Tests:
1. Issue fetching
2. Issue closing
3. Comment adding
4. Label management
5. Error handling
"""

import json
import os
import subprocess
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.github_client import GitHubClient, GitHubIssue


class TestGitHubIssue(unittest.TestCase):
    """Tests for GitHubIssue dataclass."""

    def test_from_json_full(self):
        """Test creating GitHubIssue from full JSON."""
        data = {
            "number": 42,
            "title": "Test Issue",
            "body": "This is the body",
            "labels": [{"name": "bug"}, {"name": "harness"}],
            "state": "open",
            "url": "https://github.com/owner/repo/issues/42",
            "createdAt": "2024-01-15T10:30:00Z",
            "updatedAt": "2024-01-16T14:00:00Z",
            "assignees": [{"login": "user1"}],
            "milestone": {"title": "v1.0"},
        }

        issue = GitHubIssue.from_json(data)

        self.assertEqual(issue.number, 42)
        self.assertEqual(issue.title, "Test Issue")
        self.assertEqual(issue.body, "This is the body")
        self.assertEqual(issue.labels, ["bug", "harness"])
        self.assertEqual(issue.state, "open")
        self.assertEqual(issue.assignees, ["user1"])
        self.assertEqual(issue.milestone, "v1.0")

    def test_from_json_minimal(self):
        """Test creating GitHubIssue from minimal JSON."""
        data = {
            "number": 1,
            "title": "Minimal Issue",
        }

        issue = GitHubIssue.from_json(data)

        self.assertEqual(issue.number, 1)
        self.assertEqual(issue.title, "Minimal Issue")
        self.assertEqual(issue.body, "")
        self.assertEqual(issue.labels, [])

    def test_to_task_description(self):
        """Test converting issue to task description."""
        issue = GitHubIssue(
            number=42,
            title="Implement feature X",
            body="Details about feature X\n\nMore details...",
            labels=["feature", "harness"],
        )

        description = issue.to_task_description()

        self.assertIn("Implement feature X", description)
        self.assertIn("Details about feature X", description)
        self.assertIn("feature", description)

    def test_to_task_description_truncates_long_body(self):
        """Test that long issue bodies are truncated."""
        issue = GitHubIssue(
            number=42,
            title="Long Issue",
            body="x" * 2000,
            labels=[],
        )

        description = issue.to_task_description()

        # Body should be truncated and show ellipsis
        self.assertTrue(len(description) < 2200)


class TestGitHubClientMocked(unittest.TestCase):
    """Tests for GitHubClient using mocks."""

    def setUp(self):
        self.client = GitHubClient(repo="owner/repo")

    @patch("subprocess.run")
    def test_fetch_issues_success(self, mock_run):
        """Test fetching issues successfully."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"number": 1, "title": "Issue 1", "labels": [{"name": "harness"}], "state": "open"},
                {"number": 2, "title": "Issue 2", "labels": [{"name": "harness"}], "state": "open"},
            ]),
            stderr="",
        )

        issues = self.client.fetch_issues(labels=["harness"])

        self.assertEqual(len(issues), 2)
        self.assertEqual(issues[0].number, 1)
        self.assertEqual(issues[1].number, 2)

    @patch("subprocess.run")
    def test_fetch_issues_empty(self, mock_run):
        """Test fetching when no issues match."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="[]",
            stderr="",
        )

        issues = self.client.fetch_issues(labels=["nonexistent"])

        self.assertEqual(len(issues), 0)

    @patch("subprocess.run")
    def test_fetch_issues_error(self, mock_run):
        """Test handling fetch errors gracefully."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="Not authenticated")

        issues = self.client.fetch_issues()

        self.assertEqual(len(issues), 0)

    @patch("subprocess.run")
    def test_get_issue_success(self, mock_run):
        """Test getting a single issue."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "number": 42,
                "title": "Specific Issue",
                "body": "Body text",
                "labels": [],
                "state": "open",
            }),
            stderr="",
        )

        issue = self.client.get_issue(42)

        self.assertIsNotNone(issue)
        self.assertEqual(issue.number, 42)
        self.assertEqual(issue.title, "Specific Issue")

    @patch("subprocess.run")
    def test_get_issue_not_found(self, mock_run):
        """Test getting non-existent issue."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="issue not found")

        issue = self.client.get_issue(99999)

        self.assertIsNone(issue)

    @patch("subprocess.run")
    def test_close_issue_success(self, mock_run):
        """Test closing an issue."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = self.client.close_issue(42, comment="Completed by harness")

        self.assertTrue(result)
        # Should have been called twice: once for comment, once for close
        self.assertEqual(mock_run.call_count, 2)

    @patch("subprocess.run")
    def test_close_issue_no_comment(self, mock_run):
        """Test closing an issue without comment."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = self.client.close_issue(42)

        self.assertTrue(result)
        # Should only be called once for close
        self.assertEqual(mock_run.call_count, 1)

    @patch("subprocess.run")
    def test_close_issue_error(self, mock_run):
        """Test handling close errors."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="Permission denied")

        result = self.client.close_issue(42)

        self.assertFalse(result)

    @patch("subprocess.run")
    def test_add_comment_success(self, mock_run):
        """Test adding a comment to an issue."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = self.client.add_issue_comment(42, "Test comment")

        self.assertTrue(result)

    @patch("subprocess.run")
    def test_add_label_success(self, mock_run):
        """Test adding a label to an issue."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = self.client.add_label(42, "in-progress")

        self.assertTrue(result)
        call_args = mock_run.call_args[0][0]
        self.assertIn("--add-label", call_args)

    @patch("subprocess.run")
    def test_remove_label_success(self, mock_run):
        """Test removing a label from an issue."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = self.client.remove_label(42, "in-progress")

        self.assertTrue(result)
        call_args = mock_run.call_args[0][0]
        self.assertIn("--remove-label", call_args)

    @patch("subprocess.run")
    def test_check_gh_cli_available(self, mock_run):
        """Test checking gh CLI availability."""
        mock_run.return_value = MagicMock(returncode=0)

        result = self.client.check_gh_cli()

        self.assertTrue(result)

    @patch("subprocess.run")
    def test_check_gh_cli_not_authenticated(self, mock_run):
        """Test checking gh CLI when not authenticated."""
        mock_run.return_value = MagicMock(returncode=1)

        result = self.client.check_gh_cli()

        self.assertFalse(result)


class TestGitHubClientRepoDetection(unittest.TestCase):
    """Tests for repository auto-detection."""

    @patch("subprocess.run")
    def test_auto_detect_repo(self, mock_run):
        """Test auto-detecting repository from git."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"nameWithOwner": "owner/auto-repo"}),
            stderr="",
        )

        client = GitHubClient()  # No repo specified
        repo = client.get_repo()

        self.assertEqual(repo, "owner/auto-repo")

    @patch("subprocess.run")
    def test_explicit_repo_overrides_detection(self, mock_run):
        """Test that explicit repo is used over detection."""
        client = GitHubClient(repo="explicit/repo")
        repo = client.get_repo()

        self.assertEqual(repo, "explicit/repo")
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_detection_caches_result(self, mock_run):
        """Test that repo detection result is cached."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"nameWithOwner": "owner/cached-repo"}),
            stderr="",
        )

        client = GitHubClient()
        repo1 = client.get_repo()
        repo2 = client.get_repo()

        self.assertEqual(repo1, repo2)
        # Should only call gh once due to caching
        self.assertEqual(mock_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
