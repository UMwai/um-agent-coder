"""
Unit tests for IterationTracker and RalphPersistence.

Tests:
1. Iteration increment and limit enforcement
2. Iteration record tracking
3. Persistence save/load
4. Resume from interruption
5. Summary statistics
"""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.ralph.iteration_tracker import (
    IterationRecord,
    IterationTracker,
)
from um_agent_coder.harness.ralph.persistence import RalphPersistence


class TestIterationRecord(unittest.TestCase):
    """Test IterationRecord dataclass."""

    def test_basic_creation(self):
        """Test basic record creation."""
        now = datetime.utcnow()
        record = IterationRecord(
            iteration_num=1,
            started_at=now,
        )

        self.assertEqual(record.iteration_num, 1)
        self.assertEqual(record.started_at, now)
        self.assertIsNone(record.ended_at)
        self.assertFalse(record.promise_found)

    def test_duration_calculation(self):
        """Test duration property."""
        start = datetime.utcnow()
        end = start + timedelta(seconds=30)

        record = IterationRecord(
            iteration_num=1,
            started_at=start,
            ended_at=end,
        )

        self.assertEqual(record.duration, timedelta(seconds=30))

    def test_duration_none_when_incomplete(self):
        """Test duration is None when not ended."""
        record = IterationRecord(
            iteration_num=1,
            started_at=datetime.utcnow(),
        )

        self.assertIsNone(record.duration)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        now = datetime.utcnow()
        record = IterationRecord(
            iteration_num=5,
            started_at=now,
            ended_at=now + timedelta(seconds=10),
            output_snippet="test output",
            promise_found=True,
            error=None,
        )

        data = record.to_dict()
        restored = IterationRecord.from_dict(data)

        self.assertEqual(restored.iteration_num, 5)
        self.assertEqual(restored.output_snippet, "test output")
        self.assertTrue(restored.promise_found)


class TestIterationTracker(unittest.TestCase):
    """Test IterationTracker class."""

    def test_basic_creation(self):
        """Test basic tracker creation."""
        tracker = IterationTracker(
            task_id="task-001",
            max_iterations=10,
        )

        self.assertEqual(tracker.task_id, "task-001")
        self.assertEqual(tracker.max_iterations, 10)
        self.assertEqual(tracker.current_iteration, 0)
        self.assertTrue(tracker.can_continue())

    def test_increment(self):
        """Test iteration increment."""
        tracker = IterationTracker(task_id="test", max_iterations=5)

        tracker.increment()
        self.assertEqual(tracker.current_iteration, 1)

        tracker.increment()
        self.assertEqual(tracker.current_iteration, 2)

    def test_max_iterations_limit(self):
        """Test that max_iterations is enforced."""
        tracker = IterationTracker(task_id="test", max_iterations=3)

        self.assertTrue(tracker.can_continue())
        tracker.increment()
        self.assertTrue(tracker.can_continue())
        tracker.increment()
        self.assertTrue(tracker.can_continue())
        tracker.increment()
        self.assertFalse(tracker.can_continue())

    def test_start_and_end_iteration(self):
        """Test start_iteration and end_iteration."""
        tracker = IterationTracker(task_id="test", max_iterations=10)

        record = tracker.start_iteration()
        self.assertEqual(record.iteration_num, 1)
        self.assertIsNone(record.ended_at)

        completed = tracker.end_iteration(
            output="Test output",
            promise_found=False,
        )

        self.assertIsNotNone(completed.ended_at)
        self.assertEqual(completed.output_snippet, "Test output")
        self.assertFalse(completed.promise_found)

    def test_iteration_history_tracking(self):
        """Test that iterations are tracked in history."""
        tracker = IterationTracker(task_id="test", max_iterations=10)

        # Do 3 iterations
        for i in range(3):
            tracker.start_iteration()
            tracker.end_iteration(output=f"Output {i+1}", promise_found=False)

        self.assertEqual(len(tracker.iteration_history), 3)
        self.assertEqual(tracker.iteration_history[0].output_snippet, "Output 1")
        self.assertEqual(tracker.iteration_history[2].output_snippet, "Output 3")

    def test_promise_found_marks_complete(self):
        """Test that finding promise marks tracker as complete."""
        tracker = IterationTracker(task_id="test", max_iterations=10)

        tracker.start_iteration()
        tracker.end_iteration(output="Done", promise_found=True)

        self.assertTrue(tracker.is_complete)
        self.assertEqual(tracker.completion_reason, "promise_found")
        self.assertFalse(tracker.can_continue())

    def test_output_snippet_truncation(self):
        """Test that output is truncated."""
        tracker = IterationTracker(task_id="test", max_iterations=10)

        tracker.start_iteration()
        long_output = "x" * 1000
        tracker.end_iteration(output=long_output, output_snippet_length=100)

        self.assertEqual(len(tracker.iteration_history[0].output_snippet), 100)

    def test_iterations_remaining(self):
        """Test iterations_remaining property."""
        tracker = IterationTracker(task_id="test", max_iterations=5)

        self.assertEqual(tracker.iterations_remaining, 5)
        tracker.increment()
        self.assertEqual(tracker.iterations_remaining, 4)
        tracker.increment()
        tracker.increment()
        self.assertEqual(tracker.iterations_remaining, 2)

    def test_mark_exceeded(self):
        """Test mark_exceeded method."""
        tracker = IterationTracker(task_id="test", max_iterations=5)

        tracker.mark_exceeded()

        self.assertTrue(tracker.is_complete)
        self.assertEqual(tracker.completion_reason, "max_iterations_exceeded")
        self.assertFalse(tracker.can_continue())

    def test_get_summary(self):
        """Test get_summary method."""
        tracker = IterationTracker(task_id="test-summary", max_iterations=10)

        # Do some iterations
        for i in range(3):
            tracker.start_iteration()
            tracker.end_iteration(
                output=f"Output {i}",
                promise_found=(i == 2),  # Found on iteration 3
            )

        summary = tracker.get_summary()

        self.assertEqual(summary["task_id"], "test-summary")
        self.assertEqual(summary["max_iterations"], 10)
        self.assertEqual(summary["current_iteration"], 3)
        self.assertEqual(summary["iterations_remaining"], 7)
        self.assertTrue(summary["is_complete"])
        self.assertEqual(summary["completion_reason"], "promise_found")
        self.assertEqual(summary["successful_iterations"], 1)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        tracker = IterationTracker(task_id="serialize-test", max_iterations=15)

        tracker.start_iteration()
        tracker.end_iteration(output="First", promise_found=False)
        tracker.start_iteration()
        tracker.end_iteration(output="Second", promise_found=True)

        data = tracker.to_dict()
        restored = IterationTracker.from_dict(data)

        self.assertEqual(restored.task_id, "serialize-test")
        self.assertEqual(restored.max_iterations, 15)
        self.assertEqual(restored.current_iteration, 2)
        self.assertTrue(restored.is_complete)
        self.assertEqual(len(restored.iteration_history), 2)


class TestRalphPersistence(unittest.TestCase):
    """Test RalphPersistence class."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ralph_test.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_tracker(self):
        """Test saving and loading a tracker."""
        tracker = IterationTracker(task_id="persist-001", max_iterations=20)

        tracker.start_iteration()
        tracker.end_iteration(output="Test output", promise_found=False)

        self.persistence.save_tracker(tracker)

        # Load it back
        loaded = self.persistence.load_tracker("persist-001")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.task_id, "persist-001")
        self.assertEqual(loaded.max_iterations, 20)
        self.assertEqual(loaded.current_iteration, 1)
        self.assertEqual(len(loaded.iteration_history), 1)

    def test_load_nonexistent_tracker(self):
        """Test loading a tracker that doesn't exist."""
        loaded = self.persistence.load_tracker("nonexistent")
        self.assertIsNone(loaded)

    def test_resume_from_interruption(self):
        """Test resuming after simulated interruption."""
        # Create and save tracker partway through
        tracker = IterationTracker(task_id="resume-001", max_iterations=10)

        for i in range(3):
            tracker.start_iteration()
            tracker.end_iteration(output=f"Iteration {i+1}")

        self.persistence.save_tracker(tracker)

        # Simulate restart - load fresh instance
        resumed = self.persistence.load_tracker("resume-001")

        self.assertEqual(resumed.current_iteration, 3)
        self.assertEqual(len(resumed.iteration_history), 3)
        self.assertTrue(resumed.can_continue())
        self.assertEqual(resumed.iterations_remaining, 7)

    def test_update_existing_tracker(self):
        """Test updating an existing tracker."""
        tracker = IterationTracker(task_id="update-001", max_iterations=10)

        tracker.start_iteration()
        tracker.end_iteration(output="First")
        self.persistence.save_tracker(tracker)

        # Continue and save again
        tracker.start_iteration()
        tracker.end_iteration(output="Second", promise_found=True)
        self.persistence.save_tracker(tracker)

        # Load and verify
        loaded = self.persistence.load_tracker("update-001")

        self.assertEqual(loaded.current_iteration, 2)
        self.assertTrue(loaded.is_complete)
        self.assertEqual(len(loaded.iteration_history), 2)

    def test_delete_tracker(self):
        """Test deleting a tracker."""
        tracker = IterationTracker(task_id="delete-001", max_iterations=5)
        tracker.start_iteration()
        tracker.end_iteration(output="Test")
        self.persistence.save_tracker(tracker)

        # Verify it exists
        self.assertIsNotNone(self.persistence.load_tracker("delete-001"))

        # Delete it
        result = self.persistence.delete_tracker("delete-001")
        self.assertTrue(result)

        # Verify it's gone
        self.assertIsNone(self.persistence.load_tracker("delete-001"))

    def test_delete_nonexistent_tracker(self):
        """Test deleting a tracker that doesn't exist."""
        result = self.persistence.delete_tracker("nonexistent")
        self.assertFalse(result)

    def test_list_active_trackers(self):
        """Test listing active trackers."""
        # Create some trackers
        for i in range(3):
            tracker = IterationTracker(task_id=f"active-{i}", max_iterations=10)
            if i == 1:  # Mark one as complete
                tracker.mark_complete("test")
            self.persistence.save_tracker(tracker)

        active = self.persistence.list_active_trackers()

        self.assertEqual(len(active), 2)
        self.assertIn("active-0", active)
        self.assertIn("active-2", active)
        self.assertNotIn("active-1", active)

    def test_list_all_trackers(self):
        """Test listing all trackers."""
        for i in range(3):
            tracker = IterationTracker(task_id=f"list-{i}", max_iterations=10+i)
            tracker.start_iteration()
            tracker.end_iteration(output="Test")
            self.persistence.save_tracker(tracker)

        all_trackers = self.persistence.list_all_trackers()

        self.assertEqual(len(all_trackers), 3)
        # Check it includes summary fields
        for t in all_trackers:
            self.assertIn("task_id", t)
            self.assertIn("max_iterations", t)
            self.assertIn("iteration_count", t)

    def test_get_iteration_history(self):
        """Test getting iteration history."""
        tracker = IterationTracker(task_id="history-001", max_iterations=10)

        for i in range(5):
            tracker.start_iteration()
            tracker.end_iteration(output=f"Output {i}", promise_found=(i == 4))

        self.persistence.save_tracker(tracker)

        history = self.persistence.get_iteration_history("history-001")

        self.assertEqual(len(history), 5)
        self.assertEqual(history[0].output_snippet, "Output 0")
        self.assertTrue(history[4].promise_found)

    def test_reset(self):
        """Test reset clears all data."""
        # Create some trackers
        for i in range(3):
            tracker = IterationTracker(task_id=f"reset-{i}", max_iterations=10)
            self.persistence.save_tracker(tracker)

        # Reset
        self.persistence.reset()

        # Verify all gone
        all_trackers = self.persistence.list_all_trackers()
        self.assertEqual(len(all_trackers), 0)

    def test_persistence_with_errors(self):
        """Test persistence handles iteration errors."""
        tracker = IterationTracker(task_id="error-001", max_iterations=10)

        tracker.start_iteration()
        tracker.end_iteration(
            output="Failed output",
            promise_found=False,
            error="Execution failed with timeout"
        )

        self.persistence.save_tracker(tracker)

        loaded = self.persistence.load_tracker("error-001")

        self.assertEqual(loaded.iteration_history[0].error, "Execution failed with timeout")


class TestPersistenceEdgeCases(unittest.TestCase):
    """Test edge cases for persistence."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "edge.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_empty_output_snippet(self):
        """Test handling empty output."""
        tracker = IterationTracker(task_id="empty-001", max_iterations=5)
        tracker.start_iteration()
        tracker.end_iteration(output="")
        self.persistence.save_tracker(tracker)

        loaded = self.persistence.load_tracker("empty-001")
        self.assertEqual(loaded.iteration_history[0].output_snippet, "")

    def test_unicode_in_output(self):
        """Test handling unicode in output."""
        tracker = IterationTracker(task_id="unicode-001", max_iterations=5)
        tracker.start_iteration()
        # Use valid UTF-8 characters (not surrogate pairs)
        tracker.end_iteration(output="Unicode: \u2713 \u2717 \U0001F600")
        self.persistence.save_tracker(tracker)

        loaded = self.persistence.load_tracker("unicode-001")
        self.assertIn("\u2713", loaded.iteration_history[0].output_snippet)

    def test_concurrent_access(self):
        """Test multiple persistence instances."""
        p1 = RalphPersistence(self.db_path)
        p2 = RalphPersistence(self.db_path)

        # Save from one
        tracker = IterationTracker(task_id="concurrent-001", max_iterations=10)
        p1.save_tracker(tracker)

        # Load from other
        loaded = p2.load_tracker("concurrent-001")
        self.assertIsNotNone(loaded)

    def test_large_iteration_history(self):
        """Test handling many iterations."""
        tracker = IterationTracker(task_id="large-001", max_iterations=100)

        for i in range(50):
            tracker.start_iteration()
            tracker.end_iteration(output=f"Iteration {i}")

        self.persistence.save_tracker(tracker)

        loaded = self.persistence.load_tracker("large-001")
        self.assertEqual(len(loaded.iteration_history), 50)


if __name__ == "__main__":
    unittest.main()
