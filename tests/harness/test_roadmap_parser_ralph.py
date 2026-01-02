"""
Tests for ralph-specific roadmap parser extensions.

Tests:
1. Parsing ralph-enabled tasks
2. Default values when ralph fields omitted
3. Validation (max_iterations > 0)
4. Mixed ralph and non-ralph tasks
"""

import os
import shutil
import sys
import tempfile
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.models import RalphConfig
from um_agent_coder.harness.roadmap_parser import RoadmapParser

# Sample roadmap with ralph tasks
SAMPLE_RALPH_ROADMAP = """# Roadmap: Ralph Test Project

## Objective
Test ralph loop parsing

## Constraints
- Max time per task: 30 min
- Max retries per task: 3

## Success Criteria
- [ ] All tasks pass

## Tasks

### Phase 1: Basic Ralph Tasks
- [ ] **ralph-001**: Implement feature X with full test coverage
  - ralph: true
  - max_iterations: 30
  - completion_promise: FEATURE_X_COMPLETE
  - success: Tests pass, coverage > 80%
  - timeout: 60min

- [ ] **ralph-002**: Ralph task with defaults
  - ralph: true
  - success: Feature works

- [ ] **non-ralph-001**: Regular task
  - timeout: 15min
  - success: Done

### Phase 2: Mixed Tasks
- [ ] **ralph-003**: Custom promise task
  - ralph: true
  - max_iterations: 50
  - completion_promise: CUSTOM_PROMISE_TEXT
  - depends: ralph-001, ralph-002
  - cli: codex

- [ ] **non-ralph-002**: Another regular task
  - depends: ralph-003
  - cli: gemini

## Growth Mode
1. Improve test coverage
2. Add documentation
"""


class TestRoadmapParserRalph(unittest.TestCase):
    """Test ralph-specific parsing functionality."""

    def setUp(self):
        """Set up temporary roadmap file."""
        self.temp_dir = tempfile.mkdtemp()
        self.roadmap_path = os.path.join(self.temp_dir, "roadmap.md")
        with open(self.roadmap_path, 'w') as f:
            f.write(SAMPLE_RALPH_ROADMAP)
        self.parser = RoadmapParser(self.roadmap_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_parse_ralph_enabled_task(self):
        """Test parsing a task with ralph: true."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-001")

        self.assertIsNotNone(task)
        self.assertTrue(task.is_ralph_task)
        self.assertIsNotNone(task.ralph_config)
        self.assertTrue(task.ralph_config.enabled)

    def test_parse_ralph_max_iterations(self):
        """Test parsing max_iterations field."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-001")

        self.assertEqual(task.ralph_config.max_iterations, 30)

    def test_parse_ralph_completion_promise(self):
        """Test parsing completion_promise field."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-001")

        self.assertEqual(task.ralph_config.completion_promise, "FEATURE_X_COMPLETE")

    def test_ralph_defaults(self):
        """Test default values for ralph config."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-002")

        self.assertTrue(task.is_ralph_task)
        self.assertEqual(task.ralph_config.max_iterations, 30)  # default
        self.assertEqual(task.ralph_config.completion_promise, "COMPLETE")  # default

    def test_non_ralph_task(self):
        """Test that non-ralph tasks have no ralph config."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("non-ralph-001")

        self.assertIsNotNone(task)
        self.assertFalse(task.is_ralph_task)
        self.assertIsNone(task.ralph_config)

    def test_custom_promise_text(self):
        """Test custom promise text parsing."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-003")

        self.assertEqual(task.ralph_config.completion_promise, "CUSTOM_PROMISE_TEXT")
        self.assertEqual(task.ralph_config.max_iterations, 50)

    def test_ralph_with_dependencies(self):
        """Test ralph task with dependencies."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-003")

        self.assertTrue(task.is_ralph_task)
        self.assertEqual(task.depends, ["ralph-001", "ralph-002"])

    def test_ralph_with_cli_override(self):
        """Test ralph task with CLI override."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-003")

        self.assertTrue(task.is_ralph_task)
        self.assertEqual(task.cli, "codex")

    def test_mixed_ralph_and_regular_tasks(self):
        """Test roadmap with mix of ralph and regular tasks."""
        roadmap = self.parser.parse()

        ralph_tasks = [t for t in roadmap.all_tasks if t.is_ralph_task]
        non_ralph_tasks = [t for t in roadmap.all_tasks if not t.is_ralph_task]

        self.assertEqual(len(ralph_tasks), 3)
        self.assertEqual(len(non_ralph_tasks), 2)

    def test_ralph_task_timeout(self):
        """Test that timeout is preserved for ralph tasks."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-001")

        self.assertEqual(task.timeout_minutes, 60)

    def test_ralph_task_success_criteria(self):
        """Test that success criteria is preserved for ralph tasks."""
        roadmap = self.parser.parse()
        task = roadmap.get_task("ralph-001")

        self.assertEqual(task.success_criteria, "Tests pass, coverage > 80%")


class TestRalphConfigModel(unittest.TestCase):
    """Test RalphConfig model."""

    def test_ralph_config_defaults(self):
        """Test RalphConfig default values."""
        config = RalphConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.max_iterations, 30)
        self.assertEqual(config.completion_promise, "COMPLETE")
        self.assertTrue(config.require_xml_format)

    def test_ralph_config_custom(self):
        """Test RalphConfig with custom values."""
        config = RalphConfig(
            enabled=True,
            max_iterations=50,
            completion_promise="CUSTOM",
            require_xml_format=False,
        )

        self.assertEqual(config.max_iterations, 50)
        self.assertEqual(config.completion_promise, "CUSTOM")
        self.assertFalse(config.require_xml_format)

    def test_ralph_config_serialization(self):
        """Test RalphConfig to_dict and from_dict."""
        config = RalphConfig(
            enabled=True,
            max_iterations=25,
            completion_promise="TEST_COMPLETE",
        )

        data = config.to_dict()
        restored = RalphConfig.from_dict(data)

        self.assertEqual(restored.enabled, config.enabled)
        self.assertEqual(restored.max_iterations, config.max_iterations)
        self.assertEqual(restored.completion_promise, config.completion_promise)


class TestRalphEdgeCases(unittest.TestCase):
    """Test edge cases for ralph parsing."""

    def test_ralph_false(self):
        """Test explicitly setting ralph: false."""
        roadmap_content = """# Roadmap: Test

## Objective
Test

## Tasks

### Phase 1: Test
- [ ] **task-001**: Test task
  - ralph: false
  - max_iterations: 50
"""
        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "roadmap.md")
            with open(path, 'w') as f:
                f.write(roadmap_content)

            parser = RoadmapParser(path)
            roadmap = parser.parse()
            task = roadmap.get_task("task-001")

            self.assertFalse(task.is_ralph_task)
            self.assertIsNone(task.ralph_config)
        finally:
            shutil.rmtree(temp_dir)

    def test_ralph_yes_variant(self):
        """Test ralph: yes as alternative to true."""
        roadmap_content = """# Roadmap: Test

## Objective
Test

## Tasks

### Phase 1: Test
- [ ] **task-001**: Test task
  - ralph: yes
"""
        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "roadmap.md")
            with open(path, 'w') as f:
                f.write(roadmap_content)

            parser = RoadmapParser(path)
            roadmap = parser.parse()
            task = roadmap.get_task("task-001")

            self.assertTrue(task.is_ralph_task)
        finally:
            shutil.rmtree(temp_dir)

    def test_ralph_1_variant(self):
        """Test ralph: 1 as alternative to true."""
        roadmap_content = """# Roadmap: Test

## Objective
Test

## Tasks

### Phase 1: Test
- [ ] **task-001**: Test task
  - ralph: 1
"""
        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "roadmap.md")
            with open(path, 'w') as f:
                f.write(roadmap_content)

            parser = RoadmapParser(path)
            roadmap = parser.parse()
            task = roadmap.get_task("task-001")

            self.assertTrue(task.is_ralph_task)
        finally:
            shutil.rmtree(temp_dir)

    def test_promise_with_spaces(self):
        """Test completion_promise with spaces (should preserve)."""
        roadmap_content = """# Roadmap: Test

## Objective
Test

## Tasks

### Phase 1: Test
- [ ] **task-001**: Test task
  - ralph: true
  - completion_promise: FEATURE COMPLETE WITH SPACES
"""
        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "roadmap.md")
            with open(path, 'w') as f:
                f.write(roadmap_content)

            parser = RoadmapParser(path)
            roadmap = parser.parse()
            task = roadmap.get_task("task-001")

            self.assertEqual(
                task.ralph_config.completion_promise,
                "FEATURE COMPLETE WITH SPACES"
            )
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
