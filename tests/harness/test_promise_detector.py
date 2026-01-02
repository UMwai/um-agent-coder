"""
Unit tests for the PromiseDetector class.

Tests:
1. XML-style promise detection
2. Plain text detection
3. No-match scenarios
4. Edge cases (partial matches, nested tags, whitespace)
5. Case sensitivity
"""

import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.ralph.promise_detector import PromiseDetector


class TestXMLStyleDetection(unittest.TestCase):
    """Test XML-style promise detection (<promise>TEXT</promise>)."""

    def test_basic_xml_detection(self):
        """Test basic XML promise detection."""
        detector = PromiseDetector(promise="COMPLETE")
        output = "Task finished. <promise>COMPLETE</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "COMPLETE")
        self.assertEqual(result.match_type, "xml")

    def test_xml_detection_with_whitespace(self):
        """Test XML detection with internal whitespace."""
        detector = PromiseDetector(promise="COMPLETE")
        output = "Output: <promise>  COMPLETE  </promise> more text"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "COMPLETE")
        self.assertEqual(result.match_type, "xml")

    def test_xml_detection_multiline(self):
        """Test XML detection spanning multiple lines."""
        detector = PromiseDetector(promise="TASK_DONE")
        output = """Some output
        <promise>
        TASK_DONE
        </promise>
        More output"""

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "TASK_DONE")

    def test_xml_detection_case_insensitive_tags(self):
        """Test that promise tags are case-insensitive."""
        detector = PromiseDetector(promise="DONE")
        output = "Output: <PROMISE>DONE</PROMISE> end"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "DONE")

    def test_xml_detection_underscore_promise(self):
        """Test XML detection with underscore in promise."""
        detector = PromiseDetector(promise="RALPH_LOOP_IMPLEMENTATION_COMPLETE")
        output = "All done! <promise>RALPH_LOOP_IMPLEMENTATION_COMPLETE</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "RALPH_LOOP_IMPLEMENTATION_COMPLETE")

    def test_xml_detection_returns_position(self):
        """Test that XML detection returns match position."""
        detector = PromiseDetector(promise="DONE")
        output = "Prefix text <promise>DONE</promise> suffix"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.position, 12)  # Position of <promise>


class TestPlainTextDetection(unittest.TestCase):
    """Test plain text promise detection."""

    def test_basic_plain_text_detection(self):
        """Test basic plain text detection."""
        detector = PromiseDetector(promise="FEATURE_COMPLETE")
        output = "The feature is done. FEATURE_COMPLETE. Goodbye."

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "FEATURE_COMPLETE")
        self.assertEqual(result.match_type, "plain")

    def test_plain_text_case_insensitive(self):
        """Test plain text is case-insensitive by default."""
        detector = PromiseDetector(promise="COMPLETE")
        output = "Task is complete now."

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "complete")

    def test_plain_text_case_sensitive(self):
        """Test plain text with case-sensitive mode."""
        detector = PromiseDetector(promise="COMPLETE", case_sensitive=True)
        output = "Task is complete now."  # lowercase

        result = detector.detect(output)

        self.assertFalse(result.found)

    def test_plain_text_case_sensitive_match(self):
        """Test plain text case-sensitive match succeeds."""
        detector = PromiseDetector(promise="COMPLETE", case_sensitive=True)
        output = "Task is COMPLETE now."

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "COMPLETE")

    def test_xml_preferred_over_plain(self):
        """Test that XML format is preferred over plain text."""
        detector = PromiseDetector(promise="DONE")
        output = "DONE earlier. And also <promise>DONE</promise> here."

        result = detector.detect(output)

        # Should find XML version first
        self.assertTrue(result.found)
        self.assertEqual(result.match_type, "xml")


class TestNoMatchScenarios(unittest.TestCase):
    """Test scenarios where promise is not found."""

    def test_empty_output(self):
        """Test with empty output."""
        detector = PromiseDetector(promise="COMPLETE")
        result = detector.detect("")

        self.assertFalse(result.found)
        self.assertIsNone(result.promise_text)

    def test_none_like_output(self):
        """Test with None-like empty string."""
        detector = PromiseDetector(promise="COMPLETE")
        result = detector.detect("")

        self.assertFalse(result.found)

    def test_wrong_promise_text(self):
        """Test with different promise text."""
        detector = PromiseDetector(promise="COMPLETE")
        output = "<promise>FAILED</promise>"

        result = detector.detect(output)

        self.assertFalse(result.found)

    def test_partial_promise_text(self):
        """Test with partial match."""
        detector = PromiseDetector(promise="TASK_COMPLETE")
        output = "<promise>TASK</promise>"

        result = detector.detect(output)

        self.assertFalse(result.found)

    def test_promise_in_comment(self):
        """Test promise appearing only in code comment."""
        detector = PromiseDetector(promise="DONE", require_xml_format=True)
        output = "# This outputs DONE when complete"

        result = detector.detect(output)

        self.assertFalse(result.found)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    def test_nested_tags_outer(self):
        """Test with nested tags - should match outer."""
        detector = PromiseDetector(promise="OUTER")
        output = "<promise>OUTER</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "OUTER")

    def test_multiple_promises_same_text(self):
        """Test with multiple identical promises."""
        detector = PromiseDetector(promise="DONE")
        output = "<promise>DONE</promise> and <promise>DONE</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        # Should find first occurrence
        self.assertEqual(result.promise_text, "DONE")

    def test_multiple_different_promises(self):
        """Test with multiple different promises."""
        detector = PromiseDetector(promise="SECOND")
        output = "<promise>FIRST</promise> then <promise>SECOND</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "SECOND")

    def test_malformed_tags(self):
        """Test with malformed tags."""
        detector = PromiseDetector(promise="DONE")
        output = "<promise>DONE</promis>"  # Missing 'e'

        result = detector.detect(output)

        # Should not match malformed XML, but may match plain text
        if result.found:
            self.assertEqual(result.match_type, "plain")

    def test_promise_with_special_chars(self):
        """Test promise with special characters."""
        detector = PromiseDetector(promise="DONE_v1.0")
        output = "<promise>DONE_v1.0</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "DONE_v1.0")

    def test_very_long_output(self):
        """Test with very long output."""
        detector = PromiseDetector(promise="COMPLETE")
        output = "x" * 100000 + "<promise>COMPLETE</promise>" + "y" * 100000

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "COMPLETE")

    def test_promise_at_start(self):
        """Test promise at the very start of output."""
        detector = PromiseDetector(promise="START")
        output = "<promise>START</promise> rest of output"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.position, 0)

    def test_promise_at_end(self):
        """Test promise at the very end of output."""
        detector = PromiseDetector(promise="END")
        output = "Some output <promise>END</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "END")


class TestRequireXMLFormat(unittest.TestCase):
    """Test require_xml_format option."""

    def test_require_xml_blocks_plain(self):
        """Test that require_xml_format blocks plain text matches."""
        detector = PromiseDetector(promise="DONE", require_xml_format=True)
        output = "The task is DONE now."

        result = detector.detect(output)

        self.assertFalse(result.found)

    def test_require_xml_allows_xml(self):
        """Test that require_xml_format allows XML matches."""
        detector = PromiseDetector(promise="DONE", require_xml_format=True)
        output = "Output: <promise>DONE</promise>"

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.match_type, "xml")


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods."""

    def test_detect_any_promise(self):
        """Test detect_any_promise finds any promise."""
        detector = PromiseDetector(promise="NOT_THIS")
        output = "Output: <promise>SOMETHING_ELSE</promise>"

        result = detector.detect_any_promise(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "SOMETHING_ELSE")

    def test_detect_any_promise_empty(self):
        """Test detect_any_promise with no promise."""
        detector = PromiseDetector(promise="ANYTHING")
        output = "No promise here"

        result = detector.detect_any_promise(output)

        self.assertFalse(result.found)

    def test_extract_all_promises(self):
        """Test extracting all promises from output."""
        output = """
        <promise>FIRST</promise>
        Some text
        <promise>SECOND</promise>
        More text
        <promise>THIRD</promise>
        """

        promises = PromiseDetector.extract_all_promises(output)

        self.assertEqual(len(promises), 3)
        self.assertEqual(promises[0], "FIRST")
        self.assertEqual(promises[1], "SECOND")
        self.assertEqual(promises[2], "THIRD")

    def test_extract_all_promises_empty(self):
        """Test extracting promises from empty output."""
        promises = PromiseDetector.extract_all_promises("")

        self.assertEqual(promises, [])

    def test_extract_all_promises_none(self):
        """Test extracting promises when none present."""
        promises = PromiseDetector.extract_all_promises("No promises here")

        self.assertEqual(promises, [])


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""

    def test_claude_style_output(self):
        """Test with Claude-style verbose output."""
        detector = PromiseDetector(promise="RALPH_LOOP_IMPLEMENTATION_COMPLETE")
        output = """
        I've completed all the implementation tasks:

        1. Created PromiseDetector class
        2. Implemented IterationTracker
        3. Built RalphExecutor wrapper
        4. Extended roadmap parser
        5. Integrated with harness

        All tests pass and the implementation is complete.

        <promise>RALPH_LOOP_IMPLEMENTATION_COMPLETE</promise>
        """

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.promise_text, "RALPH_LOOP_IMPLEMENTATION_COMPLETE")

    def test_codex_style_output(self):
        """Test with Codex-style output."""
        detector = PromiseDetector(promise="FEATURE_X_COMPLETE")
        output = """
        > Implementing feature X...
        > Running tests...
        > All 15 tests passed

        FEATURE_X_COMPLETE

        The feature has been implemented successfully.
        """

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.match_type, "plain")

    def test_mixed_output_xml_preferred(self):
        """Test that XML is found even when plain text also present."""
        detector = PromiseDetector(promise="DONE")
        output = """
        Status: DONE with phase 1
        Moving to phase 2...
        All phases DONE
        <promise>DONE</promise>
        Cleanup complete.
        """

        result = detector.detect(output)

        self.assertTrue(result.found)
        self.assertEqual(result.match_type, "xml")


if __name__ == "__main__":
    unittest.main()
