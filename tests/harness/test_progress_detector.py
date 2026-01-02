"""Tests for progress detection system."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.um_agent_coder.harness.autonomous.progress_detector import (
    DEFAULT_WEIGHTS,
    GitDiffStats,
    ProgressDetector,
    ProgressSignal,
    _normalize_output,
    _parse_git_diff_stat,
    calculate_progress_score,
    file_changes_score,
    output_diff_score,
)
from src.um_agent_coder.harness.autonomous.progress_markers import (
    count_progress_markers,
    extract_progress_markers,
    has_progress_markers,
)


class TestProgressMarkers:
    """Tests for progress marker extraction."""

    def test_extract_single_marker(self):
        """Test extracting a single progress marker."""
        output = "Working... <progress>Completed step 1</progress> Done."
        markers = extract_progress_markers(output)
        assert markers == ["Completed step 1"]

    def test_extract_multiple_markers(self):
        """Test extracting multiple progress markers."""
        output = """
        <progress>Started task</progress>
        Working on implementation...
        <progress>Completed implementation</progress>
        Running tests...
        <progress>Tests passing</progress>
        """
        markers = extract_progress_markers(output)
        assert markers == ["Started task", "Completed implementation", "Tests passing"]

    def test_extract_no_markers(self):
        """Test with no progress markers."""
        output = "Just regular output without any markers."
        markers = extract_progress_markers(output)
        assert markers == []

    def test_extract_empty_output(self):
        """Test with empty output."""
        assert extract_progress_markers("") == []
        assert extract_progress_markers(None) == []

    def test_extract_case_insensitive(self):
        """Test that extraction is case insensitive."""
        output = "<PROGRESS>Upper case</PROGRESS> <Progress>Mixed</Progress>"
        markers = extract_progress_markers(output)
        assert markers == ["Upper case", "Mixed"]

    def test_extract_multiline_content(self):
        """Test extracting markers with multiline content."""
        output = """<progress>
        Line 1
        Line 2
        </progress>"""
        markers = extract_progress_markers(output)
        assert len(markers) == 1
        assert "Line 1" in markers[0]
        assert "Line 2" in markers[0]

    def test_extract_strips_whitespace(self):
        """Test that extracted markers have whitespace stripped."""
        output = "<progress>  spaces around  </progress>"
        markers = extract_progress_markers(output)
        assert markers == ["spaces around"]

    def test_extract_empty_marker_skipped(self):
        """Test that empty markers are skipped."""
        output = "<progress></progress> <progress>  </progress> <progress>real</progress>"
        markers = extract_progress_markers(output)
        assert markers == ["real"]

    def test_has_progress_markers_true(self):
        """Test has_progress_markers returns True when markers exist."""
        output = "Some output <progress>marker</progress> more output"
        assert has_progress_markers(output) is True

    def test_has_progress_markers_false(self):
        """Test has_progress_markers returns False when no markers."""
        output = "No markers here"
        assert has_progress_markers(output) is False

    def test_count_progress_markers(self):
        """Test counting progress markers."""
        output = "<progress>1</progress><progress>2</progress><progress>3</progress>"
        assert count_progress_markers(output) == 3


class TestProgressSignal:
    """Tests for ProgressSignal dataclass."""

    def test_default_values(self):
        """Test default signal values."""
        signal = ProgressSignal()
        assert signal.output_diff_score == 0.0
        assert signal.file_changes_score == 0.0
        assert signal.explicit_markers == []
        assert signal.checklist_progress == 0.0

    def test_custom_values(self):
        """Test signal with custom values."""
        signal = ProgressSignal(
            output_diff_score=0.5,
            file_changes_score=0.3,
            explicit_markers=["marker1", "marker2"],
            checklist_progress=0.75,
        )
        assert signal.output_diff_score == 0.5
        assert signal.file_changes_score == 0.3
        assert signal.explicit_markers == ["marker1", "marker2"]
        assert signal.checklist_progress == 0.75

    def test_serialization(self):
        """Test signal serialization to dict."""
        signal = ProgressSignal(
            output_diff_score=0.5,
            file_changes_score=0.3,
            explicit_markers=["marker1"],
            checklist_progress=0.25,
        )
        data = signal.to_dict()
        assert data["output_diff_score"] == 0.5
        assert data["file_changes_score"] == 0.3
        assert data["explicit_markers"] == ["marker1"]
        assert data["checklist_progress"] == 0.25

    def test_deserialization(self):
        """Test signal deserialization from dict."""
        data = {
            "output_diff_score": 0.7,
            "file_changes_score": 0.4,
            "explicit_markers": ["m1", "m2"],
            "checklist_progress": 0.5,
        }
        signal = ProgressSignal.from_dict(data)
        assert signal.output_diff_score == 0.7
        assert signal.file_changes_score == 0.4
        assert signal.explicit_markers == ["m1", "m2"]
        assert signal.checklist_progress == 0.5

    def test_deserialization_missing_fields(self):
        """Test deserialization with missing fields uses defaults."""
        data = {"output_diff_score": 0.5}
        signal = ProgressSignal.from_dict(data)
        assert signal.output_diff_score == 0.5
        assert signal.file_changes_score == 0.0
        assert signal.explicit_markers == []
        assert signal.checklist_progress == 0.0


class TestCalculateProgressScore:
    """Tests for progress score calculation."""

    def test_all_zero_signals(self):
        """Test with all zero signals."""
        signal = ProgressSignal()
        score = calculate_progress_score(signal)
        assert score == 0.0

    def test_all_max_signals(self):
        """Test with all maximum signals."""
        signal = ProgressSignal(
            output_diff_score=1.0,
            file_changes_score=1.0,
            explicit_markers=["m1", "m2"],  # 2 markers = 1.0 score (capped)
            checklist_progress=1.0,
        )
        score = calculate_progress_score(signal)
        assert score == 1.0

    def test_output_diff_only(self):
        """Test with only output diff signal."""
        signal = ProgressSignal(output_diff_score=1.0)
        score = calculate_progress_score(signal)
        assert score == pytest.approx(0.30)  # 30% weight

    def test_file_changes_only(self):
        """Test with only file changes signal."""
        signal = ProgressSignal(file_changes_score=1.0)
        score = calculate_progress_score(signal)
        assert score == pytest.approx(0.30)  # 30% weight

    def test_explicit_markers_only(self):
        """Test with only explicit markers."""
        # One marker = 0.5 score, with 25% weight = 0.125
        signal = ProgressSignal(explicit_markers=["marker"])
        score = calculate_progress_score(signal)
        assert score == pytest.approx(0.125)

        # Two markers = 1.0 score (capped), with 25% weight = 0.25
        signal = ProgressSignal(explicit_markers=["m1", "m2"])
        score = calculate_progress_score(signal)
        assert score == pytest.approx(0.25)

    def test_checklist_only(self):
        """Test with only checklist progress."""
        signal = ProgressSignal(checklist_progress=1.0)
        score = calculate_progress_score(signal)
        assert score == pytest.approx(0.15)  # 15% weight

    def test_partial_signals(self):
        """Test with partial signals."""
        signal = ProgressSignal(
            output_diff_score=0.5,  # 0.5 * 0.30 = 0.15
            file_changes_score=0.5,  # 0.5 * 0.30 = 0.15
            explicit_markers=[],  # 0
            checklist_progress=0.5,  # 0.5 * 0.15 = 0.075
        )
        score = calculate_progress_score(signal)
        assert score == pytest.approx(0.375)

    def test_custom_weights(self):
        """Test with custom weights."""
        signal = ProgressSignal(
            output_diff_score=1.0,
            file_changes_score=0.0,
            explicit_markers=[],
            checklist_progress=0.0,
        )
        # Default: 30% weight = 0.30
        default_score = calculate_progress_score(signal)
        assert default_score == pytest.approx(0.30)

        # Custom: 100% weight to output_diff
        custom_weights = {
            "output_diff": 1.0,
            "file_changes": 0.0,
            "explicit_markers": 0.0,
            "checklist": 0.0,
        }
        custom_score = calculate_progress_score(signal, custom_weights)
        assert custom_score == pytest.approx(1.0)

    def test_score_capped_at_1(self):
        """Test that score is capped at 1.0."""
        signal = ProgressSignal(
            output_diff_score=2.0,  # Invalid but should be handled
            file_changes_score=2.0,
            explicit_markers=["m1", "m2", "m3", "m4", "m5"],
            checklist_progress=2.0,
        )
        score = calculate_progress_score(signal)
        assert score <= 1.0

    def test_default_weights_sum_to_1(self):
        """Test that default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0)


class TestOutputDiffScore:
    """Tests for output diff scoring."""

    def test_first_iteration_always_progress(self):
        """Test that first iteration (empty prev) is always progress."""
        score = output_diff_score("", "Some output")
        assert score == 1.0

        score = output_diff_score(None, "Some output")
        assert score == 1.0

    def test_empty_current_output(self):
        """Test that empty current output is no progress."""
        score = output_diff_score("Previous output", "")
        assert score == 0.0

    def test_identical_outputs(self):
        """Test identical outputs have zero diff."""
        output = "The exact same output"
        score = output_diff_score(output, output)
        assert score == 0.0

    def test_completely_different_outputs(self):
        """Test completely different outputs have high diff."""
        prev = "aaaaaaaaaaaaaaaaaaaaaaaaaaa"
        curr = "zzzzzzzzzzzzzzzzzzzzzzzzzzz"
        score = output_diff_score(prev, curr)
        assert score > 0.9

    def test_partial_similarity(self):
        """Test partially similar outputs."""
        prev = "Working on step 1. Making progress."
        curr = "Working on step 2. Making progress."
        score = output_diff_score(prev, curr)
        # Should be between 0 and 1, with some similarity
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        """Test that comparison is case insensitive."""
        prev = "HELLO WORLD"
        curr = "hello world"
        score = output_diff_score(prev, curr)
        assert score == pytest.approx(0.0)

    def test_whitespace_normalized(self):
        """Test that whitespace is normalized."""
        prev = "hello    world"
        curr = "hello world"
        score = output_diff_score(prev, curr)
        assert score == pytest.approx(0.0)

    def test_timestamps_ignored(self):
        """Test that timestamps are ignored in comparison."""
        prev = "Log entry at 2024-01-01T12:00:00"
        curr = "Log entry at 2024-01-02T15:30:00"
        score = output_diff_score(prev, curr)
        # Should be very similar after timestamp removal
        assert score < 0.2


class TestNormalizeOutput:
    """Tests for output normalization."""

    def test_lowercases_text(self):
        """Test that text is lowercased."""
        result = _normalize_output("HELLO World")
        assert result == "hello world"

    def test_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        result = _normalize_output("hello    \n\t   world")
        assert result == "hello world"

    def test_removes_iso_timestamps(self):
        """Test that ISO timestamps are removed."""
        result = _normalize_output("event at 2024-01-15T14:30:00 happened")
        # The timestamp pattern should be removed
        assert "2024-01-15" not in result
        assert "14:30:00" not in result

    def test_removes_unix_timestamps(self):
        """Test that Unix timestamps are removed."""
        result = _normalize_output("timestamp 1704067200 recorded")
        assert "1704067200" not in result

    def test_empty_string(self):
        """Test empty string handling."""
        assert _normalize_output("") == ""
        assert _normalize_output(None) == ""


class TestParseGitDiffStat:
    """Tests for git diff stat parsing."""

    def test_parse_basic_diff(self):
        """Test parsing basic git diff --stat output."""
        output = """
 src/foo.py | 10 +++++-----
 src/bar.py |  5 +++++
 2 files changed, 10 insertions(+), 5 deletions(-)
"""
        stats = _parse_git_diff_stat(output)
        assert stats.files_changed == 2
        assert stats.insertions == 10
        assert stats.deletions == 5

    def test_parse_insertions_only(self):
        """Test parsing diff with only insertions."""
        output = """
 new_file.py | 50 ++++++++++++++
 1 file changed, 50 insertions(+)
"""
        stats = _parse_git_diff_stat(output)
        assert stats.files_changed == 1
        assert stats.insertions == 50
        assert stats.deletions == 0

    def test_parse_deletions_only(self):
        """Test parsing diff with only deletions."""
        output = """
 old_file.py | 30 --------------
 1 file changed, 30 deletions(-)
"""
        stats = _parse_git_diff_stat(output)
        assert stats.files_changed == 1
        assert stats.insertions == 0
        assert stats.deletions == 30

    def test_parse_empty_output(self):
        """Test parsing empty output."""
        stats = _parse_git_diff_stat("")
        assert stats.files_changed == 0
        assert stats.insertions == 0
        assert stats.deletions == 0

    def test_parse_no_changes(self):
        """Test parsing when no files changed."""
        stats = _parse_git_diff_stat("no changes found")
        assert stats.files_changed == 0
        assert stats.insertions == 0
        assert stats.deletions == 0


class TestFileChangesScore:
    """Tests for file changes scoring."""

    def test_no_changes(self):
        """Test with no git changes."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
            )
            score = file_changes_score(Path("/fake/path"))
            assert score == 0.0

    def test_small_changes(self):
        """Test with small changes (< 100 lines)."""
        with patch(
            "src.um_agent_coder.harness.autonomous.progress_detector.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="1 file changed, 10 insertions(+), 5 deletions(-)",
            )
            score = file_changes_score(Path("/fake/path"))
            # Both unstaged and staged diffs are checked, so 15 * 2 = 30 / 100 = 0.3
            assert score == pytest.approx(0.30)

    def test_large_changes(self):
        """Test with large changes (>= 100 lines)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="5 files changed, 150 insertions(+), 50 deletions(-)",
            )
            score = file_changes_score(Path("/fake/path"))
            assert score == 1.0  # Capped at 1.0

    def test_git_error(self):
        """Test with git error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            score = file_changes_score(Path("/fake/path"))
            assert score == 0.0

    def test_timeout(self):
        """Test with timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=10)
            score = file_changes_score(Path("/fake/path"))
            assert score == 0.0

    def test_git_not_found(self):
        """Test when git is not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            score = file_changes_score(Path("/fake/path"))
            assert score == 0.0


class TestProgressDetector:
    """Tests for ProgressDetector class."""

    def test_init_defaults(self):
        """Test detector initialization with defaults."""
        detector = ProgressDetector()
        assert detector.workspace == Path.cwd()
        assert detector.weights == DEFAULT_WEIGHTS
        assert detector.no_progress_threshold == 0.15

    def test_init_custom(self):
        """Test detector initialization with custom values."""
        custom_weights = {"output_diff": 0.5, "file_changes": 0.5, "explicit_markers": 0.0, "checklist": 0.0}
        detector = ProgressDetector(
            workspace=Path("/custom/path"),
            weights=custom_weights,
            no_progress_threshold=0.2,
        )
        assert detector.workspace == Path("/custom/path")
        assert detector.weights == custom_weights
        assert detector.no_progress_threshold == 0.2

    def test_detect_with_progress_markers(self):
        """Test detection with progress markers."""
        detector = ProgressDetector()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            signal = detector.detect(
                prev_output="Working...",
                curr_output="<progress>Completed step 1</progress>",
            )
        assert signal.explicit_markers == ["Completed step 1"]

    def test_detect_with_checklist(self):
        """Test detection with checklist progress."""
        detector = ProgressDetector()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            signal = detector.detect(
                prev_output="prev",
                curr_output="curr",
                checklist_completed=3,
                checklist_total=10,
            )
        assert signal.checklist_progress == 0.3

    def test_calculate_score(self):
        """Test score calculation via detector."""
        detector = ProgressDetector()
        signal = ProgressSignal(output_diff_score=0.5, file_changes_score=0.5)
        score = detector.calculate_score(signal)
        assert score == pytest.approx(0.30)  # 0.5 * 0.3 + 0.5 * 0.3

    def test_has_progress_true(self):
        """Test has_progress returns True above threshold."""
        detector = ProgressDetector(no_progress_threshold=0.15)
        signal = ProgressSignal(output_diff_score=1.0)  # Score = 0.30
        assert detector.has_progress(signal) is True

    def test_has_progress_false(self):
        """Test has_progress returns False below threshold."""
        detector = ProgressDetector(no_progress_threshold=0.15)
        signal = ProgressSignal(output_diff_score=0.1)  # Score = 0.03
        assert detector.has_progress(signal) is False

    def test_detect_and_score(self):
        """Test combined detect_and_score method."""
        detector = ProgressDetector()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            signal, score, has_prog = detector.detect_and_score(
                prev_output="",
                curr_output="<progress>Done</progress>",
            )
        assert signal.explicit_markers == ["Done"]
        assert score > 0
        assert isinstance(has_prog, bool)


class TestProgressDetectorIntegration:
    """Integration tests for progress detection with real git repo."""

    def test_real_git_repo(self):
        """Test with a real (temporary) git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir,
                capture_output=True,
            )

            # Create initial file and commit
            (tmpdir / "test.py").write_text("print('hello')")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "initial"],
                cwd=tmpdir,
                capture_output=True,
            )

            # Initially no changes
            score = file_changes_score(tmpdir)
            assert score == 0.0

            # Make changes
            (tmpdir / "test.py").write_text("print('hello world')\nprint('more lines')\n")

            # Now should detect changes
            score = file_changes_score(tmpdir)
            assert score > 0.0


class TestNoProgressThreshold:
    """Tests for no-progress threshold behavior."""

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        detector = ProgressDetector(no_progress_threshold=0.15)

        # Just below threshold
        signal = ProgressSignal(output_diff_score=0.49)  # 0.49 * 0.3 = 0.147
        assert detector.has_progress(signal) is False

        # Just above threshold
        signal = ProgressSignal(output_diff_score=0.51)  # 0.51 * 0.3 = 0.153
        assert detector.has_progress(signal) is True

    def test_custom_threshold(self):
        """Test with custom threshold."""
        detector = ProgressDetector(no_progress_threshold=0.5)
        signal = ProgressSignal(output_diff_score=1.0)  # Score = 0.30
        assert detector.has_progress(signal) is False  # Below 0.5 threshold
