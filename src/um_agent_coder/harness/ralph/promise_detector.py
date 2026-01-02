"""
Promise detection for ralph loop completion.

Detects completion promises in CLI executor output to determine
when a ralph loop task has successfully completed.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Result of promise detection."""
    found: bool
    promise_text: Optional[str] = None
    match_type: Optional[str] = None  # "xml" or "plain"
    position: Optional[int] = None  # Start position of match in output


class PromiseDetector:
    """Detects completion promises in executor output.

    Supports two formats:
    1. XML-style: <promise>COMPLETION_TEXT</promise>
    2. Plain text: Exact string matching

    Example:
        detector = PromiseDetector(promise="TASK_COMPLETE")
        result = detector.detect("Output with <promise>TASK_COMPLETE</promise>")
        assert result.found == True
        assert result.promise_text == "TASK_COMPLETE"
    """

    # Regex for XML-style promise tags
    XML_PATTERN = re.compile(
        r'<promise>\s*(.+?)\s*</promise>',
        re.IGNORECASE | re.DOTALL
    )

    def __init__(
        self,
        promise: str,
        case_sensitive: bool = False,
        require_xml_format: bool = False,
    ):
        """Initialize the promise detector.

        Args:
            promise: The promise text to look for
            case_sensitive: Whether matching is case-sensitive (default: False)
            require_xml_format: Only match XML-style promises (default: False)
        """
        self.promise = promise.strip()
        self.case_sensitive = case_sensitive
        self.require_xml_format = require_xml_format

        # Pre-compile the normalized promise for comparison
        self._normalized_promise = self.promise if case_sensitive else self.promise.lower()

    def detect(self, output: str) -> DetectionResult:
        """Detect if the promise is present in the output.

        Args:
            output: The executor output to scan

        Returns:
            DetectionResult with found=True if promise detected,
            including the matched promise text and match type.
        """
        if not output:
            return DetectionResult(found=False)

        # First, try XML-style detection
        xml_result = self._detect_xml(output)
        if xml_result.found:
            return xml_result

        # If XML not required, try plain text detection
        if not self.require_xml_format:
            plain_result = self._detect_plain(output)
            if plain_result.found:
                return plain_result

        return DetectionResult(found=False)

    def _detect_xml(self, output: str) -> DetectionResult:
        """Detect XML-style promise tags."""
        matches = self.XML_PATTERN.finditer(output)

        for match in matches:
            promise_content = match.group(1).strip()
            normalized_content = promise_content if self.case_sensitive else promise_content.lower()

            if normalized_content == self._normalized_promise:
                return DetectionResult(
                    found=True,
                    promise_text=promise_content,
                    match_type="xml",
                    position=match.start(),
                )

        return DetectionResult(found=False)

    def _detect_plain(self, output: str) -> DetectionResult:
        """Detect plain text promise."""
        search_output = output if self.case_sensitive else output.lower()
        search_promise = self._normalized_promise

        position = search_output.find(search_promise)
        if position != -1:
            # Extract the actual matched text from original output
            actual_text = output[position:position + len(self.promise)]
            return DetectionResult(
                found=True,
                promise_text=actual_text,
                match_type="plain",
                position=position,
            )

        return DetectionResult(found=False)

    def detect_any_promise(self, output: str) -> DetectionResult:
        """Detect any promise tag in output, regardless of content.

        Useful for finding promises even when the expected text is unknown.

        Args:
            output: The executor output to scan

        Returns:
            DetectionResult with the first promise found, or found=False
        """
        if not output:
            return DetectionResult(found=False)

        match = self.XML_PATTERN.search(output)
        if match:
            return DetectionResult(
                found=True,
                promise_text=match.group(1).strip(),
                match_type="xml",
                position=match.start(),
            )

        return DetectionResult(found=False)

    @staticmethod
    def extract_all_promises(output: str) -> list[str]:
        """Extract all promise tags from output.

        Args:
            output: The executor output to scan

        Returns:
            List of all promise texts found (may be empty)
        """
        if not output:
            return []

        matches = PromiseDetector.XML_PATTERN.findall(output)
        return [m.strip() for m in matches]
