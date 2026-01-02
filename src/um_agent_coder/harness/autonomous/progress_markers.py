"""Extract progress markers from output.

Progress markers are XML-style tags that agents use to indicate incremental
progress toward a goal: <progress>description</progress>
"""

import re

# Pattern to match <progress>...</progress> tags
PROGRESS_PATTERN = re.compile(
    r"<progress>(.*?)</progress>",
    re.IGNORECASE | re.DOTALL,
)


def extract_progress_markers(output: str) -> list[str]:
    """Extract all progress markers from output.

    Args:
        output: The output text to search for progress markers.

    Returns:
        List of progress marker contents, in order of appearance.

    Example:
        >>> extract_progress_markers("Working... <progress>Completed step 1</progress>")
        ['Completed step 1']
    """
    if not output:
        return []

    matches = PROGRESS_PATTERN.findall(output)
    # Strip whitespace from each match
    return [m.strip() for m in matches if m.strip()]


def has_progress_markers(output: str) -> bool:
    """Check if output contains any progress markers.

    Args:
        output: The output text to check.

    Returns:
        True if at least one progress marker is found.
    """
    if not output:
        return False
    return bool(PROGRESS_PATTERN.search(output))


def count_progress_markers(output: str) -> int:
    """Count the number of progress markers in output.

    Args:
        output: The output text to count markers in.

    Returns:
        Number of progress markers found.
    """
    return len(extract_progress_markers(output))
