import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from um_agent_coder.utils.formatting import format_compact_number


def test_format_compact_number():
    assert format_compact_number(100) == "100"
    assert format_compact_number(999) == "999"
    assert format_compact_number(1000) == "1k"
    assert format_compact_number(1200) == "1.2k"
    assert format_compact_number(1500) == "1.5k"
    assert format_compact_number(10000) == "10k"
    assert format_compact_number(100000) == "100k"
    assert format_compact_number(1000000) == "1M"
    assert format_compact_number(1200000) == "1.2M"
    assert format_compact_number(1500000000) == "1.5B"
    assert format_compact_number(2000000000000) == "2T"
