import pytest
from um_agent_coder.utils.formatting import format_compact_number


def test_format_compact_number_small():
    assert format_compact_number(500) == "500"
    assert format_compact_number(999) == "999"
    assert format_compact_number(0) == "0"


def test_format_compact_number_thousands():
    assert format_compact_number(1000) == "1k"
    assert format_compact_number(1200) == "1.2k"
    assert format_compact_number(1500) == "1.5k"
    assert format_compact_number(9900) == "9.9k"


def test_format_compact_number_millions():
    assert format_compact_number(1_000_000) == "1M"
    assert format_compact_number(1_500_000) == "1.5M"
    assert format_compact_number(2_000_000) == "2M"
    assert format_compact_number(12_500_000) == "12.5M"


def test_format_compact_number_billions():
    assert format_compact_number(1_000_000_000) == "1B"
    assert format_compact_number(2_500_000_000) == "2.5B"


def test_format_compact_number_floats():
    assert format_compact_number(1200.0) == "1.2k"
    assert format_compact_number(500.5) == "500.5"
