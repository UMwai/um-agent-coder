from um_agent_coder.utils.formatting import format_compact_number

def test_format_compact_number():
    assert format_compact_number(500) == "500"
    assert format_compact_number(1000) == "1K"
    assert format_compact_number(1200) == "1.2K"
    assert format_compact_number(1500) == "1.5K"
    assert format_compact_number(1000000) == "1M"
    assert format_compact_number(1500000) == "1.5M"
    assert format_compact_number(2000000000) == "2B"
