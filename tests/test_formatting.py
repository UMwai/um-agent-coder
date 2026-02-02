import pytest
from um_agent_coder.utils.formatting import format_compact_number


class TestFormatCompactNumber:
    def test_small_numbers(self):
        """Test formatting numbers smaller than 1000."""
        assert format_compact_number(0) == "0"
        assert format_compact_number(5) == "5"
        assert format_compact_number(999) == "999"
        assert format_compact_number(-500) == "-500"

    def test_thousands(self):
        """Test formatting numbers in thousands."""
        assert format_compact_number(1000) == "1k"
        assert format_compact_number(1200) == "1.2k"
        assert format_compact_number(1500) == "1.5k"
        assert format_compact_number(9999) == "10k"  # Rounding might happen depending on implementation, let's check
        # With default decimals=1: 9999 / 1000 = 9.999 -> "10.0" -> "10k"
        assert format_compact_number(9999) == "10k"
        assert format_compact_number(10500) == "10.5k"

    def test_millions(self):
        """Test formatting numbers in millions."""
        assert format_compact_number(1_000_000) == "1M"
        assert format_compact_number(1_200_000) == "1.2M"
        assert format_compact_number(1_500_000) == "1.5M"
        assert format_compact_number(12_500_000) == "12.5M"

    def test_billions(self):
        """Test formatting numbers in billions."""
        assert format_compact_number(1_000_000_000) == "1B"
        assert format_compact_number(1_500_000_000) == "1.5B"

    def test_trillions(self):
        """Test formatting numbers in trillions."""
        assert format_compact_number(1_000_000_000_000) == "1T"

    def test_decimals_parameter(self):
        """Test the decimals parameter."""
        assert format_compact_number(1234, decimals=2) == "1.23k"
        assert format_compact_number(1234, decimals=0) == "1k"
        assert format_compact_number(1_234_567, decimals=2) == "1.23M"

    def test_negative_numbers(self):
        """Test formatting negative large numbers."""
        assert format_compact_number(-1200) == "-1.2k"
        assert format_compact_number(-1_500_000) == "-1.5M"

    def test_floats(self):
        """Test formatting float inputs."""
        assert format_compact_number(1200.5) == "1.2k"
        assert format_compact_number(1_500_000.123) == "1.5M"
