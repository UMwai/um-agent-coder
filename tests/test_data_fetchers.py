import time
from unittest.mock import patch

from um_agent_coder.orchestrator.data_fetchers import FetchResult, YahooFinanceFetcher


class TestYahooFinanceFetcher:
    def test_fetch_multiple_skips_delay_when_cached(self):
        """Test that rate limiting sleep is skipped when data is cached."""
        fetcher = YahooFinanceFetcher()

        # Mock fetch to return a "cached" result
        with patch.object(fetcher, "fetch") as mock_fetch:
            mock_fetch.return_value = FetchResult(
                success=True, data={"some": "data"}, source="yahoo_finance", cached=True
            )

            tickers = ["AAPL", "GOOG", "MSFT"]

            start_time = time.time()
            fetcher.fetch_multiple(tickers)
            end_time = time.time()

            duration = end_time - start_time

            # If optimizations are working, this should be very fast (< 0.1s)
            # If not, it would be > 1.5s
            # We assert < 0.5s to be safe but definitely faster than 1.5s
            assert (
                duration < 0.5
            ), f"fetch_multiple took too long ({duration}s), possibly sleeping on cached results"

            assert mock_fetch.call_count == 3

    def test_fetch_multiple_respects_delay_when_not_cached(self):
        """Test that rate limiting sleep is applied when data is not cached."""
        fetcher = YahooFinanceFetcher()

        # Mock fetch to return a FRESH result (not cached)
        with patch.object(fetcher, "fetch") as mock_fetch:
            mock_fetch.return_value = FetchResult(
                success=True, data={"some": "data"}, source="yahoo_finance", cached=False
            )

            tickers = ["AAPL", "GOOG"]

            start_time = time.time()
            fetcher.fetch_multiple(tickers)
            end_time = time.time()

            duration = end_time - start_time

            # Should sleep 0.5s * 2 = 1.0s
            # We check for at least 0.9s to account for slight timing variations
            assert duration >= 0.9, f"fetch_multiple was too fast ({duration}s), should have slept"
