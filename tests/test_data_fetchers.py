import unittest
from unittest.mock import MagicMock, patch
from um_agent_coder.orchestrator.data_fetchers import YahooFinanceFetcher, FetchResult

class TestYahooFinanceFetcher(unittest.TestCase):
    @patch('um_agent_coder.orchestrator.data_fetchers.time.sleep')
    def test_fetch_multiple_caching_optimization(self, mock_sleep):
        fetcher = YahooFinanceFetcher()
        # Mocking the session to avoid warnings or errors during init if checks happen there
        # But YahooFinanceFetcher only creates session in init, which is fine.

        # Mock the fetch method
        fetcher.fetch = MagicMock()

        # Mock results: 2 cached, 1 fresh
        # The fetcher iterates over tickers. We need side_effect to return these in order.
        results = [
            FetchResult(success=True, data={}, source="yahoo", cached=True),
            FetchResult(success=True, data={}, source="yahoo", cached=True),
            FetchResult(success=True, data={}, source="yahoo", cached=False),
        ]
        fetcher.fetch.side_effect = results

        tickers = ["T1", "T2", "T3"]
        fetcher.fetch_multiple(tickers)

        # Should only sleep once (for the non-cached result)
        self.assertEqual(mock_sleep.call_count, 1)

if __name__ == '__main__':
    unittest.main()
