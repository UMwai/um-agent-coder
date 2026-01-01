"""
Data Fetchers - Integrations with external data sources.

Provides standardized interfaces for:
- SEC EDGAR (financial filings)
- Yahoo Finance (stock data)
- ClinicalTrials.gov (trial data)
- News APIs
- PubMed (research papers)
"""

import hashlib
import json
import os
import time
import re
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


@dataclass
class FetchResult:
    """Result from a data fetch operation."""
    success: bool
    data: Any
    source: str
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    cached: bool = False


class DataFetcher(ABC):
    """Abstract base class for data fetchers."""

    def __init__(self, cache_dir: str = ".data_cache", cache_ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._memory_cache: Dict[str, FetchResult] = {}

        # Use a persistent session for connection pooling
        self.session = requests.Session() if requests else None

    @abstractmethod
    def fetch(self, **kwargs) -> FetchResult:
        """Fetch data from the source."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of this data source."""
        pass

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        sorted_items = sorted(kwargs.items())
        key_str = "_".join(f"{k}={v}" for k, v in sorted_items)
        # Use SHA256 for stable hashing across runs/platforms
        hash_digest = hashlib.sha256(key_str.encode()).hexdigest()
        return f"{self.source_name}_{hash_digest}"

    def _check_cache(self, cache_key: str) -> Optional[FetchResult]:
        """Check if cached data exists and is fresh."""
        # Check in-memory cache first (L1)
        if cache_key in self._memory_cache:
            result = self._memory_cache[cache_key]
            fetched_at = datetime.fromisoformat(result.fetched_at)
            if datetime.now() - fetched_at < self.cache_ttl:
                return result
            else:
                del self._memory_cache[cache_key]

        # Check disk cache (L2)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if datetime.now() - fetched_at < self.cache_ttl:
                result = FetchResult(
                    success=True,
                    data=cached["data"],
                    source=self.source_name,
                    fetched_at=cached["fetched_at"],
                    cached=True
                )
                # Populate memory cache
                self._memory_cache[cache_key] = result
                return result
        except Exception:
            pass

        return None

    def _save_cache(self, cache_key: str, result: FetchResult):
        """Save result to cache."""
        if not result.success:
            return

        # Update memory cache
        self._memory_cache[cache_key] = result

        # Update disk cache
        cache_path = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "data": result.data,
                    "fetched_at": result.fetched_at
                }, f)
        except Exception:
            pass


class SECEdgarFetcher(DataFetcher):
    """
    Fetches data from SEC EDGAR.

    Supports:
    - Company filings (10-K, 10-Q, 8-K)
    - Company facts (financials)
    - Full-text search
    """

    BASE_URL = "https://data.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, user_agent: str = "um-agent-coder/1.0", **kwargs):
        super().__init__(**kwargs)
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json"
        }
        if self.session:
            self.session.headers.update(self.headers)

    @property
    def source_name(self) -> str:
        return "sec_edgar"

    def fetch(
        self,
        ticker: Optional[str] = None,
        cik: Optional[str] = None,
        filing_type: str = "10-K",
        limit: int = 10,
        **kwargs
    ) -> FetchResult:
        """
        Fetch SEC filings.

        Args:
            ticker: Stock ticker symbol
            cik: SEC CIK number (alternative to ticker)
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            limit: Max number of filings to return
        """
        if not self.session:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error="requests library not installed"
            )

        # Check cache
        cache_key = self._get_cache_key(ticker=ticker, cik=cik, filing_type=filing_type)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # First, get CIK if we have ticker
            if ticker and not cik:
                cik = self._ticker_to_cik(ticker)
                if not cik:
                    return FetchResult(
                        success=False,
                        data=None,
                        source=self.source_name,
                        error=f"Could not find CIK for ticker {ticker}"
                    )

            # Format CIK (pad to 10 digits)
            cik_padded = str(cik).zfill(10)

            # Fetch submissions
            url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Filter filings by type
            filings = []
            recent = data.get("filings", {}).get("recent", {})

            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            descriptions = recent.get("primaryDocument", [])

            for i, form in enumerate(forms[:100]):  # Check first 100
                if filing_type.upper() in form.upper():
                    filings.append({
                        "form": form,
                        "date": dates[i] if i < len(dates) else None,
                        "accession": accessions[i] if i < len(accessions) else None,
                        "document": descriptions[i] if i < len(descriptions) else None
                    })
                    if len(filings) >= limit:
                        break

            result = FetchResult(
                success=True,
                data={
                    "company": data.get("name"),
                    "cik": cik,
                    "ticker": ticker,
                    "filings": filings
                },
                source=self.source_name
            )

            self._save_cache(cache_key, result)
            return result

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error=str(e)
            )

    def _ticker_to_cik(self, ticker: str) -> Optional[str]:
        """Convert ticker symbol to CIK."""
        try:
            url = f"{self.BASE_URL}/submissions/CIK-lookup-data.txt"
            # This would need proper implementation
            # For now, return None to indicate lookup needed
            return None
        except Exception:
            return None

    def fetch_company_facts(self, cik: str) -> FetchResult:
        """Fetch company financial facts."""
        if not self.session:
            return FetchResult(
                success=False, data=None, source=self.source_name,
                error="requests library not installed"
            )

        try:
            cik_padded = str(cik).zfill(10)
            url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            return FetchResult(
                success=True,
                data=response.json(),
                source=self.source_name
            )

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error=str(e)
            )


class YahooFinanceFetcher(DataFetcher):
    """
    Fetches stock data from Yahoo Finance.

    Note: Uses unofficial API endpoints. For production, consider yfinance library.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance"

    @property
    def source_name(self) -> str:
        return "yahoo_finance"

    def fetch(
        self,
        ticker: str,
        include_financials: bool = True,
        include_profile: bool = True,
        **kwargs
    ) -> FetchResult:
        """
        Fetch stock data for a ticker.

        Args:
            ticker: Stock ticker symbol
            include_financials: Include financial statements
            include_profile: Include company profile
        """
        if not self.session:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error="requests library not installed"
            )

        cache_key = self._get_cache_key(ticker=ticker)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Fetch quote data
            modules = ["price", "summaryDetail"]
            if include_profile:
                modules.append("assetProfile")
            if include_financials:
                modules.extend(["incomeStatementHistory", "balanceSheetHistory", "cashflowStatementHistory"])

            url = f"{self.BASE_URL}/chart/{ticker}"
            params = {
                "modules": ",".join(modules),
                "interval": "1d",
                "range": "1mo"
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            chart_data = response.json()

            # Also try to get key statistics
            quote_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            quote_params = {"modules": ",".join(modules)}

            quote_response = self.session.get(quote_url, params=quote_params, timeout=30)

            data = {
                "ticker": ticker,
                "chart": chart_data.get("chart", {}).get("result", [{}])[0],
            }

            if quote_response.ok:
                quote_data = quote_response.json()
                data["summary"] = quote_data.get("quoteSummary", {}).get("result", [{}])[0]

            result = FetchResult(
                success=True,
                data=data,
                source=self.source_name
            )

            self._save_cache(cache_key, result)
            return result

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error=str(e)
            )

    def fetch_multiple(self, tickers: List[str]) -> Dict[str, FetchResult]:
        """Fetch data for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.fetch(ticker)
            time.sleep(0.5)  # Rate limiting
        return results


class ClinicalTrialsFetcher(DataFetcher):
    """
    Fetches clinical trial data from ClinicalTrials.gov.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    @property
    def source_name(self) -> str:
        return "clinical_trials"

    def fetch(
        self,
        sponsor: Optional[str] = None,
        condition: Optional[str] = None,
        phase: Optional[str] = None,
        status: str = "RECRUITING,ACTIVE_NOT_RECRUITING",
        limit: int = 50,
        **kwargs
    ) -> FetchResult:
        """
        Fetch clinical trials.

        Args:
            sponsor: Company/organization sponsor name
            condition: Disease or condition
            phase: Trial phase (PHASE1, PHASE2, PHASE3, PHASE4)
            status: Trial status filter
            limit: Max results
        """
        if not self.session:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error="requests library not installed"
            )

        cache_key = self._get_cache_key(sponsor=sponsor, condition=condition, phase=phase)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            url = f"{self.BASE_URL}/studies"

            params = {
                "format": "json",
                "pageSize": min(limit, 100),
                "countTotal": "true"
            }

            # Build query
            query_parts = []
            if sponsor:
                query_parts.append(f"AREA[LeadSponsorName]{sponsor}")
            if condition:
                query_parts.append(f"AREA[Condition]{condition}")

            if query_parts:
                params["query.term"] = " AND ".join(query_parts)

            if phase:
                params["filter.advanced"] = f"AREA[Phase]{phase}"

            if status:
                params["filter.overallStatus"] = status

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract relevant trial info
            trials = []
            for study in data.get("studies", []):
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

                trials.append({
                    "nct_id": id_module.get("nctId"),
                    "title": id_module.get("briefTitle"),
                    "status": status_module.get("overallStatus"),
                    "phase": design_module.get("phases", []),
                    "sponsor": sponsor_module.get("leadSponsor", {}).get("name"),
                    "start_date": status_module.get("startDateStruct", {}).get("date"),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date")
                })

            result = FetchResult(
                success=True,
                data={
                    "total_count": data.get("totalCount", len(trials)),
                    "trials": trials
                },
                source=self.source_name
            )

            self._save_cache(cache_key, result)
            return result

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error=str(e)
            )


class NewsFetcher(DataFetcher):
    """
    Fetches news articles.

    Supports multiple backends:
    - NewsAPI.org (requires API key)
    - Google News RSS (free)
    """

    # Pre-compiled regex patterns for performance
    ITEM_PATTERN = re.compile(r'<item>(.*?)</item>', re.DOTALL)
    TITLE_PATTERN = re.compile(r'<title>(.*?)</title>')
    LINK_PATTERN = re.compile(r'<link>(.*?)</link>')
    PUB_DATE_PATTERN = re.compile(r'<pubDate>(.*?)</pubDate>')

    @property
    def source_name(self) -> str:
        return "news"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("NEWS_API_KEY")

    def fetch(
        self,
        query: str,
        days_back: int = 7,
        limit: int = 20,
        **kwargs
    ) -> FetchResult:
        """
        Fetch news articles.

        Args:
            query: Search query
            days_back: How many days back to search
            limit: Max articles
        """
        if not self.session:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error="requests library not installed"
            )

        cache_key = self._get_cache_key(query=query, days_back=days_back)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Try NewsAPI if we have a key
        if self.api_key:
            return self._fetch_newsapi(query, days_back, limit)

        # Fall back to Google News RSS
        return self._fetch_google_news(query, limit)

    def _fetch_newsapi(self, query: str, days_back: int, limit: int) -> FetchResult:
        """Fetch from NewsAPI.org."""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": min(limit, 100),
                "apiKey": self.api_key
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            articles = [
                {
                    "title": a.get("title"),
                    "source": a.get("source", {}).get("name"),
                    "published": a.get("publishedAt"),
                    "url": a.get("url"),
                    "description": a.get("description")
                }
                for a in data.get("articles", [])
            ]

            result = FetchResult(
                success=True,
                data={"articles": articles, "total": data.get("totalResults", len(articles))},
                source=self.source_name
            )

            self._save_cache(self._get_cache_key(query=query, days_back=days_back), result)
            return result

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error=str(e)
            )

    def _fetch_google_news(self, query: str, limit: int) -> FetchResult:
        """Fetch from Google News RSS."""
        try:
            import urllib.parse

            encoded_query = urllib.parse.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse RSS (using pre-compiled regex for performance)
            articles = []

            # Use finditer with islice to avoid processing the whole document if not needed
            for match in itertools.islice(self.ITEM_PATTERN.finditer(response.text), limit):
                item = match.group(1)
                title = self.TITLE_PATTERN.search(item)
                link = self.LINK_PATTERN.search(item)
                pub_date = self.PUB_DATE_PATTERN.search(item)

                articles.append({
                    "title": title.group(1) if title else None,
                    "url": link.group(1) if link else None,
                    "published": pub_date.group(1) if pub_date else None,
                    "source": "Google News"
                })

            return FetchResult(
                success=True,
                data={"articles": articles},
                source=self.source_name
            )

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                error=str(e)
            )


class DataFetcherRegistry:
    """Registry of available data fetchers."""

    def __init__(self):
        self.fetchers: Dict[str, DataFetcher] = {}

        # Register default fetchers
        self.register("sec_edgar", SECEdgarFetcher())
        self.register("yahoo_finance", YahooFinanceFetcher())
        self.register("clinical_trials", ClinicalTrialsFetcher())
        self.register("news", NewsFetcher())

    def register(self, name: str, fetcher: DataFetcher):
        """Register a data fetcher."""
        self.fetchers[name] = fetcher

    def get(self, name: str) -> Optional[DataFetcher]:
        """Get a fetcher by name."""
        return self.fetchers.get(name)

    def fetch(self, source: str, **kwargs) -> FetchResult:
        """Fetch from a named source."""
        fetcher = self.get(source)
        if not fetcher:
            return FetchResult(
                success=False,
                data=None,
                source=source,
                error=f"Unknown data source: {source}"
            )
        return fetcher.fetch(**kwargs)

    def list_sources(self) -> List[str]:
        """List available data sources."""
        return list(self.fetchers.keys())
