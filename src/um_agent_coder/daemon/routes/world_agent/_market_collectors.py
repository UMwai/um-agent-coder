"""Market event collectors for the World Agent.

Collect financial market data and convert into Events that the orient
layer can match against alpha-seeking goals. These run on Cloud Run
so they use free/public APIs (no local repos needed).

Collectors:
- MarketMoversCollector: Large price moves, volume spikes, unusual activity
- NewsCollector: Financial news via Google News RSS (free, no key needed)
- EarningsCollector: Upcoming earnings from Yahoo Finance
- VolatilityCollector: VIX, put/call ratios, IV regime changes
- SECFilingsCollector: Recent 8-K, 13F, insider transactions
- CryptoFundingCollector: Funding rates for basis trade opportunities
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from um_agent_coder.daemon.routes.world_agent._collectors import EventCollector
from um_agent_coder.daemon.routes.world_agent.models import (
    Event,
    EventCategory,
    EventSeverity,
)

logger = logging.getLogger(__name__)

# Symbols to monitor
EQUITY_WATCHLIST = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "CRM",
    "NFLX",
]
CRYPTO_WATCHLIST = ["bitcoin", "ethereum", "solana"]
VOL_SYMBOLS = ["^VIX"]


class MarketMoversCollector(EventCollector):
    """Detect large price moves and volume spikes via Yahoo Finance."""

    def __init__(
        self,
        symbols: List[str] | None = None,
        move_threshold_pct: float = 2.0,
    ):
        self._symbols = symbols or EQUITY_WATCHLIST
        self._move_threshold = move_threshold_pct

    def source_id(self) -> str:
        return "market.movers"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        events: List[Event] = []
        now = datetime.now(timezone.utc)

        try:
            quotes = await self._fetch_quotes()
            for q in quotes:
                symbol = q.get("symbol", "")
                change_pct = q.get("regularMarketChangePercent", 0)
                price = q.get("regularMarketPrice", 0)
                volume = q.get("regularMarketVolume", 0)
                avg_volume = q.get("averageDailyVolume3Month", 1)
                vol_ratio = volume / avg_volume if avg_volume else 0

                # Large price move
                if abs(change_pct) >= self._move_threshold:
                    direction = "up" if change_pct > 0 else "down"
                    severity = (
                        EventSeverity.urgent if abs(change_pct) >= 5 else EventSeverity.notable
                    )
                    events.append(
                        Event(
                            id=f"mkt-{uuid.uuid4().hex[:8]}",
                            source=self.source_id(),
                            timestamp=now,
                            category=EventCategory.financial,
                            severity=severity,
                            title=f"{symbol} {direction} {abs(change_pct):.1f}% to ${price:.2f}",
                            body=f"Volume: {volume:,} ({vol_ratio:.1f}x avg)",
                            metadata={
                                "symbol": symbol,
                                "change_pct": round(change_pct, 2),
                                "price": price,
                                "volume": volume,
                                "vol_ratio": round(vol_ratio, 1),
                                "scan_type": "price_move",
                            },
                        )
                    )

                # Volume spike (>2x average)
                if vol_ratio >= 2.0 and abs(change_pct) < self._move_threshold:
                    events.append(
                        Event(
                            id=f"mkt-{uuid.uuid4().hex[:8]}",
                            source=self.source_id(),
                            timestamp=now,
                            category=EventCategory.financial,
                            severity=EventSeverity.notable,
                            title=f"{symbol} volume spike: {vol_ratio:.1f}x average ({volume:,})",
                            body=f"Price: ${price:.2f} ({change_pct:+.1f}%)",
                            metadata={
                                "symbol": symbol,
                                "vol_ratio": round(vol_ratio, 1),
                                "volume": volume,
                                "change_pct": round(change_pct, 2),
                                "scan_type": "volume_spike",
                            },
                        )
                    )

        except Exception as e:
            logger.warning("MarketMovers collection failed: %s", e)

        return events

    async def _fetch_quotes(self) -> List[Dict[str, Any]]:
        """Fetch quotes via Yahoo Finance v8 chart API (public, no auth)."""
        quotes = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for symbol in self._symbols:
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
                    resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code != 200:
                        continue
                    result = resp.json().get("chart", {}).get("result", [])
                    if not result:
                        continue
                    meta = result[0].get("meta", {})
                    prev = meta.get("chartPreviousClose", 0)
                    price = meta.get("regularMarketPrice", 0)
                    volume = meta.get("regularMarketVolume", 0)
                    change_pct = ((price - prev) / prev * 100) if prev else 0
                    quotes.append(
                        {
                            "symbol": symbol,
                            "regularMarketPrice": price,
                            "regularMarketChangePercent": change_pct,
                            "regularMarketVolume": volume,
                            "averageDailyVolume3Month": meta.get(
                                "averageDailyVolume3Month", volume
                            ),
                            "regularMarketPreviousClose": prev,
                        }
                    )
                except Exception:
                    continue
        return quotes


class NewsCollector(EventCollector):
    """Collect financial news via Google News RSS (free, no API key)."""

    def __init__(self, queries: List[str] | None = None, max_articles: int = 15):
        self._queries = queries or [
            "stock market today",
            "Federal Reserve interest rates",
            "earnings surprise",
            "IPO market",
            "options unusual activity",
            "crypto funding rates",
        ]
        self._max = max_articles

    def source_id(self) -> str:
        return "market.news"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        import re
        import urllib.parse

        events: List[Event] = []
        now = datetime.now(timezone.utc)
        seen_titles: set[str] = set()

        item_pat = re.compile(r"<item>(.*?)</item>", re.DOTALL)
        title_pat = re.compile(r"<title>(.*?)</title>")
        link_pat = re.compile(r"<link>(.*?)</link>")
        pub_pat = re.compile(r"<pubDate>(.*?)</pubDate>")

        async with httpx.AsyncClient(timeout=10.0) as client:
            for query in self._queries:
                if len(events) >= self._max:
                    break
                try:
                    encoded = urllib.parse.quote(query)
                    url = f"https://news.google.com/rss/search?q={encoded}+when:1d&hl=en-US&gl=US&ceid=US:en"
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue

                    for match in item_pat.finditer(resp.text):
                        if len(events) >= self._max:
                            break
                        item = match.group(1)
                        title_m = title_pat.search(item)
                        link_m = link_pat.search(item)
                        pub_m = pub_pat.search(item)

                        title = title_m.group(1) if title_m else ""
                        if not title or title in seen_titles:
                            continue
                        seen_titles.add(title)

                        # Classify severity by keywords
                        severity = EventSeverity.info
                        lower = title.lower()
                        if any(
                            w in lower
                            for w in ["crash", "plunge", "surge", "halt", "emergency", "crisis"]
                        ):
                            severity = EventSeverity.urgent
                        elif any(
                            w in lower
                            for w in ["beat", "miss", "downgrade", "upgrade", "cut", "hike"]
                        ):
                            severity = EventSeverity.notable

                        events.append(
                            Event(
                                id=f"news-{uuid.uuid4().hex[:8]}",
                                source=self.source_id(),
                                timestamp=now,
                                category=EventCategory.news,
                                severity=severity,
                                title=title[:200],
                                body=f"Query: {query}",
                                metadata={
                                    "url": link_m.group(1) if link_m else "",
                                    "published": pub_m.group(1) if pub_m else "",
                                    "query": query,
                                    "scan_type": "news",
                                },
                            )
                        )
                except Exception as e:
                    logger.debug("News fetch failed for '%s': %s", query, e)

        return events


class VolatilityCollector(EventCollector):
    """Monitor VIX level and regime changes."""

    def __init__(self, vix_alert_level: float = 20.0, vix_crisis_level: float = 30.0):
        self._alert = vix_alert_level
        self._crisis = vix_crisis_level

    def source_id(self) -> str:
        return "market.volatility"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        events: List[Event] = []
        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=2d",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                result = resp.json().get("chart", {}).get("result", [])
                if not result:
                    return events

                meta = result[0].get("meta", {})
                level = meta.get("regularMarketPrice", 0)
                prev = meta.get("chartPreviousClose", 0)
                change = ((level - prev) / prev * 100) if prev else 0

                if level >= self._crisis:
                    severity = EventSeverity.critical
                    regime = "crisis"
                elif level >= self._alert:
                    severity = EventSeverity.urgent
                    regime = "risk-off"
                elif level >= 15:
                    severity = EventSeverity.info
                    regime = "neutral"
                else:
                    severity = EventSeverity.info
                    regime = "risk-on"

                events.append(
                    Event(
                        id=f"vol-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.financial,
                        severity=severity,
                        title=f"VIX at {level:.1f} ({change:+.1f}%) — regime: {regime}",
                        body=f"Previous close: {prev:.1f}",
                        metadata={
                            "vix": level,
                            "change_pct": round(change, 2),
                            "regime": regime,
                            "scan_type": "volatility",
                        },
                    )
                )

        except Exception as e:
            logger.warning("Volatility collection failed: %s", e)

        return events


class CryptoFundingCollector(EventCollector):
    """Monitor crypto perpetual funding rates for basis trade opportunities.

    High positive funding = shorts pay longs (basis trade: long spot, short perp).
    High negative funding = longs pay shorts (reverse basis).
    Uses Binance public API (no auth needed).
    """

    def __init__(self, funding_threshold: float = 0.01):
        self._threshold = funding_threshold  # 0.01 = 1% per 8h = ~137% APR

    def source_id(self) -> str:
        return "market.crypto_funding"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        events: List[Event] = []
        now = datetime.now(timezone.utc)

        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try Binance US first, fall back to global
                resp = None
                for base in [
                    "https://fapi.binance.us/fapi/v1/premiumIndex",
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                ]:
                    try:
                        resp = await client.get(base)
                        if resp.status_code == 200:
                            break
                    except Exception:
                        continue

                if not resp or resp.status_code != 200:
                    return events

                data = resp.json()
                for item in data:
                    symbol = item.get("symbol", "")
                    if symbol not in symbols:
                        continue

                    funding_rate = float(item.get("lastFundingRate", 0))
                    mark_price = float(item.get("markPrice", 0))

                    # Annualized funding (3 payments/day * 365)
                    apr = funding_rate * 3 * 365 * 100

                    if abs(funding_rate) >= self._threshold:
                        direction = "positive" if funding_rate > 0 else "negative"
                        severity = EventSeverity.urgent if abs(apr) > 50 else EventSeverity.notable
                        events.append(
                            Event(
                                id=f"fund-{uuid.uuid4().hex[:8]}",
                                source=self.source_id(),
                                timestamp=now,
                                category=EventCategory.financial,
                                severity=severity,
                                title=f"{symbol} funding {direction}: {funding_rate*100:.3f}% ({apr:.0f}% APR)",
                                body=f"Mark price: ${mark_price:,.2f}. "
                                f"{'Basis trade opportunity: long spot, short perp.' if funding_rate > 0 else 'Reverse basis: short spot, long perp.'}",
                                metadata={
                                    "symbol": symbol,
                                    "funding_rate": funding_rate,
                                    "apr": round(apr, 1),
                                    "mark_price": mark_price,
                                    "direction": direction,
                                    "scan_type": "funding_rate",
                                },
                            )
                        )
                    else:
                        # Still report rates for awareness
                        events.append(
                            Event(
                                id=f"fund-{uuid.uuid4().hex[:8]}",
                                source=self.source_id(),
                                timestamp=now,
                                category=EventCategory.financial,
                                severity=EventSeverity.info,
                                title=f"{symbol} funding: {funding_rate*100:.4f}% ({apr:.1f}% APR) at ${mark_price:,.0f}",
                                body="Normal range — no immediate opportunity.",
                                metadata={
                                    "symbol": symbol,
                                    "funding_rate": funding_rate,
                                    "apr": round(apr, 1),
                                    "mark_price": mark_price,
                                    "scan_type": "funding_rate",
                                },
                            )
                        )

        except Exception as e:
            logger.warning("Crypto funding collection failed: %s", e)

        return events


class SECFilingsCollector(EventCollector):
    """Monitor recent SEC filings (8-K, insider trades) for event-driven signals."""

    def source_id(self) -> str:
        return "market.sec_filings"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        events: List[Event] = []
        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # EDGAR full-text search for recent 8-K filings
                resp = await client.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q": "*",
                        "dateRange": "custom",
                        "startdt": now.strftime("%Y-%m-%d"),
                        "enddt": now.strftime("%Y-%m-%d"),
                        "forms": "8-K",
                    },
                    headers={"User-Agent": "um-agent-coder/1.0 admin@example.com"},
                )
                if resp.status_code != 200:
                    # Fallback: use EDGAR RSS
                    return await self._collect_rss(client, now)

                data = resp.json()
                for filing in data.get("hits", {}).get("hits", [])[:10]:
                    source = filing.get("_source", {})
                    company = (
                        source.get("display_names", ["Unknown"])[0]
                        if source.get("display_names")
                        else "Unknown"
                    )
                    form = source.get("form_type", "8-K")
                    filed = source.get("file_date", "")

                    events.append(
                        Event(
                            id=f"sec-{uuid.uuid4().hex[:8]}",
                            source=self.source_id(),
                            timestamp=now,
                            category=EventCategory.financial,
                            severity=EventSeverity.notable,
                            title=f"{company} filed {form}",
                            body=f"Filed: {filed}",
                            metadata={
                                "company": company,
                                "form_type": form,
                                "filed_date": filed,
                                "scan_type": "sec_filing",
                            },
                        )
                    )

        except Exception as e:
            logger.warning("SEC filings collection failed: %s", e)

        return events

    async def _collect_rss(self, client: httpx.AsyncClient, now: datetime) -> List[Event]:
        """Fallback: fetch from EDGAR RSS feed."""
        import re

        events: List[Event] = []
        try:
            resp = await client.get(
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=10&search_text=&action=getcurrent&output=atom",
                headers={"User-Agent": "um-agent-coder/1.0 admin@example.com"},
            )
            if resp.status_code != 200:
                return events

            # Parse Atom feed entries
            entry_pat = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
            title_pat = re.compile(r"<title[^>]*>(.*?)</title>")

            for match in entry_pat.finditer(resp.text):
                entry = match.group(1)
                title_m = title_pat.search(entry)
                if title_m:
                    events.append(
                        Event(
                            id=f"sec-{uuid.uuid4().hex[:8]}",
                            source=self.source_id(),
                            timestamp=now,
                            category=EventCategory.financial,
                            severity=EventSeverity.info,
                            title=title_m.group(1)[:200],
                            body="",
                            metadata={"scan_type": "sec_rss"},
                        )
                    )
        except Exception as e:
            logger.debug("SEC RSS fallback failed: %s", e)

        return events
