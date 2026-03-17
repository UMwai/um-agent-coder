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
CREDIT_WATCHLIST = [
    # BDCs (private credit public window)
    "ARCC",   # Ares Capital — largest BDC
    "BXSL",   # Blackstone Secured Lending
    "OBDC",   # Blue Owl Capital
    "FSK",    # FS KKR Capital
    "MAIN",   # Main Street Capital
    "HTGC",   # Hercules Capital (tech lending)
    # CLO equity (first to break)
    "ECC",    # Eagle Point Credit
    "OXLC",   # Oxford Lane Capital
    # Leveraged loans
    "BKLN",   # Invesco Senior Loan ETF
    "SRLN",   # SPDR Blackstone Senior Loan ETF
    # High yield
    "HYG",    # iShares HY Corporate Bond
    "JNK",    # SPDR HY Bond
    # Alt managers (contagion transmitters)
    "APO",    # Apollo
    "BX",     # Blackstone
    "ARES",   # Ares Management
    "OWL",    # Blue Owl Capital (manager)
]
CRYPTO_WATCHLIST = ["bitcoin", "ethereum", "solana"]
VOL_SYMBOLS = ["^VIX"]

# Credit stress thresholds
CREDIT_STRESS_THRESHOLDS = {
    # BDC discount to book: >10% = warning, >20% = danger
    "bdc_pb_warning": 0.90,    # P/B below 0.90
    "bdc_pb_danger": 0.80,     # P/B below 0.80
    # Price drops
    "credit_move_pct": 1.5,    # Lower threshold for credit instruments
    # Yield spikes (signals distress)
    "yield_spike_pct": 12.0,   # BDC yield above 12% = stress
}


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
            "private credit default",
            "CLO market stress",
            "leveraged loan defaults",
            "commercial real estate debt",
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


class CreditStressCollector(EventCollector):
    """Monitor private credit stress via BDC prices, CLO equity, HY spreads.

    Tracks:
    - BDC price-to-book ratios (discount = market calling BS on private marks)
    - CLO equity fund performance (first tranche to break)
    - Leveraged loan ETF discounts to NAV
    - HY bond ETF price action
    - Alt manager stock prices (contagion transmitters)

    Thresholds:
    - BDC P/B < 0.90 → warning
    - BDC P/B < 0.80 → contagion imminent
    - CLO equity yield > 30% → tranche impairment
    - BKLN NAV discount > 2% → loan market liquidity stress
    """

    def __init__(self, symbols: List[str] | None = None):
        self._symbols = symbols or CREDIT_WATCHLIST

    def source_id(self) -> str:
        return "market.credit_stress"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        events: List[Event] = []
        now = datetime.now(timezone.utc)

        try:
            quotes = await self._fetch_quotes()

            bdcs = ["ARCC", "BXSL", "OBDC", "FSK", "MAIN", "HTGC"]
            clo_equity = ["ECC", "OXLC"]
            loan_etfs = ["BKLN", "SRLN"]
            hy_etfs = ["HYG", "JNK"]
            alt_mgrs = ["APO", "BX", "ARES", "OWL"]

            # Aggregate BDC stress
            bdc_moves = []
            for q in quotes:
                sym = q.get("symbol", "")
                change_pct = q.get("regularMarketChangePercent", 0)
                price = q.get("regularMarketPrice", 0)
                pb = q.get("priceToBook", 0)
                div_yield = q.get("dividendYield", 0)
                fifty_two_low = q.get("fiftyTwoWeekLow", 0)

                if sym in bdcs:
                    bdc_moves.append({
                        "symbol": sym,
                        "change_pct": change_pct,
                        "price": price,
                        "pb": pb,
                        "div_yield": div_yield,
                        "near_52w_low": (
                            price <= fifty_two_low * 1.05 if fifty_two_low > 0 else False
                        ),
                    })

                    # Individual BDC stress signals
                    if pb > 0 and pb < CREDIT_STRESS_THRESHOLDS["bdc_pb_danger"]:
                        events.append(Event(
                            id=f"cred-{uuid.uuid4().hex[:8]}",
                            source=self.source_id(),
                            timestamp=now,
                            category=EventCategory.financial,
                            severity=EventSeverity.critical,
                            title=f"DANGER: {sym} P/B at {pb:.2f} — market rejecting private marks",
                            body=f"Price ${price:.2f} ({change_pct:+.1f}%). "
                                 f"Yield {div_yield*100:.1f}%. "
                                 f"BDC trading >20% below book = contagion signal.",
                            metadata={
                                "symbol": sym,
                                "price_to_book": round(pb, 3),
                                "change_pct": round(change_pct, 2),
                                "price": price,
                                "div_yield": round(div_yield * 100, 2) if div_yield else 0,
                                "scan_type": "bdc_stress",
                                "stress_level": "danger",
                            },
                        ))
                    elif pb > 0 and pb < CREDIT_STRESS_THRESHOLDS["bdc_pb_warning"]:
                        events.append(Event(
                            id=f"cred-{uuid.uuid4().hex[:8]}",
                            source=self.source_id(),
                            timestamp=now,
                            category=EventCategory.financial,
                            severity=EventSeverity.urgent,
                            title=f"WARNING: {sym} P/B at {pb:.2f} — discount widening",
                            body=f"Price ${price:.2f} ({change_pct:+.1f}%). "
                                 f"Yield {div_yield*100:.1f}%. "
                                 f"BDC discount >10% = private credit stress.",
                            metadata={
                                "symbol": sym,
                                "price_to_book": round(pb, 3),
                                "change_pct": round(change_pct, 2),
                                "price": price,
                                "div_yield": round(div_yield * 100, 2) if div_yield else 0,
                                "scan_type": "bdc_stress",
                                "stress_level": "warning",
                            },
                        ))

                # CLO equity — the canary
                if sym in clo_equity:
                    severity = EventSeverity.critical if change_pct <= -3 else (
                        EventSeverity.urgent if change_pct <= -1.5 else EventSeverity.notable
                    )
                    events.append(Event(
                        id=f"cred-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.financial,
                        severity=severity,
                        title=f"CLO equity {sym}: ${price:.2f} ({change_pct:+.1f}%)"
                              + (" — NEAR 52W LOW" if price <= fifty_two_low * 1.05 and fifty_two_low > 0 else ""),
                        body=f"Yield {div_yield*100:.1f}%. "
                             f"CLO equity is first tranche to absorb loan defaults. "
                             f"Sustained decline = underlying loan book deteriorating.",
                        metadata={
                            "symbol": sym,
                            "change_pct": round(change_pct, 2),
                            "price": price,
                            "div_yield": round(div_yield * 100, 2) if div_yield else 0,
                            "near_52w_low": price <= fifty_two_low * 1.05 if fifty_two_low > 0 else False,
                            "scan_type": "clo_equity",
                        },
                    ))

                # Loan ETFs — liquidity proxy
                if sym in loan_etfs and abs(change_pct) >= CREDIT_STRESS_THRESHOLDS["credit_move_pct"]:
                    events.append(Event(
                        id=f"cred-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.financial,
                        severity=EventSeverity.urgent,
                        title=f"Loan market stress: {sym} {change_pct:+.1f}% to ${price:.2f}",
                        body="Leveraged loan ETF decline signals credit market liquidity withdrawal.",
                        metadata={
                            "symbol": sym,
                            "change_pct": round(change_pct, 2),
                            "price": price,
                            "scan_type": "loan_stress",
                        },
                    ))

                # HY ETFs
                if sym in hy_etfs and abs(change_pct) >= CREDIT_STRESS_THRESHOLDS["credit_move_pct"]:
                    events.append(Event(
                        id=f"cred-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.financial,
                        severity=EventSeverity.urgent,
                        title=f"HY spread widening: {sym} {change_pct:+.1f}% to ${price:.2f}",
                        body="High yield bond decline = credit spreads blowing out.",
                        metadata={
                            "symbol": sym,
                            "change_pct": round(change_pct, 2),
                            "price": price,
                            "scan_type": "hy_stress",
                        },
                    ))

                # Alt managers — contagion pathway
                if sym in alt_mgrs and abs(change_pct) >= 2.0:
                    events.append(Event(
                        id=f"cred-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.financial,
                        severity=EventSeverity.urgent,
                        title=f"Alt manager selloff: {sym} {change_pct:+.1f}% to ${price:.2f}",
                        body="Alt manager stock decline signals market concern about "
                             "private credit / PE exposure. Watch for fund redemption risk.",
                        metadata={
                            "symbol": sym,
                            "change_pct": round(change_pct, 2),
                            "price": price,
                            "scan_type": "alt_mgr_stress",
                        },
                    ))

            # Composite BDC stress score
            if bdc_moves:
                avg_change = sum(b["change_pct"] for b in bdc_moves) / len(bdc_moves)
                near_lows = sum(1 for b in bdc_moves if b["near_52w_low"])
                avg_pb = sum(b["pb"] for b in bdc_moves if b["pb"] > 0)
                pb_count = sum(1 for b in bdc_moves if b["pb"] > 0)
                avg_pb = avg_pb / pb_count if pb_count else 0

                if avg_change <= -1.0 or near_lows >= 3 or (avg_pb > 0 and avg_pb < 0.85):
                    events.append(Event(
                        id=f"cred-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.financial,
                        severity=EventSeverity.critical,
                        title=f"PRIVATE CREDIT STRESS: BDC composite "
                              f"avg {avg_change:+.1f}%, {near_lows}/{len(bdc_moves)} near 52w low"
                              + (f", avg P/B {avg_pb:.2f}" if avg_pb > 0 else ""),
                        body=f"Broad BDC selloff indicates systemic private credit concern. "
                             f"Monitor for: dividend cuts, non-accrual spikes, warehouse margin calls. "
                             f"Components: {', '.join(b['symbol'] + ' ' + str(round(b['change_pct'], 1)) + '%' for b in bdc_moves)}",
                        metadata={
                            "avg_change_pct": round(avg_change, 2),
                            "near_52w_lows": near_lows,
                            "avg_price_to_book": round(avg_pb, 3) if avg_pb else None,
                            "components": {b["symbol"]: round(b["change_pct"], 2) for b in bdc_moves},
                            "scan_type": "credit_composite",
                            "stress_level": "systemic",
                        },
                    ))

        except Exception as e:
            logger.warning("Credit stress collection failed: %s", e)

        return events

    async def _fetch_quotes(self) -> List[Dict[str, Any]]:
        """Fetch quotes with fundamentals via Yahoo Finance."""
        quotes = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for symbol in self._symbols:
                try:
                    # Use v8 chart for price data
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
                    resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code != 200:
                        continue
                    result = resp.json().get("chart", {}).get("result", [])
                    if not result:
                        continue
                    meta = result[0].get("meta", {})
                    prev = meta.get("chartPreviousClose", 0)
                    price = meta.get("regularMarketPrice", 0)
                    change_pct = ((price - prev) / prev * 100) if prev else 0
                    fifty_two_low = meta.get("fiftyTwoWeekLow", 0)
                    fifty_two_high = meta.get("fiftyTwoWeekHigh", 0)

                    quote = {
                        "symbol": symbol,
                        "regularMarketPrice": price,
                        "regularMarketChangePercent": change_pct,
                        "regularMarketPreviousClose": prev,
                        "fiftyTwoWeekLow": fifty_two_low,
                        "fiftyTwoWeekHigh": fifty_two_high,
                        "priceToBook": 0,
                        "dividendYield": 0,
                    }

                    # Try to get P/B and yield from quoteSummary
                    try:
                        summary_url = (
                            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
                            f"?modules=defaultKeyStatistics,summaryDetail"
                        )
                        s_resp = await client.get(
                            summary_url, headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if s_resp.status_code == 200:
                            s_data = s_resp.json().get("quoteSummary", {}).get("result", [])
                            if s_data:
                                stats = s_data[0].get("defaultKeyStatistics", {})
                                detail = s_data[0].get("summaryDetail", {})
                                pb = stats.get("priceToBook", {})
                                if isinstance(pb, dict):
                                    pb = pb.get("raw", 0)
                                dy = detail.get("dividendYield", {})
                                if isinstance(dy, dict):
                                    dy = dy.get("raw", 0)
                                quote["priceToBook"] = pb or 0
                                quote["dividendYield"] = dy or 0
                    except Exception:
                        pass

                    quotes.append(quote)
                except Exception:
                    continue
        return quotes
