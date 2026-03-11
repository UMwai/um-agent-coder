# Eval Context: um-hf Volatility Regime Classifier

## Actual API Signatures & Models (from um_ai-hedge-fund source code)

### MarketSnapshot (command/autopilot/market_monitor.py)

```python
@dataclass
class Quote:
    symbol: str
    price: float
    change_pct: float
    volume: int = 0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    source: str = ""

@dataclass
class OptionsSnapshot:
    symbol: str
    atm_iv: float = 0.0
    put_call_ratio: float = 0.0
    total_volume: int = 0
    total_oi: int = 0
    skew: float = 0.0                    # 25-delta put IV - 25-delta call IV

@dataclass
class VolatilitySnapshot:
    vix: float = 0.0
    vix_change: float = 0.0
    vix_term_structure: str = "flat"     # contango / backwardation / flat
    put_call_ratio: float = 0.0
    iv_rank_spy: float = 0.0            # 0-100 percentile
    options: dict[str, OptionsSnapshot] = field(default_factory=dict)

@dataclass
class MarketSnapshot:
    timestamp: datetime
    market_state: str = "unknown"        # premarket / open / postmarket / closed
    quotes: dict[str, Quote] = field(default_factory=dict)
    volatility: VolatilitySnapshot = field(default_factory=VolatilitySnapshot)
    news: list[NewsItem] = field(default_factory=list)
    macro: dict[str, Any] = field(default_factory=dict)
    regime: str = "unknown"
    errors: list[str] = field(default_factory=list)

    @property
    def spy_change(self) -> float: ...
    @property
    def is_risk_off(self) -> bool:
        return self.volatility.vix > 25 or self.regime == "risk-off"
    @property
    def is_market_open(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...
```

### MarketMonitor (command/autopilot/market_monitor.py)

```python
class MarketMonitor:
    def __init__(self, watchlist=None, include_crypto=True, include_options=True, options_symbols=None, timeout=15.0): ...

    async def snapshot(self) -> MarketSnapshot:
        """Collect a full market snapshot."""
        snap = MarketSnapshot(timestamp=datetime.now(timezone.utc))
        # ... fetch data ...
        # Post-processing (runs after all quotes are in)
        await asyncio.gather(
            self._extract_volatility(snap),
            self._fetch_news(snap),
            self._detect_regime(snap),
            return_exceptions=True,
        )
        return snap

    async def _detect_regime(self, snap: MarketSnapshot) -> None:
        """Current implementation (TO BE REPLACED):"""
        vix = snap.volatility.vix
        spy = snap.quotes.get("SPY")
        spy_change = spy.change_pct if spy else 0
        pcr = snap.volatility.put_call_ratio
        if vix > 35 or spy_change < -3:
            snap.regime = "crisis"
        elif vix > 25 or spy_change < -1.5 or pcr > 1.3:
            snap.regime = "risk-off"
        elif vix < 15 and spy_change > 0.5:
            snap.regime = "risk-on"
        else:
            snap.regime = "neutral"
```

### Signal & Direction (command/autopilot/signal_collector.py)

```python
class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"

@dataclass
class Signal:
    symbol: str
    direction: Direction
    confidence: float                    # 0.0 to 1.0
    source: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    setup_type: str = ""
    timeframe: str = "swing"
    factors: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.direction != Direction.HOLD and self.confidence >= 0.5
```

### DecisionEngine (command/autopilot/decision_engine.py)

```python
class Action(str, Enum):
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE = "CLOSE"
    HEDGE = "HEDGE"
    REDUCE = "REDUCE"
    HOLD = "HOLD"
    SKIP = "SKIP"

@dataclass
class Decision:
    symbol: str
    action: Action
    confidence: float
    size_pct: float                       # % of allocated capital
    signals_used: int
    reasoning: str
    target_strategy: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    urgency: str = "normal"               # immediate / normal / low
    timestamp: datetime = ...

@dataclass
class DecisionConfig:
    min_confidence: float = 0.55
    min_signals: int = 1
    max_position_pct: float = 0.10
    base_position_pct: float = 0.03
    kelly_fraction: float = 0.25
    max_open_decisions: int = 10
    regime_scale: dict[str, float] = field(default_factory=lambda: {
        "risk-on": 1.2,
        "neutral": 1.0,
        "risk-off": 0.5,
        "crisis": 0.1,
    })

class DecisionEngine:
    def __init__(self, config: DecisionConfig | None = None): ...
    def decide(self, signals: list[Signal], market: MarketSnapshot) -> list[Decision]:
        """Uses market.regime to get regime_scale from config.regime_scale dict."""
        regime_scale = self.config.regime_scale.get(market.regime, 1.0)
        # ... evaluates signals, sizes positions ...
```

### DeltaNeutralStrategy (strategies/equity_options/strategies/delta_neutral.py)

```python
@dataclass
class StrategyConfig:
    iv_hv_zscore_threshold: float = 2.0
    min_iv: float = 0.10
    max_iv: float = 1.00
    max_position_pct: float = 0.05
    max_total_exposure_pct: float = 0.30
    max_positions: int = 10
    delta_rehedge_threshold: float = 0.10
    max_portfolio_delta: float = 0.05
    max_portfolio_gamma: float = 0.10
    max_portfolio_vega: float = 0.20
    min_dte: int = 7
    max_dte: int = 60
    close_before_dte: int = 3
    take_profit_pct: float = 0.50
    stop_loss_pct: float = 2.00
    hv_lookback_days: int = 20

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class DeltaNeutralStrategy:
    def __init__(self, config: StrategyConfig, portfolio_value: float, risk_free_rate: float = 0.05): ...
    def generate_signal(self, underlying, option_symbol, option_type, strike, expiry, iv, spot_price) -> Optional[Signal]: ...
    def calculate_position_size(self, signal, option_price, spot_price) -> int: ...
    def calculate_portfolio_greeks(self) -> Dict[str, float]: ...
    def check_rehedge_needed(self) -> List[Tuple[str, float]]: ...
    def check_exit_conditions(self) -> List[Tuple[str, str]]: ...
```

### UnifiedRiskEngine (portfolio/risk/unified_engine.py)

```python
class RiskLevel(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"

@dataclass
class RiskConfig:
    var_confidence: float = 0.95
    var_horizon_days: int = 1
    historical_lookback: int = 252
    max_portfolio_var_pct: float = 0.05
    max_portfolio_drawdown_pct: float = 0.15
    max_strategy_drawdown_pct: float = 0.20
    max_daily_loss_pct: float = 0.03
    warning_drawdown_pct: float = 0.08
    halt_drawdown_pct: float = 0.12
    max_strategy_correlation: float = 0.70
    correlation_warning: float = 0.50

@dataclass
class RiskReport:
    timestamp: datetime
    portfolio_value: float
    portfolio_var_1d: float
    portfolio_cvar_1d: float
    portfolio_volatility: float
    current_drawdown: float
    max_drawdown: float
    days_in_drawdown: int
    daily_pnl: float
    mtd_pnl: float
    ytd_pnl: float
    strategy_risks: Dict[str, StrategyRisk]
    correlation_matrix: Dict[str, Dict[str, float]]
    overall_risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]
    circuit_breaker_triggered: bool
```

## Codebase Patterns

### Import Style
```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
```

### Logging Pattern
```python
log = logging.getLogger(__name__)    # in command/autopilot/ modules
logger = logging.getLogger(__name__) # in strategies/ modules
```

### Async Pattern
MarketMonitor methods are `async def`. Strategy methods are synchronous.
The classifier should be synchronous (CPU-bound math on in-memory data).
Integration in MarketMonitor._detect_regime remains `async def` but calls sync classifier.

### Regime String Values
Current regime values used in codebase: "risk-on", "neutral", "risk-off", "crisis", "unknown".
DecisionConfig.regime_scale keys MUST match regime string values.
New regime values should be: "low_vol", "normal", "elevated", "high_vol", "crisis".

## CRITICAL CHECKS:
1. `MarketSnapshot.regime` must remain a plain string for backward compatibility
2. `DecisionConfig.regime_scale` keys must match the new regime string values exactly
3. `VolRegimeClassifier` must work from `MarketSnapshot` data only — no external API calls
4. All dataclasses must use `from __future__ import annotations` and `field(default_factory=...)` for mutable defaults
5. Test file must use `pytest` with plain functions (not unittest classes)
6. Regime transition requires 2+ consecutive snapshots (not single-snapshot spikes)
7. `RegimeParameters` must include overrides for DecisionConfig, StrategyConfig, and RiskConfig
8. Module imports must use `from command.autopilot.market_monitor import MarketSnapshot, VolatilitySnapshot, OptionsSnapshot`
