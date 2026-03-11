# Task: Volatility Regime Classifier for um-hf

## Prompt

Build a **Volatility Regime Classifier** for the um-hf (AI Hedge Fund) platform. This module classifies the current market volatility environment into regimes that drive position sizing, strategy selection, and risk limits across the platform.

The current `_detect_regime()` in `market_monitor.py` uses simplistic VIX/SPY thresholds (4 regimes). Replace it with a proper multi-signal regime classifier that integrates into both the Autopilot decision pipeline and the Delta-Neutral options strategy.

### Core Capabilities:

1. **Multi-Signal Regime Detection** — Classify into 5 regimes (low_vol, normal, elevated, high_vol, crisis) using:
   - VIX level and rate of change (VIX velocity)
   - VIX term structure (contango/backwardation slope, not just > or < threshold)
   - IV rank for SPY (percentile over trailing 252 days)
   - Put/call ratio (from OptionsSnapshot)
   - SPY realized volatility vs implied (vol risk premium)
   - Credit spread widths (HYG-IEF spread as proxy)

2. **Regime Transition Detection** — Track regime history and detect transitions:
   - Maintain a rolling window of regime classifications (last 20 snapshots)
   - Detect regime shifts (e.g., normal → elevated) vs noise (single-snapshot spikes)
   - Require 2+ consecutive snapshots in new regime before confirming transition
   - Emit regime transition events with metadata (from_regime, to_regime, confidence, trigger_signals)

3. **Strategy Parameter Adjustment** — Output regime-specific parameter overrides:
   - For DecisionEngine: `regime_scale` (position sizing multiplier), `min_confidence` override, `max_open_decisions` override
   - For DeltaNeutralStrategy: `iv_hv_zscore_threshold` override, `max_position_pct` override, `max_total_exposure_pct` override
   - For UnifiedRiskEngine: `max_daily_loss_pct` override, `warning_drawdown_pct` override
   - Return as a `RegimeParameters` dataclass that consumers can apply

4. **Historical Regime Analysis** — Provide lookback analytics:
   - `regime_distribution(lookback_days)` — percentage of time in each regime
   - `avg_regime_duration()` — average duration of each regime in snapshots
   - `current_regime_age()` — how many snapshots the current regime has persisted
   - `transition_matrix()` — probability matrix of regime-to-regime transitions

5. **Integration with MarketMonitor** — Replace `_detect_regime()`:
   - `VolRegimeClassifier` takes a `MarketSnapshot` and returns `RegimeResult`
   - `RegimeResult` includes: regime, confidence, sub_scores (per signal), parameters, transition info
   - MarketMonitor.snapshot() calls classifier and sets `snap.regime` + attaches full `RegimeResult`
   - Backward compatible: `snap.regime` remains a string ("low_vol", "normal", etc.)

### Technical Requirements:
- Pure Python with numpy/scipy (already in requirements.txt)
- Async-compatible but the classifier itself is synchronous (CPU-bound math)
- All dataclasses, same patterns as existing codebase (from __future__ import annotations, dataclass, field)
- No external API calls — works entirely from the MarketSnapshot data already collected
- Store regime history in-memory (no database dependency)
- Full implementations only — no TODOs, no stubs, no placeholders

### What to Deliver:
- `command/autopilot/vol_regime.py` — Main module with VolRegimeClassifier, RegimeResult, RegimeParameters, all dataclasses
- `command/autopilot/test_vol_regime.py` — Tests covering all 5 regimes, transitions, parameter generation, and edge cases
- Updated `command/autopilot/market_monitor.py` — Show the integration changes (updated `_detect_regime` using the classifier)
- Updated `command/autopilot/decision_engine.py` — Show how DecisionEngine consumes RegimeParameters
