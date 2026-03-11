# Hedge Fund Continuous Build Roadmap

Target repo: `~/um_ai-hedge-fund` (GitHub: `UMwai/um_ai-hedge-fund`)

## Tasks

- [ ] **hf-001**: Audit equity_options strategy code — verify delta_neutral.py, iv_scanner.py, vol_surface.py are complete and testable. Add missing unit tests. Run from ~/um_ai-hedge-fund.
  - timeout: 30min
  - depends: none
  - success: All equity_options tests pass, coverage > 60%
  - cli: codex

- [ ] **hf-002**: Wire backtesting engine — connect strategies/equity_options/backtesting/engine.py to real historical data. Implement a 1-year SPY backtest that produces Sharpe, Sortino, max drawdown metrics.
  - timeout: 45min
  - depends: hf-001
  - success: `python -m pytest strategies/equity_options/backtesting/ -v` passes
  - cli: codex

- [ ] **hf-003**: Implement portfolio risk engine circuit breakers — wire portfolio/risk/unified_engine.py with VaR, CVaR, drawdown limits per specs/ARCHITECTURE.md. Add hard limits: 25% max drawdown, 3% daily VaR, 8% monthly loss.
  - timeout: 45min
  - depends: none
  - success: Risk engine unit tests pass with boundary conditions
  - cli: codex

- [ ] **hf-004**: Build command center orchestrator — make command/orchestrator/engine.py functional. Wire to strategy adapters (NativeAdapter, HttpAdapter, CliAdapter). Test with equity_options strategy.
  - timeout: 45min
  - depends: hf-001
  - success: `python -m command.cli status` returns strategy health
  - cli: codex

- [ ] **hf-005**: Implement capital allocation service — portfolio/allocation/capital_service.py should implement Kelly criterion position sizing, pulled from ev_trading_platform if available.
  - timeout: 30min
  - depends: hf-003
  - success: Allocation service computes positions for a 2-strategy portfolio
  - cli: codex

- [ ] **hf-006**: Build crypto-arb funding rate harvester — implement strategies/crypto-arb/ with Binance/Bybit funding rate collection and cash-and-carry basis trade logic.
  - timeout: 60min
  - depends: hf-003
  - success: Funding rate data collection works, backtest shows positive carry
  - cli: codex
  - ralph: true
  - max_iterations: 20
  - completion_promise: CRYPTO_ARB_COMPLETE

- [ ] **hf-007**: Integrate adjacent repo strategies — pull proven ML strategies from um-trading-assistance and wire them as remote adapters in the command center registry.
  - timeout: 45min
  - depends: hf-004
  - success: At least 2 strategies from adjacent repos registered and queryable
  - cli: codex

- [ ] **hf-008**: Build performance attribution system — portfolio/attribution/performance.py should compute alpha decomposition, factor exposures, and strategy-level P&L.
  - timeout: 30min
  - depends: hf-002, hf-005
  - success: Performance report generates for backtest results
  - cli: codex

- [ ] **hf-009**: End-to-end integration test — run full pipeline: signal collection → strategy execution → risk check → rebalance → performance report. Paper trade mode.
  - timeout: 60min
  - depends: hf-004, hf-005, hf-003
  - success: Full loop completes without errors, produces P&L report
  - cli: codex
  - ralph: true
  - max_iterations: 30
  - completion_promise: E2E_COMPLETE

- [ ] **hf-010**: CI/CD pipeline — GitHub Actions workflow for um_ai-hedge-fund: lint, test, type-check, coverage report on every PR.
  - timeout: 30min
  - depends: hf-001
  - success: .github/workflows/ci.yml exists and passes
  - cli: codex
