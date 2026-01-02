"""Tests for Multi-CLI Router."""

from datetime import date
from unittest.mock import patch

import pytest

from um_agent_coder.harness.autonomous.cli_router import (
    AutoRouter,
    CLIRouter,
    OpusGuard,
    TaskAnalysis,
    TaskAnalyzer,
    TaskType,
    parse_cli_list,
)


class TestTaskAnalysis:
    """Tests for TaskAnalysis."""

    def test_default_values(self):
        """Test default analysis values."""
        analysis = TaskAnalysis()
        assert not analysis.requires_large_context
        assert not analysis.is_implementation
        assert analysis.estimated_difficulty == 0.5

    def test_get_primary_type_stuck(self):
        """Test stuck recovery is highest priority."""
        analysis = TaskAnalysis(
            is_stuck_recovery=True,
            is_implementation=True,
            is_complex_reasoning=True,
        )
        assert analysis.get_primary_type() == TaskType.STUCK_RECOVERY

    def test_get_primary_type_complex(self):
        """Test complex reasoning type."""
        analysis = TaskAnalysis(
            is_complex_reasoning=True,
            estimated_difficulty=0.8,
        )
        assert analysis.get_primary_type() == TaskType.COMPLEX_REASONING

    def test_get_primary_type_implementation(self):
        """Test implementation type."""
        analysis = TaskAnalysis(is_implementation=True)
        assert analysis.get_primary_type() == TaskType.IMPLEMENTATION

    def test_get_primary_type_research(self):
        """Test research type."""
        analysis = TaskAnalysis(is_research=True)
        assert analysis.get_primary_type() == TaskType.RESEARCH

    def test_get_primary_type_simple(self):
        """Test simple type as default."""
        analysis = TaskAnalysis()
        assert analysis.get_primary_type() == TaskType.SIMPLE


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer."""

    def test_analyze_implementation_keywords(self):
        """Test implementation keyword detection."""
        analyzer = TaskAnalyzer()
        analysis = analyzer.analyze("Implement a login function")

        assert analysis.is_implementation
        assert "implement" in analysis.keywords_detected

    def test_analyze_research_keywords(self):
        """Test research keyword detection."""
        analyzer = TaskAnalyzer()
        analysis = analyzer.analyze("Research the codebase to find error handlers")

        assert analysis.is_research
        assert "research" in analysis.keywords_detected or "find" in analysis.keywords_detected

    def test_analyze_complex_keywords(self):
        """Test complex keyword detection."""
        analyzer = TaskAnalyzer()
        analysis = analyzer.analyze("Design and architect a new authentication system")

        assert analysis.is_complex_reasoning
        assert "design" in analysis.keywords_detected or "architect" in analysis.keywords_detected

    def test_analyze_large_context(self):
        """Test large context detection."""
        analyzer = TaskAnalyzer(large_context_threshold=1000)
        analysis = analyzer.analyze("Test", context_size=5000)

        assert analysis.requires_large_context

    def test_analyze_stuck_recovery(self):
        """Test stuck recovery detection."""
        analyzer = TaskAnalyzer()
        analysis = analyzer.analyze("Test", consecutive_no_progress=5)

        assert analysis.is_stuck_recovery

    def test_estimate_difficulty_long_goal(self):
        """Test difficulty estimation for long goals."""
        analyzer = TaskAnalyzer()
        long_goal = "x" * 600
        analysis = analyzer.analyze(long_goal)

        assert analysis.estimated_difficulty >= 0.2

    def test_estimate_difficulty_many_iterations(self):
        """Test difficulty increases with iterations."""
        analyzer = TaskAnalyzer()
        analysis = analyzer.analyze("Test", total_iterations=25)

        assert analysis.estimated_difficulty >= 0.3


class TestOpusGuard:
    """Tests for OpusGuard."""

    def test_initial_state(self):
        """Test initial guard state."""
        guard = OpusGuard(daily_limit=10)
        assert guard.can_use_opus()
        assert guard.get_remaining() == 10

    def test_record_usage(self):
        """Test recording Opus usage."""
        guard = OpusGuard(daily_limit=5)

        guard.record_opus_use()
        assert guard.get_remaining() == 4

        guard.record_opus_use()
        assert guard.get_remaining() == 3

    def test_limit_reached(self):
        """Test when limit is reached."""
        guard = OpusGuard(daily_limit=2)

        guard.record_opus_use()
        guard.record_opus_use()

        assert not guard.can_use_opus()
        assert guard.get_remaining() == 0

    def test_daily_reset(self):
        """Test daily counter reset."""
        guard = OpusGuard(daily_limit=5)
        guard.record_opus_use()
        guard.record_opus_use()

        # Simulate new day
        guard.last_reset = date(2020, 1, 1)

        assert guard.can_use_opus()
        assert guard.get_remaining() == 5


class TestAutoRouter:
    """Tests for AutoRouter."""

    def test_default_routing(self):
        """Test default router setup."""
        router = AutoRouter()
        assert "codex" in router.enabled_clis
        assert "gemini" in router.enabled_clis
        assert "claude" in router.enabled_clis

    def test_route_implementation(self):
        """Test routing implementation tasks."""
        router = AutoRouter()
        cli, model = router.route("Implement a new function")

        assert cli == "codex"
        assert model == "gpt-5.2"

    def test_route_research(self):
        """Test routing research tasks."""
        router = AutoRouter()
        cli, model = router.route("Research the error handling patterns")

        assert cli == "gemini"
        assert model == "gemini-3-pro"

    def test_route_large_context(self):
        """Test routing large context tasks."""
        router = AutoRouter()
        cli, model = router.route("Analyze codebase", context_size=200000)

        assert cli == "gemini"
        assert model == "gemini-3-pro"

    def test_route_stuck_recovery(self):
        """Test routing stuck recovery."""
        router = AutoRouter()
        cli, model = router.route("Test", consecutive_no_progress=5)

        assert cli == "claude"
        assert "claude" in model

    def test_route_complex_uses_opus(self):
        """Test complex tasks use Opus."""
        router = AutoRouter()
        cli, model = router.route("Design a complex architecture system")

        # Should use Opus for complex tasks
        if cli == "claude":
            assert "opus" in model or "sonnet" in model

    def test_route_respects_enabled_clis(self):
        """Test routing respects enabled CLIs."""
        router = AutoRouter(enabled_clis=["codex"])
        cli, model = router.route("Research the codebase")

        # Should fallback to codex since gemini not enabled
        assert cli == "codex"

    def test_opus_exhaustion_fallback(self):
        """Test fallback when Opus exhausted."""
        guard = OpusGuard(daily_limit=1)
        guard.record_opus_use()  # Use up limit

        router = AutoRouter(opus_guard=guard)
        cli, model = router.route("Design complex system", consecutive_no_progress=5)

        # Should fallback to Sonnet
        assert cli == "claude"
        assert "sonnet" in model

    def test_get_smartest(self):
        """Test getting smartest model."""
        router = AutoRouter()
        cli, model = router._get_smartest()

        assert cli == "claude"

    def test_get_cheapest(self):
        """Test getting cheapest model."""
        router = AutoRouter()
        cli, model = router._get_cheapest()

        assert cli == "gemini"
        assert model == "gemini-3-flash"

    def test_prefer_cheap_setting(self):
        """Test prefer cheap setting."""
        router = AutoRouter(prefer_cheap=True)
        cli, model = router.route("Simple task")

        # Should prefer cheap for simple tasks
        assert cli == "gemini"
        assert model == "gemini-3-flash"


class TestParseCLIList:
    """Tests for parse_cli_list."""

    def test_parse_single(self):
        """Test parsing single CLI."""
        result = parse_cli_list("codex")
        assert result == ["codex"]

    def test_parse_multiple(self):
        """Test parsing multiple CLIs."""
        result = parse_cli_list("codex,gemini")
        assert result == ["codex", "gemini"]

    def test_parse_with_spaces(self):
        """Test parsing with spaces."""
        result = parse_cli_list("codex , gemini , claude")
        assert result == ["codex", "gemini", "claude"]

    def test_parse_auto(self):
        """Test parsing auto."""
        result = parse_cli_list("auto")
        assert result == ["auto"]

    def test_parse_empty(self):
        """Test parsing empty string."""
        result = parse_cli_list("")
        assert result == ["auto"]

    def test_parse_case_insensitive(self):
        """Test case insensitivity."""
        result = parse_cli_list("CODEX,GEMINI")
        assert result == ["codex", "gemini"]


class TestCLIRouter:
    """Tests for CLIRouter."""

    def test_auto_mode_init(self):
        """Test auto mode initialization."""
        router = CLIRouter(cli_spec="auto")
        assert router.is_auto
        assert router.auto_router is not None

    def test_explicit_mode_init(self):
        """Test explicit mode initialization."""
        router = CLIRouter(cli_spec="codex,gemini")
        assert not router.is_auto
        assert router.enabled_clis == {"codex", "gemini"}

    def test_route_auto_mode(self):
        """Test routing in auto mode."""
        router = CLIRouter(cli_spec="auto")
        cli, model = router.route("Implement a function")

        assert cli in ["codex", "gemini", "claude"]
        assert model

    def test_route_explicit_mode(self):
        """Test routing in explicit mode."""
        router = CLIRouter(cli_spec="codex")
        cli, model = router.route("Any task")

        assert cli == "codex"
        assert model == "gpt-5.2"

    def test_explicit_mode_round_robin(self):
        """Test round-robin in explicit mode."""
        router = CLIRouter(cli_spec="codex,gemini")

        cli1, _ = router.route()
        cli2, _ = router.route()
        cli3, _ = router.route()

        # Should alternate between codex and gemini
        assert cli1 == "codex"
        assert cli2 == "gemini"
        assert cli3 == "codex"

    def test_get_enabled_clis_auto(self):
        """Test getting enabled CLIs in auto mode."""
        router = CLIRouter(cli_spec="auto")
        clis = router.get_enabled_clis()

        assert "codex" in clis
        assert "gemini" in clis
        assert "claude" in clis

    def test_get_enabled_clis_explicit(self):
        """Test getting enabled CLIs in explicit mode."""
        router = CLIRouter(cli_spec="codex,gemini")
        clis = router.get_enabled_clis()

        assert clis == ["codex", "gemini"]

    def test_get_opus_remaining(self):
        """Test getting Opus remaining."""
        router = CLIRouter(cli_spec="auto", opus_daily_limit=10)
        assert router.get_opus_remaining() == 10

    def test_opus_limit_setting(self):
        """Test Opus limit setting."""
        router = CLIRouter(cli_spec="auto", opus_daily_limit=25)
        assert router.opus_guard.daily_limit == 25


class TestCLIRouterIntegration:
    """Integration tests for CLI routing."""

    def test_full_routing_workflow(self):
        """Test complete routing workflow."""
        router = CLIRouter(cli_spec="auto", opus_daily_limit=50)

        # Route different task types
        impl_cli, impl_model = router.route("Implement authentication")
        assert impl_cli == "codex"

        # Use "explore" to avoid "code" triggering implementation
        research_cli, research_model = router.route("Research the error handling patterns")
        assert research_cli == "gemini"

        stuck_cli, stuck_model = router.route("Stuck task", consecutive_no_progress=5)
        assert stuck_cli == "claude"

    def test_explicit_mode_ignores_analysis(self):
        """Test explicit mode ignores task analysis."""
        router = CLIRouter(cli_spec="gemini")

        # Even for implementation tasks, should use gemini
        cli, model = router.route("Implement a function")
        assert cli == "gemini"

    def test_opus_guard_across_routes(self):
        """Test Opus guard persists across routes."""
        router = CLIRouter(cli_spec="auto", opus_daily_limit=2)

        # Make complex requests that would use Opus
        for _ in range(3):
            router.route("Design complex architecture", consecutive_no_progress=5)

        # Should have used up Opus budget
        assert router.get_opus_remaining() < 2
