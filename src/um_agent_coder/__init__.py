"""
um-agent-coder: Multi-model AI coding agent with parallel execution and orchestration.

This package provides:
- Multi-model orchestration (Gemini, Codex, Claude)
- Parallel execution with dependency tracking
- Task decomposition with model routing
- Subagent spawning for isolated execution
- Checkpointing for pause/resume capabilities
- Data fetchers for SEC, Yahoo Finance, ClinicalTrials, News

Quick Start:
    from um_agent_coder import (
        MultiModelOrchestrator,
        ParallelExecutor,
        TaskDecomposer,
        MCPLocalLLM
    )

    # Create models (uses local MCP tools - no API keys needed)
    gemini = MCPLocalLLM(backend="gemini")
    codex = MCPLocalLLM(backend="codex")
    claude = MCPLocalLLM(backend="claude")

    # Create orchestrator
    orchestrator = MultiModelOrchestrator(
        gemini=gemini, codex=codex, claude=claude
    )

    # Run a complex task
    result = orchestrator.run("your complex task here")
"""

__version__ = "0.2.0"
__author__ = "UMwai"

# Core orchestration
from um_agent_coder.orchestrator import (
    MultiModelOrchestrator,
    TaskPipeline,
    PipelineStep,
    TaskDecomposer,
    SubTask,
    DecomposedTask,
    ModelRole,
    ParallelExecutor,
    ExecutionMode,
    ExecutionGraph,
    ClaudeCodeSubagentSpawner,
    SubagentType,
    SubagentConfig,
    DataFetcherRegistry,
    SECEdgarFetcher,
    YahooFinanceFetcher,
    ClinicalTrialsFetcher,
    NewsFetcher,
    FetchResult,
)

# LLM providers
from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM, MCPOrchestrator

# Agent
from um_agent_coder.agent.enhanced_agent import EnhancedAgent

# Persistence
from um_agent_coder.persistence import TaskCheckpointer, TaskState, TaskStatus

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Orchestration
    "MultiModelOrchestrator",
    "TaskPipeline",
    "PipelineStep",
    # Task decomposition
    "TaskDecomposer",
    "SubTask",
    "DecomposedTask",
    "ModelRole",
    # Parallel execution
    "ParallelExecutor",
    "ExecutionMode",
    "ExecutionGraph",
    # Subagent spawning
    "ClaudeCodeSubagentSpawner",
    "SubagentType",
    "SubagentConfig",
    # Data fetchers
    "DataFetcherRegistry",
    "SECEdgarFetcher",
    "YahooFinanceFetcher",
    "ClinicalTrialsFetcher",
    "NewsFetcher",
    "FetchResult",
    # LLM providers
    "MCPLocalLLM",
    "MCPOrchestrator",
    # Agent
    "EnhancedAgent",
    # Persistence
    "TaskCheckpointer",
    "TaskState",
    "TaskStatus",
]
