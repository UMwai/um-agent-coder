# um-agent-coder Architecture

## System Overview

um-agent-coder is a multi-model AI coding agent framework designed for orchestrating complex, long-running development tasks across multiple LLM providers.

```
                                    +------------------+
                                    |    User/CLI      |
                                    +--------+---------+
                                             |
                                             v
+------------------------------------------------------------------------------------+
|                              ORCHESTRATION LAYER                                    |
|  +----------------+  +-------------------+  +------------------+  +--------------+ |
|  |     CLI        |  |  MultiModel       |  |    Task          |  |  Parallel    | |
|  |   Interface    |->|  Orchestrator     |->|  Decomposer      |->|  Executor    | |
|  +----------------+  +-------------------+  +------------------+  +--------------+ |
+------------------------------------------------------------------------------------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
              v                              v                              v
+-------------------+          +-------------------+          +-------------------+
|    GEMINI LLM     |          |    CODEX LLM      |          |    CLAUDE LLM     |
|  (Research/1M ctx)|          |  (Implementation) |          |  (Synthesis)      |
+-------------------+          +-------------------+          +-------------------+
              |                              |                              |
              +------------------------------+------------------------------+
                                             |
                                             v
+------------------------------------------------------------------------------------+
|                              EXECUTION LAYER                                        |
|  +----------------+  +-------------------+  +------------------+  +--------------+ |
|  |  Subagent      |  |  Checkpointer     |  |    Data          |  |   Tools      | |
|  |  Spawner       |  |  (Persistence)    |  |    Fetchers      |  |   (MCP)      | |
|  +----------------+  +-------------------+  +------------------+  +--------------+ |
+------------------------------------------------------------------------------------+
```

---

## Core Components

### 1. CLI Interface (`cli.py`)

**Responsibility**: Entry point for user interaction

```
um-agent "task description"
    |
    +-> Parse arguments
    +-> Load configuration
    +-> Initialize orchestrator
    +-> Execute task
    +-> Display results
```

**Key Features**:
- `--orchestrate`: Enable multi-model routing
- `--parallel`: Enable concurrent execution
- `--exec-mode`: Choose execution strategy (sequential/threads/async/subagent)
- `--human-approval`: Require approval at checkpoints

### 2. Multi-Model Orchestrator (`orchestrator/multi_model.py`)

**Responsibility**: Route tasks to appropriate models based on task type

```python
class MultiModelOrchestrator:
    """
    Routes tasks to specialized models:
    - Gemini: Research, large context analysis (1M tokens)
    - Codex: Code generation, implementation
    - Claude: Synthesis, judgment, complex reasoning
    """

    def run(self, task: str) -> Dict[str, Any]:
        # 1. Analyze task requirements
        # 2. Decompose into subtasks
        # 3. Assign models to subtasks
        # 4. Execute with dependency tracking
        # 5. Aggregate results
```

**Model Selection Logic**:

| Task Type | Primary Model | Fallback |
|-----------|--------------|----------|
| Research/Exploration | Gemini (1M ctx) | Claude |
| Code Generation | Codex (GPT-5.2) | Claude |
| Code Review | Claude Opus 4.5 | Gemini |
| Documentation | Codex | Claude |
| Complex Reasoning | Claude Opus 4.5 | Gemini |

### 3. Task Decomposer (`orchestrator/task_decomposer.py`)

**Responsibility**: Break complex tasks into executable subtasks

```
Complex Task
    |
    v
+-------------------+
| Task Decomposer   |
|                   |
| 1. Analyze scope  |
| 2. Identify deps  |
| 3. Assign models  |
| 4. Create DAG     |
+-------------------+
    |
    v
+-------------------+-------------------+-------------------+
| Subtask 1         | Subtask 2         | Subtask 3         |
| (Gemini-Research) | (Codex-Implement) | (Claude-Review)   |
| depends: none     | depends: 1        | depends: 1,2      |
+-------------------+-------------------+-------------------+
```

**Decomposition Output**:
```python
@dataclass
class DecomposedTask:
    subtasks: List[Subtask]
    execution_groups: List[List[str]]  # Parallel execution groups
    dependencies: Dict[str, List[str]]
    model_assignments: Dict[str, str]
```

### 4. Parallel Executor (`orchestrator/parallel_executor.py`)

**Responsibility**: Execute subtasks with dependency-aware parallelization

**Execution Modes**:

| Mode | Implementation | Use Case |
|------|----------------|----------|
| SEQUENTIAL | Single-threaded | Simple tasks, debugging |
| THREADS | ThreadPoolExecutor | CPU-bound parallelism |
| ASYNC | asyncio | I/O-bound operations |
| SUBAGENT_SPAWN | subprocess | True isolation, long-running |

```python
class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    THREADS = "threads"
    ASYNC = "async"
    SUBAGENT_SPAWN = "subagent"
```

**Execution Flow**:
```
Execution Groups (topologically sorted)
    |
    v
Group 1: [task_a, task_b] --> Execute in parallel
    |
    v (wait for completion)
    |
Group 2: [task_c]         --> Execute
    |
    v (wait for completion)
    |
Group 3: [task_d, task_e] --> Execute in parallel
```

### 5. Claude Subagent Spawner (`orchestrator/claude_subagent.py`)

**Responsibility**: Spawn isolated Claude Code processes

```
Main Process
    |
    +-> Spawn subprocess: claude --print --output-format stream-json
    +-> Pipe prompt via stdin
    +-> Stream output from stdout
    +-> Capture result
    +-> Terminate process
```

**Isolation Benefits**:
- Separate memory space
- Independent context windows
- True parallelism (no GIL)
- Crash isolation

### 6. LLM Providers (`llm/providers/`)

**Responsibility**: Unified interface for multiple LLM backends

```python
class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        pass
```

**Provider Implementations**:

| Provider | Class | Authentication |
|----------|-------|----------------|
| OpenAI | `OpenAILLM` | API Key |
| Anthropic | `AnthropicLLM` | API Key |
| Google | `GoogleLLM` | API Key |
| Claude CLI | `ClaudeCLIProvider` | OAuth |
| Google ADC | `GoogleADCProvider` | gcloud auth |
| MCP Local | `MCPLocalLLM` | None (local tools) |

### 7. MCP Local Provider (`llm/providers/mcp_local.py`)

**Responsibility**: Invoke local MCP tools without API keys

```python
class MCPLocalLLM(LLM):
    """
    Uses local MCP servers for Gemini, Codex, Claude
    No API keys required - uses CLI OAuth
    """

    backends = {
        "gemini": "mcp__gemini-cli__ask-gemini",
        "codex": "mcp__codex__codex",
        "claude": "claude_cli_subprocess"
    }
```

### 8. Checkpointer (`persistence/checkpointer.py`)

**Responsibility**: Durable task state for pause/resume

```
Task Execution
    |
    +-> Before step: Save state
    |
    +-> Execute step
    |
    +-> After step: Update checkpoint
    |
    +-> On failure: State preserved for resume
```

**Checkpoint Schema**:
```python
@dataclass
class TaskCheckpoint:
    task_id: str
    status: TaskStatus
    current_step: int
    completed_subtasks: List[str]
    intermediate_results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### 9. Data Fetchers (`orchestrator/data_fetchers.py`)

**Responsibility**: External data integration

| Fetcher | Source | Use Case |
|---------|--------|----------|
| SECEdgarFetcher | SEC EDGAR | Company filings (10-K, 10-Q) |
| YahooFinanceFetcher | Yahoo Finance | Stock data, fundamentals |
| ClinicalTrialsFetcher | ClinicalTrials.gov | Trial data, pipeline analysis |
| NewsFetcher | News APIs | Market news, sentiment |

---

## Data Flow

### Task Execution Flow

```
1. User Input
   um-agent --orchestrate "Build authentication system"
                |
                v
2. Task Analysis
   [CLI] -> [MultiModelOrchestrator.analyze()]
                |
                v
3. Task Decomposition
   [TaskDecomposer.decompose()]
   Output: 5 subtasks with dependencies
                |
                v
4. Execution Planning
   [ParallelExecutor.plan()]
   Output: 3 execution groups
                |
                v
5. Parallel Execution
   Group 1: [Research auth patterns] -> Gemini
   Group 2: [Design schema, Plan implementation] -> Codex (parallel)
   Group 3: [Implement, Review] -> Codex + Claude
                |
                v
6. Result Aggregation
   [MultiModelOrchestrator.aggregate()]
                |
                v
7. Output
   Final result with all artifacts
```

### Checkpoint Flow

```
Execution Start
    |
    v
+---------------+
| Load existing |
| checkpoint?   |
+-------+-------+
    |       |
   Yes      No
    |       |
    v       v
Resume   Start Fresh
    |       |
    +---+---+
        |
        v
    Execute Step
        |
        v
    Save Checkpoint
        |
        v
    Next Step?
    /        \
   Yes        No
    |          |
    v          v
  Loop      Complete
```

---

## Configuration Architecture

### Configuration Hierarchy

```
1. Defaults (code)
   |
   v
2. config/config.yaml
   |
   v
3. Environment variables
   |
   v
4. CLI arguments
```

### Config Structure

```yaml
# config/config.yaml
llm:
  provider: mcp_local  # or openai, anthropic, google
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-5.2
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-opus-4.5
  google:
    api_key: ${GOOGLE_API_KEY}
    model: gemini-3-pro

orchestrator:
  execution_mode: threads
  max_workers: 4
  checkpoint_dir: .task_checkpoints
  timeout_seconds: 3600

multi_agent_router:
  research_model: gemini
  code_model: codex
  review_model: claude
```

---

## Security Architecture

### Process Isolation

```
+------------------+     +------------------+     +------------------+
| Main Process     |     | Subagent 1       |     | Subagent 2       |
| (Orchestrator)   |     | (sandboxed)      |     | (sandboxed)      |
|                  |     |                  |     |                  |
| - Config         |     | - Limited access |     | - Limited access |
| - Coordination   |     | - Own memory     |     | - Own memory     |
| - Results        |     | - Timeouts       |     | - Timeouts       |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                            File System
                         (shared context)
```

### Security Controls

| Control | Implementation |
|---------|----------------|
| API Key Protection | Environment variables only |
| Process Isolation | subprocess with limits |
| Timeout Enforcement | Per-task timeouts |
| Resource Limits | Memory/CPU caps |
| Output Sanitization | No secrets in logs |

---

## Deployment Architecture

### Local Development

```
+-------------------+
|    Developer      |
|    Machine        |
|                   |
| +---------------+ |
| | um-agent-coder| |
| |               | |
| | - Python venv | |
| | - CLI tools   | |
| | - MCP servers | |
| +---------------+ |
+-------------------+
```

### Production (Future)

```
+-------------------+     +-------------------+
|   Load Balancer   |     |   Metrics Server  |
+--------+----------+     +---------+---------+
         |                          |
         v                          v
+-------------------+     +-------------------+
|  Agent Worker 1   |     |  Agent Worker 2   |
|  (Container)      |     |  (Container)      |
+--------+----------+     +---------+---------+
         |                          |
         +------------+-------------+
                      |
                      v
         +-------------------+
         |   Shared Storage  |
         |  (Checkpoints)    |
         +-------------------+
```

---

## Extension Points

### Adding New LLM Provider

```python
# llm/providers/new_provider.py
from um_agent_coder.llm.base import LLM

class NewProviderLLM(LLM):
    def __init__(self, api_key: str, model: str):
        self.client = NewProviderClient(api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        return self.client.complete(prompt, model=self.model)

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        for chunk in self.client.stream(prompt, model=self.model):
            yield chunk
```

### Adding New Data Fetcher

```python
# orchestrator/data_fetchers.py
class CustomDataFetcher:
    def __init__(self, config: Dict):
        self.config = config

    def fetch(self, query: str) -> Dict[str, Any]:
        # Implement data fetching logic
        pass
```

### Adding New Execution Mode

```python
# orchestrator/parallel_executor.py
class CustomExecutor:
    def execute(self, tasks: List[Subtask]) -> List[Result]:
        # Implement custom execution strategy
        pass
```

---

## Performance Considerations

### Bottlenecks and Solutions

| Bottleneck | Solution |
|------------|----------|
| LLM API latency | Parallel execution, streaming |
| Context size limits | Chunking, summarization |
| Memory usage | Subagent spawning |
| Disk I/O | Async writes, batching |

### Optimization Strategies

1. **Connection Pooling**: Reuse HTTP connections
2. **Response Caching**: Cache identical prompts
3. **Lazy Loading**: Load providers on demand
4. **Streaming**: Process output as it arrives

---

*Last Updated: December 2024*
