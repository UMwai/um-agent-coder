# um-agent-coder Agent Design

## Agent Architecture Overview

um-agent-coder implements a multi-agent system where specialized AI agents collaborate to complete complex development tasks. The architecture follows a hierarchical orchestration pattern with model-specific specialization.

---

## Agent Hierarchy

```
                    +----------------------+
                    |   Head Node Agent    |
                    |   (Orchestrator)     |
                    +----------+-----------+
                               |
           +-------------------+-------------------+
           |                   |                   |
           v                   v                   v
+------------------+  +------------------+  +------------------+
|  Research Agent  |  |  Builder Agent   |  | Synthesis Agent  |
|  (Gemini)        |  |  (Codex)         |  | (Claude)         |
+------------------+  +------------------+  +------------------+
```

---

## Agent Types

### 1. Head Node Agent (Orchestrator)

**Role**: Central coordinator that analyzes tasks, delegates work, and aggregates results.

**Responsibilities**:
- Task analysis and classification
- Subtask decomposition
- Agent assignment
- Progress monitoring
- Result aggregation
- Error recovery coordination

**Implementation**:
```python
class HeadNodeAgent:
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.task_queue = []
        self.results = {}

    def orchestrate(self, task: str) -> Result:
        # 1. Analyze task complexity
        analysis = self.analyze_task(task)

        # 2. Decompose into subtasks
        subtasks = self.decompose(task, analysis)

        # 3. Assign to appropriate agents
        assignments = self.assign_agents(subtasks)

        # 4. Execute with monitoring
        results = self.execute(assignments)

        # 5. Aggregate and return
        return self.aggregate(results)
```

### 2. Research Agent (Gemini-based)

**Role**: Information gathering, large context analysis, exploration.

**Capabilities**:
- 1M token context window
- Multi-document synthesis
- Web research integration
- Pattern discovery
- Exploratory analysis

**Use Cases**:
- Codebase exploration
- Documentation analysis
- Market research
- Competitive analysis
- Large file processing

**Configuration**:
```yaml
research_agent:
  model: gemini-3-pro
  context_window: 1000000
  temperature: 0.7
  specializations:
    - exploration
    - synthesis
    - documentation_analysis
```

### 3. Builder Agent (Codex-based)

**Role**: Code generation, implementation, planning.

**Capabilities**:
- Strong code generation
- Multi-language support
- Architecture planning
- Test generation
- Refactoring

**Use Cases**:
- Feature implementation
- Bug fixes
- Code refactoring
- Test writing
- Build configuration

**Configuration**:
```yaml
builder_agent:
  model: gpt-5.2
  temperature: 0.3
  specializations:
    - code_generation
    - implementation
    - testing
    - refactoring
```

### 4. Synthesis Agent (Claude-based)

**Role**: Complex reasoning, judgment, review, and final synthesis.

**Capabilities**:
- Deep reasoning
- Code review
- Security analysis
- Quality judgment
- Final synthesis

**Use Cases**:
- Code review
- Architecture decisions
- Security assessment
- Quality validation
- Final output generation

**Configuration**:
```yaml
synthesis_agent:
  model: claude-opus-4.5
  temperature: 0.5
  specializations:
    - code_review
    - security_analysis
    - architecture_decisions
    - synthesis
```

---

## Agent Communication Protocol

### Message Format

```python
@dataclass
class AgentMessage:
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str  # Links related messages

class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    CONTEXT_UPDATE = "context_update"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    ERROR = "error"
```

### Communication Flow

```
Head Node                    Worker Agents
    |                             |
    |-- TASK_ASSIGNMENT --------->|
    |                             |
    |<-- STATUS_RESPONSE ---------|
    |                             |
    |<-- CONTEXT_UPDATE ----------|
    |                             |
    |<-- TASK_RESULT -------------|
    |                             |
```

### Context Sharing

```python
@dataclass
class SharedContext:
    project_info: Dict[str, str]
    completed_tasks: List[TaskSummary]
    intermediate_results: Dict[str, Any]
    code_artifacts: List[CodeArtifact]
    decisions_made: List[Decision]
```

---

## Agent Lifecycle

### State Machine

```
                    +--------+
                    |  IDLE  |
                    +----+---+
                         |
                    (assign task)
                         |
                         v
+--------+          +--------+          +--------+
| FAILED |<--(err)--| ACTIVE |---(ok)-->| DONE   |
+--------+          +----+---+          +--------+
     |                   |
     |              (checkpoint)
     |                   |
     |              +----v---+
     +<----(err)----| PAUSED |
                    +--------+
```

### Lifecycle Hooks

```python
class AgentLifecycle:
    def on_start(self, agent: Agent, task: Task):
        """Called when agent starts a task"""
        pass

    def on_checkpoint(self, agent: Agent, state: Dict):
        """Called at checkpoint intervals"""
        pass

    def on_complete(self, agent: Agent, result: Result):
        """Called on successful completion"""
        pass

    def on_error(self, agent: Agent, error: Exception):
        """Called on error"""
        pass

    def on_terminate(self, agent: Agent):
        """Called on agent termination"""
        pass
```

---

## Task Assignment Algorithm

### Model Selection Matrix

| Task Characteristic | Primary Agent | Secondary Agent |
|--------------------|---------------|-----------------|
| Large context (>100k tokens) | Gemini | Claude |
| Code generation | Codex | Claude |
| Code review | Claude | Gemini |
| Research/exploration | Gemini | Claude |
| Complex reasoning | Claude | Gemini |
| Fast iteration | Codex | Gemini |
| Documentation | Codex | Gemini |

### Assignment Algorithm

```python
def assign_agent(subtask: Subtask) -> str:
    # 1. Check explicit preference
    if subtask.preferred_agent:
        return subtask.preferred_agent

    # 2. Analyze task characteristics
    features = analyze_task_features(subtask)

    # 3. Score each agent
    scores = {}
    for agent_type, agent in agents.items():
        scores[agent_type] = compute_score(features, agent.capabilities)

    # 4. Select best available agent
    sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for agent_type, score in sorted_agents:
        if is_available(agent_type):
            return agent_type

    # 5. Queue for next available
    return queue_for_best_match(subtask, sorted_agents)
```

### Feature Analysis

```python
def analyze_task_features(subtask: Subtask) -> TaskFeatures:
    return TaskFeatures(
        estimated_tokens=estimate_tokens(subtask),
        requires_code_generation="implement" in subtask.description.lower(),
        requires_reasoning="analyze" in subtask.description.lower(),
        requires_review="review" in subtask.description.lower(),
        language_hints=detect_languages(subtask.context),
        complexity_score=estimate_complexity(subtask)
    )
```

---

## Subagent Spawning Pattern

### Process Isolation Model

```
Main Process (Orchestrator)
    |
    +-- spawn --> Subagent Process 1
    |                  |
    |                  +-- stdin: prompt
    |                  +-- stdout: result
    |                  +-- stderr: errors
    |
    +-- spawn --> Subagent Process 2
    |                  |
    |                  +-- ...
    |
    +-- spawn --> Subagent Process N
```

### Spawn Configuration

```python
@dataclass
class SpawnConfig:
    agent_type: str
    working_directory: str
    timeout_seconds: int
    max_memory_mb: int
    environment: Dict[str, str]
    capture_output: bool = True
    stream_output: bool = True
```

### Spawner Implementation

```python
class ClaudeCodeSubagentSpawner:
    def spawn(self, task: Task, config: SpawnConfig) -> SubagentHandle:
        cmd = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=config.working_directory,
            env=config.environment
        )

        # Send prompt
        process.stdin.write(task.prompt.encode())
        process.stdin.close()

        return SubagentHandle(process, config)
```

---

## Error Handling & Recovery

### Error Categories

```python
class AgentErrorType(Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    INVALID_OUTPUT = "invalid_output"
    PROCESS_CRASH = "process_crash"
    NETWORK_ERROR = "network_error"
```

### Recovery Strategies

| Error Type | Strategy | Max Retries |
|------------|----------|-------------|
| TIMEOUT | Retry with increased timeout | 2 |
| RATE_LIMIT | Exponential backoff | 5 |
| CONTEXT_OVERFLOW | Chunk and retry | 3 |
| INVALID_OUTPUT | Retry with clarification | 2 |
| PROCESS_CRASH | Spawn new process | 3 |
| NETWORK_ERROR | Retry with backoff | 5 |

### Recovery Implementation

```python
class AgentErrorHandler:
    def handle(self, error: AgentError, context: ExecutionContext) -> Action:
        strategy = self.get_strategy(error.type)

        if context.retry_count >= strategy.max_retries:
            return Action.ESCALATE

        if error.type == AgentErrorType.RATE_LIMIT:
            wait_time = self.calculate_backoff(context.retry_count)
            return Action.RETRY_AFTER(wait_time)

        if error.type == AgentErrorType.CONTEXT_OVERFLOW:
            chunked_task = self.chunk_task(context.task)
            return Action.RETRY_WITH(chunked_task)

        return Action.RETRY
```

---

## Agent Prompting Patterns

### System Prompt Template

```python
AGENT_SYSTEM_PROMPT = """
You are a specialized {agent_type} agent in a multi-agent system.

Your role: {role_description}

Current task context:
{context}

Collaboration notes:
- Other agents may have completed related tasks
- Your output will be used by downstream agents
- Be precise and include all necessary details

Output format:
{output_format}
"""
```

### Task Prompt Template

```python
TASK_PROMPT = """
## Task
{task_description}

## Context
{shared_context}

## Dependencies
{completed_dependencies}

## Expected Output
{output_specification}

## Constraints
- Time limit: {timeout}
- Output format: {format}
"""
```

---

## Coordination Patterns

### Sequential Coordination

```
Task A --> Agent 1 --> Result A --> Task B --> Agent 2 --> Result B
```

### Parallel Coordination

```
         +-> Task A --> Agent 1 --> Result A -+
         |                                    |
Task --> +-> Task B --> Agent 2 --> Result B -+--> Aggregate
         |                                    |
         +-> Task C --> Agent 3 --> Result C -+
```

### Pipeline Coordination

```
Research (Gemini) --> Plan (Codex) --> Implement (Codex) --> Review (Claude)
```

### Iterative Coordination

```
Task --> Agent --> Result --> Evaluate --> [Pass] --> Done
                     ^                       |
                     |                    [Fail]
                     |                       |
                     +--- Refine <-----------+
```

---

## Metrics & Observability

### Agent Metrics

```python
@dataclass
class AgentMetrics:
    agent_id: str
    tasks_completed: int
    tasks_failed: int
    avg_completion_time: float
    token_usage: int
    error_rate: float
    uptime: float
```

### Telemetry Events

```python
class AgentTelemetry:
    def emit_task_started(self, agent_id: str, task_id: str):
        pass

    def emit_task_completed(self, agent_id: str, task_id: str, duration: float):
        pass

    def emit_error(self, agent_id: str, error: Exception):
        pass

    def emit_checkpoint(self, agent_id: str, state: Dict):
        pass
```

---

*Last Updated: December 2024*
