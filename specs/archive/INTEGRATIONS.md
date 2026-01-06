# um-agent-coder Integrations

## Overview

um-agent-coder integrates with multiple LLM providers, data sources, and tooling ecosystems to provide comprehensive AI-assisted development capabilities.

---

## LLM Provider Integrations

### 1. OpenAI Integration

**Models Supported**:
- GPT-5.2 (via Codex CLI)
- GPT-4 Turbo
- GPT-4

**Authentication**:
```bash
export OPENAI_API_KEY="sk-..."
```

**Configuration**:
```yaml
llm:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-5.2
    temperature: 0.3
    max_tokens: 8000
    timeout: 120
```

**Usage**:
```python
from um_agent_coder.llm.providers.openai import OpenAILLM

llm = OpenAILLM(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-5.2"
)
response = llm.generate("Write a Python function...")
```

### 2. Anthropic Integration

**Models Supported**:
- Claude Opus 4.5
- Claude Sonnet 4
- Claude 3.5 Sonnet

**Authentication**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Configuration**:
```yaml
llm:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-opus-4.5
    max_tokens: 8000
    temperature: 0.5
```

**Usage**:
```python
from um_agent_coder.llm.providers.anthropic import AnthropicLLM

llm = AnthropicLLM(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-opus-4.5"
)
response = llm.generate("Analyze this code...")
```

### 3. Google AI Integration

**Models Supported**:
- Gemini 3 Pro (1M context)
- Gemini 3 Flash
- Gemini 2.5 Pro

**Authentication Methods**:

**Option A: API Key**
```bash
export GOOGLE_API_KEY="AIza..."
```

**Option B: Application Default Credentials (ADC)**
```bash
gcloud auth application-default login
```

**Configuration**:
```yaml
llm:
  google:
    api_key: ${GOOGLE_API_KEY}
    model: gemini-3-pro
    temperature: 0.7
```

**Usage**:
```python
from um_agent_coder.llm.providers.google import GoogleLLM

llm = GoogleLLM(
    api_key=os.environ["GOOGLE_API_KEY"],
    model="gemini-3-pro"
)
response = llm.generate("Explore this codebase...")
```

### 4. MCP Local Integration (No API Keys)

**Supported Backends**:
- Gemini CLI (via `mcp__gemini-cli__ask-gemini`)
- Codex CLI (via `mcp__codex__codex`)
- Claude CLI (via subprocess)

**Benefits**:
- No API keys required
- Uses CLI OAuth authentication
- Direct tool invocation

**Configuration**:
```yaml
llm:
  provider: mcp_local
  mcp_local:
    gemini_backend: gemini
    codex_backend: codex
    claude_backend: claude
```

**Usage**:
```python
from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# No API key needed - uses CLI OAuth
gemini = MCPLocalLLM(backend="gemini", model="gemini-3-pro")
codex = MCPLocalLLM(backend="codex", model="gpt-5.2")
claude = MCPLocalLLM(backend="claude", model="claude-opus-4.5")
```

---

## Data Source Integrations

### 1. SEC EDGAR

**Purpose**: Fetch company filings (10-K, 10-Q, 8-K, etc.)

**API Endpoint**: `https://data.sec.gov/`

**Configuration**:
```yaml
data_fetchers:
  sec_edgar:
    enabled: true
    user_agent: "YourApp/1.0 (your@email.com)"
    rate_limit_per_second: 10
```

**Usage**:
```python
from um_agent_coder.orchestrator.data_fetchers import SECEdgarFetcher

fetcher = SECEdgarFetcher()
filings = fetcher.get_filings(
    cik="0001018724",  # Amazon
    filing_type="10-K",
    count=5
)
```

**Available Methods**:
- `get_filings(cik, filing_type, count)`: Fetch filings by CIK
- `search_company(name)`: Search for company CIK
- `get_filing_content(url)`: Download filing content
- `parse_10k(content)`: Parse 10-K structure

### 2. Yahoo Finance

**Purpose**: Stock data, fundamentals, market data

**Configuration**:
```yaml
data_fetchers:
  yahoo_finance:
    enabled: true
    rate_limit_per_second: 5
```

**Usage**:
```python
from um_agent_coder.orchestrator.data_fetchers import YahooFinanceFetcher

fetcher = YahooFinanceFetcher()

# Stock price data
prices = fetcher.get_stock_data("AAPL", period="1y")

# Company fundamentals
fundamentals = fetcher.get_fundamentals("AAPL")

# Financial statements
financials = fetcher.get_financials("AAPL")
```

**Available Methods**:
- `get_stock_data(ticker, period, interval)`: Historical prices
- `get_fundamentals(ticker)`: Company fundamentals
- `get_financials(ticker)`: Income statement, balance sheet
- `get_options_chain(ticker)`: Options data
- `get_analyst_ratings(ticker)`: Analyst recommendations

### 3. ClinicalTrials.gov

**Purpose**: Clinical trial data for biotech analysis

**API Endpoint**: `https://clinicaltrials.gov/api/v2/`

**Configuration**:
```yaml
data_fetchers:
  clinical_trials:
    enabled: true
    rate_limit_per_second: 3
```

**Usage**:
```python
from um_agent_coder.orchestrator.data_fetchers import ClinicalTrialsFetcher

fetcher = ClinicalTrialsFetcher()

# Search trials
trials = fetcher.search_trials(
    sponsor="Pfizer",
    condition="cancer",
    status="RECRUITING"
)

# Get trial details
details = fetcher.get_trial("NCT12345678")
```

**Available Methods**:
- `search_trials(sponsor, condition, status)`: Search trials
- `get_trial(nct_id)`: Get trial details
- `get_sponsor_trials(sponsor)`: All trials by sponsor
- `get_pipeline_analysis(company)`: Pipeline analysis

### 4. News API Integration

**Purpose**: Market news, sentiment analysis

**Supported Sources**:
- NewsAPI
- Alpha Vantage News
- RSS feeds

**Configuration**:
```yaml
data_fetchers:
  news:
    enabled: true
    provider: newsapi  # or alpha_vantage, rss
    api_key: ${NEWS_API_KEY}
```

**Usage**:
```python
from um_agent_coder.orchestrator.data_fetchers import NewsFetcher

fetcher = NewsFetcher(api_key=os.environ["NEWS_API_KEY"])

# Search news
news = fetcher.search(
    query="Apple earnings",
    from_date="2024-01-01",
    sort_by="relevancy"
)

# Get company news
company_news = fetcher.get_company_news("AAPL", days=7)
```

---

## CLI Tool Integrations

### 1. Claude Code CLI

**Integration Method**: Subprocess spawning

**Command Format**:
```bash
claude --print --output-format stream-json --dangerously-skip-permissions
```

**Configuration**:
```yaml
cli_tools:
  claude:
    binary: claude
    args:
      - "--print"
      - "--output-format"
      - "stream-json"
      - "--dangerously-skip-permissions"
    timeout: 300
```

**Usage in Harness**:
```python
from um_agent_coder.harness.executors import ClaudeExecutor

executor = ClaudeExecutor()
result = executor.execute(task="Implement feature X")
```

### 2. Codex CLI

**Integration Method**: Subprocess spawning

**Command Format**:
```bash
codex --ask-for-approval never --sandbox danger-full-access exec "prompt"
```

**Configuration**:
```yaml
cli_tools:
  codex:
    binary: codex
    args:
      - "--ask-for-approval"
      - "never"
      - "--sandbox"
      - "danger-full-access"
      - "exec"
    timeout: 600
```

### 3. Gemini CLI

**Integration Method**: MCP or subprocess

**Command Format**:
```bash
gemini -m gemini-3-pro -o stream-json -y "prompt"
```

**Configuration**:
```yaml
cli_tools:
  gemini:
    binary: gemini
    args:
      - "-m"
      - "gemini-3-pro"
      - "-o"
      - "stream-json"
      - "-y"
    timeout: 300
```

---

## MCP (Model Context Protocol) Integrations

### Gemini MCP Server

**Tool Name**: `mcp__gemini-cli__ask-gemini`

**Parameters**:
```json
{
  "prompt": "Your analysis request",
  "model": "gemini-3-pro",
  "sandbox": false,
  "changeMode": false
}
```

**Usage**:
```python
# Via MCPLocalLLM
llm = MCPLocalLLM(backend="gemini")
result = llm.generate("@file.py analyze this code")
```

### Codex MCP Server

**Tool Name**: `mcp__codex__codex`

**Parameters**:
```json
{
  "prompt": "Task description",
  "approval-policy": "never",
  "sandbox": "danger-full-access",
  "model": "gpt-5.2"
}
```

**Usage**:
```python
# Via MCPLocalLLM
llm = MCPLocalLLM(backend="codex")
result = llm.generate("Implement a REST API")
```

---

## Future Integrations (Planned)

### Version Control

**GitHub Integration**:
- PR creation/management
- Issue tracking
- Code review automation
- Actions integration

```yaml
# Planned configuration
integrations:
  github:
    enabled: true
    token: ${GITHUB_TOKEN}
    features:
      - pr_creation
      - issue_management
      - code_review
```

### Communication

**Slack Integration**:
- Task completion notifications
- Error alerts
- Status updates

```yaml
# Planned configuration
integrations:
  slack:
    enabled: true
    webhook_url: ${SLACK_WEBHOOK}
    channels:
      notifications: "#ai-agent-updates"
      errors: "#ai-agent-errors"
```

### Monitoring

**Prometheus/Grafana**:
- Metrics export
- Dashboard integration
- Alerting

```yaml
# Planned configuration
integrations:
  prometheus:
    enabled: true
    port: 9090
    metrics:
      - task_duration
      - token_usage
      - error_rate
```

### CI/CD

**GitHub Actions**:
- Automated task execution
- PR triggers
- Scheduled runs

```yaml
# Planned workflow
name: AI Agent Task
on:
  workflow_dispatch:
    inputs:
      task:
        description: 'Task to execute'
        required: true

jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run um-agent-coder
        run: um-agent "${{ inputs.task }}"
```

---

## Integration Authentication Summary

| Integration | Auth Method | Environment Variable |
|-------------|-------------|---------------------|
| OpenAI | API Key | `OPENAI_API_KEY` |
| Anthropic | API Key | `ANTHROPIC_API_KEY` |
| Google AI | API Key or ADC | `GOOGLE_API_KEY` |
| SEC EDGAR | User Agent | None (configured) |
| Yahoo Finance | None | None |
| ClinicalTrials.gov | None | None |
| News API | API Key | `NEWS_API_KEY` |
| MCP Tools | CLI OAuth | None |

---

## Integration Best Practices

### Rate Limiting

All integrations implement rate limiting:

```python
from um_agent_coder.utils.rate_limiter import RateLimiter

limiter = RateLimiter(requests_per_second=10)

@limiter.limit
def api_call():
    pass
```

### Error Handling

Consistent error handling across integrations:

```python
from um_agent_coder.integrations.errors import (
    IntegrationError,
    RateLimitError,
    AuthenticationError,
    ConnectionError
)

try:
    result = integration.fetch(...)
except RateLimitError:
    # Wait and retry
except AuthenticationError:
    # Re-authenticate
except ConnectionError:
    # Retry with backoff
```

### Caching

Response caching for expensive calls:

```python
from um_agent_coder.utils.cache import cache

@cache(ttl=3600)  # 1 hour
def get_company_data(ticker: str):
    return fetcher.get_fundamentals(ticker)
```

---

*Last Updated: December 2024*
