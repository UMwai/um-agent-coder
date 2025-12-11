"""
Task Decomposer - Breaks ambiguous, complex tasks into structured subtasks.

For tasks like "identify biotech M&A opportunities", this module:
1. Clarifies the task by identifying implicit requirements
2. Decomposes into research questions
3. Identifies data sources needed
4. Creates an execution plan with dependencies
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class SubTaskType(Enum):
    """Types of subtasks in a decomposed workflow."""
    RESEARCH = "research"           # Gather information
    DATA_FETCH = "data_fetch"       # Pull from APIs/databases
    ANALYSIS = "analysis"           # Process/analyze data
    CODE_GEN = "code_generation"    # Generate code/scripts
    SYNTHESIS = "synthesis"         # Combine findings
    VALIDATION = "validation"       # Check quality/accuracy
    REPORT = "report"               # Generate final output


class ModelRole(Enum):
    """Which model is best suited for each task type."""
    GEMINI = "gemini"       # Large context, research, web search
    CODEX = "codex"         # Code generation, data processing
    CLAUDE = "claude"       # Reasoning, synthesis, safety review


@dataclass
class DataSource:
    """A data source that may be needed for the task."""
    name: str
    type: str  # api, database, web, file
    description: str
    url: Optional[str] = None
    requires_auth: bool = False
    priority: int = 5  # 1-10, higher = more important


@dataclass
class SubTask:
    """A single subtask in a decomposed workflow."""
    id: str
    description: str
    type: SubTaskType
    model: ModelRole
    prompt: str

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Data requirements
    data_sources: List[DataSource] = field(default_factory=list)
    input_from: List[str] = field(default_factory=list)  # IDs of tasks whose output we need

    # Execution
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Metadata
    estimated_tokens: int = 1000
    priority: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type.value,
            "model": self.model.value,
            "prompt": self.prompt,
            "depends_on": self.depends_on,
            "data_sources": [
                {"name": ds.name, "type": ds.type, "description": ds.description}
                for ds in self.data_sources
            ],
            "input_from": self.input_from,
            "status": self.status,
            "priority": self.priority
        }


@dataclass
class DecomposedTask:
    """A fully decomposed task with subtasks and execution plan."""
    original_prompt: str
    clarified_goal: str
    subtasks: List[SubTask]
    execution_order: List[str]  # Ordered list of subtask IDs
    data_sources: List[DataSource]
    estimated_total_tokens: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "clarified_goal": self.clarified_goal,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "execution_order": self.execution_order,
            "data_sources": [
                {"name": ds.name, "type": ds.type, "url": ds.url}
                for ds in self.data_sources
            ],
            "estimated_total_tokens": self.estimated_total_tokens,
            "created_at": self.created_at
        }


class TaskDecomposer:
    """
    Decomposes complex, ambiguous tasks into structured subtasks.

    Uses an LLM to:
    1. Understand the implicit requirements
    2. Generate research questions
    3. Identify needed data sources
    4. Create execution plan with model assignments

    Example:
        decomposer = TaskDecomposer(llm)
        result = decomposer.decompose("identify biotech M&A opportunities")

        # Result contains:
        # - Clarified goal
        # - List of subtasks with model assignments
        # - Data sources needed
        # - Execution order respecting dependencies
    """

    # Templates for decomposition prompts
    DECOMPOSITION_PROMPT = '''You are a task planning expert. Decompose this complex task into structured subtasks.

TASK: {prompt}

Analyze this task and provide a JSON response with:

1. **clarified_goal**: What is the user actually trying to achieve? Be specific.

2. **research_questions**: List 3-7 specific questions that need to be answered.

3. **data_sources**: What data sources would help? For each:
   - name: Short identifier
   - type: "api", "database", "web", "file"
   - description: What data it provides
   - url: API endpoint or website (if known)
   - priority: 1-10

4. **subtasks**: Break into executable steps. For each:
   - id: Unique identifier (e.g., "research_1", "analyze_2")
   - description: What this step does
   - type: One of: research, data_fetch, analysis, code_generation, synthesis, validation, report
   - model: Best model for this task:
     * "gemini" - Large context, research, web search, document analysis
     * "codex" - Code generation, data processing, API calls
     * "claude" - Reasoning, synthesis, judgment, safety review
   - prompt: The actual prompt to send to that model
   - depends_on: List of subtask IDs this depends on
   - input_from: List of subtask IDs whose output this needs

5. **execution_order**: Ordered list of subtask IDs respecting dependencies

Respond ONLY with valid JSON, no markdown.'''

    BIOTECH_MA_EXAMPLE = {
        "clarified_goal": "Identify publicly traded biotech companies that are likely acquisition targets based on pipeline value, financial metrics, and market conditions",
        "research_questions": [
            "Which biotech companies have promising late-stage pipelines but limited commercialization capability?",
            "What are recent M&A trends and valuation multiples in biotech?",
            "Which companies have cash runway concerns that might motivate a sale?",
            "What therapeutic areas are large pharma actively acquiring in?",
            "Which companies have insider buying or unusual options activity?"
        ],
        "data_sources": [
            {"name": "sec_edgar", "type": "api", "description": "SEC filings for financial data", "url": "https://www.sec.gov/cgi-bin/browse-edgar", "priority": 9},
            {"name": "clinicaltrials", "type": "api", "description": "Clinical trial status and results", "url": "https://clinicaltrials.gov/api", "priority": 8},
            {"name": "yahoo_finance", "type": "api", "description": "Stock prices, market cap, financials", "url": "https://finance.yahoo.com", "priority": 8},
            {"name": "pubmed", "type": "api", "description": "Scientific publications and citations", "url": "https://pubmed.ncbi.nlm.nih.gov", "priority": 6},
            {"name": "news_api", "type": "api", "description": "Recent news and press releases", "url": "https://newsapi.org", "priority": 7}
        ],
        "subtasks": [
            {
                "id": "research_trends",
                "description": "Research recent biotech M&A trends and valuation metrics",
                "type": "research",
                "model": "gemini",
                "prompt": "Research biotech M&A activity in the past 2 years. Identify: 1) Average acquisition premiums, 2) Most active acquirers, 3) Therapeutic areas with most activity, 4) Deal size ranges. Focus on facts and specific deals.",
                "depends_on": [],
                "input_from": []
            },
            {
                "id": "fetch_biotech_list",
                "description": "Get list of small/mid-cap biotech companies",
                "type": "data_fetch",
                "model": "codex",
                "prompt": "Write Python code to fetch a list of biotech companies with market cap $500M-$10B from Yahoo Finance or a financial API. Include: ticker, name, market cap, sector.",
                "depends_on": [],
                "input_from": []
            },
            {
                "id": "fetch_pipeline_data",
                "description": "Gather clinical pipeline information",
                "type": "data_fetch",
                "model": "codex",
                "prompt": "Write code to query ClinicalTrials.gov API for Phase 2/3 trials sponsored by biotech companies. Extract: company, drug name, indication, phase, status, expected completion.",
                "depends_on": ["fetch_biotech_list"],
                "input_from": ["fetch_biotech_list"]
            },
            {
                "id": "analyze_financials",
                "description": "Analyze financial health and cash runway",
                "type": "analysis",
                "model": "codex",
                "prompt": "Given the list of biotech companies, write code to calculate: cash runway (cash / quarterly burn), debt levels, recent financing activity. Flag companies with <18 months runway.",
                "depends_on": ["fetch_biotech_list"],
                "input_from": ["fetch_biotech_list"]
            },
            {
                "id": "score_targets",
                "description": "Score and rank potential acquisition targets",
                "type": "analysis",
                "model": "claude",
                "prompt": "Given the M&A trends, pipeline data, and financial analysis, score each company as an acquisition target (1-10). Consider: pipeline value, financial pressure, strategic fit, recent news. Explain your reasoning.",
                "depends_on": ["research_trends", "fetch_pipeline_data", "analyze_financials"],
                "input_from": ["research_trends", "fetch_pipeline_data", "analyze_financials"]
            },
            {
                "id": "validate_findings",
                "description": "Validate top candidates with recent news",
                "type": "validation",
                "model": "gemini",
                "prompt": "For the top 10 scored companies, search for recent news about: acquisition rumors, partnership discussions, insider trading, analyst ratings. Flag any red flags or confirmatory signals.",
                "depends_on": ["score_targets"],
                "input_from": ["score_targets"]
            },
            {
                "id": "generate_report",
                "description": "Generate final M&A opportunities report",
                "type": "report",
                "model": "claude",
                "prompt": "Synthesize all findings into an executive report on biotech M&A opportunities. Include: methodology, top 5 candidates with detailed profiles, risk factors, recommended next steps for due diligence.",
                "depends_on": ["validate_findings"],
                "input_from": ["research_trends", "score_targets", "validate_findings"]
            }
        ],
        "execution_order": [
            "research_trends",
            "fetch_biotech_list",
            "fetch_pipeline_data",
            "analyze_financials",
            "score_targets",
            "validate_findings",
            "generate_report"
        ]
    }

    def __init__(self, llm=None):
        """
        Initialize decomposer.

        Args:
            llm: LLM instance for decomposition (uses Claude-like reasoning)
        """
        self.llm = llm

    def decompose(self, prompt: str, use_llm: bool = True) -> DecomposedTask:
        """
        Decompose a complex task into subtasks.

        Args:
            prompt: The original user prompt
            use_llm: If True, use LLM for decomposition. If False, use heuristics.

        Returns:
            DecomposedTask with all subtasks and execution plan
        """
        if use_llm and self.llm:
            return self._decompose_with_llm(prompt)
        else:
            return self._decompose_with_heuristics(prompt)

    def _decompose_with_llm(self, prompt: str) -> DecomposedTask:
        """Use LLM to decompose the task."""
        decomposition_prompt = self.DECOMPOSITION_PROMPT.format(prompt=prompt)

        response = self.llm.chat(decomposition_prompt)

        try:
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            return self._build_decomposed_task(prompt, data)

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            # Fall back to heuristics if LLM response isn't valid JSON
            print(f"Warning: Could not parse LLM response, using heuristics: {e}")
            return self._decompose_with_heuristics(prompt)

    def _decompose_with_heuristics(self, prompt: str) -> DecomposedTask:
        """Use heuristics to decompose common task types."""
        prompt_lower = prompt.lower()

        # Detect task type and use appropriate template
        if any(kw in prompt_lower for kw in ["m&a", "acquisition", "merger", "buyout"]):
            if any(kw in prompt_lower for kw in ["biotech", "pharma", "drug", "therapeutic"]):
                return self._build_decomposed_task(prompt, self.BIOTECH_MA_EXAMPLE)

        # Generic research task decomposition
        return self._build_generic_research_task(prompt)

    def _build_decomposed_task(self, original_prompt: str, data: Dict) -> DecomposedTask:
        """Build DecomposedTask from parsed data."""
        # Build data sources
        data_sources = []
        for ds in data.get("data_sources", []):
            data_sources.append(DataSource(
                name=ds["name"],
                type=ds["type"],
                description=ds.get("description", ""),
                url=ds.get("url"),
                requires_auth=ds.get("requires_auth", False),
                priority=ds.get("priority", 5)
            ))

        # Build subtasks
        subtasks = []
        for st in data.get("subtasks", []):
            # Map data sources to subtask
            st_data_sources = [
                ds for ds in data_sources
                if ds.name in st.get("data_sources", [])
            ]

            subtasks.append(SubTask(
                id=st["id"],
                description=st["description"],
                type=SubTaskType(st["type"]),
                model=ModelRole(st["model"]),
                prompt=st["prompt"],
                depends_on=st.get("depends_on", []),
                data_sources=st_data_sources,
                input_from=st.get("input_from", []),
                estimated_tokens=st.get("estimated_tokens", 1000),
                priority=st.get("priority", 5)
            ))

        # Calculate total tokens
        total_tokens = sum(st.estimated_tokens for st in subtasks)

        return DecomposedTask(
            original_prompt=original_prompt,
            clarified_goal=data.get("clarified_goal", original_prompt),
            subtasks=subtasks,
            execution_order=data.get("execution_order", [st.id for st in subtasks]),
            data_sources=data_sources,
            estimated_total_tokens=total_tokens
        )

    def _build_generic_research_task(self, prompt: str) -> DecomposedTask:
        """Build a generic research task decomposition."""
        subtasks = [
            SubTask(
                id="clarify",
                description="Clarify requirements and scope",
                type=SubTaskType.RESEARCH,
                model=ModelRole.CLAUDE,
                prompt=f"Analyze this request and identify: 1) Key objectives, 2) Implicit requirements, 3) Success criteria, 4) Potential ambiguities to resolve.\n\nRequest: {prompt}",
                depends_on=[],
                input_from=[]
            ),
            SubTask(
                id="research",
                description="Gather relevant information",
                type=SubTaskType.RESEARCH,
                model=ModelRole.GEMINI,
                prompt=f"Research and gather comprehensive information about: {prompt}\n\nFocus on facts, data, and authoritative sources.",
                depends_on=["clarify"],
                input_from=["clarify"]
            ),
            SubTask(
                id="analyze",
                description="Analyze gathered information",
                type=SubTaskType.ANALYSIS,
                model=ModelRole.CODEX,
                prompt=f"Given the research findings, analyze the data and identify patterns, insights, and actionable conclusions.\n\nOriginal request: {prompt}",
                depends_on=["research"],
                input_from=["research"]
            ),
            SubTask(
                id="synthesize",
                description="Synthesize findings into report",
                type=SubTaskType.REPORT,
                model=ModelRole.CLAUDE,
                prompt=f"Synthesize all findings into a comprehensive report addressing: {prompt}\n\nInclude: executive summary, key findings, recommendations, and next steps.",
                depends_on=["analyze"],
                input_from=["clarify", "research", "analyze"]
            )
        ]

        return DecomposedTask(
            original_prompt=prompt,
            clarified_goal=prompt,
            subtasks=subtasks,
            execution_order=["clarify", "research", "analyze", "synthesize"],
            data_sources=[],
            estimated_total_tokens=4000
        )

    def visualize(self, task: DecomposedTask) -> str:
        """Generate ASCII visualization of task dependencies."""
        lines = [
            "=" * 60,
            "TASK DECOMPOSITION",
            "=" * 60,
            f"Goal: {task.clarified_goal[:70]}...",
            "",
            "EXECUTION FLOW:",
            "-" * 40
        ]

        for i, task_id in enumerate(task.execution_order):
            subtask = next((st for st in task.subtasks if st.id == task_id), None)
            if not subtask:
                continue

            model_icon = {"gemini": "[G]", "codex": "[C]", "claude": "[A]"}
            icon = model_icon.get(subtask.model.value, "[?]")

            deps = f" <- {subtask.depends_on}" if subtask.depends_on else ""
            lines.append(f"  {i+1}. {icon} {subtask.description}{deps}")

        lines.extend([
            "",
            "DATA SOURCES:",
            "-" * 40
        ])

        for ds in task.data_sources[:5]:
            lines.append(f"  - {ds.name}: {ds.description[:40]}...")

        lines.extend([
            "",
            f"Estimated tokens: {task.estimated_total_tokens:,}",
            "=" * 60
        ])

        return "\n".join(lines)
