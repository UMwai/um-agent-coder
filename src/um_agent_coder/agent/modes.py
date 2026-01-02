"""
Agent modes inspired by Roo-Code's multi-mode architecture.
Each mode represents a specialized persona with specific capabilities and approaches.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AgentMode(Enum):
    """Available agent modes."""

    CODE = "code"
    ARCHITECT = "architect"
    ASK = "ask"
    DEBUG = "debug"
    REVIEW = "review"
    CUSTOM = "custom"


@dataclass
class ModeConfig:
    """Configuration for an agent mode."""

    name: str
    description: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    preferred_tools: list[str] = None
    auto_approve_actions: list[str] = None
    context_priorities: dict[str, int] = None

    def __post_init__(self):
        if self.preferred_tools is None:
            self.preferred_tools = []
        if self.auto_approve_actions is None:
            self.auto_approve_actions = []
        if self.context_priorities is None:
            self.context_priorities = {}


class ModeManager:
    """Manages different agent modes and their configurations."""

    def __init__(self):
        self.modes: dict[AgentMode, ModeConfig] = {}
        self._initialize_default_modes()
        self.current_mode: AgentMode = AgentMode.CODE

    def _initialize_default_modes(self):
        """Initialize default agent modes."""

        # Code Mode - General coding tasks
        self.modes[AgentMode.CODE] = ModeConfig(
            name="Code Mode",
            description="General-purpose coding mode for implementing features and modifications",
            system_prompt="""You are an expert software engineer focused on writing clean, efficient, and maintainable code.
Your priorities:
1. Write idiomatic code that follows project conventions
2. Implement robust error handling
3. Ensure code is testable and maintainable
4. Follow SOLID principles
5. Add appropriate documentation

When implementing features:
- Analyze existing code patterns first
- Use appropriate design patterns
- Consider edge cases
- Write self-documenting code
- Validate inputs and handle errors gracefully""",
            temperature=0.7,
            preferred_tools=["FileReader", "FileWriter", "CodeSearcher", "CommandExecutor"],
            auto_approve_actions=["FileReader", "CodeSearcher"],
            context_priorities={
                "project_structure": 9,
                "related_code": 10,
                "test_files": 7,
                "documentation": 6,
            },
        )

        # Architect Mode - System design and planning
        self.modes[AgentMode.ARCHITECT] = ModeConfig(
            name="Architect Mode",
            description="Technical architecture and system design mode",
            system_prompt="""You are a software architect focused on system design and technical planning.
Your priorities:
1. Design scalable and maintainable architectures
2. Consider performance, security, and reliability
3. Create clear technical specifications
4. Identify potential technical debt
5. Plan for future extensibility

When designing systems:
- Start with high-level architecture
- Consider trade-offs explicitly
- Document architectural decisions
- Plan for monitoring and observability
- Consider deployment and operations""",
            temperature=0.8,
            preferred_tools=["ProjectAnalyzer", "FileReader", "FileWriter"],
            auto_approve_actions=["ProjectAnalyzer", "FileReader"],
            context_priorities={
                "project_structure": 10,
                "architecture_docs": 9,
                "config_files": 8,
                "interfaces": 8,
            },
        )

        # Ask Mode - Information and Q&A
        self.modes[AgentMode.ASK] = ModeConfig(
            name="Ask Mode",
            description="Information retrieval and question answering mode",
            system_prompt="""You are a knowledgeable assistant focused on providing accurate and helpful information.
Your priorities:
1. Provide clear and concise answers
2. Reference specific code locations when relevant
3. Explain technical concepts clearly
4. Suggest relevant documentation
5. Anticipate follow-up questions

When answering questions:
- Be direct and to the point
- Use examples when helpful
- Reference official documentation
- Explain trade-offs when relevant
- Suggest best practices""",
            temperature=0.5,
            preferred_tools=["FileReader", "CodeSearcher", "ProjectAnalyzer"],
            auto_approve_actions=["FileReader", "CodeSearcher", "ProjectAnalyzer"],
            context_priorities={"documentation": 10, "related_code": 8, "project_info": 7},
        )

        # Debug Mode - Problem diagnosis and fixing
        self.modes[AgentMode.DEBUG] = ModeConfig(
            name="Debug Mode",
            description="Debugging and problem-solving mode",
            system_prompt="""You are a debugging expert focused on identifying and fixing issues.
Your priorities:
1. Systematically identify root causes
2. Use debugging tools effectively
3. Write comprehensive test cases
4. Document fixes and preventions
5. Consider edge cases and race conditions

When debugging:
- Start with error messages and stack traces
- Reproduce issues consistently
- Use logging and debugging tools
- Test fixes thoroughly
- Document the solution and prevention""",
            temperature=0.3,
            preferred_tools=["FileReader", "CodeSearcher", "CommandExecutor", "FileWriter"],
            auto_approve_actions=["FileReader", "CodeSearcher"],
            context_priorities={
                "error_logs": 10,
                "stack_traces": 10,
                "related_code": 9,
                "test_files": 8,
            },
        )

        # Review Mode - Code review and quality assurance
        self.modes[AgentMode.REVIEW] = ModeConfig(
            name="Review Mode",
            description="Code review and quality assurance mode",
            system_prompt="""You are a code reviewer focused on quality, security, and best practices.
Your priorities:
1. Identify potential bugs and security issues
2. Suggest performance improvements
3. Ensure code follows project standards
4. Check for proper error handling
5. Verify test coverage

When reviewing code:
- Check for common anti-patterns
- Verify security best practices
- Assess code readability
- Suggest refactoring opportunities
- Ensure proper documentation""",
            temperature=0.4,
            preferred_tools=["FileReader", "CodeSearcher", "ProjectAnalyzer"],
            auto_approve_actions=["FileReader", "CodeSearcher", "ProjectAnalyzer"],
            context_priorities={
                "changed_files": 10,
                "test_files": 9,
                "related_code": 8,
                "style_guides": 7,
            },
        )

    def set_mode(self, mode: AgentMode) -> ModeConfig:
        """Set the current agent mode."""
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}")
        self.current_mode = mode
        return self.modes[mode]

    def get_current_mode(self) -> ModeConfig:
        """Get the current mode configuration."""
        return self.modes[self.current_mode]

    def add_custom_mode(self, name: str, config: ModeConfig):
        """Add a custom mode configuration."""
        self.modes[AgentMode.CUSTOM] = config

    def detect_mode_from_prompt(self, prompt: str) -> AgentMode:
        """Detect the appropriate mode from user prompt."""
        prompt_lower = prompt.lower()

        # Mode detection keywords
        debug_keywords = ["debug", "fix", "error", "bug", "issue", "problem", "crash", "exception"]
        architect_keywords = ["design", "architecture", "structure", "plan", "scale", "system"]
        ask_keywords = ["what", "how", "why", "explain", "tell me", "describe", "understand"]
        review_keywords = ["review", "check", "audit", "quality", "improve", "refactor"]

        # Check for mode indicators
        if any(keyword in prompt_lower for keyword in debug_keywords):
            return AgentMode.DEBUG
        elif any(keyword in prompt_lower for keyword in architect_keywords):
            return AgentMode.ARCHITECT
        elif any(keyword in prompt_lower for keyword in ask_keywords):
            return AgentMode.ASK
        elif any(keyword in prompt_lower for keyword in review_keywords):
            return AgentMode.REVIEW
        else:
            return AgentMode.CODE

    def get_mode_prompt(self, mode: Optional[AgentMode] = None) -> str:
        """Get the system prompt for a mode."""
        if mode is None:
            mode = self.current_mode
        return self.modes[mode].system_prompt

    def get_mode_tools(self, mode: Optional[AgentMode] = None) -> list[str]:
        """Get preferred tools for a mode."""
        if mode is None:
            mode = self.current_mode
        return self.modes[mode].preferred_tools

    def should_auto_approve(self, action: str, mode: Optional[AgentMode] = None) -> bool:
        """Check if an action should be auto-approved in the current mode."""
        if mode is None:
            mode = self.current_mode
        return action in self.modes[mode].auto_approve_actions
