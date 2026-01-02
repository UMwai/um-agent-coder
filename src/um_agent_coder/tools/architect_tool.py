from typing import Any

from ..llm.base import LLM
from .base import Tool, ToolResult


class ArchitectTool(Tool):
    """
    Tool for generating technical specifications from abstract prompts.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = "ArchitectTool"
        self.description = "Generate detailed technical specifications and architecture from abstract requirements."

    def execute(self, prompt: str, **kwargs) -> ToolResult:
        """
        Generate a technical specification.
        """
        try:
            # Architect prompt
            architect_prompt = f"""
You are an Expert Software Architect. Your goal is to convert the following abstract user request into a detailed technical specification.

User Request: "{prompt}"

Please generate a comprehensive Markdown specification (SPEC.md) that includes:
1. **Project Overview**: High-level summary of the system.
2. **User Stories**: Key features from a user perspective.
3. **System Architecture**:
   - High-level components.
   - Data flow.
   - Tech stack recommendations (Language, Frameworks, DB).
4. **Data Models**:
   - Key entities and their relationships.
   - Schema definitions (pseudo-code or SQL).
5. **API Design**:
   - Key endpoints (REST or GraphQL).
   - Request/Response structures.
6. **Implementation Plan**:
   - Step-by-step guide to build the system.
   - logical phases (MVP, V1, V2).

Output the specification in valid Markdown format. Do not add conversational text, just the Markdown content.
"""
            # Generate spec
            response = self.llm.chat(architect_prompt)

            return ToolResult(
                success=True, data={"spec_content": response, "original_prompt": prompt}
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Architect error: {str(e)}")

    def get_parameters(self) -> dict[str, Any]:
        return {"prompt": {"type": "string", "description": "The abstract user request or idea."}}
