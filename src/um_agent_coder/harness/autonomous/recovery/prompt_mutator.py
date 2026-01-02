"""Prompt mutation strategies for stuck recovery.

When the loop is stuck, prompt mutation attempts to rephrase or
restructure the goal to help the model approach it differently.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MutationType(Enum):
    """Types of prompt mutations."""

    REPHRASE = "rephrase"  # Reword the goal differently
    DECOMPOSE = "decompose"  # Break into smaller steps
    CONSTRAIN = "constrain"  # Add specific constraints
    EXAMPLES = "examples"  # Add concrete examples
    NEGATIVE = "negative"  # Specify what NOT to do


@dataclass
class MutationResult:
    """Result of a prompt mutation."""

    mutation_type: MutationType
    original_prompt: str
    mutated_prompt: str
    success: bool = True
    error: Optional[str] = None


# Mutation templates
MUTATION_TEMPLATES = {
    MutationType.REPHRASE: """
Reword this task using different phrasing while keeping the same goal:

Original: {prompt}

Reworded version (same goal, different words):
""",
    MutationType.DECOMPOSE: """
Break this task into smaller, sequential steps:

Task: {prompt}

Step-by-step breakdown:
1. First,
2. Then,
3. Next,
4. Finally,
""",
    MutationType.CONSTRAIN: """
Add specific constraints to make this task more focused:

Task: {prompt}

Constrained version with clear boundaries:
- Focus specifically on:
- Do not attempt:
- Success criteria:
""",
    MutationType.EXAMPLES: """
Add concrete examples to clarify what success looks like:

Task: {prompt}

With examples:
Task: [same as above]
Example 1: If the input is X, the output should be Y
Example 2: For case Z, the expected behavior is W
""",
    MutationType.NEGATIVE: """
Clarify what NOT to do to avoid common mistakes:

Task: {prompt}

Clarified with negative constraints:
DO: [the task]
DO NOT:
- Do not over-engineer
- Do not change unrelated code
- Do not skip tests
""",
}


class PromptMutator:
    """Mutate prompts to help escape stuck states.

    Uses templates to transform the original prompt into variants
    that may help the model approach the problem differently.
    """

    def __init__(self, llm=None):
        """Initialize prompt mutator.

        Args:
            llm: Optional LLM instance for advanced mutations.
                 If None, uses template-based mutations only.
        """
        self.llm = llm

    def mutate(
        self,
        prompt: str,
        mutation_type: MutationType,
        context: Optional[str] = None,
    ) -> MutationResult:
        """Apply a mutation to a prompt.

        Args:
            prompt: The original prompt to mutate.
            mutation_type: Type of mutation to apply.
            context: Optional context from previous iterations.

        Returns:
            MutationResult with the mutated prompt.
        """
        try:
            if self.llm is not None:
                # Use LLM for intelligent mutation
                mutated = self._llm_mutate(prompt, mutation_type, context)
            else:
                # Use template-based mutation
                mutated = self._template_mutate(prompt, mutation_type)

            return MutationResult(
                mutation_type=mutation_type,
                original_prompt=prompt,
                mutated_prompt=mutated,
                success=True,
            )
        except Exception as e:
            return MutationResult(
                mutation_type=mutation_type,
                original_prompt=prompt,
                mutated_prompt=prompt,  # Return original on failure
                success=False,
                error=str(e),
            )

    def _template_mutate(self, prompt: str, mutation_type: MutationType) -> str:
        """Apply template-based mutation.

        Args:
            prompt: The original prompt.
            mutation_type: Type of mutation.

        Returns:
            Mutated prompt using template.
        """
        template = MUTATION_TEMPLATES.get(mutation_type)
        if not template:
            return prompt

        # Apply template with prompt
        mutated = template.format(prompt=prompt)

        # For decompose, add the original goal at the end
        if mutation_type == MutationType.DECOMPOSE:
            mutated = f"Goal: {prompt}\n\n{mutated}"

        return mutated.strip()

    def _llm_mutate(
        self,
        prompt: str,
        mutation_type: MutationType,
        context: Optional[str] = None,
    ) -> str:
        """Apply LLM-based mutation.

        Args:
            prompt: The original prompt.
            mutation_type: Type of mutation.
            context: Optional context from previous iterations.

        Returns:
            Mutated prompt from LLM.
        """
        system_prompt = f"""You are a prompt engineer helping to improve task descriptions.
Apply the '{mutation_type.value}' transformation to the given task.
Your goal is to rephrase the task so an AI assistant can approach it more effectively.
Return ONLY the improved task description, nothing else."""

        user_prompt = f"Original task: {prompt}"
        if context:
            user_prompt += f"\n\nContext from previous attempts:\n{context}"

        # Call LLM (simplified - actual implementation would use proper LLM interface)
        response = self.llm.generate(system_prompt, user_prompt)
        return response.strip()

    def get_mutation_sequence(self) -> list[MutationType]:
        """Get recommended sequence of mutations to try.

        Returns:
            List of mutation types in recommended order.
        """
        return [
            MutationType.REPHRASE,
            MutationType.DECOMPOSE,
            MutationType.CONSTRAIN,
            MutationType.EXAMPLES,
            MutationType.NEGATIVE,
        ]

    def mutate_with_fallback(
        self,
        prompt: str,
        tried_mutations: Optional[list[MutationType]] = None,
        context: Optional[str] = None,
    ) -> Optional[MutationResult]:
        """Try mutations in sequence, skipping already tried ones.

        Args:
            prompt: The original prompt.
            tried_mutations: List of already tried mutation types.
            context: Optional context from previous iterations.

        Returns:
            MutationResult or None if all mutations exhausted.
        """
        tried = set(tried_mutations or [])
        sequence = self.get_mutation_sequence()

        for mutation_type in sequence:
            if mutation_type not in tried:
                result = self.mutate(prompt, mutation_type, context)
                if result.success:
                    return result

        return None
