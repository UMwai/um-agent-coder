"""
Adaptive Model Router - Intelligent routing based on task complexity and cost
"""

import json
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class TaskComplexity(Enum):
    """Task complexity levels"""

    TRIVIAL = 1  # Simple responses, basic queries
    SIMPLE = 2  # Basic coding, straightforward tasks
    MODERATE = 3  # Standard development tasks
    COMPLEX = 4  # Complex algorithms, architecture
    EXPERT = 5  # Advanced reasoning, system design


@dataclass
class TaskSignature:
    """Signature for identifying task characteristics"""

    keywords: list[str]
    patterns: list[str]  # Regex patterns
    min_length: int
    max_length: int
    requires_code: bool
    requires_reasoning: bool
    estimated_complexity: TaskComplexity


class ComplexityAnalyzer:
    """Analyze task complexity from prompts"""

    def __init__(self):
        self.signatures = self._init_signatures()
        self.complexity_history = deque(maxlen=100)

    def _init_signatures(self) -> list[TaskSignature]:
        """Initialize task signatures for complexity detection"""

        return [
            # Trivial tasks
            TaskSignature(
                keywords=["hello", "hi", "thanks", "yes", "no", "ok"],
                patterns=[r"^.{1,20}$", r"^\w+\?$"],
                min_length=1,
                max_length=50,
                requires_code=False,
                requires_reasoning=False,
                estimated_complexity=TaskComplexity.TRIVIAL,
            ),
            # Simple tasks
            TaskSignature(
                keywords=["explain", "what is", "how to", "define", "list"],
                patterns=[r"what is \w+", r"explain \w+", r"how to \w+"],
                min_length=10,
                max_length=200,
                requires_code=False,
                requires_reasoning=False,
                estimated_complexity=TaskComplexity.SIMPLE,
            ),
            # Moderate tasks
            TaskSignature(
                keywords=["implement", "create", "function", "class", "fix", "debug"],
                patterns=[r"implement.*function", r"create.*class", r"fix.*bug"],
                min_length=20,
                max_length=500,
                requires_code=True,
                requires_reasoning=False,
                estimated_complexity=TaskComplexity.MODERATE,
            ),
            # Complex tasks
            TaskSignature(
                keywords=["optimize", "refactor", "architecture", "design pattern", "algorithm"],
                patterns=[r"optimize.*performance", r"refactor.*code", r"design.*system"],
                min_length=50,
                max_length=1000,
                requires_code=True,
                requires_reasoning=True,
                estimated_complexity=TaskComplexity.COMPLEX,
            ),
            # Expert tasks
            TaskSignature(
                keywords=[
                    "distributed",
                    "microservices",
                    "machine learning",
                    "blockchain",
                    "quantum",
                    "concurrent",
                    "parallel",
                    "scalable",
                ],
                patterns=[r"design.*distributed.*system", r"implement.*ml.*pipeline"],
                min_length=100,
                max_length=5000,
                requires_code=True,
                requires_reasoning=True,
                estimated_complexity=TaskComplexity.EXPERT,
            ),
        ]

    def analyze_prompt(self, prompt: str) -> TaskComplexity:
        """Analyze prompt to determine task complexity"""

        prompt_lower = prompt.lower()
        prompt_length = len(prompt)

        # Score each complexity level
        scores = dict.fromkeys(TaskComplexity, 0.0)

        for signature in self.signatures:
            score = 0.0

            # Check keyword matches
            keyword_matches = sum(1 for kw in signature.keywords if kw in prompt_lower)
            if keyword_matches > 0:
                score += keyword_matches * 2

            # Check pattern matches
            pattern_matches = sum(
                1 for pattern in signature.patterns if re.search(pattern, prompt_lower)
            )
            if pattern_matches > 0:
                score += pattern_matches * 3

            # Check length constraints
            if signature.min_length <= prompt_length <= signature.max_length:
                score += 1

            # Check code indicators
            if signature.requires_code:
                code_indicators = ["```", "def ", "class ", "function", "implement"]
                if any(ind in prompt_lower for ind in code_indicators):
                    score += 2

            # Check reasoning indicators
            if signature.requires_reasoning:
                reasoning_indicators = ["why", "how", "explain", "analyze", "compare"]
                if any(ind in prompt_lower for ind in reasoning_indicators):
                    score += 2

            scores[signature.estimated_complexity] += score

        # Additional heuristics

        # Long prompts tend to be more complex
        if prompt_length > 1000:
            scores[TaskComplexity.COMPLEX] += 2
            scores[TaskComplexity.EXPERT] += 1

        # Multiple questions or requirements increase complexity
        question_marks = prompt.count("?")
        if question_marks > 2:
            scores[TaskComplexity.COMPLEX] += question_marks - 2

        # Code blocks increase complexity
        code_blocks = prompt.count("```")
        if code_blocks > 0:
            scores[TaskComplexity.MODERATE] += code_blocks
            scores[TaskComplexity.COMPLEX] += code_blocks / 2

        # Select complexity with highest score
        complexity = max(scores.items(), key=lambda x: x[1])[0]

        # Store in history for adaptive learning
        self.complexity_history.append(
            {"prompt_length": prompt_length, "complexity": complexity, "scores": scores}
        )

        return complexity

    def get_complexity_distribution(self) -> dict[TaskComplexity, float]:
        """Get distribution of recent task complexities"""

        if not self.complexity_history:
            return dict.fromkeys(TaskComplexity, 0.0)

        distribution = dict.fromkeys(TaskComplexity, 0)
        for entry in self.complexity_history:
            distribution[entry["complexity"]] += 1

        total = len(self.complexity_history)
        return {c: count / total for c, count in distribution.items()}


class AdaptiveRouter:
    """Route requests to optimal models based on task analysis"""

    def __init__(self, cost_tracker=None):
        self.analyzer = ComplexityAnalyzer()
        self.cost_tracker = cost_tracker
        self.routing_history = deque(maxlen=1000)
        self.model_performance = {}

        # Model capability matrix
        self.model_capabilities = {
            # Economy tier
            "gpt-3.5-turbo": {
                "max_complexity": TaskComplexity.MODERATE,
                "strengths": ["speed", "basic_coding"],
                "cost_multiplier": 1.0,
            },
            "claude-3-haiku": {
                "max_complexity": TaskComplexity.MODERATE,
                "strengths": ["speed", "long_context"],
                "cost_multiplier": 0.8,
            },
            # Balanced tier
            "gpt-4-turbo": {
                "max_complexity": TaskComplexity.COMPLEX,
                "strengths": ["coding", "reasoning"],
                "cost_multiplier": 20.0,
            },
            "claude-3-sonnet": {
                "max_complexity": TaskComplexity.COMPLEX,
                "strengths": ["coding", "analysis", "long_context"],
                "cost_multiplier": 6.0,
            },
            # Premium tier
            "gpt-4": {
                "max_complexity": TaskComplexity.EXPERT,
                "strengths": ["advanced_reasoning", "architecture"],
                "cost_multiplier": 60.0,
            },
            "claude-3-opus": {
                "max_complexity": TaskComplexity.EXPERT,
                "strengths": ["advanced_reasoning", "complex_coding"],
                "cost_multiplier": 30.0,
            },
        }

    def route_request(
        self,
        prompt: str,
        user_preferences: Optional[dict[str, Any]] = None,
        budget_remaining: Optional[float] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Route request to optimal model"""

        # Analyze task complexity
        complexity = self.analyzer.analyze_prompt(prompt)

        # Get user preferences
        preferences = user_preferences or {}
        prefer_speed = preferences.get("prefer_speed", False)
        prefer_quality = preferences.get("prefer_quality", False)
        required_capabilities = preferences.get("required_capabilities", [])

        # Filter eligible models
        eligible_models = []
        for model, caps in self.model_capabilities.items():
            # Check complexity compatibility
            if caps["max_complexity"].value < complexity.value:
                continue

            # Check required capabilities
            if required_capabilities:
                if not all(cap in caps["strengths"] for cap in required_capabilities):
                    continue

            eligible_models.append(model)

        if not eligible_models:
            # Fallback to most capable model
            eligible_models = ["claude-3-opus", "gpt-4"]

        # Score models
        model_scores = {}
        for model in eligible_models:
            score = self._score_model(
                model, complexity, prefer_speed, prefer_quality, budget_remaining
            )
            model_scores[model] = score

        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]

        # Record routing decision
        routing_decision = {
            "timestamp": json.dumps({"time": "now"}),  # Simplified for example
            "prompt_length": len(prompt),
            "complexity": complexity.name,
            "selected_model": best_model,
            "score": model_scores[best_model],
            "all_scores": model_scores,
        }

        self.routing_history.append(routing_decision)

        # Return model and metadata
        metadata = {
            "complexity": complexity.name,
            "routing_reason": self._get_routing_reason(best_model, complexity),
            "estimated_cost_multiplier": self.model_capabilities[best_model]["cost_multiplier"],
            "alternatives": [m for m in model_scores if m != best_model],
        }

        return best_model, metadata

    def _score_model(
        self,
        model: str,
        complexity: TaskComplexity,
        prefer_speed: bool,
        prefer_quality: bool,
        budget_remaining: Optional[float],
    ) -> float:
        """Score a model for the given task"""

        caps = self.model_capabilities[model]
        score = 100.0

        # Complexity match score
        complexity_diff = caps["max_complexity"].value - complexity.value
        if complexity_diff == 0:
            score += 20  # Perfect match
        elif complexity_diff == 1:
            score += 10  # Slight overkill is ok
        elif complexity_diff > 1:
            score -= complexity_diff * 5  # Penalize overkill

        # Cost efficiency score
        cost_mult = caps["cost_multiplier"]
        if budget_remaining is not None:
            if cost_mult > budget_remaining * 100:
                score -= 50  # Too expensive
            else:
                # Prefer cheaper models when possible
                score += 20 / (cost_mult + 1)

        # Speed preference
        if prefer_speed:
            if "speed" in caps["strengths"]:
                score += 15
            if cost_mult < 5:  # Cheaper models are usually faster
                score += 10

        # Quality preference
        if prefer_quality:
            if caps["max_complexity"].value >= TaskComplexity.COMPLEX.value:
                score += 20
            if "advanced_reasoning" in caps["strengths"]:
                score += 15

        # Historical performance bonus
        if model in self.model_performance:
            perf = self.model_performance[model]
            score += perf.get("success_rate", 0.5) * 10

        return score

    def _get_routing_reason(self, model: str, complexity: TaskComplexity) -> str:
        """Generate explanation for routing decision"""

        reasons = []
        caps = self.model_capabilities[model]

        if complexity.value <= 2:
            reasons.append("Simple task routed to economy model")
        elif complexity.value >= 4:
            reasons.append("Complex task requires advanced model")

        if caps["cost_multiplier"] < 5:
            reasons.append("Cost-effective choice")

        if "speed" in caps["strengths"]:
            reasons.append("Optimized for quick response")

        if "advanced_reasoning" in caps["strengths"]:
            reasons.append("Selected for reasoning capabilities")

        return "; ".join(reasons) if reasons else "Standard routing"

    def update_performance(
        self,
        model: str,
        success: bool,
        response_time: float,
        user_satisfaction: Optional[float] = None,
    ):
        """Update model performance metrics"""

        if model not in self.model_performance:
            self.model_performance[model] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_response_time": 0,
                "satisfaction_scores": [],
            }

        perf = self.model_performance[model]
        perf["total_requests"] += 1

        if success:
            perf["successful_requests"] += 1

        # Update rolling average response time
        perf["avg_response_time"] = (
            perf["avg_response_time"] * (perf["total_requests"] - 1) + response_time
        ) / perf["total_requests"]

        # Calculate success rate
        perf["success_rate"] = perf["successful_requests"] / perf["total_requests"]

        # Track satisfaction if provided
        if user_satisfaction is not None:
            perf["satisfaction_scores"].append(user_satisfaction)
            if len(perf["satisfaction_scores"]) > 100:
                perf["satisfaction_scores"].pop(0)
            perf["avg_satisfaction"] = np.mean(perf["satisfaction_scores"])

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get routing statistics and insights"""

        if not self.routing_history:
            return {}

        # Model usage distribution
        model_usage = {}
        for decision in self.routing_history:
            model = decision["selected_model"]
            model_usage[model] = model_usage.get(model, 0) + 1

        # Complexity distribution
        complexity_dist = self.analyzer.get_complexity_distribution()

        # Average scores by model
        model_avg_scores = {}
        for decision in self.routing_history:
            for model, score in decision["all_scores"].items():
                if model not in model_avg_scores:
                    model_avg_scores[model] = []
                model_avg_scores[model].append(score)

        for model in model_avg_scores:
            model_avg_scores[model] = np.mean(model_avg_scores[model])

        return {
            "total_requests": len(self.routing_history),
            "model_usage": model_usage,
            "complexity_distribution": {k.name: v for k, v in complexity_dist.items()},
            "model_performance": self.model_performance,
            "average_routing_scores": model_avg_scores,
        }


# Example usage
def example_usage():
    """Example of using the adaptive router"""

    router = AdaptiveRouter()

    # Simple query
    model, metadata = router.route_request(
        "What is Python?", user_preferences={"prefer_speed": True}
    )
    print(f"Simple query routed to: {model}")
    print(f"Reason: {metadata['routing_reason']}\n")

    # Complex task
    model, metadata = router.route_request(
        """
        Design and implement a distributed caching system with the following requirements:
        1. Support for multiple cache nodes with consistent hashing
        2. Implement cache invalidation strategies
        3. Handle node failures gracefully
        4. Provide monitoring and metrics
        5. Support both LRU and LFU eviction policies
        """,
        user_preferences={"prefer_quality": True},
    )
    print(f"Complex task routed to: {model}")
    print(f"Reason: {metadata['routing_reason']}\n")

    # Update performance
    router.update_performance(model, success=True, response_time=2.5, user_satisfaction=0.9)

    # Get statistics
    stats = router.get_routing_statistics()
    print("Routing Statistics:")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    example_usage()
