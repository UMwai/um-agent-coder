from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ModelCategory(Enum):
    HIGH_PERFORMANCE = "high_performance"
    EFFICIENT = "efficient"
    OPEN_SOURCE = "open_source"


@dataclass
class ModelInfo:
    name: str
    provider: str
    category: ModelCategory
    context_window: int
    cost_per_1k_input: float  # in USD
    cost_per_1k_output: float  # in USD
    capabilities: List[str]
    performance_score: float  # 0-100 based on benchmarks
    description: str


class ModelRegistry:
    """Registry of available AI models with their capabilities and costs."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        # High Performance Models
        self.register(ModelInfo(
            name="claude-3.5-sonnet",
            provider="anthropic",
            category=ModelCategory.HIGH_PERFORMANCE,
            context_window=200000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            capabilities=["coding", "reasoning", "extended_thinking", "tool_use"],
            performance_score=94,
            description="Claude 3.5 Sonnet - Best for coding with 93.7% on benchmarks"
        ))
        
        self.register(ModelInfo(
            name="claude-3-opus",
            provider="anthropic",
            category=ModelCategory.HIGH_PERFORMANCE,
            context_window=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            capabilities=["coding", "reasoning", "tool_use"],
            performance_score=92,
            description="Claude 3 Opus - Powerful but more expensive"
        ))
        
        self.register(ModelInfo(
            name="gpt-4o",
            provider="openai",
            category=ModelCategory.HIGH_PERFORMANCE,
            context_window=128000,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            capabilities=["coding", "reasoning", "multimodal", "tool_use"],
            performance_score=88,
            description="GPT-4o - Fast multimodal model with good coding abilities"
        ))
        
        self.register(ModelInfo(
            name="gpt-4-turbo",
            provider="openai",
            category=ModelCategory.HIGH_PERFORMANCE,
            context_window=128000,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            capabilities=["coding", "reasoning", "vision", "tool_use"],
            performance_score=90,
            description="GPT-4 Turbo - Reliable for complex tasks"
        ))
        
        self.register(ModelInfo(
            name="gemini-2.0-flash",
            provider="google",
            category=ModelCategory.HIGH_PERFORMANCE,
            context_window=1000000,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            capabilities=["coding", "reasoning", "multimodal", "large_context", "voice", "video"],
            performance_score=89,
            description="Gemini 2.0 Flash - Ultra cost-effective with 1M context, voice/video support"
        ))
        
        # Efficient Models
        self.register(ModelInfo(
            name="claude-3-haiku",
            provider="anthropic",
            category=ModelCategory.EFFICIENT,
            context_window=200000,
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            capabilities=["coding", "reasoning", "fast_response"],
            performance_score=78,
            description="Claude 3 Haiku - Ultra-fast and cost-effective"
        ))
        
        self.register(ModelInfo(
            name="gpt-4o-mini",
            provider="openai",
            category=ModelCategory.EFFICIENT,
            context_window=128000,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            capabilities=["coding", "reasoning", "fast_response"],
            performance_score=75,
            description="GPT-4o Mini - Affordable with decent performance"
        ))
        
        self.register(ModelInfo(
            name="gemini-1.5-flash",
            provider="google",
            category=ModelCategory.EFFICIENT,
            context_window=1000000,
            cost_per_1k_input=0.0000375,
            cost_per_1k_output=0.00015,
            capabilities=["coding", "reasoning", "multimodal", "large_context"],
            performance_score=80,
            description="Gemini 1.5 Flash - Cheapest option at $0.0375 per 1M input tokens"
        ))
        
        # Open Source Models
        self.register(ModelInfo(
            name="deepseek-v3",
            provider="deepseek",
            category=ModelCategory.OPEN_SOURCE,
            context_window=128000,
            cost_per_1k_input=0.0001,  # API pricing
            cost_per_1k_output=0.0002,
            capabilities=["coding", "reasoning", "math"],
            performance_score=92,
            description="DeepSeek-V3 - Best open-source model, rivals GPT-4o"
        ))
        
        self.register(ModelInfo(
            name="deepseek-coder-33b",
            provider="deepseek",
            category=ModelCategory.OPEN_SOURCE,
            context_window=32000,
            cost_per_1k_input=0.0,  # Self-hosted
            cost_per_1k_output=0.0,
            capabilities=["coding", "code_completion"],
            performance_score=85,
            description="DeepSeek Coder - Specialized for code generation"
        ))
        
        self.register(ModelInfo(
            name="qwen-3-235b",
            provider="alibaba",
            category=ModelCategory.OPEN_SOURCE,
            context_window=32000,
            cost_per_1k_input=0.0,  # Self-hosted
            cost_per_1k_output=0.0,
            capabilities=["coding", "reasoning", "multilingual"],
            performance_score=93,
            description="Qwen 3 235B - Top open-source for coding, 70.7 on LiveCodeBench"
        ))
        
        self.register(ModelInfo(
            name="qwen-2.5-coder-32b",
            provider="alibaba",
            category=ModelCategory.OPEN_SOURCE,
            context_window=32000,
            cost_per_1k_input=0.0,  # Self-hosted
            cost_per_1k_output=0.0,
            capabilities=["coding", "code_completion", "debugging"],
            performance_score=88,
            description="Qwen 2.5 Coder - Excellent for code tasks"
        ))
        
        self.register(ModelInfo(
            name="llama-3.3-70b",
            provider="meta",
            category=ModelCategory.OPEN_SOURCE,
            context_window=128000,
            cost_per_1k_input=0.0,  # Self-hosted
            cost_per_1k_output=0.0,
            capabilities=["coding", "reasoning", "general"],
            performance_score=82,
            description="Llama 3.3 70B - Solid open-source choice"
        ))
        
        self.register(ModelInfo(
            name="codellama-70b",
            provider="meta",
            category=ModelCategory.OPEN_SOURCE,
            context_window=100000,
            cost_per_1k_input=0.0,  # Self-hosted
            cost_per_1k_output=0.0,
            capabilities=["coding", "code_completion", "infilling"],
            performance_score=80,
            description="Code Llama 70B - Specialized for coding tasks"
        ))
        
        self.register(ModelInfo(
            name="mistral-large-2",
            provider="mistral",
            category=ModelCategory.OPEN_SOURCE,
            context_window=128000,
            cost_per_1k_input=0.002,  # API pricing
            cost_per_1k_output=0.006,
            capabilities=["coding", "reasoning", "function_calling"],
            performance_score=86,
            description="Mistral Large 2 - Strong European alternative"
        ))
    
    def register(self, model: ModelInfo):
        """Register a new model."""
        self.models[model.name] = model
    
    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self.models.get(name)
    
    def get_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get all models in a category."""
        return [m for m in self.models.values() if m.category == category]
    
    def get_best_for_task(self, task: str, max_cost: Optional[float] = None) -> Optional[ModelInfo]:
        """Get the best model for a specific task within cost constraints."""
        suitable_models = [
            m for m in self.models.values() 
            if task in m.capabilities
        ]
        
        if max_cost:
            suitable_models = [
                m for m in suitable_models
                if m.cost_per_1k_input <= max_cost
            ]
        
        if not suitable_models:
            return None
        
        # Sort by performance score
        return max(suitable_models, key=lambda m: m.performance_score)
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a specific model usage."""
        model = self.get(model_name)
        if not model:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output
        return input_cost + output_cost