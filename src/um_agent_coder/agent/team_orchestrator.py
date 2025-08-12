"""
Agent Team Orchestrator - Manages multi-agent collaboration with cost optimization
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from datetime import datetime
import logging
from .specialized_agents import (
    SpecializedRole, 
    SpecializedAgentFactory,
    DomainExpertise
)

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Define general agent roles"""
    PLANNER = "planner"  # Task decomposition and planning
    RESEARCHER = "researcher"  # Information gathering and analysis
    CODER = "coder"  # Code implementation
    REVIEWER = "reviewer"  # Code review and quality assurance
    OPTIMIZER = "optimizer"  # Performance and cost optimization
    INTEGRATOR = "integrator"  # Integration and deployment
    
    # Specialized roles are handled via SpecializedRole enum


class ModelTier(Enum):
    """Model tiers for cost-aware routing"""
    ECONOMY = "economy"  # Fast, cheap models for simple tasks
    BALANCED = "balanced"  # Mid-tier models for standard tasks
    PREMIUM = "premium"  # High-end models for complex reasoning


@dataclass
class ModelConfig:
    """Configuration for model selection and cost tracking"""
    name: str
    tier: ModelTier
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_context: int
    capabilities: List[str] = field(default_factory=list)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request"""
        return (input_tokens * self.input_cost_per_1k / 1000 + 
                output_tokens * self.output_cost_per_1k / 1000)


# Model configurations with cost data
MODEL_CONFIGS = {
    # Economy tier
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        tier=ModelTier.ECONOMY,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        max_context=16000,
        capabilities=["basic_coding", "simple_analysis"]
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku",
        tier=ModelTier.ECONOMY,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        max_context=200000,
        capabilities=["basic_coding", "fast_processing"]
    ),
    
    # Balanced tier
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        tier=ModelTier.BALANCED,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        max_context=128000,
        capabilities=["complex_coding", "analysis", "reasoning"]
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet",
        tier=ModelTier.BALANCED,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_context=200000,
        capabilities=["complex_coding", "analysis", "vision"]
    ),
    
    # Premium tier
    "gpt-4": ModelConfig(
        name="gpt-4",
        tier=ModelTier.PREMIUM,
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        max_context=128000,
        capabilities=["advanced_reasoning", "complex_coding", "architecture"]
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        tier=ModelTier.PREMIUM,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        max_context=200000,
        capabilities=["advanced_reasoning", "complex_coding", "vision", "architecture"]
    ),
}


@dataclass
class AgentTask:
    """Represents a task assigned to an agent"""
    id: str
    role: AgentRole
    description: str
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    estimated_tokens: int = 0
    max_budget: Optional[float] = None
    result: Optional[Any] = None
    status: str = "pending"
    assigned_model: Optional[str] = None
    actual_cost: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class TeamContext:
    """Shared context for agent team"""
    objective: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    total_budget: float = 10.0  # Default $10 budget
    spent_budget: float = 0.0
    messages: List[Dict[str, Any]] = field(default_factory=list)


class CostOptimizer:
    """Handles cost optimization strategies"""
    
    def __init__(self):
        self.cache = {}  # Simple response cache
        self.model_performance = {}  # Track model performance metrics
        
    def select_optimal_model(
        self, 
        task: AgentTask,
        available_budget: float,
        context: TeamContext
    ) -> str:
        """Select the most cost-effective model for a task"""
        
        # Determine required tier based on task complexity
        required_tier = self._determine_required_tier(task)
        
        # Filter models by tier and capabilities
        suitable_models = [
            (name, config) for name, config in MODEL_CONFIGS.items()
            if config.tier.value >= required_tier.value
            and self._has_required_capabilities(task, config)
        ]
        
        if not suitable_models:
            # Fallback to cheapest available model
            return min(MODEL_CONFIGS.items(), 
                      key=lambda x: x[1].input_cost_per_1k)[0]
        
        # Calculate cost-effectiveness score
        scores = []
        for name, config in suitable_models:
            estimated_cost = config.estimate_cost(
                task.estimated_tokens, 
                task.estimated_tokens // 2  # Rough output estimate
            )
            
            # Skip if over budget
            if task.max_budget and estimated_cost > task.max_budget:
                continue
            if estimated_cost > available_budget:
                continue
                
            # Score based on cost and tier match
            tier_bonus = 1.0 if config.tier == required_tier else 0.8
            cost_score = 1.0 / (estimated_cost + 0.01)  # Lower cost = higher score
            
            # Performance bonus from historical data
            perf_bonus = self.model_performance.get(name, {}).get('success_rate', 0.5)
            
            total_score = cost_score * tier_bonus * (1 + perf_bonus)
            scores.append((name, total_score))
        
        if not scores:
            # If no model fits budget, use cheapest
            return min(suitable_models, 
                      key=lambda x: x[1].estimate_cost(100, 50))[0]
        
        # Return model with highest score
        return max(scores, key=lambda x: x[1])[0]
    
    def _determine_required_tier(self, task: AgentTask) -> ModelTier:
        """Determine minimum required model tier for task"""
        
        # Complex tasks require better models
        if task.role in [AgentRole.PLANNER, AgentRole.REVIEWER]:
            return ModelTier.BALANCED
        
        if task.role == AgentRole.OPTIMIZER:
            return ModelTier.PREMIUM
            
        # High priority tasks get better models
        if task.priority >= 8:
            return ModelTier.PREMIUM
        elif task.priority >= 5:
            return ModelTier.BALANCED
        
        # Default to economy for simple tasks
        return ModelTier.ECONOMY
    
    def _has_required_capabilities(self, task: AgentTask, config: ModelConfig) -> bool:
        """Check if model has required capabilities for task"""
        
        # Map roles to required capabilities
        role_requirements = {
            AgentRole.CODER: ["basic_coding"],
            AgentRole.PLANNER: ["reasoning"],
            AgentRole.RESEARCHER: ["analysis"],
            AgentRole.REVIEWER: ["complex_coding", "analysis"],
            AgentRole.OPTIMIZER: ["advanced_reasoning"],
            AgentRole.INTEGRATOR: ["complex_coding"]
        }
        
        required = role_requirements.get(task.role, [])
        return any(cap in config.capabilities for cap in required)
    
    def check_cache(self, task_hash: str) -> Optional[Any]:
        """Check if we have a cached result for similar task"""
        return self.cache.get(task_hash)
    
    def update_cache(self, task_hash: str, result: Any):
        """Cache task result for reuse"""
        # Simple LRU-style cache with size limit
        if len(self.cache) > 100:
            # Remove oldest entry
            oldest = min(self.cache.items(), key=lambda x: x[1].get('timestamp', 0))
            del self.cache[oldest[0]]
        
        self.cache[task_hash] = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_model_performance(self, model: str, success: bool, latency: float):
        """Track model performance for future selection"""
        if model not in self.model_performance:
            self.model_performance[model] = {
                'success_count': 0,
                'total_count': 0,
                'avg_latency': 0
            }
        
        perf = self.model_performance[model]
        perf['total_count'] += 1
        if success:
            perf['success_count'] += 1
        
        # Update rolling average latency
        perf['avg_latency'] = (perf['avg_latency'] * (perf['total_count'] - 1) + latency) / perf['total_count']
        perf['success_rate'] = perf['success_count'] / perf['total_count']


class TeamOrchestrator:
    """Orchestrates multi-agent collaboration with cost optimization"""
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.optimizer = CostOptimizer()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.specialized_agents = {}  # Cache for specialized agents
        
    async def execute_team_task(
        self,
        objective: str,
        budget: float = 10.0,
        constraints: Optional[Dict[str, Any]] = None,
        use_specialized_agents: bool = True
    ) -> Dict[str, Any]:
        """Execute a complex task using agent team
        
        Args:
            objective: Task objective
            budget: Maximum budget for execution
            constraints: Additional constraints
            use_specialized_agents: Whether to use domain-specific specialized agents
        """
        
        # Initialize team context
        context = TeamContext(
            objective=objective,
            total_budget=budget,
            constraints=constraints or {}
        )
        
        # Detect if specialized agents are needed
        if use_specialized_agents:
            specialized_agent = SpecializedAgentFactory.recommend_agent(objective, constraints or {})
            if specialized_agent:
                return await self._execute_specialized_task(specialized_agent, objective, context)
        
        # Phase 1: Planning
        plan = await self._planning_phase(context)
        
        # Phase 2: Task decomposition
        tasks = self._decompose_into_tasks(plan, context)
        
        # Phase 3: Parallel execution with dependencies
        results = await self._execute_tasks(tasks, context)
        
        # Phase 4: Integration and review
        final_result = await self._integration_phase(results, context)
        
        return {
            'result': final_result,
            'total_cost': context.spent_budget,
            'budget_remaining': context.total_budget - context.spent_budget,
            'tasks_completed': len(self.completed_tasks),
            'performance_metrics': self._get_performance_metrics(),
            'agents_used': self._get_agents_summary()
        }
    
    async def _planning_phase(self, context: TeamContext) -> Dict[str, Any]:
        """Initial planning phase"""
        
        planning_task = AgentTask(
            id="plan_001",
            role=AgentRole.PLANNER,
            description=f"Create execution plan for: {context.objective}",
            priority=10,
            estimated_tokens=500
        )
        
        # Select optimal model for planning
        model = self.optimizer.select_optimal_model(
            planning_task,
            context.total_budget - context.spent_budget,
            context
        )
        planning_task.assigned_model = model
        
        # Execute planning (mock for now)
        plan = await self._execute_single_task(planning_task, context)
        
        return plan
    
    def _decompose_into_tasks(
        self, 
        plan: Dict[str, Any], 
        context: TeamContext
    ) -> List[AgentTask]:
        """Decompose plan into executable tasks"""
        
        tasks = []
        
        # Create research tasks
        tasks.append(AgentTask(
            id="research_001",
            role=AgentRole.RESEARCHER,
            description="Research best practices and existing solutions",
            priority=7,
            estimated_tokens=300
        ))
        
        # Create coding tasks
        tasks.append(AgentTask(
            id="code_001",
            role=AgentRole.CODER,
            description="Implement core functionality",
            dependencies=["research_001"],
            priority=8,
            estimated_tokens=1000
        ))
        
        # Create review task
        tasks.append(AgentTask(
            id="review_001",
            role=AgentRole.REVIEWER,
            description="Review implementation for quality and correctness",
            dependencies=["code_001"],
            priority=6,
            estimated_tokens=400
        ))
        
        # Create optimization task
        tasks.append(AgentTask(
            id="optimize_001",
            role=AgentRole.OPTIMIZER,
            description="Optimize for performance and cost",
            dependencies=["review_001"],
            priority=5,
            estimated_tokens=300
        ))
        
        return tasks
    
    async def _execute_tasks(
        self, 
        tasks: List[AgentTask], 
        context: TeamContext
    ) -> Dict[str, Any]:
        """Execute tasks with dependency management"""
        
        results = {}
        pending_tasks = {task.id: task for task in tasks}
        
        while pending_tasks:
            # Find tasks ready to execute (dependencies met)
            ready_tasks = [
                task for task in pending_tasks.values()
                if all(dep in results for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                # Check for circular dependencies
                logger.warning("No tasks ready - possible circular dependency")
                break
            
            # Execute ready tasks in parallel
            task_futures = []
            for task in ready_tasks:
                # Check budget
                if context.spent_budget >= context.total_budget:
                    logger.warning(f"Budget exhausted, skipping task {task.id}")
                    continue
                
                # Select optimal model
                available_budget = context.total_budget - context.spent_budget
                model = self.optimizer.select_optimal_model(task, available_budget, context)
                task.assigned_model = model
                
                # Execute task
                task_futures.append(self._execute_single_task(task, context))
                del pending_tasks[task.id]
            
            # Wait for tasks to complete
            if task_futures:
                task_results = await asyncio.gather(*task_futures)
                for task, result in zip(ready_tasks, task_results):
                    results[task.id] = result
                    self.completed_tasks[task.id] = task
        
        return results
    
    async def _execute_single_task(
        self, 
        task: AgentTask, 
        context: TeamContext
    ) -> Any:
        """Execute a single agent task"""
        
        task.start_time = datetime.now()
        task.status = "running"
        
        # Check cache first
        task_hash = self._hash_task(task)
        cached_result = self.optimizer.check_cache(task_hash)
        if cached_result:
            logger.info(f"Using cached result for task {task.id}")
            task.result = cached_result['result']
            task.status = "completed"
            task.actual_cost = 0  # No cost for cached results
            return cached_result['result']
        
        try:
            # Simulate task execution (replace with actual LLM call)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mock result
            result = {
                'task_id': task.id,
                'output': f"Completed {task.description}",
                'tokens_used': task.estimated_tokens
            }
            
            # Calculate actual cost
            if task.assigned_model:
                config = MODEL_CONFIGS[task.assigned_model]
                task.actual_cost = config.estimate_cost(
                    task.estimated_tokens,
                    task.estimated_tokens // 2
                )
                context.spent_budget += task.actual_cost
            
            # Update cache
            self.optimizer.update_cache(task_hash, result)
            
            # Update performance metrics
            task.end_time = datetime.now()
            latency = (task.end_time - task.start_time).total_seconds()
            self.optimizer.update_model_performance(
                task.assigned_model, 
                True, 
                latency
            )
            
            task.result = result
            task.status = "completed"
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = "failed"
            
            # Update performance metrics
            if task.assigned_model:
                self.optimizer.update_model_performance(
                    task.assigned_model, 
                    False, 
                    0
                )
            
            raise
    
    async def _integration_phase(
        self, 
        results: Dict[str, Any], 
        context: TeamContext
    ) -> Any:
        """Integrate results from all agents"""
        
        integration_task = AgentTask(
            id="integrate_001",
            role=AgentRole.INTEGRATOR,
            description="Integrate and finalize all results",
            priority=9,
            estimated_tokens=200
        )
        
        # Use remaining budget
        available_budget = context.total_budget - context.spent_budget
        model = self.optimizer.select_optimal_model(
            integration_task,
            available_budget,
            context
        )
        integration_task.assigned_model = model
        
        # Execute integration
        final_result = await self._execute_single_task(integration_task, context)
        
        return final_result
    
    def _hash_task(self, task: AgentTask) -> str:
        """Generate hash for task caching"""
        import hashlib
        
        task_str = f"{task.role.value}:{task.description}:{task.dependencies}"
        return hashlib.md5(task_str.encode()).hexdigest()
    
    async def _execute_specialized_task(
        self, 
        agent, 
        objective: str, 
        context: TeamContext
    ) -> Dict[str, Any]:
        """Execute task using specialized agent"""
        
        # Cache the agent
        self.specialized_agents[agent.role.value] = agent
        
        # Prepare specialized prompt
        prompt = agent.prepare_prompt(objective, context.constraints)
        
        # Create task for tracking
        task = AgentTask(
            id=f"specialized_{agent.role.value}",
            role=AgentRole.CODER,  # Map to general role
            description=objective,
            priority=8,
            estimated_tokens=1000
        )
        
        # Select optimal model from agent's preferences
        if agent.capabilities.preferred_models:
            model = agent.capabilities.preferred_models[0]
        else:
            model = self.optimizer.select_optimal_model(
                task,
                context.total_budget - context.spent_budget,
                context
            )
        
        task.assigned_model = model
        
        # Execute task
        result = await self._execute_single_task(task, context)
        
        # Validate output
        is_valid, error_msg = agent.validate_output(str(result))
        
        return {
            'result': result,
            'total_cost': context.spent_budget,
            'budget_remaining': context.total_budget - context.spent_budget,
            'specialized_agent': agent.role.value,
            'agent_domain': agent.capabilities.domain.value,
            'validation': {'is_valid': is_valid, 'error': error_msg},
            'performance_metrics': self._get_performance_metrics(),
            'agents_used': self._get_agents_summary()
        }
    
    def _get_agents_summary(self) -> Dict[str, Any]:
        """Get summary of agents used"""
        
        summary = {
            'general_agents': [],
            'specialized_agents': []
        }
        
        # Add general agents from completed tasks
        for task in self.completed_tasks.values():
            if task.role in AgentRole:
                summary['general_agents'].append(task.role.value)
        
        # Add specialized agents
        for agent_name in self.specialized_agents.keys():
            summary['specialized_agents'].append(agent_name)
        
        return summary
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the team execution"""
        
        total_tasks = len(self.completed_tasks)
        if total_tasks == 0:
            return {}
        
        # Calculate metrics
        avg_cost_per_task = sum(
            task.actual_cost for task in self.completed_tasks.values()
        ) / total_tasks
        
        model_usage = {}
        for task in self.completed_tasks.values():
            if task.assigned_model:
                model_usage[task.assigned_model] = model_usage.get(task.assigned_model, 0) + 1
        
        return {
            'total_tasks': total_tasks,
            'avg_cost_per_task': avg_cost_per_task,
            'model_usage': model_usage,
            'cache_hits': len([t for t in self.completed_tasks.values() if t.actual_cost == 0]),
            'model_performance': self.optimizer.model_performance,
            'specialized_agents_used': len(self.specialized_agents)
        }


# Example usage
async def main():
    """Example of using the team orchestrator"""
    
    orchestrator = TeamOrchestrator()
    
    result = await orchestrator.execute_team_task(
        objective="Implement a user authentication system with JWT tokens",
        budget=5.0,  # $5 budget
        constraints={
            'language': 'python',
            'framework': 'fastapi',
            'include_tests': True
        }
    )
    
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())