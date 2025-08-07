"""
GPT-5 Reasoning Agent with advanced capabilities for complex problem-solving.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from um_agent_coder.agent.base_agent import BaseAgent
from um_agent_coder.llm.providers.openai import OpenAILLM
from um_agent_coder.models import ModelRegistry


class ReasoningMode(Enum):
    """Different reasoning modes for the agent."""
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    REFLEXION = "reflexion"
    DEBATE = "debate"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process."""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.0
    alternatives: List[str] = None


class GPT5ReasoningAgent(BaseAgent):
    """
    Advanced reasoning agent powered by GPT-5 with multiple reasoning strategies.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-5", 
                 reasoning_mode: ReasoningMode = ReasoningMode.CHAIN_OF_THOUGHT,
                 temperature: float = 0.7, max_tokens: int = 8192):
        """
        Initialize the GPT-5 reasoning agent.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5, gpt-5-mini, gpt-5-nano)
            reasoning_mode: The reasoning strategy to employ
            temperature: Model temperature for generation
            max_tokens: Maximum tokens for response
        """
        self.llm = OpenAILLM(api_key=api_key, model=model, 
                            temperature=temperature, max_tokens=max_tokens)
        self.reasoning_mode = reasoning_mode
        self.reasoning_history: List[ReasoningStep] = []
        self.model_registry = ModelRegistry()
        
    def reason(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ReasoningStep]]:
        """
        Execute reasoning based on the selected mode.
        
        Args:
            prompt: The problem or query to reason about
            context: Additional context for reasoning
            
        Returns:
            Tuple of (final answer, reasoning steps)
        """
        if self.reasoning_mode == ReasoningMode.CHAIN_OF_THOUGHT:
            return self._chain_of_thought_reasoning(prompt, context)
        elif self.reasoning_mode == ReasoningMode.TREE_OF_THOUGHTS:
            return self._tree_of_thoughts_reasoning(prompt, context)
        elif self.reasoning_mode == ReasoningMode.REFLEXION:
            return self._reflexion_reasoning(prompt, context)
        elif self.reasoning_mode == ReasoningMode.DEBATE:
            return self._debate_reasoning(prompt, context)
        else:
            return self._standard_reasoning(prompt, context)
    
    def _chain_of_thought_reasoning(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ReasoningStep]]:
        """
        Implement Chain-of-Thought reasoning.
        """
        steps = []
        
        # System prompt for chain-of-thought
        system_prompt = """You are an advanced reasoning agent. Break down the problem step by step.
        For each step:
        1. State what you're thinking about
        2. Explain your reasoning
        3. Draw intermediate conclusions
        4. Build towards the final answer
        
        Format your response as:
        Step 1: [thought]
        Reasoning: [detailed reasoning]
        Conclusion: [intermediate conclusion]
        
        Continue until you reach the final answer."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {prompt}\n\nContext: {json.dumps(context) if context else 'None'}"}
        ]
        
        response = self.llm.chat("", messages)
        
        # Parse the response into steps
        lines = response.split('\n')
        current_step = None
        step_count = 0
        
        for line in lines:
            if line.startswith('Step'):
                if current_step:
                    steps.append(current_step)
                step_count += 1
                current_step = ReasoningStep(
                    step_number=step_count,
                    thought=line.split(':', 1)[1].strip() if ':' in line else line
                )
            elif current_step and line.startswith('Reasoning:'):
                current_step.action = line.split(':', 1)[1].strip()
            elif current_step and line.startswith('Conclusion:'):
                current_step.observation = line.split(':', 1)[1].strip()
        
        if current_step:
            steps.append(current_step)
        
        # Extract final answer
        final_answer_prompt = f"Based on the reasoning above, provide a concise final answer to: {prompt}"
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": final_answer_prompt})
        
        final_answer = self.llm.chat("", messages)
        
        self.reasoning_history = steps
        return final_answer, steps
    
    def _tree_of_thoughts_reasoning(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ReasoningStep]]:
        """
        Implement Tree-of-Thoughts reasoning with multiple paths.
        """
        steps = []
        
        # Generate multiple initial thoughts
        system_prompt = """You are an advanced reasoning agent using tree-of-thoughts.
        Generate 3 different approaches to solve this problem.
        For each approach, evaluate its promise (1-10) and potential issues."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {prompt}\n\nContext: {json.dumps(context) if context else 'None'}"}
        ]
        
        initial_thoughts = self.llm.chat("", messages)
        
        # Parse and evaluate each path
        paths = self._parse_thought_paths(initial_thoughts)
        
        # Explore the most promising path
        best_path = max(paths, key=lambda p: p.get('score', 0))
        
        # Deep dive into best path
        explore_prompt = f"""Explore this approach in detail:
        {best_path.get('approach', '')}
        
        Provide step-by-step reasoning and arrive at a solution."""
        
        messages.append({"role": "assistant", "content": initial_thoughts})
        messages.append({"role": "user", "content": explore_prompt})
        
        detailed_reasoning = self.llm.chat("", messages)
        
        # Create reasoning steps
        for i, path in enumerate(paths):
            step = ReasoningStep(
                step_number=i + 1,
                thought=path.get('approach', ''),
                confidence=path.get('score', 0) / 10.0,
                alternatives=[p.get('approach', '') for p in paths if p != path]
            )
            steps.append(step)
        
        self.reasoning_history = steps
        return detailed_reasoning, steps
    
    def _reflexion_reasoning(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ReasoningStep]]:
        """
        Implement Reflexion reasoning with self-critique and improvement.
        """
        steps = []
        max_iterations = 3
        
        messages = [
            {"role": "system", "content": "You are a reasoning agent that critically evaluates and improves your own thinking."},
            {"role": "user", "content": f"Problem: {prompt}\n\nContext: {json.dumps(context) if context else 'None'}"}
        ]
        
        for iteration in range(max_iterations):
            # Generate initial solution
            if iteration == 0:
                solution_prompt = "Provide your initial solution to this problem."
            else:
                solution_prompt = "Based on the critique, provide an improved solution."
            
            messages.append({"role": "user", "content": solution_prompt})
            solution = self.llm.chat("", messages)
            messages.append({"role": "assistant", "content": solution})
            
            # Self-critique
            critique_prompt = """Critically evaluate your solution:
            1. What are the strengths?
            2. What are the weaknesses or potential errors?
            3. What could be improved?
            4. Rate your confidence (1-10)."""
            
            messages.append({"role": "user", "content": critique_prompt})
            critique = self.llm.chat("", messages)
            messages.append({"role": "assistant", "content": critique})
            
            # Parse confidence
            confidence = self._extract_confidence(critique)
            
            step = ReasoningStep(
                step_number=iteration + 1,
                thought=solution,
                action="self-critique",
                observation=critique,
                confidence=confidence
            )
            steps.append(step)
            
            # Stop if confidence is high enough
            if confidence >= 0.9:
                break
        
        self.reasoning_history = steps
        return solution, steps
    
    def _debate_reasoning(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ReasoningStep]]:
        """
        Implement debate-style reasoning with multiple perspectives.
        """
        steps = []
        
        # Generate multiple perspectives
        perspectives = ["optimistic", "pessimistic", "analytical"]
        arguments = {}
        
        for i, perspective in enumerate(perspectives):
            system_prompt = f"""You are reasoning from a {perspective} perspective.
            Provide your analysis and solution to the problem."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {prompt}\n\nContext: {json.dumps(context) if context else 'None'}"}
            ]
            
            response = self.llm.chat("", messages)
            arguments[perspective] = response
            
            step = ReasoningStep(
                step_number=i + 1,
                thought=f"{perspective} perspective",
                action="analyze",
                observation=response,
                confidence=0.7
            )
            steps.append(step)
        
        # Synthesize arguments
        synthesis_prompt = f"""Given these different perspectives:
        
        Optimistic: {arguments['optimistic']}
        
        Pessimistic: {arguments['pessimistic']}
        
        Analytical: {arguments['analytical']}
        
        Synthesize the best solution considering all viewpoints."""
        
        messages = [
            {"role": "system", "content": "You are a master synthesizer of different viewpoints."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        final_synthesis = self.llm.chat("", messages)
        
        synthesis_step = ReasoningStep(
            step_number=len(steps) + 1,
            thought="Synthesis of all perspectives",
            action="synthesize",
            observation=final_synthesis,
            confidence=0.85
        )
        steps.append(synthesis_step)
        
        self.reasoning_history = steps
        return final_synthesis, steps
    
    def _standard_reasoning(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ReasoningStep]]:
        """
        Standard GPT-5 reasoning without special techniques.
        """
        messages = [
            {"role": "system", "content": "You are GPT-5, an advanced AI assistant with superior reasoning capabilities."},
            {"role": "user", "content": f"Problem: {prompt}\n\nContext: {json.dumps(context) if context else 'None'}"}
        ]
        
        response = self.llm.chat("", messages)
        
        step = ReasoningStep(
            step_number=1,
            thought=prompt,
            action="direct_reasoning",
            observation=response,
            confidence=0.8
        )
        
        self.reasoning_history = [step]
        return response, [step]
    
    def _parse_thought_paths(self, response: str) -> List[Dict[str, Any]]:
        """Parse multiple thought paths from response."""
        paths = []
        current_path = {}
        
        for line in response.split('\n'):
            if 'Approach' in line and ':' in line:
                if current_path:
                    paths.append(current_path)
                current_path = {'approach': line.split(':', 1)[1].strip()}
            elif 'Score:' in line or 'Promise:' in line:
                try:
                    score = float(''.join(c for c in line if c.isdigit() or c == '.'))
                    current_path['score'] = score
                except:
                    current_path['score'] = 5.0
        
        if current_path:
            paths.append(current_path)
        
        return paths if paths else [{'approach': 'Default approach', 'score': 5.0}]
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text."""
        import re
        
        # Look for patterns like "confidence: 8/10" or "8 out of 10"
        patterns = [
            r'confidence[:\s]+(\d+)[/\s]',
            r'(\d+)\s*(?:out of|/)\s*10',
            r'rate[:\s]+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                score = float(match.group(1))
                return min(score / 10.0, 1.0)
        
        return 0.5  # Default confidence
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning process."""
        if not self.reasoning_history:
            return "No reasoning history available."
        
        summary = f"Reasoning Mode: {self.reasoning_mode.value}\n"
        summary += f"Total Steps: {len(self.reasoning_history)}\n\n"
        
        for step in self.reasoning_history:
            summary += f"Step {step.step_number}:\n"
            summary += f"  Thought: {step.thought[:100]}...\n" if len(step.thought) > 100 else f"  Thought: {step.thought}\n"
            if step.confidence:
                summary += f"  Confidence: {step.confidence:.2f}\n"
            summary += "\n"
        
        avg_confidence = sum(s.confidence for s in self.reasoning_history) / len(self.reasoning_history)
        summary += f"Average Confidence: {avg_confidence:.2f}\n"
        
        return summary
    
    def clear_history(self):
        """Clear the reasoning history."""
        self.reasoning_history = []
    
    def set_reasoning_mode(self, mode: ReasoningMode):
        """Change the reasoning mode."""
        self.reasoning_mode = mode
        self.clear_history()