#!/usr/bin/env python3
"""
Example demonstrating GPT-5 Reasoning Agent with different reasoning modes.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.um_agent_coder.agent.gpt5_reasoning_agent import GPT5ReasoningAgent, ReasoningMode


def demonstrate_reasoning_modes():
    """Demonstrate different reasoning modes with GPT-5."""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Example problems to solve
    problems = [
        {
            "prompt": "Design a distributed caching system that can handle 1 million requests per second with sub-millisecond latency",
            "context": {"constraints": ["high availability", "data consistency", "cost-effective"]}
        },
        {
            "prompt": "How would you implement a real-time collaborative code editor like VS Code Live Share?",
            "context": {"requirements": ["conflict resolution", "low latency", "security"]}
        }
    ]
    
    # Test different reasoning modes
    modes = [
        ReasoningMode.CHAIN_OF_THOUGHT,
        ReasoningMode.TREE_OF_THOUGHTS,
        ReasoningMode.REFLEXION,
        ReasoningMode.DEBATE
    ]
    
    for problem in problems[:1]:  # Test with first problem
        print(f"\n{'='*80}")
        print(f"Problem: {problem['prompt']}")
        print(f"{'='*80}\n")
        
        for mode in modes[:2]:  # Test first two modes for brevity
            print(f"\n{'-'*40}")
            print(f"Reasoning Mode: {mode.value}")
            print(f"{'-'*40}\n")
            
            # Create agent with specific reasoning mode
            agent = GPT5ReasoningAgent(
                api_key=api_key,
                model="gpt-5",  # Use gpt-5-mini for cheaper option
                reasoning_mode=mode,
                temperature=0.7
            )
            
            # Perform reasoning
            answer, steps = agent.reason(problem['prompt'], problem['context'])
            
            # Display results
            print(f"Final Answer:\n{answer}\n")
            print(f"\nReasoning Summary:\n{agent.get_reasoning_summary()}")
            
            # Clear history for next problem
            agent.clear_history()


def demonstrate_model_comparison():
    """Compare GPT-5 models for different tasks."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
    
    # Simple classification task (good for nano)
    simple_task = "Classify this sentiment: 'The product exceeded my expectations!'"
    
    # Complex reasoning task (needs full GPT-5)
    complex_task = "Design a fault-tolerant microservices architecture for a banking system"
    
    print(f"\n{'='*80}")
    print("Model Comparison")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\nModel: {model}")
        print(f"{'-'*40}")
        
        agent = GPT5ReasoningAgent(
            api_key=api_key,
            model=model,
            reasoning_mode=ReasoningMode.STANDARD,
            temperature=0.3
        )
        
        # Test simple task
        print(f"\nSimple Task: {simple_task}")
        answer, _ = agent.reason(simple_task)
        print(f"Response: {answer[:200]}...")
        
        # Get model info for cost estimation
        model_info = agent.llm.get_model_info()
        print(f"Estimated cost: ${model_info['cost_per_1k_input']*0.1 + model_info['cost_per_1k_output']*0.2:.6f}")


def demonstrate_agentic_workflow():
    """Demonstrate a multi-step agentic workflow with GPT-5."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print(f"\n{'='*80}")
    print("Agentic Workflow: Building a Data Pipeline")
    print(f"{'='*80}\n")
    
    # Create agent
    agent = GPT5ReasoningAgent(
        api_key=api_key,
        model="gpt-5-mini",  # Use mini for cost-effectiveness
        reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
        temperature=0.5
    )
    
    # Multi-step workflow
    workflow_steps = [
        {
            "step": "Requirements Analysis",
            "prompt": "Analyze requirements for a real-time data pipeline processing 10TB daily from IoT sensors",
            "context": {"data_sources": ["temperature sensors", "humidity sensors", "motion detectors"]}
        },
        {
            "step": "Architecture Design",
            "prompt": "Based on the requirements, design the data pipeline architecture",
            "context": {"technologies": ["Apache Kafka", "Apache Flink", "Elasticsearch"]}
        },
        {
            "step": "Implementation Plan",
            "prompt": "Create a detailed implementation plan with milestones",
            "context": {"timeline": "3 months", "team_size": 5}
        },
        {
            "step": "Risk Assessment",
            "prompt": "Identify potential risks and mitigation strategies",
            "context": {"critical_factors": ["data loss", "latency", "scalability"]}
        }
    ]
    
    results = []
    
    for workflow_step in workflow_steps:
        print(f"\n{workflow_step['step']}")
        print(f"{'-'*40}")
        
        # Add previous results as context
        if results:
            workflow_step['context']['previous_decisions'] = results[-1][:500]
        
        answer, steps = agent.reason(workflow_step['prompt'], workflow_step['context'])
        results.append(answer)
        
        print(f"Result: {answer[:300]}...")
        print(f"Reasoning confidence: {sum(s.confidence for s in steps)/len(steps) if steps else 0:.2f}")
    
    print(f"\n{'='*80}")
    print("Workflow Complete!")
    print(f"Total reasoning steps: {len(agent.reasoning_history)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-5 Reasoning Agent Examples")
    parser.add_argument("--demo", choices=["reasoning", "models", "workflow", "all"], 
                       default="reasoning", help="Which demonstration to run")
    
    args = parser.parse_args()
    
    if args.demo == "reasoning" or args.demo == "all":
        demonstrate_reasoning_modes()
    
    if args.demo == "models" or args.demo == "all":
        demonstrate_model_comparison()
    
    if args.demo == "workflow" or args.demo == "all":
        demonstrate_agentic_workflow()