import time
import uuid
import json
import traceback
from typing import Optional, Dict, Any

from um_agent_coder.llm.base import LLM
from .enhanced_agent import EnhancedAgent
from .persistence import JobStore, JobStatus, JobState
from .planner import TaskAnalysis, ExecutionPlan, TaskStep
from um_agent_coder.context import ContextManager, ContextItem, ContextType

class AsyncRunner:
    """
    Runs jobs asynchronously, handling state persistence and resumption.
    """
    def __init__(self, llm: LLM, config: Dict[str, Any], job_store: JobStore):
        self.llm = llm
        self.config = config
        self.job_store = job_store
        # We reuse EnhancedAgent for its tool registry and helpers, 
        # but we control the execution flow.
        self.agent = EnhancedAgent(llm, config)

    def create_job(self, prompt: str) -> str:
        """Initialize a new job and perform initial analysis."""
        job_id = str(uuid.uuid4())[:8]
        print(f"Creating job {job_id}...")
        
        job = self.job_store.create_job(job_id, prompt)
        
        try:
            # 1. Plan
            print("  Analyzing task...")
            analysis = self.agent.task_planner.analyze_task(prompt)
            plan = self.agent.task_planner.create_execution_plan(analysis, prompt)
            
            # Serialize objects manually since they are dataclasses
            # (In a production system, use a proper serializer)
            job.analysis = {
                "task_type": analysis.task_type.value,
                "complexity": analysis.complexity,
                "files_to_analyze": analysis.files_to_analyze,
                "estimated_tokens": analysis.estimated_tokens
            }
            
            # Serialize plan steps
            job.plan = {
                "steps": [
                    {
                        "description": step.description,
                        "action": step.action,
                        "parameters": step.parameters,
                        "estimated_tokens": step.estimated_tokens,
                        "priority": step.priority
                    }
                    for step in plan.steps
                ],
                "estimated_cost": plan.estimated_cost
            }
            
            job.status = JobStatus.PENDING.value
            self.job_store.save_job(job)
            
            # Initialize context
            context_manager = ContextManager(self.config.get("max_context_tokens", 100000))
            context_manager.add(
                content=f"Task: {prompt}\nType: {analysis.task_type.value}",
                type=ContextType.PROJECT_INFO,
                source="task_analysis",
                priority=9
            )
            self._save_context_to_store(job_id, context_manager)
            
            print(f"  Job {job_id} created with {len(plan.steps)} steps.")
            return job_id
            
        except Exception as e:
            job.status = JobStatus.FAILED.value
            job.error = str(e)
            self.job_store.save_job(job)
            raise e

    def resume_job(self, job_id: str, max_steps: int = 1):
        """
        Execute the next steps for a job.
        
        Args:
            job_id: The job to resume
            max_steps: Maximum steps to execute in this run loop (0 for all)
        """
        job = self.job_store.load_job(job_id)
        if not job:
            print(f"Job {job_id} not found.")
            return

        if job.status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
            print(f"Job {job_id} is already {job.status}.")
            return

        print(f"Resuming job {job_id} (Step {job.current_step_index + 1}/{len(job.plan['steps'])})")
        
        # Load Context
        context_data = self.job_store.load_context(job_id)
        context_manager = self._restore_context(context_data)
        
        # Sync agent's context manager
        self.agent.context_manager = context_manager
        
        steps_executed = 0
        total_steps = len(job.plan['steps'])
        
        try:
            job.status = JobStatus.RUNNING.value
            self.job_store.save_job(job)

            while job.current_step_index < total_steps:
                if max_steps > 0 and steps_executed >= max_steps:
                    print(f"Reached execution limit of {max_steps} steps. Pausing.")
                    break

                step_data = job.plan['steps'][job.current_step_index]
                
                # Reconstruct step object
                step = TaskStep(
                    description=step_data['description'],
                    action=step_data['action'],
                    parameters=step_data['parameters'],
                    estimated_tokens=step_data['estimated_tokens'],
                    priority=step_data['priority']
                )
                
                print(f"  Executing Step {job.current_step_index + 1}: {step.description}...")
                
                # EXECUTE
                result = self.agent._execute_step(step)
                
                # Save result
                job.results.append(result)
                job.current_step_index += 1
                job.updated_at = time.time()
                
                # Save Context (agent._execute_step updates self.agent.context_manager internally)
                self._save_context_to_store(job_id, self.agent.context_manager)
                
                # Save Job State
                self.job_store.save_job(job)
                
                steps_executed += 1
                
                # Optional: Sleep to respect rate limits if needed
                time.sleep(1)

            if job.current_step_index >= total_steps:
                # Generate final response
                print("  Generating final response...")
                final_response = self.agent._generate_response(job.prompt, job.results)
                
                job.status = JobStatus.COMPLETED.value
                # We can store the final response in results or a new field
                job.results.append({"final_response": final_response})
                self.job_store.save_job(job)
                print(f"Job {job_id} COMPLETED.")

        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            job.status = JobStatus.FAILED.value
            job.error = str(e)
            self.job_store.save_job(job)

    def _save_context_to_store(self, job_id: str, manager: ContextManager):
        # Serialize ContextItems
        items_data = []
        for item in manager.items:
            items_data.append({
                "content": item.content,
                "type": item.type.value,
                "source": item.source,
                "tokens": item.tokens,
                "timestamp": item.timestamp,
                "priority": item.priority,
                "metadata": item.metadata
            })
        
        self.job_store.save_context(job_id, {"items": items_data})

    def _restore_context(self, data: Dict[str, Any]) -> ContextManager:
        manager = ContextManager(self.config.get("max_context_tokens", 100000))
        for item_data in data.get("items", []):
            try:
                # Handle enum conversion
                ctype = ContextType(item_data["type"])
                
                item = ContextItem(
                    content=item_data["content"],
                    type=ctype,
                    source=item_data["source"],
                    tokens=item_data["tokens"],
                    timestamp=item_data["timestamp"],
                    priority=item_data["priority"],
                    metadata=item_data["metadata"]
                )
                manager.items.append(item)
                manager.current_tokens += item.tokens
            except Exception as e:
                print(f"Warning: Failed to restore context item: {e}")
        return manager
