from typing import Any

from um_agent_coder.agent.yolo.executor import SubAgentExecutor
from um_agent_coder.agent.yolo.planner import CompetitivePlanner


class YoloManager:
    """
    Manager for 'Yolo Mode'.
    Coordinates Competitive Planning -> Specification -> Autonomous Execution.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.planner = CompetitivePlanner(config)

    def run(self, prompt: str):
        print("\n🚀 ENTERING YOLO MODE 🚀")
        print("--------------------------------")

        # 1. Competitive Planning
        master_plan = self.planner.create_master_plan(prompt)

        print("\n📋 MASTER PLAN GENERATED 📋")
        print("--------------------------------")
        print(master_plan)
        print("--------------------------------")

        # 2. Execution
        # In true Yolo mode, we might just go.
        # But asking for a quick enter key is safer for a CLI tool unless flag is set.
        # For now, we assume implicit approval or we could add an input() check.

        executor = SubAgentExecutor(self.config, master_plan)
        result = executor.execute()

        return result
