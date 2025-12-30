
import unittest
from src.um_agent_coder.tools.base import Tool, ToolRegistry, ToolResult

class MockTool(Tool):
    TASK_TYPES = ["mock"]

    def execute(self, **kwargs):
        pass

    def get_parameters(self):
        pass

class MultiTaskTool(Tool):
    TASK_TYPES = ["mock", "other"]

    def execute(self, **kwargs):
        pass

    def get_parameters(self):
        pass

class GeneralTool(Tool):
    TASK_TYPES = ["general"]

    def execute(self, **kwargs):
        pass

    def get_parameters(self):
        pass

class NoTaskTool(Tool):
    TASK_TYPES = []

    def execute(self, **kwargs):
        pass

    def get_parameters(self):
        pass

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.mock_tool = MockTool()
        self.multi_tool = MultiTaskTool()
        self.general_tool = GeneralTool()
        self.no_task_tool = NoTaskTool()

        self.registry.register(self.mock_tool)
        self.registry.register(self.multi_tool)
        self.registry.register(self.general_tool)
        self.registry.register(self.no_task_tool)

    def test_get_tools_for_task(self):
        # Test filtering for "mock" task
        tools = self.registry.get_tools_for_task("mock")
        self.assertIn(self.mock_tool, tools)
        self.assertIn(self.multi_tool, tools)
        self.assertIn(self.general_tool, tools)
        self.assertNotIn(self.no_task_tool, tools)

        # Test filtering for "other" task
        tools = self.registry.get_tools_for_task("other")
        self.assertIn(self.multi_tool, tools)
        self.assertIn(self.general_tool, tools)
        self.assertNotIn(self.mock_tool, tools)
        self.assertNotIn(self.no_task_tool, tools)

        # Test filtering for unknown task
        # Should only return general tool
        tools = self.registry.get_tools_for_task("unknown")
        self.assertIn(self.general_tool, tools)
        self.assertNotIn(self.mock_tool, tools)
        self.assertNotIn(self.multi_tool, tools)
        self.assertNotIn(self.no_task_tool, tools)

    def test_get_all(self):
        tools = self.registry.get_all()
        self.assertEqual(len(tools), 4)

if __name__ == '__main__':
    unittest.main()
