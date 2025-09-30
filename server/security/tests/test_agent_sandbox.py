import unittest
from server.security.agent_sandbox import AgentSandbox

class TestAgentSandbox(unittest.TestCase):
    def test_sandbox_init(self):
        sandbox = AgentSandbox()
        self.assertTrue(sandbox.sandbox_dir.startswith("/tmp"))

    def test_invalid_agent_type(self):
        sandbox = AgentSandbox()
        with self.assertRaises(Exception):
            sandbox.execute_agent("invalid-agent", "test-task")

if __name__ == "__main__":
    unittest.main()