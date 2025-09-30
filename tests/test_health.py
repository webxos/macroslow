import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertIn("torch_version", response.json())

if __name__ == "__main__":
    unittest.main()