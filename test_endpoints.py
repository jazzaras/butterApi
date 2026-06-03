import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock the GEMINI_API_KEY environment variable before importing the app
# to prevent startup/validation errors during unit tests
os.environ["GEMINI_API_KEY"] = "mock_api_key_for_testing"

from main import app

client = TestClient(app)

class TestButterDoughAPI(unittest.TestCase):

    def test_root_endpoint(self):
        """Test the health check endpoint returns 200 and online status."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["app"], "Butter & Dough API")

    def test_butter_invalid_url(self):
        """Test that invalid URLs are rejected with a 400 Bad Request."""
        # Non-http URL
        response = client.post("/butter", json={"linkedin_url": "ftp://linkedin.com/posts/123"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid URL", response.json()["detail"])

        # Non-linkedin URL
        response = client.post("/butter", json={"linkedin_url": "https://google.com/posts/123"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid URL", response.json()["detail"])

    @patch("main.scrape_linkedin")
    @patch("main.get_gemini_client")
    def test_butter_endpoint_success(self, mock_get_client, mock_scrape):
        """Test the Butter endpoint under normal operating conditions with mocks."""
        # Mock scraper content
        mock_scrape.return_value = "This is a mock LinkedIn post content about AI."

        # Mock Gemini client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Return a structured JSON matching ButterSummaryLLM schema
        mock_response.text = '{"bullet_points": ["نقطة 1", "نقطة 2", "نقطة 3"]}'
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Call endpoint
        response = client.post(
            "/butter",
            json={"linkedin_url": "https://www.linkedin.com/posts/wafaa-al-harbi-123"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["feature"], "butter")
        self.assertEqual(len(data["output"]), 3)
        self.assertEqual(data["output"][0], "نقطة 1")
        self.assertEqual(data["output"][1], "نقطة 2")
        self.assertEqual(data["output"][2], "نقطة 3")

    @patch("main.get_gemini_client")
    def test_dough_endpoint_success(self, mock_get_client):
        """Test the Dough endpoint under normal operating conditions with mocks."""
        # Mock Gemini client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Return a structured JSON matching DoughLLM schema
        mock_response.text = '{"expanded_content": "هذا محتوى موسع احترافي يناسب لينكد إن."}'
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Call endpoint
        response = client.post(
            "/dough",
            json={"text": "AI changing development"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["feature"], "dough")
        self.assertEqual(data["output"], "هذا محتوى موسع احترافي يناسب لينكد إن.")

if __name__ == "__main__":
    unittest.main()
