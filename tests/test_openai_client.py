from unittest.mock import patch, MagicMock
from openai_client import AzureOpenAIClient

def test_azure_openai_client_analyze_image_with_text():
    with patch("openai_client.AzureOpenAI") as mock_openai_class:
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai

        client = AzureOpenAIClient("key", "version", "endpoint", "deployment")
        result = client.analyze_image_with_text("sys", "user", "base64img")
        assert result == "Test response"