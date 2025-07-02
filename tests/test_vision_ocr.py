import pytest
from unittest.mock import patch, MagicMock
from vision_ocr import AzureVisionOCRClient

@pytest.fixture
def fake_image_path(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"fake image data")
    return str(img)

def test_azure_vision_ocr_client_extract_ocr_text(fake_image_path):
    with patch("vision_ocr.ComputerVisionClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.read_in_stream.return_value.headers = {"Operation-Location": "https://fake/operation/123"}
        mock_result = MagicMock()
        mock_result.status = "succeeded"
        mock_result.analyze_result.read_results = [
            MagicMock(lines=[MagicMock(text="Line 1"), MagicMock(text="Line 2")])
        ]
        mock_client.get_read_result.return_value = mock_result
        mock_client_class.return_value = mock_client

        client = AzureVisionOCRClient("endpoint", "key")
        text = client.extract_ocr_text(fake_image_path)
        assert "Line 1" in text and "Line 2" in text