from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time

class AzureVisionOCRClient:
    """
    Wrapper for Azure Computer Vision OCR.
    """
    def __init__(self, endpoint, api_key):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))

    def extract_ocr_text(self, image_path, max_retries=30, retry_delay=1):
        """
        Extracts OCR text from an image using Azure Computer Vision Read API.

        Args:
            image_path (str): Path to the image file.
            max_retries (int): Maximum polling attempts.
            retry_delay (int): Delay between polling attempts (seconds).

        Returns:
            str: Extracted text.
        """
        with open(image_path, "rb") as image_stream:
            ocr_result = self.client.read_in_stream(image_stream, raw=True)
        # Get the operation location (URL with an ID at the end)
        operation_location = ocr_result.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        for _ in range(max_retries):
            result = self.client.get_read_result(operation_id)
            if result.status not in ['notStarted', 'running']:
                break
            time.sleep(retry_delay)
        else:
            raise TimeoutError("OCR operation timed out.")

        lines = []
        if result.status == OperationStatusCodes.succeeded:
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    lines.append(line.text)
        return "\n".join(lines)