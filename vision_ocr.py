from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
from PIL import Image
import os

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
        # Check if image is too large and compress if needed
        if os.path.getsize(image_path) > 4 * 1024 * 1024:
            compressed_path = image_path + "_compressed.png"
            compress_image(image_path, compressed_path)
            image_path = compressed_path

        try:
            with open(image_path, "rb") as image_stream:
                ocr_result = self.client.read_in_stream(image_stream, raw=True)
            print(ocr_result)
        except Exception as e:
            if hasattr(e, 'response') and hasattr(e.response, 'content'):
                print("Azure error response:", e.response.content)
            raise
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

def compress_image(input_path, output_path, max_size_mb=4):
    print(f"Compressing image {input_path} to {output_path} with max size {max_size_mb}MB")
    img = Image.open(input_path)
    img.save(output_path, optimize=True, quality=85)
    while os.path.getsize(output_path) > max_size_mb * 1024 * 1024:
        img = img.resize((int(img.width * 0.9), int(img.height * 0.9)))
        img.save(output_path, optimize=True, quality=85)