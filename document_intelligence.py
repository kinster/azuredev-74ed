from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import re
import base64
import io
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

class AzureDocumentIntelligenceClient:

    def __init__(self, endpoint, api_key):
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )

    def extract_layout_text(self, image_data_list):
        """
        Processes a list of base64-encoded images, extracts wall codes from legends,
        and returns unique wall types across all images.

        Args:
            image_data_list (list): List of dicts with 'filename' and 'base64' keys.
            endpoint (str): Azure Form Recognizer endpoint.
            key (str): Azure Form Recognizer key.

        Returns:
            tuple: (count, sorted list of unique wall codes)
        """
        wall_codes = set()

        for image_data in image_data_list:
            base64_str = image_data["base64"]

            # If base64 has a prefix like "data:image/png;base64,...", strip it
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_str)
            image_stream = io.BytesIO(image_bytes)

            # Call Azure Document Intelligence Layout model
            poller = self.document_analysis_client.begin_analyze_document("prebuilt-layout", document=image_stream)
            result = poller.result()

            # Extract from lines
            for page in result.pages:
                for line in page.lines:
                    text = line.content.strip()
                    match = re.match(r"^([A-Z]{2,3}\.\d{3}[a-zA-Z]?)\b", text)
                    if match:
                        wall_codes.add(match.group(1))

            # Extract from tables (if legend is tabular)
            for table in result.tables:
                for cell in table.cells:
                    cell_text = cell.content.strip()
                    if re.match(r"^[A-Z]{2,3}\.\d{3}[a-zA-Z]?$", cell_text):
                        wall_codes.add(cell_text)

        return len(wall_codes), sorted(wall_codes)
