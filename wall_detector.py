import base64
import io
import re
import unicodedata
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential


class WallDetectorClient:
    def __init__(self, endpoint, key):
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    def clean_description(self, text):
        # Normalize unicode (e.g. Ś → S), strip odd symbols
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"[^\w\s\-\/]", "", text)  # Keep alphanumerics, dash, slash
        return text.strip()

    def is_valid_description(self, text):
        # Must not be another wall code, must be plain readable text
        if re.match(r"^[A-Z]{2,4}\.\d{3}[a-zA-Z]?$", text):
            return False
        if re.search(r"\b[A-Z]{2,4}\.\d{3}[a-zA-Z]?\b", text):
            return False
        if len(text.strip()) < 5:
            return False
        return True

    def count_wall_types_from_legend(self, image_data_list):
        wall_codes = []
        seen = set()
        max_expected = 30  # Stop after collecting up to 30 wall codes

        for image_data in image_data_list:
            base64_str = image_data["base64"]
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_str)
            image_stream = io.BytesIO(image_bytes)

            poller = self.document_analysis_client.begin_analyze_document(
                "prebuilt-layout", document=image_stream
            )
            result = poller.result()

            for page in result.pages:
                lines = page.lines
                started = False
                i = 0
                while i < len(lines):
                    line = lines[i]
                    text = line.content.strip()

                    match = re.match(r"^([A-Z]{2,4}\.\d{3}[a-zA-Z]?)$", text)
                    if match:
                        started = True
                        code = match.group(1)

                        if code not in seen:
                            seen.add(code)
                            wall_codes.append(code)

                        if len(wall_codes) >= max_expected:
                            break

                    elif not started:
                        i += 1
                        continue

                    i += 1

        return len(wall_codes), wall_codes
