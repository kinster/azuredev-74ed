import base64
import io
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from collections import defaultdict

class WallDetectorClient:

    LEGEND_BOUNDING_BOX = {
        "x": 8915.0,
        "y": 1120.0,
        "width": 1163.84,
        "height": 2387.25
    }

    LEGEND_BOUNDING_BOX["x2"] = LEGEND_BOUNDING_BOX["x"] + LEGEND_BOUNDING_BOX["width"]
    LEGEND_BOUNDING_BOX["y2"] = LEGEND_BOUNDING_BOX["y"] + LEGEND_BOUNDING_BOX["height"]

    def __init__(self, endpoint, key):
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    # def clean_description(self, text):
    #     # Normalize unicode (e.g. Ś → S), strip odd symbols
    #     text = unicodedata.normalize("NFKD", text)
    #     text = re.sub(r"[^\w\s\-\/]", "", text)  # Keep alphanumerics, dash, slash
    #     return text.strip()

    # def is_valid_description(self, text):
    #     # Must not be another wall code, must be plain readable text
    #     if re.match(r"^[A-Z]{2,4}\.\d{3}[a-zA-Z]?$", text):
    #         return False
    #     if re.search(r"\b[A-Z]{2,4}\.\d{3}[a-zA-Z]?\b", text):
    #         return False
    #     if len(text.strip()) < 5:
    #         return False
    #     return True

    def count_wall_types_from_legend(self, image_data_list):
        wall_codes = []
        seen = set()

        def is_inside_legend(polygon):
            if not polygon or len(polygon) < 4:
                return False
            xs = [p.x for p in polygon]
            ys = [p.y for p in polygon]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            return (
                self.LEGEND_BOUNDING_BOX["x"] <= min_x <= self.LEGEND_BOUNDING_BOX["x2"]
                and self.LEGEND_BOUNDING_BOX["y"] <= min_y <= self.LEGEND_BOUNDING_BOX["y2"]
            )

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
                for line in page.lines:
                    # print(line)
                    if not is_inside_legend(line.polygon):
                        continue

                    text = line.content.strip()
                    match = re.match(r"^([A-Z]{2,4}\.\d{3}[a-zA-Z]?)$", text)
                    if match:
                        code = match.group(1)
                        if code not in seen:
                            seen.add(code)
                            wall_codes.append(code)

        return len(wall_codes), wall_codes
    
    def debug_detected_lines(self, image_data_list):
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

            for page_num, page in enumerate(result.pages, start=1):
                print(f"\n--- Page {page_num} ---\n")
                for line in page.lines:
                    text = line.content.strip()
                    region_str = "n/a"
                    try:
                        if hasattr(line, "bounding_regions") and line.bounding_regions:
                            region = line.bounding_regions[0]
                            point = region.bounding_polygon[0]
                            region_str = f"x={point.x:.2f}, y={point.y:.2f}"
                    except Exception as e:
                        region_str = f"error extracting: {e}"
                    print(f"Text: '{text}' | Location: {region_str}")


    def count_wall_type_occurrences(self, image_data_list, legend_codes, legend_line_cutoff=0):
        legend_codes_set = set(legend_codes)
        wall_counts = defaultdict(int)

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

                # Exclude bottom ~40 lines assumed to be legend
                plan_lines = lines[:-legend_line_cutoff] if len(lines) > legend_line_cutoff else []

                for line in plan_lines:
                    text = line.content.strip()

                    for code in legend_codes_set:
                        if code in text:
                            wall_counts[code] += 1

        return dict(wall_counts)
