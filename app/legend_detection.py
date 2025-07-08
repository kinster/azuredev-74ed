import base64
import io
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import re

LEGEND_BBOX = {
    "x1": 8915.0,
    "y1": 1120.0,
    "x2": 10078.84,
    "y2": 3507.25
}

def is_inside_legend(polygon):
    xs = [p.x for p in polygon]
    ys = [p.y for p in polygon]
    return (
        min(xs) >= LEGEND_BBOX["x1"] and max(xs) <= LEGEND_BBOX["x2"] and
        min(ys) >= LEGEND_BBOX["y1"] and max(ys) <= LEGEND_BBOX["y2"]
    )

def extract_wall_codes(base64_image: str, form_recognizer_endpoint: str, key: str):
    # Decode base64
    if "," in base64_image:
        base64_image = base64_image.split(",")[1]
    image_bytes = base64.b64decode(base64_image)
    stream = io.BytesIO(image_bytes)

    # Azure Form Recognizer
    client = DocumentAnalysisClient(endpoint=form_recognizer_endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", document=stream)
    result = poller.result()

    seen = set()
    wall_codes = []

    for page in result.pages:
        for line in page.lines:
            if not is_inside_legend(line.polygon):
                continue
            match = re.match(r"^([A-Z]{2,4}\.\d{3}[a-zA-Z]?)$", line.content.strip())
            if match:
                code = match.group(1)
                if code not in seen:
                    seen.add(code)
                    wall_codes.append(code)

    return wall_codes
