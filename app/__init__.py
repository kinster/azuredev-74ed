import json
from .legend_detection import extract_wall_codes
import os

def main(req):
    req_body = req.get_json()
    image_b64 = req_body["base64"]
    
    endpoint = os.environ["AZURE_DI_ENDPOINT"]
    key = os.environ["AZURE_DI_KEY"]

    wall_codes = extract_wall_codes(image_b64, endpoint, key)

    return func.HttpResponse(
        json.dumps({"wallCodes": wall_codes}),
        mimetype="application/json"
    )
