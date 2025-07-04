import logging
import azure.functions as func
import os
import tempfile
from vision_ocr import AzureVisionOCRClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('OCR function triggered.')

    # Load credentials from environment variables
    vision_client = AzureVisionOCRClient(
        endpoint=os.getenv("AZURE_CV_ENDPOINT"),
        api_key=os.getenv("AZURE_CV_KEY")
    )

    # Get uploaded file
    file = req.files.get('file')
    if not file:
        return func.HttpResponse("No file uploaded.", status_code=400)

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        ocr_text = vision_client.extract_ocr_text(tmp_path)
        return func.HttpResponse(ocr_text, mimetype="text/plain")
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"OCR failed: {str(e)}", status_code=500)
    finally:
        os.remove(tmp_path)