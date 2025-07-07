import os
import os
import base64
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
from document_intelligence import AzureDocumentIntelligenceClient
from openai_client import AzureOpenAIClient
from vision_ocr import AzureVisionOCRClient
from wall_detector import WallDetectorClient
from datetime import datetime
from rich.console import Console
from weasyprint import HTML
import markdown
from weasyprint import HTML, CSS
import re
from openpyxl import Workbook

# Load API credentials
load_dotenv()

deploy_version = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
openai_client = AzureOpenAIClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # api_version="2024-05-01-preview",
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=deploy_version
)

print(f"Using OpenAI API version: {deploy_version}")

vision_ai_endpoint = os.getenv("AZURE_CV_ENDPOINT")
vision_ai_key = os.getenv("AZURE_CV_KEY")

di_ai_endpoint = os.getenv("AZURE_DI_ENDPOINT")
di_ai_key= os.getenv("AZURE_DI_KEY")

print(f"Using Azure Vision AI endpoint: {vision_ai_endpoint}")
print(f"Using Azure Vision AI key: {vision_ai_key[:5]}")


di_client = AzureDocumentIntelligenceClient(
    endpoint=di_ai_endpoint,
    api_key=di_ai_key
)

wd_client = WallDetectorClient(
    endpoint=di_ai_endpoint,
    key=di_ai_key
)

# Directories
image_dir = "inputs/images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Gather images and encode as base64
image_data_list = []
for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")
        image_data_list.append({"filename": filename, "base64": base64_img})


# system_prompt = (
# "Always count only the unique internal wall types. Do not use variation. Do not count duplicates. Use only what is visible in the image and be consistent every time."
# )

# user_prompt = (
# "How many unique internal wall types are shown in this drawing? Count each type only once using both the legend and the annotations. Return only the number."
# )  

# # Step 3: Make a single API call
# result = openai_client.analyze_image_with_text(
#     system_prompt=system_prompt,
#     user_prompt=user_prompt,
#     base64_image_list=[img["base64"] for img in image_data_list],
#     temperature=0,
#     top_p=1,  # Set your desired top_p value here (e.g., 1 for deterministic, <1 for more randomness)
# )

# # Output formatting
# output = f"# 📐 Drylining Wall Type & BoQ Summary\n\n{result}"
# print(output) 

# codes, count = di_client.extract_layout_text(image_data_list)
count, codes = wd_client.count_wall_types_from_legend(image_data_list)

print(f"Total unique wall types found: {count}")
for code in codes:
    print(code)
