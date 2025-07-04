import os
import os
import base64
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai_client import AzureOpenAIClient
from vision_ocr import AzureVisionOCRClient
from datetime import datetime
from rich.console import Console
from weasyprint import HTML
import markdown
from weasyprint import HTML, CSS
import re
from openpyxl import Workbook

endpoint = "https://walldetection-resource.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"

subscription_key = "BU3QXrqlrY4D1sh0u1My3pSL9zzQlkYav20OXHPa5ppRRhsCqQDAJQQJ99BGACHYHv6XJ3w3AAAAACOGSRa3"
api_version = "2024-12-01-preview"

# Load API credentials
load_dotenv()
openai_client = AzureOpenAIClient(
    api_key=subscription_key,
    api_version=api_version,
    endpoint=endpoint,
    deployment_name=deployment
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

# Step 2: Classify images using OpenAI client
results = openai_client.classify_documents(image_data_list)

# Find the main image(s) and supplementary images
main_images = [filename for filename, label in results if label == "main"]
supplementary_images = [filename for filename, label in results if label == "detail"]

print(f"Main image(s): {main_images}")
print(f"Supplementary images: {supplementary_images}")

# Output results
for filename, label in results:
    print(f"{filename}: {label}")

# Optionally, save results to a file
output_file = os.path.join(output_dir, "image_classification_results.txt")
with open(output_file, "w") as f:
    for filename, label in results:
        f.write(f"{filename}: {label}\n")

print(f"\n✅ Classification results saved to: {output_file}\n")

# Print to console at the end
with open(output_file, "r") as f:
    content = f.read()

console = Console()
console.print(content, markup=False)
