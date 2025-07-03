import os
import base64
import time
from dotenv import load_dotenv
from openai_client import AzureOpenAIClient
from vision_ocr import AzureVisionOCRClient
from datetime import datetime
from rich.console import Console

# Load API credentials
load_dotenv()
openai_client = AzureOpenAIClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
)

# Directories
image_dir = "inputs/images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Gather images and OCR in one batch
image_data_list = []
ocr_snippets = []

for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")
        image_data_list.append({"filename": filename, "base64": base64_img})

# Step 2: Build prompt for classification
system_prompt = (
    "You are an AI assistant that classifies construction project images. "
    "Given a set of architectural images, classify each as either a 'main document' "
    "(such as a floor plan or overall drawing) or an 'additional detail document' "
    "(such as a technical detail sheet, specification, or section). "
    "Return a list of labels ('main' or 'detail') in the same order as the images, one per line."
)

user_prompt = (
    "Classify each of the following images as either a 'main document' or an 'additional detail document'. "
    "Return only the list of labels, one per line, in the same order as the images provided."
)

# Step 3: Make a single API call with all images
content = [{"type": "text", "text": user_prompt}]
for img in image_data_list:
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"} })

for attempt in range(3):
    try:
        response = openai_client.client.chat.completions.create(
            model=openai_client.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0,
            max_tokens=100
        )
        break  # Success!
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(5)
else:
    print("All attempts failed.")

labels = [line.strip().lower() for line in response.choices[0].message.content.splitlines() if line.strip()]

# Output results
for img, label in zip(image_data_list, labels):
    print(f"{img['filename']}: {label}")

# Optionally, save results to a file
output_file = os.path.join(output_dir, "image_classification_results.txt")
with open(output_file, "w") as f:
    for img, label in zip(image_data_list, labels):
        f.write(f"{img['filename']}: {label}\n")

print(f"\n✅ Classification results saved to: {output_file}\n")

# Print to console at the end
with open(output_file, "r") as f:
    content = f.read()

console = Console()
console.print(content, markup=False)
