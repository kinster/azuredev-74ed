import os
import base64
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
vision_client = AzureVisionOCRClient(
    endpoint=os.getenv("AZURE_CV_ENDPOINT"),
    api_key=os.getenv("AZURE_CV_KEY")
)

# Debug: Print environment variables
print(os.getenv("AZURE_CV_ENDPOINT"))
print(os.getenv("AZURE_CV_KEY")[:5])

# Directories
image_dir = "images"
output_dir = "results"
output_file = os.path.join(output_dir, "wall_report.md")
os.makedirs(output_dir, exist_ok=True)

# Prompts
system_prompt = """
You are a construction AI assistant. You specialize in interpreting architectural floor plans and drylining technical detail sheets.

You are given:
- A floor plan image that shows wall type labels and scale
- One or more technical detail drawings that define the meaning of each wall type (e.g. WL.403, SW.401)
- OCR-extracted text from the plan image (optional, but included if available)

Your task is to:
1. Identify and list all unique wall types (e.g. WL.401, DW.451)
2. Count how many times each appears
3. Estimate the total length for each using the scale shown on the drawing (e.g., 1:125)
4. Match each wall type to a description from the technical drawings

Return the results in a table like this:

| Wall Type | Count | Total Length (m) | Description                     |
|-----------|-------|------------------|---------------------------------|
| WL.401    | 6     | ~35              | Standard wall lining system     |
| DW.451    | 4     | ~22              | Demountable dry wall partition  |

If descriptions are not clearly visible or defined, provide a best-guess based on naming patterns.
Also include a summary of any assumptions.
"""

# Store final results
all_outputs = []

# Process each image
for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(image_dir, filename)

        # Read and encode image
        with open(filepath, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # 🔍 Extract OCR text for the current image
        ocr_text = vision_client.extract_ocr_text(filepath)
        print(f"🔍 OCR for {filename}:\n{ocr_text[:300]}...")  # Preview first 300 chars

        # 👤 User prompt including the OCR text
        user_prompt = f"""
This is the OCR text extracted from a floor plan image showing internal wall types:

{ocr_text}

Please:
- List all unique wall types (e.g. SW.401, WL.404, DW.453)
- Count how many times each type appears in the image/OCR text
- Estimate total length using the drawing scale (1:125)
- Cross-reference wall types with technical details (if separate drawings are provided)
- Return a structured markdown table with type, count, total length in metres, and description

The images of the floor plan and detail sheets will be passed alongside this prompt.
"""

        # 🧠 Send to GPT-4o (Azure OpenAI)
        result = openai_client.analyze_image_with_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base64_image=base64_image
        )

        # 📦 Collect output per image
        formatted = f"## 🖼️ {filename}\n\n{result}"
        all_outputs.append(formatted)

# 🧾 Final combined report
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
full_report = f"# 🧱 Wall Analysis Report\nGenerated on {timestamp}\n\n" + "\n\n".join(all_outputs)

# Save to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\n✅ Markdown report saved to: {output_file}\n")

# Print to console at the end
console = Console()
console.print(full_report, markup=False)
