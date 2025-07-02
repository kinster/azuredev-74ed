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

# Directories
image_dir = "images"
output_dir = "results"
output_file = os.path.join(output_dir, "wall_report.md")
os.makedirs(output_dir, exist_ok=True)

# Prompts
system_prompt = """
You are an AI assistant trained to support a drylining estimation workflow.

You receive architectural floor plans and technical detail drawings as input. Your role is to:

1. Parse the drawing and interpret the construction scale (e.g., 1:100 or 1:125).
2. Detect key features such as partitions, ceilings, fixtures, and structural elements.
3. Identify wall types (e.g. DW.451, SW.401) based on tags and line styles.
4. Measure each wall’s approximate length and height (where visible).
5. Measure ceiling areas and count openings (windows, hatches, doors).
6. Match each identified wall or ceiling type with its specification from technical detail sheets.
7. Estimate total material quantities based on standard drylining systems (e.g., boards, fixings, MF ceilings).
8. Apply labour productivity rates and waste factors to generate cost estimates.
9. Output a Bill of Quantities (BoQ) in markdown format with clear columns:
   Wall Type | Count | Total Length (m) | Description | Materials | Labour Hours

You must rely solely on visible annotations, line types, and tags in the drawings to determine the above. Where information is ambiguous or missing, annotate this clearly in the summary.

Use markdown formatting for output. Respond in a structured, concise, and professional format suitable for export.
"""

user_prompt = """
Analyze the submitted architectural floor plan and wall detail drawings. Perform the following tasks:

- Detect and count each distinct wall or ceiling type based on visible tags (e.g. DW.451, WL.403).
- Measure total lengths (and heights, if available) for walls using the plan scale.
- Calculate approximate areas for ceilings.
- Count openings (e.g. hatches, windows, doors).
- Retrieve descriptions for each wall/ceiling type from the detail sheets.
- Estimate required materials using typical drylining systems.
- Apply labour assumptions to each item using productivity norms.

Output a structured markdown report with:

1. Table: Wall/Ceiling Type | Count | Length/Area | Description | Materials | Labour
2. Summary: Key assumptions, uncertainties, and dominant system types

Only include elements that can be visually confirmed from the images.
"""

# Collect output
all_outputs = []

# Process each image
for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Extract OCR text for the image
        ocr_text = vision_client.extract_ocr_text(filepath)

        print(f"🔍 Analyzing {filename}...")

        result = openai_client.analyze_image_with_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt + "\n\nOCR Extracted Text:\n" + ocr_text,
            base64_image=base64_image
        )

        formatted = f"## 🖼️ {filename}\n\n{result}"
        all_outputs.append(formatted)

# Final markdown content
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
full_report = f"# 🧱 Wall Analysis Report\nGenerated on {timestamp}\n\n" + "\n\n".join(all_outputs)

# Save to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\n✅ Markdown report saved to: {output_file}\n")

# Print to console at the end
console = Console()
console.print(full_report, markup=False)
