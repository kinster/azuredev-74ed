
import os
import base64
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime
from rich.console import Console

# Load API credentials
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Directories
image_dir = "images"
output_dir = "results"
output_file = os.path.join(output_dir, "wall_report.md")
os.makedirs(output_dir, exist_ok=True)

# Prompts
system_prompt = """You are a construction analysis assistant trained to process architectural plans. Your task is to analyse both floor plans and technical detail sheets provided as images.

Use your capabilities to extract the following:
1. All wall types as defined by annotations (e.g. SW.401, DW.451, WL.403).
2. The count of each wall type based on occurrences in the floor plan.
3. Estimated lengths of each wall type in metres using the plan's printed scale (e.g. 1:100 or 1:125).
4. Descriptions of each wall type from technical detail drawings (e.g. internal composition, materials, fire rating).
5. Consolidate the data into a markdown table with the following columns: Wall Type, Count, Total Length (m), Description.

Among the provided images:
- Identify the image that shows the floor plan layout (with coloured wall tags and a scale).
- Identify the images that show typical wall detail sections with cross-sectional diagrams and descriptive notes.

Use the floor plan image to extract counts and measurements, and use the detail sheets to retrieve the construction specifications for each wall type.

Assume standard architectural conventions and return clear, technically accurate results in a structured format.
"""

user_prompt = """
Analyze the provided architectural drawings. Identify, count, and estimate the lengths of all wall types from the image that contains the full floor plan. Then extract technical descriptions for each wall type by cross-referencing the remaining images, which contain wall section details and specifications.

Follow these instructions precisely:

1. **Wall Counting & Length Estimation**:
   - Identify all unique wall type tags (e.g., SW.401, WL.403, DW.451) in the floor plan.
   - Count how many times each wall type appears.
   - Estimate the total length of each wall type in meters using the drawing scale (typically 1:100 or 1:125; use what's visible).
   - Return the results in a markdown table with columns: `Wall Type | Count | Estimated Total Length (m)`.

2. **Wall Description Extraction**:
   - Match each identified wall type to its corresponding section drawing and description found in the technical detail sheets.
   - Extract and summarise the construction specification for each type into a short paragraph.
   - Add these to a second markdown table: `Wall Type | Description`.

3. **Output Format**:
   - Format everything as clean markdown.
   - Output in the following order:
     - Header with the analysis title and timestamp
     - Table of wall counts and lengths
     - Table of wall type descriptions
     - Summary paragraph of key findings or notable walls

Only include wall types that appear in the floor plan and have a matching description in the technical detail sheets. Ensure that wall tag naming is preserved exactly as shown on the drawings.
"""
# Collect output
all_outputs = []

# Process each image
for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        print(f"🔍 Analyzing {filename}...")

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=2000
        )

        result = response.choices[0].message.content
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
print("📤 Final Output:\n")
# print(full_report)
console = Console()
console.print(full_report, markup=False)
