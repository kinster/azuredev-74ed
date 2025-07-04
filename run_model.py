import os
import base64
import time
from dotenv import load_dotenv
from openai_client import AzureOpenAIClient
from vision_ocr import AzureVisionOCRClient
from datetime import datetime
from rich.console import Console
from weasyprint import HTML
import markdown
from weasyprint import HTML, CSS
import re
from openpyxl import Workbook

# Load API credentials
load_dotenv()
openai_client = AzureOpenAIClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # api_version="2024-05-01-preview",
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
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
image_dir = "inputs/images"
output_dir = "results"
output_file = os.path.join(output_dir, "wall_report.md")
os.makedirs(output_dir, exist_ok=True)

# Prompts
system_prompt = """
You are a construction AI assistant with expertise in interpreting architectural floor plans, internal wall types, and drylining detail sheets. You specialise in UK construction standards, including cost estimation and generating Bill of Quantities (BoQ).

You are provided with:
- One main floor plan image (the first image) that shows internal wall type labels (e.g. SW.401, WL.403) and scale (e.g. 1:125)
- Additional supplementary images or documents (such as technical detail sheets) that describe the construction and material specifications of each wall type
- OCR-extracted text from the plan and supplementary documents (optional, included if available)

Your task is to:
1. Use **only the main floor plan image (the first image)** to identify and list all unique wall types (e.g. WL.401, DW.451), count how many times each type appears, and estimate the total linear length for each based on the scale.
2. Use the supplementary documents **only for reference** to match each wall type to its technical description and specifications. Do not use supplementary documents for wall counting or measurement.
3. Return a table with the following columns:
   | Wall Type | Count | Total Length (m) | Description                     |
   |-----------|-------|------------------|---------------------------------|

Next, using the wall type descriptions and estimated quantities, generate a UK-style Bill of Quantities (BoQ). For each wall type:
- Assume units are in linear metres (unless clearly area-based)
- Add sample unit rates (in £/m or £/m²) where applicable
- Return a markdown table like this:

| Item No. | Description of Work                      | Unit | Quantity | Rate (£) | Total (£) |
|----------|------------------------------------------|------|----------|----------|------------|
| 1.1      | WL.401: Wall lining with plasterboard     | m    | 35       | 40.00    | 1,400.00   |
| 1.2      | DW.451: Demountable drywall partition     | m    | 22       | 48.00    | 1,056.00   |

Finish with a **summary** section that:
- Lists any assumptions (e.g. rates were estimated, length measured via tag frequency and scale)
- States if any wall types could not be matched
- **Explicitly explains how the supplementary files/images were used in your analysis**

Ensure outputs are well-formatted, accurate, and concise.
"""

# Store final results
all_outputs = []

# Step 1: Gather images and OCR in one batch
image_data_list = []
ocr_snippets = []

for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")
        ocr_text = vision_client.extract_ocr_text(filepath)
        time.sleep(3)  # 🔁 Add delay to stay under rate limit
        image_data_list.append({"filename": filename, "base64": base64_img, "filepath": filepath})
        ocr_snippets.append(f"🖼️ {filename}\n\n{ocr_text}")

# --- NEW: Classify images to find main and supplementary ---
results = openai_client.classify_documents(image_data_list)
print(f"Classified (results) images as main or supplementary.")
main_images = [img for img, label in results if label == "main"]

if not main_images:
    raise Exception("No main image detected by classification.")
main_image_filename = main_images[0]  # Use the first main image found

# Use the main image and all supplementary images (restore previous logic)
main_image_data = next(img for img in image_data_list if img["filename"] == main_image_filename)
main_ocr_snippet = next(snippet for snippet in ocr_snippets if snippet.startswith(f"🖼️ {main_image_filename}"))
supplementary_image_data = [img for img in image_data_list if img["filename"] != main_image_filename]
supplementary_ocr_snippets = [snippet for snippet in ocr_snippets if not snippet.startswith(f"🖼️ {main_image_filename}")]
ordered_image_data_list = [main_image_data] + supplementary_image_data
ordered_ocr_snippets = [main_ocr_snippet] + supplementary_ocr_snippets

# Step 2: Build single prompt
combined_ocr_text = "\n\n".join(ordered_ocr_snippets)

print(ordered_image_data_list.__len__(), "images processed.")
# print(f"Combined OCR text from {len(ordered_ocr_snippets)} images:\n{combined_ocr_text}\n")

# Extract wall types from the main OCR snippet
def extract_wall_types_from_ocr(ocr_text):
    # Naive extraction: look for patterns like WL.401, DW.451, etc.
    # This may need to be adjusted based on actual OCR output format
    pattern = r"\b(WL|DW|SW)\.\d{3}\b"
    matches = re.findall(pattern, ocr_text)
    return list(set(matches))  # Return unique wall types

main_wall_types = extract_wall_types_from_ocr(main_ocr_snippet)
wall_types_str = ", ".join(main_wall_types)

user_prompt = f"""
You are given a set of architectural images and documents used in drylining takeoff and estimating.

- The **first image** is the main floor plan to be used for identifying and counting wall types.
- The **remaining images/documents** are supplementary and should only be used for reference (e.g., technical details, specifications). Do not use supplementary documents for wall counting or measurement.

OCR-extracted text from each image is included below for your reference.

---

{combined_ocr_text}

---

The following wall types were identified in the main floor plan: {wall_types_str}.
Only include these wall types in the BoQ. Do not add any others.

**When presenting any tables (including wall type summary and BoQ), order the rows by wall type code in ascending alphabetical order (e.g., DW.451, DW.452, SW.401, WL.401, etc.).**

Your task is to:
1. Identify all unique wall types, count, and measure **using only the main floor plan (first image)**.
2. Use supplementary documents only to clarify or describe wall types, not for counting or measurement.
3. **Only include wall types in the BoQ that are actually found and counted in the main floor plan image. Do not add wall types that are only present in supplementary documents.**
4. Return two markdown tables as before, and a summary of assumptions. Do not include any sample or example BoQ tables. Only output the BoQ table generated from the main floor plan image. There should be only one BoQ table in your output.
5. In your summary, clearly state how the supplementary files/images were used in your analysis.
"""

# Step 3: Make a single API call
result = openai_client.analyze_image_with_text(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    base64_image_list=[img["base64"] for img in ordered_image_data_list],
    temperature=0,
    top_p=1,  # Set your desired top_p value here (e.g., 1 for deterministic, <1 for more randomness)
)

# Output formatting
output = f"# 📐 Drylining Wall Type & BoQ Summary\n\n{result}"


timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
full_report = f"# 🧱 Wall Analysis Report\nGenerated on {timestamp}\n\n{result}"

# Save to user-specified output file (e.g. Markdown)
output_file = "wall_analysis_report.md"

# Save report to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\n✅ Markdown report saved to: {output_file}\n")

# Optional: save markdown file
with open("wall_analysis_report.md", "w") as f:
    f.write(full_report)

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# Save markdown (optional)
with open("results/wall_analysis_report.md", "w") as f:
    f.write(full_report)

# Convert to HTML
html_content = markdown.markdown(full_report, extensions=["tables"])
wrapped_html = f"<html><body>{html_content}</body></html>"

# Write to PDF with optional styling
HTML(string=wrapped_html).write_pdf(
    "results/wall_analysis_report.pdf",
    stylesheets=[CSS(string="""
        body { font-family: Arial, sans-serif; font-size: 12px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }
        th { background-color: #f2f2f2; }
    """)]
)

print("✅ PDF written to results/wall_analysis_report.pdf")



# Print to console at the end
console = Console()
console.print(full_report, markup=False)

def extract_boq_table(markdown_text):
    pattern = r"\| *Item No\. *\|.*?\|\n\|[-| ]+\|\n((?:\|.*\|\n?)+)"
    matches = list(re.finditer(pattern, markdown_text, re.DOTALL))
    if not matches:
        return []
    table_text = matches[-1].group(0)  # Use the last match
    rows = [
        [cell.strip() for cell in row.split("|")[1:-1]]
        for row in table_text.strip().split("\n")
        if row.startswith("|")
    ]
    return rows

# Extract BoQ table rows from the markdown report
boq_rows = extract_boq_table(full_report)
if boq_rows:
    wb = Workbook()
    ws = wb.active
    ws.title = "BoQ"
    for row in boq_rows:
        ws.append(row)
    excel_path = os.path.join(output_dir, "wall_analysis_report.xlsx")
    wb.save(excel_path)
    print(f"✅ Excel BoQ written to {excel_path}")
else:
    print("❌ Could not find BoQ table in the markdown report.")
