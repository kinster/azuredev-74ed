from openai import AzureOpenAI

class AzureOpenAIClient:
    def __init__(self, api_key, api_version, endpoint, deployment_name):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name

    def analyze_image_with_text(self, system_prompt, user_prompt, base64_image):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
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
        return response.choices[0].message.content

    def classify_documents(self, ocr_snippets):
        """
        Classifies each document as 'main' or 'detail' based on its OCR text.

        Args:
            ocr_snippets (list of str): List of OCR-extracted texts.

        Returns:
            list of str: List of labels ('main' or 'detail') for each document.
        """
        prompt = (
            "You are given OCR-extracted text from a set of construction project documents. "
            "For each document, classify it as either a 'main document' (e.g., floor plan, overall drawing) "
            "or an 'additional detail document' (e.g., technical detail sheet, specification, section). "
            "Return a list of labels ('main' or 'detail') in the same order as the input, one per line.\n\n"
        )
        for i, snippet in enumerate(ocr_snippets):
            prompt += f"Document {i+1}:\n{snippet}\n---\n"
        prompt += "\nLabels:"

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for document classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        labels = [line.strip().lower() for line in response.choices[0].message.content.splitlines() if line.strip()]
        return labels