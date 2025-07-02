import os
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