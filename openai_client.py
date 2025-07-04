import time
from openai import AzureOpenAI

class AzureOpenAIClient:
    def __init__(self, api_key, api_version, endpoint, deployment_name):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name

    def analyze_image_with_text(self, system_prompt, user_prompt, base64_image_list, temperature=0, top_p=1, max_tokens=2000):
        """
        Accepts a list of base64 images and sends them all in a single prompt.
        """
        content = [{"type": "text", "text": user_prompt}]
        for base64_image in base64_image_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def classify_documents(self, image_data_list):
        """
        Classifies each image as 'main' or 'detail' based on its content.

        Args:
            image_data_list (list of dict): List of dicts with 'filename' and 'base64' keys.

        Returns:
            list of (filename, label): List of tuples mapping filename to label.
        """
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

        content = [{"type": "text", "text": user_prompt}]
        for img in image_data_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"} })

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    temperature=0,
                    top_p=1,
                    max_tokens=100
                )
                break  # Success!
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(5)
        else:
            print("All attempts failed.")
            return []

        labels = [line.strip().lower() for line in response.choices[0].message.content.splitlines() if line.strip()]
        return list(zip([img['filename'] for img in image_data_list], labels))