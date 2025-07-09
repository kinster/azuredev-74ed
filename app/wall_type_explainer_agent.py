from fastapi import APIRouter, Request
from pydantic import BaseModel
import os
from string import Template
from openai_client import AzureOpenAIClient

router = APIRouter()

openai_client = AzureOpenAIClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
)
class WallCodeRequest(BaseModel):
    codes: list[str]

@router.post("/explain-wall-types")
async def explain_wall_types(request: WallCodeRequest):
    codes = request.codes

    print(codes)

    system_prompt = (
        "You are a helpful assistant with expert knowledge of construction drawings, "
        "drylining systems, and architectural legends. When provided with internal wall codes "
        "(e.g., DW.451, WL.401), return their descriptions based on construction industry standards. "
        "Be concise and specific."
    )

    template = Template("Here is a list of wall codes:\n$codes\n\nPlease tell me what type of wall each one is based on its code.")
    user_prompt = template.substitute(codes=", ".join(codes))

    result = openai_client.ask_with_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0,
        top_p=1
    )

    return {"explanations": result}
