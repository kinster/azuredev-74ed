from fastapi import APIRouter, Request
from pydantic import BaseModel
import os
from string import Template
from openai_client import AzureOpenAIClient
import json

router = APIRouter()

openai_client = AzureOpenAIClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
)
class WallCodeRequest(BaseModel):
    wallCodes: list[str]

@router.post("/explain-wall-types")
async def explain_wall_types(request: WallCodeRequest):
    wallCodes = request.wallCodes

    print(wallCodes)

    system_prompt = (
        "You are a helpful assistant with expert knowledge of construction drawings, "
        "drylining systems, and architectural legends. When provided with internal wall codes "
        "(e.g., DW.451, WL.401), respond with a short JSON array. Each entry should contain the wall 'code' "
        "and a short 'description' such as 'dry wall' or 'shaft wall'. Do not include explanations or formatting—just output valid JSON."
    )

    template = Template(
        "Here is a list of wall codes:\n$wallCodes\n\n"
        "Return a JSON array like this: [{\"code\": \"DW.451\", \"description\": \"dry wall\"}, ...]"
    )

    user_prompt = template.substitute(wallCodes=", ".join(wallCodes))

    result = openai_client.ask_with_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0,
        top_p=1
    )

    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse OpenAI result"}

    return parsed_result
