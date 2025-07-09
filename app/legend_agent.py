from fastapi import FastAPI, APIRouter, Request
from .legend_detection import extract_wall_codes
import os

legend_router = APIRouter()

app = FastAPI()

@legend_router.post("/detect")
async def detect(request: Request):
    data = await request.json()
    base64_img = data["base64"]
    endpoint = os.environ["AZURE_DI_ENDPOINT"]
    key = os.environ["AZURE_DI_KEY"]
    wall_codes = extract_wall_codes(base64_img, endpoint, key)
    return {"wallCodes": wall_codes}
