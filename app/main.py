from fastapi import FastAPI
from app.legend_agent import app as legend_app
from app.wall_type_explainer_agent import router as explainer_router

app = FastAPI()
app.mount("/legend", legend_app)  # Now: /legend/detect/
app.include_router(explainer_router)  # Now: /explain-wall-types
