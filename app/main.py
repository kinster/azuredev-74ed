# main.py
from fastapi import FastAPI
from app.legend_agent import legend_router
from app.wall_type_explainer_agent import router as explainer_router

app = FastAPI()
app.include_router(legend_router, prefix="/legend")          # /legend/detect
app.include_router(explainer_router, prefix="/explainer")    # /explainer/wall-types
