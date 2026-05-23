from fastapi import FastAPI

from api.routers import health, prediction

app = FastAPI(
    title="Fetal Health ML Pipeline API",
    version="1.0.0",
    description="REST API for fetal health classification predictions.",
)

app.include_router(health.router)
app.include_router(prediction.router)
