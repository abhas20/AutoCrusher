from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.state as state
from app.routes.mitigation import router as mitigation_router
from app.routes.prediction import router as prediction_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    state.load_model()
    yield


app = FastAPI(
    title="Bias-Aware Model Serving & Detection API",
    description="FastAPI application to serve predictions and run Agentic fairness workflows.",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(prediction_router)
app.include_router(mitigation_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
