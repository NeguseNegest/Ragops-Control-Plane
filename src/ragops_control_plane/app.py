from fastapi import FastAPI
from pydantic import BaseModel

from ragops_control_plane import __version__


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAGOps Control Plane",
        version=__version__,
        summary="Evaluation-gated control plane for RAG systems.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app


app = create_app()
