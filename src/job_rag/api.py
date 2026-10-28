"""Local FastAPI interface for retrieval and cited answer generation."""

from contextlib import asynccontextmanager
from dataclasses import replace
from threading import Lock
from typing import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from job_rag.config import Settings


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=1000, strict=True)
    top_k: int = Field(default=2, ge=1, le=5, strict=True)
    location: str | None = Field(default=None, min_length=1, max_length=300, strict=True)


class Scores(BaseModel):
    rrf: float | None
    reranker: float | None
    vector_distance: float | None
    bm25: float | None


class JobEvidence(BaseModel):
    rank: int
    job_id: str
    title: str
    company: str
    location: str | None
    level: str | None
    publication_date: str | None
    source_chunk_id: str
    evidence: str
    scores: Scores


class QueryResponse(BaseModel):
    query: str
    location_filter: str | None
    answer: str
    cited_job_ids: list[str]
    jobs: list[JobEvidence]


def build_service():
    # Load shared retrieval models once, reserving GPU memory for local Ollama.
    from job_rag.answers import GroundedAnswerService
    from job_rag.ollama_local import OllamaGenerator
    from job_rag.pipeline import JobSearch

    settings = replace(Settings(), device="cpu")
    return GroundedAnswerService(JobSearch(settings), OllamaGenerator(settings))


def create_app(service_factory: Callable = build_service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory()
        app.state.inference_lock = Lock()
        try:
            yield
        finally:
            app.state.service = None

    app = FastAPI(
        title="Job Search RAG", version="0.1.0", lifespan=lifespan,
        description="Hybrid job retrieval and cited answers using local models.",
    )

    @app.get("/health")
    def health():
        # Readiness means retrieval loaded; generation is checked when queried.
        return {"status": "ready", "generation": "local Ollama; checked on query"}

    @app.post("/api/query", response_model=QueryResponse)
    def query_jobs(body: QueryRequest):
        # Reject overlapping inference rather than multiply model memory usage.
        lock = app.state.inference_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "The local pipeline is busy. Retry shortly.",
                                headers={"Retry-After": "5"})
        try:
            result = app.state.service.answer(
                body.query, top_k=body.top_k, location=body.location,
            )
            return QueryResponse(
                query=body.query, location_filter=body.location, **result,
            )
        except ValidationError:
            raise HTTPException(500, "The local pipeline returned an invalid response.") from None
        except ValueError:
            raise HTTPException(
                422, "The query or evidence exceeds model limits. Shorten the query or reduce top_k.",
            ) from None
        except (RuntimeError, OSError):
            raise HTTPException(
                503, "The local answer pipeline could not complete. Check Ollama and local model files.",
            ) from None
        except Exception:
            # Do not expose exceptions, local paths, or source text in error bodies.
            raise HTTPException(500, "The local pipeline encountered an unexpected error.") from None
        finally:
            lock.release()

    return app


app = create_app()
