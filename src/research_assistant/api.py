from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from research_assistant.app import ResearchAssistantApp
from research_assistant.config import load_settings
from research_assistant.vectorstore import load_pdf_catalog


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    trace: list[str]
    thread_id: str


@lru_cache(maxsize=1)
def get_service() -> ResearchAssistantApp:
    return ResearchAssistantApp()


def create_api() -> FastAPI:
    api = FastAPI(
        title="LangGraph Research Assistant API",
        version="1.0.0",
    )

    @api.get("/health")
    def health() -> dict:
        settings = load_settings()
        return {
            "status": "ok",
            "collection": settings.qdrant_collection,
            "documents": len(load_pdf_catalog()),
        }

    @api.get("/documents")
    def documents() -> dict:
        return {"documents": load_pdf_catalog()}

    @api.get("/sample-queries")
    def sample_queries() -> dict:
        return {
            "queries": [
                "What are the three articles in the English language?",
                "Which document is longer, BCA-123 Basic English.pdf or Articles.pdf, and by how many pages?",
                "What is the company policy for reimbursing conference travel expenses?",
            ]
        }

    @api.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = get_service().ask(request.query, thread_id=request.thread_id)
            return AskResponse(**result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api


app = create_api()
