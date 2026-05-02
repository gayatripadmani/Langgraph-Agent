from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanOutput(BaseModel):
    planner_notes: str = Field(description="High-level reasoning about the task.")
    subquestions: list[str] = Field(default_factory=list)
    retrieval_focus: list[str] = Field(default_factory=list)


class ToolDecision(BaseModel):
    tool_name: Literal["vector_search", "metadata_lookup", "calculator"]
    tool_input: str
    rationale: str


class Citation(BaseModel):
    source: str
    chunk_id: str
    title: str
    page_number: int | None = None
    page_reference: str | None = None


class DraftAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    gaps: list[str] = Field(default_factory=list)


class CriticOutput(BaseModel):
    verdict: Literal["finalize", "revise"]
    feedback: str
    missing_information: list[str] = Field(default_factory=list)


class FinalResponse(BaseModel):
    answer: str
    citations: list[dict]
    trace: list[str]
