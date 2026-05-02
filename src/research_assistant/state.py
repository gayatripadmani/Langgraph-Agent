from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    conversation_history: Annotated[list[dict[str, str]], operator.add]
    subquestions: list[str]
    retrieval_focus: list[str]
    retrieved_docs: list[dict[str, Any]]
    draft_answer: dict[str, Any]
    critic_feedback: str
    missing_information: list[str]
    final_response: dict[str, Any]
    trace: list[str]
    loop_count: int
