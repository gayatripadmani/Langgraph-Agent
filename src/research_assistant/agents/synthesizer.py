from __future__ import annotations

from research_assistant.schemas import DraftAnswer
from research_assistant.state import AgentState
from research_assistant.tools import unique_citations


def synthesizer_node(state: AgentState, synthesis_chain) -> dict:
    retrieved_docs = state.get("retrieved_docs", [])
    source_bundle = "\n\n".join(
        [
            (
                f"Source: {item['metadata']['source']} | "
                f"Chunk: {item['metadata']['chunk_id']} | "
                f"Title: {item['metadata']['title']}\n"
                f"Subquestion: {item.get('subquestion', '')}\n"
                f"Content: {item['content']}"
            )
            for item in retrieved_docs
        ]
    )
    prompt = f"""
You are the synthesizer agent.
Write a grounded answer to the user's question using only the retrieved evidence.
If the corpus does not support the claim, say so clearly.

User query: {state["user_query"]}

Retrieved evidence:
{source_bundle or "No evidence retrieved."}
"""
    output: DraftAnswer = synthesis_chain.invoke(prompt)
    draft_payload = output.model_dump()
    if not draft_payload["citations"]:
        draft_payload["citations"] = unique_citations(retrieved_docs)
    return {
        "draft_answer": draft_payload,
        "trace": state.get("trace", []) + ["synthesizer:draft"],
    }
