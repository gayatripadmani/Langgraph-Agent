from __future__ import annotations

from research_assistant.config import Settings
from research_assistant.schemas import ToolDecision
from research_assistant.state import AgentState
from research_assistant.tools import (
    calculator_tool,
    metadata_lookup_tool,
    vector_search_tool,
)


def retriever_node(
    state: AgentState,
    decision_chain,
    settings: Settings,
) -> dict:
    retrieved_docs: list[dict] = []
    trace: list[str] = []
    for question in state.get("subquestions", []):
        prompt = f"""
You are a retriever agent that can choose one tool for each subquestion.
Available tools:
- vector_search: semantic search over chunked research documents.
- metadata_lookup: direct lookup by document name, title, page count, or keyword.
- calculator: arithmetic only, use when the question needs counting or math.

Choose the best single tool for this subquestion and provide the exact input.

Subquestion: {question}
"""
        decision: ToolDecision = decision_chain.invoke(prompt)
        if decision.tool_name == "vector_search":
            tool_output = vector_search_tool(decision.tool_input, settings)
        elif decision.tool_name == "metadata_lookup":
            tool_output = metadata_lookup_tool(decision.tool_input)
        else:
            tool_output = calculator_tool(decision.tool_input)

        for item in tool_output:
            item["subquestion"] = question
            item["rationale"] = decision.rationale
        retrieved_docs.extend(tool_output)
        trace.append(f"retriever:{decision.tool_name}")
    return {
        "retrieved_docs": retrieved_docs,
        "trace": state.get("trace", []) + trace,
    }
