from __future__ import annotations

from research_assistant.config import Settings
from research_assistant.schemas import CriticOutput
from research_assistant.state import AgentState


def critic_node(state: AgentState, critic_chain, settings: Settings) -> dict:
    prompt = f"""
You are the critic agent in a research assistant workflow.
Check whether the draft directly answers the user question, cites supporting evidence,
and avoids unsupported claims. If evidence is missing, request a revision.

User query: {state["user_query"]}

Draft answer:
{state.get("draft_answer", {})}

Current loop count: {state.get("loop_count", 0)}
Maximum loops: {settings.max_critic_loops}
"""
    output: CriticOutput = critic_chain.invoke(prompt)
    return {
        "critic_feedback": output.feedback,
        "missing_information": output.missing_information,
        "loop_count": state.get("loop_count", 0) + 1,
        "trace": state.get("trace", []) + [f"critic:{output.verdict}"],
    }
