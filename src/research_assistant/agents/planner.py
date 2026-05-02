from __future__ import annotations

from research_assistant.schemas import PlanOutput
from research_assistant.state import AgentState


def planner_node(state: AgentState, planner_chain) -> dict:
    prior_turns = state.get("conversation_history", [])
    critic_feedback = state.get("critic_feedback", "")
    prompt = f"""
You are the planner agent in a LangGraph research workflow.
Break the user request into subquestions that can be answered with the available local corpus.
Use previous conversation turns when the user is asking a follow-up.

Conversation history:
{prior_turns}

Current user query:
{state["user_query"]}

Critic feedback to address, if any:
{critic_feedback or "None"}
"""
    output: PlanOutput = planner_chain.invoke(prompt)
    return {
        "subquestions": output.subquestions or [state["user_query"]],
        "retrieval_focus": output.retrieval_focus,
        "retrieved_docs": [],
        "draft_answer": {},
        "final_response": {},
        "trace": state.get("trace", [])
        + [f"planner:{len(output.subquestions or [state['user_query']])}-subquestions"],
    }
