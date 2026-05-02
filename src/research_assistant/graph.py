from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from research_assistant.agents.critic import critic_node
from research_assistant.agents.planner import planner_node
from research_assistant.agents.retriever import retriever_node
from research_assistant.agents.synthesizer import synthesizer_node
from research_assistant.config import Settings, load_settings
from research_assistant.llm import build_chat_model
from research_assistant.schemas import CriticOutput, DraftAnswer, PlanOutput, ToolDecision
from research_assistant.state import AgentState


def build_graph(settings: Settings | None = None):
    settings = settings or load_settings()
    llm = build_chat_model(settings)
    planner_chain = llm.with_structured_output(PlanOutput)
    decision_chain = llm.with_structured_output(ToolDecision)
    synthesis_chain = llm.with_structured_output(DraftAnswer)
    critic_chain = llm.with_structured_output(CriticOutput)

    graph = StateGraph(AgentState)
    graph.add_node("planner", lambda state: planner_node(state, planner_chain))
    graph.add_node(
        "retriever",
        lambda state: retriever_node(state, decision_chain, settings),
    )
    graph.add_node(
        "synthesizer",
        lambda state: synthesizer_node(state, synthesis_chain),
    )
    graph.add_node("critic", lambda state: critic_node(state, critic_chain, settings))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    def route_after_critic(state: AgentState) -> str:
        if state.get("loop_count", 0) >= settings.max_critic_loops:
            return "finalize"
        verdict = state["trace"][-1].split(":")[-1]
        if verdict == "finalize":
            return "finalize"
        return "planner"

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "planner": "planner",
            "finalize": END,
        },
    )

    return graph.compile(checkpointer=InMemorySaver())
