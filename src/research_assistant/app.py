from __future__ import annotations

import json
from uuid import uuid4

from research_assistant.config import load_settings
from research_assistant.graph import build_graph
from research_assistant.tools import unique_citations
from research_assistant.vectorstore import ensure_vectorstore


class ResearchAssistantApp:
    def __init__(self) -> None:
        self.settings = load_settings()
        ensure_vectorstore(self.settings)
        self.graph = build_graph(self.settings)

    def ask(self, query: str, thread_id: str | None = None) -> dict:
        thread_id = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            {
                "user_query": query,
                "conversation_history": [{"role": "user", "content": query}],
                "loop_count": 0,
                "trace": [],
            },
            config=config,
        )
        response = {
            "answer": result.get("draft_answer", {}).get(
                "answer",
                "I could not produce an answer.",
            ),
            "citations": result.get("draft_answer", {}).get("citations")
            or unique_citations(result.get("retrieved_docs", [])),
            "trace": result.get("trace", []),
            "thread_id": thread_id,
        }
        self.graph.update_state(
            config,
            {
                "conversation_history": [
                    {"role": "assistant", "content": response["answer"]}
                ]
            },
        )
        return response


def main() -> None:
    app = ResearchAssistantApp()
    thread_id: str | None = None
    print("LangGraph Research Assistant. Type 'exit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        result = app.ask(user_input, thread_id=thread_id)
        thread_id = result["thread_id"]
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
