from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.app import ResearchAssistantApp


QUERIES = [
    "What are the three articles in the English language?",
    (
        "Which document is longer, BCA-123 Basic English.pdf or Articles.pdf, and by "
        "how many pages?"
    ),
    "What is the company policy for reimbursing conference travel expenses?",
]


def main() -> None:
    app = ResearchAssistantApp()
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    thread_id = "practical-eval"
    for query in QUERIES:
        results.append({"query": query, "result": app.ask(query, thread_id=thread_id)})
    output_path = out_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
