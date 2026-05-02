from __future__ import annotations

import ast
import operator
import re
from collections import Counter
from typing import Any

from research_assistant.config import Settings
from research_assistant.vectorstore import ensure_vectorstore, load_pdf_catalog


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def vector_search_tool(query: str, settings: Settings) -> list[dict[str, Any]]:
    store = ensure_vectorstore(settings)
    results = store.similarity_search_with_score(query, k=settings.retrieve_k)
    payload: list[dict[str, Any]] = []
    for doc, score in results:
        payload.append(
            {
                "tool": "vector_search",
                "query": query,
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )
    return payload


def metadata_lookup_tool(query: str) -> list[dict[str, Any]]:
    normalized = query.lower()
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2
    }
    matches: list[dict[str, Any]] = []
    for entry in load_pdf_catalog():
        searchable = " ".join(
            [
                str(entry["source"]),
                str(entry["title"]),
                str(entry["preview"]),
                f"{entry['page_count']} pages",
            ]
        ).lower()
        search_tokens = {
            token for token in re.findall(r"[a-z0-9]+", searchable) if len(token) > 2
        }
        if normalized in searchable or query_tokens.intersection(search_tokens):
            matches.append(
                {
                    "tool": "metadata_lookup",
                    "query": query,
                    "content": (
                        f"Document: {entry['title']}\n"
                        f"File: {entry['source']}\n"
                        f"Page count: {entry['page_count']}\n"
                        f"Preview: {entry['preview']}"
                    ),
                    "metadata": {
                        "source": str(entry["source"]),
                        "title": str(entry["title"]),
                        "chunk_id": "document-summary",
                        "page_number": 1,
                    },
                }
            )
    return matches


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](
            _eval_ast(node.left), _eval_ast(node.right)
        )
    raise ValueError("Unsupported expression")


def calculator_tool(expression: str) -> list[dict[str, Any]]:
    parsed = ast.parse(expression, mode="eval")
    result = _eval_ast(parsed.body)
    return [
        {
            "tool": "calculator",
            "query": expression,
            "content": f"Calculated result: {result:g}",
            "metadata": {
                "source": "calculator",
                "title": "Calculator",
                "chunk_id": "calculator-result",
            },
        }
    ]


def unique_citations(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    citations: list[dict[str, Any]] = []
    for item in retrieved_docs:
        metadata = item["metadata"]
        key = (metadata["source"], metadata["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "title": metadata["title"],
                "page_number": metadata.get("page_number"),
                "page_reference": metadata.get("page_reference"),
            }
        )
    return citations


def summarize_sources(retrieved_docs: list[dict[str, Any]]) -> str:
    counts = Counter(item["metadata"]["source"] for item in retrieved_docs)
    if not counts:
        return "No sources retrieved."
    return ", ".join(f"{source} ({count})" for source, count in counts.items())
