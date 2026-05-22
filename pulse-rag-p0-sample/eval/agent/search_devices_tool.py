"""Function tool exposed to the eval Foundry agent.

The Pulse RAG production app calls ``app.retrieval.retrieve_context`` directly
from Flask. For evaluation we need the agent itself to decide *whether* to
call retrieval and *which query* to pass, so this module re-exports that
function as a Foundry-compatible tool.

Why a function tool (and not the Azure AI Search knowledge tool):
The Foundry built-in tool-call evaluators (``tool_call_accuracy``,
``tool_input_accuracy``, ``tool_output_utilization``, ``tool_call_success``)
and ``groundedness`` have *limited support* when the agent uses the Azure AI
Search knowledge tool. Wrapping retrieval as a user-defined Function Tool is
the workaround documented by Microsoft Learn and is what makes those
evaluators return real scores instead of pass-through results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Reuse the production retrieval implementation so the eval scores what the
# app actually does.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "app"))

from retrieval import retrieve_context  # type: ignore[import-not-found]  # noqa: E402


def search_devices(query: str, top: int = 5) -> list[dict[str, Any]]:
    """Search the pulse-device-chunks index for context relevant to ``query``.

    Args:
        query: Natural-language question from the user about a Pulse device.
        top:   Maximum number of chunks to return (1-10).

    Returns:
        List of chunks with ``sourceId``, ``title``, ``content`` (the raw
        snippet to ground the answer on) and ``score`` (relevance score).
    """
    top = max(1, min(int(top or 5), 10))
    chunks = retrieve_context(query=query, top=top)
    return [
        {
            "sourceId": chunk.source_id,
            "title": chunk.title,
            "content": chunk.content,
            "score": chunk.score,
        }
        for chunk in chunks
    ]


# Foundry expects an OpenAI-style function-tool definition. Keep this in sync
# with the ``search_devices`` signature above — the eval runner reads this
# definition and passes it to ``tool_call_accuracy`` as ``tool_definitions``.
SEARCH_DEVICES_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_devices",
        "description": (
            "Look up grounding context about the customer's Pulse devices "
            "(touch panels, displays, cameras, DSPs, sensors, etc.) from the "
            "pulse-device-chunks search index. Call this for any question "
            "that references a specific device, room, building, capability, "
            "or status; do not call it for greetings or pure meta questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question, re-phrased for search if useful.",
                },
                "top": {
                    "type": "integer",
                    "description": "How many chunks to retrieve (1-10).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}
