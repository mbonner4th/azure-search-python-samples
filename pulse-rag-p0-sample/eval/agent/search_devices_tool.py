"""Built-in Azure AI Search tool for the Pulse RAG eval agent.

We previously wrapped the public ``/api/search/preview`` Function endpoint
as an :class:`~azure.ai.projects.models.OpenApiTool` because earlier
hosted Foundry agents could not invoke local Python ``FunctionTool``\\s.
Now that the agent runs on ``gpt-5.4-mini``, Foundry exposes the
**native** :class:`~azure.ai.projects.models.AzureAISearchTool` — the
model talks to Azure AI Search directly via the project's
CognitiveSearch connection, with no Function App hop. This is faster,
removes a moving part, and gives the model semantic/vector hybrid
retrieval out of the box.

Required env (set in ``eval/.env``):

* ``AZURE_SEARCH_CONNECTION_ID`` — full ARM id of the project's
  CognitiveSearch connection, e.g.
  ``/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>/projects/<proj>/connections/<conn>``.
  Discover it via ``client.connections.list()``.
* ``SearchIndexName`` — search index name (``pulse-device-chunks``).
* ``AZURE_SEARCH_QUERY_TYPE`` (optional) — one of ``simple``,
  ``semantic``, ``vector``, ``vector_simple_hybrid``,
  ``vector_semantic_hybrid``. Defaults to ``vector_semantic_hybrid``
  to match the production retrieval pipeline.
* ``AZURE_SEARCH_TOP_K`` (optional int) — chunks to retrieve. Default 5.
"""

from __future__ import annotations

import os
from typing import Any

from azure.ai.projects.models import (
    AISearchIndexResource,
    AzureAISearchQueryType,
    AzureAISearchTool,
    AzureAISearchToolResource,
)


# The hosted built-in tool surfaces under this type name in agent runs.
TOOL_NAME = "azure_ai_search"
TOOL_DESCRIPTION = (
    "Built-in Azure AI Search retrieval over the pulse-device-chunks "
    "index. Returns grounding chunks about the customer's Pulse devices "
    "(touch panels, displays, cameras, DSPs, sensors). The model issues "
    "a search query and receives matching documents to ground its answer."
)


def build_search_tool() -> AzureAISearchTool:
    """Build the native :class:`AzureAISearchTool` for the agent definition."""
    connection_id = os.environ.get("AZURE_SEARCH_CONNECTION_ID")
    if not connection_id:
        raise SystemExit(
            "Missing required environment variable: AZURE_SEARCH_CONNECTION_ID. "
            "Set it to the full ARM id of the project's CognitiveSearch "
            "connection (discover with client.connections.list())."
        )
    index_name = os.environ.get("SearchIndexName") or os.environ.get(
        "AZURE_SEARCH_INDEX_NAME"
    )
    if not index_name:
        raise SystemExit(
            "Missing required environment variable: SearchIndexName "
            "(e.g. pulse-device-chunks)."
        )

    query_type_str = (
        os.environ.get("AZURE_SEARCH_QUERY_TYPE") or "vector_semantic_hybrid"
    ).lower()
    try:
        query_type = AzureAISearchQueryType(query_type_str)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid AZURE_SEARCH_QUERY_TYPE={query_type_str!r}. Valid: "
            "simple, semantic, vector, vector_simple_hybrid, vector_semantic_hybrid."
        ) from exc

    top_k = int(os.environ.get("AZURE_SEARCH_TOP_K") or 5)

    return AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=connection_id,
                    index_name=index_name,
                    query_type=query_type,
                    top_k=top_k,
                )
            ],
        )
    )


# Tool definition used by ``tool_call_accuracy`` / ``tool_selection`` /
# ``tool_input_accuracy`` evaluators. The native Azure AI Search tool is
# invoked by the model with a single ``query`` string. We declare it in
# the FLAT ``{name, description, parameters}`` shape the evaluators
# require — the OpenAI-nested ``{type:"function", function:{...}}`` shape
# is rejected with "Each tool definitions must contain a 'name' field."
SEARCH_DEVICES_TOOL_DEFINITION: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query passed to Azure AI Search.",
            },
        },
        "required": ["query"],
    },
}
