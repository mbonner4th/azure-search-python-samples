"""Provision (or version-bump) the Pulse RAG eval agent in Microsoft Foundry.

Uses the ``azure-ai-projects`` 2.1.0 *hosted agents* API
(``client.agents.create_version`` + :class:`PromptAgentDefinition`). The
agent has exactly one tool: the built-in :class:`AzureAISearchTool`,
wired to the project's CognitiveSearch connection and the
``pulse-device-chunks`` index. Keeping the toolset to a single tool
makes ``tool_call_accuracy`` / ``tool_selection`` / ``tool_input_accuracy``
scores meaningful.

The agent runs on ``gpt-5.4-mini``, which natively supports the
``azure_ai_search`` built-in tool — no Function App / OpenAPI hop
required. Earlier versions of this agent used an ``OpenApiTool`` wrapping
``POST /api/search/preview`` for that reason; see git history.

Run once before ``run_eval.py``. Re-running creates a new version under
the same agent name (Foundry hosted-agents are immutable per-version).

Writes ``eval/agent/.eval-agent.json`` with the agent name + latest
version so the runner (and the user) know which version to evaluate
against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from azure.core.exceptions import ResourceNotFoundError  # noqa: E402
from azure.ai.projects.models import PromptAgentDefinition  # noqa: E402

from common import load_env, project_client, require_env  # noqa: E402
from agent.search_devices_tool import build_search_tool  # noqa: E402


AGENT_INSTRUCTIONS = (
    "You are the Pulse RAG assistant. You answer questions about devices in "
    "a customer's audio/video estate (touch panels, displays, cameras, DSPs, "
    "sensors). You have a built-in Azure AI Search retrieval tool wired to "
    "the `pulse-device-chunks` index. For any question that references a "
    "specific device, room, building, capability, or status, you MUST issue "
    "an Azure AI Search query first (use the user's question, or a refined "
    "version, as the search query). Use only the content returned by the "
    "search tool to ground your answer. If the search returns no relevant "
    "context, say you do not know and suggest what device or room to ask "
    "about — do not make up devices. When you answer, cite source ids in "
    "parentheses, e.g. (device-003). Do not issue a search for greetings or "
    "pure meta questions about yourself."
)


def _write_pointer(agent_name: str, version: str, model: str) -> Path:
    pointer = Path(__file__).resolve().parent / ".eval-agent.json"
    pointer.write_text(
        json.dumps(
            {"agent_name": agent_name, "version": version, "model": model},
            indent=2,
        )
    )
    return pointer


def main() -> int:
    load_env()
    agent_name = require_env("EVAL_AGENT_NAME")
    model_deployment = require_env("EVAL_AGENT_MODEL_DEPLOYMENT_NAME")

    client = project_client()
    agents = client.agents

    tool = build_search_tool()
    definition = PromptAgentDefinition(
        model=model_deployment,
        instructions=AGENT_INSTRUCTIONS,
        tools=[tool],
    )

    # Check whether the agent already exists; create_version will bump the
    # version automatically when the agent name exists.
    try:
        existing = agents.get(agent_name)
        print(
            f"[agent] Agent '{agent_name}' already exists "
            f"(latest_version={getattr(existing, 'latest_version', '?')}). "
            "Creating a new version."
        )
    except ResourceNotFoundError:
        print(f"[agent] Creating new agent '{agent_name}' on model '{model_deployment}'.")

    result = agents.create_version(
        agent_name=agent_name,
        definition=definition,
        description="Pulse RAG eval agent (built-in Azure AI Search tool over pulse-device-chunks).",
    )
    version = str(getattr(result, "version", None) or getattr(result, "agent_version", "1"))
    print(f"[agent] Created version: {version}")

    pointer = _write_pointer(agent_name=agent_name, version=version, model=model_deployment)
    print(f"[agent] Wrote pointer file: {pointer}")
    print(f"[agent] Update eval/.env: EVAL_AGENT_VERSION={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
