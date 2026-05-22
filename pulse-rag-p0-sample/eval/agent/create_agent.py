"""Provision (or update) the Pulse RAG eval agent in Microsoft Foundry.

The agent has exactly one tool: ``search_devices`` (a Function Tool wrapping
``app/retrieval.py``). This is intentional — keeping the toolset small makes
``tool_call_accuracy`` and ``tool_selection`` scores meaningful.

Run once before ``run_eval.py``. Re-running is safe: if an agent with the
configured name already exists it is updated in place.

The agent's name + ID are persisted to ``eval/agent/.eval-agent.json`` so the
eval runner can find it without re-creating.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_env, project_client, require_env  # noqa: E402
from agent.search_devices_tool import (  # noqa: E402
    SEARCH_DEVICES_TOOL_DEFINITION,
)


AGENT_INSTRUCTIONS = (
    "You are the Pulse RAG assistant. You answer questions about devices in "
    "a customer's audio/video estate (touch panels, displays, cameras, DSPs, "
    "sensors). For any question that references a specific device, room, "
    "building, capability, or status, you MUST call the `search_devices` "
    "function tool first with the user's question (or a refined version) as "
    "the `query` argument. Use only the content returned by the tool to "
    "ground your answer. If the tool returns no relevant context, say you "
    "do not know and suggest what device or room to ask about — do not make "
    "up devices. When you answer, cite source ids in parentheses, e.g. "
    "(device-003)."
)


def _write_pointer(agent_id: str, agent_name: str, model: str) -> Path:
    pointer = Path(__file__).resolve().parent / ".eval-agent.json"
    pointer.write_text(
        json.dumps(
            {"agent_id": agent_id, "agent_name": agent_name, "model": model},
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

    # Try to find an existing agent with this name. The SDK exposes
    # `list_agents`; iterate and match by name. Some SDK builds expose
    # `agents.list()` instead, so be defensive.
    existing = None
    try:
        for agent in agents.list_agents():  # type: ignore[attr-defined]
            if getattr(agent, "name", None) == agent_name:
                existing = agent
                break
    except AttributeError:
        for agent in agents.list():  # type: ignore[attr-defined]
            if getattr(agent, "name", None) == agent_name:
                existing = agent
                break

    tools = [SEARCH_DEVICES_TOOL_DEFINITION]

    if existing is not None:
        agent_id = getattr(existing, "id", None) or getattr(existing, "agent_id")
        print(f"[agent] Updating existing agent '{agent_name}' (id={agent_id}).")
        try:
            agents.update_agent(
                agent_id=agent_id,
                model=model_deployment,
                instructions=AGENT_INSTRUCTIONS,
                tools=tools,
            )
        except AttributeError:
            agents.update(  # type: ignore[attr-defined]
                agent_id=agent_id,
                model=model_deployment,
                instructions=AGENT_INSTRUCTIONS,
                tools=tools,
            )
    else:
        print(f"[agent] Creating new agent '{agent_name}' on model '{model_deployment}'.")
        created = agents.create_agent(
            model=model_deployment,
            name=agent_name,
            instructions=AGENT_INSTRUCTIONS,
            tools=tools,
        )
        agent_id = getattr(created, "id", None) or getattr(created, "agent_id")

    pointer = _write_pointer(agent_id=agent_id, agent_name=agent_name, model=model_deployment)
    print(f"[agent] Wrote pointer file: {pointer}")
    print(f"[agent] Agent id: {agent_id}")
    print(f"[agent] Set EVAL_AGENT_NAME={agent_name} in eval/.env (already required).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
