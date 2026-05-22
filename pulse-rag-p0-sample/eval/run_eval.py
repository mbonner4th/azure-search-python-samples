"""Run the Pulse RAG eval locally against the hosted Foundry agent.

Why local instead of ``client.evals.create``?
    ``azure-ai-projects==2.1.0`` ships an ``AIProjectClient`` that exposes
    ``agents``, ``datasets``, ``connections``, ``deployments``,
    ``evaluation_rules`` and ``beta`` namespaces — but **no** ``evals``
    namespace. The cloud "agent batch eval" surface is currently only
    reachable through the Microsoft Foundry MCP server, which (in this
    environment) is pinned to a different tenant. We therefore drive the
    same evaluators locally: invoke the hosted agent via the Responses API,
    capture the tool calls + answer, and score each row with the
    ``azure-ai-evaluation`` builtin evaluators.

Inputs (all from ``eval/.env``)::

    PROJECT_ENDPOINT
    EVAL_AGENT_NAME, EVAL_AGENT_VERSION
    AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME   (e.g. eval-judge-gpt5mini)

Dataset:
    ``eval/data/pulse_eval_questions.jsonl`` (the same file uploaded as
    ``EVAL_DATASET_NAME`` v``EVAL_DATASET_VERSION``). Each row has
    ``query``, ``ground_truth``, ``expected_source_ids`` and
    ``expected_tool``.

Output:
    ``eval/results/run-<UTC-timestamp>.json`` with per-row evaluator
    outputs and aggregate means; also printed to stdout.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import eval_root, load_env, project_client, require_env  # noqa: E402
from agent.search_devices_tool import (  # noqa: E402
    SEARCH_DEVICES_TOOL_DEFINITION,
    TOOL_NAME,
)


# The built-in ``azure_ai_search`` tool surfaces in the Responses API as
# items of type ``azure_ai_search_call`` (issued by the model) and
# ``azure_ai_search_call_output`` (results returned by Search). We also
# keep the older ``openapi_*`` / ``function_*`` types for backward
# compatibility with earlier agent versions.
HOSTED_TOOL_CALL_TYPES = {
    "azure_ai_search_call",
    "openapi_call",
    "function_call",
}
HOSTED_TOOL_OUTPUT_TYPES = {
    "azure_ai_search_call_output",
    "openapi_call_output",
    "function_call_output",
}


def _account_endpoint_from_project(project_endpoint: str) -> str:
    """Strip the ``/api/projects/<name>`` suffix so we get the bare account
    endpoint to use as ``azure_endpoint`` for the judge model."""
    parsed = urlparse(project_endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def _judge_model_config() -> dict[str, str]:
    project_endpoint = require_env("PROJECT_ENDPOINT")
    deployment = require_env("AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_AI_JUDGE_API_VERSION", "2024-12-01-preview")
    return {
        "azure_endpoint": _account_endpoint_from_project(project_endpoint),
        "azure_deployment": deployment,
        "api_version": api_version,
    }


def _load_dataset() -> list[dict[str, Any]]:
    path = eval_root() / "data" / "pulse_eval_questions.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_context_from_output(payload: Any) -> str:
    """Best-effort extraction of grounding text from a tool call_output.

    Different hosted tools structure their output differently:

    * ``openapi_call_output`` puts a JSON string in ``output``.
    * ``azure_ai_search_call_output`` typically returns a list of
      ``results``/``documents`` either as a structured field or as a JSON
      string in ``output``. We walk a few common shapes and concatenate
      every text-bearing field we find.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        # Try to parse as JSON to pull out result chunks; if not JSON, use raw.
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        return _extract_context_from_output(parsed)
    if isinstance(payload, list):
        return "\n\n".join(_extract_context_from_output(x) for x in payload).strip()
    if isinstance(payload, dict):
        # Common containers for hosted-tool results.
        for key in ("results", "documents", "data", "chunks", "value"):
            if key in payload:
                return _extract_context_from_output(payload[key])
        # Common text fields on a single chunk.
        parts: list[str] = []
        for key in ("content", "text", "chunk", "snippet", "body"):
            v = payload.get(key)
            if isinstance(v, str) and v:
                parts.append(v)
        if parts:
            return "\n".join(parts)
        # Fall back to the JSON dump so groundedness has *something*.
        return json.dumps(payload, default=str)
    return str(payload)


def _invoke_agent(openai_client: Any, query: str) -> dict[str, Any]:
    """Invoke the hosted agent once and normalize the response.

    Returns a dict with ``output_text``, ``tool_calls`` (OpenAI-style list
    of ``{type, tool_call_id, name, arguments}``) and ``context`` (the
    concatenated tool-output payloads, used as ``context`` by the
    Groundedness evaluator).
    """
    resp = openai_client.responses.create(input=query)

    tool_calls: list[dict[str, Any]] = []
    context_chunks: list[str] = []

    for item in resp.output:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        kind = d.get("type") or ""
        if kind in HOSTED_TOOL_CALL_TYPES:
            raw_args = d.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                # Built-in tools (e.g. azure_ai_search) often expose the
                # search query on a dedicated field instead of ``arguments``.
                query_str = d.get("query") or d.get("input") or ""
                args = {"query": query_str} if query_str else {}
            # Normalize the tool name: built-in tools don't always carry a
            # ``name`` field, but the type itself identifies them.
            name = d.get("name") or kind.removesuffix("_call")
            tool_calls.append(
                {
                    "type": "tool_call",
                    "tool_call_id": d.get("call_id") or d.get("id"),
                    "name": name,
                    "arguments": args,
                }
            )
        elif kind in HOSTED_TOOL_OUTPUT_TYPES:
            # azure_ai_search_call_output may carry structured ``results``
            # at the top level; openapi_call_output carries the response
            # JSON string in ``output``.
            payload = (
                d.get("results")
                or d.get("output")
                or d.get("content")
                or d
            )
            extracted = _extract_context_from_output(payload)
            if extracted:
                context_chunks.append(extracted)

    return {
        "output_text": resp.output_text or "",
        "tool_calls": tool_calls,
        "context": "\n\n".join(context_chunks) if context_chunks else "",
    }


def _mean(values: list[float]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(nums) if nums else None


def _coerce_score(metric_result: Any, score_key: str) -> float | None:
    if not isinstance(metric_result, dict):
        return None
    val = metric_result.get(score_key)
    if isinstance(val, (int, float)):
        return float(val)
    # Some evaluators report ``<metric>`` as a string label and the numeric
    # value as ``<metric>_score``.
    val = metric_result.get(f"{score_key}_score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def main() -> int:
    load_env()
    agent_name = require_env("EVAL_AGENT_NAME")
    agent_version = require_env("EVAL_AGENT_VERSION")  # informational only

    # Build the openai client bound to the hosted agent.
    client = project_client(allow_preview=True)
    openai_client = client.get_openai_client(agent_name=agent_name)
    print(f"[eval] agent={agent_name} v{agent_version}")
    print(f"[eval] agent endpoint: {openai_client.base_url}")

    # Build judge-backed evaluators. azure-ai-evaluation uses model_config
    # to talk to the judge deployment via Azure OpenAI; pass our
    # DefaultAzureCredential so we don't need an API key.
    from azure.identity import DefaultAzureCredential
    from azure.ai.evaluation import (
        GroundednessEvaluator,
        IntentResolutionEvaluator,
        RelevanceEvaluator,
        TaskAdherenceEvaluator,
        ToolCallAccuracyEvaluator,
    )

    model_config = _judge_model_config()
    credential = DefaultAzureCredential()
    evaluators = {
        "relevance": RelevanceEvaluator(model_config, credential=credential),
        "groundedness": GroundednessEvaluator(model_config, credential=credential),
        "intent_resolution": IntentResolutionEvaluator(model_config, credential=credential),
        "task_adherence": TaskAdherenceEvaluator(model_config, credential=credential),
        "tool_call_accuracy": ToolCallAccuracyEvaluator(model_config, credential=credential),
    }

    # ``SEARCH_DEVICES_TOOL_DEFINITION`` is already in the FLAT shape that
    # ``azure-ai-evaluation`` requires (top-level ``name`` / ``description``
    # / ``parameters``). The built-in tool surfaces in tool_calls as
    # ``azure_ai_search`` — which matches ``TOOL_NAME`` directly, no
    # ``<tool>_<operationId>`` double-prefix.
    hosted_tool_definition: dict[str, Any] = SEARCH_DEVICES_TOOL_DEFINITION

    rows = _load_dataset()
    print(f"[eval] dataset rows: {len(rows)}")

    per_row: list[dict[str, Any]] = []
    started = time.time()
    for idx, row in enumerate(rows, start=1):
        query: str = row["query"]
        ground_truth: str = row.get("ground_truth", "")
        print(f"\n[eval] row {idx}/{len(rows)}: {query[:80]}")

        try:
            invocation = _invoke_agent(openai_client, query)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! agent invoke failed: {exc!r}")
            per_row.append(
                {
                    "query": query,
                    "ground_truth": ground_truth,
                    "error": f"agent_invoke: {exc!r}",
                }
            )
            continue

        response_text = invocation["output_text"]
        tool_calls = invocation["tool_calls"]
        context_text = invocation["context"]

        # Tool calls without a context payload mean the agent answered from
        # prior knowledge; supply at least a placeholder so groundedness
        # doesn't crash.
        context_for_eval = context_text or "(no tool output)"

        scores: dict[str, Any] = {}

        def _safe(name: str, fn: Any) -> None:
            try:
                scores[name] = fn()
            except Exception as exc:  # noqa: BLE001
                scores[name] = {"error": repr(exc)}

        _safe(
            "relevance",
            lambda: evaluators["relevance"](query=query, response=response_text),
        )
        _safe(
            "groundedness",
            lambda: evaluators["groundedness"](
                query=query, response=response_text, context=context_for_eval
            ),
        )
        _safe(
            "intent_resolution",
            lambda: evaluators["intent_resolution"](
                query=query,
                response=response_text,
                tool_calls=tool_calls or None,
                tool_definitions=[hosted_tool_definition],
            ),
        )
        _safe(
            "task_adherence",
            lambda: evaluators["task_adherence"](
                query=query,
                response=response_text,
                tool_calls=tool_calls or None,
                tool_definitions=[hosted_tool_definition],
            ),
        )
        if tool_calls:
            _safe(
                "tool_call_accuracy",
                lambda: evaluators["tool_call_accuracy"](
                    query=query,
                    tool_calls=tool_calls,
                    tool_definitions=[hosted_tool_definition],
                    response=response_text,
                ),
            )
        else:
            scores["tool_call_accuracy"] = {"skipped": "no tool calls"}

        for name, result in scores.items():
            if isinstance(result, dict) and "error" in result:
                print(f"  - {name}: ERROR {result['error']}")
                continue
            if isinstance(result, dict) and "skipped" in result:
                print(f"  - {name}: skipped ({result['skipped']})")
                continue
            score = _coerce_score(result, name) if isinstance(result, dict) else None
            print(f"  - {name}: {score}")

        per_row.append(
            {
                "query": query,
                "ground_truth": ground_truth,
                "expected_source_ids": row.get("expected_source_ids"),
                "expected_tool": row.get("expected_tool"),
                "response": response_text,
                "tool_calls": tool_calls,
                "context_preview": context_text[:800],
                "scores": scores,
            }
        )

    elapsed = time.time() - started

    # Aggregate per-criterion means.
    aggregate: dict[str, float | None] = {}
    for criterion in (
        "relevance",
        "groundedness",
        "intent_resolution",
        "task_adherence",
        "tool_call_accuracy",
    ):
        per_score = [
            _coerce_score(r["scores"].get(criterion), criterion)
            for r in per_row
            if "scores" in r
        ]
        aggregate[criterion] = _mean([v for v in per_score if v is not None])

    summary = {
        "agent_name": agent_name,
        "agent_version": agent_version,
        "rows_evaluated": len(per_row),
        "elapsed_seconds": round(elapsed, 1),
        "judge_deployment": model_config["azure_deployment"],
        "judge_endpoint": model_config["azure_endpoint"],
        "aggregate_mean_scores": aggregate,
        "per_row": per_row,
    }

    print("\n========== AGGREGATE MEANS ==========")
    for k, v in aggregate.items():
        print(f"  {k:22s} {v}")

    results_dir = eval_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[eval] Wrote {out_path}")

    # Non-zero exit only if every row errored out.
    if not any("scores" in r for r in per_row):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(2)
