"""Run a Foundry cloud evaluation against the Pulse RAG eval agent.

This script wires everything together:

* **Target**: the ``pulse-device-agent`` agent created by ``create_agent.py``
  (an ``azure_ai_agent`` target — Foundry runs the agent for each row).
* **Dataset**: ``EVAL_DATASET_NAME`` v``EVAL_DATASET_VERSION`` (uploaded by
  ``data/upload_dataset.py``). Each row supplies ``query`` and
  ``ground_truth``.
* **Judge model**: ``AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME`` (provisioned by
  ``scripts/deploy_judge_model.py``). All LLM-based evaluators score with
  this deployment.
* **Evaluators**: the nine builtin evaluators required for a RAG-with-tools
  agent (tool-call hygiene + answer quality + retrieval grounding +
  intent/task adherence).

It then polls the run until it terminates and prints a short summary plus
the Foundry portal URL so the user can drill into per-row evidence.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_env, project_client, require_env  # noqa: E402
from agent.search_devices_tool import SEARCH_DEVICES_TOOL_DEFINITION  # noqa: E402


POLL_INTERVAL_SECONDS = 15
TERMINAL_STATES = {"completed", "failed", "canceled", "cancelled"}


def _evaluator(name: str, judge_deployment: str, **extra: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "name": name,
        "type": f"builtin.{name}",
        "init_params": {
            "deployment_name": judge_deployment,
        },
    }
    if extra:
        cfg["init_params"].update(extra)  # type: ignore[union-attr]
    return cfg


def _build_eval_config(judge_deployment: str) -> dict[str, object]:
    tool_definitions = [SEARCH_DEVICES_TOOL_DEFINITION]

    # Each entry follows the cloud-eval JSON shape: name + type + init params.
    # ``data_mapping`` tells the evaluator which sample/item fields to read.
    testing_criteria = [
        # --- Tool-use evaluators (only meaningful because we use a Function Tool,
        # not the Azure AI Search knowledge tool). ---
        {
            **_evaluator("tool_call_accuracy", judge_deployment),
            "init_params": {
                "deployment_name": judge_deployment,
                "tool_definitions": tool_definitions,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "tool_calls": "{{sample.output_items}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            **_evaluator("tool_selection", judge_deployment),
            "init_params": {
                "deployment_name": judge_deployment,
                "tool_definitions": tool_definitions,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "tool_calls": "{{sample.output_items}}",
            },
        },
        {
            **_evaluator("tool_input_accuracy", judge_deployment),
            "init_params": {
                "deployment_name": judge_deployment,
                "tool_definitions": tool_definitions,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "tool_calls": "{{sample.output_items}}",
            },
        },
        # --- Answer-quality evaluators. ---
        {
            **_evaluator("relevance", judge_deployment),
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            **_evaluator("groundedness", judge_deployment),
            "data_mapping": {
                "query": "{{item.query}}",
                "context": "{{sample.output_items}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            **_evaluator("retrieval", judge_deployment),
            "data_mapping": {
                "query": "{{item.query}}",
                "context": "{{sample.output_items}}",
            },
        },
        {
            **_evaluator("response_completeness", judge_deployment),
            "data_mapping": {
                "response": "{{sample.output_text}}",
                "ground_truth": "{{item.ground_truth}}",
            },
        },
        {
            **_evaluator("intent_resolution", judge_deployment),
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            **_evaluator("task_adherence", judge_deployment),
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
                "tool_calls": "{{sample.output_items}}",
            },
        },
    ]

    return {"testing_criteria": testing_criteria}


def _data_source(dataset_name: str, dataset_version: str, agent_name: str, agent_version: str) -> dict[str, object]:
    return {
        "type": "azure_ai_target_completions",
        "input_dataset": {
            "type": "dataset",
            "name": dataset_name,
            "version": dataset_version,
        },
        "target": {
            "type": "azure_ai_agent",
            "name": agent_name,
            "version": agent_version,
        },
        "config": {
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ground_truth": {"type": "string"},
                    "expected_source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expected_tool": {"type": ["string", "null"]},
                },
                "required": ["query", "ground_truth"],
            },
            "include_sample_schema": True,
        },
    }


def main() -> int:
    load_env()
    judge_deployment = require_env("AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME")
    agent_name = require_env("EVAL_AGENT_NAME")
    agent_version = require_env("EVAL_AGENT_VERSION")
    dataset_name = require_env("EVAL_DATASET_NAME")
    dataset_version = require_env("EVAL_DATASET_VERSION")

    client = project_client()

    print("[eval] Creating eval definition...")
    eval_def = client.evals.create(
        name="pulse-rag-eval",
        data_source_config={"type": "custom"},
        **_build_eval_config(judge_deployment),
    )
    eval_id = getattr(eval_def, "id", None) or getattr(eval_def, "eval_id")
    print(f"[eval] eval_id = {eval_id}")

    print("[eval] Starting eval run against agent target...")
    run = client.evals.runs.create(
        eval_id=eval_id,
        data_source=_data_source(dataset_name, dataset_version, agent_name, agent_version),
    )
    run_id = getattr(run, "id", None) or getattr(run, "run_id")
    print(f"[eval] run_id = {run_id}")

    # Poll for completion.
    while True:
        run = client.evals.runs.retrieve(eval_id=eval_id, run_id=run_id)
        status = str(getattr(run, "status", "")).lower()
        print(f"[eval] status={status}")
        if status in TERMINAL_STATES:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    report_url = getattr(run, "report_url", None) or getattr(run, "results_url", None)
    results = getattr(run, "per_testing_criteria_results", None)

    print("\n========== RESULTS ==========")
    if results:
        for criterion in results:
            name = criterion.get("name") if isinstance(criterion, dict) else getattr(criterion, "name", "?")
            passed = criterion.get("passed") if isinstance(criterion, dict) else getattr(criterion, "passed", None)
            failed = criterion.get("failed") if isinstance(criterion, dict) else getattr(criterion, "failed", None)
            print(f"  {name:30s}  passed={passed}  failed={failed}")
    else:
        # Some SDK builds put the summary inside ``result_counts``.
        result_counts = getattr(run, "result_counts", None)
        print(f"  result_counts = {result_counts}")
        print(f"  raw run = {json.dumps(getattr(run, 'as_dict', lambda: {})(), indent=2, default=str)[:2000]}")
    if report_url:
        print(f"\nOpen in Foundry portal: {report_url}")
    return 0 if str(getattr(run, "status", "")).lower() == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
