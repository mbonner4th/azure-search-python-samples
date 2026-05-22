"""Create a persistent Foundry batch evaluation for the Pulse device agent.

Uses ``client.evals.create`` + ``client.evals.runs.create`` (cloud evaluation)
against the project's OpenAI-compatible endpoint. The evaluation is a named,
re-runnable resource visible in the Foundry portal under Project > Evaluation.

Required env (eval/.env):
  PROJECT_ENDPOINT      Foundry project endpoint
                        (https://{account}.services.ai.azure.com/api/projects/{project})
  EVAL_AGENT_NAME       Foundry agent name (default: pulse-device-agent)
  EVAL_AGENT_VERSION    Agent version to evaluate (default: latest)
  AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME
                        Deployment used by LLM-judge evaluators
                        (e.g. eval-judge-gpt5mini)

Usage:
  python eval/scripts/create_foundry_eval.py

The script uploads the JSONL dataset under eval/data, creates (or re-uses) the
evaluation, then starts a new run. It prints the eval id, run id, and the
portal URL.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Make `eval` package importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from common import load_env, require_env  # noqa: E402
from agent.search_devices_tool import SEARCH_DEVICES_TOOL_DEFINITION  # noqa: E402

from azure.ai.projects import AIProjectClient  # noqa: E402
from azure.identity import DefaultAzureCredential  # noqa: E402


def _services_ai_endpoint() -> str:
    """Evaluations API lives under the services.ai.azure.com data-plane host.

    Translate the cognitiveservices form to the services.ai form when needed.
    """
    ep = require_env("PROJECT_ENDPOINT")
    return ep.replace(".cognitiveservices.azure.com", ".services.ai.azure.com")

EVAL_NAME = "pulse-device-agent-eval"
DATASET_NAME = "pulse-device-eval-questions"
DATASET_FILE = ROOT / "eval" / "data" / "pulse_eval_questions.jsonl"


def _build_testing_criteria(judge_deployment: str) -> list[dict]:
    """Built-in evaluator definitions matching the local run_eval.py runner."""
    common_text = {
        "query": "{{item.query}}",
        "response": "{{sample.output_text}}",
        "ground_truth": "{{item.ground_truth}}",
    }
    common_items = {
        "query": "{{item.query}}",
        "response": "{{sample.output_items}}",
        "ground_truth": "{{item.ground_truth}}",
    }
    return [
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {"deployment_name": judge_deployment},
            "data_mapping": common_text,
        },
        {
            "type": "azure_ai_evaluator",
            "name": "groundedness",
            "evaluator_name": "builtin.groundedness",
            "initialization_parameters": {"deployment_name": judge_deployment},
            "data_mapping": common_text,
        },
        {
            "type": "azure_ai_evaluator",
            "name": "intent_resolution",
            "evaluator_name": "builtin.intent_resolution",
            "initialization_parameters": {"deployment_name": judge_deployment},
            "data_mapping": common_items,
        },
        {
            "type": "azure_ai_evaluator",
            "name": "task_adherence",
            "evaluator_name": "builtin.task_adherence",
            "initialization_parameters": {"deployment_name": judge_deployment},
            "data_mapping": common_items,
        },
        {
            "type": "azure_ai_evaluator",
            "name": "tool_call_accuracy",
            "evaluator_name": "builtin.tool_call_accuracy",
            "initialization_parameters": {"deployment_name": judge_deployment},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_items}}",
                "tool_definitions": "{{item.tool_definitions}}",
            },
        },
    ]


def main() -> int:
    load_env()
    agent_name = os.environ.get("EVAL_AGENT_NAME", "pulse-device-agent")
    agent_version = os.environ.get("EVAL_AGENT_VERSION", "").strip()
    judge_deployment = require_env("AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME")

    if not DATASET_FILE.exists():
        print(f"[eval] dataset file not found: {DATASET_FILE}", file=sys.stderr)
        return 2

    proj = AIProjectClient(
        endpoint=_services_ai_endpoint(),
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )

    # Build an enriched JSONL that embeds the agent's tool definitions per row
    # (required by builtin.tool_call_accuracy data_mapping).
    tool_defs = [SEARCH_DEVICES_TOOL_DEFINITION]
    enriched_path = Path(tempfile.gettempdir()) / "pulse_eval_with_tooldefs.jsonl"
    with DATASET_FILE.open("r", encoding="utf-8") as src, enriched_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["tool_definitions"] = tool_defs
            dst.write(json.dumps(row) + "\n")

    # Upload (or version) the dataset
    dataset_version = time.strftime("%Y%m%d%H%M%S")
    print(f"[eval] uploading dataset {DATASET_NAME}:{dataset_version} from {enriched_path}")
    ds = proj.datasets.upload_file(
        name=DATASET_NAME,
        version=dataset_version,
        file_path=str(enriched_path),
    )
    data_id = ds.id
    print(f"[eval] dataset id: {data_id}")

    client = proj.get_openai_client()

    item_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "ground_truth": {"type": "string"},
            "tool_definitions": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["query"],
    }
    data_source_config = {
        "type": "custom",
        "item_schema": item_schema,
        "include_sample_schema": True,
    }

    testing_criteria = _build_testing_criteria(judge_deployment)

    print(f"[eval] creating/getting evaluation: {EVAL_NAME}")
    eval_object = client.evals.create(
        name=EVAL_NAME,
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,
        metadata={"agent_name": agent_name, "purpose": "pulse-rag-baseline"},
    )
    print(f"[eval] eval id: {eval_object.id}")

    target = {"type": "azure_ai_agent", "name": agent_name}
    if agent_version:
        target["version"] = agent_version

    input_messages = {
        "type": "template",
        "template": [
            {
                "type": "message",
                "role": "user",
                "content": {"type": "input_text", "text": "{{item.query}}"},
            }
        ],
    }

    data_source = {
        "type": "azure_ai_target_completions",
        "source": {"type": "file_id", "id": data_id},
        "input_messages": input_messages,
        "target": target,
    }

    run_name = f"{agent_name}-v{agent_version or 'latest'}-{dataset_version}"
    print(f"[eval] starting run: {run_name}")
    run = client.evals.runs.create(
        eval_id=eval_object.id,
        name=run_name,
        data_source=data_source,
    )
    print(f"[eval] run id:     {run.id}")
    print(f"[eval] run status: {getattr(run, 'status', '?')}")
    report_url = getattr(run, "report_url", None)
    if report_url:
        print(f"[eval] report:     {report_url}")
    print("\n[eval] Done. Open the run in the Foundry portal under")
    print("       Project > Evaluation to view live progress and results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
