"""Deploy the LLM judge model used by Foundry evaluators.

Recommended judge model is ``gpt-5-mini`` (per the Foundry evaluator docs for
tool-call accuracy and groundedness). If ``gpt-5-mini`` is not available in
the target Foundry account's region, this script automatically falls back to
``gpt-4.1-mini``.

Idempotent: if a deployment with the configured name already exists, this
script prints its model + version and exits successfully.

Usage:
    python eval/scripts/deploy_judge_model.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Allow running as a script: add eval/ to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_env, require_env  # noqa: E402  pylint: disable=wrong-import-position

import os


CANDIDATE_MODELS = [
    # (model_name, model_version, sku_name, sku_capacity)
    # Capacity is in K tokens-per-minute; 50 is plenty for batch eval.
    ("gpt-5-mini", None, "GlobalStandard", 50),
    ("gpt-4.1-mini", "2025-04-14", "GlobalStandard", 50),
]


def _run_az(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["az", *args],
        capture_output=True,
        text=True,
        check=False,
        shell=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_account_id(resource_id: str) -> tuple[str, str, str]:
    # /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<acct>
    parts = resource_id.strip("/").split("/")
    if len(parts) < 8 or parts[5] != "Microsoft.CognitiveServices":
        raise SystemExit(f"FOUNDRY_ACCOUNT_RESOURCE_ID is not a CognitiveServices account id: {resource_id}")
    return parts[1], parts[3], parts[7]


def _existing_deployment(rg: str, account: str, deployment_name: str) -> dict | None:
    rc, out, _ = _run_az(
        [
            "cognitiveservices", "account", "deployment", "show",
            "--name", account,
            "--resource-group", rg,
            "--deployment-name", deployment_name,
            "-o", "json",
        ]
    )
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _try_deploy(
    rg: str,
    account: str,
    deployment_name: str,
    model_name: str,
    model_version: str | None,
    sku_name: str,
    sku_capacity: int,
) -> tuple[bool, str]:
    args = [
        "cognitiveservices", "account", "deployment", "create",
        "--name", account,
        "--resource-group", rg,
        "--deployment-name", deployment_name,
        "--model-name", model_name,
        "--model-format", "OpenAI",
        "--sku-name", sku_name,
        "--sku-capacity", str(sku_capacity),
    ]
    if model_version:
        args.extend(["--model-version", model_version])
    rc, out, err = _run_az(args)
    return rc == 0, (out + err).strip()


def main() -> int:
    load_env()
    deployment_name = require_env("AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME")
    account_id = require_env("FOUNDRY_ACCOUNT_RESOURCE_ID")

    sub_id, rg, account = _parse_account_id(account_id)
    print(f"[judge] Foundry account: {account} (rg={rg}, sub={sub_id})")
    print(f"[judge] Target deployment name: {deployment_name}")

    # Make sure we are pointed at the right subscription so az calls use it.
    _run_az(["account", "set", "--subscription", sub_id])

    existing = _existing_deployment(rg, account, deployment_name)
    if existing:
        model = existing.get("properties", {}).get("model", {})
        print(
            f"[judge] Deployment already exists: model={model.get('name')} "
            f"version={model.get('version')}. Nothing to do."
        )
        return 0

    last_error = ""
    for model_name, model_version, sku_name, sku_capacity in CANDIDATE_MODELS:
        print(f"[judge] Attempting deploy: model={model_name} version={model_version or 'latest'} sku={sku_name}/{sku_capacity}k tpm")
        ok, message = _try_deploy(rg, account, deployment_name, model_name, model_version, sku_name, sku_capacity)
        if ok:
            print(f"[judge] SUCCESS — deployed {model_name} as '{deployment_name}'.")
            print(f"[judge] Set this in eval/.env: AZURE_AI_JUDGE_MODEL_DEPLOYMENT_NAME={deployment_name}")
            return 0
        print(f"[judge]   failed: {message.splitlines()[-1] if message else 'unknown error'}")
        last_error = message

    print("[judge] ERROR — none of the candidate models could be deployed.")
    print(f"[judge] Last error:\n{last_error}")
    print(
        "[judge] Hint: check `az cognitiveservices account list-models --name "
        f"{account} --resource-group {rg}` for what's available in this region."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
