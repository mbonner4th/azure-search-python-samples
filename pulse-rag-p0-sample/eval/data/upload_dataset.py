"""Upload the hand-curated golden questions to the Foundry project as a dataset.

The dataset becomes the input for ``eval/run_eval.py``. Re-running this script
publishes a new dataset *version* (the version string in ``EVAL_DATASET_VERSION``
must be incremented in ``eval/.env`` for that to happen) — it does NOT update
an existing version in-place, because Foundry treats datasets as immutable
per-version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_env, project_client, require_env  # noqa: E402


def main() -> int:
    load_env()
    dataset_name = require_env("EVAL_DATASET_NAME")
    dataset_version = require_env("EVAL_DATASET_VERSION")

    jsonl_path = Path(__file__).resolve().parent / "pulse_eval_questions.jsonl"
    if not jsonl_path.exists():
        print(f"[dataset] ERROR: {jsonl_path} not found.")
        return 1

    # Sanity-check the JSONL before pushing.
    rows = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[dataset] ERROR: invalid JSON on line {line_no}: {exc}")
                return 1
            for required in ("query", "ground_truth"):
                if required not in obj:
                    print(f"[dataset] ERROR: line {line_no} missing '{required}'")
                    return 1
            rows += 1

    print(f"[dataset] Validated {rows} rows in {jsonl_path.name}.")

    client = project_client()
    print(f"[dataset] Uploading as name='{dataset_name}' version='{dataset_version}'.")
    result = client.datasets.upload_file(
        name=dataset_name,
        version=dataset_version,
        file_path=str(jsonl_path),
    )

    dataset_id = getattr(result, "id", None) or getattr(result, "dataset_id", None) or repr(result)
    print(f"[dataset] Upload complete. dataset_id={dataset_id}")
    print("[dataset] Use this dataset name+version in EVAL_DATASET_NAME / EVAL_DATASET_VERSION when running eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
