# Pulse RAG Foundry Evaluator

Cloud-side Foundry evaluation harness for the Pulse RAG sample. It exercises the
production retrieval path (`app/retrieval.py`) end-to-end through a Foundry agent
and scores it with nine LLM-judged evaluators.

## Why this exists (and why it is shaped this way)

The Pulse RAG production app is a tight RAG loop: Flask → `retrieve_context()`
→ Foundry chat completion. There is no agent in production. So why does the
eval scaffold one?

Two reasons:

1. **Foundry's tool-call evaluators (`tool_call_accuracy`, `tool_selection`,
   `tool_input_accuracy`) and `groundedness` have limited support when the
   agent uses the Azure AI Search _knowledge tool_.** Wrapping retrieval as
   a user-defined **Function Tool** is the documented workaround and gives
   the evaluators a tool definition + tool call trace they can reason over.
   That is exactly what `agent/search_devices_tool.py` does — it imports
   `retrieve_context()` from the production app, so the eval scores the
   same retrieval code the app runs.
2. The agent target lets us reuse Foundry's `azure_ai_target_completions`
   data source, which executes the agent for each dataset row server-side
   and feeds the trace straight into the evaluators. No custom batch loop.

## Dataset rationale

`data/pulse_eval_questions.jsonl` is **21 hand-curated rows**, not
synthetically generated. The Foundry Simulator preview can synthesize Q/A
pairs, but for a tiny fixed inventory (15 devices) human curation gives
better coverage of the patterns we actually want to grade:

- **Single-device lookup** ("What model is the Main Lobby Touch Panel?")
- **Multi-device aggregation** ("Which devices are in the Boardroom?",
  "Which devices support Dante audio?")
- **Capability filters** ("Which devices support wireless content sharing?")
- **Status / operational** ("Which devices are degraded or offline?")
- **Negative cases** ("Is there an espresso machine in the Cafe?" → expect
  "I do not know") to score `groundedness` on refusals.
- **No-tool case** ("Hello, can you tell me your name?") to score
  `tool_selection` on when _not_ to call the tool.

Each row carries `expected_source_ids` and `expected_tool` so the
evaluators can compare ground truth to the agent's tool calls and final
answer.

## Prerequisites

1. **Pulse RAG stack must be deployed.** This eval calls into
   `app/retrieval.py`, which needs the `pulse-device-chunks` Azure AI Search
   index populated. From the repo root:

   ```pwsh
   cd pulse-rag-p0-sample
   azd up
   ```

   That provisions Cosmos, Search, Foundry, the indexer Function, and the
   chat ACA app. Seed sample data with `scripts/seed_cosmos.py` (see
   `pulse-rag-p0-sample/README.md`).

2. **Azure CLI logged in** as a user with `Cognitive Services Contributor`
   (or higher) on the Foundry account — required to deploy the judge model.

3. **Python 3.10+** with the eval dependencies installed:

   ```pwsh
   cd pulse-rag-p0-sample/eval
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. **`eval/.env`** populated. Copy `sample.env` to `.env` and fill in:
   - `PROJECT_ENDPOINT` — the Foundry project endpoint from `azd env get-values`.
   - `FOUNDRY_ACCOUNT_RESOURCE_ID` — full ARM id of the Foundry
     CognitiveServices account (the management-plane parent of
     `PROJECT_ENDPOINT`). Needed only by `deploy_judge_model.py`.
   - `SearchServiceEndpoint`, `SearchIndexName` — the values
     `app/retrieval.py` uses.
   - The dataset / agent / judge variables (defaults in `sample.env` are fine).

## Run the evaluation

Three commands, in order:

```pwsh
# 1. Provision the LLM judge (tries gpt-5-mini, falls back to gpt-4.1-mini).
python scripts/deploy_judge_model.py

# 2. Provision the eval agent with the search_devices function tool.
python agent/create_agent.py

# 3. Upload the golden dataset and kick off the cloud eval.
python data/upload_dataset.py
python run_eval.py
```

`run_eval.py` polls the run until it reaches a terminal state and prints a
per-evaluator pass/fail summary plus the Foundry portal URL for the full
report.

## File layout

```
eval/
├── README.md                            (this file)
├── requirements.txt                     pinned-floor eval-only deps
├── sample.env                           env template
├── common.py                            shared bootstrap (env, AIProjectClient)
├── run_eval.py                          cloud eval entrypoint
├── scripts/
│   └── deploy_judge_model.py            Phase 1b — provision judge LLM
├── agent/
│   ├── search_devices_tool.py           function-tool wrapping app/retrieval.py
│   └── create_agent.py                  provision the pulse-device-agent
└── data/
    ├── pulse_eval_questions.jsonl       21 hand-curated golden rows
    └── upload_dataset.py                push dataset to Foundry
```

## Regional availability note

The `deploy_judge_model.py` script prefers `gpt-5-mini` because the Foundry
docs recommend it for tool-call and groundedness evaluators. If `gpt-5-mini`
is not available in your Foundry account's region, the script automatically
falls back to `gpt-4.1-mini` (`2025-04-14`). See the script's log output for
which model actually landed.
