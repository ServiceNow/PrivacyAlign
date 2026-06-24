# Privacy Evaluation

Runs the **PrivacyAlign** benchmark end to end. It generates agent responses to privacy-sensitive scenarios under a chosen agent prompt, then judges each response for two failure modes: **leaks** (disclosing information that should stay private from the recipient) and **omits** (dropping non-sensitive information that would have been helpful).

Judging optionally shows the judge human reference annotations as a calibration signal. Data comes from [`ServiceNow/PrivacyAlign`](https://huggingface.co/datasets/ServiceNow/PrivacyAlign) on the Hub. Scripts support OpenRouter or Azure OpenAI backends, and judging can also run a local vLLM judge.

## Scripts

| Script | What it does |
| --- | --- |
| `generate_responses.py` | Runs an agent prompt (`naive` or `privacy_enhanced`) over the test split for one or more models. One results file per model. |
| `judge_responses.py` | Judges responses for `leaks`/`omits` using a local vLLM judge or an OpenRouter/Azure judge, with or without annotations shown. |

## Dataset

Scripts download the requested split (default `test`) of `ServiceNow/PrivacyAlign` and cache it as JSONL under `.cache/privalign-dataset/`. Requires `pip install datasets`. Pass `--dataset-split` to choose a split, or use a local JSONL with `--dataset-path` (`generate_responses.py`) / `--samples-path` (`judge_responses.py`).

`generate_responses.py` builds prompts via the shared builder in `synthetic_data_generation/`, so edits to `resources/prompts/naive_agent_prompt.txt` (or `privacy_enhanced_agent_prompt.txt`) there are picked up automatically. Judge templates live here under `resources/prompts/`: `leak_omit_judge_prompt_with_annotations.txt` shows the two reference responses plus per-annotator human labels, and `leak_omit_judge_prompt_no_annotations.txt` shows only scenario context and the response.

## Quickstart

```bash
export OPENROUTER_API_KEY=...   # required for the OpenRouter backend
```

**1. Generate agent responses** (downloads the `test` split on first run):

```bash
python3 privacy_evaluation/generate_responses.py \
  --prompt naive \
  --models anthropic/claude-opus-4.7 google/gemini-3.1-pro-preview \
  --max-tokens 65536 \
  --retry-errors
```

Each model writes to `privacy_evaluation/runs/naive_openrouter_test/<model_slug>/results.jsonl`.

**2. Judge the responses** (OpenRouter, annotations shown):

```bash
python3 privacy_evaluation/judge_responses.py \
  --results-paths \
    privacy_evaluation/runs/naive_openrouter_test/anthropic_claude_opus_4_7/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/google_gemini_3_1_pro_preview/results.jsonl \
  --backend openrouter \
  --judge-model google/gemini-3.1-flash-lite \
  --reasoning-effort high \
  --max-tokens 65536 \
  --judge-prompt with_annotations
```

`with_annotations` is the default. Each results file gets a sibling judgments file named after the judge variant and model, e.g. `judgments__google_gemini_3_1_flash_lite.jsonl` (or `judgments_no_annotations__<judge>.jsonl` for `--judge-prompt no_annotations`).

Smoke-test prompt construction without spending credits:

```bash
python3 privacy_evaluation/generate_responses.py --dry-run --limit 2 --print-first-prompt
```

## Agent prompts

`generate_responses.py --prompt {naive,privacy_enhanced}`. `naive` is the default leakage-phase prompt; `privacy_enhanced` adds an instruction to consider sender/recipient and withhold information inappropriate for the recipient. Each variant writes to its own `runs/` subdir (`naive_openrouter_test/`, `privacy_enhanced_openrouter_test/`) so resume is isolated per prompt.

## Judge prompts: with vs without annotations

`judge_responses.py --judge-prompt {with_annotations,no_annotations}` (default `with_annotations`) controls whether human reference annotations are shown.

- **`with_annotations`** the judge sees the user instruction, prior memories, prior tool trajectory, both reference responses (A/B), and per-annotator human labels (anonymized as `annotator_1`, `annotator_2`, ...) for each flagged leak/omission. Labels are framed as calibration guidance, not ground truth. Writes `judgments__<judge>.jsonl`.
- **`no_annotations`** the judge sees only scenario context and the response. Writes `judgments_no_annotations__<judge>.jsonl`.

Annotations are read from the samples file's embedded `annotations` field. Each judgment row records `leaks` / `leaks_explanation` / `omits` / `omits_explanation`, plus a `judge_status` of `ok`, `parse_error`, `empty_response`, `skipped`, or `judge_error`. Thinking traces are stripped before parsing, and the parser tolerates `key: value` lines instead of strict JSON.

## Backends

All scripts share the same backend plumbing.

- **OpenRouter** (default) set `OPENROUTER_API_KEY` and pass OpenRouter model IDs. Reasoning is controlled with `--reasoning-effort {none,minimal,low,medium,high,xhigh}`, `--reasoning-enabled` (adaptive thinking), `--reasoning-max-tokens`, `--verbosity`, and `--include-reasoning`. Controls are model-family dependent, so some frontier models ignore `--reasoning-effort` and only honor `--reasoning-enabled`. `--temperature` / `--top-p` are accepted but **not sent** on the OpenRouter/Azure path.
- **Azure OpenAI** pass `--azure-endpoint https://<resource>.openai.azure.com/openai/v1` and use Azure deployment names for `--models` / `--judge-model`. The key is read from `AZURE_OPENAI_API_KEY` (override with `--azure-api-key-env`). The path translates OpenRouter-shaped requests into Azure Chat Completions shape (`reasoning` flattens to `reasoning_effort`, `max_tokens` becomes `max_completion_tokens`).
- **Local vLLM judge** `judge_responses.py --backend vllm` (its default) loads the judge offline. It honors `--temperature` / `--top-p` / `--top-k` and the usual `--vllm-*` flags (`--vllm-tensor-parallel-size`, `--vllm-gpu-memory-utilization`, `--vllm-max-model-len`, `--enable-thinking`, etc.). See `--help`.

## Resume and retry

Generation and judging are resumable and append-only.

- **Resume is on by default.** `generate_responses.py` skips a sample whose latest row for the matching prompt hash has `status: ok`. `judge_responses.py` skips `(sample, model, judge)` keys already judged `ok`. Use `--no-resume` or `--overwrite`.
- **`--retry-errors`** (`generate_responses.py`) retries the retryable set (`api_error`, `empty_response`, `parse_error`) up to `--max-attempts` (default `5`) per model/sample/prompt-hash. Results stream to disk as each call completes.
- Per-request timeouts use `--openrouter-request-timeout` (seconds, `0` disables).

## Output layout

`runs/` is git-ignored.

```text
privacy_evaluation/runs/<prompt-subdir>/<model_slug>/results.jsonl
privacy_evaluation/runs/<prompt-subdir>/<model_slug>/results.summary.json
.../judgments__<judge_slug>.jsonl                 # --judge-prompt with_annotations
.../judgments_no_annotations__<judge_slug>.jsonl  # --judge-prompt no_annotations
.../<stem>.summary.json                           # per-file judge summary
```

`<model_slug>` is the slugified model ID (e.g. `anthropic/claude-opus-4.7` becomes `anthropic_claude_opus_4_7`). Each results row carries `sample_name`, `model`, `status`, `attempt`, `prompt_sha1`, `expected_final_action`, `raw_output`, `parsed_action`, and timing. Pass `--save-prompts` to embed the full prompt.
