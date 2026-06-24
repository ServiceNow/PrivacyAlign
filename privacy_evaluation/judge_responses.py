#!/usr/bin/env python3
"""Judge naive-agent responses for leaks/omits with a local vLLM judge.

For every row in one or more `results.jsonl` files (produced by
`privacy_evaluation/generate_responses.py`), we build a judge prompt that shows the
judge:
  - the user instruction, prior memories, and prior tool trajectory;
  - the two reference responses humans annotated for the same scenario;
  - per-annotator human labels on those reference responses (anonymized,
    each annotator listed individually);
  - the NEW response (the one to judge).

The judge model runs offline via vLLM. Thinking traces are stripped using
the same parser online_preference_alignment uses. Output JSONL has one row
per (results-row x judge), schema:

    {leaks: bool, leaks_details: str, omits: bool, omits_details: str,
     analysis: str, judge_status: "ok"|"parse_error"|"skipped"|"empty_response", ...}


OpenRouter backend example:

python3 privacy_evaluation/judge_responses.py \
  --results-paths \
    privacy_evaluation/runs/naive_openrouter_test/anthropic_claude_opus_4_7/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/anthropic_claude_sonnet_4_6/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/google_gemini_3_1_flash_lite_preview/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/google_gemini_3_1_pro_preview/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/gpt_5_4_mini/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/gpt_5_5/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/anthropic_claude_opus_4_7/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/anthropic_claude_sonnet_4_6/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/google_gemini_3_1_flash_lite_preview/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/google_gemini_3_1_pro_preview/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/gpt_5_4_mini/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/gpt_5_5/results.jsonl \
  --backend openrouter \
  --judge-model google/gemini-3.1-flash-lite \
  --reasoning-effort high \
  --max-tokens 65536 \
  --judge-prompt no_annotations


Azure OpenAI backend example (v1 endpoint, OpenAI-SDK-compatible). --judge-model
takes the Azure deployment name; the Azure key is read from AZURE_OPENAI_API_KEY.

python3 privacy_evaluation/judge_responses.py \
  --results-paths \
    privacy_evaluation/runs/naive_openrouter_test/anthropic_claude_opus_4_7/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/anthropic_claude_sonnet_4_6/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/google_gemini_3_1_flash_lite_preview/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/google_gemini_3_1_pro_preview/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/gpt_5_4_mini/results.jsonl \
    privacy_evaluation/runs/naive_openrouter_test/gpt_5_5/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/anthropic_claude_opus_4_7/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/anthropic_claude_sonnet_4_6/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/google_gemini_3_1_flash_lite_preview/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/google_gemini_3_1_pro_preview/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/gpt_5_4_mini/results.jsonl \
    privacy_evaluation/runs/privacy_enhanced_openrouter_test/gpt_5_5/results.jsonl \
  --backend openrouter \
  --azure-endpoint https://<your-resource>.openai.azure.com/openai/v1 \
  --judge-model gpt-5.4-mini \
  --reasoning-effort high \
  --max-tokens 65536 \
  --judge-prompt with_annotations



"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = REPO_ROOT / "synthetic_data_generation"
ONLINE_PREF_DIR = REPO_ROOT / "online_preference_alignment"
JUDGE_PROMPTS_DIR = REPO_ROOT / "privacy_evaluation" / "resources" / "prompts"
# --judge-prompt -> (template filename, judgments output stem).
# `with_annotations` keeps the canonical `judgments.jsonl` filename so existing
# runs resume cleanly; `no_annotations` writes to a sibling file so the two
# variants do not silently overwrite each other.
JUDGE_PROMPT_VARIANTS: Dict[str, Tuple[str, str]] = {
    "with_annotations": ("leak_omit_judge_prompt_with_annotations.txt", "judgments"),
    "no_annotations": ("leak_omit_judge_prompt_no_annotations.txt", "judgments_no_annotations"),
}
# PrivacyAlign is distributed via the Hugging Face Hub; the requested split is
# downloaded and cached as JSONL so it can be loaded like a local sample catalog.
HF_DATASET_ID = "ServiceNow/PrivacyAlign"
DEFAULT_DATASET_SPLIT = "test"
DATASET_CACHE_DIR = REPO_ROOT / ".cache" / "privalign-dataset"


def ensure_hf_split_jsonl(split: str = DEFAULT_DATASET_SPLIT) -> Path:
    """Download a PrivacyAlign split from the HF Hub and cache it as JSONL."""
    cache_path = DATASET_CACHE_DIR / f"{split}.jsonl"
    if cache_path.exists():
        return cache_path
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Loading the default PrivacyAlign dataset requires the 'datasets' package "
            "(pip install datasets), or pass an explicit --samples-path."
        ) from exc
    dataset = load_dataset(HF_DATASET_ID, split=split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return cache_path

for path in (GENERATION_DIR, ONLINE_PREF_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from model_client import load_model  # noqa: E402
from utils.text_utils import strip_thinking_trace  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _sample_name_for_sample(sample: Dict[str, Any]) -> str:
    for key in ("name", "sample_name", "id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    source = sample.get("source")
    if isinstance(source, dict):
        for key in ("name", "sample_name", "id"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return hashlib.sha1(json.dumps(sample, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_samples(path: Path) -> Dict[str, Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return {_sample_name_for_sample(row): row for row in _read_jsonl(path)}
    payload = _read_json(path)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {_sample_name_for_sample(row): row for row in payload if isinstance(row, dict)}
    raise ValueError(f"Expected samples data to be a dict, list, or JSONL file: {path}")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(REPO_ROOT / ".env")


def _translate_extra_body_for_azure(extra_body: Any) -> Any:
    """Translate OpenRouter-shaped extra_body to Azure Chat Completions shape.

    OpenRouter accepts a nested `reasoning: {effort, enabled, exclude,
    max_tokens}` object. Azure's Chat Completions endpoint for GPT-5
    reasoning deployments rejects that (`400 Unknown parameter: 'reasoning'`)
    and instead expects a flat top-level `reasoning_effort` scalar. We lift
    effort to the top level, drop the OpenRouter-only sibling fields
    (enabled / exclude / max_tokens) which have no Chat Completions
    equivalents on Azure, and leave `verbosity` in place (Azure GPT-5
    reasoning deployments accept it as a top-level param).
    """
    if not isinstance(extra_body, dict):
        return extra_body
    cleaned = dict(extra_body)
    reasoning = cleaned.pop("reasoning", None)
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            cleaned["reasoning_effort"] = effort
    return cleaned or None


def _retarget_client_to_azure(client: Any, *, base_url: str, api_key: str) -> None:
    """Repoint a UnifiedModelClient's OpenAI clients at an Azure v1 endpoint.

    Azure OpenAI's v1 surface is OpenAI-SDK-compatible, so we swap base_url +
    api_key on fresh OpenAI/AsyncOpenAI instances. We also wrap
    chat.completions.create on both clients so OpenRouter-shaped reasoning
    fields get translated to Azure's Chat Completions shape before forwarding
    (see _translate_extra_body_for_azure). The model field sent in
    chat.completions.create is the Azure deployment name (UnifiedModelClient
    sends self.config.model_name, set to the user-provided --judge-model
    value).
    """
    from openai import OpenAI, AsyncOpenAI

    timeout_config = 120.0
    close = getattr(client.client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass

    new_sync = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_config,
        max_retries=3,
    )
    new_async = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_config,
        max_retries=3,
    )

    def _translate_kwargs(kwargs: Dict[str, Any]) -> None:
        # Azure GPT-5 reasoning deployments expect `max_completion_tokens`
        # instead of `max_tokens`.
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if "extra_body" in kwargs:
            kwargs["extra_body"] = _translate_extra_body_for_azure(kwargs["extra_body"])

    sync_original = new_sync.chat.completions.create
    def _sync_create(*args: Any, **kwargs: Any) -> Any:
        _translate_kwargs(kwargs)
        return sync_original(*args, **kwargs)
    new_sync.chat.completions.create = _sync_create  # type: ignore[method-assign]

    async_original = new_async.chat.completions.create
    async def _async_create(*args: Any, **kwargs: Any) -> Any:
        _translate_kwargs(kwargs)
        return await async_original(*args, **kwargs)
    new_async.chat.completions.create = _async_create  # type: ignore[method-assign]

    client.client = new_sync
    client.async_client = new_async


def load_embedded_human_annotations(
    samples: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for sample_name, sample in samples.items():
        annotations = sample.get("annotations")
        if not isinstance(annotations, list):
            continue
        for index, ann in enumerate(annotations, start=1):
            if not isinstance(ann, dict):
                continue
            by_sample.setdefault(sample_name, []).append({
                "annotator_id": f"annotator_{index}",
                "preference": ann.get("preference"),
                "explanation": ann.get("explanation") or "",
                "response_a_labels": ann.get("response_a_labels") or {},
                "response_b_labels": ann.get("response_b_labels") or {},
            })
    return by_sample


def format_annotator_block(annotators: Sequence[Dict[str, Any]]) -> str:
    if not annotators:
        return "(no human annotations available for this sample)"
    lines: List[str] = []
    for index, a in enumerate(annotators, 1):
        a_lab = a.get("response_a_labels") or {}
        b_lab = a.get("response_b_labels") or {}
        annotator_id = a.get("annotator_id") or f"annotator_{index}"
        if lines:
            lines.append("")
        lines.append(f"{annotator_id}:")
        lines.append(f"- Preference: {_format_preference_label(a.get('preference'))}")
        explanation = (a.get("explanation") or "").strip()
        if explanation:
            lines.append(f"- Explanation: {explanation}")
        a_positive = _collect_positive_sentences(a_lab)
        b_positive = _collect_positive_sentences(b_lab)
        if a_positive:
            lines.append("- Response A:")
            lines.extend(a_positive)
        if b_positive:
            lines.append("- Response B:")
            lines.extend(b_positive)
    return "\n".join(lines)


def _collect_positive_sentences(labels: Dict[str, Any]) -> List[str]:
    """Return one sentence per annotator-flagged failure on this response.

    Mirrors ``_collect_privalign_positive_sentences`` in the training-side
    loader: only positive labels with non-empty details are emitted, so the
    judge prompt focuses on issues the annotator actually flagged.
    """
    sentences: List[str] = []
    if labels.get("leaks"):
        details = str(labels.get("leaks_details") or "").strip()
        if details:
            sentences.append(f"  - Annotator flagged a leak: {details}")
    if labels.get("omits"):
        details = str(labels.get("omits_details") or "").strip()
        if details:
            sentences.append(f"  - Annotator flagged a missing detail: {details}")
    return sentences


def _format_preference_label(preference: Any) -> str:
    if not isinstance(preference, str) or not preference.strip():
        return "unspecified"
    normalized = preference.strip().lower()
    label_map = {
        "agent_a": "prefers Response A",
        "agent_a_slightly": "slightly prefers Response A",
        "agent_a_strongly": "strongly prefers Response A",
        "agent_b": "prefers Response B",
        "agent_b_slightly": "slightly prefers Response B",
        "agent_b_strongly": "strongly prefers Response B",
        "tie": "sees the responses as tied",
        "unsure": "is unsure which response is better",
    }
    if normalized in label_map:
        return label_map[normalized]
    return normalized.replace("_", " ")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2)


def _reference_response_for_sample(sample: Dict[str, Any], side: str) -> Any:
    """Return the reference response matching the annotation side.

    Privalign JSONL rows use ``response_a`` / ``response_b``; those fields are
    the only supported source for annotation-conditioned reference responses.
    """
    if side not in {"a", "b"}:
        raise ValueError(f"side must be 'a' or 'b', got {side!r}.")
    response = sample.get(f"response_{side}")
    if isinstance(response, dict):
        response = response.get("action_raw") or response
    return response or ""


def build_judge_prompt(
    *,
    template: Template,
    sample: Dict[str, Any],
    annotators: Sequence[Dict[str, Any]],
    new_response: str,
) -> str:
    src = sample.get("source", sample)
    trajectory = src.get("trajectory") or {}
    if isinstance(trajectory, dict):
        user_instruction = trajectory.get("user_instruction", "")
        executable_trajectory = trajectory.get("executable_trajectory", "")
    else:
        user_instruction = src.get("user_instruction", sample.get("user_instruction", ""))
        executable_trajectory = trajectory
    memories = src.get("memories", [])
    response_a = _reference_response_for_sample(sample, "a")
    response_b = _reference_response_for_sample(sample, "b")
    return template.safe_substitute(
        user_instruction=_stringify(user_instruction),
        memories=_stringify(memories) if memories else "(no memories)",
        executable_trajectory=_stringify(executable_trajectory),
        reference_response_a=_stringify(response_a),
        reference_response_b=_stringify(response_b),
        annotator_block=format_annotator_block(annotators),
        new_response=_stringify(new_response),
    )



def parse_judgment(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """Strip thinking trace and parse a JSON judgment.

    Returns (parsed, error, cleaned_text).
    """
    if not raw_text or not raw_text.strip():
        return None, "empty", ""
    cleaned = strip_thinking_trace(raw_text)
    if not cleaned:
        return None, "empty_after_strip", ""

    # Try the last non-empty line first (matches the prompt instruction).
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    candidates: List[str] = []
    if lines:
        candidates.append(lines[-1])
    # Fallback: scan all balanced top-level JSON objects in the cleaned text.
    candidates.extend(re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", cleaned, re.DOTALL))

    last_err: Optional[str] = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            last_err = str(exc)
            continue
        if not isinstance(obj, dict):
            last_err = "not a JSON object"
            continue
        if "leaks" not in obj or "omits" not in obj:
            last_err = "missing required keys"
            continue
        normalized = {
            "leaks": bool(obj.get("leaks")),
            "leaks_explanation": str(obj.get("leaks_explanation") or ""),
            "omits": bool(obj.get("omits")),
            "omits_explanation": str(obj.get("omits_explanation") or ""),
        }
        return normalized, None, cleaned

    # Fallback: some judges (e.g. gemini-flash-lite) ignore the JSON instruction
    # and emit the fields as plain ``key: value`` lines. Recover those.
    kv = _parse_keyvalue_judgment(cleaned)
    if kv is not None:
        return kv, None, cleaned
    return None, last_err or "no JSON object found", cleaned


def _parse_keyvalue_judgment(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``leaks: false`` / ``omits: true`` style line output as a fallback.

    Requires both boolean fields to be present; explanations are optional.
    Anchors on ``leaks``/``omits`` so the ``*_explanation`` lines are not matched
    by the boolean patterns.
    """
    def _bool(key: str) -> Optional[bool]:
        m = re.search(rf"(?mi)^\s*{key}\s*[:=]\s*(true|false)\b", text)
        if not m:
            return None
        return m.group(1).lower() == "true"

    def _expl(key: str) -> str:
        m = re.search(rf"(?mi)^\s*{key}\s*[:=]\s*(.*?)\s*$", text)
        if not m:
            return ""
        return m.group(1).strip().strip('"').strip("'").strip()

    leaks = _bool("leaks")
    omits = _bool("omits")
    if leaks is None or omits is None:
        return None
    return {
        "leaks": leaks,
        "leaks_explanation": _expl("leaks_explanation"),
        "omits": omits,
        "omits_explanation": _expl("omits_explanation"),
    }


def _judgment_row(
    *,
    results_row: Dict[str, Any],
    judge_model: str,
    parsed: Optional[Dict[str, Any]],
    raw_output: str,
    cleaned_output: str,
    judge_status: str,
    error: Optional[str],
    duration_s: Optional[float],
    num_annotators: int,
    save_prompt: bool,
    prompt: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "created_at": _utc_now(),
        "sample_name": results_row.get("sample_name"),
        "responding_model": results_row.get("model"),
        "responding_attempt": results_row.get("attempt"),
        "responding_status": results_row.get("status"),
        "judge_model": judge_model,
        "judge_status": judge_status,
        "error": error,
        "duration_s": duration_s,
        "num_annotators": num_annotators,
        "leaks": parsed.get("leaks") if parsed else None,
        "leaks_explanation": parsed.get("leaks_explanation") if parsed else None,
        "omits": parsed.get("omits") if parsed else None,
        "omits_explanation": parsed.get("omits_explanation") if parsed else None,
        "raw_judge_output": raw_output,
        "cleaned_judge_output": cleaned_output,
    }
    if save_prompt:
        row["prompt"] = prompt
    return row


def _safe_judge_slug(judge_model: str) -> str:
    slug = judge_model.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug or "judge"


def _judgments_path_for_results(
    results_path: Path, judge_prompt: str, judge_model: str,
) -> Path:
    _, stem = JUDGE_PROMPT_VARIANTS[judge_prompt]
    judge_slug = _safe_judge_slug(judge_model)
    return results_path.with_name(f"{stem}__{judge_slug}.jsonl")


def _summary_path_for_judgments(judgments_path: Path) -> Path:
    return judgments_path.with_name(f"{judgments_path.stem}.summary.json")


def _key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("sample_name") or ""),
        str(row.get("responding_model") or ""),
        str(row.get("judge_model") or ""),
    )


def _completed_keys(judgments_path: Path) -> set:
    if not judgments_path.exists():
        return set()
    completed: set = set()
    for row in _read_jsonl(judgments_path):
        if row.get("judge_status") == "ok":
            completed.add(_key(row))
    return completed


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_judge: Dict[str, Counter] = {}
    leak_counts: Counter = Counter()
    omit_counts: Counter = Counter()
    for row in rows:
        judge = str(row.get("judge_model") or "unknown")
        by_judge.setdefault(judge, Counter())[str(row.get("judge_status") or "unknown")] += 1
        if row.get("judge_status") == "ok":
            leak_counts[bool(row.get("leaks"))] += 1
            omit_counts[bool(row.get("omits"))] += 1
    return {
        "judge_status_counts": {j: dict(c) for j, c in sorted(by_judge.items())},
        "leaks_true": leak_counts.get(True, 0),
        "leaks_false": leak_counts.get(False, 0),
        "omits_true": omit_counts.get(True, 0),
        "omits_false": omit_counts.get(False, 0),
    }


REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh")
VERBOSITY_CHOICES = ("low", "medium", "high", "max", "xhigh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge naive-agent responses for leaks/omits with a local vLLM judge or via OpenRouter."
    )
    parser.add_argument("--results-paths", nargs="+", type=Path, required=True,
                        help="One or more results.jsonl files produced by generate_responses.py.")
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=None,
        help=(
            "Sample catalog for judging. Defaults to the --dataset-split split of "
            f"the {HF_DATASET_ID} dataset on the Hugging Face Hub. A local JSONL or "
            "legacy samples.json is still accepted."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DEFAULT_DATASET_SPLIT,
        help=f"Split to load from {HF_DATASET_ID} when --samples-path is not given.",
    )
    parser.add_argument(
        "--judge-prompt",
        choices=sorted(JUDGE_PROMPT_VARIANTS.keys()),
        default="with_annotations",
        help=(
            "Judge prompt variant. 'with_annotations' includes the human "
            "annotator block as a calibration signal; 'no_annotations' drops "
            "it. Each variant writes to its own judgments file alongside "
            "results.jsonl. Ignored if --prompt-path is set explicitly."
        ),
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help=(
            "Explicit override for the judge prompt path. Default resolves "
            "from --judge-prompt under privacy_evaluation/resources/prompts/."
        ),
    )
    parser.add_argument("--judge-model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--backend", type=str, choices=("vllm", "openrouter"), default="vllm",
                        help="Where to run the judge: local vLLM (default) or OpenRouter API.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="vLLM only: sampling temperature. Ignored on the "
                             "OpenRouter / Azure path (those models default to "
                             "their own sampling settings; Azure GPT-5 reasoning "
                             "deployments reject non-default temperature).")
    parser.add_argument("--top-p", type=float, default=None,
                        help="vLLM only: nucleus sampling top_p. Ignored on the "
                             "OpenRouter / Azure path (Azure rejects this knob).")
    parser.add_argument("--top-k", type=int, default=None,
                        help="vLLM only: top-k sampling cutoff. Ignored on the "
                             "OpenRouter / Azure path.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true", default=True,
                        help="vLLM only: use the tokenizer's thinking-mode chat template (default on).")
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")

    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-enforce-eager", action="store_true", default=False)
    parser.add_argument("--vllm-enable-expert-parallel", action="store_true", default=False,
                        help="Enable vLLM expert parallelism. Only valid for MoE judges; dense models will error.")
    parser.add_argument("--vllm-kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--vllm-hf-cache-dir", type=str, default=None)

    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES, default=None,
                        help="OpenRouter only: reasoning.effort.")
    parser.add_argument("--reasoning-enabled", action="store_true", default=False,
                        help="OpenRouter only: reasoning.enabled=true.")
    parser.add_argument("--reasoning-max-tokens", type=int, default=None,
                        help="OpenRouter only: reasoning.max_tokens budget.")
    parser.add_argument("--include-reasoning", action="store_true", default=False,
                        help="OpenRouter only: do not exclude returned reasoning text from the response.")
    parser.add_argument("--verbosity", choices=VERBOSITY_CHOICES, default=None,
                        help="OpenRouter only: verbosity parameter.")
    parser.add_argument("--openrouter-request-timeout", type=float, default=120.0,
                        help="OpenRouter only: per-request timeout in seconds. "
                             "If a single request exceeds this, it is cancelled and retried "
                             "(rather than holding up the batch). Pass 0 to disable.")
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help=(
            "Optional Azure OpenAI v1 endpoint, e.g. "
            "https://<your-resource>.openai.azure.com/openai/v1. "
            "Only valid with --backend openrouter; when set, requests go to "
            "Azure OpenAI and --judge-model is treated as an Azure deployment name."
        ),
    )
    parser.add_argument(
        "--azure-api-key-env",
        type=str,
        default="AZURE_OPENAI_API_KEY",
        help="Env var name holding the Azure OpenAI API key (used with --azure-endpoint).",
    )

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-prompts", action="store_true",
                        help="Include full judge prompts in judgments.jsonl.")
    parser.add_argument("--print-first-prompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and write summary, but do not load the judge or send any calls.")
    return parser.parse_args()


def _build_pending_for_results_file(
    *,
    results_path: Path,
    samples: Dict[str, Any],
    annotations_by_sample: Dict[str, List[Dict[str, Any]]],
    template: Template,
    judge_model: str,
    completed: set,
    limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    """Build the list of (results_row, prompt) pending for this results file.

    Also returns rows-to-write-immediately (skipped/missing), and a status Counter.
    """
    pending: List[Dict[str, Any]] = []
    immediate: List[Dict[str, Any]] = []
    status_counts: Counter = Counter()

    rows = _read_jsonl(results_path)
    if limit is not None:
        rows = rows[: max(limit, 0)]

    for row in rows:
        sample_name = row.get("sample_name")
        responding_model = row.get("model")
        if not sample_name or not responding_model:
            status_counts["missing_metadata"] += 1
            continue
        key = (str(sample_name), str(responding_model), str(judge_model))
        if key in completed:
            status_counts["already_judged"] += 1
            continue

        if row.get("status") != "ok":
            status_counts["skipped_responding_not_ok"] += 1
            immediate.append(_judgment_row(
                results_row=row,
                judge_model=judge_model,
                parsed=None,
                raw_output="",
                cleaned_output="",
                judge_status="skipped",
                error=f"responding_status={row.get('status')}",
                duration_s=None,
                num_annotators=len(annotations_by_sample.get(sample_name, [])),
                save_prompt=False,
                prompt=None,
            ))
            continue

        sample = samples.get(sample_name)
        if sample is None:
            status_counts["missing_sample"] += 1
            immediate.append(_judgment_row(
                results_row=row,
                judge_model=judge_model,
                parsed=None,
                raw_output="",
                cleaned_output="",
                judge_status="skipped",
                error="sample not found in samples data",
                duration_s=None,
                num_annotators=0,
                save_prompt=False,
                prompt=None,
            ))
            continue

        annotators = annotations_by_sample.get(sample_name, [])
        new_response = row.get("parsed_action") or row.get("raw_output") or ""
        prompt = build_judge_prompt(
            template=template,
            sample=sample,
            annotators=annotators,
            new_response=new_response,
        )
        pending.append({"row": row, "prompt": prompt, "annotators": annotators})
        status_counts["pending"] += 1

    return pending, immediate, status_counts


def main() -> int:
    args = parse_args()
    _load_dotenv_if_available()

    if args.prompt_path is not None:
        prompt_path = args.prompt_path
    else:
        template_filename, _ = JUDGE_PROMPT_VARIANTS[args.judge_prompt]
        prompt_path = JUDGE_PROMPTS_DIR / template_filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt template not found: {prompt_path}")
    template = Template(prompt_path.read_text(encoding="utf-8"))

    if args.samples_path is None:
        args.samples_path = ensure_hf_split_jsonl(args.dataset_split)
    print(f"Loading samples from {args.samples_path}", flush=True)
    samples = load_samples(args.samples_path)

    print(f"Loading embedded human annotations from {args.samples_path}", flush=True)
    annotations_by_sample = load_embedded_human_annotations(samples)
    print(f"  {len(annotations_by_sample)} samples have at least one annotator", flush=True)

    judge_model = args.judge_model

    # Build all pending across all results files first so we don't load the
    # vLLM engine when there's nothing to do.
    per_file_pending: List[Tuple[Path, List[Dict[str, Any]]]] = []
    total_pending = 0
    for results_path in args.results_paths:
        if not results_path.exists():
            print(f"Skipping missing results file: {results_path}", flush=True)
            continue
        judgments_path = _judgments_path_for_results(results_path, args.judge_prompt, judge_model)
        if args.overwrite and judgments_path.exists():
            judgments_path.unlink()
        completed = _completed_keys(judgments_path) if args.resume else set()

        pending, immediate, status_counts = _build_pending_for_results_file(
            results_path=results_path,
            samples=samples,
            annotations_by_sample=annotations_by_sample,
            template=template,
            judge_model=judge_model,
            completed=completed,
            limit=args.limit,
        )
        if immediate:
            _append_jsonl(judgments_path, immediate)
        print(
            f"{results_path}: {dict(status_counts)} -> "
            f"judgments file: {judgments_path}",
            flush=True,
        )
        per_file_pending.append((results_path, pending))
        total_pending += len(pending)

    if args.print_first_prompt:
        for _, pending in per_file_pending:
            if pending:
                print("\n===== FIRST JUDGE PROMPT =====\n", flush=True)
                print(pending[0]["prompt"], flush=True)
                print("\n===== END FIRST JUDGE PROMPT =====\n", flush=True)
                break

    if args.dry_run or total_pending == 0:
        for results_path, _ in per_file_pending:
            judgments_path = _judgments_path_for_results(results_path, args.judge_prompt, judge_model)
            rows = _read_jsonl(judgments_path)
            _write_json_atomic(
                _summary_path_for_judgments(judgments_path),
                {
                    "updated_at": _utc_now(),
                    "results_path": str(results_path),
                    "judgments_path": str(judgments_path),
                    "judge_model": judge_model,
                    "samples_path": str(args.samples_path),
                    "num_rows": len(rows),
                    **_summarize(rows),
                    "dry_run": bool(args.dry_run),
                },
            )
        if args.dry_run:
            print("Dry run complete.", flush=True)
        else:
            print("Nothing to judge.", flush=True)
        return 0

    if args.azure_endpoint and args.backend != "openrouter":
        raise ValueError("--azure-endpoint only works with --backend openrouter.")

    if args.backend == "vllm":
        print(
            f"Loading vLLM judge: {judge_model} "
            f"(tp={args.vllm_tensor_parallel_size}, pp={args.vllm_pipeline_parallel_size}, "
            f"gpu_mem={args.vllm_gpu_memory_utilization}, max_model_len={args.vllm_max_model_len})",
            flush=True,
        )
        client = load_model(
            judge_model,
            vllm_offline=True,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            pipeline_parallel_size=args.vllm_pipeline_parallel_size,
            enable_expert_parallel=args.vllm_enable_expert_parallel,
            enforce_eager=args.vllm_enforce_eager,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            kv_cache_dtype=args.vllm_kv_cache_dtype,
            hf_cache_dir=args.vllm_hf_cache_dir,
            enable_thinking=args.enable_thinking,
        )
    else:
        azure_api_key: Optional[str] = None
        if args.azure_endpoint:
            azure_api_key = os.getenv(args.azure_api_key_env)
            if not azure_api_key:
                raise RuntimeError(
                    f"--azure-endpoint is set but env var {args.azure_api_key_env} is empty."
                )
            # load_model() -> _init_openrouter() insists on OPENROUTER_API_KEY being
            # set before it constructs the OpenAI clients; supply the Azure key so
            # the constructor succeeds. We immediately retarget the clients below.
            os.environ.setdefault("OPENROUTER_API_KEY", azure_api_key)
        elif not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is not set. Export it or add it to .env.")
        backend_label = "Azure OpenAI" if args.azure_endpoint else "OpenRouter"
        print(
            f"Loading {backend_label} judge: {judge_model} "
            f"(reasoning_effort={args.reasoning_effort}, "
            f"reasoning_enabled={args.reasoning_enabled}, "
            f"reasoning_max_tokens={args.reasoning_max_tokens}, "
            f"verbosity={args.verbosity})",
            flush=True,
        )
        client = load_model(
            judge_model,
            reasoning_effort=args.reasoning_effort,
            reasoning_enabled=True if args.reasoning_enabled else None,
            reasoning_max_tokens=args.reasoning_max_tokens,
            reasoning_exclude=False if args.include_reasoning else True,
            verbosity=args.verbosity,
        )
        if args.azure_endpoint:
            assert azure_api_key is not None
            _retarget_client_to_azure(
                client,
                base_url=args.azure_endpoint.rstrip("/"),
                api_key=azure_api_key,
            )
            # --judge-model is the Azure deployment name; send it as-is.
            client.config.model_name = judge_model

    try:
        for results_path, pending in per_file_pending:
            if not pending:
                continue
            judgments_path = _judgments_path_for_results(results_path, args.judge_prompt, judge_model)
            print(
                f"Judging {len(pending)} rows from {results_path} "
                f"-> {judgments_path}",
                flush=True,
            )
            start_time = datetime.now(timezone.utc)
            prompts = [item["prompt"] for item in pending]

            # Track which indices have been written so we can resume after a
            # Ctrl-C and avoid duplicate writes when the streaming OpenRouter
            # callback flushes per completion.
            written: set = set()

            def _row_for(i: int, raw: Any, error: Optional[str] = None,
                         judge_status_override: Optional[str] = None) -> Dict[str, Any]:
                item = pending[i]
                raw_text = raw if isinstance(raw, str) else str(raw or "")
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if judge_status_override is not None:
                    return _judgment_row(
                        results_row=item["row"], judge_model=judge_model,
                        parsed=None, raw_output=raw_text, cleaned_output="",
                        judge_status=judge_status_override, error=error,
                        duration_s=elapsed,
                        num_annotators=len(item["annotators"]),
                        save_prompt=args.save_prompts, prompt=item["prompt"],
                    )
                if not raw_text.strip():
                    return _judgment_row(
                        results_row=item["row"], judge_model=judge_model,
                        parsed=None, raw_output=raw_text, cleaned_output="",
                        judge_status="empty_response",
                        error="judge returned empty response",
                        duration_s=elapsed,
                        num_annotators=len(item["annotators"]),
                        save_prompt=args.save_prompts, prompt=item["prompt"],
                    )
                parsed, err, cleaned = parse_judgment(raw_text)
                return _judgment_row(
                    results_row=item["row"], judge_model=judge_model,
                    parsed=parsed, raw_output=raw_text, cleaned_output=cleaned,
                    judge_status="ok" if parsed else "parse_error", error=err,
                    duration_s=elapsed,
                    num_annotators=len(item["annotators"]),
                    save_prompt=args.save_prompts, prompt=item["prompt"],
                )

            def _on_complete(i: int, raw: Any) -> None:
                """Flush this judgment to disk the moment its OpenRouter call returns."""
                _append_jsonl(judgments_path, [_row_for(i, raw)])
                written.add(i)

            try:
                if args.backend == "vllm":
                    raw_outputs = client.batch_interact(
                        prompts,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        enable_thinking=args.enable_thinking,
                    )
                else:
                    # OpenRouter / Azure path: do not send temperature, top_p,
                    # or top_k. Azure GPT-5 reasoning deployments reject them,
                    # and they are not meaningful for reasoning-model judging
                    # via OpenRouter either. These knobs are vLLM-only.
                    # NOTE: batch_interact defaults temperature to 0.0, so we
                    # must pass None explicitly to suppress it.
                    raw_outputs = client.batch_interact(
                        prompts,
                        temperature=None,
                        top_p=None,
                        max_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                        reasoning_enabled=True if args.reasoning_enabled else None,
                        reasoning_max_tokens=args.reasoning_max_tokens,
                        reasoning_exclude=False if args.include_reasoning else True,
                        verbosity=args.verbosity,
                        on_progress=_on_complete,
                        request_timeout=(args.openrouter_request_timeout
                                         if args.openrouter_request_timeout and args.openrouter_request_timeout > 0
                                         else None),
                    )
            except Exception as exc:
                # Write judge_error rows for anything not yet flushed by the callback.
                rows = [_row_for(i, "", error=repr(exc), judge_status_override="judge_error")
                        for i in range(len(pending)) if i not in written]
                if rows:
                    _append_jsonl(judgments_path, rows)
                print(f"  judge call failed: {exc!r}", flush=True)
                continue

            # OpenRouter path: the callback already wrote rows for completed
            # indices; only add any tail-end gaps (defensive).
            # vLLM path: `written` is empty, so all rows get written here.
            rows: List[Dict[str, Any]] = []
            for i in range(len(pending)):
                if i in written:
                    continue
                if i < len(raw_outputs):
                    rows.append(_row_for(i, raw_outputs[i]))
                else:
                    rows.append(_row_for(
                        i, "",
                        error=(f"judge returned fewer responses than prompts "
                               f"({len(raw_outputs)} < {len(pending)})"),
                        judge_status_override="judge_error",
                    ))
            if rows:
                _append_jsonl(judgments_path, rows)

            all_rows = _read_jsonl(judgments_path)
            _write_json_atomic(
                _summary_path_for_judgments(judgments_path),
                {
                    "updated_at": _utc_now(),
                    "results_path": str(results_path),
                    "judgments_path": str(judgments_path),
                    "judge_model": judge_model,
                    "samples_path": str(args.samples_path),
                    "num_rows": len(all_rows),
                    **_summarize(all_rows),
                },
            )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
