#!/usr/bin/env python3
"""Generate responses over PrivacyAlign test samples via OpenRouter or Azure OpenAI.

Pick the agent prompt with --prompt {naive,privacy_enhanced}. Each prompt
variant writes to its own runs subdirectory so resume is isolated:

  --prompt naive             -> privacy_evaluation/runs/naive_openrouter_test/<model_slug>/results.jsonl
  --prompt privacy_enhanced  -> privacy_evaluation/runs/privacy_enhanced_openrouter_test/<model_slug>/results.jsonl

# Reasoning knobs by model family (OpenRouter docs, May 2026):
#  - Opus 4.7: --reasoning-effort is accepted but IGNORED. Thinking depth on
#    4.7 is fully adaptive with NO user-facing knob; the only thinking
#    control is --reasoning-enabled (on/off). --verbosity is a SEPARATE,
#    orthogonal output-effort knob (maps to Anthropic output_config.effort)
#    and does NOT influence reasoning depth.
#  - Sonnet 4.6: --reasoning-effort high works (maps to a thinking budget).
#  - Gemini 3.1 (Pro / Flash Lite): --reasoning-effort high maps to Google's
#    thinkingLevel.
#  - GPT-5.5 / 5.4-mini via Azure OpenAI v1: --reasoning-effort high. The
#    Azure path translates the OpenRouter-shaped nested `reasoning` object
#    into Azure Chat Completions' flat top-level `reasoning_effort` scalar
#    (see _translate_extra_body_for_azure). Requires a reasoning-capable
#    GPT-5 deployment.
# OpenAI models are accessed via Azure OpenAI only (not OpenRouter). Azure
# key is read from AZURE_OPENAI_API_KEY by default.

# --- naive prompt ---

# Opus 4.7 (adaptive thinking on; reasoning depth not user-controllable)
python3 privacy_evaluation/generate_responses.py \
  --prompt naive \
  --models anthropic/claude-opus-4.7 \
  --reasoning-enabled \
  --max-tokens 65536 \
  --retry-errors

# Sonnet 4.6 (effort=high)
python3 privacy_evaluation/generate_responses.py \
  --prompt naive \
  --models anthropic/claude-sonnet-4.6 \
  --reasoning-effort high \
  --max-tokens 65536 \
  --retry-errors

# Gemini 3.1 Pro + Flash Lite (effort=high)
python3 privacy_evaluation/generate_responses.py \
  --prompt naive \
  --models google/gemini-3.1-pro-preview google/gemini-3.1-flash-lite-preview \
  --reasoning-effort high \
  --max-tokens 65536 \
  --retry-errors

# GPT-5.5 + GPT-5.4-mini via Azure OpenAI (effort=high)
python3 privacy_evaluation/generate_responses.py \
  --prompt naive \
  --azure-endpoint https://<your-resource>.openai.azure.com/openai/v1 \
  --models gpt-5.5 gpt-5.4-mini \
  --reasoning-effort high \
  --max-tokens 65536 \
  --retry-errors

# --- privacy_enhanced prompt ---

# Opus 4.7 (adaptive thinking on; reasoning depth not user-controllable)
python3 privacy_evaluation/generate_responses.py \
  --prompt privacy_enhanced \
  --models anthropic/claude-opus-4.7 \
  --reasoning-enabled \
  --max-tokens 65536 \
  --retry-errors

# Sonnet 4.6 (effort=high)
python3 privacy_evaluation/generate_responses.py \
  --prompt privacy_enhanced \
  --models anthropic/claude-sonnet-4.6 \
  --reasoning-effort high \
  --max-tokens 65536 \
  --retry-errors

# Gemini 3.1 Pro + Flash Lite (effort=high)
python3 privacy_evaluation/generate_responses.py \
  --prompt privacy_enhanced \
  --models google/gemini-3.1-pro-preview google/gemini-3.1-flash-lite-preview \
  --reasoning-effort high \
  --max-tokens 65536 \
  --retry-errors

# GPT-5.5 + GPT-5.4-mini via Azure OpenAI (effort=high)
python3 privacy_evaluation/generate_responses.py \
  --prompt privacy_enhanced \
  --azure-endpoint https://<your-resource>.openai.azure.com/openai/v1 \
  --models gpt-5.5 gpt-5.4-mini \
  --reasoning-effort high \
  --max-tokens 65536 \
  --retry-errors

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
# PrivacyAlign is distributed via the Hugging Face Hub. By default the requested
# split is downloaded and cached locally as JSONL so the rest of the pipeline can
# treat it like any other --dataset-path file.
HF_DATASET_ID = "ServiceNow/PrivacyAlign"
DEFAULT_DATASET_SPLIT = "test"
DATASET_CACHE_DIR = REPO_ROOT / ".cache" / "privalign-dataset"
DEFAULT_PROMPTS_DIR = GENERATION_DIR / "resources" / "prompts"
DEFAULT_TOOLKIT_SPECS_PATH = GENERATION_DIR / "resources" / "assets" / "all_toolkits.json"
RUNS_ROOT = REPO_ROOT / "privacy_evaluation" / "runs"

# --prompt -> (template filename, default runs subdir).
# Each prompt variant gets its own runs subdir so resume isolates results by
# prompt and existing naive runs are preserved.
PROMPT_VARIANTS: Dict[str, Tuple[str, str]] = {
    "naive": ("naive_agent_prompt.txt", "naive_openrouter_test"),
    "privacy_enhanced": ("privacy_enhanced_agent_prompt.txt", "privacy_enhanced_openrouter_test"),
}
REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high", "xhigh")
VERBOSITY_CHOICES = ("low", "medium", "high", "max", "xhigh")
RETRYABLE_STATUSES = {"api_error", "empty_response", "parse_error"}


if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from model_client import load_model  # noqa: E402
from pipeline.context import PipelineContext  # noqa: E402
from pipeline.leakage_filter import build_naive_agent_prompt, parse_naive_agent_result  # noqa: E402
from pipeline.toolkit_registry import ToolkitRegistry  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _short_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _safe_model_slug(model_name: str) -> str:
    slug = model_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug or "model"


def default_output_path_for_model(model_name: str, prompt_variant: str) -> Path:
    _, runs_subdir = PROMPT_VARIANTS[prompt_variant]
    return RUNS_ROOT / runs_subdir / _safe_model_slug(model_name) / "results.jsonl"


def default_summary_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(".summary.json")


def prompt_to_chat_messages(prompt: Any) -> List[Dict[str, str]]:
    """Mirror model_client's OpenRouter prompt splitting for inspectability."""
    if isinstance(prompt, list):
        return [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in prompt
            if isinstance(message, dict)
        ]

    text = str(prompt)
    if "\n===\n" in text:
        system_part, user_part = text.split("\n===\n", 1)
        return [
            {"role": "system", "content": system_part.strip()},
            {"role": "user", "content": user_part.strip()},
        ]
    return [{"role": "user", "content": text}]


def print_first_prompt_messages(prompt: str) -> None:
    print("\n===== FIRST PROMPT AS OPENROUTER MESSAGES =====\n", flush=True)
    for idx, message in enumerate(prompt_to_chat_messages(prompt), start=1):
        role = message["role"].upper()
        print(f"--- MESSAGE {idx}: {role} ---", flush=True)
        print(message["content"], flush=True)
        print("", flush=True)
    print("===== END FIRST PROMPT =====\n", flush=True)


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
    sends self.config.model_name, set to the user-provided --models value).
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


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(GENERATION_DIR / ".env")


def load_prompt_context(
    prompts_dir: Path,
    toolkit_specs_path: Path,
    *,
    prompt_variant: str = "naive",
) -> PipelineContext:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    template_filename, _ = PROMPT_VARIANTS[prompt_variant]
    prompt_path = prompts_dir / template_filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    registry = ToolkitRegistry(toolkit_specs_path=toolkit_specs_path)
    prompt_templates = {
        template_filename: Template(prompt_path.read_text(encoding="utf-8")),
    }

    return PipelineContext(
        args=argparse.Namespace(
            print_prompts=False,
            diverse_generation_temperature=0.0,
            filter_temperature=0.0,
            reasoning_effort=None,
            filter_reasoning_effort=None,
            filter_top_p=1.0,
            filter_top_k=-1,
        ),
        registry=registry,
        prompt_templates=prompt_templates,
        seed_options={},
        model=None,
        diversity=None,
        name_rows=[],
    )


def get_default_models(manifest: Dict[str, Any]) -> List[str]:
    models = manifest.get("source_metadata", {}).get("response_models", [])
    if isinstance(models, list):
        cleaned = [str(model).strip() for model in models if str(model).strip()]
        if cleaned:
            return cleaned
    return ["openai/gpt-oss-120b"]


def unique_ordered(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def ensure_hf_split_jsonl(split: str = DEFAULT_DATASET_SPLIT) -> Path:
    """Download a PrivacyAlign split from the HF Hub and cache it as JSONL.

    Returns the path to the cached JSONL so the caller can treat it exactly like
    an explicit ``--dataset-path`` file.
    """
    cache_path = DATASET_CACHE_DIR / f"{split}.jsonl"
    if cache_path.exists():
        return cache_path
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Loading the default PrivacyAlign dataset requires the 'datasets' package "
            "(pip install datasets), or pass an explicit --dataset-path."
        ) from exc
    dataset = load_dataset(HF_DATASET_ID, split=split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return cache_path


def load_annotation_test_items(
    *,
    dataset_path: Optional[Path] = None,
    manifest_path: Optional[Path],
    samples_path: Optional[Path],
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if dataset_path is not None:
        items = _read_jsonl(dataset_path)
        if limit is not None:
            items = items[: max(limit, 0)]
        manifest = {
            "dataset_path": str(dataset_path),
            "source_metadata": {"response_models": []},
            "test_names": [_sample_name_for_item(item) for item in items],
        }
        return manifest, items

    if manifest_path is None or samples_path is None:
        raise ValueError(
            "Legacy batches mode requires both --manifest-path and --samples-path."
        )
    manifest = _read_json(manifest_path)
    samples = _read_json(samples_path)

    test_names = manifest.get("test_names")
    if not isinstance(test_names, list):
        raise ValueError(f"Manifest does not contain a list-valued test_names: {manifest_path}")

    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for raw_name in test_names:
        name = str(raw_name)
        sample = samples.get(name) if isinstance(samples, dict) else None
        if sample is None:
            missing.append(name)
            continue
        items.append(sample)

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} test sample names are missing from {samples_path}: {preview}")

    if limit is not None:
        items = items[: max(limit, 0)]

    return manifest, items


def _sample_name_for_item(annotation_item: Dict[str, Any]) -> str:
    for key in ("name", "sample_name", "id"):
        value = annotation_item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    source = annotation_item.get("source")
    if isinstance(source, dict):
        for key in ("name", "sample_name", "id"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return _short_sha1(json.dumps(annotation_item, sort_keys=True))


def _resolve_final_action(annotation_item: Dict[str, Any]) -> str:
    value = annotation_item.get("expected_final_action")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for response_key in ("response_a", "response_b"):
        response = annotation_item.get(response_key)
        if isinstance(response, dict):
            tool_name = response.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                return tool_name.strip()
    value = annotation_item.get("generated_final_action")
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    raise ValueError(f"Privalign item {_sample_name_for_item(annotation_item)} has no final action")


def _normalize_privalign_item(
    annotation_item: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Convert an exported privalign-dataset row into prompt-builder inputs."""
    final_action = _resolve_final_action(annotation_item)
    user_name = str(annotation_item.get("user_name") or "")
    trajectory = {
        "user_name": user_name,
        "user_email": str(annotation_item.get("user_email") or ""),
        "user_instruction": str(annotation_item.get("user_instruction") or ""),
        "toolkits": annotation_item.get("toolkits") or [],
        "executable_trajectory": str(annotation_item.get("trajectory") or ""),
        "final_action": final_action,
    }
    seed_candidate = {
        "data_subject": user_name,
        "data_sender": user_name,
        "data_recipient": "",
        "final_action": final_action,
        "toolkits": trajectory["toolkits"],
        "domains": annotation_item.get("domains") or [],
    }
    vignette = {
        "memories": annotation_item.get("memories") or [],
        "sensitive_info_items": [],
        "relevant_info_items": [],
    }
    return {"trajectory": trajectory}, seed_candidate, vignette


def reconstruct_prompt_inputs(
    annotation_item: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Convert an PrivacyAlign catalog item into prompt-builder inputs."""
    if "source" not in annotation_item and "trajectory" in annotation_item:
        return _normalize_privalign_item(annotation_item)

    source = annotation_item.get("source")
    if not isinstance(source, dict):
        source = annotation_item

    trajectory = source["trajectory"]
    seed = source["seed"]
    vignette = dict(source["vignette"])
    vignette["memories"] = source.get("memories", vignette.get("memories", []))

    metadata = source.get("generation_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    raw_domains = metadata.get("domains") or seed.get("domains")
    if not raw_domains:
        legacy_domain = metadata.get("scenario_domain") or seed.get("scenario_domain")
        raw_domains = [legacy_domain] if legacy_domain else []

    seed_candidate = {
        "data_subject": seed["data_subject"],
        "data_sender": seed["data_sender"],
        "data_recipient": seed["data_recipient"],
        "final_action": trajectory["final_action"],
        "toolkits": trajectory["toolkits"],
        "domains": raw_domains,
    }
    return {"trajectory": trajectory}, seed_candidate, vignette


def build_eval_case(
    ctx: PipelineContext,
    annotation_item: Dict[str, Any],
    *,
    prompt_variant: str = "naive",
) -> Dict[str, Any]:
    template_filename, _ = PROMPT_VARIANTS[prompt_variant]
    trajectory_payload, seed_candidate, vignette = reconstruct_prompt_inputs(annotation_item)
    prompt = build_naive_agent_prompt(
        ctx, trajectory_payload, seed_candidate, vignette,
        template_name=template_filename,
    )
    source = annotation_item.get("source", annotation_item)
    trajectory = trajectory_payload["trajectory"]
    return {
        "sample_name": _sample_name_for_item(annotation_item),
        "prompt": prompt,
        "prompt_sha1": _short_sha1(prompt),
        "expected_final_action": trajectory["final_action"],
        "toolkits": trajectory.get("toolkits", []),
        "user_instruction": trajectory.get("user_instruction", ""),
        "source_model_name": source.get("model_name") or source.get("source_model_name"),
        "pair_family": annotation_item.get("pair_family"),
        "comparative_verdict": annotation_item.get("comparative_verdict"),
        "original_action": annotation_item.get("original_action"),
        "new_action": annotation_item.get("new_action"),
    }


def _rows_by_model_sample(output_path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if not output_path.exists():
        return grouped

    for row in _load_existing_rows(output_path):
        model = row.get("model")
        sample_name = row.get("sample_name")
        if not model or not sample_name:
            continue
        grouped.setdefault((str(model), str(sample_name)), []).append(row)
    return grouped


def _attempt_count(rows: Sequence[Dict[str, Any]], prompt_sha1: str) -> int:
    return sum(1 for row in rows if row.get("prompt_sha1") == prompt_sha1)


def _latest_matching_row(
    rows: Sequence[Dict[str, Any]],
    prompt_sha1: str,
) -> Optional[Dict[str, Any]]:
    for row in reversed(rows):
        if row.get("prompt_sha1") == prompt_sha1:
            return row
    return None


def should_run_case(
    *,
    existing_rows: Sequence[Dict[str, Any]],
    prompt_sha1: str,
    retry_errors: bool,
    max_attempts: int,
) -> bool:
    latest = _latest_matching_row(existing_rows, prompt_sha1)
    if latest is None:
        return True
    if latest.get("status") == "ok":
        return False
    if not retry_errors:
        return False
    if latest.get("status") not in RETRYABLE_STATUSES:
        return False
    return _attempt_count(existing_rows, prompt_sha1) < max_attempts


def _completed_keys(output_path: Path, retry_errors: bool) -> Dict[Tuple[str, str], str]:
    completed: Dict[Tuple[str, str], str] = {}
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = row.get("model")
            sample_name = row.get("sample_name")
            prompt_sha1 = row.get("prompt_sha1")
            if not model or not sample_name or not prompt_sha1:
                continue
            if retry_errors and row.get("status") in RETRYABLE_STATUSES:
                continue
            completed[(str(model), str(sample_name))] = str(prompt_sha1)
    return completed


def _load_existing_rows(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Counter] = {}
    for row in rows:
        model = str(row.get("model") or "unknown")
        by_model.setdefault(model, Counter())[str(row.get("status") or "unknown")] += 1
    return {
        model: dict(counter)
        for model, counter in sorted(by_model.items())
    }


def _summarize_latest_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        model = row.get("model")
        sample_name = row.get("sample_name")
        if not model or not sample_name:
            continue
        latest[(str(model), str(sample_name))] = row

    by_model: Dict[str, Counter] = {}
    for (model, _sample_name), row in latest.items():
        by_model.setdefault(model, Counter())[str(row.get("status") or "unknown")] += 1
    return {
        model: dict(counter)
        for model, counter in sorted(by_model.items())
    }


def write_summary(
    *,
    summary_path: Path,
    output_path: Path,
    dataset_path: Optional[Path],
    manifest_path: Path,
    samples_path: Path,
    models: Sequence[str],
    num_cases: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    rows = _load_existing_rows(output_path)
    payload: Dict[str, Any] = {
        "updated_at": _utc_now(),
        "output_path": str(output_path),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "manifest_path": str(manifest_path),
        "samples_path": str(samples_path),
        "models": list(models),
        "num_cases": num_cases,
        "num_rows": len(rows),
        "status_counts_by_model": _summarize_rows(rows),
        "latest_status_counts_by_model": _summarize_latest_rows(rows),
    }
    if extra:
        payload.update(extra)
    _write_json_atomic(summary_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses over PrivacyAlign test samples using OpenRouter "
            "or Azure OpenAI. Pick the agent prompt with --prompt."
        )
    )
    parser.add_argument(
        "--prompt",
        choices=sorted(PROMPT_VARIANTS.keys()),
        default="naive",
        help=(
            "Agent prompt to use. 'naive' is the default leakage-phase-1 prompt; "
            "'privacy_enhanced' adds an instruction to consider sender/recipient and "
            "withhold information inappropriate for the recipient."
        ),
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Privalign JSONL split to evaluate. Defaults to the --dataset-split split "
            f"of the {HF_DATASET_ID} dataset on the Hugging Face Hub. Use "
            "--use-legacy-batches to read --manifest-path/--samples-path instead."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DEFAULT_DATASET_SPLIT,
        help=f"Split to load from {HF_DATASET_ID} when --dataset-path is not given.",
    )
    parser.add_argument(
        "--use-legacy-batches",
        action="store_true",
        help="Read a local manifest.json + samples.json (--manifest-path/--samples-path) instead of --dataset-path.",
    )
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--toolkit-specs-path", type=Path, default=DEFAULT_TOOLKIT_SPECS_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit results JSONL path. By default each model writes to "
            "privacy_evaluation/runs/<prompt-subdir>/<model_slug>/results.jsonl."
        ),
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="OpenRouter model IDs. Defaults to manifest source_metadata.response_models.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N test samples.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Accepted but unused: this script only runs through "
                             "OpenRouter / Azure, where temperature is not sent. "
                             "Azure GPT-5 reasoning deployments reject non-default "
                             "temperature.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Accepted but unused: this script only runs through "
                             "OpenRouter / Azure, where top_p is not sent. Azure "
                             "GPT-5 reasoning deployments reject this knob.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=None,
        help="Optional OpenRouter reasoning effort to send as reasoning.effort.",
    )
    parser.add_argument(
        "--reasoning-enabled",
        action="store_true",
        help="Send OpenRouter reasoning.enabled=true. For Claude 4.6 this enables adaptive thinking.",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help="Optional OpenRouter reasoning.max_tokens budget. For Claude 4.6 this uses budget-based thinking.",
    )
    parser.add_argument(
        "--verbosity",
        choices=VERBOSITY_CHOICES,
        default=None,
        help="Optional OpenRouter verbosity parameter. For Claude 4.6 this maps to Anthropic output_config.effort.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Ask OpenRouter to return reasoning text when available. By default it is excluded.",
    )
    parser.add_argument("--openrouter-request-timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds. If a single OpenRouter call "
                             "exceeds this, it is cancelled and retried (rather than holding up "
                             "the batch). Pass 0 to disable.")
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help=(
            "Optional Azure OpenAI v1 endpoint, e.g. "
            "https://<your-resource>.openai.azure.com/openai/v1. "
            "When set, requests go to Azure OpenAI instead of OpenRouter and "
            "--models values are treated as Azure deployment names."
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
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry api_error, empty_response, and parse_error rows while below --max-attempts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum attempts per model/sample/prompt hash when --retry-errors is set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts but do not call OpenRouter.")
    parser.add_argument("--print-first-prompt", action="store_true")
    parser.add_argument("--save-prompts", action="store_true", help="Include full prompts in results JSONL.")
    return parser.parse_args()


def _validate_output_path(output_path: Path, *, overwrite: bool, resume: bool) -> None:
    if overwrite and output_path.exists():
        output_path.unlink()
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output exists and --no-resume was set: {output_path}. "
            "Use --overwrite to start fresh."
        )


def _result_row(
    *,
    case: Dict[str, Any],
    model: str,
    attempt: int,
    max_attempts: int,
    raw_output: str,
    status: str,
    parsed_action: Optional[str],
    error: Optional[str],
    duration_s: Optional[float],
    save_prompt: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "created_at": _utc_now(),
        "sample_name": case["sample_name"],
        "model": model,
        "model_slug": _safe_model_slug(model),
        "status": status,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "prompt_sha1": case["prompt_sha1"],
        "expected_final_action": case["expected_final_action"],
        "toolkits": case["toolkits"],
        "user_instruction": case["user_instruction"],
        "source_model_name": case["source_model_name"],
        "pair_family": case["pair_family"],
        "comparative_verdict": case["comparative_verdict"],
        "raw_output": raw_output,
        "parsed_action": parsed_action,
        "error": error,
        "duration_s": duration_s,
    }
    if save_prompt:
        row["prompt"] = case["prompt"]
    return row


def _pending_cases_and_latest_counts(
    *,
    cases: Sequence[Dict[str, Any]],
    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]],
    model: str,
    retry_errors: bool,
    max_attempts: int,
) -> Tuple[List[Dict[str, Any]], Counter]:
    pending: List[Dict[str, Any]] = []
    progress_counts: Counter = Counter()
    for case in cases:
        rows = grouped_rows.get((model, case["sample_name"]), [])
        latest = _latest_matching_row(rows, case["prompt_sha1"])
        should_run = should_run_case(
            existing_rows=rows,
            prompt_sha1=case["prompt_sha1"],
            retry_errors=retry_errors,
            max_attempts=max_attempts,
        )
        if should_run:
            pending.append(case)

        if latest is None:
            progress_counts["not_started"] += 1
            continue

        status = str(latest.get("status") or "unknown")
        if status == "ok":
            progress_counts["ok"] += 1
        elif status in RETRYABLE_STATUSES:
            if should_run:
                progress_counts["retryable_pending"] += 1
            elif retry_errors and _attempt_count(rows, case["prompt_sha1"]) >= max_attempts:
                progress_counts["exhausted"] += 1
            else:
                progress_counts["failed_retry_disabled"] += 1
        else:
            progress_counts["other_final"] += 1
    return pending, progress_counts


def _print_model_progress(
    *,
    model: str,
    pending: Sequence[Dict[str, Any]],
    progress_counts: Counter,
    output_path: Path,
    retry_pass: int,
) -> None:
    parts = [
        f"{progress_counts.get('ok', 0)} ok",
    ]
    if progress_counts.get("not_started", 0):
        parts.append(f"{progress_counts['not_started']} not started")
    if progress_counts.get("retryable_pending", 0):
        parts.append(f"{progress_counts['retryable_pending']} retryable")
    if progress_counts.get("exhausted", 0):
        parts.append(f"{progress_counts['exhausted']} exhausted")
    if progress_counts.get("failed_retry_disabled", 0):
        parts.append(f"{progress_counts['failed_retry_disabled']} failed/retry disabled")
    if progress_counts.get("other_final", 0):
        parts.append(f"{progress_counts['other_final']} other final")

    print(
        f"Model {model}: pass {retry_pass}: {len(pending)} pending "
        f"({', '.join(parts)}). Results: {output_path}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    _load_dotenv_if_available()

    if not args.use_legacy_batches and args.dataset_path is None:
        args.dataset_path = ensure_hf_split_jsonl(args.dataset_split)

    manifest, annotation_items = load_annotation_test_items(
        dataset_path=None if args.use_legacy_batches else args.dataset_path,
        manifest_path=args.manifest_path,
        samples_path=args.samples_path,
        limit=args.limit,
    )
    models = unique_ordered(args.models or get_default_models(manifest))
    if not models:
        raise ValueError("No models specified.")

    ctx = load_prompt_context(
        args.prompts_dir, args.toolkit_specs_path, prompt_variant=args.prompt,
    )
    cases = [
        build_eval_case(ctx, item, prompt_variant=args.prompt)
        for item in annotation_items
    ]

    if args.output_path is not None and len(models) > 1:
        raise ValueError(
            "--output-path can only be used with one model. "
            "Omit --output-path to get one default results file per model."
        )
    if args.summary_path is not None and len(models) > 1:
        raise ValueError(
            "--summary-path can only be used with one model. "
            "Omit --summary-path to get one default summary file per model."
        )

    print(
        f"Loaded {len(cases)} test samples from "
        f"{args.manifest_path if args.use_legacy_batches else args.dataset_path} "
        f"for {len(models)} model(s).",
        flush=True,
    )

    if args.print_first_prompt and cases:
        print_first_prompt_messages(cases[0]["prompt"])

    if args.dry_run:
        for model in models:
            output_path = args.output_path or default_output_path_for_model(model, args.prompt)
            summary_path = args.summary_path or default_summary_path_for_output(output_path)
            print(
                f"Dry run complete for {model}. No files written. "
                f"Would write results to {output_path} and summary to {summary_path}.",
                flush=True,
            )
        return 0

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

    for model in models:
        output_path = args.output_path or default_output_path_for_model(model, args.prompt)
        summary_path = args.summary_path or default_summary_path_for_output(output_path)
        _validate_output_path(output_path, overwrite=args.overwrite, resume=args.resume)
        grouped_rows = _rows_by_model_sample(output_path) if args.resume else {}
        max_attempts = max(args.max_attempts, 1)
        pending, progress_counts = _pending_cases_and_latest_counts(
            cases=cases,
            grouped_rows=grouped_rows,
            model=model,
            retry_errors=args.retry_errors,
            max_attempts=max_attempts,
        )
        _print_model_progress(
            model=model,
            pending=pending,
            progress_counts=progress_counts,
            output_path=output_path,
            retry_pass=1,
        )
        if not pending:
            continue

        client = load_model(
            model,
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
            # --models value is the Azure deployment name; send it as-is.
            client.config.model_name = model
        try:
            retry_pass = 1
            while pending:
                start_time = datetime.now(timezone.utc)
                prompts = [case["prompt"] for case in pending]
                attempt_by_sample = {
                    case["sample_name"]: _attempt_count(
                        grouped_rows.get((model, case["sample_name"]), []),
                        case["prompt_sha1"],
                    ) + 1
                    for case in pending
                }
                print(
                    f"Model {model}: pass {retry_pass} ({len(pending)} sample(s), "
                    f"first={pending[0]['sample_name']})",
                    flush=True,
                )

                # Track which indices have been flushed so post-call cleanup +
                # exception handling don't double-write rows already on disk.
                written: set = set()

                def _row_for(
                    i: int,
                    raw: Any,
                    *,
                    error: Optional[str] = None,
                    status_override: Optional[str] = None,
                ) -> Dict[str, Any]:
                    case = pending[i]
                    raw_text = raw if isinstance(raw, str) else str(raw or "")
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    if status_override is not None:
                        return _result_row(
                            case=case, model=model,
                            attempt=attempt_by_sample[case["sample_name"]],
                            max_attempts=max_attempts,
                            raw_output=raw_text,
                            status=status_override,
                            parsed_action=None,
                            error=error,
                            duration_s=elapsed,
                            save_prompt=args.save_prompts,
                        )
                    if not raw_text.strip():
                        return _result_row(
                            case=case, model=model,
                            attempt=attempt_by_sample[case["sample_name"]],
                            max_attempts=max_attempts,
                            raw_output=raw_text,
                            status="empty_response",
                            parsed_action=None,
                            error="model returned empty response",
                            duration_s=elapsed,
                            save_prompt=args.save_prompts,
                        )
                    parsed_action = parse_naive_agent_result(
                        raw_text, expected_action=case["expected_final_action"],
                    )
                    return _result_row(
                        case=case, model=model,
                        attempt=attempt_by_sample[case["sample_name"]],
                        max_attempts=max_attempts,
                        raw_output=raw_text,
                        status="ok" if parsed_action is not None else "parse_error",
                        parsed_action=parsed_action,
                        error=None if parsed_action is not None
                              else "failed to parse expected tool_use JSON",
                        duration_s=elapsed,
                        save_prompt=args.save_prompts,
                    )

                def _on_complete(i: int, raw: Any) -> None:
                    """Flush this result the moment its OpenRouter call returns."""
                    row = _row_for(i, raw)
                    _append_jsonl(output_path, [row])
                    grouped_rows.setdefault((model, row["sample_name"]), []).append(row)
                    written.add(i)

                try:
                    # OpenRouter / Azure path: do not send temperature, top_p,
                    # or top_k. Those knobs are vLLM-only; Azure GPT-5
                    # reasoning deployments reject them outright, and they're
                    # not meaningful for reasoning models on OpenRouter.
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
                    # Write api_error rows for anything not already flushed by the callback.
                    rows = []
                    for i in range(len(pending)):
                        if i in written:
                            continue
                        row = _row_for(i, "", error=repr(exc), status_override="api_error")
                        rows.append(row)
                        grouped_rows.setdefault((model, row["sample_name"]), []).append(row)
                    if rows:
                        _append_jsonl(output_path, rows)
                    print(f"Batch failed for {model}: {exc!r}", flush=True)
                    raw_outputs = []

                # Defensive: write rows for any indices the callback didn't visit.
                tail_rows: List[Dict[str, Any]] = []
                for i in range(len(pending)):
                    if i in written:
                        continue
                    if i < len(raw_outputs):
                        row = _row_for(i, raw_outputs[i])
                    else:
                        row = _row_for(
                            i, "",
                            error=(f"client returned fewer responses than prompts "
                                   f"({len(raw_outputs)} < {len(pending)})"),
                            status_override="api_error",
                        )
                    tail_rows.append(row)
                    grouped_rows.setdefault((model, row["sample_name"]), []).append(row)
                if tail_rows:
                    _append_jsonl(output_path, tail_rows)

                write_summary(
                    summary_path=summary_path,
                    output_path=output_path,
                    dataset_path=None if args.use_legacy_batches else args.dataset_path,
                    manifest_path=args.manifest_path,
                    samples_path=args.samples_path,
                    models=[model],
                    num_cases=len(cases),
                )

                retry_pass += 1
                pending, progress_counts = _pending_cases_and_latest_counts(
                    cases=cases,
                    grouped_rows=grouped_rows,
                    model=model,
                    retry_errors=args.retry_errors,
                    max_attempts=max_attempts,
                )
                _print_model_progress(
                    model=model,
                    pending=pending,
                    progress_counts=progress_counts,
                    output_path=output_path,
                    retry_pass=retry_pass,
                )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        write_summary(
            summary_path=summary_path,
            output_path=output_path,
            dataset_path=None if args.use_legacy_batches else args.dataset_path,
            manifest_path=args.manifest_path,
            samples_path=args.samples_path,
            models=[model],
            num_cases=len(cases),
        )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
