from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from string import Template

from data_loaders import get_hf_load_dataset, load_prompt, register_dataset

logger = logging.getLogger(__name__)


PRIVALIGN_DATASET_DIRNAME = "privalign-dataset"
# PrivacyAlign is published on the Hugging Face Hub. When no local export is
# found, the requested split is downloaded and cached as JSONL.
PRIVALIGN_HF_DATASET_ID = "ServiceNow/PrivacyAlign"
PRIVALIGN_VALIDATION_BUCKET_COUNT = 20
PRIVALIGN_VALIDATION_BUCKETS = frozenset({0, 1})  # deterministic ~10% holdout of train.jsonl for dev eval
PRIVALIGN_STUDENT_TEMPLATE_NAME = "naive_agent_prompt"
REFERENCE_RESPONSE_FIELD_CANDIDATES = (
    "reference_response",
    "policy_reference_response",
)


def is_privalign_dataset_name(dataset_name: str) -> bool:
    """Return whether *dataset_name* points at the local Privalign export."""
    name = dataset_name.strip()
    if not name:
        return False
    if os.path.isdir(name):
        return os.path.basename(os.path.normpath(name)) == PRIVALIGN_DATASET_DIRNAME
    if name.endswith(".jsonl") and os.path.isfile(name):
        parent = os.path.basename(os.path.dirname(os.path.dirname(name)))
        return parent == PRIVALIGN_DATASET_DIRNAME
    return name == PRIVALIGN_DATASET_DIRNAME


def _stringify_privalign_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _format_privalign_memories(memories: object) -> str:
    if isinstance(memories, list):
        lines = [str(memory).strip() for memory in memories if str(memory).strip()]
        if lines:
            return "\n".join(f"- {line}" for line in lines)
    return _stringify_privalign_field(memories)


def _format_privalign_response(response: object) -> str:
    if isinstance(response, dict):
        raw = response.get("action_raw")
        if isinstance(raw, str) and raw.strip():
            return raw
        args_json = response.get("arguments_json")
        if isinstance(args_json, str) and args_json.strip():
            tool_name = response.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                try:
                    arguments = json.loads(args_json)
                except json.JSONDecodeError:
                    arguments = args_json
                return json.dumps(
                    {"type": "tool_use", "name": tool_name, "arguments": arguments},
                    ensure_ascii=True,
                    sort_keys=True,
                )
            return args_json
    return _stringify_privalign_field(response)


def _json_dumps_pretty(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)


def _parse_json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _infer_json_schema_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _resolve_privalign_final_action(row: dict) -> str:
    final_action = row.get("expected_final_action")
    if isinstance(final_action, str) and final_action.strip():
        return final_action.strip()
    for response_key in ("response_b", "response_a"):
        response = row.get(response_key)
        if isinstance(response, dict):
            tool_name = response.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                return tool_name.strip()
            action = _parse_json_object(response.get("action_raw"))
            if action is not None and isinstance(action.get("name"), str):
                return str(action["name"]).strip()
    return ""


def _resolve_privalign_action_arguments(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        return {}
    action = _parse_json_object(response.get("action_raw"))
    if action is not None and isinstance(action.get("arguments"), dict):
        return dict(action["arguments"])
    args = _parse_json_object(response.get("arguments_json"))
    return {} if args is None else dict(args)


def _build_privalign_final_action_schema(row: dict) -> dict[str, object]:
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for response_key in ("response_a", "response_b"):
        for name, value in _resolve_privalign_action_arguments(row.get(response_key)).items():
            if not isinstance(name, str) or not name:
                continue
            properties.setdefault(name, {"type": _infer_json_schema_type(value)})
            if name not in required:
                required.append(name)
    schema: dict[str, object] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _iter_privalign_tool_names_from_trajectory(trajectory: str) -> list[str]:
    tool_names: list[str] = []
    seen: set[str] = set()
    for line in trajectory.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        event = _parse_json_object(stripped)
        if event is None or event.get("type") != "tool_use":
            continue
        name = event.get("name")
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            tool_names.append(name)
    return tool_names


def _build_privalign_toolkit_descriptions(row: dict, *, trajectory: str, final_action: str) -> str:
    toolkits = row.get("toolkits")
    if not isinstance(toolkits, list) or not toolkits:
        return "(no toolkits)"
    observed_actions = _iter_privalign_tool_names_from_trajectory(trajectory)
    if final_action and final_action not in observed_actions:
        observed_actions.append(final_action)
    lines: list[str] = []
    for toolkit in toolkits:
        toolkit_name = str(toolkit).strip()
        if not toolkit_name:
            continue
        actions = [action for action in observed_actions if action.startswith(toolkit_name)]
        if actions:
            lines.append(f"- **{toolkit_name}**: {', '.join(actions)}")
        else:
            lines.append(f"- **{toolkit_name}**")
    return "\n".join(lines) if lines else "(no toolkits)"


def _render_privalign_student_prompt(row: dict) -> str:
    user_instruction = str(row.get("user_instruction") or "").strip()
    if not user_instruction:
        raise ValueError("Privalign row must include a non-empty user_instruction.")
    trajectory = str(row.get("trajectory") or "")
    final_action = _resolve_privalign_final_action(row)
    if not final_action:
        raise ValueError("Privalign row must include an expected_final_action or response tool_name.")
    template = Template(load_prompt(PRIVALIGN_STUDENT_TEMPLATE_NAME))
    output_format_instructions = (
        "Output a single JSON object:\n"
        "{\"type\": \"tool_use\", \"name\": \"" + final_action + "\", \"arguments\": { ... }}"
    )
    rendered_vars = {
        "user_name": str(row.get("user_name") or ""),
        "toolkit_descriptions": _build_privalign_toolkit_descriptions(
            row,
            trajectory=trajectory,
            final_action=final_action,
        ),
        "user_instruction": user_instruction,
        "memories": _json_dumps_pretty(row.get("memories") or []),
        "executable_trajectory": trajectory,
        "task_instructions": f"Generate exactly one `{final_action}` tool call based on the trajectory evidence.",
        "output_format_instructions": output_format_instructions,
        "final_action": final_action,
        "final_action_input_schema_json": _json_dumps_pretty(
            _build_privalign_final_action_schema(row)
        ),
    }
    return template.safe_substitute(**rendered_vars)


def _normalize_privalign_label_block(labels: object) -> dict[str, object]:
    if not isinstance(labels, dict):
        return {"leaks": False, "leaks_details": "", "omits": False, "omits_details": ""}
    return {
        "leaks": bool(labels.get("leaks", False)),
        "leaks_details": str(labels.get("leaks_details") or ""),
        "omits": bool(labels.get("omits", False)),
        "omits_details": str(labels.get("omits_details") or ""),
    }


def _normalize_privalign_annotations(annotations: object) -> list[dict[str, object]]:
    if not isinstance(annotations, list):
        return []
    normalized: list[dict[str, object]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        normalized.append(
            {
                "preference": str(annotation.get("preference") or ""),
                "explanation": str(annotation.get("explanation") or ""),
                "gold": bool(annotation.get("gold", False)),
                "response_a_labels": _normalize_privalign_label_block(
                    annotation.get("response_a_labels")
                ),
                "response_b_labels": _normalize_privalign_label_block(
                    annotation.get("response_b_labels")
                ),
            }
        )
    return normalized


def _build_privalign_demo(row: dict) -> dict[str, object]:
    return {
        "demo_type": "privalign_pairwise",
        "source_dataset": "privalign-dataset",
        "item_id": row.get("id"),
        "user_instruction": str(row.get("user_instruction") or ""),
        "memories": _format_privalign_memories(row.get("memories")),
        "executable_trajectory": str(row.get("trajectory") or ""),
        "expected_final_action": _resolve_privalign_final_action(row),
        "reference_response_a": _format_privalign_response(row.get("response_a")),
        "reference_response_b": _format_privalign_response(row.get("response_b")),
        "annotations": _normalize_privalign_annotations(row.get("annotations")),
        "majority_bucket": str(row.get("majority_bucket") or ""),
        "preference_counts": (
            row.get("preference_counts") if isinstance(row.get("preference_counts"), dict) else {}
        ),
    }


def _build_privalign_context(row: dict) -> list[dict[str, str]]:
    return [{"role": "user", "content": _render_privalign_student_prompt(row)}]


def _iter_privalign_preference_examples(row: dict):
    if not isinstance(row, dict):
        raise ValueError("Privalign row must be a dict.")
    yield {
        "context": _build_privalign_context(row),
        "judge_demo": _build_privalign_demo(row),
    }


def _format_privalign_annotation_block(annotations: list[dict[str, object]]) -> str:
    if not annotations:
        return "No human annotations were provided."
    parts: list[str] = []
    for index, annotation in enumerate(annotations, 1):
        a_labels = _normalize_privalign_label_block(annotation.get("response_a_labels"))
        b_labels = _normalize_privalign_label_block(annotation.get("response_b_labels"))
        lines: list[str] = [
            f"Annotator {index}:",
            f"- Preference: {_format_privalign_preference_label(annotation.get('preference'))}",
            f"- Explanation: {annotation.get('explanation') or ''}",
        ]
        a_positive = _collect_privalign_positive_sentences(a_labels)
        b_positive = _collect_privalign_positive_sentences(b_labels)
        if a_positive:
            lines.append("- Response A:")
            lines.extend(a_positive)
        if b_positive:
            lines.append("- Response B:")
            lines.extend(b_positive)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _collect_privalign_positive_sentences(labels: dict[str, object]) -> list[str]:
    """Return one sentence per annotator-flagged failure on this response.

    Negative annotations ("the annotator did NOT mark this as leaking") are
    intentionally skipped to keep the judge prompt concise: 67% of omit
    labels and 30% of leak labels are negatives in the Privalign training
    split, and showing them adds verbose boilerplate without much signal.
    Positive labels without an explanatory ``leaks_details`` / ``omits_details``
    are also skipped, since the bare yes/no offers little for the judge to
    calibrate against.
    """
    sentences: list[str] = []
    if labels.get("leaks"):
        details = str(labels.get("leaks_details") or "").strip()
        if details:
            sentences.append(f"  - Annotator flagged a leak: {details}")
    if labels.get("omits"):
        details = str(labels.get("omits_details") or "").strip()
        if details:
            sentences.append(f"  - Annotator flagged a missing detail: {details}")
    return sentences


def _format_privalign_preference_label(preference: object) -> str:
    if not isinstance(preference, str) or not preference.strip():
        return "not provided"
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


def render_privalign_template(
    template: str,
    *,
    judge_demo: dict[str, object],
    eval_response1: str = "",
    eval_response2: str = "",
) -> str:
    annotations = judge_demo.get("annotations")
    if not isinstance(annotations, list):
        annotations = []
    result = template.format(
        user_instruction=str(judge_demo.get("user_instruction") or ""),
        memories=str(judge_demo.get("memories") or ""),
        executable_trajectory=str(judge_demo.get("executable_trajectory") or ""),
        reference_response_a=str(judge_demo.get("reference_response_a") or ""),
        reference_response_b=str(judge_demo.get("reference_response_b") or ""),
        annotator_block=_format_privalign_annotation_block(annotations),
        eval_response1=eval_response1,
        eval_response2=eval_response2,
    )
    return re.sub(r"\n{3,}", "\n\n", result)


def render_privalign_pointwise_eval_template(
    template: str,
    *,
    judge_demo: dict[str, object],
    new_response: str = "",
) -> str:
    annotations = judge_demo.get("annotations")
    if not isinstance(annotations, list):
        annotations = []
    result = Template(template).safe_substitute(
        user_instruction=str(judge_demo.get("user_instruction") or ""),
        memories=str(judge_demo.get("memories") or ""),
        executable_trajectory=str(judge_demo.get("executable_trajectory") or ""),
        reference_response_a=str(judge_demo.get("reference_response_a") or ""),
        reference_response_b=str(judge_demo.get("reference_response_b") or ""),
        annotator_block=_format_privalign_annotation_block(annotations),
        new_response=new_response,
    )
    return re.sub(r"\n{3,}", "\n\n", result)


def _format_preference_example(
    example: dict,
) -> dict[str, object]:
    reference_responses: dict[str, str] = {}
    for field_name in REFERENCE_RESPONSE_FIELD_CANDIDATES:
        candidate = example.get(field_name)
        if isinstance(candidate, str) and candidate:
            reference_responses[field_name] = candidate

    judge_demo = build_preference_judge_demo(example)
    result = {
        "prompt": list(example["context"]),
        "judge_demo": judge_demo,
    }
    ranking_demo = example.get("ranking_demo")
    if isinstance(ranking_demo, dict):
        result["ranking_demo"] = dict(ranking_demo)
    result.update(reference_responses)

    return result


def build_preference_judge_demo(example: dict) -> dict[str, object]:
    """Return the normalized preference/eval demo for an example."""
    judge_demo = example.get("judge_demo")
    if isinstance(judge_demo, dict):
        return dict(judge_demo)
    ranking_demo = example.get("ranking_demo")
    if not isinstance(ranking_demo, dict):
        raise ValueError(
            "Example is missing a 'judge_demo' or 'ranking_demo'; HelpSteer-style "
            "examples are no longer supported."
        )
    return dict(ranking_demo)


def _preference_example_dedup_key(
    example: dict,
) -> str:
    formatted_example = _format_preference_example(example)
    return json.dumps(formatted_example, sort_keys=True, ensure_ascii=True)


def _deduplicate_preference_dataset(
    dataset,
):
    keep_indices: list[int] = []
    seen_keys: set[str] = set()

    for index, example in enumerate(dataset):
        dedup_key = _preference_example_dedup_key(example)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        keep_indices.append(index)

    if len(keep_indices) == len(dataset):
        logger.info("No duplicate preference examples detected before formatting.")
        return dataset

    if hasattr(dataset, "select"):
        deduplicated = dataset.select(keep_indices)
    else:
        deduplicated = [dataset[index] for index in keep_indices]
    logger.info(
        "Deduplicated preference dataset from %s to %s examples before formatting.",
        len(dataset),
        len(deduplicated),
    )
    return deduplicated


def stable_text_hash_key(text: str) -> str:
    """SHA-256 digest of *text* used as a stable cache/split key."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _privalign_row_belongs_to_split(row_id: object, split: str) -> bool:
    """Return whether a train.jsonl row belongs to the requested split.

    `train` and `validation` are deterministic hash-bucket carve-outs of
    train.jsonl keyed on the row `id`; `test` rows live in test.jsonl and
    bypass this filter.
    """
    digest = hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % PRIVALIGN_VALIDATION_BUCKET_COUNT
    is_validation = bucket in PRIVALIGN_VALIDATION_BUCKETS
    return is_validation if split == "validation" else not is_validation


def _resolve_privalign_jsonl_path(dataset_name: str, *, split: str) -> str:
    requested = dataset_name.strip()
    # The dev split is a deterministic hash holdout of train.jsonl, so both
    # 'train' and 'validation' read the same file and the loader filters rows.
    file_split = "train" if split == "validation" else split
    if file_split not in {"train", "test", "all"}:
        raise ValueError("Privalign only supports 'train', 'test', 'validation', and 'all' splits.")
    if requested.endswith(".jsonl") and os.path.isfile(requested):
        return requested

    candidate_dirs: list[str] = []
    if os.path.isdir(requested):
        candidate_dirs.append(requested)
    candidate_dirs.append(os.path.join(os.getcwd(), requested))

    for directory in candidate_dirs:
        path = os.path.join(directory, "data", f"{file_split}.jsonl")
        if os.path.isfile(path):
            return path

    # No local export found: fall back to the published dataset on the Hub.
    try:
        return _download_privalign_hf_jsonl(file_split)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear FileNotFoundError
        raise FileNotFoundError(
            f"Could not resolve Privalign {split!r} JSONL from dataset_name={dataset_name!r} "
            f"locally, and downloading {PRIVALIGN_HF_DATASET_ID} from the Hugging Face Hub "
            f"failed: {exc}"
        ) from exc


def _privalign_cache_jsonl_path(file_split: str) -> str:
    cache_root = os.environ.get("PRIVALIGN_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "privacyalign"
    )
    return os.path.join(cache_root, "data", f"{file_split}.jsonl")


def _download_privalign_hf_jsonl(file_split: str) -> str:
    """Download a PrivacyAlign split from the HF Hub and cache it as JSONL.

    ``all`` concatenates the published ``train`` and ``test`` splits.
    """
    cache_path = _privalign_cache_jsonl_path(file_split)
    if os.path.isfile(cache_path):
        return cache_path
    load_dataset = get_hf_load_dataset()
    hf_splits = ["train", "test"] if file_split == "all" else [file_split]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for hf_split in hf_splits:
            dataset = load_dataset(PRIVALIGN_HF_DATASET_ID, split=hf_split)
            for row in dataset:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(tmp_path, cache_path)
    logger.info("Cached PrivacyAlign %s split from the Hub at %s.", file_split, cache_path)
    return cache_path


def _load_raw_privalign_preference_dataset(
    dataset_name: str,
    *,
    split: str,
):
    path = _resolve_privalign_jsonl_path(dataset_name, split=split)
    logger.info("Loading Privalign dataset from %s for split=%s.", path, split)
    apply_bucket_filter = split in {"train", "validation"}
    examples: list[dict] = []
    skipped_rows = 0
    holdout_filtered = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if apply_bucket_filter and not _privalign_row_belongs_to_split(row.get("id"), split):
                    holdout_filtered += 1
                    continue
                examples.extend(_iter_privalign_preference_examples(row))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                skipped_rows += 1
                logger.warning(
                    "Skipping malformed Privalign row %s in %s: %s",
                    line_number,
                    path,
                    exc,
                )
    logger.info(
        "Prepared %s Privalign %s examples (skipped_rows=%s holdout_filtered=%s).",
        len(examples),
        split,
        skipped_rows,
        holdout_filtered,
    )
    if not examples:
        raise ValueError(f"Privalign produced no usable examples for split={split!r}.")
    return examples


def load_privalign_pairwise_genrm_dataset(
    dataset_name: str,
    *,
    split: str,
):
    """Build pairwise gen-RM training rows in memory from a Privalign dataset path.

    Used when ``policy_scorer == "privalign_pairwise_margin"`` and
    ``dataset_name`` points at the local Privalign export. This mirrors the
    on-disk ``genrm.build_privalign_genrm_training_data`` pre-pass without
    requiring callers to materialize ``train.jsonl`` and ``validation.jsonl``
    first.
    """
    from genrm.build_privalign_genrm_training_data import build_privalign_genrm_rows

    input_path = _resolve_privalign_jsonl_path(dataset_name, split="train")
    per_split, summary = build_privalign_genrm_rows(input_path)
    if split not in per_split:
        raise ValueError(
            f"Unknown pairwise gen-RM split {split!r}; expected one of {sorted(per_split)}."
        )
    rows = per_split[split]
    logger.info(
        "Prepared %s pairwise gen-RM rows for Privalign split=%s (source=%s; summary=%s).",
        len(rows), split, input_path, summary.get(split, {}),
    )
    if not rows:
        raise ValueError(f"Pairwise gen-RM builder produced no rows for split={split!r}.")
    from datasets import Dataset
    return Dataset.from_list(rows)


def append_user_prompt_suffix_to_row(row: dict, user_prompt_suffix: str | None) -> dict:
    """Return a row copy with ``user_prompt_suffix`` appended to the last user turn."""
    if not user_prompt_suffix:
        return row
    field = "prompt" if "prompt" in row else "context"
    msgs = list(row.get(field) or [])
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
        ):
            content = msg["content"].rstrip()
            if content.endswith(user_prompt_suffix):
                return row
            new_msg = dict(msg)
            new_msg["content"] = user_prompt_suffix if not content else content + "\n\n" + user_prompt_suffix
            return {**row, field: msgs[:i] + [new_msg] + msgs[i + 1:]}
    return row


def load_raw_preference_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    dataset_config: str | None = None,
):
    """Load raw preference rows before prompt formatting."""
    if is_privalign_dataset_name(dataset_name):
        return _load_raw_privalign_preference_dataset(dataset_name, split=split)

    load_dataset = get_hf_load_dataset()

    logger.info("Loading preference dataset %s split=%s from Hugging Face.", dataset_name, split)
    if dataset_config is None:
        ds = load_dataset(dataset_name, "preference", split=split)
    else:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    logger.info("Loaded raw preference dataset with %s examples.", len(ds))
    ds = ds.filter(lambda x: x["overall_preference"] != 0)
    logger.info("Filtered preference dataset to %s examples with non-tied preferences.", len(ds))
    return ds


def format_preference_dataset(
    dataset,
    *,
    seed: int = 42,
    deduplicate_examples: bool = True,
    shuffle: bool = True,
):
    """Format raw preference rows into training prompts."""
    if deduplicate_examples:
        dataset = _deduplicate_preference_dataset(dataset)
    else:
        logger.info("Preference dataset deduplication disabled; keeping all filtered examples.")

    logger.info("Formatting preference dataset into policy-optimization prompts.")
    if hasattr(dataset, "map") and hasattr(dataset, "column_names"):
        dataset = dataset.map(
            lambda example: _format_preference_example(example),
            remove_columns=dataset.column_names,
        )
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
    else:
        dataset = [
            _format_preference_example(example)
            for example in dataset
        ]
        if shuffle:
            shuffled_dataset = list(dataset)
            random.Random(seed).shuffle(shuffled_dataset)
            dataset = shuffled_dataset
    if shuffle:
        logger.info("Shuffled preference dataset with seed=%s and returning %s training examples.", seed, len(dataset))
    else:
        logger.info("Returning %s formatted preference examples without shuffling.", len(dataset))
    return dataset


@register_dataset("preference")
def load_preference_dataset(
    dataset_name: str,
    dataset_config: str | None = None,
    seed: int = 42,
    deduplicate_examples: bool = True,
    user_prompt_suffix: str | None = None,
):
    """Load a HuggingFace preference dataset and format for training.

    Expects a schema with `context` and a `judge_demo` or `ranking_demo` per
    example. The formatted rows keep a compact `judge_demo` payload so
    LLM-judge RL can reuse the human annotations in-context without carrying
    the full raw row.
    """
    ds = load_raw_preference_dataset(dataset_name, split="train", dataset_config=dataset_config)
    if user_prompt_suffix:
        if hasattr(ds, "map"):
            ds = ds.map(lambda example: append_user_prompt_suffix_to_row(example, user_prompt_suffix))
        else:
            ds = [append_user_prompt_suffix_to_row(example, user_prompt_suffix) for example in ds]
    return format_preference_dataset(
        ds,
        seed=seed,
        deduplicate_examples=deduplicate_examples,
    )


def load_preference_validation_dataset(
    dataset_name: str,
    *,
    split: str = "validation",
    dataset_config: str | None = None,
):
    """Load raw preference rows for evaluation without prompt reformatting."""
    return load_raw_preference_dataset(dataset_name, split=split, dataset_config=dataset_config)
