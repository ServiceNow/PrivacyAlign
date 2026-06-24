"""Pure utility functions: text normalization, JSON parsing, file I/O."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class StageGenerationError(Exception):
    """Raised when one pipeline stage cannot produce valid output."""


def build_ci_fields(vignette: Dict[str, Any], seed_candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Build contextual-integrity fields used by quality check and filter prompts."""
    return {
        "sensitive_info_items": vignette.get("sensitive_info_items", []),
        "relevant_info_items": vignette.get("relevant_info_items", []),
        "data_subject": vignette.get("data_subject_concrete", seed_candidate.get("data_subject", "")),
        "data_sender": vignette.get("data_sender_concrete", seed_candidate.get("data_sender", "")),
        "data_recipient": vignette.get("data_recipient_concrete", seed_candidate.get("data_recipient", "")),
        "final_action": seed_candidate.get("final_action", ""),
    }


@dataclass
class NameRow:
    name: str
    frequency: int


def coerce_pass_flag(value: Any) -> bool:
    """Parse LLM pass/fail values safely.

    Accepts booleans, common string forms, and numeric 0/1.
    Falls back to False for ambiguous values.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "pass", "passed"}:
            return True
        if normalized in {"false", "no", "n", "0", "fail", "failed", ""}:
            return False

    return False


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip())
    return value


def normalize_key(value: str) -> str:
    value = normalize_text(value)
    # Convert camelCase/PascalCase into snake_case before lowercasing.
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = value.lower()
    value = value.replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", value)


def normalize_string_list(value: Any) -> List[str]:
    """Return a deduplicated list of normalized strings.

    Accepts either a scalar string-like value or a list/tuple of values.
    Deduplication preserves the first occurrence order.
    """
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = [value]

    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        text = normalize_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def short_hash(value: str) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", text)
    text = re.sub(r"\n?\s*```$", "", text)
    return text.strip()


def extract_balanced_fragment(text: str, opener: str, closer: str) -> Optional[str]:
    start = text.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            # Only handle backslash escapes inside quoted strings.
            # Outside strings, a backslash is just a literal character
            # and should not cause the next char (e.g. { or }) to be skipped.
            if in_string and ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find(opener, start + 1)
    return None


def parse_json_loose(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        raise ValueError("Model returned empty response.")
    if hasattr(raw, "dict"):
        return raw.dict()

    text = strip_code_fence(str(raw))
    if not text:
        raise ValueError("Model returned empty text response.")
    for candidate in (text, extract_balanced_fragment(text, "{", "}"), extract_balanced_fragment(text, "[", "]")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from model output: {text[:400]}")


def coerce_json_object(raw: Any, preferred_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        keys = tuple(preferred_keys or [])
        if keys:
            for item in raw:
                if isinstance(item, dict) and any(key in item for key in keys):
                    return item
        for item in raw:
            if isinstance(item, dict):
                return item
    return {}

