"""Leakage filter: prompt building and result parsing for batched two-phase filter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from pipeline.constants import _ANSWER_PERMISSIVE_RE, _ANSWER_YN_RE, _BARE_YN_RE, _LAST_YN_RE
from pipeline.utils import extract_balanced_fragment

LOGGER = logging.getLogger("scenario_generator")


def _strip_reasoning_and_fences(text: str) -> str:
    """Remove reasoning tags and markdown fences from model output."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    return cleaned


def _parse_valid_tool_use(
    obj: Any,
    expected_action: Optional[str],
) -> Optional[str]:
    """Validate and serialize a tool_use object for the expected final action."""
    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "tool_use":
        return None

    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if expected_action is not None and name != expected_action:
        return None

    action_input = obj.get("arguments", obj.get("input"))
    if not isinstance(action_input, dict):
        return None

    return json.dumps(obj)


def _parse_yes_no_answer(raw_text: str, label: str) -> Optional[bool]:
    """Parse a Yes/No verdict from judge output.

    Tries progressively more permissive patterns:
    1. Strict "Answer: Yes/No" (with optional "Final" prefix)
    2. Bare "Yes"/"No" on its own line
    3. Permissive answer-like phrases ("the answer is: Yes", "verdict: No")
    4. Last standalone Yes/No word in the text
    """
    cleaned = _strip_reasoning_and_fences(raw_text)
    matches = list(_ANSWER_YN_RE.finditer(cleaned))
    if not matches:
        matches = list(_BARE_YN_RE.finditer(cleaned))
    if not matches:
        matches = list(_ANSWER_PERMISSIVE_RE.finditer(cleaned))
    if not matches:
        matches = list(_LAST_YN_RE.finditer(cleaned))
    if not matches:
        LOGGER.warning("Could not parse %s: %s", label, cleaned)
        return None
    return matches[-1].group(1).lower() == "yes"


def _format_final_action_input_schema(ctx: Any, final_action: str) -> Dict[str, Any]:
    """Build a compact JSON schema for the final action input."""
    return ctx.registry.format_action_input_schema(final_action)


def build_naive_agent_prompt(
    ctx: Any,
    trajectory_payload: Dict[str, Any],
    seed_candidate: Dict[str, Any],
    vignette: Optional[Dict[str, Any]] = None,
    *,
    template_name: str = "naive_agent_prompt.txt",
) -> str:
    """Build the naive agent prompt (leakage phase 1).

    Includes prior-interaction memories from the vignette so the naive agent has
    the same context a real agent would have. Pass template_name to render a
    different prompt (e.g. "privacy_enhanced_agent_prompt.txt") that uses the
    same placeholder schema.
    """
    trajectory = trajectory_payload["trajectory"]
    final_action = trajectory["final_action"]
    toolkit_descriptions = ctx.registry.build_toolkit_descriptions(trajectory["toolkits"])

    final_action_input_schema_json = _format_final_action_input_schema(ctx, final_action)
    task_instructions = f'Generate exactly one `{final_action}` tool call based on the trajectory evidence.'
    output_format_instructions = (
        "Output a single JSON object:\n"
        "{\"type\": \"tool_use\", \"name\": \"" + final_action + "\", \"arguments\": { ... }}"
    )

    # Extract memories from vignette so the naive agent has the same context
    # a real memory-augmented agent would have.
    memories: list = []
    if vignette is not None:
        memories = vignette.get("memories", [])

    return ctx.render_prompt(
        template_name,
        user_name=trajectory["user_name"],
        user_email=trajectory["user_email"],
        toolkit_descriptions=toolkit_descriptions,
        user_instruction=trajectory["user_instruction"],
        memories=memories,
        executable_trajectory=trajectory["executable_trajectory"],
        final_action=final_action,
        final_action_input_schema_json=final_action_input_schema_json,
        task_instructions=task_instructions,
        output_format_instructions=output_format_instructions,
    )


def parse_naive_agent_result(
    raw_text: str,
    expected_action: Optional[str] = None,
) -> Optional[str]:
    """Extract tool_use JSON from naive agent output. Returns None on failure."""
    text = _strip_reasoning_and_fences(raw_text)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            parsed = _parse_valid_tool_use(obj, expected_action=expected_action)
            if parsed is not None:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    fragment = extract_balanced_fragment(text, "{", "}")
    if fragment:
        try:
            obj = json.loads(fragment)
            parsed = _parse_valid_tool_use(obj, expected_action=expected_action)
            if parsed is not None:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def build_sensibility_check_prompt(
    ctx: Any,
    trajectory_payload: Dict[str, Any],
    generated_action: str,
    vignette: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the sensibility check prompt (leakage phase 1.5)."""
    trajectory = trajectory_payload["trajectory"]
    final_action = trajectory["final_action"]
    final_action_input_schema_json = _format_final_action_input_schema(ctx, final_action)
    memories: list = []
    if vignette is not None:
        memories = vignette.get("memories", [])
    return ctx.render_prompt(
        "sensibility_check_prompt.txt",
        user_name=trajectory["user_name"],
        user_instruction=trajectory["user_instruction"],
        final_action=final_action,
        final_action_input_schema_json=final_action_input_schema_json,
        memories=memories,
        executable_trajectory=trajectory["executable_trajectory"],
        generated_action=generated_action,
    )


def parse_sensibility_check_result(raw_text: str) -> Optional[bool]:
    """Parse Yes/No from sensibility check output.

    Returns True if sensible, False if not sensible, None if unparseable.
    """
    return _parse_yes_no_answer(raw_text, label="sensibility check")


def build_leakage_judge_prompt(
    ctx: Any,
    trajectory_payload: Dict[str, Any],
    seed_candidate: Dict[str, Any],
    vignette: Dict[str, Any],
    generated_action: str,
) -> str:
    """Build the leakage judge prompt (leakage phase 2)."""
    trajectory = trajectory_payload["trajectory"]
    return ctx.render_prompt(
        "leakage_judge_prompt.txt",
        story=vignette.get("story", ""),
        sensitive_info_items=vignette.get("sensitive_info_items", []),
        relevant_info_items=vignette.get("relevant_info_items", []),
        data_subject=vignette.get("data_subject_concrete", seed_candidate.get("data_subject", "")),
        data_sender=vignette.get("data_sender_concrete", seed_candidate.get("data_sender", "")),
        data_recipient=vignette.get("data_recipient_concrete", seed_candidate.get("data_recipient", "")),
        user_name=trajectory["user_name"],
        user_instruction=trajectory["user_instruction"],
        memories=vignette.get("memories", []),
        executable_trajectory=trajectory["executable_trajectory"],
        generated_action=generated_action,
    )


def parse_leakage_judge_result(raw_text: str) -> Optional[bool]:
    """Parse Yes/No from leakage judge output.

    Returns True if leaked, False if not leaked, None if unparseable.
    Returning None (instead of a default) lets the caller decide whether
    to keep or discard the candidate on parse failure.
    """
    return _parse_yes_no_answer(raw_text, label="leakage judgment")
