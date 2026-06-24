"""Quality filter: prompt building and result parsing for batched LLM judge."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pipeline.utils import build_ci_fields, coerce_json_object, coerce_pass_flag, normalize_text

LOGGER = logging.getLogger("scenario_generator")


def _parse_step_field(raw_value: Any) -> Any:
    """Decode trajectory step fields when they are JSON-encoded strings.

    We pass parsed structures to the quality judge so it sees complete evidence
    rather than aggressively truncated snippets.
    """
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError, TypeError):
            return stripped
    return raw_value


def build_quality_filter_prompt(
    ctx: Any,
    trajectory_payload: Dict[str, Any],
    vignette: Dict[str, Any],
    seed_candidate: Dict[str, Any],
) -> Optional[str]:
    """Build the quality filter LLM judge prompt.

    Returns None if the quality filter is disabled or the model is unavailable.
    """
    if getattr(ctx.args, "disable_quality_filter", False):
        return None
    if ctx.model is None:
        return None
    if not isinstance(seed_candidate, dict):
        seed_candidate = {}

    steps = trajectory_payload.get("steps", [])
    trajectory = trajectory_payload.get("trajectory", {})
    user_instruction = normalize_text(trajectory.get("user_instruction", "")) if isinstance(trajectory, dict) else ""

    observations = []
    for idx, step in enumerate(steps):
        observations.append({
            "step": idx + 1,
            "action": step.get("action", ""),
            "action_input": _parse_step_field(step.get("action_input", "")),
            "observation": _parse_step_field(step.get("observation", "")),
        })

    ci_fields = build_ci_fields(vignette, seed_candidate)

    return ctx.render_prompt(
        "quality_filter_prompt.txt",
        story=vignette.get("story", ""),
        user_instruction=user_instruction,
        observations_json=observations,
        memories=vignette.get("memories", []),
        sensitive_info_items=ci_fields["sensitive_info_items"],
        relevant_info_items=ci_fields["relevant_info_items"],
        contextual_integrity_json=ci_fields,
    )


def parse_quality_filter_result(raw_json: Any) -> Tuple[bool, List[str]]:
    """Parse quality filter result. Returns (passed, issues)."""
    try:
        result = coerce_json_object(raw_json, preferred_keys=["pass", "issues"])
        if not isinstance(result, dict):
            LOGGER.warning("LLM quality check returned non-object response, rejecting")
            return False, ["llm_check_failed: non-object response"]

        passed = coerce_pass_flag(result.get("pass", False))
        llm_issues = result.get("issues", [])
        if not isinstance(llm_issues, list):
            llm_issues = [str(llm_issues)]
        llm_issues = [str(issue) for issue in llm_issues]
        return passed, llm_issues
    except Exception as exc:
        LOGGER.warning("LLM quality check parse failed, rejecting: %s", exc)
        return False, [f"llm_check_failed: {exc}"]
