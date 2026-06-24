"""Profile generation: prompt building and result parsing."""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from pipeline.constants import PROFILE_REQUIRED_KEYS
from pipeline.utils import StageGenerationError, coerce_json_object, normalize_key, normalize_text


def _profile_seed(ctx: Any, first_name: str) -> Dict[str, Any]:
    outbound_action_hint = ctx.registry.sample_final_action()
    return {
        "first_name": first_name,
        "outbound_action_hint": outbound_action_hint,
    }


def _match_option(value: str, options: list) -> str:
    """Match a value against a list of options, case-insensitively.

    Returns the canonical option string (preserving the original casing from
    the seed options file).  Raises StageGenerationError if no match is found.
    """
    for option in options:
        if value.lower() == option.lower():
            return option
    raise StageGenerationError(
        f"Invalid value: {value!r} (not in allowed options)"
    )


def _normalize_profile(ctx: Any, raw_profile: Dict[str, Any], seed: Dict[str, Any]) -> Dict[str, Any]:
    sex_options = ctx.seed_options["sex_options"]
    ethnicity_options = ctx.seed_options["ethnicity_options"]
    religion_options = ctx.seed_options["religion_options"]
    citizenship_options = ctx.seed_options["default_citizenship"]

    normalized: Dict[str, Any] = {}
    for key, value in raw_profile.items():
        normalized[normalize_key(key)] = normalize_text(value)

    normalized["first_name"] = seed["first_name"]

    for key in PROFILE_REQUIRED_KEYS:
        if not normalized.get(key):
            raise StageGenerationError(f"Missing required profile key: {key}")

    # Case-insensitive matching against seed options. The canonical casing
    # from the seed options file is preserved so downstream code (diversity
    # tracking, output formatting) sees consistent values.
    try:
        normalized["sex"] = _match_option(normalized["sex"], sex_options)
    except StageGenerationError:
        raise StageGenerationError(f"Invalid sex value: {normalized['sex']}")
    try:
        normalized["ethnicity"] = _match_option(normalized["ethnicity"], ethnicity_options)
    except StageGenerationError:
        raise StageGenerationError(f"Invalid ethnicity value: {normalized['ethnicity']}")
    try:
        normalized["religion"] = _match_option(normalized["religion"], religion_options)
    except StageGenerationError:
        raise StageGenerationError(f"Invalid religion value: {normalized['religion']}")
    try:
        normalized["citizenship"] = _match_option(normalized["citizenship"], citizenship_options)
    except StageGenerationError:
        raise StageGenerationError(f"Invalid citizenship value: {normalized['citizenship']}")

    if "@" not in normalized["email"]:
        raise StageGenerationError("Profile email missing '@'.")

    return normalized


def build_profile_prompt(ctx: Any, first_name: str) -> Tuple[str, Dict[str, Any]]:
    """Build profile generation prompt. Returns (prompt, seed_dict)."""
    seed = _profile_seed(ctx, first_name)
    sex_opts = list(ctx.seed_options["sex_options"])
    ethnicity_opts = list(ctx.seed_options["ethnicity_options"])
    religion_opts = list(ctx.seed_options["religion_options"])
    citizenship_opts = list(ctx.seed_options["default_citizenship"])
    random.shuffle(sex_opts)
    random.shuffle(ethnicity_opts)
    random.shuffle(religion_opts)
    random.shuffle(citizenship_opts)
    prompt = ctx.render_prompt(
        "profile_prompt.txt",
        seed_json=seed,
        first_name=seed["first_name"],
        outbound_action_hint=seed["outbound_action_hint"],
        sex_options_json=sex_opts,
        ethnicity_options_json=ethnicity_opts,
        religion_options_json=religion_opts,
        citizenship_options_json=citizenship_opts,
    )
    return prompt, seed


def parse_profile_result(ctx: Any, raw_json: Any, seed: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate profile from raw JSON response."""
    payload = coerce_json_object(raw_json, preferred_keys=["profile"])
    raw_profile = payload.get("profile", payload)
    if not isinstance(raw_profile, dict):
        raise StageGenerationError("Profile stage returned non-object profile.")
    return _normalize_profile(ctx, raw_profile, seed)
