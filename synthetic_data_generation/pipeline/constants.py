"""Validation keys and regex helpers used across generation stages."""

from __future__ import annotations

import re

REQUIRED_SEED_OPTION_KEYS = [
    "sex_options",
    "ethnicity_options",
    "religion_options",
    "default_citizenship",
]

PROMPT_TEMPLATE_FILES = [
    "profile_prompt.txt",
    "scenario_prompt.txt",
    "vignette_quality_check_prompt.txt",
    "trajectory_memories_prompt.txt",
    "quality_filter_prompt.txt",
    "naive_agent_prompt.txt",
    "sensibility_check_prompt.txt",
    "leakage_judge_prompt.txt",
]

PROFILE_REQUIRED_KEYS = [
    "first_name",
    "last_name",
    "sex",
    "ethnicity",
    "religion",
    "citizenship",
    "email",
    "occupation",
]

SEED_REQUIRED_KEYS = [
    "data_subject",
    "data_recipient",
    "toolkits",
    "final_action",
]

SCENARIO_REQUIRED_KEYS = SEED_REQUIRED_KEYS + [
    "domains",
    "story",
    "user_instruction",
    "sensitive_info_items",
    "relevant_info_items",
]

TRAJECTORY_REQUIRED_KEYS = [
    "executable_trajectory",
]

MIN_MEMORIES = 5

_ANSWER_YN_RE = re.compile(r"(?im)^\s*\*{0,2}\s*(?:Final\s+)?Answer\s*:?\s*\*{0,2}\s*(Yes|No)\b")
_BARE_YN_RE = re.compile(r"(?im)^\s*\*{0,2}\s*(Yes|No)\s*\*{0,2}\s*\.?\s*$")
# Permissive fallback: catches Yes/No after any answer-like phrase
# (e.g. "the final answer is: Yes", "verdict: No", "answer is Yes").
_ANSWER_PERMISSIVE_RE = re.compile(r"(?i)(?:answer|verdict|judgment|conclusion|decision)\s*(?:is|:)\s*\*{0,2}\s*(Yes|No)\b")
# Last resort: the last standalone Yes/No word in the text.
_LAST_YN_RE = re.compile(r"(?i)\b(Yes|No)\b")

# Names that indicate the LLM fell back to placeholders instead of generating
# realistic character names. Matched case-insensitively against the name portion
# of data_subject, data_sender, and data_recipient.
PLACEHOLDER_NAMES = frozenset({
    "john doe", "jane doe", "john smith", "jane smith",
    "bob smith", "alice smith", "test user", "example user",
})
