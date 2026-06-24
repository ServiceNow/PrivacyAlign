#!/usr/bin/env python3
"""
Post-processing script for generated benchmark data.

Stages:
  1. Link-passthrough filtering — removes samples where leakage is caused by
     forwarding a pre-existing URL rather than composing content.
  2. Name deduplication — replaces overrepresented character names with fresh
     ethnicity-appropriate alternatives so scenarios feel unique.
  3. Diversity sampling — caps over-represented categories of configurable
     fields (final_action, domains) while honoring requested minimum-share
     constraints such as subject scope and model family.
  4. Reporting — prints frequency tables for key fields before and after
     processing.

Usage:
    python postprocess.py --input-path outputs/main_data_generated_gpt_oss_120b.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import re
import time
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pipeline.domains import domain_signature as build_domain_signature
from pipeline.domains import normalize_domain_list
from pipeline.link_passthrough_filter import check_sample as check_link_passthrough
from pipeline.utils import append_jsonl, write_json_atomic

try:  # pragma: no cover - exercised via opt-in exact-solver tests.
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    SCIPY_MILP_AVAILABLE = True
except ImportError:  # pragma: no cover - exact solver is optional.
    np = None
    Bounds = None
    LinearConstraint = None
    milp = None
    coo_array = None
    SCIPY_MILP_AVAILABLE = False

LOGGER = logging.getLogger("postprocess")

SCRIPT_DIR = Path(__file__).resolve().parent

PRIMARY_MODEL_FAMILIES = ["qwen", "gpt", "nvidia"]
DEFAULT_CAP_BALANCE_FIELDS = [
    "generation_metadata.domains",
    "generation_metadata.domain_signature",
    "trajectory.final_action",
    "generation_metadata.toolkit_signature",
]

DOMAIN_SIGNATURE_FIELD = "generation_metadata.domain_signature"
TOOLKIT_FIELD = "trajectory.toolkits"

# ─── Name banks (loaded from resources/replacement_names.json) ─────────────

def _load_name_data() -> Dict[str, Any]:
    """Load replacement name banks from the external JSON resource file."""
    names_path = SCRIPT_DIR / "resources" / "replacement_names.json"
    with names_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

_NAME_DATA = _load_name_data()

FIRST_NAMES: Dict[str, Dict[str, List[str]]] = _NAME_DATA["first_names"]
LAST_NAMES: Dict[str, List[str]] = _NAME_DATA["last_names"]

# ─── Lookup tables (built from name banks + extra entries) ────────────────

LAST_NAME_TO_ETHNICITY: Dict[str, str] = {}
for _eth, _names in LAST_NAMES.items():
    for _n in _names:
        LAST_NAME_TO_ETHNICITY[_n.lower()] = _eth
LAST_NAME_TO_ETHNICITY.update(_NAME_DATA.get("extra_last_name_ethnicities", {}))

FIRST_NAME_TO_GENDER: Dict[str, str] = {}
for _eth_data in FIRST_NAMES.values():
    for _n in _eth_data.get("male", []):
        FIRST_NAME_TO_GENDER[_n.lower()] = "male"
    for _n in _eth_data.get("female", []):
        FIRST_NAME_TO_GENDER[_n.lower()] = "female"
FIRST_NAME_TO_GENDER.update(_NAME_DATA.get("extra_first_name_genders", {}))

# Placeholder names that should always be replaced, regardless of frequency.
_PLACEHOLDER_NAMES: Set[str] = set(_NAME_DATA.get("placeholder_names", []))

_PLACEHOLDER_NAME_PATTERNS: Dict[str, re.Pattern] = {
    name: re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
    for name in _PLACEHOLDER_NAMES
}

# Surnames that should trigger replacement even when the full name itself is
# not globally overrepresented.
_FORCED_LAST_NAMES: Set[str] = {
    name.lower() for name in _NAME_DATA.get("forced_last_names", ["Patel", "Chen"])
}

# ─── Name extraction and classification ─────────────────────────────────────

_TITLE_PREFIXES = {"dr", "mr", "mrs", "ms", "prof", "rev", "hon"}


def _is_plausible_name(name: str) -> bool:
    """Return True if *name* looks like a person's name.

    Rejects strings where any word starts with a lowercase letter, digit,
    or punctuation — these are almost certainly role descriptions or other
    non-name text that leaked through compound-field splitting.
    """
    words = name.split()
    return bool(words) and all(w[0].isupper() and w[0].isalpha() for w in words)


def _extract_single_name(segment: str) -> str:
    """Extract a person's name from a single 'Name, Role at Org' segment."""
    segment = segment.strip()
    if not segment:
        return ""
    name_part = segment.split(",")[0].strip()
    words = name_part.split()
    if words and words[0].lower().rstrip(".") in _TITLE_PREFIXES:
        words = words[1:]
    return " ".join(words).strip()


def extract_person_names(field_value: str) -> List[str]:
    """Extract person names from a possibly compound field value.

    Handles both single entries ("Sara Patel, analyst at Acme") and compound
    entries joined with "and" ("Sara Patel, analyst ... and John Doe, accountant ...").
    Returns a list of plausible person names (>= 2 words, all capitalised).
    """
    if not field_value:
        return []
    names: List[str] = []
    for segment in re.split(r"\band\b", field_value):
        name = _extract_single_name(segment)
        if name and len(name.split()) >= 2 and _is_plausible_name(name):
            names.append(name)
    return names


def split_name(full_name: str) -> Tuple[str, str]:
    """Split a full name into (first_name, last_name)."""
    parts = full_name.strip().split()
    if len(parts) == 0:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], " ".join(parts[1:]))


def compute_name_abs_threshold(sample_count: int, threshold_pct: float) -> int:
    """Convert a percentage threshold into the minimum matching sample count."""
    if sample_count <= 0:
        return 1
    return max(math.ceil(sample_count * threshold_pct / 100.0), 1)


def find_placeholder_names_in_text(text: str) -> Set[str]:
    """Return canonical placeholder names found anywhere in *text*."""
    found: Set[str] = set()
    for placeholder, pattern in _PLACEHOLDER_NAME_PATTERNS.items():
        if pattern.search(text):
            found.add(placeholder)
    return found


def find_forced_last_name_names(names: Set[str]) -> Set[str]:
    """Return names whose surname is on the forced-replacement list."""
    forced: Set[str] = set()
    for name in names:
        _, last_name = split_name(name)
        if last_name and last_name.lower() in _FORCED_LAST_NAMES:
            forced.add(name)
    return forced


def plan_forced_last_name_replacements(
    samples: List[Dict[str, Any]],
    surname_abs_threshold: int,
    rng: random.Random | None = None,
) -> Dict[int, Set[str]]:
    """Select only the excess forced-surname samples for replacement.

    The threshold is interpreted as a maximum number of samples that may still
    contain the surname after replacement. Samples are selected uniformly at
    random per surname, and all matching structured names in a selected sample
    are replaced so the sample no longer contributes to that surname's count.
    """
    if rng is None:
        rng = random.Random()

    per_last_name: Dict[str, List[Tuple[int, Set[str]]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        names = find_forced_last_name_names(extract_names_from_sample(sample))
        if not names:
            continue

        grouped: Dict[str, Set[str]] = defaultdict(set)
        for name in names:
            _, last_name = split_name(name)
            if last_name:
                grouped[last_name.lower()].add(name)
        for last_name, matched_names in grouped.items():
            per_last_name[last_name].append((idx, matched_names))

    planned: Dict[int, Set[str]] = defaultdict(set)
    for last_name, occurrences in per_last_name.items():
        if len(occurrences) <= surname_abs_threshold:
            continue
        replace_count = len(occurrences) - surname_abs_threshold
        selected = rng.sample(occurrences, replace_count)
        for idx, names in selected:
            planned[idx].update(names)

    return dict(planned)


def guess_ethnicity(full_name: str) -> str:
    """Infer ethnicity from a person's last name."""
    _, last = split_name(full_name)
    if not last:
        return "anglo"
    return LAST_NAME_TO_ETHNICITY.get(last.lower(), "anglo")


def guess_gender(full_name: str) -> str:
    """Infer gender from a person's first name."""
    first, _ = split_name(full_name)
    if not first:
        return "male"
    return FIRST_NAME_TO_GENDER.get(first.lower(), "male")


def extract_names_from_sample(sample: Dict[str, Any]) -> Set[str]:
    """Extract all character names from a sample's structured fields."""
    names: Set[str] = set()
    seed = sample.get("seed", {})
    traj = sample.get("trajectory", {})

    for field in ("data_subject", "data_sender", "data_recipient"):
        for name in extract_person_names(seed.get(field, "")):
            names.add(name)

    user_name = traj.get("user_name", "")
    if user_name and len(user_name.split()) >= 2:
        names.add(user_name)

    return names


# ─── Name replacement ───────────────────────────────────────────────────────

class NameReplacer:
    """Generates unique ethnicity-appropriate replacement names."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._used: Dict[str, Set[str]] = defaultdict(set)

    def generate(self, original_name: str) -> Tuple[str, str, str]:
        """Return (full_name, first_name, last_name) replacement.

        Avoids reusing the original's first or last name so that
        standalone first-name patterns remain effective.
        """
        ethnicity = guess_ethnicity(original_name)
        gender = guess_gender(original_name)

        orig_first, orig_last = split_name(original_name)
        orig_first_lower = orig_first.lower()
        orig_last_lower = orig_last.lower()

        eth_firsts = FIRST_NAMES.get(ethnicity, FIRST_NAMES["anglo"])
        first_pool = eth_firsts.get(gender, eth_firsts.get("male", []))
        last_pool = LAST_NAMES.get(ethnicity, LAST_NAMES["anglo"])

        for _ in range(200):
            first = self.rng.choice(first_pool)
            last = self.rng.choice(last_pool)
            if first.lower() == orig_first_lower or last.lower() == orig_last_lower:
                continue
            full = f"{first} {last}"
            if full not in self._used[ethnicity]:
                self._used[ethnicity].add(full)
                return full, first, last

        # Fallback (pool exhausted): allow repeats but still avoid same names
        for _ in range(50):
            first = self.rng.choice(first_pool)
            last = self.rng.choice(last_pool)
            if first.lower() != orig_first_lower and last.lower() != orig_last_lower:
                return f"{first} {last}", first, last

        # Last resort: accept any combination
        first = self.rng.choice(first_pool)
        last = self.rng.choice(last_pool)
        return f"{first} {last}", first, last


def _make_pattern(name: str) -> re.Pattern:
    """Create a word-boundary regex for a name, handling apostrophes."""
    escaped = re.escape(name)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


def _make_handle_pattern(name: str) -> re.Pattern:
    """Create a pattern for user-handle / email-local-part variants.

    Matches underscore-separated (liam_oconnor), dot-separated (liam.oconnor),
    and concatenated (liamoconnor) forms.  Uses non-word-char boundaries so
    it works inside @handles and email addresses.
    """
    escaped = re.escape(name)
    return re.compile(r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])", re.IGNORECASE)


def _build_handle_variants(
    orig_first: str,
    orig_last: str,
    new_first: str,
    new_last: str,
) -> List[Tuple[re.Pattern, str]]:
    """Build replacement pairs for email/handle name variants.

    Covers:
    - first_last / first-last
    - first.last
    - firstlast
    - f_last / f-last / f.last / flast
    - apostrophe-stripped variants
    """
    pairs: List[Tuple[re.Pattern, str]] = []
    # Normalize: lowercase, strip apostrophes
    of = orig_first.lower()
    ol = orig_last.lower().replace("'", "")
    nf = new_first.lower()
    nl = new_last.lower().replace("'", "")
    oi = of[:1]
    ni = nf[:1]

    # underscore-separated: liam_oconnor -> sean_murphy
    pairs.append((_make_handle_pattern(f"{of}_{ol}"), f"{nf}_{nl}"))
    # hyphen-separated: liam-oconnor -> sean-murphy
    pairs.append((_make_handle_pattern(f"{of}-{ol}"), f"{nf}-{nl}"))
    # dot-separated is already handled by individual first/last word-boundary
    # patterns, but add the compound form for robustness
    pairs.append((_make_handle_pattern(f"{of}.{ol}"), f"{nf}.{nl}"))
    # concatenated: liamoconnor -> seanmurphy (only if >= 8 chars to avoid
    # false matches on short combos)
    concat_orig = f"{of}{ol}"
    if len(concat_orig) >= 8:
        concat_new = f"{nf}{nl}"
        pairs.append((_make_handle_pattern(concat_orig), concat_new))

    # first-initial variants are common in email local parts.
    if oi and ni:
        pairs.append((_make_handle_pattern(f"{oi}_{ol}"), f"{ni}_{nl}"))
        pairs.append((_make_handle_pattern(f"{oi}-{ol}"), f"{ni}-{nl}"))
        pairs.append((_make_handle_pattern(f"{oi}.{ol}"), f"{ni}.{nl}"))
        if len(ol) >= 4:
            pairs.append((_make_handle_pattern(f"{oi}{ol}"), f"{ni}{nl}"))

    return pairs


def _case_preserving_sub(pattern: re.Pattern, replacement: str, text: str) -> str:
    """Replace pattern matches while preserving original casing pattern."""
    def _replacer(match: re.Match) -> str:
        original = match.group()
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return " ".join(w.capitalize() for w in replacement.split())
        else:
            return replacement.lower()
    return pattern.sub(_replacer, text)


def replace_in_obj(
    obj: Any,
    replacements: List[Tuple[re.Pattern, str]],
) -> Any:
    """Recursively replace name patterns in a nested dict/list/str structure."""
    if isinstance(obj, str):
        for pattern, repl in replacements:
            obj = _case_preserving_sub(pattern, repl, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: replace_in_obj(v, replacements) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_in_obj(item, replacements) for item in obj]
    else:
        return obj


def _replace_in_serialized_json_string(
    text: str,
    replacements: List[Tuple[re.Pattern, str]],
) -> str:
    """Apply replacements inside JSON/NDJSON payload strings when possible.

    Some fields (for example ``generated_final_action`` and
    ``executable_trajectory``) store structured JSON as escaped strings.
    Word-boundary regexes can miss names that appear right after escape
    sequences like ``\\n`` in those payloads, so parse-and-rewrite when the
    string is actually JSON or newline-delimited JSON.
    """
    stripped = text.strip()
    if not stripped:
        return text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    else:
        return json.dumps(replace_in_obj(parsed, replacements), ensure_ascii=False)

    lines = text.splitlines()
    if not lines:
        return text

    parsed_lines: List[Any] = []
    saw_payload = False
    for line in lines:
        if not line.strip():
            parsed_lines.append(None)
            continue
        try:
            parsed_line = json.loads(line)
        except json.JSONDecodeError:
            return text
        parsed_lines.append(parsed_line)
        saw_payload = True

    if not saw_payload:
        return text

    rendered_lines: List[str] = []
    for parsed_line in parsed_lines:
        if parsed_line is None:
            rendered_lines.append("")
            continue
        replaced_line = replace_in_obj(parsed_line, replacements)
        rendered_lines.append(json.dumps(replaced_line, ensure_ascii=False))
    return "\n".join(rendered_lines)


def replace_names_in_sample(
    sample: Dict[str, Any],
    names_to_replace: Set[str],
    replacer: NameReplacer,
) -> bool:
    """Apply full-name replacement rules to one sample in place."""
    if not names_to_replace:
        return False

    sample_names = extract_names_from_sample(sample)
    sample_first_counts: Counter = Counter()
    for sample_name in sample_names:
        first_name, _ = split_name(sample_name)
        if first_name:
            sample_first_counts[first_name.lower()] += 1

    replacement_pairs: List[Tuple[re.Pattern, str]] = []
    exact_name_replacements: Dict[str, Tuple[str, str, str]] = {}
    for orig_name in sorted(names_to_replace, key=len, reverse=True):
        new_full, new_first, new_last = replacer.generate(orig_name)
        exact_name_replacements[orig_name] = (new_full, new_first, new_last)
        orig_first, orig_last = split_name(orig_name)

        replacement_pairs.append((_make_pattern(orig_name), new_full))
        if orig_first and orig_last:
            replacement_pairs.extend(
                _build_handle_variants(
                    orig_first, orig_last, new_first, new_last,
                )
            )
        if (
            orig_first
            and sample_first_counts.get(orig_first.lower(), 0) == 1
        ):
            replacement_pairs.append((_make_pattern(orig_first), new_first))

    new_sample = replace_in_obj(sample, replacement_pairs)
    traj = new_sample.get("trajectory", {})
    if isinstance(traj, dict):
        for field in ("generated_final_action", "executable_trajectory"):
            value = traj.get(field)
            if isinstance(value, str) and value:
                traj[field] = _replace_in_serialized_json_string(value, replacement_pairs)

    sender_name = _extract_single_name(sample.get("seed", {}).get("data_sender", ""))
    if sender_name in exact_name_replacements:
        _, sender_first, _ = exact_name_replacements[sender_name]
        seed = new_sample.get("seed", {})
        if seed.get("data_sender_name"):
            seed["data_sender_name"] = sender_first

    unresolved = [
        name for name in names_to_replace
        if name in json.dumps(new_sample, ensure_ascii=False)
    ]
    if unresolved:
        LOGGER.warning("Name replacement left unresolved names: %s", ", ".join(sorted(unresolved)))

    sample.clear()
    sample.update(new_sample)
    return True


# ─── Domain normalization (shared with generation via pipeline.domains) ─────


def _set_nested(obj: Dict, path: str, value: Any) -> None:
    """Set a nested field value by dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        if isinstance(obj, dict):
            obj = obj.setdefault(key, {})
        else:
            return
    if isinstance(obj, dict):
        obj[keys[-1]] = value


def normalize_domains(
    samples: List[Dict[str, Any]],
    field_paths: List[str] | None = None,
) -> int:
    """Normalize domain labels in-place. Returns count of relabelled samples.

    Checks multiple field paths so that domains stored in either the new
    ``*.domains`` list fields or legacy ``*.scenario_domain`` scalar fields
    are normalized consistently.
    """
    if field_paths is None:
        field_paths = [
            "generation_metadata.domains",
            "seed.domains",
            "generation_metadata.scenario_domain",
            "seed.scenario_domain",
        ]
    changed = 0
    for s in samples:
        sample_changed = False
        for field_path in field_paths:
            if field_path.endswith(".scenario_domain") and not _has_nested_explicit(s, field_path):
                continue
            raw = _get_nested_raw(s, field_path)
            if raw in (None, "", []):
                continue
            canonical = normalize_domain_list(raw)
            if _path_is_domain_field(field_path):
                if field_path.endswith(".domains") or field_path == "domains":
                    new_value: Any = canonical
                else:
                    new_value = canonical[0] if len(canonical) == 1 else canonical
            else:
                new_value = canonical
            if raw != new_value:
                _set_nested(s, field_path, new_value)
                sample_changed = True
        if sample_changed:
            changed += 1
    return changed


_HEALTHCARE_TOOLKIT_CALL_RE = re.compile(r"(epicfhir|teladoc)", re.IGNORECASE)

_HEALTHCARE_KEYWORD_PATTERNS: Dict[str, re.Pattern[str]] = {
    "diagnosis": re.compile(r"\bdiagnos(?:is|es|ed|ing)?\b", re.IGNORECASE),
    "hipaa": re.compile(r"\bhipaa\b", re.IGNORECASE),
    "phi": re.compile(r"\bphi\b", re.IGNORECASE),
    "fmla": re.compile(r"\bfmla\b", re.IGNORECASE),
    "medical_information": re.compile(r"\bmedical information\b", re.IGNORECASE),
    "health_information": re.compile(r"\bhealth information\b", re.IGNORECASE),
    "sensitive_health": re.compile(r"\bsensitive health\b", re.IGNORECASE),
    "medical_leave": re.compile(r"\bmedical leave\b", re.IGNORECASE),
    "medical_clearance": re.compile(r"\bmedical clearance\b", re.IGNORECASE),
    "medical_condition": re.compile(r"\bmedical condition\b", re.IGNORECASE),
    "health_services": re.compile(r"\bhealth services\b", re.IGNORECASE),
    "health_support_services": re.compile(
        r"\bhealth support services?\b",
        re.IGNORECASE,
    ),
    "medication": re.compile(r"\bmedication(?:s)?\b", re.IGNORECASE),
    "prescription": re.compile(r"\bprescription\b", re.IGNORECASE),
    "pharmacy": re.compile(r"\bpharmacy\b", re.IGNORECASE),
    "test_results": re.compile(r"\btest results?\b", re.IGNORECASE),
    "lab_results": re.compile(r"\blab results?\b", re.IGNORECASE),
    "symptoms": re.compile(r"\bsymptoms?\b", re.IGNORECASE),
    "medical_record": re.compile(r"\bmedical record\b", re.IGNORECASE),
    "patient_record": re.compile(r"\bpatient record\b", re.IGNORECASE),
    "medical_history": re.compile(r"\bmedical history\b", re.IGNORECASE),
    "treatment_records": re.compile(r"\btreatment records?\b", re.IGNORECASE),
    "mental_health": re.compile(r"\bmental health\b", re.IGNORECASE),
    "therapy": re.compile(r"\btherapy\b", re.IGNORECASE),
    "therapist": re.compile(r"\btherapist\b", re.IGNORECASE),
    "physiotherapy": re.compile(r"\bphysiotherap\w*\b", re.IGNORECASE),
    "mental_illness": re.compile(
        r"\b(?:depress(?:ion|ive)?|anxiety|ptsd|bipolar|suicid(?:e|al))\b",
        re.IGNORECASE,
    ),
    "doctor_note": re.compile(r"\bdoctor'?s note\b", re.IGNORECASE),
    "injury": re.compile(r"\binjur(?:y|ies|ed)\b", re.IGNORECASE),
    "workers_comp": re.compile(
        r"\bworkers?'?\s+(?:comp|compensation)\b",
        re.IGNORECASE,
    ),
    "medical_review": re.compile(r"\bmedical review\b", re.IGNORECASE),
    "medical_accommodation": re.compile(
        r"\bmedical accommodation(?:s)?\b",
        re.IGNORECASE,
    ),
    "asthma": re.compile(r"\basthma\b", re.IGNORECASE),
    "allergy": re.compile(r"\ballerg(?:y|ies|ic)\b", re.IGNORECASE),
    "miscarriage": re.compile(
        r"\b(?:miscarriage|pregnancy loss)\b",
        re.IGNORECASE,
    ),
    "fertility": re.compile(r"\bfertility\b", re.IGNORECASE),
    "ivf": re.compile(r"\bivf\b", re.IGNORECASE),
    "prenatal": re.compile(r"\bprenatal\b", re.IGNORECASE),
    "postpartum": re.compile(r"\bpostpartum\b", re.IGNORECASE),
    "vaccine": re.compile(r"\bvaccin\w*\b", re.IGNORECASE),
    "pregnancy": re.compile(r"\bpregnan(?:cy|t)\b", re.IGNORECASE),
    "cancer": re.compile(r"\bcancer\b", re.IGNORECASE),
    "alzheimers": re.compile(r"\balzheimer'?s\b", re.IGNORECASE),
    "dementia": re.compile(r"\bdementia\b", re.IGNORECASE),
    "fracture": re.compile(r"\bfracture\b", re.IGNORECASE),
    "spinal_injury": re.compile(r"\b(?:spinal|lumbar)\b", re.IGNORECASE),
    "rehab": re.compile(r"\brehab(?:ilitation)?\b", re.IGNORECASE),
    "blood_pressure": re.compile(r"\bblood pressure\b", re.IGNORECASE),
    "glucose": re.compile(r"\bglucose\b", re.IGNORECASE),
    "heart_rate": re.compile(r"\bheart rate\b", re.IGNORECASE),
    "health_insurance": re.compile(r"\bhealth insurance\b", re.IGNORECASE),
    "health_plan": re.compile(r"\bhealth plan\b", re.IGNORECASE),
    "oncology": re.compile(r"\boncology\b", re.IGNORECASE),
    "neurology": re.compile(r"\bneurology\b", re.IGNORECASE),
    "cardiology": re.compile(r"\bcardiology\b", re.IGNORECASE),
    "infectious_disease": re.compile(r"\binfectious disease\b", re.IGNORECASE),
    "reproductive_health": re.compile(r"\breproductive health\b", re.IGNORECASE),
    "disability_accommodation": re.compile(
        r"\bdisability accommodation(?:s)?\b",
        re.IGNORECASE,
    ),
    "clinic": re.compile(r"\bclinic\b", re.IGNORECASE),
    "hospital": re.compile(r"\bhospital\b", re.IGNORECASE),
}


def _healthcare_mismatch_text(sample: Dict[str, Any]) -> str:
    traj = sample.get("trajectory", {})
    parts: List[str] = []
    for value in (
        sample.get("vignette"),
        traj.get("user_instruction"),
        traj.get("generated_final_action"),
        traj.get("executable_trajectory"),
    ):
        if isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=False))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return "\n".join(parts)


def detect_healthcare_mismatch_signals(sample: Dict[str, Any]) -> Dict[str, List[str]]:
    domains = _sample_domains(sample)
    if "healthcare" in domains:
        return {"tool_calls": [], "keywords": []}

    executable_trajectory = str(
        sample.get("trajectory", {}).get("executable_trajectory", "") or ""
    )
    toolkit_aliases = {
        "epicfhir": "EpicFHIR",
        "teladoc": "Teladoc",
    }
    tool_calls = sorted(
        {
            toolkit_aliases[match.group(1).lower()]
            for match in _HEALTHCARE_TOOLKIT_CALL_RE.finditer(executable_trajectory)
        },
        key=str.lower,
    )

    text = _healthcare_mismatch_text(sample)
    keywords = [
        label
        for label, pattern in _HEALTHCARE_KEYWORD_PATTERNS.items()
        if pattern.search(text)
    ]
    return {"tool_calls": tool_calls, "keywords": keywords}


def filter_healthcare_domain_mismatches(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejection_rows: List[Dict[str, Any]] = []

    for sample in samples:
        signals = detect_healthcare_mismatch_signals(sample)
        if not signals["tool_calls"] and not signals["keywords"]:
            accepted.append(sample)
            continue

        rejected.append(sample)
        rejection_rows.append(
            {
                "name": sample.get("name", "?"),
                "stage": "healthcare_domain_mismatch",
                "domains": _sample_domains(sample),
                "tool_calls": signals["tool_calls"],
                "keywords": signals["keywords"],
            }
        )

    return accepted, rejected, rejection_rows


# ─── Computed diversity fields ───────────────────────────────────────────────


def compute_toolkit_signature(sample: Dict[str, Any]) -> str:
    """Derive an order-invariant toolkit signature from trajectory.toolkits."""
    toolkits = sample.get("trajectory", {}).get("toolkits", [])
    if not toolkits:
        return "unknown"
    return "|".join(sorted(toolkits))


def _extract_name(field_value: str) -> str:
    """Extract the person name (text before the first comma) from a seed field."""
    name = field_value.split(",")[0].strip()
    # Strip title prefixes (Dr., Mr., etc.)
    words = name.split()
    if words and words[0].lower().rstrip(".") in _TITLE_PREFIXES:
        words = words[1:]
    return " ".join(words).lower()


def compute_subject_scope(sample: Dict[str, Any]) -> str:
    """Derive subject scope (self / third_party / multi_subject) from seed fields."""
    seed = sample.get("seed", {})
    data_subject = seed.get("data_subject", "").strip()
    data_sender = seed.get("data_sender", "").strip()
    if not data_subject:
        return "unknown"
    sender_name = _extract_name(data_sender)
    if not sender_name:
        return "unknown"
    # Check for multi-subject first — " and " is a strong structural signal
    # that multiple people are named.  This matches DiversityTracker._subject_scope.
    if " and " in data_subject.lower():
        return "multi_subject"
    if sender_name in data_subject.lower():
        return "self"
    return "third_party"


def _normalize_family_label(family: str) -> str:
    candidate = str(family).strip().lower()
    if candidate in {"nvidia", "nemotron"}:
        return "nvidia"
    if candidate in {"gpt", "gpt-oss", "gpt_oss"}:
        return "gpt"
    if candidate == "qwen":
        return "qwen"
    if candidate == "step":
        return "step"
    if candidate in {"unknown", "other"}:
        return candidate
    return candidate


def _detect_model_family_from_probe(probe: str) -> str:
    normalized_probe = str(probe or "").strip().lower()
    if not normalized_probe:
        return ""

    normalized_label = _normalize_family_label(normalized_probe)
    if normalized_label != normalized_probe:
        return normalized_label

    if "nvidia" in normalized_probe or "nemotron" in normalized_probe:
        return "nvidia"
    if (
        "gpt-oss" in normalized_probe
        or "gpt_oss" in normalized_probe
        or re.search(r"(^|[^a-z0-9])gpt([^a-z0-9]|$)", normalized_probe)
    ):
        return "gpt"
    if "stepfun" in normalized_probe or re.search(r"(^|[^a-z0-9])step([^a-z0-9]|$)", normalized_probe):
        return "step"
    if "qwen" in normalized_probe:
        return "qwen"
    return normalized_label


def compute_model_family(sample: Dict[str, Any]) -> str:
    """Derive a coarse model family label for balancing and reporting."""
    meta = sample.get("generation_metadata", {})

    explicit_family = str(
        meta.get("model_family")
        or meta.get("combined_source_family")
        or ""
    ).strip().lower()
    if explicit_family:
        return _detect_model_family_from_probe(explicit_family)

    probe = " ".join(
        str(value or "")
        for value in (
            meta.get("combined_source_model_name"),
            sample.get("model_name"),
        )
    ).lower()
    return _detect_model_family_from_probe(probe) or "unknown"


def materialize_computed_fields(samples: List[Dict[str, Any]]) -> None:
    """Write computed diversity fields into generation_metadata.

    These computed fields are used by diversity sampling so they can be
    referenced via the standard dot-path mechanism (e.g.
    ``generation_metadata.toolkit_signature``).
    """
    for s in samples:
        meta = s.setdefault("generation_metadata", {})
        meta["domain_signature"] = compute_domain_signature(s)
        meta["toolkit_signature"] = compute_toolkit_signature(s)
        meta["subject_scope"] = compute_subject_scope(s)
        meta["model_family"] = compute_model_family(s)


# ─── Diversity sampling ─────────────────────────────────────────────────────

def _path_is_domain_field(path: str) -> bool:
    return path.endswith(".domains") or path.endswith(".scenario_domain") or path in {"domains", "scenario_domain"}


def _path_is_toolkit_field(path: str) -> bool:
    return path.endswith(".toolkits") or path == "toolkits"


def _has_nested_explicit(obj: Dict[str, Any], path: str) -> bool:
    value: Any = obj
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def _get_nested_raw(obj: Dict[str, Any], path: str) -> Any:
    """Get a nested field value by dot-separated path with domain-schema fallback."""
    value: Any = obj
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        if key in value:
            value = value[key]
            continue
        if key == "domains" and "scenario_domain" in value:
            value = value["scenario_domain"]
            continue
        if key == "scenario_domain" and "domains" in value:
            value = value["domains"]
            continue
        return None
    return value


_DOMAIN_FIELD_PATHS: Tuple[str, ...] = (
    "generation_metadata.domains",
    "seed.domains",
    "generation_metadata.scenario_domain",
    "seed.scenario_domain",
)


def _sample_domains(sample: Dict[str, Any]) -> List[str]:
    """Return the canonical domain list for a sample, with schema fallback."""
    for field_path in _DOMAIN_FIELD_PATHS:
        raw = _get_nested_raw(sample, field_path)
        if raw in (None, "", []):
            continue
        domains = normalize_domain_list(raw)
        if domains:
            return domains
    return []


def compute_domain_signature(sample: Dict[str, Any]) -> str:
    """Derive an order-invariant domain signature from sample metadata."""
    return build_domain_signature(_sample_domains(sample))


def compute_domain_pairs(sample: Dict[str, Any]) -> List[str]:
    """Return normalized co-occurring domain-pair labels for audit reporting."""
    domains = _sample_domains(sample)
    return ["|".join(pair) for pair in itertools.combinations(domains, 2)]


def normalize_toolkit_list(raw: Any) -> List[str]:
    if raw in (None, ""):
        return []

    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        candidates = [raw]

    normalized: List[str] = []
    seen: Set[str] = set()
    for item in candidates:
        toolkit = str(item).strip().lower()
        if not toolkit or toolkit in seen:
            continue
        seen.add(toolkit)
        normalized.append(toolkit)
    return normalized


def _normalize_field_key(value: Any, path: str) -> str:
    if _path_is_domain_field(path):
        domains = normalize_domain_list(value)
        if not domains:
            return ""
        return json.dumps(domains, ensure_ascii=False)
    if _path_is_toolkit_field(path):
        toolkits = normalize_toolkit_list(value)
        if not toolkits:
            return ""
        return json.dumps(toolkits, ensure_ascii=False)
    if isinstance(value, list):
        normalized = [str(item).strip().lower() for item in value if str(item).strip()]
        if not normalized:
            return ""
        return json.dumps(normalized, ensure_ascii=False)
    return str(value).strip().lower() if value not in (None, "") else ""


def _get_nested(obj: Dict[str, Any], path: str) -> str:
    """Get a normalized nested field value by dot-separated path."""
    return _normalize_field_key(_get_nested_raw(obj, path), path)


def _field_contributions(sample: Dict[str, Any], field_path: str) -> List[Tuple[str, float]]:
    """Return category contributions for balancing.

    Most fields contribute fully to one category. Domain-list fields contribute
    one full count to each normalized domain or toolkit in the list.
    """
    raw = _get_nested_raw(sample, field_path)
    if _path_is_domain_field(field_path):
        domains = normalize_domain_list(raw)
        if not domains:
            return [("unknown", 1.0)]
        return [(domain, 1.0) for domain in domains]
    if _path_is_toolkit_field(field_path):
        toolkits = normalize_toolkit_list(raw)
        if not toolkits:
            return [("unknown", 1.0)]
        return [(toolkit, 1.0) for toolkit in toolkits]

    key = _normalize_field_key(raw, field_path) or "unknown"
    return [(key, 1.0)]


def _field_mass_counter(samples: List[Dict[str, Any]], field_path: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample in samples:
        for category, weight in _field_contributions(sample, field_path):
            counts[category] += weight
    return counts


def _normalize_field_max_pcts(
    field_max_pcts: Dict[str, float] | None,
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for field_path, pct in (field_max_pcts or {}).items():
        if not field_path or pct is None or pct <= 0:
            continue
        normalized[field_path.strip()] = float(pct)
    return normalized


def build_field_max_pcts(args: argparse.Namespace) -> Dict[str, float]:
    return _normalize_field_max_pcts(
        {
            DOMAIN_SIGNATURE_FIELD: args.domain_signature_max_pct,
            TOOLKIT_FIELD: args.toolkit_max_pct,
        }
    )


def resolve_balance_fields(args: argparse.Namespace) -> List[str]:
    balance_fields: List[str] = []
    seen: Set[str] = set()
    for field_path in args.balance_fields:
        normalized = str(field_path).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        balance_fields.append(normalized)

    if args.toolkit_max_pct is not None and args.toolkit_max_pct > 0:
        if TOOLKIT_FIELD not in seen:
            balance_fields.append(TOOLKIT_FIELD)

    return balance_fields


def _resolve_field_max_pct(
    field_path: str,
    default_max_pct: float,
    field_max_pcts: Dict[str, float] | None = None,
) -> float:
    normalized = field_max_pcts or {}
    return float(normalized.get(field_path, default_max_pct))


def _cap_count_for_field(
    total: int,
    field_path: str,
    default_max_pct: float,
    field_max_pcts: Dict[str, float] | None = None,
) -> int:
    max_pct = _resolve_field_max_pct(field_path, default_max_pct, field_max_pcts)
    return max(int(total * max_pct / 100.0), 1)


def format_field_cap_pcts(
    field_paths: List[str],
    default_max_pct: float,
    field_max_pcts: Dict[str, float] | None = None,
) -> str:
    normalized = _normalize_field_max_pcts(field_max_pcts)
    if not normalized:
        return f"default={default_max_pct:.1f}%"

    formatted = []
    seen: Set[str] = set()
    for field_path in field_paths:
        if field_path in seen:
            continue
        seen.add(field_path)
        pct = _resolve_field_max_pct(field_path, default_max_pct, normalized)
        if abs(pct - default_max_pct) <= 1e-9:
            continue
        formatted.append(f"{field_path}={pct:.1f}%")

    if not formatted:
        return f"default={default_max_pct:.1f}%"
    return f"default={default_max_pct:.1f}% with overrides: " + ", ".join(formatted)


def _field_score_against_counts(
    sample: Dict[str, Any],
    field_path: str,
    counts: Counter[str],
) -> float:
    return sum(
        weight * counts.get(category, 0.0)
        for category, weight in _field_contributions(sample, field_path)
    )


def _sample_matches_category(sample: Dict[str, Any], field_path: str, category: str) -> bool:
    normalized_category = str(category).strip().lower()
    return any(
        contribution_category == normalized_category
        for contribution_category, _ in _field_contributions(sample, field_path)
    )


def enforce_min_category_pcts(
    samples: List[Dict[str, Any]],
    field_path: str,
    min_pcts: Dict[str, float],
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Maximize retained samples while enforcing per-category minimum shares."""
    if rng is None:
        rng = random.Random()

    if not samples:
        return samples

    normalized_min_pcts = {
        str(category).strip().lower(): float(pct)
        for category, pct in min_pcts.items()
        if float(pct) > 0
    }
    if not normalized_min_pcts:
        return samples

    total_requested = sum(normalized_min_pcts.values())
    if total_requested > 100.0:
        LOGGER.warning(
            "  Minimum percentages for '%s' sum to %.1f%% (> 100%%) — skipping",
            field_path,
            total_requested,
        )
        return samples

    category_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        category = _get_nested(sample, field_path) or "unknown"
        category_to_indices[category].append(idx)

    active_min_pcts: Dict[str, float] = {}
    for category, pct in normalized_min_pcts.items():
        if category_to_indices.get(category):
            active_min_pcts[category] = pct
        else:
            LOGGER.warning(
                "  No '%s' samples found for '%s' minimum %.1f%% — ignoring that constraint",
                category,
                field_path,
                pct,
            )

    if not active_min_pcts:
        return samples

    target_total = min(
        len(samples),
        min(
            math.floor(len(category_to_indices[category]) * 100.0 / pct)
            for category, pct in active_min_pcts.items()
        ),
    )
    if target_total >= len(samples):
        LOGGER.info(
            "  Minimum percentages for '%s' already satisfied — no trimming needed",
            field_path,
        )
        return samples

    required_counts = {
        category: math.ceil(target_total * pct / 100.0)
        for category, pct in active_min_pcts.items()
    }
    kept_by_category: Dict[str, List[int]] = {
        category: list(indices)
        for category, indices in category_to_indices.items()
    }
    excess = len(samples) - target_total

    LOGGER.info(
        "  Minimum-percentage balance on '%s' with target total %d",
        field_path,
        target_total,
    )

    while excess > 0:
        eligible = [
            category
            for category, indices in kept_by_category.items()
            if len(indices) > required_counts.get(category, 0)
        ]
        if not eligible:
            LOGGER.warning(
                "    Unable to trim '%s' further without violating minimum-percentage constraints",
                field_path,
            )
            break

        category = max(
            eligible,
            key=lambda cat: (
                len(kept_by_category[cat]) - required_counts.get(cat, 0),
                len(kept_by_category[cat]),
            ),
        )
        remove_pos = rng.randrange(len(kept_by_category[category]))
        kept_by_category[category].pop(remove_pos)
        excess -= 1

    keep_indices: Set[int] = set()
    for indices in kept_by_category.values():
        keep_indices.update(indices)

    for category in sorted(kept_by_category):
        before = len(category_to_indices[category])
        after = len(kept_by_category[category])
        if before != after:
            LOGGER.info("    '%s': %d -> %d (-%d)", category, before, after, before - after)

    return [samples[i] for i in sorted(keep_indices)]


def _normalize_requested_families(families: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for family in families:
        candidate = _detect_model_family_from_probe(family)
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def resolve_model_families(
    samples: List[Dict[str, Any]],
    families: List[str] | None,
) -> List[str]:
    """Return explicit requested families or infer sensible defaults from data."""
    normalized_families = _normalize_requested_families(families or [])
    if normalized_families:
        return normalized_families

    present_counts = Counter(
        compute_model_family(sample)
        for sample in samples
    )
    inferred_primary = [
        family
        for family in PRIMARY_MODEL_FAMILIES
        if present_counts[family] > 0
    ]
    if inferred_primary:
        return inferred_primary

    if present_counts["step"] > 0:
        return ["step"]

    return [
        family
        for family, count in present_counts.most_common()
        if family not in {"", "unknown"} and count > 0
    ]


def build_min_pct_constraints(
    samples: List[Dict[str, Any]],
    scope_min_pcts: Dict[str, float],
    families: List[str] | None = None,
    model_family_min_pct: float | None = None,
) -> List[Tuple[str, str, float]]:
    """Collect minimum-share constraints enforced jointly during Stage 4/7."""
    constraints: List[Tuple[str, str, float]] = []
    for scope, pct in scope_min_pcts.items():
        if pct > 0:
            constraints.append(("generation_metadata.subject_scope", scope, pct))

    if model_family_min_pct is None or model_family_min_pct <= 0:
        return constraints

    present_families = {
        compute_model_family(sample)
        for sample in samples
    }
    for family in _normalize_requested_families(families or []):
        if family in present_families:
            constraints.append(
                ("generation_metadata.model_family", family, model_family_min_pct)
            )
    return constraints


def format_min_pct_constraints(
    constraints: List[Tuple[str, str, float]],
) -> str:
    """Return a readable summary of minimum-share constraints."""
    if not constraints:
        return "none"

    labels = {
        "generation_metadata.subject_scope": "subject_scope minimums",
        "generation_metadata.model_family": "model_family minimums",
    }
    by_field: Dict[str, List[str]] = defaultdict(list)
    for field, value, pct in constraints:
        by_field[field].append(f"{value}={pct:.1f}%")

    ordered_fields = [
        "generation_metadata.subject_scope",
        "generation_metadata.model_family",
    ]
    parts: List[str] = []
    for field in ordered_fields:
        values = by_field.pop(field, [])
        if values:
            parts.append(f"{labels.get(field, field)}: {', '.join(values)}")
    for field in sorted(by_field):
        parts.append(f"{labels.get(field, field)}: {', '.join(by_field[field])}")
    return "; ".join(parts)


def _max_equalized_family_target_with_self_pct(
    family_total_counts: Dict[str, int],
    family_self_counts: Dict[str, int],
    self_pct: float,
) -> int:
    """Return the largest equal per-family target that can satisfy *self_pct*."""
    if not family_total_counts:
        return 0

    max_target = min(family_total_counts.values())
    if self_pct <= 0:
        return max_target
    if self_pct > 100.0:
        return 0

    families = list(family_total_counts.keys())
    low = 0
    high = max_target
    while low < high:
        target = (low + high + 1) // 2
        required_self = math.ceil(len(families) * target * self_pct / 100.0)
        available_self = sum(min(family_self_counts.get(family, 0), target) for family in families)
        if available_self >= required_self:
            low = target
        else:
            high = target - 1
    return low


def equalize_model_families_with_self_scope(
    samples: List[Dict[str, Any]],
    families: List[str],
    self_pct: float,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Jointly satisfy equal model-family counts and a minimum self-scope share.

    This is more sample-efficient than running scope balancing and family
    equalization as two separate random downsampling stages.
    """
    if rng is None:
        rng = random.Random()

    if not samples:
        return samples

    normalized_families = _normalize_requested_families(families)
    if not normalized_families:
        LOGGER.warning("  No model families requested — skipping joint family/scope balance")
        return samples

    family_self_indices: Dict[str, List[int]] = {family: [] for family in normalized_families}
    family_other_indices: Dict[str, List[int]] = {family: [] for family in normalized_families}
    dropped_non_target = 0

    for idx, sample in enumerate(samples):
        family = _get_nested(sample, "generation_metadata.model_family") or compute_model_family(sample)
        if family not in family_self_indices:
            dropped_non_target += 1
            continue
        scope = _get_nested(sample, "generation_metadata.subject_scope") or compute_subject_scope(sample)
        if scope == "self":
            family_self_indices[family].append(idx)
        else:
            family_other_indices[family].append(idx)

    present = {
        family: (family_self_indices[family], family_other_indices[family])
        for family in normalized_families
        if family_self_indices[family] or family_other_indices[family]
    }
    if not present:
        LOGGER.warning(
            "  None of the requested model families are present (%s) — skipping joint family/scope balance",
            ", ".join(normalized_families),
        )
        return samples

    family_total_counts = {
        family: len(self_indices) + len(other_indices)
        for family, (self_indices, other_indices) in present.items()
    }
    family_self_counts = {
        family: len(self_indices)
        for family, (self_indices, _) in present.items()
    }
    target_size = _max_equalized_family_target_with_self_pct(
        family_total_counts,
        family_self_counts,
        self_pct,
    )
    if target_size == 0:
        LOGGER.warning(
            "  Unable to satisfy self_pct=%.1f%% jointly with equal family counts across %s; "
            "falling back to model-family balance only",
            self_pct,
            ", ".join(present.keys()),
        )
        return equalize_model_families(samples, families, rng)

    required_self = math.ceil(len(present) * target_size * self_pct / 100.0)
    selected_self_counts = {
        family: max(0, target_size - len(other_indices))
        for family, (_, other_indices) in present.items()
    }
    remaining_self = required_self - sum(selected_self_counts.values())

    family_order = list(present.keys())
    rng.shuffle(family_order)
    family_order.sort(
        key=lambda family: (
            min(family_self_counts[family], target_size) - selected_self_counts[family],
            family_self_counts[family],
        ),
        reverse=True,
    )
    for family in family_order:
        if remaining_self <= 0:
            break
        extra_self_capacity = min(family_self_counts[family], target_size) - selected_self_counts[family]
        if extra_self_capacity <= 0:
            continue
        add = min(extra_self_capacity, remaining_self)
        selected_self_counts[family] += add
        remaining_self -= add

    if remaining_self > 0:
        LOGGER.warning(
            "  Joint family/scope allocation could not place %d required self samples; "
            "falling back to model-family balance only",
            remaining_self,
        )
        return equalize_model_families(samples, families, rng)

    keep_indices: Set[int] = set()
    LOGGER.info(
        "  Joint family/scope balance across %s with target size %d each and self_pct=%.1f%%",
        ", ".join(present.keys()),
        target_size,
        self_pct,
    )

    for family in normalized_families:
        family_indices = present.get(family)
        if not family_indices:
            LOGGER.warning("    '%s': absent", family)
            continue

        self_indices, other_indices = family_indices
        self_keep = selected_self_counts[family]
        other_keep = target_size - self_keep

        if self_keep >= len(self_indices):
            keep_indices.update(self_indices)
        elif self_keep > 0:
            keep_indices.update(rng.sample(self_indices, self_keep))

        if other_keep >= len(other_indices):
            keep_indices.update(other_indices)
        elif other_keep > 0:
            keep_indices.update(rng.sample(other_indices, other_keep))

        LOGGER.info(
            "    '%s': total %d -> %d (-%d), self %d -> %d, non-self %d -> %d",
            family,
            len(self_indices) + len(other_indices),
            target_size,
            len(self_indices) + len(other_indices) - target_size,
            len(self_indices),
            self_keep,
            len(other_indices),
            other_keep,
        )

    final = [samples[i] for i in sorted(keep_indices)]
    final_self = sum(selected_self_counts.values())
    LOGGER.info(
        "  Joint family/scope balance kept %d samples, dropped %d non-target samples, "
        "achieved self_pct=%.1f%%",
        len(final),
        dropped_non_target,
        100.0 * final_self / len(final) if final else 0.0,
    )
    return final


def equalize_model_families(
    samples: List[Dict[str, Any]],
    families: List[str],
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Downsample target model families to the smallest present family size.

    This maximizes retention while producing an equal-count mixture across the
    requested families. Samples from non-target families are dropped when this
    stage is enabled so the final family mix can sum to 100%.
    """
    if rng is None:
        rng = random.Random()

    if not samples:
        return samples

    normalized_families = _normalize_requested_families(families)
    if not normalized_families:
        LOGGER.warning("  No model families requested — skipping model-family balance")
        return samples

    family_to_indices: Dict[str, List[int]] = {family: [] for family in normalized_families}
    dropped_non_target = 0
    for idx, sample in enumerate(samples):
        family = _get_nested(sample, "generation_metadata.model_family") or compute_model_family(sample)
        if family in family_to_indices:
            family_to_indices[family].append(idx)
        else:
            dropped_non_target += 1

    present = {family: indices for family, indices in family_to_indices.items() if indices}
    if not present:
        LOGGER.warning(
            "  None of the requested model families are present (%s) — skipping model-family balance",
            ", ".join(normalized_families),
        )
        return samples

    target_size = min(len(indices) for indices in present.values())
    keep_indices: Set[int] = set()

    LOGGER.info(
        "  Model-family balance across %s with target size %d each",
        ", ".join(present.keys()),
        target_size,
    )

    for family in normalized_families:
        indices = present.get(family)
        if not indices:
            LOGGER.warning("    '%s': absent", family)
            continue
        if len(indices) > target_size:
            selected = rng.sample(indices, target_size)
            keep_indices.update(selected)
            LOGGER.info(
                "    '%s': %d -> %d (-%d)",
                family, len(indices), target_size, len(indices) - target_size,
            )
        else:
            keep_indices.update(indices)
            LOGGER.info("    '%s': %d -> %d", family, len(indices), len(indices))

    final = [samples[i] for i in sorted(keep_indices)]
    LOGGER.info(
        "  Model-family balance kept %d samples and dropped %d non-target samples",
        len(final),
        dropped_non_target,
    )
    return final


def enforce_min_model_family_pct(
    samples: List[Dict[str, Any]],
    families: List[str],
    min_pct: float,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Downsample only as much as needed so each target family reaches *min_pct*.

    This is softer than exact equalization. It keeps the maximum number of
    samples subject to the constraint that every requested model family
    occupies at least ``min_pct`` percent of the final dataset.
    """
    if rng is None:
        rng = random.Random()

    if not samples:
        return samples

    normalized_families = _normalize_requested_families(families)
    if not normalized_families:
        LOGGER.warning("  No model families requested — skipping model-family min-pct balance")
        return samples

    if min_pct <= 0:
        LOGGER.warning("  Non-positive min model-family pct %.2f — skipping", min_pct)
        return samples

    if len(normalized_families) * min_pct > 100.0:
        LOGGER.warning(
            "  Requested min model-family pct %.2f is impossible for %d families — skipping",
            min_pct, len(normalized_families),
        )
        return samples

    target_family_indices: Dict[str, List[int]] = {family: [] for family in normalized_families}
    non_target_indices: List[int] = []
    for idx, sample in enumerate(samples):
        family = _get_nested(sample, "generation_metadata.model_family") or compute_model_family(sample)
        if family in target_family_indices:
            target_family_indices[family].append(idx)
        else:
            non_target_indices.append(idx)

    present = {family: indices for family, indices in target_family_indices.items() if indices}
    if not present:
        LOGGER.warning(
            "  None of the requested model families are present (%s) — skipping model-family min-pct balance",
            ", ".join(normalized_families),
        )
        return samples

    max_total_by_constraint = min(
        math.floor(len(indices) * 100.0 / min_pct)
        for indices in present.values()
    )
    target_total = min(len(samples), max_total_by_constraint)
    if target_total >= len(samples):
        LOGGER.info(
            "  Model-family min-pct already satisfied at %.2f%% for %s — no trimming needed",
            min_pct, ", ".join(present.keys()),
        )
        return samples

    required_counts = {
        family: math.ceil(target_total * min_pct / 100.0)
        for family in present.keys()
    }

    kept_by_family: Dict[str, List[int]] = {
        family: list(indices) for family, indices in target_family_indices.items()
    }
    kept_non_target = list(non_target_indices)

    excess = len(samples) - target_total

    LOGGER.info(
        "  Model-family min-pct balance across %s with min_pct=%.2f%% and target total %d",
        ", ".join(present.keys()),
        min_pct,
        target_total,
    )

    # Drop non-target samples first since they do not help satisfy the target constraint.
    if excess > 0 and kept_non_target:
        drop_count = min(excess, len(kept_non_target))
        drop_positions = sorted(rng.sample(range(len(kept_non_target)), drop_count), reverse=True)
        for pos in drop_positions:
            kept_non_target.pop(pos)
        excess -= drop_count
        LOGGER.info("    non-target: %d -> %d (-%d)", len(non_target_indices), len(kept_non_target), drop_count)

    # Then trim the most overrepresented target families until the target total is reached.
    while excess > 0:
        eligible = [
            family
            for family, indices in kept_by_family.items()
            if family in required_counts and len(indices) > required_counts[family]
        ]
        if not eligible:
            LOGGER.warning("    Unable to trim further without violating min-pct constraints")
            break

        family = max(
            eligible,
            key=lambda fam: (len(kept_by_family[fam]) - required_counts[fam], len(kept_by_family[fam])),
        )
        remove_pos = rng.randrange(len(kept_by_family[family]))
        kept_by_family[family].pop(remove_pos)
        excess -= 1

    final_counts = {family: len(indices) for family, indices in kept_by_family.items() if indices}
    for family in normalized_families:
        before = len(target_family_indices[family])
        after = len(kept_by_family[family])
        if before or after:
            LOGGER.info("    '%s': %d -> %d (-%d)", family, before, after, before - after)

    keep_indices: Set[int] = set(kept_non_target)
    for indices in kept_by_family.values():
        keep_indices.update(indices)

    final = [samples[i] for i in sorted(keep_indices)]
    achieved = {
        family: (100.0 * count / len(final)) if final else 0.0
        for family, count in final_counts.items()
    }
    LOGGER.info(
        "  Model-family min-pct balance kept %d samples; achieved percentages: %s",
        len(final),
        ", ".join(f"{family}={pct:.1f}%" for family, pct in achieved.items()),
    )
    return final


def project_model_family_balanced_size(
    samples: List[Dict[str, Any]],
    families: List[str],
    *,
    equalize: bool = False,
    min_pct: float | None = None,
    self_pct: float | None = None,
) -> int:
    """Return the projected final size after model-family balancing."""
    if not samples:
        return 0

    normalized_families = _normalize_requested_families(families)
    if not normalized_families:
        return len(samples)

    family_counts = Counter(compute_model_family(sample) for sample in samples)
    present_counts = {
        family: family_counts[family]
        for family in normalized_families
        if family_counts[family] > 0
    }

    if min_pct is not None:
        if min_pct <= 0 or len(normalized_families) * min_pct > 100.0 or not present_counts:
            return len(samples)
        max_total_by_constraint = min(
            math.floor(count * 100.0 / min_pct)
            for count in present_counts.values()
        )
        return min(len(samples), max_total_by_constraint)

    if equalize:
        if not present_counts:
            return len(samples)
        if self_pct is not None:
            family_self_counts: Dict[str, int] = {family: 0 for family in present_counts}
            for sample in samples:
                family = compute_model_family(sample)
                if family in family_self_counts and compute_subject_scope(sample) == "self":
                    family_self_counts[family] += 1
            target_size = _max_equalized_family_target_with_self_pct(
                present_counts,
                family_self_counts,
                self_pct,
            )
            if target_size > 0:
                return target_size * len(present_counts)
        return min(present_counts.values()) * len(present_counts)

    return len(samples)


def compute_cap_excess(
    samples: List[Dict[str, Any]],
    field_paths: List[str],
    max_pct: float,
    field_max_pcts: Dict[str, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Return total samples above cap across the requested balance fields."""
    if not samples:
        return 0, {field_path: 0 for field_path in field_paths}

    normalized_field_max_pcts = _normalize_field_max_pcts(field_max_pcts)
    excess_by_field: Dict[str, float] = {}
    total_excess = 0.0
    for field_path in field_paths:
        cap = _cap_count_for_field(
            len(samples),
            field_path,
            max_pct,
            normalized_field_max_pcts,
        )
        counts = _field_mass_counter(samples, field_path)
        field_excess = sum(max(count - cap, 0.0) for count in counts.values())
        excess_by_field[field_path] = field_excess
        total_excess += field_excess
    return total_excess, excess_by_field


def _field_cap_is_enforceable(
    samples: List[Dict[str, Any]],
    field_path: str,
    max_pct: float,
    field_max_pcts: Dict[str, float] | None = None,
) -> bool:
    """Return whether a field can be meaningfully tightened at *max_pct*."""
    if not samples:
        return False

    field_max_pct = _resolve_field_max_pct(
        field_path,
        max_pct,
        _normalize_field_max_pcts(field_max_pcts),
    )
    contributions_by_sample = [
        _field_contributions(sample, field_path)
        for sample in samples
    ]
    categories = {
        category
        for contributions in contributions_by_sample
        for category, _ in contributions
    }
    if not categories:
        return False

    multi_category_field = any(len(contributions) > 1 for contributions in contributions_by_sample)
    if not multi_category_field and len(categories) * field_max_pct < 100.0:
        return False
    return True


def select_enforceable_balance_fields(
    samples: List[Dict[str, Any]],
    field_paths: List[str],
    max_pct: float,
    field_max_pcts: Dict[str, float] | None = None,
) -> Tuple[List[str], List[str]]:
    """Split requested balance fields into enforceable vs skipped."""
    normalized_field_max_pcts = _normalize_field_max_pcts(field_max_pcts)
    enforceable: List[str] = []
    skipped: List[str] = []
    for field_path in field_paths:
        if _field_cap_is_enforceable(
            samples,
            field_path,
            max_pct,
            field_max_pcts=normalized_field_max_pcts,
        ):
            enforceable.append(field_path)
        else:
            skipped.append(field_path)
    return enforceable, skipped


def _small_total_cap_upper_bound(max_pct: float) -> int:
    """Return the largest total for which the cap is forced to 1 sample."""
    if max_pct <= 0:
        return 0
    threshold = (
        Decimal("100") / Decimal(str(max_pct))
    ).to_integral_value(rounding=ROUND_CEILING)
    return max(int(threshold) - 1, 0)


def _finalize_kept_subset(
    samples: List[Dict[str, Any]],
    keep_indices: List[int],
    field_paths: List[str],
    max_pct: float,
    *,
    field_max_pcts: Dict[str, float] | None = None,
    pct_constraints: List[Tuple[str, str, float]] | None = None,
    count_constraints: Dict[Tuple[str, str], int] | None = None,
    progress_desc: str = "Cap tightening",
) -> List[Dict[str, Any]]:
    """Validate and report the final kept subset for any sampling backend."""
    final_samples = [samples[idx] for idx in keep_indices]
    LOGGER.info("  %s: %d -> %d samples", progress_desc, len(samples), len(final_samples))

    final_excess, excess_by_field = compute_cap_excess(
        final_samples,
        field_paths,
        max_pct,
        field_max_pcts=field_max_pcts,
    )
    if final_excess > 1e-9:
        LOGGER.warning(
            "  %s: some balance caps remain infeasible under minimum-share constraints: %s",
            progress_desc,
            ", ".join(
                f"{field} excess={excess:.2f}"
                for field, excess in excess_by_field.items()
                if excess > 1e-9
            ),
        )

    deficits, count_shortfalls = _evaluate_min_constraint_violations(
        final_samples,
        pct_constraints,
        count_constraints,
    )

    if deficits:
        LOGGER.warning(
            "  %s: some minimum-share constraints remain infeasible: %s",
            progress_desc,
            ", ".join(
                f"{field}={value} deficit={deficit}"
                for (field, value), deficit in sorted(deficits.items())
            ),
        )
    if count_shortfalls:
        LOGGER.warning(
            "  %s: some minimum-count constraints remain infeasible: %s",
            progress_desc,
            ", ".join(
                f"{field}={value} shortfall={shortfall}"
                for (field, value), shortfall in sorted(count_shortfalls.items())
            ),
        )

    return final_samples


def _evaluate_min_constraint_violations(
    samples: List[Dict[str, Any]],
    pct_constraints: List[Tuple[str, str, float]] | None = None,
    count_constraints: Dict[Tuple[str, str], int] | None = None,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
    """Return shortfalls for minimum-share and minimum-count constraints."""
    deficits: Dict[Tuple[str, str], int] = {}
    for field, value, pct in (pct_constraints or []):
        required = math.ceil(len(samples) * pct / 100.0)
        actual = sum(
            1
            for sample in samples
            if _sample_matches_category(sample, field, value)
        )
        deficit = max(required - actual, 0)
        if deficit > 0:
            deficits[(field, value)] = deficit

    count_shortfalls: Dict[Tuple[str, str], int] = {}
    for key, min_count in (count_constraints or {}).items():
        field, value = key
        actual = sum(
            1
            for sample in samples
            if _sample_matches_category(sample, field, value)
        )
        shortfall = max(min_count - actual, 0)
        if shortfall > 0:
            count_shortfalls[key] = shortfall

    return deficits, count_shortfalls


def tighten_diversity_caps(
    samples: List[Dict[str, Any]],
    field_paths: List[str],
    max_pct: float,
    *,
    rng: random.Random | None = None,
    preferred_families: List[str] | None = None,
    field_max_pcts: Dict[str, float] | None = None,
    max_total_count: int | None = None,
    protected_pct_constraints: List[Tuple[str, str, float]] | None = None,
    protected_count_constraints: Dict[Tuple[str, str], int] | None = None,
    progress_desc: str = "Cap tightening",
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
) -> List[Dict[str, Any]]:
    """Solve diversity sampling exactly with a MILP backend when available."""
    del rng, preferred_families
    if not SCIPY_MILP_AVAILABLE:
        raise RuntimeError(
            "Exact sampling requires SciPy MILP support. Install scipy and retry."
        )
    if not samples or not field_paths:
        return samples
    if max_total_count is not None and max_total_count <= 0:
        raise ValueError("max_total_count must be positive when provided")

    normalized_field_max_pcts = _normalize_field_max_pcts(field_max_pcts)
    active_fields, skipped_fields = select_enforceable_balance_fields(
        samples,
        field_paths,
        max_pct,
        field_max_pcts=normalized_field_max_pcts,
    )
    if skipped_fields:
        LOGGER.info(
            "  %s: skipping mathematically infeasible balance fields: %s",
            progress_desc,
            ", ".join(skipped_fields),
        )
    if not active_fields:
        return samples

    normalized_pct_constraints = [
        (field.strip(), value.strip().lower(), float(pct))
        for field, value, pct in (protected_pct_constraints or [])
        if field and value and pct > 0
    ]
    normalized_count_constraints = {
        (field.strip(), value.strip().lower()): int(min_count)
        for (field, value), min_count in (protected_count_constraints or {}).items()
        if field and value and min_count > 0
    }

    n_samples = len(samples)
    total_var_idx = n_samples
    n_vars = n_samples + 1
    small_total_upper_by_field: Dict[str, int] = {}
    small_total_var_idx_by_threshold: Dict[int, int] = {}
    for field_path in active_fields:
        field_max_pct = _resolve_field_max_pct(
            field_path,
            max_pct,
            normalized_field_max_pcts,
        )
        small_total_upper = _small_total_cap_upper_bound(field_max_pct)
        small_total_upper_by_field[field_path] = small_total_upper
        if 0 < small_total_upper < n_samples and small_total_upper not in small_total_var_idx_by_threshold:
            small_total_var_idx_by_threshold[small_total_upper] = n_vars
            n_vars += 1

    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    def add_row(entries: List[Tuple[int, float]], lb: float, ub: float) -> None:
        row_idx = len(lower_bounds)
        for col_idx, coeff in entries:
            if abs(coeff) <= 1e-12:
                continue
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            values.append(float(coeff))
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    # Total selected samples: sum(x_i) == total.
    total_entries = [(idx, 1.0) for idx in range(n_samples)]
    total_entries.append((total_var_idx, -1.0))
    add_row(total_entries, 0.0, 0.0)

    # Link each "small total" binary to whether the cap floor is still 1 sample.
    for small_total_upper, small_total_var_idx in small_total_var_idx_by_threshold.items():
        add_row(
            [(total_var_idx, 1.0), (small_total_var_idx, float(n_samples))],
            float(small_total_upper + 1),
            np.inf,
        )
        add_row(
            [(total_var_idx, 1.0), (small_total_var_idx, float(n_samples))],
            -np.inf,
            float(small_total_upper + n_samples),
        )

    for field, value, pct in normalized_pct_constraints:
        entries = [
            (idx, 100.0)
            for idx, sample in enumerate(samples)
            if _sample_matches_category(sample, field, value)
        ]
        entries.append((total_var_idx, -pct))
        add_row(entries, 0.0, np.inf)

    for (field, value), min_count in normalized_count_constraints.items():
        entries = [
            (idx, 1.0)
            for idx, sample in enumerate(samples)
            if _sample_matches_category(sample, field, value)
        ]
        add_row(entries, float(min_count), np.inf)

    category_constraints = 0
    field_contributions: Dict[str, List[List[Tuple[str, float]]]] = {
        field_path: [
            _field_contributions(sample, field_path)
            for sample in samples
        ]
        for field_path in active_fields
    }
    for field_path, contributions_by_sample in field_contributions.items():
        field_max_pct = _resolve_field_max_pct(
            field_path,
            max_pct,
            normalized_field_max_pcts,
        )
        small_total_upper = small_total_upper_by_field[field_path]
        small_total_var_idx = small_total_var_idx_by_threshold.get(small_total_upper)
        category_to_entries: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for idx, contributions in enumerate(contributions_by_sample):
            for category, weight in contributions:
                category_to_entries[category].append((idx, float(weight)))

        for entries in category_to_entries.values():
            category_constraints += 1
            if small_total_upper >= n_samples:
                add_row(entries, -np.inf, 1.0)
                continue

            if small_total_var_idx is None:
                cap_entries = [(idx, 100.0 * weight) for idx, weight in entries]
                cap_entries.append((total_var_idx, -field_max_pct))
                add_row(cap_entries, -np.inf, 0.0)
                continue

            # If the final total is still tiny, the cap remains exactly 1 sample.
            one_cap_entries = [(idx, weight) for idx, weight in entries]
            one_cap_entries.append((small_total_var_idx, float(n_samples)))
            add_row(one_cap_entries, -np.inf, float(n_samples + 1))

            # Otherwise use the usual percentage cap.
            relax_m = float(100 * n_samples)
            cap_entries = [(idx, 100.0 * weight) for idx, weight in entries]
            cap_entries.append((total_var_idx, -field_max_pct))
            cap_entries.append((small_total_var_idx, -relax_m))
            add_row(cap_entries, -np.inf, 0.0)

    LOGGER.info(
        "  %s: using exact MILP balancing across %d fields, %d cap categories, %d minimum-share constraints, %d minimum-count constraints (%s)",
        progress_desc,
        len(active_fields),
        category_constraints,
        len(normalized_pct_constraints),
        len(normalized_count_constraints),
        format_field_cap_pcts(active_fields, max_pct, normalized_field_max_pcts),
    )

    bounds_lb = np.zeros(n_vars, dtype=float)
    bounds_ub = np.ones(n_vars, dtype=float)
    bounds_ub[total_var_idx] = float(
        min(n_samples, max_total_count) if max_total_count is not None else n_samples
    )
    for small_total_var_idx in small_total_var_idx_by_threshold.values():
        bounds_ub[small_total_var_idx] = 1.0

    # SciPy's HiGHS wrapper expects C-int index/integrality buffers on this
    # platform; NumPy 2 can otherwise default these to long/int64.
    integrality = np.ones(n_vars, dtype=np.intc)
    objective = np.zeros(n_vars, dtype=float)
    objective[total_var_idx] = -1.0

    row_index_array = np.asarray(row_indices, dtype=np.intc)
    col_index_array = np.asarray(col_indices, dtype=np.intc)
    matrix = coo_array(
        (np.asarray(values, dtype=float), (row_index_array, col_index_array)),
        shape=(len(lower_bounds), n_vars),
    )
    constraints = LinearConstraint(
        matrix,
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )
    options: Dict[str, Any] = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = float(mip_rel_gap)

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(bounds_lb, bounds_ub),
        constraints=constraints,
        options=options,
    )

    status_to_label = {
        0: "optimal",
        1: "time-or-node-limit",
        2: "infeasible",
        3: "unbounded",
        4: "error",
    }
    LOGGER.info(
        "  %s: MILP status=%s (%s), objective=%s, nodes=%s, gap=%s",
        progress_desc,
        result.status,
        status_to_label.get(result.status, "unknown"),
        None if result.fun is None else f"{-result.fun:.0f}",
        result.mip_node_count,
        result.mip_gap,
    )

    if result.status == 2:
        raise RuntimeError(
            f"{progress_desc}: no feasible subset satisfies all caps and minimum-share constraints"
        )
    if result.status not in {0, 1} or result.x is None:
        raise RuntimeError(f"{progress_desc}: exact MILP solve failed: {result.message}")

    keep_indices = [
        idx
        for idx, value in enumerate(result.x[:n_samples])
        if value >= 0.5
    ]
    candidate_samples = [samples[idx] for idx in keep_indices]
    candidate_excess, _ = compute_cap_excess(
        candidate_samples,
        active_fields,
        max_pct,
        field_max_pcts=normalized_field_max_pcts,
    )
    deficits, count_shortfalls = _evaluate_min_constraint_violations(
        candidate_samples,
        normalized_pct_constraints,
        normalized_count_constraints,
    )
    incumbent_feasible = (
        candidate_excess <= 1e-9
        and not deficits
        and not count_shortfalls
    )
    incumbent_has_mass = bool(keep_indices)

    if result.status == 1 and (not incumbent_has_mass or not incumbent_feasible):
        raise RuntimeError(
            f"{progress_desc}: MILP hit a solve limit before finding a feasible incumbent; increase --exact-time-limit or relax the constraints"
        )

    if result.status == 1:
        LOGGER.warning(
            "  %s: MILP hit a solve limit; returning the best feasible incumbent found (not certified optimal)",
            progress_desc,
        )

    if result.status == 0 and not incumbent_feasible:
        raise RuntimeError(
            f"{progress_desc}: exact MILP reported optimality but returned an infeasible incumbent"
        )

    return _finalize_kept_subset(
        samples,
        keep_indices,
        active_fields,
        max_pct,
        field_max_pcts=normalized_field_max_pcts,
        pct_constraints=normalized_pct_constraints,
        count_constraints=normalized_count_constraints,
        progress_desc=progress_desc,
    )

# ─── Frequency reporting ────────────────────────────────────────────────────

def report_frequencies(
    samples: List[Dict[str, Any]],
    title: str = "Dataset",
) -> None:
    """Print frequency tables for key fields."""
    n = len(samples)
    print(f"\n{'=' * 70}")
    print(f" {title} ({n} samples)")
    print(f"{'=' * 70}")

    # Single pass over all samples to collect every counter.
    action_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    domain_signature_counts: Counter = Counter()
    domain_pair_counts: Counter = Counter()
    domain_arity_counts: Counter = Counter()
    toolkit_marginal_counts: Counter = Counter()
    model_family_counts: Counter = Counter()
    scope_counts: Counter = Counter()
    toolkit_counts: Counter = Counter()
    name_counter: Counter = Counter()

    for s in samples:
        action_counts[s.get("trajectory", {}).get("final_action", "unknown")] += 1
        domains = _sample_domains(s)
        counted_domains = domains or ["unknown"]
        for domain in counted_domains:
            domain_counts[domain] += 1
        domain_signature_counts[build_domain_signature(counted_domains)] += 1
        for pair in itertools.combinations(counted_domains, 2):
            domain_pair_counts["|".join(pair)] += 1
        domain_arity_counts[str(len(domains)) if domains else "missing"] += 1
        toolkits = normalize_toolkit_list(s.get("trajectory", {}).get("toolkits", []))
        counted_toolkits = toolkits or ["unknown"]
        for toolkit in counted_toolkits:
            toolkit_marginal_counts[toolkit] += 1
        model_family_counts[compute_model_family(s)] += 1
        scope_counts[compute_subject_scope(s)] += 1
        toolkit_counts[compute_toolkit_signature(s)] += 1
        for name in extract_names_from_sample(s):
            name_counter[name] += 1

    # Final action frequencies
    print("\n  Final Action Frequencies:")
    print(f"  {'Action':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for action, count in action_counts.most_common(50):
        pct = 100.0 * count / n if n else 0
        print(f"  {action:<40} {count:>6} {pct:>5.1f}%")
    if len(action_counts) > 50:
        remaining = sum(c for _, c in action_counts.most_common()[50:])
        print(f"  {'(others)':<40} {remaining:>6}")

    # Domain frequencies
    print("\n  Domain Frequencies (top 50, raw counts):")
    print(f"  {'Domain':<40} {'Count':>8} {'%':>6}")
    print(f"  {'-' * 54}")
    for domain, count in domain_counts.most_common(50):
        pct = 100.0 * count / n if n else 0
        print(f"  {domain:<40} {int(count):>8} {pct:>5.1f}%")
    if len(domain_counts) > 50:
        remaining = sum(c for _, c in domain_counts.most_common()[50:])
        print(f"  {'(others)':<40} {int(remaining):>8}")
    print(f"  (total unique domains: {len(domain_counts)})")

    print("\n  Domain Signature Frequencies (top 25):")
    print(f"  {'Signature':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for signature, count in domain_signature_counts.most_common(25):
        pct = 100.0 * count / n if n else 0
        print(f"  {signature:<40} {count:>6} {pct:>5.1f}%")
    if len(domain_signature_counts) > 25:
        remaining = sum(c for _, c in domain_signature_counts.most_common()[25:])
        print(f"  {'(others)':<40} {remaining:>6}")
    print(f"  (total unique signatures: {len(domain_signature_counts)})")

    print("\n  Domain Pair Frequencies (top 25):")
    print(f"  {'Pair':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for pair, count in domain_pair_counts.most_common(25):
        pct = 100.0 * count / n if n else 0
        print(f"  {pair:<40} {count:>6} {pct:>5.1f}%")
    if len(domain_pair_counts) > 25:
        remaining = sum(c for _, c in domain_pair_counts.most_common()[25:])
        print(f"  {'(others)':<40} {remaining:>6}")
    print(f"  (total unique domain pairs: {len(domain_pair_counts)})")

    print("\n  Domains Per Sample:")
    print(f"  {'# Domains':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for arity, count in sorted(
        domain_arity_counts.items(),
        key=lambda item: (item[0] == "missing", int(item[0]) if item[0].isdigit() else 999),
    ):
        pct = 100.0 * count / n if n else 0
        print(f"  {arity:<40} {count:>6} {pct:>5.1f}%")

    # Model family frequencies
    print("\n  Model Family Frequencies:")
    print(f"  {'Family':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for family, count in model_family_counts.most_common():
        pct = 100.0 * count / n if n else 0
        print(f"  {family:<40} {count:>6} {pct:>5.1f}%")

    # Subject scope frequencies
    print("\n  Subject Scope Frequencies:")
    print(f"  {'Scope':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for scope, count in scope_counts.most_common():
        pct = 100.0 * count / n if n else 0
        print(f"  {scope:<40} {count:>6} {pct:>5.1f}%")

    # Toolkit marginal frequencies
    print("\n  Toolkit Frequencies (top 50, per-sample marginals):")
    print(f"  {'Toolkit':<40} {'Count':>8} {'%':>6}")
    print(f"  {'-' * 54}")
    for toolkit, count in toolkit_marginal_counts.most_common(50):
        pct = 100.0 * count / n if n else 0
        print(f"  {toolkit:<40} {int(count):>8} {pct:>5.1f}%")
    if len(toolkit_marginal_counts) > 50:
        remaining = sum(c for _, c in toolkit_marginal_counts.most_common()[50:])
        print(f"  {'(others)':<40} {int(remaining):>8}")
    print(f"  (total unique toolkits: {len(toolkit_marginal_counts)})")

    # Toolkit signature frequencies (top 50)
    print("\n  Toolkit Signature Frequencies (top 50):")
    print(f"  {'Signature':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 54}")
    for sig, count in toolkit_counts.most_common(50):
        pct = 100.0 * count / n if n else 0
        print(f"  {sig:<40} {count:>6} {pct:>5.1f}%")
    if len(toolkit_counts) > 50:
        remaining = sum(c for _, c in toolkit_counts.most_common()[50:])
        print(f"  {'(others)':<40} {remaining:>6}")
    print(f"  (total unique signatures: {len(toolkit_counts)})")

    # Character name frequencies (top 50)
    print("\n  Character Name Frequencies (top 50):")
    print(f"  {'Name':<40} {'Samples':>6}")
    print(f"  {'-' * 48}")
    for name, count in name_counter.most_common(50):
        eth = guess_ethnicity(name)
        print(f"  {name:<40} {count:>6}  ({eth})")

    print()


# ─── CLI and main ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-process generated benchmark data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python postprocess.py --input-path outputs/main_data_generated_gpt_oss_120b.json

  # Dry run (report only, no writes)
  python postprocess.py --input-path outputs/main_data_generated_gpt_oss_120b.json --dry-run

  # Only filter + report (no name replace or sampling)
  python postprocess.py --input-path outputs/data.json --skip-name-replace --skip-sampling
""",
    )

    io = p.add_argument_group("I/O")
    io.add_argument(
        "--input-path", required=True,
        help="Input JSON file.",
    )
    io.add_argument(
        "--output-path", default=None,
        help="Output JSON file (default: <input>_postprocessed.json).",
    )
    io.add_argument(
        "--rejected-path", default=None,
        help="JSONL log for rejected samples.",
    )

    filt = p.add_argument_group("Filtering")
    filt.add_argument(
        "--skip-healthcare-mismatch-filter",
        action="store_true",
        help=(
            "Skip rejecting non-healthcare samples whose trajectory contains "
            "clear healthcare tool-call or keyword signals."
        ),
    )
    filt.add_argument("--skip-link-filter", action="store_true")

    names = p.add_argument_group("Name replacement")
    names.add_argument("--skip-name-replace", action="store_true")
    names.add_argument(
        "--name-threshold", type=float, default=1.0,
        help="Replace names appearing in >= this percent of samples (default: 1.0).",
    )

    samp = p.add_argument_group("Diversity sampling")
    samp.add_argument("--skip-sampling", action="store_true")
    samp.add_argument(
        "--balance-fields", nargs="+",
        default=list(DEFAULT_CAP_BALANCE_FIELDS),
        help=(
            "Fields to balance via percentage-cap sampling. Minimum-share "
            "constraints for subject_scope and model_family are configured "
            "separately via --*-scope-pct, --min-model-family-pct, or "
            "--equalize-model-families and are not part of the default cap "
            "field set."
        ),
    )
    samp.add_argument(
        "--max-pct", type=float, default=5.0,
        help=(
            "Percentage used to derive per-category caps during each sampling "
            "stage (default: 5.0)."
        ),
    )
    samp.add_argument(
        "--domain-signature-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap percentage for "
            "'generation_metadata.domain_signature'. When set, "
            "domain signatures use this value instead of --max-pct."
        ),
    )
    samp.add_argument(
        "--toolkit-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap percentage for 'trajectory.toolkits'. "
            "When set, per-toolkit marginals use this value instead of "
            "--max-pct and 'trajectory.toolkits' is automatically added to "
            "--balance-fields."
        ),
    )
    samp.add_argument(
        "--self-scope-pct", type=float, default=None,
        help=(
            "Minimum percentage for 'self' subject_scope samples. Downsamples "
            "non-self categories (third_party, multi_subject) so that 'self' "
            "makes up at least this share of the final dataset. When exact "
            "model-family equalization is enabled, both constraints are "
            "optimized jointly to maximize retention. Skipped if not set. "
            "Example: --self-scope-pct 50"
        ),
    )
    samp.add_argument(
        "--third-party-scope-pct",
        "--other-scope-pct",
        dest="third_party_scope_pct",
        type=float,
        default=None,
        help=(
            "Minimum percentage for 'third_party' subject_scope samples. "
            "Useful when self-scope trimming would otherwise push third-party "
            "coverage too low. Example: --third-party-scope-pct 30"
        ),
    )
    samp.add_argument(
        "--multi-subject-scope-pct",
        dest="multi_subject_scope_pct",
        type=float,
        default=None,
        help=(
            "Minimum percentage for 'multi_subject' subject_scope samples. "
            "Useful when self-scope or third-party trimming would otherwise "
            "push multi-subject coverage too low. Example: "
            "--multi-subject-scope-pct 15"
        ),
    )
    samp.add_argument(
        "--equalize-model-families",
        action="store_true",
        help=(
            "Downsample requested model families to equal counts using the "
            "smallest present family size. Runs after generic diversity "
            "sampling so the final dataset keeps as much data as possible."
        ),
    )
    samp.add_argument(
        "--min-model-family-pct",
        type=float,
        default=None,
        help=(
            "Softer alternative to exact equalization. Downsample only enough "
            "so each requested model family occupies at least this percentage "
            "of the final dataset (for example 30.0). When sampling is "
            "enabled, this minimum is enforced jointly inside Stage 4's "
            "diversity optimizer."
        ),
    )
    samp.add_argument(
        "--model-families",
        nargs="+",
        default=None,
        help=(
            "Model families to equalize when --equalize-model-families or "
            "--min-model-family-pct is set. When omitted, postprocess "
            "infers the primary families present in the data (preferring "
            "qwen/gpt/nvidia)."
        ),
    )
    samp.add_argument(
        "--exact-time-limit",
        type=float,
        default=None,
        help=(
            "Optional wall-clock limit in seconds for exact diversity "
            "sampling. If reached after a feasible incumbent is found, "
            "postprocess returns the best incumbent found so far; otherwise "
            "the solve fails."
        ),
    )
    samp.add_argument(
        "--exact-mip-rel-gap",
        type=float,
        default=None,
        help=(
            "Optional relative MIP gap target for exact diversity sampling. "
            "Lower is stricter; omitted means SciPy/HiGHS default."
        ),
    )

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42, help="Random seed.")
    misc.add_argument("--dry-run", action="store_true",
                      help="Report statistics without writing output.")
    misc.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    rng = random.Random(args.seed)

    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        stem = input_path.stem + "_postprocessed"
        output_path = input_path.parent / (stem + input_path.suffix)
    rejected_path = Path(args.rejected_path) if args.rejected_path else (
        input_path.parent / "postprocess_rejected.jsonl"
    )

    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as fh:
        samples: List[Dict[str, Any]] = json.load(fh)

    LOGGER.info("Loaded %d samples from %s", len(samples), input_path)

    # ── Stage 0: Domain normalization ──
    LOGGER.info("Stage 0: Domain normalization")
    relabelled = normalize_domains(samples)
    LOGGER.info("  Relabelled %d samples", relabelled)

    # ── Stage 0.5: Healthcare domain mismatch filter ──
    if not args.skip_healthcare_mismatch_filter:
        LOGGER.info("Stage 0.5: Healthcare domain mismatch filter")
        accepted, rejected, rejection_rows = filter_healthcare_domain_mismatches(samples)
        if rejected:
            LOGGER.info(
                "  Rejected %d non-healthcare samples with healthcare-sensitive signals",
                len(rejected),
            )
            if not args.dry_run:
                for row in rejection_rows:
                    append_jsonl(rejected_path, row)
        else:
            LOGGER.info("  No healthcare domain mismatches found")
        samples = accepted

    # ── Report BEFORE ──
    report_frequencies(samples, title="BEFORE post-processing")

    # ── Stage 1: Link passthrough filter ──
    if not args.skip_link_filter:
        LOGGER.info("Stage 1: Link-passthrough filter")
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for s in samples:
            if check_link_passthrough(s):
                rejected.append(s)
            else:
                accepted.append(s)

        if rejected:
            LOGGER.info("  Rejected %d link-passthrough samples", len(rejected))
            if not args.dry_run:
                for r in rejected:
                    append_jsonl(rejected_path, {
                        "name": r.get("name", "?"),
                        "stage": "link_passthrough",
                    })
        else:
            LOGGER.info("  No link-passthrough samples found")
        samples = accepted

    # ── Stage 2: Name replacement ──
    if not args.skip_name_replace:
        name_counter: Counter = Counter()
        for s in samples:
            for name in extract_names_from_sample(s):
                name_counter[name] += 1

        n_samples = len(samples)
        name_abs_threshold = compute_name_abs_threshold(n_samples, args.name_threshold)
        LOGGER.info(
            "Stage 2: Name replacement (threshold=%.1f%% -> %d samples)",
            args.name_threshold, name_abs_threshold,
        )

        overrep = {
            name: count for name, count in name_counter.items()
            if count >= name_abs_threshold
        }

        # Always replace placeholder names regardless of frequency.
        for placeholder in _PLACEHOLDER_NAMES:
            if placeholder not in overrep:
                overrep[placeholder] = name_counter.get(placeholder, 0)

        replacer = NameReplacer(rng)
        if overrep:
            LOGGER.info(
                "  Found %d overrepresented names (>= %d samples / %.1f%%):",
                len(overrep), name_abs_threshold, args.name_threshold,
            )
            for name, count in sorted(overrep.items(), key=lambda x: -x[1])[:50]:
                eth = guess_ethnicity(name)
                gen = guess_gender(name)
                LOGGER.info("    %-30s %3d samples  (%s, %s)", name, count, eth, gen)

            replaced_count = 0

            for s in samples:
                names_to_replace = extract_names_from_sample(s) & set(overrep.keys())

                # Catch placeholder names even when they only appear in
                # free text (e.g. generated_final_action) rather than in
                # structured name fields.
                sample_text = json.dumps(s)
                names_to_replace.update(find_placeholder_names_in_text(sample_text))

                if replace_names_in_sample(s, names_to_replace, replacer):
                    replaced_count += 1

            LOGGER.info("  Replaced names in %d samples", replaced_count)
        else:
            LOGGER.info("  No names above threshold — skipping")

        forced_last_name_plan = plan_forced_last_name_replacements(
            samples,
            name_abs_threshold,
            rng,
        )
        if forced_last_name_plan:
            forced_counts: Counter[str] = Counter()
            for names in forced_last_name_plan.values():
                for name in names:
                    _, last_name = split_name(name)
                    if last_name:
                        forced_counts[last_name] += 1
            LOGGER.info(
                "  Replacing excess forced-surname samples (keeping up to %d samples per surname): %s",
                name_abs_threshold,
                ", ".join(
                    f"{last_name}={count}"
                    for last_name, count in sorted(forced_counts.items())
                ),
            )
            replaced_count = 0
            for idx, names_to_replace in sorted(forced_last_name_plan.items()):
                if replace_names_in_sample(samples[idx], names_to_replace, replacer):
                    replaced_count += 1
            LOGGER.info("  Replaced forced-surname names in %d samples", replaced_count)

    # ── Stage 2b: Materialize computed diversity fields ──
    LOGGER.info("Stage 2b: Materializing computed diversity fields")
    materialize_computed_fields(samples)
    resolved_model_families = resolve_model_families(samples, args.model_families)
    balance_fields = resolve_balance_fields(args)
    if resolved_model_families:
        if args.model_families:
            LOGGER.info(
                "  Using requested model families for balancing: %s",
                ", ".join(resolved_model_families),
            )
        else:
            LOGGER.info(
                "  Inferred model families for balancing: %s",
                ", ".join(resolved_model_families),
            )
    elif args.equalize_model_families or args.min_model_family_pct is not None:
        LOGGER.warning("  No model families available for balancing")

    scope_min_pcts: Dict[str, float] = {}
    if args.self_scope_pct is not None:
        scope_min_pcts["self"] = args.self_scope_pct
    if args.third_party_scope_pct is not None:
        scope_min_pcts["third_party"] = args.third_party_scope_pct
    if args.multi_subject_scope_pct is not None:
        scope_min_pcts["multi_subject"] = args.multi_subject_scope_pct

    joint_family_scope_balance = (
        args.self_scope_pct is not None
        and args.equalize_model_families
        and args.min_model_family_pct is None
        and args.third_party_scope_pct is None
        and args.multi_subject_scope_pct is None
    )
    field_cap_pcts = build_field_max_pcts(args)
    stage4_min_pct_constraints = build_min_pct_constraints(
        samples,
        scope_min_pcts,
        resolved_model_families,
        args.min_model_family_pct,
    )
    stage4_integrates_min_pct_constraints = (
        not args.skip_sampling
        and not joint_family_scope_balance
        and bool(stage4_min_pct_constraints)
    )

    # ── Stage 3: Subject scope minimum fallback when not integrated into Stage 4 ──
    if scope_min_pcts:
        if joint_family_scope_balance:
            LOGGER.info(
                "Stage 3: Deferring self-scope minimum (%.1f%%) to Stage 5 for joint optimization",
                args.self_scope_pct,
            )
        elif stage4_integrates_min_pct_constraints:
            pass
        else:
            LOGGER.info(
                "Stage 3: Subject scope minimums (%s)",
                ", ".join(f"{scope}={pct:.1f}%%" for scope, pct in sorted(scope_min_pcts.items())),
            )
            before = len(samples)
            samples = enforce_min_category_pcts(
                samples,
                "generation_metadata.subject_scope",
                scope_min_pcts,
                rng,
            )
            LOGGER.info("  %d -> %d samples", before, len(samples))

    sampling_preferred_families = (
        resolved_model_families
        if (args.equalize_model_families or args.min_model_family_pct is not None)
        else None
    )

    # ── Stage 4: Global one-by-one diversity sampling ──
    if not args.skip_sampling:
        fields, skipped_fields = select_enforceable_balance_fields(
            samples,
            balance_fields,
            args.max_pct,
            field_max_pcts=field_cap_pcts,
        )
        if skipped_fields:
            LOGGER.info(
                "Stage 4: Skipping mathematically infeasible balance fields: %s",
                ", ".join(skipped_fields),
            )
        if not fields:
            LOGGER.info("Stage 4: Diversity sampling skipped; no enforceable balance fields remain")
        else:
            if stage4_min_pct_constraints:
                LOGGER.info(
                    "Stage 4: Exact diversity sampling with integrated minimum-share constraints (%s, %d enforceable fields)",
                    format_field_cap_pcts(fields, args.max_pct, field_cap_pcts),
                    len(fields),
                )
                LOGGER.info(
                    "  Integrated minimum-share constraints: %s",
                    format_min_pct_constraints(stage4_min_pct_constraints),
                )
            else:
                LOGGER.info(
                    "Stage 4: Exact diversity sampling (%s, %d enforceable fields)",
                    format_field_cap_pcts(fields, args.max_pct, field_cap_pcts),
                    len(fields),
                )
            before = len(samples)
            samples = tighten_diversity_caps(
                samples,
                fields,
                args.max_pct,
                rng=rng,
                preferred_families=sampling_preferred_families,
                field_max_pcts=field_cap_pcts,
                protected_pct_constraints=stage4_min_pct_constraints,
                progress_desc="Stage 4 cap tightening",
                time_limit=args.exact_time_limit,
                mip_rel_gap=args.exact_mip_rel_gap,
            )
            LOGGER.info("  %d -> %d samples", before, len(samples))
            projected_final_size = project_model_family_balanced_size(
                samples,
                resolved_model_families,
                equalize=args.equalize_model_families and args.min_model_family_pct is None,
                min_pct=args.min_model_family_pct,
                self_pct=args.self_scope_pct if joint_family_scope_balance else None,
            )
            if projected_final_size != len(samples):
                LOGGER.info(
                    "  Stage 4 projected final size after downstream model-family balancing: %d",
                    projected_final_size,
                )

    # ── Stage 5: Model-family balancing ──
    if args.min_model_family_pct is not None:
        if stage4_integrates_min_pct_constraints:
            LOGGER.info(
                "Stage 5: Model-family minimum verification (%s, min_pct=%.2f%%)",
                ", ".join(resolved_model_families),
                args.min_model_family_pct,
            )
        else:
            LOGGER.info(
                "Stage 5: Model-family min-pct balancing (%s, min_pct=%.2f%%)",
                ", ".join(resolved_model_families),
                args.min_model_family_pct,
            )
        projected_size = project_model_family_balanced_size(
            samples,
            resolved_model_families,
            min_pct=args.min_model_family_pct,
        )
        if stage4_integrates_min_pct_constraints and projected_size >= len(samples):
            LOGGER.info("  Already satisfied after Stage 4 joint optimization — no trimming needed")
        else:
            before = len(samples)
            samples = enforce_min_model_family_pct(
                samples,
                resolved_model_families,
                args.min_model_family_pct,
                rng,
            )
            LOGGER.info("  %d -> %d samples", before, len(samples))
    elif args.equalize_model_families:
        if joint_family_scope_balance:
            LOGGER.info(
                "Stage 5: Joint model-family/self-scope balancing (%s, min self=%.1f%%)",
                ", ".join(resolved_model_families),
                args.self_scope_pct,
            )
        else:
            LOGGER.info(
                "Stage 5: Model-family balancing (%s)",
                ", ".join(resolved_model_families),
            )
        before = len(samples)
        if joint_family_scope_balance:
            samples = equalize_model_families_with_self_scope(
                samples,
                resolved_model_families,
                args.self_scope_pct,
                rng,
            )
        else:
            samples = equalize_model_families(samples, resolved_model_families, rng)
        LOGGER.info("  %d -> %d samples", before, len(samples))

    # ── Stage 6: Re-assert subject scope minimums after downstream trimming ──
    if scope_min_pcts and not joint_family_scope_balance:
        if stage4_integrates_min_pct_constraints:
            LOGGER.info(
                "Stage 6: Final subject scope minimum verification (%s)",
                ", ".join(f"{scope}={pct:.1f}%%" for scope, pct in sorted(scope_min_pcts.items())),
            )
        else:
            LOGGER.info(
                "Stage 6: Final subject scope minimums (%s)",
                ", ".join(f"{scope}={pct:.1f}%%" for scope, pct in sorted(scope_min_pcts.items())),
            )
        active_scope_targets = []
        for scope, pct in scope_min_pcts.items():
            count = sum(1 for sample in samples if compute_subject_scope(sample) == scope)
            if count > 0:
                active_scope_targets.append(math.floor(count * 100.0 / pct))
        projected_scope_size = (
            min(len(samples), min(active_scope_targets))
            if active_scope_targets
            else len(samples)
        )
        if stage4_integrates_min_pct_constraints and projected_scope_size >= len(samples):
            LOGGER.info("  Already satisfied after joint optimization — no trimming needed")
        else:
            before = len(samples)
            samples = enforce_min_category_pcts(
                samples,
                "generation_metadata.subject_scope",
                scope_min_pcts,
                rng,
            )
            LOGGER.info("  %d -> %d samples", before, len(samples))

    # ── Stage 7: Final cap tightening on balance fields ──
    if not args.skip_sampling and balance_fields:
        final_min_pct_constraints = build_min_pct_constraints(
            samples,
            scope_min_pcts,
            resolved_model_families,
            args.min_model_family_pct,
        )

        protected_count_constraints: Dict[Tuple[str, str], int] = {}
        if args.equalize_model_families and args.min_model_family_pct is None:
            family_counts = Counter(
                compute_model_family(sample)
                for sample in samples
            )
            for family in resolved_model_families:
                if family_counts[family] > 0:
                    protected_count_constraints[("generation_metadata.model_family", family)] = family_counts[family]

        LOGGER.info(
            "Stage 7: Final exact diversity cap tightening (%s)",
            format_field_cap_pcts(balance_fields, args.max_pct, field_cap_pcts),
        )
        final_cap_fields, skipped_fields = select_enforceable_balance_fields(
            samples,
            balance_fields,
            args.max_pct,
            field_max_pcts=field_cap_pcts,
        )
        if skipped_fields:
            LOGGER.info(
                "  Stage 7: Skipping mathematically infeasible balance fields: %s",
                ", ".join(skipped_fields),
            )
        if not final_cap_fields:
            LOGGER.info("  Stage 7: No enforceable balance fields remain")
        else:
            before = len(samples)
            samples = tighten_diversity_caps(
                samples,
                final_cap_fields,
                args.max_pct,
                rng=rng,
                preferred_families=sampling_preferred_families,
                field_max_pcts=field_cap_pcts,
                protected_pct_constraints=final_min_pct_constraints,
                protected_count_constraints=protected_count_constraints,
                progress_desc="Stage 7 final tightening",
                time_limit=args.exact_time_limit,
                mip_rel_gap=args.exact_mip_rel_gap,
            )
            LOGGER.info("  %d -> %d samples", before, len(samples))

    # ── Report AFTER ──
    report_frequencies(samples, title="AFTER post-processing")

    # ── Write output ──
    if args.dry_run:
        LOGGER.info("Dry run — not writing output")
    else:
        write_json_atomic(output_path, samples)
        LOGGER.info("Wrote %d samples to %s", len(samples), output_path)


if __name__ == "__main__":
    main()
