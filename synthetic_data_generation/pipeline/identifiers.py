"""Programmatic identifier generation and replacement.

LLMs frequently default to lazy identifiers like 123456789 or 987654321.
This module detects those patterns in generated text and replaces them with
unique, random values of the same format, keeping a registry to ensure
cross-sample uniqueness and intra-sample consistency.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Set, Tuple

LOGGER = logging.getLogger("scenario_generator")

# ---------------------------------------------------------------------------
# Known-lazy sequences that models love to generate
# ---------------------------------------------------------------------------
_LAZY_DIGIT_SEQUENCES: Set[str] = {
    "123456789", "987654321", "1234567890", "9876543210",
    "111111111", "222222222", "000000000",
    "111111111111", "123456789012",
}

# ---------------------------------------------------------------------------
# Regex patterns for identifier types, ordered by specificity
# ---------------------------------------------------------------------------
# Each entry: (name, compiled regex, generator function name)
# The regex must capture the FULL identifier including delimiters.

# SSN: 3-2-4 with dashes
_SSN_RE = re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")
# SIN (Canadian): 3-3-3 with dashes
_SIN_RE = re.compile(r"\b(\d{3}-\d{3}-\d{3})\b")
# Phone: various formats — (555) 219-4839, 604-555-0198, +1-416-555-0198, +353 86 987 6543
_PHONE_PAREN_RE = re.compile(r"\((\d{3})\)\s*(\d{3})-(\d{4})")
_PHONE_DASH_RE = re.compile(r"\b(\d{3})-(\d{3})-(\d{4})\b")
# Generic long digit sequence (account numbers, policy numbers, etc.) — 6+ digits.
# Also matches digits preceded by a letter prefix (e.g., H123456789).
_LONG_DIGITS_RE = re.compile(r"(?<![0-9])(\d{6,})(?![0-9])")


def _random_digits(n: int) -> str:
    """Generate n random digits, first digit always nonzero."""
    first = str(random.randint(1, 9))
    rest = "".join(str(random.randint(0, 9)) for _ in range(n - 1))
    return first + rest


def _generate_ssn() -> str:
    """Generate a random SSN-formatted string (not a valid SSN)."""
    # Avoid real SSN area numbers starting with 9xx (ITIN) or 000/666
    area = random.randint(1, 899)
    while area in (0, 666):
        area = random.randint(1, 899)
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def _generate_sin() -> str:
    """Generate a random SIN-formatted string."""
    return f"{_random_digits(3)}-{_random_digits(3)}-{_random_digits(3)}"


def _generate_phone_paren() -> str:
    """Generate (XXX) XXX-XXXX phone number."""
    return f"({_random_digits(3)}) {_random_digits(3)}-{_random_digits(4)}"


def _generate_phone_dash() -> str:
    """Generate XXX-XXX-XXXX phone number."""
    return f"{_random_digits(3)}-{_random_digits(3)}-{_random_digits(4)}"


def _generate_long_digits(length: int) -> str:
    """Generate a random digit string of the given length."""
    return _random_digits(length)


class IdentifierRegistry:
    """Tracks used identifiers across the entire generation run to ensure uniqueness."""

    def __init__(self) -> None:
        # All identifiers that have been assigned (stripped of formatting)
        self._used_raw: Set[str] = set()
        # Per-sample replacement map: old -> new (for consistent propagation)
        self._current_replacements: Dict[str, str] = {}
        # Raw digits first seen in the *current* sample — these are legitimate
        # within-sample reuses (e.g. same SSN in story and sensitive_info_items)
        # and must not trigger replacement.
        self._current_sample_raws: Set[str] = set()

    def _apply_replacements_single_pass(self, text: str) -> str:
        """Apply all current replacements in a single pass to avoid cascading corruption.

        Iterating str.replace(old, new) can corrupt text when a replacement's
        output contains a substring matching a later replacement key.  This
        method builds a regex alternation (longest keys first) so every
        replacement is applied exactly once without interference.
        """
        if not self._current_replacements:
            return text
        sorted_keys = sorted(self._current_replacements, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
        return pattern.sub(lambda m: self._current_replacements[m.group()], text)

    def _unique(self, generator, *args, max_attempts: int = 50) -> str:
        """Call generator until we get a value not yet used."""
        for _ in range(max_attempts):
            value = generator(*args)
            raw = re.sub(r"[^0-9]", "", value)
            if raw not in self._used_raw and raw not in _LAZY_DIGIT_SEQUENCES:
                self._used_raw.add(raw)
                return value
        # Fallback: accept the last generated value
        return value  # noqa: F821 — always assigned in loop

    def _is_lazy(self, raw_digits: str) -> bool:
        """Check if a digit string is a known-lazy sequence."""
        return raw_digits in _LAZY_DIGIT_SEQUENCES

    def begin_sample(self) -> None:
        """Start a new sample — clears the per-sample replacement map."""
        self._current_replacements = {}
        self._current_sample_raws = set()

    @property
    def replacements(self) -> Dict[str, str]:
        """Current sample's old->new replacement map."""
        return dict(self._current_replacements)

    def restore_replacements(self, replacements: Dict[str, str]) -> None:
        """Restore a previously saved replacement map (e.g. from a cached sample)."""
        self._current_replacements.update(replacements)

    def fix_identifiers(self, text: str) -> str:
        """Scan text for lazy/reused identifiers and replace them.

        Returns the fixed text. Builds up self._current_replacements so the
        same mapping can be applied to other fields (trajectory, memories).
        """
        # Apply any already-known replacements first (for consistency within a sample)
        text = self._apply_replacements_single_pass(text)

        # Raw digits of values we just substituted in — these should not be
        # re-detected as "reused" identifiers during the scan below.
        _replacement_value_raws: Set[str] = {
            re.sub(r"[^0-9]", "", v) for v in self._current_replacements.values()
        }

        # --- SSN (3-2-4) ---
        for match in _SSN_RE.finditer(text):
            old = match.group(1)
            raw = re.sub(r"[^0-9]", "", old)
            if raw in _replacement_value_raws:
                continue
            if self._is_lazy(raw) or (raw in self._used_raw and raw not in self._current_sample_raws):
                if old not in self._current_replacements:
                    new = self._unique(_generate_ssn)
                    self._current_replacements[old] = new
            else:
                self._used_raw.add(raw)
                self._current_sample_raws.add(raw)

        # --- SIN (3-3-3) --- only match if not already caught by SSN
        for match in _SIN_RE.finditer(text):
            old = match.group(1)
            if old in self._current_replacements:
                continue
            raw = re.sub(r"[^0-9]", "", old)
            if raw in _replacement_value_raws:
                continue
            if self._is_lazy(raw) or (raw in self._used_raw and raw not in self._current_sample_raws):
                if old not in self._current_replacements:
                    new = self._unique(_generate_sin)
                    self._current_replacements[old] = new
            else:
                self._used_raw.add(raw)
                self._current_sample_raws.add(raw)

        # --- Phone (parenthesized) ---
        for match in _PHONE_PAREN_RE.finditer(text):
            old = match.group(0)
            raw = re.sub(r"[^0-9]", "", old)
            if raw in _replacement_value_raws:
                continue
            if self._is_lazy(raw) or (raw in self._used_raw and raw not in self._current_sample_raws):
                if old not in self._current_replacements:
                    new = self._unique(_generate_phone_paren)
                    self._current_replacements[old] = new
            else:
                self._used_raw.add(raw)
                self._current_sample_raws.add(raw)

        # --- Phone (dashed, 10-digit) — skip if already caught by SSN/SIN ---
        for match in _PHONE_DASH_RE.finditer(text):
            old = match.group(0)
            if old in self._current_replacements:
                continue
            raw = re.sub(r"[^0-9]", "", old)
            if raw in _replacement_value_raws:
                continue
            if len(raw) == 10 and (self._is_lazy(raw) or (raw in self._used_raw and raw not in self._current_sample_raws)):
                if old not in self._current_replacements:
                    new = self._unique(_generate_phone_dash)
                    self._current_replacements[old] = new
            elif len(raw) == 10:
                self._used_raw.add(raw)
                self._current_sample_raws.add(raw)

        # --- Long digit sequences (account numbers, policy numbers, etc.) ---
        for match in _LONG_DIGITS_RE.finditer(text):
            old = match.group(1)
            if old in self._current_replacements:
                continue
            if old in _replacement_value_raws:
                continue
            if self._is_lazy(old) or (old in self._used_raw and old not in self._current_sample_raws):
                if old not in self._current_replacements:
                    new = self._unique(_generate_long_digits, len(old))
                    self._current_replacements[old] = new
            else:
                self._used_raw.add(old)
                self._current_sample_raws.add(old)

        # Apply all replacements (single-pass to avoid cascading corruption)
        text = self._apply_replacements_single_pass(text)

        return text

    def apply_replacements(self, text: str) -> str:
        """Apply the current sample's replacement map to a text field.

        Use this for trajectory and memories after fix_identifiers() has
        been called on the scenario fields.
        """
        return self._apply_replacements_single_pass(text)

    def fix_string_list(self, items: List[str]) -> List[str]:
        """Apply fix_identifiers to each string in a list."""
        return [self.fix_identifiers(item) for item in items]

    def apply_replacements_to_list(self, items: List[str]) -> List[str]:
        """Apply existing replacements to each string in a list."""
        return [self.apply_replacements(item) for item in items]


# ---------------------------------------------------------------------------
# Identifier reconciliation: fix trajectory values that diverge from the
# canonical identifiers in sensitive_info_items
# ---------------------------------------------------------------------------

# Broad regex to extract identifier-like values from free text.
# Covers international phones, SSNs, alphanumeric IDs, IBANs, long digit runs.
_IDENTIFIER_RE = re.compile(
    r"(?:"
    # International phone: +44 7700 123456, +1-213-555-0198, +353 86 987 6543
    r"\+\d[\d\s\-\(\)]{6,}"
    r"|"
    # IBAN-style: 2 uppercase letters + 2 digits + 10-30 alphanumeric
    r"[A-Z]{2}\d{2}[A-Z0-9]{10,}"
    r"|"
    # Alphanumeric ID with letter prefix: MRN-320678139, BCBSA-4523987123, AET-9384739
    r"[A-Z][A-Z0-9]*-\d{4,}[\d\-]*"
    r"|"
    # SSN/SIN: 3-2-4 or 3-3-3 with dashes
    r"\d{3}-\d{2,3}-\d{3,4}"
    r"|"
    # Parenthesized phone: (555) 219-4839
    r"\(\d{3}\)\s*\d{3}-\d{4}"
    r"|"
    # Generic long digit run (6+ digits)
    r"(?<![0-9A-Za-z])\d{6,}(?![0-9])"
    r")"
)


def _build_format_pattern(identifier: str) -> re.Pattern:
    """Build a regex that matches the same format but with different digits.

    Non-digit characters are kept as literals; each contiguous run of digits
    is replaced with ``\\d{N}`` where N is the run length.
    """
    parts: List[str] = []
    i = 0
    while i < len(identifier):
        if identifier[i].isdigit():
            j = i
            while j < len(identifier) and identifier[j].isdigit():
                j += 1
            parts.append(rf"\d{{{j - i}}}")
            i = j
        else:
            parts.append(re.escape(identifier[i]))
            i += 1
    return re.compile("".join(parts))


def _extract_context_words(item_text: str, identifier: str, window: int = 5) -> List[str]:
    """Extract context words surrounding an identifier within its canonical item sentence.

    Returns lowercased words from the item text excluding the identifier itself,
    useful for disambiguating which trajectory occurrence corresponds to which
    canonical item when multiple identifiers share the same format.
    """
    # Remove the identifier from the text to get surrounding context
    context = item_text.replace(identifier, " ")
    words = re.findall(r"[a-zA-Z]{3,}", context)
    return [w.lower() for w in words[:window]]


def _context_score(context_words: List[str], text: str, match_pos: int, radius: int = 300) -> int:
    """Score how many context words appear near a match position in the text."""
    start = max(0, match_pos - radius)
    end = min(len(text), match_pos + radius)
    window = text[start:end].lower()
    return sum(1 for w in context_words if w in window)


def reconcile_identifiers(canonical_items: List[str], text: str) -> str:
    """Replace mismatched identifiers in *text* with canonical values.

    Uses a two-pass approach:
    1. Collect all canonical identifiers that are missing from the text and
       find all candidate replacement positions.
    2. Assign canonical→candidate matches using context-aware scoring to
       handle multiple same-format identifiers correctly.
    """
    # Collect all canonical digit strings so we never overwrite one canonical
    # value with another (e.g. two phone numbers with the same format).
    all_canonical_vals: Set[str] = set()
    all_canonical_digits: Set[str] = set()
    for item in canonical_items:
        for match in _IDENTIFIER_RE.finditer(item):
            val = match.group().rstrip(" ,;.)")
            all_canonical_vals.add(val)
            digits = re.sub(r"[^0-9]", "", val)
            if len(digits) >= 4:
                all_canonical_digits.add(digits)

    # --- Pass 1: collect missing canonical identifiers and their candidates ---
    # Each entry: (canonical_val, item_text, format_regex, context_words)
    missing: List[Tuple[str, str, re.Pattern, List[str]]] = []
    for item in canonical_items:
        for match in _IDENTIFIER_RE.finditer(item):
            canonical_val = match.group().rstrip(" ,;.)")
            canonical_digits = re.sub(r"[^0-9]", "", canonical_val)
            if len(canonical_digits) < 4:
                continue
            # Already present — nothing to fix
            if canonical_val in text:
                continue
            if canonical_digits in re.sub(r"[^0-9]", "", text):
                continue
            fmt_re = _build_format_pattern(canonical_val)
            context_words = _extract_context_words(item, canonical_val)
            missing.append((canonical_val, item, fmt_re, context_words))

    if not missing:
        return text

    # --- Pass 2: for each missing canonical, find all candidate matches ---
    # Build a list of (canonical_val, old_val, match_pos, context_score)
    # then greedily assign best matches, ensuring each text occurrence is
    # used at most once.
    assignments: List[Tuple[str, str, int, int]] = []  # (canonical, old, pos, score)
    for canonical_val, item, fmt_re, context_words in missing:
        for fmt_match in fmt_re.finditer(text):
            old_val = fmt_match.group()
            if old_val == canonical_val:
                continue
            # Don't replace a value that is itself canonical
            if old_val in all_canonical_vals:
                continue
            old_digits = re.sub(r"[^0-9]", "", old_val)
            if old_digits in all_canonical_digits:
                continue
            score = _context_score(context_words, text, fmt_match.start())
            assignments.append((canonical_val, old_val, fmt_match.start(), score))

    # Greedily assign: sort by context score descending, then commit
    # replacements ensuring each canonical value and each text position
    # are used at most once.
    assignments.sort(key=lambda x: x[3], reverse=True)
    used_canonicals: Set[str] = set()
    used_positions: Set[int] = set()
    replacements: List[Tuple[str, str]] = []  # (old_val, canonical_val)

    for canonical_val, old_val, pos, score in assignments:
        if canonical_val in used_canonicals:
            continue
        if pos in used_positions:
            continue
        used_canonicals.add(canonical_val)
        used_positions.add(pos)
        replacements.append((old_val, canonical_val))

    # Apply replacements — use single-pass to avoid cascading
    if replacements:
        sorted_repls = sorted(replacements, key=lambda x: len(x[0]), reverse=True)
        pattern = re.compile("|".join(re.escape(old) for old, _ in sorted_repls))
        repl_map = {old: new for old, new in sorted_repls}
        text = pattern.sub(lambda m: repl_map[m.group()], text)
        for old_val, canonical_val in replacements:
            LOGGER.debug("Reconciled identifier: %r -> %r", old_val, canonical_val)

    return text
