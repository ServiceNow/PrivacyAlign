"""Shared helpers for canonicalizing domain labels during generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pipeline.utils import normalize_string_list


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ALIASES_PATH = SCRIPT_DIR / "resources" / "domain_aliases.json"


def _load_domain_aliases() -> Dict[str, str]:
    with ALIASES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["aliases"]


_DOMAIN_ALIASES = _load_domain_aliases()
_DOMAIN_SEPARATOR_RE = re.compile(r"[\u2010-\u2015\u2212\u2011_-]+")
_DOMAIN_SPACE_RE = re.compile(r"\s+")


def canonicalize_domain_label(label: str) -> str:
    """Normalize punctuation and whitespace for a single domain label."""
    label = str(label or "").strip().lower()
    label = _DOMAIN_SEPARATOR_RE.sub(" ", label)
    label = _DOMAIN_SPACE_RE.sub(" ", label)
    return label.strip()


_NORMALIZED_DOMAIN_ALIASES: Dict[str, str] = {
    canonicalize_domain_label(key): canonicalize_domain_label(value)
    for key, value in _DOMAIN_ALIASES.items()
}


def normalize_domain_list(value: Any) -> List[str]:
    """Return a canonicalized, deduplicated domain list.

    Alias resolution happens before deduplication, so values like
    ``["human_resources", "hr"]`` collapse to ``["hr"]``.
    """
    normalized: List[str] = []
    seen = set()
    for item in normalize_string_list(value):
        canonical = _NORMALIZED_DOMAIN_ALIASES.get(
            canonicalize_domain_label(item),
            canonicalize_domain_label(item),
        )
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return sorted(normalized)


def domain_signature(value: Any) -> str:
    """Return an order-invariant domain signature for a domain value."""
    domains = normalize_domain_list(value)
    if not domains:
        return "unknown"
    return "|".join(domains)
