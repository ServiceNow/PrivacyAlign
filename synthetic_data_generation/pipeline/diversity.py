"""DiversityTracker: holds diversity counters for generation."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.domains import domain_signature, normalize_domain_list
from pipeline.utils import NameRow, normalize_text

LOGGER = logging.getLogger("scenario_generator")


class DiversityTracker:
    """Tracks diversity counters for name, domain, toolkit signature, etc."""

    def __init__(self) -> None:
        self.name_counts: Counter[str] = Counter()
        self.domain_counts: Counter[str] = Counter()
        self.domain_signature_counts: Counter[str] = Counter()
        self.toolkit_signature_counts: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.subject_name_counts: Counter[str] = Counter()
        self.recipient_name_counts: Counter[str] = Counter()
        self.subject_scope_counts: Counter[str] = Counter()
        self.toolkit_count_counts: Counter[str] = Counter()
        self.sensitive_item_count_counts: Counter[str] = Counter()
        # Profile-level counters (warmed from generation_metadata on resume)
        self.sex_counts: Counter[str] = Counter()
        self.ethnicity_counts: Counter[str] = Counter()
        self.religion_counts: Counter[str] = Counter()
        self.citizenship_counts: Counter[str] = Counter()
        self.occupation_counts: Counter[str] = Counter()

    @staticmethod
    def _extract_subject_name(data_subject: str) -> str:
        """Extract just the name(s) from a data_subject string, stripping roles.

        Handles formats like:
          "Marcus Thorne, Senior Software Engineer at Novak Industries"
          "Eleanor Whitfield, homeowner and James Whitfield, homeowner"
        Returns lowercased name(s) joined by " and ".
        """
        raw = normalize_text(data_subject).lower()
        if not raw:
            return ""
        # Split on " and " for multi-subject, extract name before first comma
        parts = raw.split(" and ")
        names = []
        for part in parts:
            part = part.strip()
            name = part.split(",")[0].strip()
            if name:
                names.append(name)
        return " and ".join(names)

    @staticmethod
    def _subject_scope(data_subject: str, data_sender: str) -> str:
        subject = normalize_text(data_subject).lower()
        sender = normalize_text(data_sender).split(",")[0].strip().lower()
        if not subject:
            return "unknown"
        if " and " in subject:
            return "multi_subject"
        if sender and sender in subject:
            return "self"
        return "third_party"

    @staticmethod
    def _bucket_count(count: int) -> str:
        if count <= 0:
            return "0"
        if count >= 5:
            return "5+"
        return str(count)

    def _seed_dimension_values(
        self,
        seed_candidate: Dict[str, Any],
        vignette: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        seed = seed_candidate if isinstance(seed_candidate, dict) else {}
        vignette_obj = vignette if isinstance(vignette, dict) else {}

        action = normalize_text(seed.get("final_action", "")) or "unknown_action"
        toolkit_sig = "|".join(sorted(seed.get("toolkits", []))) or "unknown_toolkits"
        subject_name = self._extract_subject_name(seed.get("data_subject", ""))
        recipient_name = self._extract_subject_name(seed.get("data_recipient", ""))
        subject_scope = self._subject_scope(
            seed.get("data_subject", ""),
            seed.get("data_sender", ""),
        )
        toolkit_count_bucket = self._bucket_count(len(seed.get("toolkits", [])))
        sensitive_items = vignette_obj.get("sensitive_info_items", [])
        if not isinstance(sensitive_items, list):
            sensitive_items = []
        normalized_sensitive_items = [
            normalize_text(item)
            for item in sensitive_items
            if isinstance(item, str) and normalize_text(item)
        ]
        sensitive_item_count_bucket = self._bucket_count(len(normalized_sensitive_items))
        raw_domains = seed.get("domains")
        if raw_domains is None and "scenario_domain" in seed:
            raw_domains = [seed.get("scenario_domain", "")]
        domains = normalize_domain_list(raw_domains)
        if not domains:
            domains = ["unknown"]
        return {
            "action": action,
            "domain_signature": domain_signature(domains),
            "toolkit_sig": toolkit_sig,
            "subject_name": subject_name,
            "recipient_name": recipient_name,
            "subject_scope": subject_scope,
            "toolkit_count_bucket": toolkit_count_bucket,
            "sensitive_item_count_bucket": sensitive_item_count_bucket,
            "domains": domains,
        }

    @staticmethod
    def _increment(counter: Counter[str], key: str, weight: float = 1.0) -> None:
        if key:
            counter[key] += weight

    @staticmethod
    def _decrement(counter: Counter[str], key: str, weight: float = 1.0) -> None:
        if key and counter[key] > 0:
            counter[key] = max(0.0, counter[key] - weight)

    def warm_from_dataset(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Failed to read bootstrap dataset %s: %s", path, exc)
            return

        if not isinstance(data, list):
            return

        self.warm_from_samples(data)

    def warm_from_samples(self, samples: List[Dict[str, Any]]) -> None:
        for item in samples:
            seed = item.get("seed", {}) if isinstance(item, dict) else {}
            trajectory = item.get("trajectory", {}) if isinstance(item, dict) else {}
            vignette = item.get("vignette", {}) if isinstance(item, dict) else {}
            metadata = item.get("generation_metadata", {}) if isinstance(item, dict) else {}

            sender_name = normalize_text(seed.get("data_sender_name", ""))
            if sender_name:
                self.name_counts[sender_name] += 1

            raw_domains = seed.get("domains") or metadata.get("domains")
            if not raw_domains:
                legacy_domain = (
                    normalize_text(seed.get("scenario_domain", ""))
                    or normalize_text(metadata.get("scenario_domain", ""))
                    or normalize_text(seed.get("data_type", ""))
                )
                raw_domains = [legacy_domain] if legacy_domain else []

            warm_seed = {
                "domains": raw_domains,
                "final_action": normalize_text(trajectory.get("final_action", "")),
                "toolkits": trajectory.get("toolkits", []) if isinstance(trajectory, dict) else [],
                "data_subject": normalize_text(seed.get("data_subject", "")),
                "data_sender": normalize_text(seed.get("data_sender", "")),
                "data_recipient": normalize_text(seed.get("data_recipient", "")),
            }
            warm_vignette = {
                "story": vignette.get("story", "") if isinstance(vignette, dict) else "",
                "sensitive_info_items": (
                    vignette.get("sensitive_info_items", []) if isinstance(vignette, dict) else []
                ),
            }
            dims = self._seed_dimension_values(warm_seed, warm_vignette)

            self._increment(self.subject_name_counts, dims["subject_name"])
            self._increment(self.recipient_name_counts, dims["recipient_name"])
            domain_weight = 1.0 / len(dims["domains"])
            for domain in dims["domains"]:
                self._increment(self.domain_counts, domain, domain_weight)
            self._increment(self.domain_signature_counts, dims["domain_signature"])
            self._increment(self.action_counts, dims["action"])
            self._increment(self.toolkit_signature_counts, dims["toolkit_sig"])
            self._increment(self.subject_scope_counts, dims["subject_scope"])
            self._increment(self.toolkit_count_counts, dims["toolkit_count_bucket"])
            self._increment(self.sensitive_item_count_counts, dims["sensitive_item_count_bucket"])

            # Warm profile-level counters from generation_metadata
            sex = normalize_text(metadata.get("profile_sex", "")).lower()
            ethnicity = normalize_text(metadata.get("profile_ethnicity", "")).lower()
            religion = normalize_text(metadata.get("profile_religion", "")).lower()
            citizenship = normalize_text(metadata.get("profile_citizenship", "")).lower()
            occupation = normalize_text(metadata.get("profile_occupation", "")).lower()
            if sex:
                self.sex_counts[sex] += 1
            if ethnicity:
                self.ethnicity_counts[ethnicity] += 1
            if religion:
                self.religion_counts[religion] += 1
            if citizenship:
                self.citizenship_counts[citizenship] += 1
            if occupation:
                self.occupation_counts[occupation] += 1

    def update_seed(
        self,
        seed_candidate: Dict[str, Any],
        vignette: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update domain/toolkit/action counters after seed selection.

        Name counts are updated eagerly in sample_first_name(), so they
        are intentionally excluded here to avoid double-counting.
        """
        dims = self._seed_dimension_values(seed_candidate, vignette)
        domain_weight = 1.0 / len(dims["domains"])
        for domain in dims["domains"]:
            self._increment(self.domain_counts, domain, domain_weight)
        self._increment(self.domain_signature_counts, dims["domain_signature"])
        self._increment(self.subject_name_counts, dims["subject_name"])
        self._increment(self.recipient_name_counts, dims["recipient_name"])
        self._increment(self.toolkit_signature_counts, dims["toolkit_sig"])
        self._increment(self.action_counts, dims["action"])
        self._increment(self.subject_scope_counts, dims["subject_scope"])
        self._increment(self.toolkit_count_counts, dims["toolkit_count_bucket"])
        self._increment(self.sensitive_item_count_counts, dims["sensitive_item_count_bucket"])

    def rollback_seed(
        self,
        seed_candidate: Dict[str, Any],
        vignette: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reverse the counters incremented by update_seed().

        Called when a sample that was eagerly counted in Stage 2 fails in a
        later stage, so that the counters stay accurate for subsequent batches.
        """
        dims = self._seed_dimension_values(seed_candidate, vignette)
        domain_weight = 1.0 / len(dims["domains"])
        for domain in dims["domains"]:
            self._decrement(self.domain_counts, domain, domain_weight)
        self._decrement(self.domain_signature_counts, dims["domain_signature"])
        self._decrement(self.subject_name_counts, dims["subject_name"])
        self._decrement(self.recipient_name_counts, dims["recipient_name"])
        self._decrement(self.toolkit_signature_counts, dims["toolkit_sig"])
        self._decrement(self.action_counts, dims["action"])
        self._decrement(self.subject_scope_counts, dims["subject_scope"])
        self._decrement(self.toolkit_count_counts, dims["toolkit_count_bucket"])
        self._decrement(self.sensitive_item_count_counts, dims["sensitive_item_count_bucket"])

    # ------------------------------------------------------------------
    # Profile scoring / tracking (overgeneration support)
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_score(count: float, counter: Counter[str]) -> float:
        """Score how under-represented a value is relative to its dimension's mean.

        Normalizes the raw count by the average count across all distinct
        values in the dimension, then applies an inverse-square penalty.
        This keeps all dimensions on a comparable scale regardless of
        cardinality: ~1.0 for novel values, ~0.25 at the mean, approaching
        0 for over-represented values.
        """
        if not counter:
            return 1.0
        total = sum(counter.values())
        num_distinct = len(counter)
        mean = total / max(1, num_distinct)
        normalized = count / mean if mean > 0 else 0.0
        return 1.0 / (1.0 + normalized) ** 2

    def score_profile(self, profile: Dict[str, Any]) -> float:
        """Score a profile candidate by how underrepresented its attributes are."""
        sex = normalize_text(profile.get("sex", "")).lower()
        ethnicity = normalize_text(profile.get("ethnicity", "")).lower()
        religion = normalize_text(profile.get("religion", "")).lower()
        citizenship = normalize_text(profile.get("citizenship", "")).lower()
        occupation = normalize_text(profile.get("occupation", "")).lower()
        return (
            self._relative_score(self.sex_counts[sex], self.sex_counts)
            + self._relative_score(self.ethnicity_counts[ethnicity], self.ethnicity_counts)
            + self._relative_score(self.religion_counts[religion], self.religion_counts)
            + self._relative_score(self.citizenship_counts[citizenship], self.citizenship_counts)
            + self._relative_score(self.occupation_counts[occupation], self.occupation_counts)
        )

    # Weights for scenario diversity dimensions. Domain mass carries the most
    # weight because it drives the broadest contextual diversity. Repeated
    # domain signatures get a lighter penalty so we discourage overused
    # combinations without overpowering the marginal domain signal.
    SCENARIO_WEIGHTS: Dict[str, float] = {
        "domain": 4.0,
        "domain_signature": 2.0,
        "action": 1.0,
        "toolkit_sig": 1.0,
        "subject_name": 1.0,
        "recipient_name": 1.0,
        "subject_scope": 1.0,
        "toolkit_count_bucket": 1.0,
        "sensitive_item_count_bucket": 1.0,
    }

    def score_scenario(
        self,
        seed_candidate: Dict[str, Any],
        vignette: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Score scenario novelty using weighted structural + semantic dimensions."""
        dims = self._seed_dimension_values(seed_candidate, vignette)
        w = self.SCENARIO_WEIGHTS
        domain_weight = 1.0 / len(dims["domains"])
        domain_score = sum(
            domain_weight * self._relative_score(self.domain_counts[domain], self.domain_counts)
            for domain in dims["domains"]
        )
        return (
            w["domain"] * domain_score
            + w["domain_signature"] * self._relative_score(
                self.domain_signature_counts[dims["domain_signature"]],
                self.domain_signature_counts,
            )
            + w["action"] * self._relative_score(self.action_counts[dims["action"]], self.action_counts)
            + w["toolkit_sig"] * self._relative_score(self.toolkit_signature_counts[dims["toolkit_sig"]], self.toolkit_signature_counts)
            + w["subject_name"] * self._relative_score(self.subject_name_counts[dims["subject_name"]], self.subject_name_counts)
            + w["recipient_name"] * self._relative_score(self.recipient_name_counts[dims["recipient_name"]], self.recipient_name_counts)
            + w["subject_scope"] * self._relative_score(self.subject_scope_counts[dims["subject_scope"]], self.subject_scope_counts)
            + w["toolkit_count_bucket"] * self._relative_score(self.toolkit_count_counts[dims["toolkit_count_bucket"]], self.toolkit_count_counts)
            + w["sensitive_item_count_bucket"] * self._relative_score(self.sensitive_item_count_counts[dims["sensitive_item_count_bucket"]], self.sensitive_item_count_counts)
        )

    def update_profile(self, profile: Dict[str, Any]) -> None:
        """Increment profile-level diversity counters."""
        sex = normalize_text(profile.get("sex", "")).lower()
        ethnicity = normalize_text(profile.get("ethnicity", "")).lower()
        religion = normalize_text(profile.get("religion", "")).lower()
        citizenship = normalize_text(profile.get("citizenship", "")).lower()
        occupation = normalize_text(profile.get("occupation", "")).lower()
        if sex:
            self.sex_counts[sex] += 1
        if ethnicity:
            self.ethnicity_counts[ethnicity] += 1
        if religion:
            self.religion_counts[religion] += 1
        if citizenship:
            self.citizenship_counts[citizenship] += 1
        if occupation:
            self.occupation_counts[occupation] += 1

    def rollback_profile(self, profile: Dict[str, Any]) -> None:
        """Reverse the counters incremented by update_profile()."""
        sex = normalize_text(profile.get("sex", "")).lower()
        ethnicity = normalize_text(profile.get("ethnicity", "")).lower()
        religion = normalize_text(profile.get("religion", "")).lower()
        citizenship = normalize_text(profile.get("citizenship", "")).lower()
        occupation = normalize_text(profile.get("occupation", "")).lower()
        if sex and self.sex_counts[sex] > 0:
            self.sex_counts[sex] -= 1
        if ethnicity and self.ethnicity_counts[ethnicity] > 0:
            self.ethnicity_counts[ethnicity] -= 1
        if religion and self.religion_counts[religion] > 0:
            self.religion_counts[religion] -= 1
        if citizenship and self.citizenship_counts[citizenship] > 0:
            self.citizenship_counts[citizenship] -= 1
        if occupation and self.occupation_counts[occupation] > 0:
            self.occupation_counts[occupation] -= 1

    def rollback_name(self, first_name: str) -> None:
        """Reverse the name_counts increment from sample_first_name()."""
        name = first_name.strip()
        if name and self.name_counts[name] > 0:
            self.name_counts[name] -= 1

    def sample_first_name(
        self,
        name_rows: List[NameRow],
        frequency_alpha: float,
        repeat_penalty: float,
    ) -> str:
        weights: List[float] = []
        for row in name_rows:
            if self.name_counts[row.name] > 0:
                weights.append(0.0)
            else:
                weights.append(float(row.frequency) ** frequency_alpha)
        if not any(w > 0 for w in weights):
            LOGGER.warning("All names exhausted; falling back to soft penalty.")
            weights = []
            for row in name_rows:
                base = float(row.frequency) ** frequency_alpha
                penalty = 1.0 / (1.0 + self.name_counts[row.name] * repeat_penalty) ** 2
                weights.append(base * penalty)
        picked = random.choices(name_rows, weights=weights, k=1)[0]
        self.name_counts[picked.name] += 1
        return picked.name
