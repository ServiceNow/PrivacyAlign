#!/usr/bin/env python3
"""Prune mined selected pairs with the exact postprocess diversity solver.

This script uses the original source samples for diversity signals
(domains, domain signatures, final actions, toolkits, model families, and
subject scope), optionally filters to unanimously sensible selected pairs,
and writes the pruned result back into the mined JSON schema.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import postprocess
from pipeline.utils import write_json_atomic


LOGGER = logging.getLogger("prune_mined_pairs")
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune mined selected pairs with the current exact postprocess "
            "diversity logic, optionally after requiring unanimous sensibility."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the mined JSON produced by mine_samples.py.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for the pruned mined JSON.",
    )
    parser.add_argument(
        "--source-input-path",
        type=str,
        default=None,
        help=(
            "Optional explicit source dataset path. Defaults to "
            "metadata.input_path from the mined file."
        ),
    )
    parser.add_argument(
        "--require-unanimous-sensible",
        action="store_true",
        help=(
            "Keep only selected pairs whose two chosen responses were judged "
            "sensible by every judge model before diversity pruning."
        ),
    )
    parser.add_argument(
        "--balance-fields",
        nargs="+",
        default=list(postprocess.DEFAULT_CAP_BALANCE_FIELDS),
        help=(
            "Fields to balance with the exact postprocess diversity solver. "
            "Defaults match postprocess.py."
        ),
    )
    parser.add_argument(
        "--max-pct",
        type=float,
        default=15.0,
        help="Default max percentage cap for active balance fields.",
    )
    parser.add_argument(
        "--action-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap for 'trajectory.final_action'. When set, "
            "this overrides --max-pct for final actions."
        ),
    )
    parser.add_argument(
        "--domain-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap for 'generation_metadata.domains'. When set, "
            "this overrides --max-pct for raw domains."
        ),
    )
    parser.add_argument(
        "--domain-signature-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap for 'generation_metadata.domain_signature'. "
            "When set, this overrides --max-pct for domain signatures."
        ),
    )
    parser.add_argument(
        "--toolkit-signature-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap for 'generation_metadata.toolkit_signature'. "
            "When set, this overrides --max-pct for toolkit signatures."
        ),
    )
    parser.add_argument(
        "--toolkit-max-pct",
        type=float,
        default=None,
        help=(
            "Optional stricter cap for 'trajectory.toolkits'. When set, "
            "this overrides --max-pct for per-toolkit marginals and "
            "automatically adds 'trajectory.toolkits' to --balance-fields."
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help=(
            "Optional upper bound on the final number of selected pairs. The "
            "exact solver keeps the largest feasible subset up to this count."
        ),
    )
    parser.add_argument(
        "--self-scope-pct",
        type=float,
        default=None,
        help="Optional minimum percentage for 'self' subject_scope samples.",
    )
    parser.add_argument(
        "--third-party-scope-pct",
        "--other-scope-pct",
        dest="third_party_scope_pct",
        type=float,
        default=None,
        help="Optional minimum percentage for 'third_party' subject_scope samples.",
    )
    parser.add_argument(
        "--multi-subject-scope-pct",
        dest="multi_subject_scope_pct",
        type=float,
        default=None,
        help="Optional minimum percentage for 'multi_subject' subject_scope samples.",
    )
    parser.add_argument(
        "--min-model-family-pct",
        type=float,
        default=None,
        help=(
            "Optional minimum percentage for each requested model family. "
            "These minimum-share constraints are enforced jointly with caps."
        ),
    )
    parser.add_argument(
        "--model-families",
        nargs="+",
        default=None,
        help=(
            "Model families to use when --min-model-family-pct is set. "
            "Defaults to the primary families inferred from the data."
        ),
    )
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=None,
        help="Optional solve limit in seconds for the exact diversity solver.",
    )
    parser.add_argument(
        "--exact-mip-rel-gap",
        type=float,
        default=None,
        help="Optional relative MIP gap target for the exact diversity solver.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed recorded in metadata for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info-level logs from this script and postprocess helpers.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_source_input_path(
    mined_path: Path,
    metadata: Dict[str, Any],
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    input_path = metadata.get("input_path")
    if not input_path:
        raise ValueError("mined metadata.input_path is missing; pass --source-input-path")

    candidate = Path(str(input_path)).expanduser()
    if candidate.is_absolute():
        return candidate

    for base in (SCRIPT_DIR, mined_path.parent, SCRIPT_DIR.parent):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not resolve source input path '{input_path}' from mined metadata"
    )


def collect_stats(samples: Iterable[Dict[str, Any]]) -> Dict[str, Counter[str]]:
    action_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    domain_signature_counts: Counter[str] = Counter()
    toolkit_signature_counts: Counter[str] = Counter()
    toolkit_counts: Counter[str] = Counter()

    for sample in samples:
        action_counts[sample.get("trajectory", {}).get("final_action", "unknown")] += 1
        family_counts[postprocess.compute_model_family(sample)] += 1
        scope_counts[postprocess.compute_subject_scope(sample)] += 1
        for domain, weight in postprocess._field_contributions(
            sample,
            "generation_metadata.domains",
        ):
            domain_counts[domain] += weight
        domain_signature_counts[postprocess.compute_domain_signature(sample)] += 1
        toolkit_signature_counts[postprocess.compute_toolkit_signature(sample)] += 1
        for toolkit, weight in postprocess._field_contributions(
            sample,
            "trajectory.toolkits",
        ):
            toolkit_counts[toolkit] += weight

    return {
        "action": action_counts,
        "family": family_counts,
        "scope": scope_counts,
        "domain": domain_counts,
        "domain_signature": domain_signature_counts,
        "toolkit_signature": toolkit_signature_counts,
        "toolkit": toolkit_counts,
    }


def top_items(counter: Counter[str], limit: int = 8) -> List[Tuple[str, float]]:
    return counter.most_common(limit)


def _format_counter_items(items: Sequence[Tuple[str, float]]) -> str:
    formatted: List[str] = []
    for key, value in items:
        if isinstance(value, float) and not float(value).is_integer():
            formatted.append(f"{key}={value:.2f}")
        else:
            formatted.append(f"{key}={int(value)}")
    return ", ".join(formatted)


def count_sensibility_yes(
    sample_record: Dict[str, Any],
    response_model: str,
    judge_models: Sequence[str],
) -> int:
    sensibility = sample_record.get("sensibility_judgments", {})
    return sum(
        1
        for judge_model in judge_models
        if sensibility.get(judge_model, {}).get(response_model, {}).get("sensible") is True
    )


def selected_pair_is_unanimously_sensible(
    sample_record: Dict[str, Any],
    selected_pair: Dict[str, Any],
    judge_models: Sequence[str],
) -> bool:
    total_judges = len(judge_models)
    a_yes = count_sensibility_yes(sample_record, selected_pair["model_a"], judge_models)
    b_yes = count_sensibility_yes(sample_record, selected_pair["model_b"], judge_models)
    return a_yes == total_judges and b_yes == total_judges


def build_field_max_pcts(args: argparse.Namespace) -> Dict[str, float]:
    return {
        field_path: pct
        for field_path, pct in {
            "trajectory.final_action": args.action_max_pct,
            "trajectory.toolkits": args.toolkit_max_pct,
            "generation_metadata.domains": args.domain_max_pct,
            "generation_metadata.domain_signature": args.domain_signature_max_pct,
            "generation_metadata.toolkit_signature": args.toolkit_signature_max_pct,
        }.items()
        if pct is not None and pct > 0
    }


def resolve_balance_fields(args: argparse.Namespace) -> List[str]:
    balance_fields: List[str] = []
    seen: set[str] = set()
    for field_path in args.balance_fields:
        normalized = str(field_path).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        balance_fields.append(normalized)

    if args.toolkit_max_pct is not None and args.toolkit_max_pct > 0:
        if "trajectory.toolkits" not in seen:
            balance_fields.append("trajectory.toolkits")

    return balance_fields


def build_min_pct_constraints(
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, str, float]], List[str], Dict[str, float]]:
    scope_min_pcts: Dict[str, float] = {}
    if args.self_scope_pct is not None:
        scope_min_pcts["self"] = args.self_scope_pct
    if args.third_party_scope_pct is not None:
        scope_min_pcts["third_party"] = args.third_party_scope_pct
    if args.multi_subject_scope_pct is not None:
        scope_min_pcts["multi_subject"] = args.multi_subject_scope_pct

    resolved_model_families: List[str] = []
    if args.min_model_family_pct is not None:
        resolved_model_families = postprocess.resolve_model_families(
            samples,
            args.model_families,
        )

    constraints = postprocess.build_min_pct_constraints(
        samples,
        scope_min_pcts,
        resolved_model_families,
        args.min_model_family_pct,
    )
    return constraints, resolved_model_families, scope_min_pcts


def apply_pruning(
    mined_data: Dict[str, Any],
    source_samples: List[Dict[str, Any]],
    *,
    require_unanimous_sensible: bool,
    balance_fields: List[str],
    max_pct: float,
    field_max_pcts: Dict[str, float],
    target_count: int | None,
    args: argparse.Namespace,
    exact_time_limit: float | None,
    exact_mip_rel_gap: float | None,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    selected_pairs = list(mined_data.get("selected_pairs", []))
    selected_names = [entry.get("name") for entry in selected_pairs if entry.get("name")]

    source_by_name = {sample.get("name"): sample for sample in source_samples}
    missing_source = sorted(name for name in selected_names if name not in source_by_name)
    if missing_source:
        preview = ", ".join(missing_source[:10])
        raise KeyError(
            f"{len(missing_source)} selected names are missing from the source dataset: {preview}"
        )

    filtered_pairs = list(selected_pairs)
    filtered_names = list(selected_names)
    unanimous_removed_names: List[str] = []
    if require_unanimous_sensible:
        judge_models = list(mined_data.get("metadata", {}).get("judge_models", []))
        if not judge_models:
            raise ValueError(
                "--require-unanimous-sensible needs metadata.judge_models in the mined file"
            )

        sample_records = {
            sample.get("name"): sample
            for sample in mined_data.get("samples", [])
            if sample.get("name")
        }
        missing_records = sorted(name for name in selected_names if name not in sample_records)
        if missing_records:
            preview = ", ".join(missing_records[:10])
            raise KeyError(
                f"{len(missing_records)} selected names are missing from mined samples: {preview}"
            )

        filtered_pairs = [
            entry
            for entry in selected_pairs
            if selected_pair_is_unanimously_sensible(
                sample_records[entry["name"]],
                entry,
                judge_models,
            )
        ]
        filtered_names = [entry["name"] for entry in filtered_pairs]
        filtered_name_set = set(filtered_names)
        unanimous_removed_names = sorted(
            name for name in selected_names if name not in filtered_name_set
        )

    filtered_source_samples = [
        source_by_name[entry["name"]]
        for entry in filtered_pairs
    ]
    postprocess.materialize_computed_fields(filtered_source_samples)
    min_pct_constraints, resolved_model_families, scope_min_pcts = build_min_pct_constraints(
        filtered_source_samples,
        args,
    )

    before_stats = collect_stats(filtered_source_samples)
    final_samples = postprocess.tighten_diversity_caps(
        list(filtered_source_samples),
        balance_fields,
        max_pct,
        field_max_pcts=field_max_pcts,
        max_total_count=target_count,
        protected_pct_constraints=min_pct_constraints,
        progress_desc="Selected-pair cap tightening",
        time_limit=exact_time_limit,
        mip_rel_gap=exact_mip_rel_gap,
    )
    after_stats = collect_stats(final_samples)

    keep_names = {sample["name"] for sample in final_samples}
    cap_removed_names = sorted(name for name in filtered_names if name not in keep_names)
    removed_names = sorted(set(unanimous_removed_names) | set(cap_removed_names))

    pruned = copy.deepcopy(mined_data)
    pruned_selected_pairs = [
        entry for entry in pruned.get("selected_pairs", []) if entry.get("name") in keep_names
    ]
    pruned["selected_pairs"] = pruned_selected_pairs

    unanimous_removed_name_set = set(unanimous_removed_names)
    for sample in pruned.get("samples", []):
        name = sample.get("name")
        if name in removed_names and sample.get("selected_pair") is not None:
            sample["selected_pair"] = None
            sample["postprocess_pruned"] = True
            sample["prune_reason"] = (
                "not_unanimously_sensible"
                if name in unanimous_removed_name_set
                else "diversity_cap_trim"
            )

    metadata = dict(pruned.get("metadata", {}))
    metadata["num_selected_pairs"] = len(pruned_selected_pairs)
    metadata["selected_judge_counts"] = dict(
        Counter(entry.get("judge_model") for entry in pruned_selected_pairs)
    )
    metadata["selected_pair_family_counts"] = dict(
        Counter(entry.get("pair_family") for entry in pruned_selected_pairs)
    )
    metadata["postprocess_pruning"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "derived_from": None,
        "source_input_path": None,
        "seed": seed,
        "require_unanimous_sensible": require_unanimous_sensible,
        "balance_fields": list(balance_fields),
        "max_pct": max_pct,
        "field_max_pcts": dict(field_max_pcts),
        "target_count": target_count,
        "exact_time_limit": exact_time_limit,
        "exact_mip_rel_gap": exact_mip_rel_gap,
        "resolved_model_families": list(resolved_model_families),
        "scope_min_pcts": dict(scope_min_pcts),
        "min_pct_constraints": [
            {"field": field, "value": value, "pct": pct}
            for field, value, pct in min_pct_constraints
        ],
        "selected_pairs_before": len(selected_pairs),
        "selected_pairs_after_prefilter": len(filtered_pairs),
        "selected_pairs_after": len(pruned_selected_pairs),
        "removed_before_cap": len(unanimous_removed_names),
        "removed_by_caps": len(cap_removed_names),
        "removed_total": len(removed_names),
        "removed_names": removed_names,
    }
    metadata.pop("unanimous_subset_pruning", None)
    pruned["metadata"] = metadata

    summary = {
        "selected_before": len(selected_pairs),
        "selected_after_prefilter": len(filtered_pairs),
        "selected_after_final": len(final_samples),
        "require_unanimous_sensible": require_unanimous_sensible,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "field_max_pcts": dict(field_max_pcts),
        "max_pct": max_pct,
        "target_count": target_count,
        "min_pct_constraints": list(min_pct_constraints),
        "removed_before_cap": len(unanimous_removed_names),
        "removed_by_caps": len(cap_removed_names),
        "removed_total": len(removed_names),
    }
    return pruned, summary


def emit_summary(summary: Dict[str, Any], balance_fields: Sequence[str]) -> None:
    before_count = summary["selected_after_prefilter"]
    after_count = summary["selected_after_final"]
    before_stats = summary["before_stats"]
    after_stats = summary["after_stats"]

    if summary["require_unanimous_sensible"]:
        print(
            "Selected pairs:",
            f"{summary['selected_before']} total",
            f"-> {before_count} after unanimous filter",
            f"-> {after_count} final",
        )
    else:
        print(
            "Selected pairs:",
            f"{summary['selected_before']} total",
            f"-> {after_count} final",
        )
    if summary.get("target_count") is not None:
        print(
            "Target count:",
            summary["target_count"],
            f"(final: {after_count})",
        )
    print(
        "Cap configuration:",
        postprocess.format_field_cap_pcts(
            list(balance_fields),
            summary["max_pct"],
            summary["field_max_pcts"],
        ),
    )
    if summary["min_pct_constraints"]:
        print(
            "Minimum-share constraints:",
            postprocess.format_min_pct_constraints(summary["min_pct_constraints"]),
        )
    print(
        "Removed:",
        f"{summary['removed_total']} total",
        f"({summary['removed_before_cap']} prefilter, {summary['removed_by_caps']} cap trim)",
    )
    print(
        "Model family final:",
        dict(after_stats["family"]),
        "(before:",
        dict(before_stats["family"]),
        ")",
    )
    print(
        "Subject scope final:",
        dict(after_stats["scope"]),
        "(before:",
        dict(before_stats["scope"]),
        ")",
    )
    print("Top actions final:", _format_counter_items(top_items(after_stats["action"])))
    print("Top domains final:", _format_counter_items(top_items(after_stats["domain"])))
    print(
        "Top domain signatures final:",
        _format_counter_items(top_items(after_stats["domain_signature"])),
    )
    print("Top toolkits final:", _format_counter_items(top_items(after_stats["toolkit"])))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.target_count is not None and args.target_count <= 0:
        raise ValueError("--target-count must be positive when provided")

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    mined_data = load_json(input_path)
    if not isinstance(mined_data, dict):
        raise TypeError("Expected mined JSON to be a top-level object")

    source_input_path = resolve_source_input_path(
        input_path,
        mined_data.get("metadata", {}),
        args.source_input_path,
    )
    source_samples = load_json(source_input_path)
    if not isinstance(source_samples, list):
        raise TypeError("Expected source dataset to be a list of samples")

    balance_fields = resolve_balance_fields(args)
    field_max_pcts = build_field_max_pcts(args)
    pruned, summary = apply_pruning(
        mined_data,
        source_samples,
        require_unanimous_sensible=args.require_unanimous_sensible,
        balance_fields=balance_fields,
        max_pct=args.max_pct,
        field_max_pcts=field_max_pcts,
        target_count=args.target_count,
        args=args,
        exact_time_limit=args.exact_time_limit,
        exact_mip_rel_gap=args.exact_mip_rel_gap,
        seed=args.seed,
    )

    pruning_metadata = pruned["metadata"]["postprocess_pruning"]
    pruning_metadata["derived_from"] = str(input_path)
    pruning_metadata["source_input_path"] = str(source_input_path)

    write_json_atomic(output_path, pruned)
    emit_summary(summary, balance_fields)
    print(f"Wrote pruned mined file to {output_path}")


if __name__ == "__main__":
    main()
