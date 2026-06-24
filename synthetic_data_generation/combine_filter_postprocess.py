#!/usr/bin/env python3
"""
Combine generated outputs, run embedding diversity filtering, then postprocess.

Default behavior:
1. Discover `outputs/main_data_generated*.json`.
2. Merge files in primary family order: qwen -> gpt -> nvidia, then any
   legacy step-family outputs, then other files.
3. Make sample names globally unique while preserving original names in metadata.
4. Run embedding diversity filtering on the merged stream. By default, this
   preserves source order so qwen/gpt samples are seen before nvidia during
   deduplication; the explicit 'protected' mode is still available when you
   want to defer the most abundant family.
5. Run postprocess with retention-friendly defaults and equal model-family balancing.

Examples:
  python3 combine_filter_postprocess.py

  python3 combine_filter_postprocess.py \
    --max-similarity 0.995 \
    --enable-postprocess-sampling \
    --name-threshold 2.0 \
    --max-pct 20.0 \
    --min-model-family-pct 30.0 \
    --self-scope-pct 50.0
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pipeline.embedding_diversity import EmbeddingDiversityFilter
from pipeline.utils import append_jsonl, write_json_atomic

LOGGER = logging.getLogger("combine_filter_postprocess")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_COMBINED_STEM = "main_data_generated_combined_qwen_gpt_nvidia"
DEFAULT_COMBINED_PATH = DEFAULT_OUTPUTS_DIR / f"{DEFAULT_COMBINED_STEM}.json"
DEFAULT_FILTERED_PATH = (
    DEFAULT_OUTPUTS_DIR / f"{DEFAULT_COMBINED_STEM}_diversity_filtered.json"
)
DEFAULT_POSTPROCESSED_PATH = (
    DEFAULT_OUTPUTS_DIR / f"{DEFAULT_COMBINED_STEM}_diversity_filtered_postprocessed.json"
)
DEFAULT_EMBED_REJECTED_PATH = (
    DEFAULT_OUTPUTS_DIR / f"{DEFAULT_COMBINED_STEM}_diversity_rejected.jsonl"
)
DEFAULT_POSTPROCESS_REJECTED_PATH = (
    DEFAULT_OUTPUTS_DIR / f"{DEFAULT_COMBINED_STEM}_postprocess_rejected.jsonl"
)

PRIMARY_FAMILY_ORDER = ["qwen", "gpt", "nvidia"]
DEFAULT_MODEL_FAMILIES = list(PRIMARY_FAMILY_ORDER)
FAMILY_ORDER = [*PRIMARY_FAMILY_ORDER, "step", "other"]
DERIVED_NAME_MARKERS = ("combined", "diversity_filtered", "postprocessed", "mined")

_EMAIL_LOCAL_PATTERN_RE = re.compile(
    r"^(?P<prefix>[a-z0-9]+)"
    r"(?:(?P<sep>[._-])(?P<surname>[a-z]+)(?P<digits>\d*)(?P<rest>(?:[._-][a-z0-9]+)*)|"
    r"(?P<joined>[a-z]+)(?P<joined_digits>\d*))$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge generated outputs in qwen->gpt->nvidia order, run embedding "
            "diversity filtering, then run postprocess."
        ),
    )

    io = parser.add_argument_group("Input / output")
    io.add_argument(
        "--input-paths",
        nargs="*",
        default=None,
        help=(
            "Explicit JSON files to combine. If omitted, auto-discovers "
            "outputs/main_data_generated*.json and skips derived outputs."
        ),
    )
    io.add_argument(
        "--outputs-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Directory to scan when --input-paths is omitted.",
    )
    io.add_argument(
        "--combined-output-path",
        type=str,
        default=str(DEFAULT_COMBINED_PATH),
        help="Path for the merged, ordered dataset.",
    )
    io.add_argument(
        "--filtered-output-path",
        type=str,
        default=str(DEFAULT_FILTERED_PATH),
        help="Path for the embedding-diversity-filtered dataset.",
    )
    io.add_argument(
        "--postprocessed-output-path",
        type=str,
        default=str(DEFAULT_POSTPROCESSED_PATH),
        help="Path for the final postprocessed dataset.",
    )
    io.add_argument(
        "--embedding-rejected-path",
        type=str,
        default=str(DEFAULT_EMBED_REJECTED_PATH),
        help="JSONL file for embedding-diversity rejections.",
    )
    io.add_argument(
        "--postprocess-rejected-path",
        type=str,
        default=str(DEFAULT_POSTPROCESS_REJECTED_PATH),
        help="JSONL file for postprocess rejections.",
    )

    emb = parser.add_argument_group("Embedding diversity")
    emb.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model used for semantic deduplication.",
    )
    emb.add_argument(
        "--embedding-device",
        type=str,
        default="cuda",
        help="Embedding device (cuda, cpu, etc.).",
    )
    emb.add_argument(
        "--embed-batch-size",
        type=int,
        default=1,
        help="Batch size for embedding forward passes. Increase if your GPU has headroom.",
    )
    emb.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Max token length for the embedding model.",
    )
    emb.add_argument(
        "--max-similarity",
        type=float,
        default=0.99,
        help=(
            "Embedding similarity threshold. Higher keeps more data. "
            "Default 0.99 is more retention-friendly than the standalone script default."
        ),
    )
    emb.add_argument(
        "--embedding-order",
        choices=["auto", "source", "balanced", "protected"],
        default="auto",
        help=(
            "Order used for the embedding-dedup scan. 'source' preserves the "
            "merged qwen->gpt->nvidia order. 'balanced' round-robins the "
            "requested model families before overflow. 'protected' defers a "
            "uniquely most-abundant requested family until the scarcer "
            "families are scanned. 'auto' currently aliases 'source' so "
            "earlier qwen/gpt families are retained first during dedup."
        ),
    )

    post = parser.add_argument_group("Postprocess")
    post.add_argument(
        "--skip-link-filter",
        action="store_true",
        help="Pass through to postprocess.py.",
    )
    post.add_argument(
        "--skip-name-replace",
        action="store_true",
        help="Pass through to postprocess.py.",
    )
    post.add_argument(
        "--enable-postprocess-sampling",
        action="store_true",
        help=(
            "Enable postprocess diversity sampling. Disabled by default because "
            "it is the most size-reducing postprocess stage."
        ),
    )
    post.add_argument(
        "--max-pct",
        "--postprocess-max-pct",
        dest="postprocess_max_pct",
        type=float,
        default=20.0,
        help="Used only when --enable-postprocess-sampling is set.",
    )
    post.add_argument(
        "--name-threshold",
        "--postprocess-name-threshold",
        dest="postprocess_name_threshold",
        type=float,
        default=1.0,
        help="Pass through to postprocess.py --name-threshold.",
    )
    post.add_argument(
        "--self-scope-pct",
        "--postprocess-self-scope-pct",
        dest="postprocess_self_scope_pct",
        type=float,
        default=None,
        help="Optional pass-through to postprocess.py --self-scope-pct minimum.",
    )
    post.add_argument(
        "--third-party-scope-pct",
        "--other-scope-pct",
        dest="postprocess_third_party_scope_pct",
        type=float,
        default=None,
        help="Optional pass-through to postprocess.py --third-party-scope-pct minimum.",
    )
    post.add_argument(
        "--multi-subject-scope-pct",
        dest="postprocess_multi_subject_scope_pct",
        type=float,
        default=None,
        help="Optional pass-through to postprocess.py --multi-subject-scope-pct minimum.",
    )
    post.add_argument(
        "--postprocess-seed",
        type=int,
        default=42,
        help="Random seed passed to postprocess.py.",
    )
    post.add_argument(
        "--equalize-model-families",
        dest="equalize_model_families",
        action="store_true",
        default=True,
        help=(
            "Force equal counts across qwen/gpt/nvidia in postprocess. "
            "Enabled by default."
        ),
    )
    post.add_argument(
        "--skip-model-family-balance",
        dest="equalize_model_families",
        action="store_false",
        help=(
            "Do not force equal counts across qwen/gpt/nvidia in postprocess. "
            "By default, the final dataset is balanced across those families."
        ),
    )
    post.add_argument(
        "--model-families",
        nargs="+",
        default=list(DEFAULT_MODEL_FAMILIES),
        help="Model families to equalize in postprocess (default: qwen gpt nvidia).",
    )
    post.add_argument(
        "--min-model-family-pct",
        type=float,
        default=None,
        help=(
            "Softer alternative to exact equalization. Keep as much data as possible "
            "while ensuring each requested model family is at least this percentage "
            "of the final dataset."
        ),
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--stop-after-merge",
        action="store_true",
        help="Write the combined file and exit.",
    )
    misc.add_argument(
        "--stop-after-filter",
        action="store_true",
        help="Write the combined and filtered files and exit.",
    )
    misc.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


def is_discovered_input(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if not path.name.startswith("main_data_generated"):
        return False
    lowered = path.stem.lower()
    return not any(marker in lowered for marker in DERIVED_NAME_MARKERS)


def input_path_sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    if stem.endswith("_2"):
        return (stem[:-2], 0, path.name)
    return (stem, 1, path.name)


def discover_input_paths(outputs_dir: Path) -> List[Path]:
    paths = sorted(
        (path for path in outputs_dir.glob("main_data_generated*.json") if is_discovered_input(path)),
        key=input_path_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No generated JSON files found under {outputs_dir}")
    return paths


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    return payload


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
    if candidate == "other":
        return "other"
    return candidate


def detect_family(path: Path, samples: Sequence[Dict[str, Any]]) -> str:
    model_name = ""
    if samples:
        model_name = str(samples[0].get("model_name", ""))
    probe = f"{model_name} {path.name}".lower()
    if "nvidia" in probe or "nemotron" in probe:
        return "nvidia"
    if "gpt-oss" in probe or "gpt_oss" in probe or re.search(r"(^|[^a-z0-9])gpt([^a-z0-9]|$)", probe):
        return "gpt"
    if "stepfun" in probe or re.search(r"(^|[^a-z0-9])step([^a-z0-9]|$)", probe):
        return "step"
    if "qwen" in probe:
        return "qwen"
    return "other"


def order_paths(paths: Sequence[Path]) -> List[Tuple[str, Path, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Tuple[Path, List[Dict[str, Any]]]]] = defaultdict(list)
    for path in paths:
        samples = load_samples(path)
        family = detect_family(path, samples)
        grouped[family].append((path, samples))

    ordered: List[Tuple[str, Path, List[Dict[str, Any]]]] = []
    for family in FAMILY_ORDER:
        for path, samples in sorted(
            grouped.get(family, []),
            key=lambda item: input_path_sort_key(item[0]),
        ):
            ordered.append((family, path, samples))
    return ordered


def make_unique_name(name: str, seen: Counter[str]) -> str:
    if seen[name] == 0:
        seen[name] += 1
        return name
    seen[name] += 1
    return f"{name}__dup{seen[name]}"


def combine_samples(
    ordered_inputs: Sequence[Tuple[str, Path, List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    combined: List[Dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    seen_names: Counter[str] = Counter()

    for family, path, samples in ordered_inputs:
        LOGGER.info("Adding %d samples from %s [%s]", len(samples), path.name, family)
        for sample in samples:
            item = copy.deepcopy(sample)
            original_name = str(item.get("name", "unnamed"))
            unique_base = f"{family}__{path.stem}__{original_name}"
            unique_name = make_unique_name(unique_base, seen_names)

            metadata = item.setdefault("generation_metadata", {})
            metadata["combined_original_name"] = original_name
            metadata["combined_source_file"] = path.name
            metadata["combined_source_family"] = family
            metadata["combined_source_model_name"] = item.get("model_name", "")
            metadata["combined_order_index"] = len(combined)

            item["name"] = unique_name
            combined.append(item)
            family_counts[family] += 1

    return combined, family_counts


def clear_output_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _normalize_name_token(text: str) -> str:
    """Return an ASCII-ish lowercase token for email/name comparisons."""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _extract_email_name_parts(user_name: str, user_email: str) -> Dict[str, str] | None:
    """Parse a personal-looking user email into first/surname components."""
    email = str(user_email).strip().lower()
    if "@" not in email:
        return None

    parts = str(user_name).strip().split()
    if len(parts) < 2:
        return None

    first_norm = _normalize_name_token(parts[0])
    last_norm = _normalize_name_token(parts[-1])
    if len(first_norm) < 2 or len(last_norm) < 3:
        return None

    local, domain = email.split("@", 1)
    match = _EMAIL_LOCAL_PATTERN_RE.match(local)
    if not match:
        return None

    prefix = _normalize_name_token(match.group("prefix") or "")
    if prefix not in {first_norm, first_norm[0]}:
        return None

    sep = match.group("sep") or ""
    if sep:
        surname_token = _normalize_name_token(match.group("surname") or "")
        digits = match.group("digits") or ""
        rest = match.group("rest") or ""
    else:
        surname_token = _normalize_name_token(match.group("joined") or "")
        digits = match.group("joined_digits") or ""
        rest = ""

    if len(surname_token) < 3:
        return None

    return {
        "email": email,
        "local": local,
        "domain": domain,
        "prefix": prefix,
        "sep": sep,
        "surname_token": surname_token,
        "digits": digits,
        "rest": rest,
        "first_norm": first_norm,
        "last_norm": last_norm,
    }


def repair_postprocessed_user_emails(samples: Sequence[Dict[str, Any]]) -> int:
    """Repair stale surname tokens in personal-looking user emails.

    This runs after postprocess because some merged inputs still contain emails
    whose local-part surname token no longer matches the rewritten user name.
    The repair is intentionally conservative: it only rewrites addresses when
    the stale surname token is reused across multiple distinct visible last
    names, which is a strong signal of name-replacement drift rather than a
    legitimate alias.
    """
    parsed_rows: List[Tuple[Dict[str, Any], Dict[str, str]]] = []
    token_to_last_names: Dict[str, set[str]] = defaultdict(set)

    for sample in samples:
        traj = sample.get("trajectory", {})
        parsed = _extract_email_name_parts(
            traj.get("user_name", ""),
            traj.get("user_email", ""),
        )
        if not parsed:
            continue
        parsed_rows.append((sample, parsed))
        token_to_last_names[parsed["surname_token"]].add(parsed["last_norm"])

    changed = 0
    for sample, parsed in parsed_rows:
        if parsed["surname_token"] == parsed["last_norm"]:
            continue
        if len(token_to_last_names[parsed["surname_token"]]) < 2:
            continue

        if parsed["sep"]:
            new_local = (
                f"{parsed['prefix']}{parsed['sep']}{parsed['last_norm']}"
                f"{parsed['digits']}{parsed['rest']}"
            )
        else:
            new_local = (
                f"{parsed['prefix']}{parsed['last_norm']}{parsed['digits']}"
            )

        traj = sample.setdefault("trajectory", {})
        repaired_email = f"{new_local}@{parsed['domain']}"
        if traj.get("user_email") != repaired_email:
            traj["user_email"] = repaired_email
            changed += 1

    return changed


def summarize_counts(samples: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample in samples:
        family = (
            sample.get("generation_metadata", {}).get("combined_source_family")
            or "unknown"
        )
        counts[str(family)] += 1
    return counts


def normalize_requested_families(families: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for family in families:
        candidate = _normalize_family_label(str(family).strip().lower())
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def resolve_embedding_order_strategy(args: argparse.Namespace) -> str:
    if args.embedding_order != "auto":
        return str(args.embedding_order)

    return "source"


def split_family_queues_for_embedding_order(
    samples: Sequence[Dict[str, Any]],
    target_families: Sequence[str],
) -> Tuple[Dict[str, deque[Dict[str, Any]]], List[Dict[str, Any]]]:
    family_queues: Dict[str, deque[Dict[str, Any]]] = {
        family: deque() for family in target_families
    }
    overflow: List[Dict[str, Any]] = []

    for sample in samples:
        family = str(
            sample.get("generation_metadata", {}).get("combined_source_family")
            or ""
        ).strip().lower()
        if family in family_queues:
            family_queues[family].append(sample)
        else:
            overflow.append(sample)

    return family_queues, overflow


def round_robin_family_queues(
    family_queues: Dict[str, deque[Dict[str, Any]]],
    family_order: Sequence[str],
) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    while True:
        added = False
        for family in family_order:
            queue = family_queues.get(family)
            if queue:
                ordered.append(queue.popleft())
                added = True
        if not added:
            break
    return ordered


def plan_embedding_filter_inputs(
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], str]:
    strategy = resolve_embedding_order_strategy(args)
    if strategy == "source":
        return list(samples), strategy

    target_families = normalize_requested_families(args.model_families)
    if not target_families:
        return list(samples), "source"

    family_queues, overflow = split_family_queues_for_embedding_order(
        samples,
        target_families,
    )
    present_families = [family for family in target_families if family_queues[family]]
    if not present_families:
        return list(samples), "source"

    if strategy == "balanced":
        ordered = round_robin_family_queues(family_queues, target_families)
    else:
        present_counts = {
            family: len(family_queues[family])
            for family in present_families
        }
        largest_count = max(present_counts.values())
        largest_families = [
            family
            for family in present_families
            if present_counts[family] == largest_count
        ]
        if len(largest_families) == 1 and len(present_families) > 1:
            deferred_family = largest_families[0]
            protected_order = [
                family for family in target_families if family != deferred_family
            ]
            LOGGER.info(
                "Embedding scan order: deferring abundant family '%s' (%d samples) "
                "until scarcer requested families are scanned first",
                deferred_family,
                largest_count,
            )
            ordered = round_robin_family_queues(family_queues, protected_order)
            ordered.extend(round_robin_family_queues(family_queues, [deferred_family]))
        else:
            LOGGER.info(
                "Embedding scan order: no unique abundant requested family to defer; "
                "falling back to balanced round-robin"
            )
            ordered = round_robin_family_queues(family_queues, target_families)

    ordered.extend(overflow)
    return ordered, strategy


def sort_by_combined_order(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        samples,
        key=lambda sample: int(
            sample.get("generation_metadata", {}).get("combined_order_index", 0)
        ),
    )


def run_embedding_filter(
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    rejected_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clear_output_file(rejected_path)
    LOGGER.info(
        "Starting embedding diversity filter: samples=%d model=%s device=%s batch_size=%d max_similarity=%.4f",
        len(samples),
        args.embedding_model,
        args.embedding_device,
        args.embed_batch_size,
        args.max_similarity,
    )

    embedding_filter = EmbeddingDiversityFilter(
        model_name=args.embedding_model,
        device=args.embedding_device,
        embed_batch_size=args.embed_batch_size,
        max_length=args.max_length,
        max_similarity=args.max_similarity,
    )
    accepted, rejected = embedding_filter.filter_batch(samples)
    accepted = sort_by_combined_order(accepted)
    rejected = sort_by_combined_order(rejected)

    for sample in rejected:
        metadata = sample.get("generation_metadata", {})
        append_jsonl(
            rejected_path,
            {
                "name": sample.get("name", "?"),
                "combined_original_name": metadata.get("combined_original_name", ""),
                "source_file": metadata.get("combined_source_file", ""),
                "source_family": metadata.get("combined_source_family", ""),
                "error": "Filtered by embedding diversity",
            },
        )

    return accepted, rejected


def build_postprocess_command(args: argparse.Namespace, filtered_path: Path, output_path: Path, rejected_path: Path) -> List[str]:
    requested_families = normalize_requested_families(args.model_families)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "postprocess.py"),
        "--input-path",
        str(filtered_path),
        "--output-path",
        str(output_path),
        "--rejected-path",
        str(rejected_path),
        "--name-threshold",
        str(args.postprocess_name_threshold),
        "--seed",
        str(args.postprocess_seed),
    ]

    if args.skip_link_filter:
        command.append("--skip-link-filter")
    if args.skip_name_replace:
        command.append("--skip-name-replace")
    if args.enable_postprocess_sampling:
        command.extend(["--max-pct", str(args.postprocess_max_pct)])
    else:
        command.append("--skip-sampling")
    if args.min_model_family_pct is not None:
        command.extend(["--model-families", *requested_families])
        command.extend(["--min-model-family-pct", str(args.min_model_family_pct)])
    elif args.equalize_model_families:
        command.append("--equalize-model-families")
        command.extend(["--model-families", *requested_families])
    self_scope_pct = getattr(args, "postprocess_self_scope_pct", None)
    third_party_scope_pct = getattr(args, "postprocess_third_party_scope_pct", None)
    multi_subject_scope_pct = getattr(args, "postprocess_multi_subject_scope_pct", None)
    if self_scope_pct is not None:
        command.extend(["--self-scope-pct", str(self_scope_pct)])
    if third_party_scope_pct is not None:
        command.extend(["--third-party-scope-pct", str(third_party_scope_pct)])
    if multi_subject_scope_pct is not None:
        command.extend(["--multi-subject-scope-pct", str(multi_subject_scope_pct)])
    if args.verbose:
        command.append("--verbose")

    return command


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    outputs_dir = Path(args.outputs_dir)
    combined_output_path = Path(args.combined_output_path)
    filtered_output_path = Path(args.filtered_output_path)
    postprocessed_output_path = Path(args.postprocessed_output_path)
    embedding_rejected_path = Path(args.embedding_rejected_path)
    postprocess_rejected_path = Path(args.postprocess_rejected_path)

    if args.input_paths:
        input_paths = [Path(raw) for raw in args.input_paths]
    else:
        input_paths = discover_input_paths(outputs_dir)

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    ordered_inputs = order_paths(input_paths)
    LOGGER.info("Merge order:")
    for family, path, samples in ordered_inputs:
        LOGGER.info("  %s -> %s (%d samples)", family, path.name, len(samples))

    combined_samples, combined_counts = combine_samples(ordered_inputs)
    LOGGER.info("Combined %d samples", len(combined_samples))
    for family in FAMILY_ORDER:
        if combined_counts[family]:
            LOGGER.info("  %s: %d", family, combined_counts[family])

    write_json_atomic(combined_output_path, combined_samples)
    LOGGER.info("Wrote combined dataset to %s", combined_output_path)

    if args.stop_after_merge:
        return

    embedding_filter_inputs, embedding_order_strategy = plan_embedding_filter_inputs(
        combined_samples,
        args,
    )
    LOGGER.info(
        "Embedding filter scan order: %s",
        "scarcity-protecting order across requested families"
        if embedding_order_strategy == "protected"
        else "balance-aware round-robin across requested families"
        if embedding_order_strategy == "balanced"
        else "merged source order",
    )

    accepted, rejected = run_embedding_filter(
        embedding_filter_inputs,
        args,
        embedding_rejected_path,
    )
    write_json_atomic(filtered_output_path, accepted)
    LOGGER.info(
        "Embedding filter complete: accepted=%d rejected=%d output=%s",
        len(accepted),
        len(rejected),
        filtered_output_path,
    )

    accepted_counts = summarize_counts(accepted)
    rejected_counts = summarize_counts(rejected)
    for family in FAMILY_ORDER:
        if accepted_counts[family] or rejected_counts[family]:
            LOGGER.info(
                "  %s -> accepted=%d rejected=%d",
                family,
                accepted_counts[family],
                rejected_counts[family],
            )

    if args.stop_after_filter:
        return

    clear_output_file(postprocess_rejected_path)
    postprocess_cmd = build_postprocess_command(
        args,
        filtered_output_path,
        postprocessed_output_path,
        postprocess_rejected_path,
    )
    LOGGER.info("Running postprocess: %s", " ".join(postprocess_cmd))
    subprocess.run(postprocess_cmd, check=True, cwd=SCRIPT_DIR)

    postprocessed_samples = load_samples(postprocessed_output_path)
    repaired_emails = repair_postprocessed_user_emails(postprocessed_samples)
    if repaired_emails:
        write_json_atomic(postprocessed_output_path, postprocessed_samples)
        LOGGER.info(
            "Postprocess email consistency repair: updated %d user_email fields",
            repaired_emails,
        )
    LOGGER.info("Postprocess complete: %s", postprocessed_output_path)


if __name__ == "__main__":
    main()
