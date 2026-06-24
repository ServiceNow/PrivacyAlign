"""Build gen-RM training rows for the Privalign dataset (Phase A).

For each Privalign row in the train split of ``ServiceNow/PrivacyAlign``:

1. Compute the signed pairwise target from the mean per-annotator vote score::

       agent_a          = -2
       agent_a_slightly = -1
       tie              =  0
       agent_b_slightly = +1
       agent_b          = +2

   ``unsure`` votes are excluded from the denominator. The resulting
   ``target_score`` is clipped to [-2, +2]. Positive ``target_score`` ⇒
   Response B was the consensus preference for the ``(response_a, response_b)``
   ordering.

2. Keep tie/near-tie rows by default so the gen-RM learns to emit ``Score: 0``.
   Pass ``--drop_near_tie`` to remove rows whose consensus is ambiguous
   (majority_bucket is tie/neutral AND |target_score| < 1).

3. Emit BOTH orderings (ab, ba). For ``ba`` the target sign flips and
   ``preferred_slot`` swaps so the gen-RM sees each pair "both ways".

4. Apply the deterministic hash-bucket train/dev split from
   ``_privalign_row_belongs_to_split`` (mirrors the Privalign loader).

Each emitted row::

    {
      "context": [{"role": "user", "content": <rendered privalign_genrm_pairwise>}],
      "ranking_demo": {
        "demo_type": "privalign_pairwise_genrm",
        "source_dataset": "privalign-dataset",
        "prompt_id": "<row.id>__<ab|ba>",
        "preferred_slot": 1 or 2,
        "target_score": float -2..+2,             # consumed by PairwiseMarginScorer soft-target path
        "target_signed_margin": int -2..+2,       # legacy rounded target for compatibility
        ...
      }
    }

Output: ``outputs/genrm-privalign-qwen3.5-4b/training/{train,dev}.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from data_loaders import load_prompt
from data_loaders.preference import (
    PRIVALIGN_VALIDATION_BUCKETS,
    PRIVALIGN_VALIDATION_BUCKET_COUNT,
    _format_privalign_memories,
    _format_privalign_response,
)


GENRM_PROMPT_TEMPLATE_NAME = "privalign_genrm_pairwise"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _privalign_split_for_row_id(row_id: object) -> str:
    """Return 'validation' or 'train' for a Privalign row id (matches loader)."""
    digest = hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % PRIVALIGN_VALIDATION_BUCKET_COUNT
    return "validation" if bucket in PRIVALIGN_VALIDATION_BUCKETS else "train"


def _round_half_away_from_zero(value: float) -> int:
    if value > 0:
        return math.floor(value + 0.5)
    if value < 0:
        return math.ceil(value - 0.5)
    return 0


def _compute_target_score(preference_counts: dict | None) -> tuple[float, float, int]:
    """Return (clipped_mean_target, raw_vote_sum, counted_vote_count).

    Strong preferences are scored as +/-2, slight preferences as +/-1, and ties
    as 0. ``unsure`` votes are treated as missing preference signal and excluded
    from the average, so target magnitude reflects average annotator preference
    strength rather than the number of annotations collected.
    """
    counts = preference_counts if isinstance(preference_counts, dict) else {}
    a = float(counts.get("agent_a", 0) or 0)
    a_slight = float(counts.get("agent_a_slightly", 0) or 0)
    b = float(counts.get("agent_b", 0) or 0)
    b_slight = float(counts.get("agent_b_slightly", 0) or 0)
    tie = float(counts.get("tie", 0) or 0)
    raw = (2.0 * b + b_slight) - (2.0 * a + a_slight)
    vote_count = int(a + a_slight + b + b_slight + tie)
    if vote_count <= 0:
        return 0.0, raw, 0
    target = raw / float(vote_count)
    target = float(max(-2.0, min(2.0, target)))
    return target, raw, vote_count


def _compute_target_signed_margin(preference_counts: dict | None) -> tuple[int, float]:
    """Legacy rounded target helper retained for compatibility."""
    target_score, raw, _vote_count = _compute_target_score(preference_counts)
    return int(max(-2, min(2, _round_half_away_from_zero(target_score)))), raw


def _soft_gold_leak(
    annotations: list,
    *,
    slot_key: str,
) -> tuple[float | None, int]:
    """Return (gold_leak, n_annotators) for one response slot."""
    leak_votes: list[float] = []
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            labels = ann.get(slot_key)
            if not isinstance(labels, dict):
                continue
            leak = labels.get("leaks")
            if isinstance(leak, bool):
                leak_votes.append(1.0 if leak else 0.0)
    if not leak_votes:
        return None, 0
    return sum(leak_votes) / len(leak_votes), len(leak_votes)


def _render_rate_prompt(*, template: str, row: dict, response1: str, response2: str) -> str:
    user_instruction = str(row.get("user_instruction") or "").strip()
    memories = _format_privalign_memories(row.get("memories")) or "(no memories provided)"
    executable_trajectory = str(row.get("trajectory") or "").strip() or "(no prior tool use)"
    content = template.format(
        user_instruction=user_instruction,
        memories=memories,
        executable_trajectory=executable_trajectory,
        eval_response1=response1,
        eval_response2=response2,
    )
    return re.sub(r"\n{3,}", "\n\n", content)


def build_privalign_genrm_rows(
    input_path: str | Path,
    *,
    drop_near_tie: bool = False,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Return pairwise Privalign gen-RM rows split into train/validation."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Privalign input JSONL not found at {input_path}")

    template = load_prompt(GENRM_PROMPT_TEMPLATE_NAME)
    rows = _read_jsonl(input_path)

    per_split: dict[str, list[dict]] = {"train": [], "validation": []}
    per_split_summary: dict[str, dict] = {
        split: {
            "input_rows": 0,
            "kept_pair_rows": 0,
            "dropped_near_tie": 0,
            "dropped_missing_responses": 0,
            "target_histogram": {str(t): 0 for t in (-2, -1, 0, 1, 2)},
            "preferred_slot_counts": {"1": 0, "2": 0},
            "missing_leak_label_rows": 0,
            # Averaged preference scores can be neutral while `majority_bucket`
            # still gives the row a direction. Track how often the two views
            # disagree so we can quantify in the writeup.
            "tie_disagrees_with_majority_bucket": 0,
        }
        for split in ("train", "validation")
    }

    for row in rows:
        row_id = row.get("id")
        split = _privalign_split_for_row_id(row_id)
        summary = per_split_summary[split]
        summary["input_rows"] += 1

        target_score, raw, vote_count = _compute_target_score(row.get("preference_counts"))
        target_signed_margin = int(
            max(-2, min(2, _round_half_away_from_zero(target_score)))
        )
        majority_bucket = str(row.get("majority_bucket") or "").lower()

        if drop_near_tie and majority_bucket in {"tie", "n"} and abs(target_score) < 1.0:
            summary["dropped_near_tie"] += 1
            continue

        # Disagreement = the averaged target is neutral but the dataset's
        # `majority_bucket` field gives it a direction. Counts the source row
        # (each row produces 2 training rows after both orderings).
        if target_score == 0 and majority_bucket in {"agent_a", "agent_b", "a", "b"}:
            summary["tie_disagrees_with_majority_bucket"] += 1

        response_a_text = _format_privalign_response(row.get("response_a"))
        response_b_text = _format_privalign_response(row.get("response_b"))
        if not response_a_text.strip() or not response_b_text.strip():
            summary["dropped_missing_responses"] += 1
            continue
        annotations = row.get("annotations")
        gold_leak_a, _ = _soft_gold_leak(annotations, slot_key="response_a_labels")
        gold_leak_b, _ = _soft_gold_leak(annotations, slot_key="response_b_labels")
        has_leak_labels = gold_leak_a is not None and gold_leak_b is not None
        if not has_leak_labels:
            summary["missing_leak_label_rows"] += 1

        # Emit BOTH orderings. For "ab" the row's natural sign holds; for "ba"
        # the target sign flips because response_a is now in slot 2.
        for (
            ordering,
            response1,
            response2,
            signed_target_score_for_ordering,
            signed_margin_for_ordering,
            leak1,
            leak2,
        ) in (
            (
                "ab",
                response_a_text,
                response_b_text,
                target_score,
                target_signed_margin,
                gold_leak_a,
                gold_leak_b,
            ),
            (
                "ba",
                response_b_text,
                response_a_text,
                -target_score,
                -target_signed_margin,
                gold_leak_b,
                gold_leak_a,
            ),
        ):
            # preferred_slot = slot holding the consensus-preferred response.
            # When target score > 0, slot 2 is preferred; <0 ⇒ slot 1.
            # Zero-target rows are retained by default; arbitrarily place
            # preferred_slot=1 so dev-eval can still compute a sign metric
            # (treated as "Response 1 better or equal").
            if signed_target_score_for_ordering > 0:
                preferred_slot = 2
            elif signed_target_score_for_ordering < 0:
                preferred_slot = 1
            else:
                preferred_slot = 1

            summary["target_histogram"][str(signed_margin_for_ordering)] += 1
            summary["preferred_slot_counts"][str(preferred_slot)] += 1

            rate_content = _render_rate_prompt(
                template=template,
                row=row,
                response1=response1,
                response2=response2,
            )

            ranking_demo = {
                "demo_type": "privalign_pairwise_genrm",
                "source_dataset": "privalign-dataset",
                "prompt_id": f"privalign:{row_id}__{ordering}",
                "responses": {},
                "assessments": [],
                "preferred_slot": int(preferred_slot),
                "target_score": float(signed_target_score_for_ordering),
                "target_signed_margin": int(signed_margin_for_ordering),
                "target_vote_sum": float(raw if ordering == "ab" else -raw),
                "target_vote_count": int(vote_count),
                "ordering": ordering,
                "original_row_id": row_id,
                "majority_bucket": majority_bucket,
                "raw_margin": float(raw if ordering == "ab" else -raw),
            }
            if has_leak_labels:
                ranking_demo["gold_leak_response1"] = float(leak1)
                ranking_demo["gold_leak_response2"] = float(leak2)

            per_split[split].append(
                {
                    "context": [{"role": "user", "content": rate_content}],
                    "ranking_demo": ranking_demo,
                }
            )
            summary["kept_pair_rows"] += 1

    return per_split, per_split_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default=None,
        help=(
            "Privalign train.jsonl (the hash-bucket split carves train/dev out of this "
            "file). Defaults to the train split of ServiceNow/PrivacyAlign, downloaded "
            "from the Hugging Face Hub."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/genrm-privalign-qwen3.5-4b/training",
        help="Where to write the gen-RM training/dev JSONLs.",
    )
    parser.add_argument(
        "--drop_near_tie",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If majority_bucket is tie/neutral AND |target_score|<1, drop the row "
            "instead of keeping it as a target=0 row. Default: keep ties."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input_path
    if input_path is None:
        from data_loaders.preference import _resolve_privalign_jsonl_path

        input_path = _resolve_privalign_jsonl_path("privalign-dataset", split="train")
    per_split, per_split_summary = build_privalign_genrm_rows(
        Path(input_path),
        drop_near_tie=args.drop_near_tie,
    )

    for split, written_rows in per_split.items():
        out_path = output_dir / f"{split}.jsonl"
        _write_jsonl(out_path, written_rows)
        print(f"[{split}] wrote {len(written_rows)} rows -> {out_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(per_split_summary, indent=2, sort_keys=True))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
