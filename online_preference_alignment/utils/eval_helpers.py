"""Shared utilities for offline evaluation scripts (vLLM, tokenization, IO)."""

from __future__ import annotations

import math


_EVAL_ENGINE_SEED_OFFSETS = {
    "reference": 1,
    "judge": 2,
    "trained_genrm": 4,
}


def build_eval_engine_seed(seed: int, *, purpose: str) -> int:
    """Return a reproducible but decorrelated seed for temporary eval engines."""
    offset = _EVAL_ENGINE_SEED_OFFSETS.get(purpose)
    if offset is None:
        supported = ", ".join(sorted(_EVAL_ENGINE_SEED_OFFSETS))
        raise ValueError(f"Unknown eval engine seed purpose {purpose!r}. Expected one of: {supported}.")
    return int(seed) + offset


def summarize_pairwise_dev_eval_metrics(
    candidate_first_outputs: list[str],
    reference_first_outputs: list[str],
    *,
    metric_prefix: str = "preference",
    candidate_label: str = "student",
    shorter_label: str | None = None,
    candidate_responses: list[str] | None = None,
    reference_responses: list[str] | None = None,
) -> dict[str, float]:
    """Summarize wandb-friendly metrics for a dual-order pairwise dev-eval pass.

    `candidate_first_outputs` are judge outputs where the policy completion is
    response 1; `reference_first_outputs` are judge outputs where the cached
    reference is response 1.

    If both `candidate_responses` and `reference_responses` are provided, also
    emit a length-conditioned slice (`{shorter_label}_shorter_*`, defaulting to
    `{candidate_label}_shorter_*`) covering the subset of samples where the
    candidate is strictly shorter than the reference by word count.
    """
    from judges.pairwise import combine_pairwise_judge_outputs

    if len(candidate_first_outputs) != len(reference_first_outputs):
        raise ValueError("candidate_first_outputs and reference_first_outputs must have the same length.")
    total_count = len(candidate_first_outputs)
    has_response_lengths = candidate_responses is not None and reference_responses is not None
    if has_response_lengths:
        if len(candidate_responses) != total_count or len(reference_responses) != total_count:
            raise ValueError(
                "candidate_responses and reference_responses must align with the judge output batches."
            )
        candidate_shorter_flags = [
            len(candidate.split()) < len(reference.split())
            for candidate, reference in zip(candidate_responses, reference_responses)
        ]
    else:
        candidate_shorter_flags = [False] * total_count
    shorter_total_count = sum(candidate_shorter_flags)
    parsed_scores: list[float] = []
    parsed_score_is_shorter: list[bool] = []
    invalid_count = 0
    for index, (candidate_first_output, reference_first_output) in enumerate(
        zip(candidate_first_outputs, reference_first_outputs)
    ):
        combined_score, _scores, _reasonings = combine_pairwise_judge_outputs(
            reference_first_output=reference_first_output,
            candidate_first_output=candidate_first_output,
            require_both=False,
        )
        if combined_score is None or math.isnan(combined_score):
            invalid_count += 1
            continue
        parsed_scores.append(combined_score)
        parsed_score_is_shorter.append(candidate_shorter_flags[index])
    valid_count = len(parsed_scores)
    prefix = metric_prefix.rstrip("/")
    shorter_scores = [
        score for score, is_shorter in zip(parsed_scores, parsed_score_is_shorter) if is_shorter
    ]
    shorter_valid_count = len(shorter_scores)
    shorter_mean_score = (
        sum(shorter_scores) / shorter_valid_count if shorter_valid_count > 0 else 0.0
    )
    shorter_metrics: dict[str, float] = {}
    if has_response_lengths:
        shorter_metric_label = candidate_label if shorter_label is None else shorter_label
        shorter_metrics = {
            f"{prefix}/{shorter_metric_label}_shorter_count": float(shorter_total_count),
            f"{prefix}/{shorter_metric_label}_shorter_mean_score": shorter_mean_score,
        }
    if valid_count <= 0:
        return {
            f"{prefix}/num_examples": float(total_count),
            f"{prefix}/valid_judgments": 0.0,
            f"{prefix}/invalid_judgments": float(invalid_count),
            f"{prefix}/{candidate_label}_mean_score": 0.0,
            f"{prefix}/{candidate_label}_win_rate": 0.0,
            f"{prefix}/tie_rate": 0.0,
            f"{prefix}/reference_win_rate": 0.0,
            **shorter_metrics,
        }
    mean_score = sum(parsed_scores) / valid_count
    return {
        f"{prefix}/num_examples": float(total_count),
        f"{prefix}/valid_judgments": float(valid_count),
        f"{prefix}/invalid_judgments": float(invalid_count),
        f"{prefix}/{candidate_label}_mean_score": mean_score,
        f"{prefix}/{candidate_label}_win_rate": sum(score > 0 for score in parsed_scores) / valid_count,
        f"{prefix}/tie_rate": sum(score == 0 for score in parsed_scores) / valid_count,
        f"{prefix}/reference_win_rate": sum(score < 0 for score in parsed_scores) / valid_count,
        **shorter_metrics,
    }


def summarize_pointwise_dev_eval_metrics(
    judge_outputs: list[str],
    *,
    metric_prefix: str = "privalign",
    auto_fail_mask: list[bool] | None = None,
) -> dict[str, float]:
    """Summarize wandb-friendly metrics for a single-pass leak/omit dev-eval."""
    from judges.pairwise import parse_pointwise_judge_output

    total_count = len(judge_outputs)
    if auto_fail_mask is None:
        auto_fail_mask = [False] * total_count
    if len(auto_fail_mask) != total_count:
        raise ValueError("auto_fail_mask must align with the judge output batch.")
    prefix = metric_prefix.rstrip("/")

    leak_count = 0
    omit_count = 0
    clean_count = 0
    valid_count = 0
    invalid_count = 0
    for judge_output, auto_failed in zip(judge_outputs, auto_fail_mask):
        if auto_failed:
            leak_count += 1
            omit_count += 1
            valid_count += 1
            continue
        leaks, omits, _reasoning = parse_pointwise_judge_output(judge_output)
        if leaks is None or omits is None:
            invalid_count += 1
            continue
        valid_count += 1
        if leaks:
            leak_count += 1
        if omits:
            omit_count += 1
        if not leaks and not omits:
            clean_count += 1
    if valid_count <= 0:
        return {
            f"{prefix}/num_examples": float(total_count),
            f"{prefix}/valid_judgments": 0.0,
            f"{prefix}/invalid_judgments": float(invalid_count),
            f"{prefix}/leak_rate": 0.0,
            f"{prefix}/omit_rate": 0.0,
            f"{prefix}/clean_rate": 0.0,
        }
    return {
        f"{prefix}/num_examples": float(total_count),
        f"{prefix}/valid_judgments": float(valid_count),
        f"{prefix}/invalid_judgments": float(invalid_count),
        f"{prefix}/leak_rate": leak_count / valid_count,
        f"{prefix}/omit_rate": omit_count / valid_count,
        f"{prefix}/clean_rate": clean_count / valid_count,
    }


def summarize_reference_word_count_metrics(
    reference_responses: list[str],
    *,
    metric_prefix: str = "preference",
) -> dict[str, float]:
    prefix = metric_prefix.rstrip("/")
    if not reference_responses:
        return {f"{prefix}/reference_word_count_mean": 0.0}
    word_counts = [len(response.split()) for response in reference_responses]
    return {
        f"{prefix}/reference_word_count_mean": sum(word_counts) / len(word_counts),
    }
