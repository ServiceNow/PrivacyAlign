"""Coordinator-side helpers that turn trajectories into policy batches."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch

from training.batch_types import PolicyOptimizationBatch, TrajectoryBatch
from training.batching import materialize_packed_logprobs
from training.config import TrainingConfig
from experience.builders import PolicyExperienceBuilder, PolicyExperienceConfig
from experience.scorers import (
    TrajectoryScoreContext,
    TrajectoryScorer,
)


@dataclass
class PolicyBatchBuildResult:
    """One coordinator-side conversion from trajectory batch to policy batch."""

    batch: PolicyOptimizationBatch
    metrics: dict[str, float]
    sample_texts: dict[str, str] | None = None


class PolicyTrainingBatchBuilder:
    """Builds policy-training batches from generic trajectories."""

    def __init__(
        self,
        *,
        scorer: TrajectoryScorer,
        experience_builder: PolicyExperienceBuilder,
    ) -> None:
        self.scorer = scorer
        self.experience_builder = experience_builder

    def build_batch(
        self,
        trajectory_batch: TrajectoryBatch,
        *,
        score_context: TrajectoryScoreContext | None = None,
    ) -> PolicyBatchBuildResult:
        old_log_probs = materialize_packed_rollout_logprobs(trajectory_batch)
        score_outputs = self.scorer.score(trajectory_batch, context=score_context)
        raw_sequence_scores = score_outputs.scores.detach().float()
        policy_batch = self.experience_builder.build_batch(
            trajectory_batch,
            old_log_probs=old_log_probs,
            score_outputs=score_outputs,
        )
        metrics = summarize_policy_batch(
            policy_batch,
            raw_sequence_scores=raw_sequence_scores,
            score_outputs=score_outputs,
        )
        return PolicyBatchBuildResult(
            batch=policy_batch,
            metrics=metrics,
            sample_texts=build_policy_score_sample_texts(score_outputs),
        )


def build_policy_training_batch_builder(
    args: TrainingConfig,
    *,
    scorer: TrajectoryScorer,
) -> PolicyTrainingBatchBuilder:
    experience_builder = PolicyExperienceBuilder(
        PolicyExperienceConfig(
            reward_clip_min=args.policy_reward_clip_min,
            reward_clip_max=args.policy_reward_clip_max,
        )
    )
    return PolicyTrainingBatchBuilder(
        scorer=scorer,
        experience_builder=experience_builder,
    )


def materialize_packed_rollout_logprobs(trajectory_batch: TrajectoryBatch) -> torch.Tensor:
    """Turn packed vLLM sampled-token logprobs into a padded [B, T] tensor."""
    completion_mask = trajectory_batch.get("completion_mask")
    sequence_offsets = trajectory_batch.get("vllm_sequence_offsets")
    logprobs_flat = trajectory_batch.get("vllm_logprobs_flat")
    if not torch.is_tensor(completion_mask):
        raise KeyError("trajectory_batch must include completion_mask.")
    if not torch.is_tensor(sequence_offsets):
        raise KeyError("trajectory_batch must include vllm_sequence_offsets.")
    if not torch.is_tensor(logprobs_flat):
        raise KeyError("trajectory_batch must include vllm_logprobs_flat.")
    return materialize_packed_logprobs(logprobs_flat, sequence_offsets, completion_mask)


def build_policy_score_sample_texts(score_outputs) -> dict[str, str] | None:
    """Extract text artifacts from policy scorer metadata for sample logging."""
    if score_outputs is None:
        return None
    metadata = getattr(score_outputs, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    sample_texts: dict[str, str] = {}
    for metadata_key, sample_key in (
        ("sample_policy_judge_prompt", "policy_judge_prompt"),
        ("sample_policy_judge_prompt_swapped", "policy_judge_prompt_swapped"),
    ):
        value = metadata.get(metadata_key)
        if isinstance(value, str) and value:
            sample_texts[sample_key] = value
    return sample_texts or None


def summarize_policy_batch(
    policy_batch: PolicyOptimizationBatch,
    *,
    raw_sequence_scores: torch.Tensor | None = None,
    score_outputs=None,
) -> dict[str, float]:
    """Summarize coordinator-side policy batch signals for logging/debugging."""
    metrics: dict[str, float] = {}
    action_mask = policy_batch.get("action_mask")
    advantages = policy_batch.get("advantages")
    if torch.is_tensor(action_mask):
        action_count = action_mask.sum().item()
        metrics["policy/action_tokens"] = float(action_count)
    if torch.is_tensor(advantages) and torch.is_tensor(action_mask):
        mask = action_mask.to(dtype=advantages.dtype)
        denom = max(float(mask.sum().item()), 1.0)
        metrics["policy/advantage_mean"] = float((advantages * mask).sum().item() / denom)
    if torch.is_tensor(raw_sequence_scores) and raw_sequence_scores.numel() > 0:
        metrics["policy/sequence_score_mean"] = float(raw_sequence_scores.mean().item())
        metrics["policy/sequence_score_std"] = float(raw_sequence_scores.std(unbiased=False).item())
    if score_outputs is not None:
        length_penalties = score_outputs.metadata.get("length_penalties")
        if torch.is_tensor(length_penalties):
            length_penalty_tensor = length_penalties.detach().float().reshape(-1)
        elif isinstance(length_penalties, (list, tuple)):
            length_penalty_tensor = torch.tensor(length_penalties, dtype=torch.float32).reshape(-1)
        else:
            length_penalty_tensor = None
        if torch.is_tensor(length_penalty_tensor) and length_penalty_tensor.numel() > 0:
            nonzero_count = float((length_penalty_tensor > 0).sum().item())
            penalty_count = float(length_penalty_tensor.numel())
            metrics["policy/length_penalty_mean"] = float(length_penalty_tensor.mean().item())
            metrics["policy/length_penalty_max"] = float(length_penalty_tensor.max().item())
            metrics["policy/length_penalty_nonzero_count"] = nonzero_count
            metrics["policy/length_penalty_nonzero_rate"] = nonzero_count / penalty_count
        _add_optional_float_metric(
            metrics,
            score_outputs.metadata,
            metadata_key="undershort_penalty_mean",
            metric_key="policy/undershort_penalty_mean",
        )
        _add_optional_float_metric(
            metrics,
            score_outputs.metadata,
            metadata_key="undershort_penalty_rate",
            metric_key="policy/undershort_penalty_rate",
        )
        for scorer_prefix in ("judge", "genrm"):
            _add_optional_float_metric(
                metrics,
                score_outputs.metadata,
                metadata_key=f"{scorer_prefix}/undershort_penalty_mean",
                metric_key=f"policy/{scorer_prefix}_undershort_penalty_mean",
            )
            _add_optional_float_metric(
                metrics,
                score_outputs.metadata,
                metadata_key=f"{scorer_prefix}/undershort_penalty_rate",
                metric_key=f"policy/{scorer_prefix}_undershort_penalty_rate",
            )
        invalid_sequence_count = score_outputs.metadata.get("invalid_sequence_count")
        if invalid_sequence_count is not None:
            invalid_count = float(invalid_sequence_count)
            metrics["policy/invalid_sequence_count"] = invalid_count
            sequence_count = float(raw_sequence_scores.numel()) if torch.is_tensor(raw_sequence_scores) else 0.0
            metrics["policy/invalid_sequence_rate"] = (
                invalid_count / sequence_count if sequence_count > 0 else 0.0
            )
        format_violation_count = score_outputs.metadata.get("format_violation_count")
        if format_violation_count is not None:
            metrics["policy/format_violation_count"] = float(format_violation_count)
            format_violation_rate = score_outputs.metadata.get("format_violation_rate")
            if format_violation_rate is not None:
                metrics["policy/format_violation_rate"] = float(format_violation_rate)
            violation_reasons = score_outputs.metadata.get("format_violation_reasons")
            if isinstance(violation_reasons, list):
                for reason, count in Counter(
                    str(reason) for reason in violation_reasons if reason is not None
                ).items():
                    reason_key = re.sub(r"[^a-zA-Z0-9_]+", "_", reason).strip("_").lower()
                    if reason_key:
                        metrics[f"policy/format_violation/{reason_key}"] = float(count)
        for metadata_key in (
            "pair_accuracy",
            "leak_accuracy",
        ):
            _add_optional_float_metric(
                metrics,
                score_outputs.metadata,
                metadata_key=metadata_key,
                metric_key=f"policy/{metadata_key}",
            )
    return metrics


def _add_optional_float_metric(
    metrics: dict[str, float],
    metadata: dict,
    *,
    metadata_key: str,
    metric_key: str,
) -> None:
    value = metadata.get(metadata_key)
    if value is not None:
        metrics[metric_key] = float(value)
