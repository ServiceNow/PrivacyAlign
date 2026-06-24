"""Reusable builders that turn trajectories into policy-optimization batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.batch_types import PolicyOptimizationBatch, TrajectoryBatch
from experience.scorers import SequenceScoreOutputs


_TRAJECTORY_BATCH_KEYS = (
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_mask",
    "num_prompts",
    "vllm_sequence_offsets",
    "vllm_logprobs_flat",
)

@dataclass(frozen=True)
class PolicyExperienceConfig:
    """Controls how sequence scores become token-level advantages."""

    reward_clip_min: float = -1000.0
    reward_clip_max: float = 1000.0

    def __post_init__(self) -> None:
        if self.reward_clip_min > self.reward_clip_max:
            raise ValueError("reward_clip_min must be <= reward_clip_max.")


class PolicyExperienceBuilder:
    """Builds OpenRLHF-like policy batches from trajectory tensors.

    This keeps advantage construction out of the trainer loop so scalar reward
    models and LLM judges can produce the same downstream batch.
    """

    def __init__(self, config: PolicyExperienceConfig | None = None) -> None:
        self.config = PolicyExperienceConfig() if config is None else config

    def build_batch(
        self,
        trajectory_batch: TrajectoryBatch,
        *,
        old_log_probs: torch.Tensor,
        score_outputs: SequenceScoreOutputs | None = None,
        sequence_scores: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> PolicyOptimizationBatch:
        completion_ids = _require_tensor(trajectory_batch, "completion_ids")
        completion_mask = _require_tensor(trajectory_batch, "completion_mask")
        num_sequences = int(completion_ids.size(0))
        sequence_shape = completion_ids.shape

        _validate_tensor_shape(old_log_probs, sequence_shape, "old_log_probs", "completion_ids")
        compute_device = old_log_probs.device

        resolved_action_mask = completion_mask.bool() if action_mask is None else action_mask.bool()
        resolved_action_mask = resolved_action_mask.to(device=compute_device)
        _validate_tensor_shape(resolved_action_mask, sequence_shape, "action_mask", "completion_ids")
        empty_action_sequence_mask = ~resolved_action_mask.any(dim=1)
        invalid_sequence_mask = (
            empty_action_sequence_mask.clone()
            if bool(empty_action_sequence_mask.any())
            else None
        )

        if score_outputs is not None:
            score_outputs.validate(num_sequences=num_sequences)
            if sequence_scores is not None:
                raise ValueError("Pass either score_outputs or sequence_scores, not both.")
            sequence_scores = score_outputs.scores
            score_invalid_sequence_mask = _resolve_invalid_sequence_mask(
                score_outputs,
                num_sequences=num_sequences,
                device=compute_device,
            )
            invalid_sequence_mask = _merge_invalid_sequence_masks(
                invalid_sequence_mask,
                score_invalid_sequence_mask,
            )
        if sequence_scores is None:
            raise ValueError("PolicyExperienceBuilder requires sequence_scores or score_outputs.")
        if not torch.is_tensor(sequence_scores):
            raise TypeError("sequence_scores must be a torch.Tensor.")
        if sequence_scores.ndim != 1 or sequence_scores.size(0) != num_sequences:
            raise ValueError(
                "sequence_scores must have shape [num_sequences] aligned with completion_ids."
            )
        sequence_scores = sequence_scores.to(device=compute_device)
        sequence_scores = sequence_scores.clamp(
            min=self.config.reward_clip_min,
            max=self.config.reward_clip_max,
        )
        sequence_scores = _apply_prompt_group_baseline(
            sequence_scores,
            num_prompts=_require_num_prompts(trajectory_batch),
            invalid_sequence_mask=invalid_sequence_mask,
        )

        valid_sequence_mask = (
            torch.ones(num_sequences, device=compute_device, dtype=torch.bool)
            if invalid_sequence_mask is None
            else ~invalid_sequence_mask
        )
        effective_action_mask = resolved_action_mask & valid_sequence_mask.unsqueeze(1)
        advantages = self._broadcast_sequence_scores_to_tokens(
            sequence_scores=sequence_scores,
            action_mask=effective_action_mask,
            invalid_sequence_mask=invalid_sequence_mask,
        )

        policy_batch: PolicyOptimizationBatch = {
            key: trajectory_batch[key]
            for key in _TRAJECTORY_BATCH_KEYS
            if key in trajectory_batch
        }
        policy_batch["action_mask"] = effective_action_mask
        policy_batch["old_log_probs"] = old_log_probs
        policy_batch["advantages"] = advantages
        policy_batch["sequence_scores"] = sequence_scores
        policy_batch["valid_sequence_mask"] = valid_sequence_mask
        return policy_batch

    def _broadcast_sequence_scores_to_tokens(
        self,
        *,
        sequence_scores: torch.Tensor,
        action_mask: torch.Tensor,
        invalid_sequence_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Turn per-sequence scores into per-token advantages.

        The reward signal is sequence-level, so the (clipped, group-centered)
        score is broadcast unchanged to every action token. Following Dr. GRPO,
        we skip std normalization: the advantage scale tracks the reward scale
        directly, and degenerate batches (all rollouts tied) cannot blow up.
        """
        valid_sequence_mask = action_mask.any(dim=1)
        if invalid_sequence_mask is not None:
            valid_sequence_mask = valid_sequence_mask & ~invalid_sequence_mask
        score_per_sequence = sequence_scores * valid_sequence_mask.to(dtype=sequence_scores.dtype)

        action_mask_f = action_mask.to(dtype=sequence_scores.dtype)
        per_token = score_per_sequence.unsqueeze(1) * action_mask_f
        return per_token


def _require_tensor(batch: TrajectoryBatch, key: str) -> torch.Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"trajectory_batch must include tensor key {key!r}.")
    return value


def _require_num_prompts(batch: TrajectoryBatch) -> int:
    value = batch.get("num_prompts")
    if not isinstance(value, int) or value <= 0:
        raise KeyError("trajectory_batch must include a positive integer num_prompts.")
    return value


def _apply_prompt_group_baseline(
    sequence_scores: torch.Tensor,
    *,
    num_prompts: int,
    invalid_sequence_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center scores within each prompt group for REINFORCE++-baseline."""
    if num_prompts <= 0:
        raise ValueError("num_prompts must be > 0.")
    if sequence_scores.numel() % num_prompts != 0:
        raise ValueError(
            "sequence_scores length must be divisible by num_prompts."
        )
    group_size = sequence_scores.numel() // num_prompts
    if group_size <= 1:
        raise ValueError(
            "Prompt-group baseline requires more than one sampled completion per prompt."
        )
    grouped_scores = sequence_scores.reshape(num_prompts, group_size)
    if invalid_sequence_mask is None:
        centered_scores = grouped_scores - grouped_scores.mean(dim=1, keepdim=True)
        return centered_scores.reshape(-1)
    grouped_invalid_mask = invalid_sequence_mask.reshape(num_prompts, group_size)
    grouped_valid_mask = ~grouped_invalid_mask
    valid_counts = grouped_valid_mask.sum(dim=1, keepdim=True)
    safe_valid_counts = valid_counts.clamp_min(1).to(dtype=grouped_scores.dtype)
    valid_score_sums = (grouped_scores * grouped_valid_mask.to(dtype=grouped_scores.dtype)).sum(
        dim=1,
        keepdim=True,
    )
    valid_means = valid_score_sums / safe_valid_counts
    centered_scores = torch.where(
        grouped_valid_mask,
        grouped_scores - valid_means,
        torch.zeros_like(grouped_scores),
    )
    return centered_scores.reshape(-1)


def _resolve_invalid_sequence_mask(
    score_outputs: SequenceScoreOutputs,
    *,
    num_sequences: int,
    device: torch.device,
) -> torch.Tensor | None:
    invalid_sequence_mask = score_outputs.metadata.get("invalid_sequence_mask")
    if invalid_sequence_mask is None:
        return None
    if not torch.is_tensor(invalid_sequence_mask):
        raise TypeError("score_outputs.metadata['invalid_sequence_mask'] must be a torch.Tensor.")
    if invalid_sequence_mask.ndim != 1 or invalid_sequence_mask.size(0) != num_sequences:
        raise ValueError(
            "invalid_sequence_mask must have shape [num_sequences] aligned with completion_ids."
        )
    return invalid_sequence_mask.to(device=device, dtype=torch.bool)


def _merge_invalid_sequence_masks(
    existing_mask: torch.Tensor | None,
    new_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if existing_mask is None:
        return new_mask
    if new_mask is None:
        return existing_mask
    return existing_mask | new_mask


def _validate_tensor_shape(
    value: torch.Tensor,
    expected_shape: torch.Size,
    value_name: str,
    reference_name: str,
) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{value_name} must be a torch.Tensor.")
    if value.shape != expected_shape:
        raise ValueError(
            f"{value_name} must match {reference_name} shape "
            f"({tuple(value.shape)} != {tuple(expected_shape)})."
        )
