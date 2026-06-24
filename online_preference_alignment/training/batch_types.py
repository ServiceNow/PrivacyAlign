"""Shared serialized batch types."""

from __future__ import annotations

from typing import TypedDict

import torch


class TrajectoryBatch(TypedDict, total=False):
    """Objective-agnostic prompt/completion tensors plus rollout diagnostics.

    Future RL objectives can extend this base shape with sequence-level rewards,
    advantages, or old log-probs without reworking the batch sharding utilities.
    """

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    completion_texts: list[str]
    num_prompts: int
    vllm_sequence_offsets: torch.Tensor
    vllm_logprobs_flat: torch.Tensor


class PolicyOptimizationBatch(TrajectoryBatch, total=False):
    """Trajectory tensors commonly needed by PPO/REINFORCE-style objectives.

    The field names are intentionally close to OpenRLHF's `Experience`
    structure so reward-model or LLM-judge RL can reuse the same vocabulary:
    action masks, old log-probs, advantages, and sequence-level diagnostics.
    """

    action_mask: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    sequence_scores: torch.Tensor
    valid_sequence_mask: torch.Tensor
