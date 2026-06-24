"""Training-objective interfaces shared by RL objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from training.batching import BatchFieldLayout


@dataclass
class ObjectiveMicroBatchResult:
    """One objective evaluation for one training micro-batch.

    `loss` is the scalar used for backward().
    `components` are objective-specific tensors or floats accumulated into
    step-level metrics.
    `log_details` is a flat dictionary rendered in phase-progress logs.
    """

    loss: torch.Tensor
    components: dict[str, torch.Tensor | float]
    log_details: dict[str, str]


@dataclass(frozen=True)
class ObjectiveRolloutRequirements:
    """Coordinator-side rollout requirements for one objective.

    This keeps vLLM diagnostics near the objective definition instead of
    scattering those decisions through the Ray driver.
    """

    include_sampled_logprobs: bool = False


class TrainingObjective(Protocol):
    """Interface implemented by each optimization objective.

    This follows the same high-level separation used in OpenRLHF:
    trajectory construction happens outside the objective, while the objective
    owns normalization, loss construction, and metric reduction.
    """

    name: str

    def training_batch_layout(self) -> BatchFieldLayout:
        """Return the batch layout this objective expects on workers."""

    def rollout_requirements(self, args: Any) -> ObjectiveRolloutRequirements:
        """Describe which rollout diagnostics the coordinator must collect."""

    def requires_reference_model(self, args: Any) -> bool:
        """Return whether workers must own a frozen reference model."""

    def build_metrics_accumulator(self) -> Any:
        """Return an accumulator object used across one optimizer step."""

    def count_normalization_items(
        self,
        trainer: Any,
        batch: dict[str, torch.Tensor],
    ) -> int:
        """Count the local optimization items for one micro-batch."""

    def compute_micro_batch(
        self,
        trainer: Any,
        batch: dict[str, torch.Tensor],
        *,
        normalization: torch.Tensor,
        stage: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
    ) -> ObjectiveMicroBatchResult:
        """Compute the backward loss and objective-specific logging payload."""

    def aggregate_step_metrics(
        self,
        trainer: Any,
        accum: Any,
        *,
        global_normalization_count: int,
        stage: str,
    ) -> dict[str, float]:
        """Reduce one optimizer step of accumulated metrics into scalars."""
