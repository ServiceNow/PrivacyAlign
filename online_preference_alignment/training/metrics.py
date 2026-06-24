"""Training-step metric accumulators for policy optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _detach_scalar(tensor: torch.Tensor) -> torch.Tensor:
    detached = tensor.detach()
    if detached.dim() != 0:
        detached = detached.reshape(())
    return detached.to(dtype=torch.float32)


class _DeferredSyncMixin:
    """Accumulate tensor-valued scalars on-device and sync them once in `finalize()`.

    Each accumulator holds two views of every sum: a Python float (canonical,
    read by step-end aggregators) and an optional 0-d device tensor used while
    the optimizer step is still in progress. `_accumulate_scalar` routes tensor
    inputs into the deferred buffer so we don't force a D2H sync per micro-batch.
    `finalize()` pulls those buffers once before the aggregator reads the float
    fields.
    """

    _tensor_sums: dict[str, torch.Tensor]

    def _accumulate_scalar(self, field_name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            detached = _detach_scalar(value)
            existing = self._tensor_sums.get(field_name)
            self._tensor_sums[field_name] = detached if existing is None else existing + detached
        else:
            setattr(self, field_name, getattr(self, field_name) + float(value))

    def finalize(self) -> None:
        """Drain deferred tensor sums into their float fields in a single sync each."""
        for field_name, tensor_sum in self._tensor_sums.items():
            setattr(
                self,
                field_name,
                getattr(self, field_name) + float(tensor_sum.item()),
            )
        self._tensor_sums.clear()


@dataclass
class PolicyOptimizationTrainingStepMetrics(_DeferredSyncMixin):
    """Accumulates actor-only policy-optimization metrics across one optimizer step."""

    policy_loss_sum: float = 0.0
    reference_kl_sum: float = 0.0
    action_token_count: float = 0.0
    approx_kl_sum: float = 0.0
    advantage_sum: float = 0.0
    old_log_prob_sum: float = 0.0
    grad_norm: float | None = None
    _tensor_sums: dict[str, torch.Tensor] = field(default_factory=dict)

    def accumulate(self, loss_components: dict[str, Any]) -> None:
        """Add one micro-batch's policy metrics to the running totals."""
        self._accumulate_scalar("policy_loss_sum", loss_components["policy_loss_sum"])
        self._accumulate_scalar("reference_kl_sum", loss_components["reference_kl_sum"])
        self._accumulate_scalar(
            "action_token_count", loss_components.get("action_token_count", 0.0)
        )
        self._accumulate_scalar("approx_kl_sum", loss_components["approx_kl_sum"])
        self._accumulate_scalar("advantage_sum", loss_components["advantage_sum"])
        self._accumulate_scalar("old_log_prob_sum", loss_components["old_log_prob_sum"])
