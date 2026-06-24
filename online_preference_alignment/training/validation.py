"""Finite-value validation helpers for the trainer stack.

Extracted from ``training.loss`` so loss computation stays focused on
objective math while validation concerns are reusable across subsystems.
"""

from __future__ import annotations

import math

import torch


class TrainerValidationMixin:
    """Numeric validation guards shared by all trainer subsystems."""

    def _ensure_finite_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        stage: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
    ) -> None:
        # `bool(finite_mask.all())` forces a D2H sync on every call. In the hot path we
        # skip the guard unless the user has opted into finite-tensor debugging.
        if not bool(getattr(self.args, "debug_finite_tensor_checks", False)):
            return
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return

        detached = tensor.detach()
        non_finite_count = int((~finite_mask).sum().item())
        finite_values = detached[finite_mask]
        stats = ""
        if finite_values.numel() > 0:
            stats = (
                f" finite_min={finite_values.min().item():.6f}"
                f" finite_max={finite_values.max().item():.6f}"
            )
        raise FloatingPointError(
            f"{self._phase_log_prefix(stage=stage, micro_batch_index=micro_batch_index, num_micro_batches=num_micro_batches)} "
            f"tensor={name} contains {non_finite_count} non-finite values.{stats}"
        )

    def _ensure_finite_metrics(self, metrics: dict[str, float], *, stage: str) -> None:
        non_finite_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if not math.isfinite(float(value))
        }
        if not non_finite_metrics:
            return

        rendered = ", ".join(
            f"{key}={value}"
            for key, value in sorted(non_finite_metrics.items())
        )
        raise FloatingPointError(
            f"{self._phase_log_prefix(stage=stage, micro_batch_index=None, num_micro_batches=None)} "
            f"metrics contain non-finite values: {rendered}"
        )
