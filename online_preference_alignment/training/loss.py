"""Policy-objective execution, validation checks, and metric aggregation.

The heavy model-forward infrastructure lives in ``training.forward``
(``TrainerForwardMixin``), and checkpoint I/O lives in ``training.checkpoint``
(``TrainerCheckpointMixin``).  This module owns the objective-level training
loop and the RL reference-model regularization helpers.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from training.forward import TrainerForwardMixin
from training.checkpoint import TrainerCheckpointMixin
from training.phase_logging import TrainerPhaseLoggingMixin
from training.validation import TrainerValidationMixin
from utils.trainer_utils import is_distributed_ready


class TrainerLossMixin(
    TrainerForwardMixin,
    TrainerCheckpointMixin,
    TrainerPhaseLoggingMixin,
    TrainerValidationMixin,
):
    """Training-objective execution plus policy numerical helpers."""

    # ------------------------------------------------------------------
    # Micro-batch diagnostics
    # ------------------------------------------------------------------

    def _tensor_token_stats(self, batch: dict[str, torch.Tensor], mask_key: str) -> tuple[int, int]:
        mask = batch.get(mask_key)
        if not torch.is_tensor(mask) or mask.ndim != 2:
            return 0, 0
        token_sums = mask.sum(dim=1)
        total_tokens = int(token_sums.sum().item())
        max_tokens = int(token_sums.max().item()) if token_sums.numel() > 0 else 0
        return total_tokens, max_tokens

    def _cuda_memory_stats(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {}
        mb = 1024 * 1024
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        free, total = torch.cuda.mem_get_info(self.device)
        device_used = total - free
        return {
            "cuda_allocated_mb": allocated // mb,
            "cuda_reserved_mb": reserved // mb,
            "cuda_device_used_mb": device_used // mb,
            "cuda_device_free_mb": free // mb,
            "cuda_external_mb": max(0, device_used - reserved) // mb,
        }

    def _maybe_get_global_grad_norm(self) -> float | None:
        deepspeed_engine = getattr(self, "deepspeed_engine", None)
        engine = deepspeed_engine if deepspeed_engine is not None else self.model
        get_global_grad_norm = getattr(engine, "get_global_grad_norm", None)
        if not callable(get_global_grad_norm):
            return None
        try:
            value = float(get_global_grad_norm())
        except Exception:
            return None
        if not math.isfinite(value):
            return None
        return value

    def _backward_step_debug_details(
        self,
        micro_batch: dict[str, torch.Tensor],
        *,
        will_update_parameters: bool,
    ) -> dict[str, Any]:
        prompt_tokens, max_prompt_tokens = self._tensor_token_stats(micro_batch, "prompt_mask")
        completion_tokens, max_completion_tokens = self._tensor_token_stats(micro_batch, "completion_mask")
        loss_tokens = self._require_training_objective().count_normalization_items(self, micro_batch)
        details: dict[str, Any] = {
            "rank": self.args.rank,
            "sequences": micro_batch["completion_mask"].size(0),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "loss_tokens": loss_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "max_completion_tokens": max_completion_tokens,
            "optimizer_step_boundary": str(bool(will_update_parameters)).lower(),
        }
        details.update(self._cuda_memory_stats())
        return details

    def _gradient_average_world_size(self) -> int:
        for candidate in (getattr(self, "deepspeed_engine", None), getattr(self, "model", None)):
            dp_world_size = getattr(candidate, "dp_world_size", None)
            if isinstance(dp_world_size, int) and dp_world_size > 0:
                return dp_world_size
        if is_distributed_ready():
            return dist.get_world_size()
        return 1

    def _distributed_debug_barrier(
        self,
        *,
        stage: str,
        phase: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if (
            not self.args.log_phase_progress
            or not bool(getattr(self.args, "debug_distributed_phase_barriers", False))
            or not is_distributed_ready()
        ):
            return
        start_details = {
            "rank": self.args.rank,
            "world_size": dist.get_world_size(),
        }
        if details:
            start_details.update(details)
        self._log_phase_event(
            stage=stage,
            phase=phase,
            status="start",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=start_details,
        )
        dist.barrier()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._log_phase_event(
            stage=stage,
            phase=phase,
            status="done",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details={
                "rank": self.args.rank,
                **self._cuda_memory_stats(),
            },
        )

    # ------------------------------------------------------------------
    # Distributed scalar helper
    # ------------------------------------------------------------------

    def _to_device_scalar(self, v: float) -> torch.Tensor:
        return torch.tensor(v, device=self.device, dtype=torch.float32)

    def _distributed_sum_scalar(self, value: float) -> torch.Tensor:
        """Convenience: convert a float to a device tensor and all-reduce sum."""
        return self._distributed_sum(self._to_device_scalar(value))

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _require_training_objective(self):
        objective = getattr(self, "training_objective", None)
        if objective is None:
            raise RuntimeError(
                "Trainer is missing training_objective. "
                "Initialize it before calling _run_training_step()."
            )
        return objective

    def _run_training_step(self, micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        if not micro_batches:
            raise ValueError("Training step received no micro-batches.")
        if len(micro_batches) != self.args.gradient_accumulation_steps:
            raise RuntimeError(
                "Each optimizer step must contain exactly gradient_accumulation_steps micro-batches "
                f"({self.args.gradient_accumulation_steps}), but got {len(micro_batches)}. "
                "Check the train dataloader batch sizing."
            )

        objective = self._require_training_objective()
        count_micro_batches = getattr(objective, "count_normalization_items_for_micro_batches", None)
        if callable(count_micro_batches):
            local_normalization_count = count_micro_batches(self, micro_batches)
        else:
            local_normalization_count = sum(
                objective.count_normalization_items(self, micro_batch)
                for micro_batch in micro_batches
            )
        global_normalization_count = self._distributed_sum_int(local_normalization_count)
        if global_normalization_count <= 0:
            raise ValueError(
                f"Optimizer batch produced no valid normalization items for objective={objective.name!r}."
            )

        normalization = torch.tensor(global_normalization_count, device=self.device, dtype=torch.float32)
        accum = objective.build_metrics_accumulator()

        for micro_batch_index, micro_batch in enumerate(micro_batches, start=1):
            self._log_phase_event(
                stage="train",
                phase="micro_batch",
                status="start",
                micro_batch_index=micro_batch_index,
                num_micro_batches=len(micro_batches),
            )
            micro_batch = self._move_batch_to_device(micro_batch)
            objective_result = objective.compute_micro_batch(
                self,
                micro_batch,
                normalization=normalization,
                stage="train",
                micro_batch_index=micro_batch_index,
                num_micro_batches=len(micro_batches),
            )
            accum.accumulate(objective_result.components)

            self._log_phase_event(
                stage="train",
                phase="loss",
                status="computed",
                micro_batch_index=micro_batch_index,
                num_micro_batches=len(micro_batches),
                details=objective_result.log_details,
            )
            self._ensure_finite_tensor(
                "total_loss",
                objective_result.loss,
                stage="train",
                micro_batch_index=micro_batch_index,
                num_micro_batches=len(micro_batches),
            )
            grad_norm = self._backward_and_step(
                objective_result.loss, micro_batch, micro_batch_index, len(micro_batches),
            )
            if hasattr(accum, "grad_norm"):
                accum.grad_norm = grad_norm
            del objective_result, micro_batch
            self._log_phase_event(
                stage="train",
                phase="micro_batch",
                status="done",
                micro_batch_index=micro_batch_index,
                num_micro_batches=len(micro_batches),
            )

        return objective.aggregate_step_metrics(
            self,
            accum,
            global_normalization_count=global_normalization_count,
            stage="train",
        )

    def _run_eval_forward(
        self, training_batch: dict[str, torch.Tensor], *, num_generations: int = 1,
    ) -> dict[str, float]:
        """Forward-only loss computation for validation: no backward pass, no optimizer step."""
        from training.batching import split_training_batch_into_micro_batches

        objective = self._require_training_objective()

        local_batch = self._trim_training_batch_tensors(training_batch)
        layout = self._training_batch_layout()
        micro_batches = split_training_batch_into_micro_batches(
            local_batch,
            layout=layout,
            num_generations=num_generations,
            prompts_per_micro_batch=self.args.per_device_train_batch_size,
            max_sequences_per_micro_batch=self.args.per_device_train_batch_size,
        )
        micro_batches = [self._trim_training_batch_tensors(mb) for mb in micro_batches]
        del training_batch, local_batch

        if not micro_batches:
            raise ValueError("Eval forward received no micro-batches.")

        count_micro_batches = getattr(objective, "count_normalization_items_for_micro_batches", None)
        if callable(count_micro_batches):
            local_normalization_count = count_micro_batches(self, micro_batches)
        else:
            local_normalization_count = sum(
                objective.count_normalization_items(self, micro_batch)
                for micro_batch in micro_batches
            )
        global_normalization_count = self._distributed_sum_int(local_normalization_count)
        if global_normalization_count <= 0:
            return {}

        normalization = torch.tensor(global_normalization_count, device=self.device, dtype=torch.float32)
        accum = objective.build_metrics_accumulator()

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for micro_batch in micro_batches:
                    micro_batch = self._move_batch_to_device(micro_batch)
                    objective_result = objective.compute_micro_batch(
                        self,
                        micro_batch,
                        normalization=normalization,
                        stage="eval",
                        micro_batch_index=None,
                        num_micro_batches=None,
                    )
                    accum.accumulate(objective_result.components)
                    del objective_result, micro_batch
        finally:
            if was_training:
                self.model.train()

        return objective.aggregate_step_metrics(
            self,
            accum,
            global_normalization_count=global_normalization_count,
            stage="eval",
        )

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        micro_batch: dict[str, torch.Tensor],
        micro_batch_index: int,
        num_micro_batches: int,
    ) -> float | None:
        backward_loss = loss * float(self._gradient_average_world_size())
        will_update_parameters = bool(self.model.is_gradient_accumulation_boundary())
        phase_logging_enabled = bool(self.args.log_phase_progress)
        backward_step_details = (
            self._backward_step_debug_details(
                micro_batch,
                will_update_parameters=will_update_parameters,
            )
            if phase_logging_enabled
            else None
        )
        backward_step_start = self._log_phase_start(
            stage="train",
            phase="backward_step",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=backward_step_details,
        )
        self._log_phase_event(
            stage="train",
            phase="backward",
            status="start",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=self._cuda_memory_stats() if phase_logging_enabled else None,
        )
        self._distributed_debug_barrier(
            stage="train",
            phase="pre_backward_barrier",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=backward_step_details,
        )
        self.model.backward(backward_loss)
        self._log_phase_event(
            stage="train",
            phase="backward",
            status="done",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=self._cuda_memory_stats() if phase_logging_enabled else None,
        )
        grad_norm: float | None = None
        if will_update_parameters:
            self._log_phase_event(
                stage="train",
                phase="grad_norm",
                status="start",
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
                details=self._cuda_memory_stats() if phase_logging_enabled else None,
            )
            grad_norm = self._maybe_get_global_grad_norm()
            grad_norm_details = None
            if phase_logging_enabled:
                grad_norm_details = self._cuda_memory_stats()
                if grad_norm is not None:
                    grad_norm_details["value"] = f"{grad_norm:.6f}"
            self._log_phase_event(
                stage="train",
                phase="grad_norm",
                status="done",
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
                details=grad_norm_details,
            )
        self._log_phase_event(
            stage="train",
            phase="optimizer_step",
            status="start",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=self._cuda_memory_stats() if phase_logging_enabled else None,
        )
        self.model.step()
        self._log_phase_event(
            stage="train",
            phase="optimizer_step",
            status="done",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            details=self._cuda_memory_stats() if phase_logging_enabled else None,
        )
        self._log_phase_end(
            stage="train",
            phase="backward_step",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            start_time=backward_step_start,
        )
        return grad_norm

    # ------------------------------------------------------------------
    # Policy metric aggregation
    # ------------------------------------------------------------------

    def _aggregate_policy_optimization_metrics(
        self,
        accum,
        global_policy_normalization_count: int,
    ) -> dict[str, float]:
        accum.finalize()
        dss = self._distributed_sum_scalar
        normalization = float(global_policy_normalization_count)
        kl_coef = float(self.args.policy_reward_kl_coef)
        global_policy_loss_sum = dss(accum.policy_loss_sum)
        global_reference_kl_sum = dss(accum.reference_kl_sum)
        global_action_token_count = dss(accum.action_token_count)
        action_normalization = max(float(global_action_token_count.item()), 1.0)

        metrics = {
            "policy/loss": float(
                (global_policy_loss_sum.item() + kl_coef * global_reference_kl_sum.item())
                / normalization
            ),
            "policy/surrogate_loss": float(global_policy_loss_sum.item() / normalization),
            "policy/reference_kl": float(global_reference_kl_sum.item() / normalization),
            "policy/vllm_kl": float(dss(accum.approx_kl_sum).item() / action_normalization),
            "policy/advantage_mean": float(dss(accum.advantage_sum).item() / action_normalization),
            "policy/old_log_prob_mean": float(dss(accum.old_log_prob_sum).item() / action_normalization),
            "policy/active_token_fraction": float(global_action_token_count.item() / normalization),
        }
        if accum.grad_norm is not None:
            metrics["sanity/grad_norm"] = self._distributed_mean(
                torch.tensor(accum.grad_norm, device=self.device, dtype=torch.float32)
            ).item()
        self._ensure_finite_metrics(metrics, stage="train")
        return metrics

    # ------------------------------------------------------------------
    # Policy optimization helpers
    # ------------------------------------------------------------------

    def _compute_policy_log_probs(
        self,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
    ) -> torch.Tensor:
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        completion_length = completion_ids.size(1)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        policy_forward_start = self._log_phase_start(
            stage=stage, phase="policy_forward",
            micro_batch_index=micro_batch_index, num_micro_batches=num_micro_batches,
        )
        current_log_probs = self._get_per_token_logps(
            self.model, input_ids, attention_mask, completion_length,
        )
        self._log_phase_end(
            stage=stage, phase="policy_forward",
            micro_batch_index=micro_batch_index, num_micro_batches=num_micro_batches,
            start_time=policy_forward_start,
        )
        self._ensure_finite_tensor(
            "policy_log_probs", current_log_probs,
            stage=stage, micro_batch_index=micro_batch_index, num_micro_batches=num_micro_batches,
        )
        return current_log_probs

    def _compute_policy_log_probs_and_reference_kl_full_vocab(
        self,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ref_model is None:
            raise RuntimeError(
                "policy_reward_kl_coef > 0 requires Trainer to own a frozen reference model."
            )
        if self._lm_head_chunk_size() is None:
            raise RuntimeError(
                "_compute_policy_log_probs_and_reference_kl_full_vocab requires lm_head_chunk_size."
            )

        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        action_mask = batch["action_mask"].bool()
        completion_length = completion_ids.size(1)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        policy_forward_start = self._log_phase_start(
            stage=stage,
            phase="policy_forward",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
        )
        student_completion_hidden_states = self._forward_student_completion_hidden_states_subbatched(
            input_ids,
            attention_mask,
            completion_length,
        )
        self._log_phase_end(
            stage=stage,
            phase="policy_forward",
            micro_batch_index=micro_batch_index,
            num_micro_batches=num_micro_batches,
            start_time=policy_forward_start,
        )

        ref_model = self.ref_model
        ref_was_training = ref_model.training
        ref_model.eval()
        try:
            reference_start = self._log_phase_start(
                stage=stage,
                phase="reference_full_vocab",
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            )
            with torch.no_grad():
                ref_completion_hidden_states = self._get_completion_hidden_states(
                    ref_model,
                    input_ids,
                    attention_mask,
                    completion_length,
                )
            self._log_phase_end(
                stage=stage,
                phase="reference_full_vocab",
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
                start_time=reference_start,
            )

            reference_kl, current_log_probs = self._compute_full_vocab_kl_and_logps_from_hidden_states(
                self.model,
                student_completion_hidden_states,
                ref_model,
                ref_completion_hidden_states,
                completion_ids,
            )
            del ref_completion_hidden_states, student_completion_hidden_states
            self._ensure_finite_tensor(
                "policy_log_probs",
                current_log_probs,
                stage=stage,
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            )
            self._ensure_finite_tensor(
                "policy_reference_kl_per_token_full_vocab",
                reference_kl,
                stage=stage,
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            )

            reference_kl = reference_kl * action_mask.to(dtype=reference_kl.dtype)
        finally:
            if ref_was_training:
                ref_model.train()

        return current_log_probs, reference_kl.sum(dtype=torch.float32)
