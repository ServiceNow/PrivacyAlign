"""Actor-only SAPO objective for RL training."""

from __future__ import annotations

import torch

from training.metrics import PolicyOptimizationTrainingStepMetrics
from objectives.base import (
    ObjectiveMicroBatchResult,
    ObjectiveRolloutRequirements,
)
from training.batching import POLICY_OPTIMIZATION_BATCH_LAYOUT


def compute_sapo_loss_stats(
    current_log_probs: torch.Tensor,
    *,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    algorithm: str = "sapo",
) -> dict[str, torch.Tensor]:
    """Compute policy-loss statistics over one micro-batch.

    ``algorithm``:
      * ``"sapo"`` (default): SAPO soft-clipped surrogate with asymmetric
        temperatures.
      * ``"vanilla"``: importance-weighted REINFORCE without clipping or soft
        gating: per-token loss = ``-(ratio * advantage)``. Use as a neutral
        RL baseline.
    """
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current_log_probs and old_log_probs must share the same shape.")
    if current_log_probs.shape != advantages.shape:
        raise ValueError("current_log_probs and advantages must share the same shape.")
    if current_log_probs.shape != action_mask.shape:
        raise ValueError("current_log_probs and action_mask must share the same shape.")
    if algorithm not in {"sapo", "vanilla"}:
        raise ValueError(f"algorithm must be 'sapo' or 'vanilla'; got {algorithm!r}.")

    action_mask_f = action_mask.to(dtype=current_log_probs.dtype)
    log_ratio = current_log_probs - old_log_probs
    ratio = log_ratio.exp()
    approx_kl = -log_ratio.detach() * action_mask_f

    if algorithm == "vanilla":
        per_token_loss = -(ratio * advantages)
    else:
        tau = torch.where(advantages >= 0, tau_pos, tau_neg).to(dtype=current_log_probs.dtype)
        gate_probability = torch.sigmoid(tau * (ratio - 1.0))
        gated_multiplier = gate_probability * (4.0 / tau)
        per_token_loss = -(gated_multiplier * advantages)

    return {
        "policy_loss_sum": (per_token_loss * action_mask_f).sum(dtype=torch.float32),
        "approx_kl_sum": approx_kl.sum(dtype=torch.float32),
    }


class PolicyOptimizationObjective:
    """Actor-only SAPO objective over precomputed rollout advantages."""

    name = "policy_optimization"

    def training_batch_layout(self):
        return POLICY_OPTIMIZATION_BATCH_LAYOUT

    def rollout_requirements(self, args) -> ObjectiveRolloutRequirements:
        del args
        return ObjectiveRolloutRequirements(include_sampled_logprobs=True)

    def requires_reference_model(self, args) -> bool:
        return float(getattr(args, "policy_reward_kl_coef", 0.0)) > 0.0

    def build_metrics_accumulator(self) -> PolicyOptimizationTrainingStepMetrics:
        return PolicyOptimizationTrainingStepMetrics()

    def count_normalization_items(
        self,
        trainer,
        batch,
    ) -> int:
        del trainer
        # Dr. GRPO-style length normalization: reduce summed token losses by the
        # total number of action tokens in the optimizer batch, never by each
        # sequence's own length.
        return int(batch["action_mask"].bool().sum().item())

    def compute_micro_batch(
        self,
        trainer,
        batch,
        *,
        normalization,
        stage: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
    ) -> ObjectiveMicroBatchResult:
        tau_pos = float(getattr(trainer.args, "policy_sapo_tau_pos", 1.0))
        tau_neg = float(getattr(trainer.args, "policy_sapo_tau_neg", 1.05))
        algorithm = str(getattr(trainer.args, "policy_algorithm", "sapo"))
        kl_coef = float(getattr(trainer.args, "policy_reward_kl_coef", 0.0))
        action_mask = batch["action_mask"].bool()
        if kl_coef > 0.0:
            current_log_probs, reference_kl_sum = (
                trainer._compute_policy_log_probs_and_reference_kl_full_vocab(
                    batch,
                    stage=stage,
                    micro_batch_index=micro_batch_index,
                    num_micro_batches=num_micro_batches,
                )
            )
        else:
            current_log_probs = trainer._compute_policy_log_probs(
                batch,
                stage=stage,
                micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            )
            reference_kl_sum = torch.zeros((), device=current_log_probs.device, dtype=torch.float32)
        stats = compute_sapo_loss_stats(
            current_log_probs,
            old_log_probs=batch["old_log_probs"],
            advantages=batch["advantages"],
            action_mask=action_mask,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            algorithm=algorithm,
        )
        loss = (stats["policy_loss_sum"] + kl_coef * reference_kl_sum) / normalization

        action_mask_f = action_mask.to(dtype=current_log_probs.dtype)
        advantage_sum = (batch["advantages"].detach() * action_mask_f).sum(dtype=torch.float32)
        old_log_prob_sum = (batch["old_log_probs"].detach() * action_mask_f).sum(dtype=torch.float32)

        components: dict[str, torch.Tensor | float] = {
            "policy_loss_sum": stats["policy_loss_sum"],
            "reference_kl_sum": reference_kl_sum,
            "approx_kl_sum": stats["approx_kl_sum"],
            "action_token_count": action_mask_f.sum(dtype=torch.float32),
            "advantage_sum": advantage_sum,
            "old_log_prob_sum": old_log_prob_sum,
        }

        log_details: dict[str, str] = {}
        if bool(getattr(trainer.args, "log_phase_progress", False)):
            log_details = {
                "objective": self.name,
                "loss_type": algorithm,
                "total_loss": f"{loss.detach().float().item():.6f}",
                "surrogate_loss": f"{(stats['policy_loss_sum'] / normalization).detach().float().item():.6f}",
                "reference_kl": f"{(reference_kl_sum / normalization).detach().float().item():.6f}",
                "vllm_kl": f"{(stats['approx_kl_sum'] / normalization).detach().float().item():.6f}",
                "tau_pos": f"{tau_pos:.4f}",
                "tau_neg": f"{tau_neg:.4f}",
                "normalization": f"{normalization.detach().float().item():.1f}",
            }

        return ObjectiveMicroBatchResult(
            loss=loss,
            components=components,
            log_details=log_details,
        )

    def aggregate_step_metrics(
        self,
        trainer,
        accum: PolicyOptimizationTrainingStepMetrics,
        *,
        global_normalization_count: int,
        stage: str,
    ) -> dict[str, float]:
        del stage
        return trainer._aggregate_policy_optimization_metrics(
            accum,
            global_policy_normalization_count=global_normalization_count,
        )
