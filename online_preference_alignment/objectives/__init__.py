"""Factories for trainer objectives."""

from __future__ import annotations

from objectives.base import ObjectiveRolloutRequirements, TrainingObjective
from objectives.policy_optimization import PolicyOptimizationObjective


def build_training_objective(name: str) -> TrainingObjective:
    normalized = name.strip().lower()
    if normalized == "policy_optimization":
        return PolicyOptimizationObjective()
    raise ValueError(f"Unsupported training_objective: {name!r}")


__all__ = [
    "ObjectiveRolloutRequirements",
    "PolicyOptimizationObjective",
    "TrainingObjective",
    "build_training_objective",
]
