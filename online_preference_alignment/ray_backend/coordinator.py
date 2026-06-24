"""Ray coordinator – main driver for preference-conditioned training.

Owns the CLI, Ray cluster bootstrap, dataset preparation, logging / metrics,
and the outer training loop.  Rollout dispatch, vLLM engine lifecycle, and
dev-eval orchestration live in sibling modules:

- ``ray_backend/rollout_dispatch.py`` – vLLM prompt dispatch, engine helpers,
  training payload construction, single-step iteration runner.
- ``ray_backend/dev_eval.py``          – preference evaluation, reference
  response caching, temporary vLLM engine context manager.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import ray
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer

import wandb
from cli import build_arg_parser, build_training_config
from data_loaders import get_dataset
from data_loaders.preference import (
    format_preference_dataset,
    is_privalign_dataset_name,
    load_privalign_pairwise_genrm_dataset,
)
from objectives import build_training_objective
from ray_backend.dev_eval import (
    build_dev_eval_state,
    build_policy_batch_builder,
    run_dev_eval,
    should_run_dev_eval,
)
from ray_backend.launcher import RayActorGroup
from ray_backend.ray_utils import wait_for_ray_refs
from ray_backend.rollout_dispatch import (
    VLLMEnginePool,
    build_training_payload,
    destroy_policy_judge_vllm_engines,
    destroy_trained_genrm_vllm_engines,
    requested_vllm_logprob_count,
    run_training_iteration,
    sync_actor_weights_to_vllm,
)
from ray_backend.rollout_utils import OfflineRolloutBatchBuilder
from ray_backend.training_worker import (
    TrainingModelActor,
    build_training_config_init_kwargs,
)
from ray_backend.vllm_engine import (
    create_offline_vllm_engine_bundle,
)
logger = logging.getLogger(__name__)
_WANDB_HISTORY_TABLES: dict[str, object] = {}
_WANDB_HISTORY_COLUMNS: dict[str, tuple[str, ...]] = {}
_WANDB_HISTORY_ROWS: dict[str, list[dict[str, object]]] = {}


# ---------------------------------------------------------------------------
# Small dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTrainingBatchLayout:
    global_prompt_batch_size: int
    local_prompt_batch_size: int
    global_sequence_batch_size: int
    local_sequence_batch_size: int
    per_device_train_batch_size: int
    training_gradient_accumulation_steps: int


# ---------------------------------------------------------------------------
# Phase-timing context manager
# ---------------------------------------------------------------------------


@contextmanager
def _log_phase(description: str):
    start_time = time.monotonic()
    logger.info("%s...", description)
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.1fs", description, time.monotonic() - start_time)
        raise
    logger.info("%s completed in %.1fs", description, time.monotonic() - start_time)


# ---------------------------------------------------------------------------
# Ray / cluster helpers
# ---------------------------------------------------------------------------


def _format_ray_resources(resources: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(resources.items()))


def should_use_cuda_ipc_sync(args) -> bool:
    return bool(args.ray_colocate_models and args.vllm_sync_backend == "nccl")


def resolve_total_train_workers(args) -> int:
    total_workers = args.ray_num_nodes * args.ray_num_train_workers
    if total_workers <= 0:
        raise ValueError("Ray training requires at least one worker.")
    return total_workers


def resolve_total_vllm_engines(args) -> int:
    total_engines = int(args.ray_vllm_num_engines)
    if total_engines <= 0:
        raise ValueError("--ray_vllm_num_engines must be > 0.")
    return total_engines


def resolve_ray_worker_model_load_path(model_name: str, *, world_size: int) -> str:
    if world_size <= 1 or os.path.isdir(model_name):
        return model_name
    return snapshot_download(repo_id=model_name)


def resolve_training_batch_layout_for_world_size(
    args,
    *,
    world_size: int,
) -> ResolvedTrainingBatchLayout:
    if args.global_prompt_batch_size <= 0:
        raise ValueError("--global_prompt_batch_size must be > 0.")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per_device_train_batch_size must be > 0.")
    num_generations = int(args.num_generations)
    if num_generations <= 0:
        raise ValueError("--num_generations must be > 0.")
    if args.global_prompt_batch_size % world_size != 0:
        raise ValueError(
            "--global_prompt_batch_size must be divisible by world_size so prompt batches shard evenly "
            f"({args.global_prompt_batch_size} prompts across {world_size} workers)."
        )
    local_prompt_batch_size = args.global_prompt_batch_size // world_size
    local_sequence_batch_size = local_prompt_batch_size * num_generations
    if local_sequence_batch_size % args.per_device_train_batch_size != 0:
        raise ValueError(
            "--per_device_train_batch_size must divide the per-worker generated-sequence count "
            f"({local_prompt_batch_size} prompts * {num_generations} generations = "
            f"{local_sequence_batch_size} sequences per worker)."
        )
    training_gradient_accumulation_steps = (
        local_sequence_batch_size // args.per_device_train_batch_size
    )
    return ResolvedTrainingBatchLayout(
        global_prompt_batch_size=args.global_prompt_batch_size,
        local_prompt_batch_size=local_prompt_batch_size,
        global_sequence_batch_size=args.global_prompt_batch_size * num_generations,
        local_sequence_batch_size=local_sequence_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        training_gradient_accumulation_steps=training_gradient_accumulation_steps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_ray_arg_parser():
    parser = build_arg_parser()
    parser.description = "Ray-backed preference-conditioned training runner"
    parser.add_argument("--ray_address", type=str, default=None, help="Ray cluster address. Defaults to local.")
    parser.add_argument("--ray_num_nodes", type=int, default=1, help="Number of Ray training nodes.")
    parser.add_argument(
        "--ray_num_train_workers",
        type=int,
        default=1,
        help="Number of Ray training workers per node.",
    )
    parser.add_argument("--ray_vllm_num_engines", type=int, default=1, help="Number of offline vLLM Ray engines.")
    parser.add_argument(
        "--ray_colocate_models",
        action="store_true",
        default=True,
        help="Pack training workers and offline vLLM engines into the same Ray placement group.",
    )
    parser.add_argument(
        "--vllm_sync_backend",
        type=str,
        default="nccl",
        help="Weight-sync backend used between rank-0 training worker and offline vLLM workers.",
    )
    return parser


def parse_args():
    args = build_ray_arg_parser().parse_args()
    return args


# ---------------------------------------------------------------------------
# Seeding / data loading
# ---------------------------------------------------------------------------


def seed_ray_driver(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompt_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
) -> DataLoader:
    """Build the prompt-batch dataloader for one training epoch."""
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    sampler = RandomSampler(dataset, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=list,
    )


def _build_shared_placement_group(args):
    from ray.util.placement_group import placement_group

    total_train_workers = resolve_total_train_workers(args)
    expected_vllm_workers = args.ray_vllm_num_engines * args.vllm_tensor_parallel_size
    if total_train_workers != expected_vllm_workers:
        raise ValueError(
            "Ray colocation requires total Ray train workers to match "
            "ray_vllm_num_engines * vllm_tensor_parallel_size "
            f"({total_train_workers} != {expected_vllm_workers})."
        )

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_train_workers)]
    logger.info(
        "Creating shared Ray placement group with %s bundles for colocated training/vLLM workers.",
        len(bundles),
    )
    shared_pg = placement_group(bundles, strategy="PACK")
    wait_for_ray_refs([shared_pg.ready()], description="shared Ray placement group scheduling")
    return shared_pg


def _filter_dataset_by_prompt_length(
    dataset,
    *,
    rollout_builder: OfflineRolloutBatchBuilder,
    max_prompt_length: int | None,
):
    """Drop rows whose tokenized prompt exceeds ``max_prompt_length``."""
    if max_prompt_length is None:
        return dataset

    def keep_example(example: dict[str, Any]) -> bool:
        prompt_lengths = rollout_builder._example_prompt_token_lengths(example)
        return all(length <= max_prompt_length for length in prompt_lengths.values())

    original_size = len(dataset)
    if hasattr(dataset, "filter"):
        filtered_dataset = dataset.filter(keep_example)
    else:
        filtered_dataset = [example for example in dataset if keep_example(example)]

    filtered_size = len(filtered_dataset)
    removed_count = original_size - filtered_size
    logger.info(
        "Filtered training dataset by max_prompt_length=%s; kept=%s removed=%s.",
        max_prompt_length, filtered_size, removed_count,
    )
    if filtered_size <= 0:
        raise ValueError(
            "All training examples were filtered out by max_prompt_length. "
            "Increase --max_prompt_length or use a dataset with shorter prompts."
        )
    return filtered_dataset


def _limit_dataset_samples(dataset, *, max_samples: int | None, label: str):
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError(f"{label} max_samples must be > 0 when set.")
    capped = min(len(dataset), int(max_samples))
    if hasattr(dataset, "select"):
        limited = dataset.select(range(capped))
    else:
        limited = list(dataset)[:capped]
    logger.info("Limited %s dataset to %s/%s examples.", label, len(limited), len(dataset))
    return limited


# ---------------------------------------------------------------------------
# Resume / checkpoint utilities
# ---------------------------------------------------------------------------


def _resume_position(global_step: int, *, steps_per_epoch: int) -> tuple[int, int]:
    if global_step < 0:
        raise ValueError("global_step must be >= 0.")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0.")
    return divmod(global_step, steps_per_epoch)


def _read_latest_checkpoint_metadata(output_dir: str) -> dict[str, Any] | None:
    latest_path = os.path.join(output_dir, "latest_checkpoint.json")
    if not os.path.isfile(latest_path):
        return None
    with open(latest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_resume_checkpoint_path(output_dir: str, resume: str | None) -> str | None:
    if resume is None:
        return None

    normalized = resume.strip()
    if not normalized:
        return None

    if normalized.lower() == "latest":
        latest_state = _read_latest_checkpoint_metadata(output_dir)
        if latest_state is None:
            raise FileNotFoundError(f"No latest_checkpoint.json found under {output_dir}.")
        checkpoint_path = latest_state.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(f"latest_checkpoint.json under {output_dir} is missing checkpoint_path.")
        return os.path.abspath(str(checkpoint_path))

    candidate = normalized
    if not os.path.isabs(candidate):
        direct_candidate = os.path.join(output_dir, candidate)
        if os.path.isdir(direct_candidate):
            candidate = direct_candidate

    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "latest_checkpoint.json")):
        latest_state = _read_latest_checkpoint_metadata(candidate)
        if latest_state is None or not latest_state.get("checkpoint_path"):
            raise ValueError(f"latest_checkpoint.json under {candidate} is missing checkpoint_path.")
        candidate = os.path.abspath(str(latest_state["checkpoint_path"]))

    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Resume checkpoint directory not found: {candidate}")

    trainer_state_path = os.path.join(candidate, "trainer_state.json")
    if not os.path.isfile(trainer_state_path):
        raise FileNotFoundError(f"Resume checkpoint is missing trainer_state.json: {candidate}")

    return os.path.abspath(candidate)


# ---------------------------------------------------------------------------
# Logging / metrics
# ---------------------------------------------------------------------------


def _format_metric_value(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return "0.0000"
    if 1e-3 <= magnitude < 1e4:
        return f"{value:.4f}"
    return f"{value:.6e}"


_METRIC_NAME_PRIORITY = {
    "policy/loss": 0,
    "policy/surrogate_loss": 1,
    "policy/reference_kl": 2,
    "policy/vllm_kl": 3,
    "progress/epochs_completed": 4,
    "sanity/vllm_to_student_kl_k2_estimator": 5,
    "sanity/student_response_entropy_mean": 6,
    "sanity/grad_norm": 7,
}


def _metric_sort_key(name: str) -> tuple[int, int, str]:
    explicit_priority = _METRIC_NAME_PRIORITY.get(name)
    if explicit_priority is not None:
        return (0, explicit_priority, name)
    if name.startswith("loss/"):
        return (1, 0, name)
    if name.startswith("sanity/"):
        return (2, 0, name)
    if name.startswith("diag/"):
        return (3, 0, name)
    return (4, 0, name)


def _ordered_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return dict(sorted(metrics.items(), key=lambda item: _metric_sort_key(item[0])))


def _compute_epochs_completed(global_step: int, *, steps_per_epoch: int) -> float:
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0.")
    if global_step < 0:
        raise ValueError("global_step must be >= 0.")
    return float(global_step) / float(steps_per_epoch)


def _wandb_requested(report_to: list[str] | None) -> bool:
    return report_to is not None and "wandb" in report_to


def _reset_wandb_sample_history() -> None:
    _WANDB_HISTORY_TABLES.clear()
    _WANDB_HISTORY_COLUMNS.clear()
    _WANDB_HISTORY_ROWS.clear()


def _get_wandb_sample_history_table(
    history_key: str,
    columns: list[str],
    *,
    rows: list[dict[str, object]] | None = None,
):
    existing_columns = list(_WANDB_HISTORY_COLUMNS.get(history_key, ()))
    if not existing_columns:
        existing_columns = list(columns)
    else:
        for column in columns:
            if column not in existing_columns:
                existing_columns.append(column)
    column_tuple = tuple(existing_columns)

    stored_rows = _WANDB_HISTORY_ROWS.setdefault(history_key, [])
    pending_rows = list(rows or [])
    if pending_rows:
        stored_rows.extend(dict(row) for row in pending_rows)

    table_missing = history_key not in _WANDB_HISTORY_TABLES
    schema_changed = _WANDB_HISTORY_COLUMNS.get(history_key) != column_tuple
    if table_missing or schema_changed:
        history_table = wandb.Table(columns=existing_columns, log_mode="INCREMENTAL")
        for row in stored_rows:
            history_table.add_data(*[row.get(column, "") for column in existing_columns])
        _WANDB_HISTORY_TABLES[history_key] = history_table
        _WANDB_HISTORY_COLUMNS[history_key] = column_tuple
        return history_table

    history_table = _WANDB_HISTORY_TABLES[history_key]
    for row in pending_rows:
        history_table.add_data(*[row.get(column, "") for column in existing_columns])
    return history_table


def _setup_wandb(config) -> bool:
    if not _wandb_requested(config.report_to):
        return False

    os.makedirs(config.output_dir, exist_ok=True)
    _reset_wandb_sample_history()
    try:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "preference-rl"),
            name=os.path.basename(os.path.normpath(config.output_dir)) or None,
            dir=config.output_dir,
            config=config.to_dict(),
        )
        if wandb.run is not None:
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step", step_sync=True)
    except Exception as exc:
        logger.warning("Disabling Weights & Biases logging after initialization failure: %s", exc)
        return False
    return True


def _finish_wandb(wandb_enabled: bool) -> None:
    if not wandb_enabled or wandb.run is None:
        return
    wandb.finish()
    _reset_wandb_sample_history()


def _log_wandb_payload(
    global_step: int,
    payload: dict[str, object],
    *,
    wandb_enabled: bool = False,
) -> None:
    if not wandb_enabled or wandb.run is None or not payload:
        return
    wandb.log({"global_step": global_step, **payload}, step=global_step, commit=True)


def _log_metrics(global_step: int, metrics: dict[str, float]) -> None:
    ordered = _ordered_metrics(metrics)
    rendered = ", ".join(f"{key}={_format_metric_value(value)}" for key, value in ordered.items())
    logger.info("step=%s train %s", global_step, rendered)


def _is_better_metric(new_value: float, prior_best: float | None, *, higher_is_better: bool) -> bool:
    if prior_best is None:
        return True
    if higher_is_better:
        return new_value > prior_best
    return new_value < prior_best


def _maybe_save_best_checkpoint(
    *,
    config,
    model_group,
    eval_metrics: dict[str, float],
    global_step: int,
    best_state: dict,
    wandb_enabled: bool,
) -> None:
    metric_name = getattr(config, "save_best_metric", None)
    if not metric_name:
        return
    if metric_name not in eval_metrics:
        logger.warning(
            "save_best_metric=%s not found in dev eval metrics at step=%s; skipping best-checkpoint update.",
            metric_name, global_step,
        )
        return
    new_value = float(eval_metrics[metric_name])
    higher_is_better = bool(getattr(config, "save_best_higher_is_better", True))
    prior_best = best_state.get("value")
    if not _is_better_metric(new_value, prior_best, higher_is_better=higher_is_better):
        return
    ray.get(model_group.async_run_method("save_checkpoint", "best"))
    best_state["value"] = new_value
    best_state["step"] = int(global_step)
    best_state["metric"] = metric_name
    logger.info(
        "Saved best checkpoint at step=%s (%s=%.6f, prior_best=%s).",
        global_step, metric_name, new_value,
        "n/a" if prior_best is None else f"{prior_best:.6f}",
    )
    if wandb_enabled and wandb.run is not None:
        wandb.run.summary["best_checkpoint"] = os.path.join(config.output_dir, "best")
        wandb.run.summary["best_checkpoint_step"] = int(global_step)
        wandb.run.summary[f"best_{metric_name}"] = new_value


def _log_eval_metrics(global_step: int, metrics: dict[str, float]) -> None:
    ordered = _ordered_metrics(metrics)
    rendered = ", ".join(f"{key}={_format_metric_value(value)}" for key, value in ordered.items())
    logger.info("step=%s eval %s", global_step, rendered)


def _build_wandb_metrics_payload(
    prefix: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    ordered = _ordered_metrics(metrics)
    return {f"{prefix}/{key}": value for key, value in ordered.items()}


def _resolve_startup_eval_log_step(resume_global_step: int) -> int:
    # Fresh-run evals happen before the first optimizer step, so log them at
    # step 0 to avoid colliding with the first training metrics at step 1.
    return max(0, int(resume_global_step))


def _log_startup_eval_metrics(
    global_step: int,
    *,
    eval_metrics: dict[str, float] | None = None,
    startup_eval_sample_texts: dict[str, str] | list[dict[str, str]] | None = None,
    wandb_enabled: bool = False,
) -> None:
    payload: dict[str, object] = {}

    if eval_metrics is not None:
        _log_eval_metrics(global_step, eval_metrics)
        payload.update(_build_wandb_metrics_payload("eval", eval_metrics))

    payload.update(
        _build_wandb_eval_samples_payload(global_step, startup_eval_sample_texts)
    )
    _log_wandb_payload(global_step, payload, wandb_enabled=wandb_enabled)


def should_log_training_samples(config, global_step: int) -> bool:
    sample_log_steps = int(getattr(config, "sample_log_steps", 1))
    if sample_log_steps <= 0 or global_step <= 0:
        return False
    return global_step == 1 or global_step % sample_log_steps == 0


def _render_sample_texts(sample_texts: dict[str, str] | None) -> str | None:
    if not sample_texts:
        return None
    rendered_lines: list[str] = []
    for key, value in sample_texts.items():
        if not value:
            continue
        rendered_lines.extend([f"{key}:", value])
    if not rendered_lines:
        return None
    return "\n".join(rendered_lines)


def _build_wandb_batch_samples_payload(
    global_step: int,
    sample_texts: dict[str, str] | None,
) -> dict[str, object]:
    if not sample_texts:
        return {}
    columns = list(sample_texts.keys())
    history_columns = ["global_step", *columns]
    history_table = _get_wandb_sample_history_table(
        "training_samples",
        history_columns,
        rows=[{"global_step": global_step, **sample_texts}],
    )
    payload: dict[str, object] = {
        "samples/history_table": history_table,
        "samples/logged_step": float(global_step),
    }
    for column, value in sample_texts.items():
        payload[f"samples/{column}"] = value
    return payload


def _build_wandb_eval_samples_payload(
    global_step: int,
    sample_texts: dict[str, str] | list[dict[str, str]] | None,
) -> dict[str, object]:
    """Build a wandb payload that logs eval sample texts into an incremental Table.

    Accepts either a single sample dict or a list of sample dicts. Each row is
    appended to a persistent ``eval_samples`` history table so that all samples
    across training steps are visible in one wandb Table panel.
    """
    if not sample_texts:
        return {}
    sample_rows = [sample_texts] if isinstance(sample_texts, dict) else list(sample_texts)
    if not sample_rows:
        return {}
    columns: list[str] = []
    for row in sample_rows:
        for column in row.keys():
            if column not in columns:
                columns.append(column)
    history_columns = ["global_step", *columns]
    history_table = _get_wandb_sample_history_table(
        "eval_samples",
        history_columns,
        rows=[
            {"global_step": global_step, **{column: row.get(column, "") for column in columns}}
            for row in sample_rows
        ],
    )
    payload: dict[str, object] = {
        "eval_samples/history_table": history_table,
        "eval_samples/logged_step": float(global_step),
    }
    # Also log the first sample's fields as individual scalar strings for the
    # wandb default panels (matches the training sample logging pattern).
    for column, value in sample_rows[0].items():
        payload[f"eval_samples/{column}"] = value
    return payload


def _get_logged_startup_eval_samples(dev_eval_state) -> dict[str, str] | list[dict[str, str]] | None:
    if dev_eval_state is None:
        return None
    return (
        getattr(dev_eval_state, "logged_eval_sample_texts_batch", None)
        or getattr(dev_eval_state, "logged_eval_sample_texts", None)
    )


def _log_checkpoint_saved(
    *,
    checkpoint_name: str,
    global_step: int,
    output_dir: str,
) -> None:
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    logger.info("step=%s saved checkpoint=%s path=%s", global_step, checkpoint_name, checkpoint_path)


def _build_wandb_checkpoint_payload(global_step: int) -> dict[str, object]:
    return {
        "checkpoint/save_step": float(global_step),
        "checkpoint/save_event": 1.0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    ray_vllm_num_engines = resolve_total_vllm_engines(args)
    seed_ray_driver(args.seed)

    with _log_phase("Initializing Ray runtime"):
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    logger.info("Ray cluster resources: %s", _format_ray_resources(ray.cluster_resources()))
    logger.info("Ray available resources: %s", _format_ray_resources(ray.available_resources()))

    total_train_workers = resolve_total_train_workers(args)

    batch_layout = resolve_training_batch_layout_for_world_size(args, world_size=total_train_workers)
    os.environ["WORLD_SIZE"] = str(total_train_workers)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    logger.info(
        "Resolved Ray training layout: workers=%s, global_prompt_batch_size=%s, "
        "global_sequence_batch_size=%s, local_prompt_batch_size=%s, "
        "per_device_train_batch_size=%s, training_gradient_accumulation_steps=%s",
        total_train_workers,
        batch_layout.global_prompt_batch_size,
        batch_layout.global_sequence_batch_size,
        batch_layout.local_prompt_batch_size,
        batch_layout.per_device_train_batch_size,
        batch_layout.training_gradient_accumulation_steps,
    )
    use_cuda_ipc_sync = should_use_cuda_ipc_sync(args)
    if use_cuda_ipc_sync:
        logger.info(
            "Using CUDA IPC for vLLM weight sync because ray_colocate_models=true and vllm_sync_backend=nccl."
        )
    else:
        logger.info("Using direct %s communicator for vLLM weight sync.", args.vllm_sync_backend)

    # -- Dataset ----------------------------------------------------------------
    raw_preference_dataset = None
    user_prompt_suffix = getattr(args, "user_prompt_suffix", None)
    policy_scorer = getattr(args, "policy_scorer", None)
    with _log_phase(f"Loading dataset {args.dataset_name} ({args.dataset})"):
        if policy_scorer == "privalign_pairwise_margin" and is_privalign_dataset_name(
            args.dataset_name
        ):
            logger.info(
                "Building pairwise gen-RM training rows in memory from Privalign at %s.",
                args.dataset_name,
            )
            raw_preference_dataset = load_privalign_pairwise_genrm_dataset(
                args.dataset_name, split="train",
            )
            raw_preference_dataset = _limit_dataset_samples(
                raw_preference_dataset,
                max_samples=args.train_max_samples,
                label="training",
            )
            dataset = raw_preference_dataset
        else:
            dataset_kwargs = {
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "seed": args.seed,
            }
            if args.dataset == "preference":
                dataset_kwargs["deduplicate_examples"] = args.deduplicate_preference_examples
                dataset_kwargs["user_prompt_suffix"] = user_prompt_suffix
            dataset = get_dataset(args.dataset, **dataset_kwargs)
            dataset = _limit_dataset_samples(
                dataset,
                max_samples=args.train_max_samples,
                label="training",
            )
    if raw_preference_dataset is None:
        logger.info("Loaded Ray training dataset with %s train examples.", len(dataset))
    else:
        logger.info(
            "Loaded raw Ray training dataset with %s preference examples before policy formatting.",
            len(dataset),
        )

    config = build_training_config(
        args,
        gradient_accumulation_steps=batch_layout.training_gradient_accumulation_steps,
    )
    # Keep driver-side vLLM helper paths (dev eval, policy judge, trained genRM)
    # aligned with the rollout engine shape without adding another config knob.
    config.ray_vllm_num_engines = ray_vllm_num_engines
    training_objective = build_training_objective(config.training_objective)
    config.resume_from = _resolve_resume_checkpoint_path(config.output_dir, args.resume_from)
    wandb_enabled = _setup_wandb(config)
    ray_worker_model_load_path = args.model_name
    if total_train_workers > 1 and not os.path.isdir(args.model_name):
        with _log_phase(f"Resolving local model snapshot for Ray worker loads ({args.model_name})"):
            ray_worker_model_load_path = resolve_ray_worker_model_load_path(
                args.model_name, world_size=total_train_workers,
            )
        logger.info(
            "Using local model snapshot for Ray training workers to avoid concurrent HF cache races: %s",
            ray_worker_model_load_path,
        )

    # -- Tokenizer & rollout builder -------------------------------------------
    with _log_phase(f"Loading tokenizer and generation config for {args.model_name}"):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=args.trust_remote_code,
        )
        try:
            from transformers import GenerationConfig

            generation_config = GenerationConfig.from_pretrained(
                args.model_name, trust_remote_code=args.trust_remote_code,
            )
        except Exception:
            generation_config = None
    rollout_builder = OfflineRolloutBatchBuilder(config, tokenizer, generation_config=generation_config)

    max_model_len = None
    if args.max_prompt_length is not None and args.max_completion_length is not None:
        max_model_len = args.max_prompt_length + args.max_completion_length

    if raw_preference_dataset is not None:
        with _log_phase("Formatting preference dataset for training"):
            dataset = format_preference_dataset(
                raw_preference_dataset,
                seed=args.seed,
                deduplicate_examples=args.deduplicate_preference_examples,
            )
        logger.info("Loaded Ray training dataset with %s train examples.", len(dataset))

    dev_eval_state = None
    if (
        config.dev_eval_steps > 0
        or config.dev_eval_before_training
    ):
        with _log_phase("Preparing preference validation dev eval"):
            dev_eval_state = build_dev_eval_state(
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                config=config,
                rollout_builder=rollout_builder,
                reference_model_path=args.model_name,
                judge_model_path=config.dev_eval_judge_model_name,
            )
    dataset = _filter_dataset_by_prompt_length(
        dataset,
        rollout_builder=rollout_builder,
        max_prompt_length=config.max_prompt_length,
    )

    prompts_per_worker = config.local_prompt_batch_size
    steps_per_epoch = len(dataset) // batch_layout.global_prompt_batch_size
    if steps_per_epoch <= 0:
        raise ValueError("Training dataset produced no full optimizer batches for the Ray pipeline.")
    total_steps = steps_per_epoch * args.num_train_epochs
    logger.info("Running with fresh synchronous rollouts only; rollout/training overlap is disabled.")

    # -- Ray actors & vLLM engines --------------------------------------------
    with _log_phase("Preparing Ray placement and training actors"):
        shared_pg = _build_shared_placement_group(args)
        model_group = RayActorGroup(
            num_nodes=args.ray_num_nodes,
            num_gpus_per_node=args.ray_num_train_workers,
            ray_actor_type=TrainingModelActor,
            pg=shared_pg,
            num_gpus_per_actor=0.2 if shared_pg is not None else 1.0,
        )

    rollout_vllm_create_kwargs = {
        "num_engines": ray_vllm_num_engines,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "full_determinism": False,
        "max_model_len": max_model_len,
        "max_logprobs": requested_vllm_logprob_count(args),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_enable_sleep": args.vllm_enable_sleep_mode,
        "vllm_enforce_eager": args.vllm_enforce_eager,
        "log_phase_progress": args.log_phase_progress,
        "shared_pg": shared_pg,
    }
    with _log_phase("Creating offline vLLM Ray engines"):
        vllm_engines, rollout_engine_bundle_pg = create_offline_vllm_engine_bundle(
            **rollout_vllm_create_kwargs,
        )
    logger.info("Created %s offline vLLM engine(s).", len(vllm_engines))

    pool = VLLMEnginePool(
        engines=vllm_engines,
        lock=threading.RLock(),
        shared_pg=shared_pg,
        rollout_engine_bundle_pg=rollout_engine_bundle_pg,
        rollout_engine_create_kwargs=rollout_vllm_create_kwargs,
    )

    # -- Worker init -----------------------------------------------------------
    worker_config_kwargs = build_training_config_init_kwargs(config)
    report_to = worker_config_kwargs.get("report_to")
    if isinstance(report_to, str):
        worker_config_kwargs["report_to"] = None if report_to == "wandb" else report_to
    elif report_to is not None:
        remaining_targets = [target for target in report_to if target != "wandb"]
        worker_config_kwargs["report_to"] = remaining_targets or None

    actor_init_refs = model_group.async_init_model_from_pretrained(
        config_kwargs=worker_config_kwargs,
        model_name=args.model_name,
        model_load_path=ray_worker_model_load_path,
        max_steps=total_steps,
        vllm_engines=pool.engines,
        vllm_num_engines=ray_vllm_num_engines,
        vllm_sync_backend=args.vllm_sync_backend,
        ray_colocate_models=args.ray_colocate_models,
    )
    actor_init_results = wait_for_ray_refs(actor_init_refs, description="Ray training worker initialization")
    rank_zero_state = next(result for result in actor_init_results if result["rank"] == 0)
    resume_global_step = int(rank_zero_state.get("global_step", 0))
    resume_tokens_seen = int(rank_zero_state.get("num_input_tokens_seen", 0))
    if resume_global_step > 0:
        logger.info(
            "Resuming training from checkpoint=%s at global_step=%s.",
            config.resume_from,
            resume_global_step,
        )

    rollout_builder.num_input_tokens_seen = resume_tokens_seen
    vllm_sampling_kwargs = rollout_builder.build_sampling_kwargs(
        args, stop_token_ids=rollout_builder._resolve_stop_token_ids(),
    )

    sync_actor_weights_to_vllm(model_group=model_group, config=config, pool=pool)
    startup_eval_log_step = _resolve_startup_eval_log_step(resume_global_step)
    policy_batch_builder = build_policy_batch_builder(
        config=config,
        training_objective=training_objective,
        rollout_builder=rollout_builder,
        model_group=model_group,
        pool=pool,
    )

    startup_eval_metrics: dict[str, float] | None = None
    if resume_global_step < total_steps and config.dev_eval_before_training:
        with _log_phase("Running preference dev eval before training"):
            startup_eval_metrics = run_dev_eval(
                model_group=model_group,
                config=config,
                rollout_builder=rollout_builder,
                eval_state=dev_eval_state,
                pool=pool,
                training_objective=training_objective,
                policy_batch_builder=policy_batch_builder,
            )

    _log_startup_eval_metrics(
        startup_eval_log_step,
        eval_metrics=startup_eval_metrics,
        startup_eval_sample_texts=_get_logged_startup_eval_samples(dev_eval_state),
        wandb_enabled=wandb_enabled,
    )

    best_checkpoint_state: dict[str, Any] = {}
    if startup_eval_metrics is not None:
        _maybe_save_best_checkpoint(
            config=config,
            model_group=model_group,
            eval_metrics=startup_eval_metrics,
            global_step=startup_eval_log_step,
            best_state=best_checkpoint_state,
            wandb_enabled=wandb_enabled,
        )

    # -- Training loop ---------------------------------------------------------
    try:
        if resume_global_step >= total_steps:
            logger.info(
                "Checkpoint is already at global_step=%s, which meets or exceeds total_steps=%s. Skipping training.",
                resume_global_step,
                total_steps,
            )
        else:
            logger.info("Running rollout generation synchronously.")
            start_epoch, skip_batches_in_start_epoch = _resume_position(
                resume_global_step, steps_per_epoch=steps_per_epoch,
            )
            for _epoch in range(start_epoch, args.num_train_epochs):
                dataloader = _build_prompt_dataloader(
                    dataset,
                    batch_size=args.global_prompt_batch_size,
                    num_workers=args.dataloader_num_workers,
                    seed=args.seed,
                    epoch=_epoch,
                )
                for batch_index, prompt_batch in enumerate(dataloader):
                    if _epoch == start_epoch and batch_index < skip_batches_in_start_epoch:
                        continue
                    payload = build_training_payload(
                        args=args,
                        training_objective=training_objective,
                        rollout_builder=rollout_builder,
                        policy_batch_builder=policy_batch_builder,
                        pool=pool,
                        vllm_sampling_kwargs=vllm_sampling_kwargs,
                        prompt_batch=prompt_batch,
                        update_token_count=True,
                    )
                    if training_objective.name == "policy_optimization":
                        valid_sequence_mask = payload.training_batch.get("valid_sequence_mask")
                        if torch.is_tensor(valid_sequence_mask) and not bool(valid_sequence_mask.any()):
                            logger.warning(
                                "Skipping policy batch at epoch=%s batch_index=%s because every sampled completion "
                                "was invalid for training.",
                                _epoch,
                                batch_index,
                            )
                            continue
                    global_step, metrics = run_training_iteration(
                        payload=payload,
                        training_objective=training_objective,
                        model_group=model_group,
                        config=config,
                        prompts_per_worker=prompts_per_worker,
                        pool=pool,
                    )
                    epochs_completed = _compute_epochs_completed(
                        global_step,
                        steps_per_epoch=steps_per_epoch,
                    )
                    progress_metrics = {"progress/epochs_completed": epochs_completed}
                    wandb_payload: dict[str, object] = {}
                    wandb_payload.update(_build_wandb_metrics_payload("train", progress_metrics))
                    if should_log_training_samples(config, global_step):
                        rendered_sample_texts = _render_sample_texts(payload.sample_texts)
                        if rendered_sample_texts:
                            logger.info("step=%s sample\n%s", global_step, rendered_sample_texts)
                        wandb_payload.update(
                            _build_wandb_batch_samples_payload(global_step, payload.sample_texts)
                        )
                    if global_step % config.logging_steps == 0:
                        metrics_with_progress = dict(metrics)
                        metrics_with_progress.update(progress_metrics)
                        _log_metrics(global_step, metrics_with_progress)
                        wandb_payload.update(_build_wandb_metrics_payload("train", metrics_with_progress))
                    if should_run_dev_eval(config, global_step):
                        eval_metrics = run_dev_eval(
                            model_group=model_group,
                            config=config,
                            rollout_builder=rollout_builder,
                            eval_state=dev_eval_state,
                            pool=pool,
                            training_objective=training_objective,
                            policy_batch_builder=policy_batch_builder,
                        )
                        _log_eval_metrics(global_step, eval_metrics)
                        wandb_payload.update(_build_wandb_metrics_payload("eval", eval_metrics))
                        wandb_payload.update(
                            _build_wandb_eval_samples_payload(
                                global_step,
                                _get_logged_startup_eval_samples(dev_eval_state),
                            )
                        )
                        _maybe_save_best_checkpoint(
                            config=config,
                            model_group=model_group,
                            eval_metrics=eval_metrics,
                            global_step=global_step,
                            best_state=best_checkpoint_state,
                            wandb_enabled=wandb_enabled,
                        )
                    if global_step % config.save_steps == 0:
                        checkpoint_name = f"checkpoint-{global_step}"
                        ray.get(model_group.async_run_method("save_checkpoint", checkpoint_name))
                        _log_checkpoint_saved(
                            checkpoint_name=checkpoint_name,
                            global_step=global_step,
                            output_dir=config.output_dir,
                        )
                        wandb_payload.update(_build_wandb_checkpoint_payload(global_step))
                        if wandb_enabled and wandb.run is not None:
                            wandb.run.summary["latest_checkpoint"] = os.path.join(
                                config.output_dir,
                                checkpoint_name,
                            )
                    _log_wandb_payload(global_step, wandb_payload, wandb_enabled=wandb_enabled)

        ray.get(model_group.async_run_method("save_checkpoint", "final"))
        final_state = ray.get(model_group.actor_handlers[0].get_training_state.remote())
        final_step = int(final_state["global_step"])
        _log_checkpoint_saved(
            checkpoint_name="final",
            global_step=final_step,
            output_dir=config.output_dir,
        )
        if wandb_enabled and wandb.run is not None:
            _log_wandb_payload(final_step, _build_wandb_checkpoint_payload(final_step), wandb_enabled=True)
            wandb.run.summary["latest_checkpoint"] = os.path.join(config.output_dir, "final")
    finally:
        destroy_policy_judge_vllm_engines(pool)
        destroy_trained_genrm_vllm_engines(pool)
        ray.shutdown()
        _finish_wandb(wandb_enabled)


if __name__ == "__main__":
    main()
