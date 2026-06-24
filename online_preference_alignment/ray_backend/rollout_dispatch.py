"""vLLM rollout dispatch and training iteration orchestration.

Owns the low-level prompt dispatch to vLLM engines, engine lifecycle helpers
(weight sync, engine rebuild), and the per-step training payload / iteration
runners used by the coordinator main loop.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import ray
import torch

from experience import TrajectoryScoreContext
from objectives import build_training_objective
from ray_backend.launcher import RayActorGroup
from ray_backend.rollout_utils import (
    OfflineRolloutBatchBuilder,
    PolicyOptimizationPackage,
    TrainingBatchPackage,
)
from ray_backend.vllm_engine import (
    batch_vllm_engine_call,
    create_offline_vllm_engine_bundle,
    destroy_offline_vllm_engine_bundle,
)
from training.batching import shard_training_batch_for_workers


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state bundle for vLLM engine handles
# ---------------------------------------------------------------------------


@dataclass
class VLLMEnginePool:
    """Mutable bundle of vLLM engine handles and their coordination state.

    Replaces the (vllm_engines, vllm_lock, vllm_state) parameter triple that
    was previously threaded through every coordinator helper.
    """

    engines: list
    lock: threading.RLock
    shared_pg: Any = None
    rollout_engine_bundle_pg: Any = None
    rollout_engine_create_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_judge_engines: list = field(default_factory=list)
    policy_judge_engine_bundle_pg: Any = None
    policy_judge_model_path: str | None = None
    policy_judge_max_model_len: int | None = None
    trained_genrm_engines: list = field(default_factory=list)
    trained_genrm_engine_bundle_pg: Any = None
    trained_genrm_model_path: str | None = None
    trained_genrm_max_model_len: int | None = None


# ---------------------------------------------------------------------------
# Rollout requirements helpers
# ---------------------------------------------------------------------------


def _objective_rollout_requirements(args):
    return build_training_objective(args.training_objective).rollout_requirements(args)


def requested_vllm_logprob_count(args) -> int | None:
    requirements = _objective_rollout_requirements(args)
    return 1 if requirements.include_sampled_logprobs else None


def resolve_vllm_num_engines(config) -> int:
    num_engines = int(getattr(config, "ray_vllm_num_engines", 1))
    if num_engines <= 0:
        raise ValueError("ray_vllm_num_engines must be > 0.")
    return num_engines


# ---------------------------------------------------------------------------
# Engine lifecycle helpers
# ---------------------------------------------------------------------------


def sync_actor_weights_to_vllm(
    *,
    model_group: RayActorGroup,
    config,
    pool: VLLMEnginePool,
) -> None:
    if not pool.engines:
        return
    with pool.lock:
        if config.vllm_enable_sleep_mode:
            # Wake weights only so sync avoids allocating KV cache on top of
            # training-time memory pressure.
            batch_vllm_engine_call(pool.engines, "wake_up", tags=("weights",))
        ray.get(model_group.async_run_method("broadcast_to_vllm"))
        # Keep the rollout engines in a weights-ready state after sync. The
        # next generation/eval round will wake any remaining resources and then
        # return the engines to level-1 sleep after use.


def replace_vllm_engines(pool: VLLMEnginePool, new_engines: list) -> None:
    pool.engines.clear()
    pool.engines.extend(new_engines)


def destroy_policy_judge_vllm_engines(pool: VLLMEnginePool) -> None:
    """Destroy the persistent frozen policy-judge engine bundle, if present."""
    with pool.lock:
        if not pool.policy_judge_engines:
            return
        destroy_offline_vllm_engine_bundle(
            list(pool.policy_judge_engines),
            pool.policy_judge_engine_bundle_pg,
        )
        pool.policy_judge_engines.clear()
        pool.policy_judge_engine_bundle_pg = None
        pool.policy_judge_model_path = None
        pool.policy_judge_max_model_len = None


def destroy_trained_genrm_vllm_engines(pool: VLLMEnginePool) -> None:
    """Destroy the persistent frozen trained-genRM engine bundle, if present."""
    with pool.lock:
        if not pool.trained_genrm_engines:
            return
        destroy_offline_vllm_engine_bundle(
            list(pool.trained_genrm_engines),
            pool.trained_genrm_engine_bundle_pg,
        )
        pool.trained_genrm_engines.clear()
        pool.trained_genrm_engine_bundle_pg = None
        pool.trained_genrm_model_path = None
        pool.trained_genrm_max_model_len = None


def _ensure_frozen_vllm_engines(
    *,
    pool: VLLMEnginePool,
    config,
    model_path: str,
    seed: int,
    max_model_len: int | None,
    engine_attr: str,
    pg_attr: str,
    model_attr: str,
    max_len_attr: str,
    destroy_existing,
    label: str,
) -> list:
    requested_max_model_len = None if max_model_len is None else int(max_model_len)
    with pool.lock:
        current_engines = getattr(pool, engine_attr)
        current_model_path = getattr(pool, model_attr)
        has_compatible_engine = bool(current_engines) and current_model_path == model_path
        if has_compatible_engine and requested_max_model_len is not None:
            current_max_model_len = getattr(pool, max_len_attr)
            has_compatible_engine = (
                current_max_model_len is None
                or current_max_model_len >= requested_max_model_len
            )
        if has_compatible_engine:
            return list(current_engines)

        if current_engines:
            logger.info(
                "Recreating persistent frozen %s vLLM engines "
                "(model=%s requested_max_model_len=%s current_model=%s current_max_model_len=%s).",
                label,
                model_path,
                requested_max_model_len,
                current_model_path,
                getattr(pool, max_len_attr),
            )
            destroy_existing(pool)
        else:
            logger.info(
                "Creating persistent frozen %s vLLM engines "
                "for model=%s max_model_len=%s.",
                label,
                model_path,
                requested_max_model_len,
            )

        engines, placement_group_obj = create_offline_vllm_engine_bundle(
            num_engines=resolve_vllm_num_engines(config),
            tensor_parallel_size=config.vllm_tensor_parallel_size,
            model_name=model_path,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
            seed=int(seed),
            full_determinism=False,
            max_model_len=requested_max_model_len,
            max_logprobs=None,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            vllm_enable_sleep=config.vllm_enable_sleep_mode,
            vllm_enforce_eager=config.vllm_enforce_eager,
            log_phase_progress=config.log_phase_progress,
            shared_pg=pool.shared_pg,
        )
        setattr(pool, engine_attr, engines)
        setattr(pool, pg_attr, placement_group_obj)
        setattr(pool, model_attr, model_path)
        setattr(pool, max_len_attr, requested_max_model_len)
        return list(engines)


def ensure_policy_judge_vllm_engines(
    *,
    pool: VLLMEnginePool,
    config,
    model_path: str,
    seed: int,
    max_model_len: int | None,
) -> list:
    """Return a persistent frozen policy-judge vLLM engine bundle.

    Policy-judge rewards should be scored by a frozen model, not by the rollout
    engine whose weights are synced from the actively trained policy. This
    helper creates that frozen engine once and reuses it across training batches;
    it only rebuilds if the requested context window grows beyond the current
    engine's ``max_model_len``.
    """
    return _ensure_frozen_vllm_engines(
        pool=pool,
        config=config,
        model_path=model_path,
        seed=seed,
        max_model_len=max_model_len,
        engine_attr="policy_judge_engines",
        pg_attr="policy_judge_engine_bundle_pg",
        model_attr="policy_judge_model_path",
        max_len_attr="policy_judge_max_model_len",
        destroy_existing=destroy_policy_judge_vllm_engines,
        label="policy-judge",
    )


def ensure_trained_genrm_vllm_engines(
    *,
    pool: VLLMEnginePool,
    config,
    model_path: str,
    seed: int,
    max_model_len: int | None,
) -> list:
    """Return a persistent frozen trained-genRM vLLM engine bundle."""
    return _ensure_frozen_vllm_engines(
        pool=pool,
        config=config,
        model_path=model_path,
        seed=seed,
        max_model_len=max_model_len,
        engine_attr="trained_genrm_engines",
        pg_attr="trained_genrm_engine_bundle_pg",
        model_attr="trained_genrm_model_path",
        max_len_attr="trained_genrm_max_model_len",
        destroy_existing=destroy_trained_genrm_vllm_engines,
        label="trained-genRM",
    )


@contextmanager
def suspended_rollout_engines(
    *,
    model_group: RayActorGroup,
    config,
    pool: VLLMEnginePool,
):
    """Free GPU memory held by rollout engines for the duration of the block.

    Destroys rollout engines while temporary judge/reference/RM engines run,
    then rebuilds and re-syncs them afterward. This is slower than vLLM level-2
    sleep, but avoids CuMem wake-up failures when trying to change sleep levels
    on an already-sleeping colocated Ray engine.
    """
    with pool.lock:
        if pool.engines:
            destroy_offline_vllm_engine_bundle(list(pool.engines), pool.rollout_engine_bundle_pg)
            replace_vllm_engines(pool, [])
            pool.rollout_engine_bundle_pg = None
        try:
            yield None
        finally:
            rebuild_rollout_vllm_engines(model_group=model_group, config=config, pool=pool)


def rebuild_rollout_vllm_engines(
    *,
    model_group: RayActorGroup,
    config,
    pool: VLLMEnginePool,
) -> None:
    rollout_engine_create_kwargs = dict(pool.rollout_engine_create_kwargs)
    logger.info("Recreating rollout vLLM engines after reference eval.")
    new_engines, rollout_engine_bundle_pg = create_offline_vllm_engine_bundle(
        **rollout_engine_create_kwargs,
    )
    replace_vllm_engines(pool, new_engines)
    pool.rollout_engine_bundle_pg = rollout_engine_bundle_pg
    ray.get(
        model_group.async_run_method(
            "set_vllm_engines",
            vllm_engines=pool.engines,
            vllm_num_engines=len(pool.engines),
        )
    )
    sync_actor_weights_to_vllm(model_group=model_group, config=config, pool=pool)


# ---------------------------------------------------------------------------
# Prompt dispatch to vLLM engines
# ---------------------------------------------------------------------------


def dispatch_prompts(
    *,
    args,
    vllm_engines: list,
    vllm_sampling_kwargs: dict[str, Any],
    prompt_token_ids_batch: list[list[int]],
    include_logprobs: bool,
    requested_logprob_count: int | None = None,
    expected_num_generations: int | None = None,
) -> tuple[list[list[int]], dict[str, torch.Tensor] | None]:
    dispatch_start = time.monotonic()
    if not prompt_token_ids_batch:
        return [], None
    if not vllm_engines:
        raise ValueError("Ray rollout generation requires at least one offline vLLM engine.")

    num_engines = len(vllm_engines)
    chunk_size = math.ceil(len(prompt_token_ids_batch) / num_engines)
    request_sampling_kwargs = dict(vllm_sampling_kwargs)
    if requested_logprob_count is not None:
        request_sampling_kwargs["logprobs"] = requested_logprob_count
    else:
        request_sampling_kwargs.pop("logprobs", None)

    refs = []
    for engine_index, start in enumerate(range(0, len(prompt_token_ids_batch), chunk_size)):
        prompt_chunk = prompt_token_ids_batch[start : start + chunk_size]
        use_tqdm = args.vllm_show_progress and engine_index == 0
        refs.append(
            vllm_engines[engine_index].generate.remote(
                prompt_chunk,
                request_sampling_kwargs,
                use_tqdm=use_tqdm,
                include_sampled_logprobs=include_logprobs,
            )
        )

    completion_sequences: list[list[int]] = []
    completion_vllm_diagnostics = None
    sequence_offset_parts: list[torch.Tensor] = []
    logprob_parts: list[torch.Tensor] = []
    total_completion_tokens = 0

    fetch_start = time.monotonic()
    engine_result_batches = ray.get(refs)
    fetch_duration = time.monotonic() - fetch_start
    merge_start = time.monotonic()
    for engine_results in engine_result_batches:
        completion_sequences.extend(engine_results["sequences"])
        total_completion_tokens += sum(len(sequence) for sequence in engine_results["sequences"])
        if include_logprobs:
            sequence_offsets = engine_results["vllm_sequence_offsets"]
            if not torch.is_tensor(sequence_offsets):
                raise RuntimeError("Offline vLLM diagnostics must include vllm_sequence_offsets.")
            if sequence_offsets.ndim != 1:
                raise ValueError("vllm_sequence_offsets must be a 1D tensor.")
            if sequence_offsets.numel() > 1:
                previous_total = total_completion_tokens - int(sequence_offsets[-1].item())
                sequence_offset_parts.append(sequence_offsets[1:] + previous_total)
            logprobs_flat = engine_results["vllm_logprobs_flat"]
            if not torch.is_tensor(logprobs_flat):
                raise RuntimeError("Offline vLLM diagnostics must include vllm_logprobs_flat.")
            logprob_parts.append(logprobs_flat)

    expected = len(prompt_token_ids_batch) * (
        int(expected_num_generations) if expected_num_generations is not None else args.num_generations
    )
    if len(completion_sequences) != expected:
        raise RuntimeError(
            f"Offline vLLM returned {len(completion_sequences)} completions but expected {expected}."
        )
    if include_logprobs:
        completion_vllm_diagnostics = {
            "vllm_sequence_offsets": torch.cat(
                [torch.zeros(1, dtype=torch.int32), *sequence_offset_parts]
            ),
            "vllm_logprobs_flat": (
                torch.cat(logprob_parts) if logprob_parts else torch.empty(0, dtype=torch.float32)
            ),
        }
        if completion_vllm_diagnostics["vllm_sequence_offsets"].numel() != expected + 1:
            raise RuntimeError(
                "Offline vLLM returned packed sequence offsets for "
                f"{completion_vllm_diagnostics['vllm_sequence_offsets'].numel() - 1} completions but expected {expected}."
            )

    if args.log_phase_progress:
        logger.info(
            "stage=rollout phase=dispatch_prompts status=done prompts=%s completions=%s completion_tokens=%s "
            "ray_get_seconds=%.4f merge_seconds=%.4f total_seconds=%.4f",
            len(prompt_token_ids_batch),
            len(completion_sequences),
            total_completion_tokens,
            fetch_duration,
            time.monotonic() - merge_start,
            time.monotonic() - dispatch_start,
        )
    return completion_sequences, completion_vllm_diagnostics


# ---------------------------------------------------------------------------
# Training payload construction
# ---------------------------------------------------------------------------


def build_training_payload(
    *,
    args,
    training_objective,
    rollout_builder: OfflineRolloutBatchBuilder,
    policy_batch_builder,
    pool: VLLMEnginePool,
    vllm_sampling_kwargs: dict[str, Any],
    prompt_batch: list[dict[str, Any]],
    update_token_count: bool,
) -> TrainingBatchPackage:
    build_start = time.monotonic()

    prompt_build_start = time.monotonic()
    prompts = [example["prompt"] for example in prompt_batch]
    prompt_texts = [rollout_builder._format_student_prompt(prompt) for prompt in prompts]
    prompt_token_ids_batch = rollout_builder._tokenize_prompt_text_sequences(prompt_texts)
    prompt_tensors = rollout_builder._build_prompt_tensors_from_token_sequences(prompt_token_ids_batch)
    prompt_build_duration = time.monotonic() - prompt_build_start

    rollout_requirements = training_objective.rollout_requirements(args)
    include_logprobs = rollout_requirements.include_sampled_logprobs
    logprob_count = requested_vllm_logprob_count(args)

    dispatch_duration = 0.0
    with pool.lock:
        if args.vllm_enable_sleep_mode:
            batch_vllm_engine_call(pool.engines, "wake_up")
        dispatch_start = time.monotonic()
        completion_sequences, completion_vllm_diagnostics = dispatch_prompts(
            args=args,
            vllm_engines=pool.engines,
            vllm_sampling_kwargs=vllm_sampling_kwargs,
            prompt_token_ids_batch=prompt_token_ids_batch,
            include_logprobs=include_logprobs,
            requested_logprob_count=logprob_count,
        )
        dispatch_duration = time.monotonic() - dispatch_start
        if args.vllm_enable_sleep_mode:
            batch_vllm_engine_call(pool.engines, "sleep")

    prepare_start = time.monotonic()
    if training_objective.name == "policy_optimization":
        if policy_batch_builder is None:
            raise RuntimeError("policy_optimization requires a policy_batch_builder.")
        trajectory_package = rollout_builder.prepare_trajectory_batch(
            prompt_batch,
            completion_sequences=completion_sequences,
            completion_vllm_diagnostics=completion_vllm_diagnostics,
            update_token_count=update_token_count,
            prompt_texts=prompt_texts,
            prompt_tensors=prompt_tensors,
        )
        policy_batch_result = policy_batch_builder.build_batch(
            trajectory_package.trajectory_batch,
            score_context=TrajectoryScoreContext(
                examples=prompt_batch,
                prompt_texts=prompt_texts,
            ),
        )
        sample_texts = dict(trajectory_package.sample_texts or {})
        if policy_batch_result.sample_texts:
            sample_texts.update(policy_batch_result.sample_texts)
        package = PolicyOptimizationPackage(
            training_batch=policy_batch_result.batch,
            rollout_metrics={
                **trajectory_package.rollout_metrics,
                **policy_batch_result.metrics,
            },
            num_input_tokens_seen=trajectory_package.num_input_tokens_seen,
            sample_texts=sample_texts or None,
        )
    else:
        raise ValueError(f"Unsupported training_objective: {training_objective.name!r}")
    prepare_duration = time.monotonic() - prepare_start

    if args.log_phase_progress:
        total_completion_tokens = int(package.training_batch["completion_mask"].sum().item())
        logger.info(
            "stage=rollout phase=build_training_payload status=done objective=%s prompts=%s completions=%s completion_tokens=%s "
            "prompt_build_seconds=%.4f dispatch_seconds=%.4f prepare_seconds=%.4f total_seconds=%.4f",
            training_objective.name,
            len(prompt_batch),
            len(completion_sequences),
            total_completion_tokens,
            prompt_build_duration,
            dispatch_duration,
            prepare_duration,
            time.monotonic() - build_start,
        )
    return package
# ---------------------------------------------------------------------------
# Single training step execution
# ---------------------------------------------------------------------------


def run_training_iteration(
    *,
    payload: TrainingBatchPackage,
    training_objective,
    model_group: RayActorGroup,
    config,
    prompts_per_worker: int,
    pool: VLLMEnginePool,
) -> tuple[int, dict[str, float]]:
    worker_batches = shard_training_batch_for_workers(
        payload.training_batch,
        layout=training_objective.training_batch_layout(),
        world_size=config.world_size,
        prompts_per_worker=prompts_per_worker,
        num_generations=config.num_generations,
    )
    if config.log_phase_progress:
        logger.info(
            "stage=rollout phase=shard_for_workers status=done workers=%s prompts_per_worker=%s total_prompts=%s",
            config.world_size,
            prompts_per_worker,
            payload.training_batch["num_prompts"],
        )
    payload.training_batch = {
        key: value
        for key, value in payload.training_batch.items()
        if not isinstance(value, torch.Tensor)
    }

    train_refs = [
        actor.train_step.remote(
            worker_batch,
            num_input_tokens_seen=payload.num_input_tokens_seen,
        )
        for actor, worker_batch in zip(model_group.actor_handlers, worker_batches)
    ]
    train_results = ray.get(train_refs)

    rank_zero_result = next(result for result in train_results if result["rank"] == 0)
    global_step = rank_zero_result["global_step"]

    sync_actor_weights_to_vllm(model_group=model_group, config=config, pool=pool)

    metrics = dict(payload.rollout_metrics)
    metrics.update(rank_zero_result["metrics"])
    return global_step, metrics
