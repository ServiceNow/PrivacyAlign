"""preference dev evaluation, reference response caching, and temporary vLLM engine management.

All evaluation-time vLLM orchestration lives here so that the main coordinator
loop stays focused on the training cadence.
"""

from __future__ import annotations

import gc
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ray
import torch
from transformers import AutoTokenizer

from data_loaders import load_prompt
from data_loaders.preference import (
    is_privalign_dataset_name,
    load_preference_validation_dataset,
    load_privalign_pairwise_genrm_dataset,
    stable_text_hash_key,
)
from experience import (
    PairwiseMarginScorer,
    PrivalignPairwiseMarginScorer,
    PrivalignRLPairwiseJudgeScorer,
    TrajectoryScoreContext,
    TrainedGenRMScorer,
    build_policy_training_batch_builder,
)
from judges.pairwise import (
    build_privalign_rl_judge_prompt_token_ids_batch,
    build_privalign_rl_pointwise_judge_prompt_token_ids_batch,
)
from utils.eval_helpers import (
    build_eval_engine_seed,
    summarize_pairwise_dev_eval_metrics,
    summarize_pointwise_dev_eval_metrics,
    summarize_reference_word_count_metrics,
)
from ray_backend.launcher import RayActorGroup
from ray_backend.rollout_dispatch import (
    VLLMEnginePool,
    dispatch_prompts,
    ensure_policy_judge_vllm_engines,
    ensure_trained_genrm_vllm_engines,
    requested_vllm_logprob_count,
    resolve_vllm_num_engines,
    suspended_rollout_engines,
)
from ray_backend.rollout_utils import OfflineRolloutBatchBuilder
from training.batching import shard_training_batch_for_workers
from training.config import (
    TRAINED_GENRM_POLICY_SCORERS,
    resolve_trained_genrm_prompt_template,
)
from ray_backend.vllm_engine import (
    batch_vllm_engine_call,
    create_offline_vllm_engine_bundle,
    destroy_offline_vllm_engine_bundle,
)
from utils.text_utils import decode_response_tokens, strip_thinking_trace
logger = logging.getLogger(__name__)
PRIVALIGN_RL_JUDGE_TEMPLATE_NAME = "privalign_rl_pairwise_judge"
PRIVALIGN_RL_POINTWISE_JUDGE_TEMPLATE_NAME = "privalign_rl_pointwise_judge"
STARTUP_EVAL_WANDB_SAMPLE_COUNT = 5
PAIRWISE_NO_RESPONSE_CANDIDATE_FIRST_OUTPUT = (
    "Reasoning: The candidate did not produce a usable response.\n"
    "Score: 2"
)
PAIRWISE_NO_RESPONSE_REFERENCE_FIRST_OUTPUT = (
    "Reasoning: The candidate did not produce a usable response.\n"
    "Score: -2"
)


# ---------------------------------------------------------------------------
# Dev-eval state
# ---------------------------------------------------------------------------


@dataclass
class DevEvalState:
    eval_kind: str
    examples: list[dict[str, Any]]
    prompt_texts: list[str]
    prompt_token_ids_batch: list[list[int]]
    judge_template: str
    reference_model_path: str
    judge_model_path: str
    judge_rollout_builder: OfflineRolloutBatchBuilder
    rubric_items_batch: list[list[dict[str, Any]]]
    eval_prompts_for_judge: list[str]
    reference_responses: list[str] | None = None
    raw_reference_responses: list[str] | None = None
    logged_sample_index: int | None = None
    logged_eval_sample_texts: dict[str, str] | None = None
    logged_eval_sample_texts_batch: list[dict[str, str]] | None = None


# ---------------------------------------------------------------------------
# vLLM text generation helpers
# ---------------------------------------------------------------------------


def _build_eval_vllm_sampling_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    num_generations: int = 1,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0.")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0 for eval/judge generation.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the range (0, 1].")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0 when set.")
    if repetition_penalty is not None and repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be > 0 when set.")
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1.")
    sampling_kwargs = {
        "n": int(num_generations),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": max_new_tokens,
        "detokenize": False,
    }
    if top_k is not None:
        sampling_kwargs["top_k"] = int(top_k)
    if presence_penalty is not None:
        sampling_kwargs["presence_penalty"] = float(presence_penalty)
    if repetition_penalty is not None:
        sampling_kwargs["repetition_penalty"] = float(repetition_penalty)
    return sampling_kwargs


def _decode_completion_sequences(
    rollout_builder: OfflineRolloutBatchBuilder,
    completion_sequences: list[list[int]],
    *,
    strip_thinking_traces: bool = True,
) -> list[str]:
    return [
        decode_response_tokens(
            rollout_builder.processing_class,
            sequence,
            strip_thinking_traces=strip_thinking_traces,
        )
        for sequence in completion_sequences
    ]


def _completion_is_unusable_for_eval(
    rollout_builder: OfflineRolloutBatchBuilder,
    completion_sequence: list[int],
) -> bool:
    if not completion_sequence:
        return True
    try:
        if not rollout_builder._completion_has_response_tokens(completion_sequence):
            return True
    except Exception:
        # Fall back to decoded text below for lightweight test doubles that do not
        # expose the rollout-builder response-mask helper.
        pass
    text = decode_response_tokens(rollout_builder.processing_class, completion_sequence)
    return not text.strip()


def _release_cuda_cache_after_eval() -> None:
    """Defragment the CUDA allocator after a dev-eval wake/sleep cycle.

    vLLM holding 60–70%% of VRAM plus repeated wake/sleep cycles leave the
    PyTorch allocator with multi-GB of reserved-but-unallocated memory. On the
    next training backward this fragmentation can fail an otherwise-fitting
    allocation. ``gc.collect()`` + ``torch.cuda.empty_cache()`` returns those
    reserved blocks to the allocator. Cheap (~milliseconds) and safe.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _summarize_no_response_eval_metrics(
    unusable_mask: list[bool],
    *,
    metric_prefix: str,
) -> dict[str, float]:
    total = len(unusable_mask)
    count = sum(1 for value in unusable_mask if value)
    prefix = metric_prefix.rstrip("/")
    return {
        f"{prefix}/no_response_count": float(count),
        f"{prefix}/no_response_rate": (float(count) / float(total)) if total > 0 else 0.0,
    }


def _filter_completion_diagnostics_by_sequence_indices(
    completion_vllm_diagnostics: dict[str, torch.Tensor] | None,
    sequence_indices: list[int],
) -> dict[str, torch.Tensor] | None:
    if completion_vllm_diagnostics is None:
        return None
    offsets = completion_vllm_diagnostics.get("vllm_sequence_offsets")
    if not torch.is_tensor(offsets):
        return dict(completion_vllm_diagnostics)
    if not sequence_indices:
        filtered: dict[str, torch.Tensor] = {
            "vllm_sequence_offsets": offsets.new_zeros((1,)),
        }
        tensor = completion_vllm_diagnostics.get("vllm_logprobs_flat")
        if torch.is_tensor(tensor):
            filtered["vllm_logprobs_flat"] = tensor[:0]
        return filtered

    new_offsets = [0]
    logprob_slices: list[torch.Tensor] = []
    logprobs_flat = completion_vllm_diagnostics.get("vllm_logprobs_flat")
    total_tokens = 0
    for sequence_index in sequence_indices:
        start = int(offsets[sequence_index].item())
        end = int(offsets[sequence_index + 1].item())
        total_tokens += end - start
        new_offsets.append(total_tokens)
        if torch.is_tensor(logprobs_flat):
            logprob_slices.append(logprobs_flat[start:end])

    filtered = {
        "vllm_sequence_offsets": torch.tensor(
            new_offsets,
            dtype=offsets.dtype,
            device=offsets.device,
        )
    }
    if torch.is_tensor(logprobs_flat):
        filtered["vllm_logprobs_flat"] = (
            torch.cat(logprob_slices, dim=0) if logprob_slices else logprobs_flat[:0]
        )
    return filtered


def _resolve_eval_candidate_temperature(args) -> float:
    """Sampling temperature for eval-candidate generation (student/reference/teacher).

    Prefers ``eval_reference_vllm_temperature`` when set, otherwise falls back to
    ``eval_judge_vllm_temperature``, then to the training rollout temperature.
    """
    ref_temp = getattr(args, "eval_reference_vllm_temperature", None)
    if ref_temp is not None:
        return float(ref_temp)
    return float(
        getattr(
            args,
            "eval_judge_vllm_temperature",
            getattr(args, "train_vllm_temperature", 0.6),
        )
    )


def _resolve_eval_candidate_top_p(args) -> float:
    """Top-p sampling for eval-candidate generation (student/reference/teacher)."""
    return float(getattr(args, "eval_reference_vllm_top_p", 0.95))


def _resolve_eval_candidate_top_k(args) -> int | None:
    """Top-k sampling for eval-candidate generation (student/reference/teacher)."""
    top_k = getattr(args, "eval_reference_vllm_top_k", 20)
    return None if top_k is None else int(top_k)


def _resolve_eval_candidate_repetition_penalty(args) -> float | None:
    """Repetition penalty for eval-candidate generation (student/reference/teacher)."""
    rep_penalty = getattr(args, "eval_reference_vllm_repetition_penalty", None)
    return None if rep_penalty is None else float(rep_penalty)


def _resolve_eval_candidate_presence_penalty(args) -> float | None:
    """Presence penalty for eval-candidate generation (student/reference/teacher)."""
    presence_penalty = getattr(args, "eval_reference_vllm_presence_penalty", None)
    return None if presence_penalty is None else float(presence_penalty)


def _resolve_eval_judge_temperature(args) -> float:
    """Sampling temperature for LLM-judge scoring."""
    return float(
        getattr(
            args,
            "eval_judge_vllm_temperature",
            getattr(args, "train_vllm_temperature", 0.6),
        )
    )


def generate_texts_with_vllm(
    *,
    args,
    rollout_builder: OfflineRolloutBatchBuilder,
    vllm_engines: list,
    prompt_token_ids_batch: list[list[int]],
    max_new_tokens: int,
    strip_thinking_traces: bool = True,
    top_p: float = 1.0,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    num_generations: int = 1,
    temperature: float | None = None,
) -> list[str]:
    if temperature is None:
        eval_temperature = _resolve_eval_judge_temperature(args)
    else:
        eval_temperature = float(temperature)
    completion_sequences, _ = dispatch_prompts(
        args=args,
        vllm_engines=vllm_engines,
        vllm_sampling_kwargs=_build_eval_vllm_sampling_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=eval_temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            num_generations=num_generations,
        ),
        prompt_token_ids_batch=prompt_token_ids_batch,
        include_logprobs=False,
        expected_num_generations=num_generations,
    )
    return _decode_completion_sequences(
        rollout_builder,
        completion_sequences,
        strip_thinking_traces=strip_thinking_traces,
    )


# ---------------------------------------------------------------------------
# Temporary eval vLLM engine context
# ---------------------------------------------------------------------------


def _temporary_engine_max_model_len(
    prompt_token_ids_batch: list[list[int]],
    *,
    max_new_tokens: int,
) -> int:
    max_prompt_tokens = max((len(token_ids) for token_ids in prompt_token_ids_batch), default=0)
    # Leave a tiny cushion for chat-template/control-token drift between prompt
    # construction and vLLM validation without falling back to the model's full
    # native context window.
    return max_prompt_tokens + int(max_new_tokens) + 16


def _default_temporary_engine_max_model_len(config) -> int | None:
    max_prompt_length = getattr(config, "max_prompt_length", None)
    max_completion_length = getattr(config, "max_completion_length", None)
    if max_prompt_length is None or max_completion_length is None:
        return None
    return int(max_prompt_length) + int(max_completion_length) + 16


@contextmanager
def _suspend_rollout_vllm_engines(
    *,
    model_group: RayActorGroup,
    config,
    pool: VLLMEnginePool,
):
    """Temporarily free the GPU memory held by rollout engines for eval."""
    with suspended_rollout_engines(model_group=model_group, config=config, pool=pool):
        yield


@contextmanager
def _temporary_loaded_eval_vllm_engines(
    *,
    config,
    pool: VLLMEnginePool,
    model_path: str,
    seed: int | None = None,
    max_model_len: int | None = None,
):
    """Spin up a temporary eval engine bundle without touching rollout-engine lifecycle."""
    with pool.lock:
        eval_engines = []
        eval_pg = None
        resolved_seed = int(config.seed if seed is None else seed)
        resolved_max_model_len = (
            int(max_model_len)
            if max_model_len is not None
            else _default_temporary_engine_max_model_len(config)
        )
        logger.info(
            "Creating temporary eval vLLM engine for model=%s max_model_len=%s.",
            model_path,
            resolved_max_model_len,
        )
        try:
            eval_engines, eval_pg = create_offline_vllm_engine_bundle(
                num_engines=resolve_vllm_num_engines(config),
                tensor_parallel_size=config.vllm_tensor_parallel_size,
                model_name=model_path,
                dtype=config.dtype,
                trust_remote_code=config.trust_remote_code,
                seed=resolved_seed,
                full_determinism=False,
                max_model_len=resolved_max_model_len,
                max_logprobs=None,
                gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                vllm_enable_sleep=False,
                vllm_enforce_eager=config.vllm_enforce_eager,
                log_phase_progress=config.log_phase_progress,
                shared_pg=pool.shared_pg,
            )
            yield eval_engines
        finally:
            destroy_offline_vllm_engine_bundle(eval_engines, eval_pg)


@contextmanager
def temporary_eval_vllm_engines(
    *,
    model_group: RayActorGroup,
    config,
    pool: VLLMEnginePool,
    model_path: str,
    seed: int | None = None,
    max_model_len: int | None = None,
):
    """Destroy rollout engines, spin up a single eval engine, then restore rollout engines."""
    with _suspend_rollout_vllm_engines(
        model_group=model_group,
        config=config,
        pool=pool,
    ):
        with _temporary_loaded_eval_vllm_engines(
            config=config,
            pool=pool,
            model_path=model_path,
            seed=seed,
            max_model_len=max_model_len,
        ) as eval_engines:
            yield eval_engines


# ---------------------------------------------------------------------------
# preference dev-eval construction and execution
# ---------------------------------------------------------------------------


def build_dev_eval_state(
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    reference_model_path: str,
    judge_model_path: str,
) -> DevEvalState:
    # Short-circuit for the Phase A pairwise-margin scorer: dev eval here is
    # "run the in-training gen-RM on the dev split, parse Score, compute
    # pair_accuracy" -- no judge model, no candidate vs reference comparison.
    if (
        getattr(config, "policy_scorer", None) == "privalign_pairwise_margin"
        and is_privalign_dataset_name(dataset_name)
    ):
        return _build_pairwise_margin_dev_eval_state(
            dataset_name=dataset_name,
            config=config,
            rollout_builder=rollout_builder,
        )

    raw_dataset = load_preference_validation_dataset(
        dataset_name,
        split="validation",
        dataset_config=dataset_config,
    )
    if config.dev_eval_max_samples is not None:
        max_samples = min(len(raw_dataset), int(config.dev_eval_max_samples))
        if hasattr(raw_dataset, "select"):
            raw_dataset = raw_dataset.select(range(max_samples))
        else:
            raw_dataset = raw_dataset[:max_samples]
    if is_privalign_dataset_name(dataset_name):
        return _build_privalign_dev_eval_state(
            raw_dataset=raw_dataset,
            dataset_name=dataset_name,
            config=config,
            rollout_builder=rollout_builder,
            reference_model_path=reference_model_path,
            judge_model_path=judge_model_path,
        )
    raise ValueError(f"Unsupported dev eval dataset_name={dataset_name!r}.")


def _build_pairwise_margin_dev_eval_state(
    *,
    dataset_name: str,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
) -> DevEvalState:
    """Lightweight dev-eval state for Phase A gen-RM training.

    Loads pairwise dev rows, tokenizes each rate-prompt, and stashes the
    consensus ``preferred_slot`` per row. ``run_dev_eval`` then runs the
    in-training policy on these prompts and computes pair_accuracy directly.
    """
    if is_privalign_dataset_name(dataset_name):
        raw_dataset = load_privalign_pairwise_genrm_dataset(
            dataset_name, split="validation",
        )
    else:
        raise ValueError(
            "Pairwise-margin dev eval requires a Privalign dataset_name; "
            f"got dataset_name={dataset_name!r}."
        )
    if config.dev_eval_max_samples is not None:
        max_samples = min(len(raw_dataset), int(config.dev_eval_max_samples))
        if hasattr(raw_dataset, "select"):
            raw_dataset = raw_dataset.select(range(max_samples))
        else:
            raw_dataset = raw_dataset[:max_samples]

    examples: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    prompt_token_ids_batch: list[list[int]] = []
    skipped = 0
    for example in raw_dataset:
        prompt_messages = example.get("prompt") or example.get("context") or []
        if not prompt_messages:
            skipped += 1
            continue
        prompt_text = rollout_builder._format_student_prompt(prompt_messages)
        prompt_token_ids = rollout_builder._tokenize_prompt_text_sequences([prompt_text])[0]
        if (
            config.max_prompt_length is not None
            and len(prompt_token_ids) > config.max_prompt_length
        ):
            skipped += 1
            continue
        ranking_demo = example.get("ranking_demo") or example.get("judge_demo")
        if not isinstance(ranking_demo, dict) or "preferred_slot" not in ranking_demo:
            skipped += 1
            continue
        examples.append(dict(example))
        prompt_texts.append(prompt_text)
        prompt_token_ids_batch.append(prompt_token_ids)

    logger.info(
        "Prepared pairwise-margin dev eval set with %s prompts (skipped=%s).",
        len(examples),
        skipped,
    )
    if not examples:
        raise ValueError(
            "Pairwise-margin dev eval produced no usable rows from "
            f"{dataset_name!r}."
        )
    return DevEvalState(
        eval_kind="pairwise_margin",
        examples=examples,
        prompt_texts=prompt_texts,
        prompt_token_ids_batch=prompt_token_ids_batch,
        judge_template="",
        reference_model_path="",
        judge_model_path="",
        judge_rollout_builder=rollout_builder,
        rubric_items_batch=[],
        eval_prompts_for_judge=[],
    )


def _build_privalign_dev_eval_state(
    *,
    raw_dataset,
    dataset_name: str,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    reference_model_path: str,
    judge_model_path: str,
) -> DevEvalState:
    # Keep Privalign policy training and dev evaluation deliberately different:
    # policy rewards may be pairwise (better relative training signal), while
    # dev eval reports absolute leak/omit/clean rates with the pointwise judge.
    judge_template = load_prompt(PRIVALIGN_RL_POINTWISE_JUDGE_TEMPLATE_NAME)
    judge_rollout_builder = _resolve_rollout_builder_for_model(
        config=config,
        model_path=judge_model_path,
        fallback_rollout_builder=rollout_builder,
        enable_thinking=bool(getattr(config, "policy_judge_enable_thinking", True)),
    )

    examples: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    prompt_token_ids_batch: list[list[int]] = []
    for example in raw_dataset:
        prompt_messages = example.get("context") or example.get("prompt") or []
        if not prompt_messages:
            continue
        prompt_text = rollout_builder._format_student_prompt(prompt_messages)
        prompt_token_ids = rollout_builder._tokenize_prompt_text_sequences([prompt_text])[0]
        if config.max_prompt_length is not None and len(prompt_token_ids) > config.max_prompt_length:
            continue
        examples.append(example)
        prompt_texts.append(prompt_text)
        prompt_token_ids_batch.append(prompt_token_ids)

    logger.info(
        "Prepared Privalign validation eval set for %s with %s prompts (requested max_samples=%s).",
        dataset_name,
        len(examples),
        config.dev_eval_max_samples,
    )
    if not examples:
        raise ValueError(
            "Privalign dev eval produced no usable validation prompts. "
            "Increase --max_prompt_length or disable dev eval."
        )

    return DevEvalState(
        eval_kind="privalign_pointwise",
        examples=examples,
        prompt_texts=prompt_texts,
        prompt_token_ids_batch=prompt_token_ids_batch,
        judge_template=judge_template,
        reference_model_path=reference_model_path,
        judge_model_path=judge_model_path,
        judge_rollout_builder=judge_rollout_builder,
        rubric_items_batch=[],
        eval_prompts_for_judge=[],
    )


def should_run_dev_eval(config, global_step: int) -> bool:
    return config.dev_eval_steps > 0 and global_step > 0 and global_step % config.dev_eval_steps == 0


def _ensure_reference_responses(
    *,
    model_group: RayActorGroup,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
    num_generations: int | None = None,
) -> tuple[list[str], list[str]]:
    num_generations = (
        int(getattr(config, "dev_eval_num_generations", 1)) or 1
        if num_generations is None
        else int(num_generations)
    )
    expected_count = len(eval_state.prompt_token_ids_batch) * num_generations
    cached = eval_state.raw_reference_responses
    if cached is None or len(cached) != expected_count:
        eval_state.raw_reference_responses = _load_cached_reference_responses(
            config=config,
            eval_state=eval_state,
            expected_num_generations=num_generations,
        )
        if eval_state.raw_reference_responses is None:
            logger.info(
                "Generating cached preference reference responses (num_generations=%s).",
                num_generations,
            )
            reference_temperature = _resolve_eval_candidate_temperature(config)
            with temporary_eval_vllm_engines(
                model_group=model_group,
                config=config,
                pool=pool,
                model_path=eval_state.reference_model_path,
                seed=build_eval_engine_seed(config.seed, purpose="reference"),
                max_model_len=_temporary_engine_max_model_len(
                    eval_state.prompt_token_ids_batch,
                    max_new_tokens=config.max_completion_length,
                ),
            ) as reference_engines:
                eval_state.raw_reference_responses = generate_texts_with_vllm(
                    args=config,
                    rollout_builder=rollout_builder,
                    vllm_engines=reference_engines,
                    prompt_token_ids_batch=eval_state.prompt_token_ids_batch,
                    max_new_tokens=config.max_completion_length,
                    strip_thinking_traces=False,
                    num_generations=num_generations,
                    temperature=reference_temperature,
                    top_p=_resolve_eval_candidate_top_p(config),
                    top_k=_resolve_eval_candidate_top_k(config),
                    repetition_penalty=_resolve_eval_candidate_repetition_penalty(config),
                    presence_penalty=_resolve_eval_candidate_presence_penalty(config),
                )
            _write_cached_reference_responses(
                config=config,
                eval_state=eval_state,
                raw_reference_responses=eval_state.raw_reference_responses,
                expected_num_generations=num_generations,
            )
        eval_state.reference_responses = None
    if eval_state.reference_responses is None:
        eval_state.reference_responses = [
            strip_thinking_trace(text) for text in (eval_state.raw_reference_responses or [])
        ]
    return eval_state.reference_responses, eval_state.raw_reference_responses or []


def _reference_response_cache_path(config) -> Path | None:
    path_value = getattr(config, "dev_eval_reference_response_cache", None)
    if path_value is None or str(path_value).strip() == "":
        return None
    return Path(str(path_value)).expanduser()


def _reference_response_cache_readonly(config) -> bool:
    return bool(getattr(config, "dev_eval_reference_response_cache_readonly", False))


def _reference_response_cache_key(example: dict[str, Any], *, example_index: int) -> str:
    judge_demo = example.get("judge_demo")
    if isinstance(judge_demo, dict) and judge_demo.get("demo_type") == "privalign_pairwise":
        item_id = judge_demo.get("item_id")
        if item_id is not None:
            return f"privalign:{item_id}"
        user_instruction = judge_demo.get("user_instruction")
        if isinstance(user_instruction, str) and user_instruction.strip():
            return f"privalign:{stable_text_hash_key(user_instruction)}"
    prompt_id = example.get("prompt_id")
    if prompt_id is not None:
        return f"prompt_id:{prompt_id}"
    return f"index:{example_index}"


def _expected_reference_response_cache_keys(
    eval_state: DevEvalState,
    *,
    expected_num_generations: int,
) -> list[tuple[str, int]]:
    return [
        (
            _reference_response_cache_key(example, example_index=example_index),
            generation_index,
        )
        for example_index, example in enumerate(eval_state.examples)
        for generation_index in range(expected_num_generations)
    ]


def _load_cached_reference_responses(
    *,
    config,
    eval_state: DevEvalState,
    expected_num_generations: int,
) -> list[str] | None:
    cache_path = _reference_response_cache_path(config)
    if cache_path is None:
        return None
    if not cache_path.is_file():
        if _reference_response_cache_readonly(config):
            raise FileNotFoundError(
                "Reference response cache is required but does not exist: "
                f"{cache_path.resolve()}"
            )
        logger.info("Reference response cache does not exist yet: %s", cache_path)
        return None

    rows_by_key: dict[tuple[str, int], str] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in reference response cache {cache_path} line {line_number}: {exc}"
                ) from exc
            cache_key = row.get("cache_key")
            generation_index = row.get("generation_index")
            raw_response = row.get("raw_reference_response")
            if not isinstance(cache_key, str):
                raise ValueError(f"Reference response cache row {line_number} is missing string cache_key.")
            if not isinstance(generation_index, int):
                raise ValueError(
                    f"Reference response cache row {line_number} is missing integer generation_index."
                )
            if not isinstance(raw_response, str):
                raise ValueError(
                    f"Reference response cache row {line_number} is missing string raw_reference_response."
                )
            key = (cache_key, generation_index)
            if key in rows_by_key:
                raise ValueError(f"Duplicate reference response cache key in {cache_path}: {key!r}")
            rows_by_key[key] = raw_response

    expected_keys = _expected_reference_response_cache_keys(
        eval_state,
        expected_num_generations=expected_num_generations,
    )
    missing_keys = [key for key in expected_keys if key not in rows_by_key]
    if missing_keys:
        if _reference_response_cache_readonly(config):
            preview = ", ".join(repr(key) for key in missing_keys[:5])
            raise ValueError(
                "Reference response cache is missing expected rows in read-only mode: "
                f"{cache_path.resolve()} missing {len(missing_keys)}/{len(expected_keys)} "
                f"rows. First missing keys: {preview}"
            )
        logger.warning(
            "Reference response cache %s is missing %s/%s expected rows; regenerating and overwriting it.",
            cache_path,
            len(missing_keys),
            len(expected_keys),
        )
        return None

    logger.info(
        "Loaded %s frozen reference dev-eval responses from %s.",
        len(expected_keys),
        cache_path,
    )
    return [rows_by_key[key] for key in expected_keys]


def _write_cached_reference_responses(
    *,
    config,
    eval_state: DevEvalState,
    raw_reference_responses: list[str],
    expected_num_generations: int,
) -> None:
    cache_path = _reference_response_cache_path(config)
    if cache_path is None:
        return
    expected_keys = _expected_reference_response_cache_keys(
        eval_state,
        expected_num_generations=expected_num_generations,
    )
    if len(raw_reference_responses) != len(expected_keys):
        raise ValueError(
            "Cannot write reference response cache because response count does not match eval examples "
            f"({len(raw_reference_responses)} != {len(expected_keys)})."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row_index, ((cache_key, generation_index), raw_response) in enumerate(
            zip(expected_keys, raw_reference_responses)
        ):
            row = {
                "cache_key": cache_key,
                "example_index": row_index // expected_num_generations,
                "generation_index": generation_index,
                "reference_model_path": eval_state.reference_model_path,
                "max_completion_length": int(config.max_completion_length),
                "eval_reference_vllm_temperature": _resolve_eval_candidate_temperature(config),
                "eval_reference_vllm_top_p": _resolve_eval_candidate_top_p(config),
                "eval_reference_vllm_top_k": _resolve_eval_candidate_top_k(config),
                "raw_reference_response": raw_response,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(cache_path)
    logger.info(
        "Wrote %s frozen reference dev-eval responses to %s.",
        len(raw_reference_responses),
        cache_path,
    )


def _tokenizer_name_or_path(tokenizer: Any) -> str | None:
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        return name
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        resolved_name = init_kwargs.get("name_or_path")
        if isinstance(resolved_name, str) and resolved_name:
            return resolved_name
    return None


def _copy_config_with_overrides(config, **overrides):
    if is_dataclass(config):
        return replace(config, **overrides)
    values = dict(vars(config))
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolve_rollout_builder_for_model(
    *,
    config,
    model_path: str,
    fallback_rollout_builder: OfflineRolloutBatchBuilder,
    enable_thinking: bool | None = None,
) -> OfflineRolloutBatchBuilder:
    fallback_tokenizer = getattr(fallback_rollout_builder, "processing_class", None)
    fallback_args = getattr(fallback_rollout_builder, "args", None)
    current_thinking = None if fallback_args is None else bool(
        getattr(fallback_args, "qwen_enable_thinking", True)
        and not getattr(fallback_args, "disable_student_thinking", False)
    )
    if enable_thinking is None:
        builder_config = config
    elif current_thinking is None or current_thinking == enable_thinking:
        builder_config = config
    else:
        builder_config = _copy_config_with_overrides(
            config,
            qwen_enable_thinking=enable_thinking,
            disable_student_thinking=not enable_thinking,
        )
    if _tokenizer_name_or_path(fallback_tokenizer) == model_path:
        if builder_config is config:
            return fallback_rollout_builder
        return OfflineRolloutBatchBuilder(builder_config, fallback_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if enable_thinking is not None:
        builder_config = _copy_config_with_overrides(
            config,
            qwen_enable_thinking=enable_thinking,
            disable_student_thinking=not enable_thinking,
        )
    return OfflineRolloutBatchBuilder(builder_config, tokenizer)


def _build_eval_sample_texts(
    *,
    prompt_text: str,
    student_response: str | None,
    reference_response: str | None,
    judge_prompt: str | None = None,
    gold_solution: str | None = None,
    gold_answer: str | None = None,
    judge_response: str | None = None,
    judge_correct: str | None = None,
) -> dict[str, str]:
    return {
        "prompt": prompt_text,
        "student_response": student_response or "",
        "reference_response": reference_response or "",
        "judge_prompt": judge_prompt or "",
        "gold_solution": gold_solution or "",
        "gold_answer": gold_answer or "",
        "judge_response": judge_response or "",
        "judge_correct": judge_correct or "",
    }


def _log_eval_sample(
    *,
    sample_texts: dict[str, str],
) -> None:
    rendered_lines = ["preference eval sample:"]
    field_labels = (
        ("prompt", "Prompt"),
        ("student_response", "Student response"),
        ("reference_response", "Reference response"),
        ("judge_prompt", "Judge prompt"),
        ("gold_solution", "Gold solution"),
        ("gold_answer", "Gold answer"),
        ("judge_response", "Judge response"),
        ("judge_correct", "Judge correct"),
    )
    for field_name, label in field_labels:
        value = sample_texts.get(field_name, "")
        if not value:
            continue
        rendered_lines.extend([f"{label}:", value])
    logger.info("\n%s", "\n".join(rendered_lines))


def _build_eval_sample_batch(
    *,
    rows: list[dict[str, str]],
    max_samples: int = STARTUP_EVAL_WANDB_SAMPLE_COUNT,
) -> list[dict[str, str]]:
    if max_samples <= 0:
        return []
    return [dict(row) for row in rows[:max_samples]]


def _decode_prompt_token_ids_batch(
    rollout_builder: OfflineRolloutBatchBuilder,
    prompt_token_ids_batch: list[list[int]],
) -> list[str]:
    return [
        rollout_builder.processing_class.decode(prompt_token_ids, skip_special_tokens=False)
        for prompt_token_ids in prompt_token_ids_batch
    ]


def _expand_for_multi_sample(items: list, num_generations: int) -> list:
    """Repeat each item ``num_generations`` times consecutively.

    Matches vLLM's interleaved output order where all generations for prompt *i*
    precede those for prompt *i+1*.
    """
    if num_generations <= 1:
        return items
    return [item for item in items for _ in range(num_generations)]


def _resolve_dev_eval_generation_count(config, *, compute_eval_loss: bool) -> int:
    requested = int(getattr(config, "dev_eval_num_generations", 1)) or 1
    if not compute_eval_loss or requested > 1:
        return requested
    fallback = int(getattr(config, "num_generations", requested)) or requested
    if fallback > requested:
        logger.warning(
            "--dev-eval-loss needs at least two candidate generations per prompt for the "
            "policy objective baseline; using num_generations=%s for this dev eval.",
            fallback,
        )
        return fallback
    return requested


def _generate_student_responses(
    *,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
    num_generations: int = 1,
    include_loss_diagnostics: bool = False,
) -> tuple[list[str], list[list[int]] | None, dict[str, Any] | None, list[bool]]:
    """Generate student responses and optionally return raw sequences + vLLM diagnostics.

    Returns (decoded_texts, raw_completion_sequences, vllm_diagnostics, unusable_mask).
    When include_loss_diagnostics is False, the raw sequences and diagnostics are None.
    """
    eval_temperature = _resolve_eval_candidate_temperature(config)
    sampling_kwargs = _build_eval_vllm_sampling_kwargs(
        max_new_tokens=config.max_completion_length,
        temperature=eval_temperature,
        top_p=_resolve_eval_candidate_top_p(config),
        top_k=_resolve_eval_candidate_top_k(config),
        num_generations=num_generations,
    )
    include_logprobs = include_loss_diagnostics
    logprob_count = requested_vllm_logprob_count(config) if include_loss_diagnostics else None
    with pool.lock:
        if config.vllm_enable_sleep_mode:
            batch_vllm_engine_call(pool.engines, "wake_up")
        try:
            completion_sequences, vllm_diagnostics = dispatch_prompts(
                args=config,
                vllm_engines=pool.engines,
                vllm_sampling_kwargs=sampling_kwargs,
                prompt_token_ids_batch=eval_state.prompt_token_ids_batch,
                include_logprobs=include_logprobs,
                requested_logprob_count=logprob_count,
                expected_num_generations=num_generations,
            )
        finally:
            if config.vllm_enable_sleep_mode:
                batch_vllm_engine_call(pool.engines, "sleep")
    _release_cuda_cache_after_eval()
    decoded_texts = _decode_completion_sequences(rollout_builder, completion_sequences)
    unusable_mask = [
        _completion_is_unusable_for_eval(rollout_builder, sequence)
        for sequence in completion_sequences
    ]
    if include_loss_diagnostics:
        return decoded_texts, completion_sequences, vllm_diagnostics, unusable_mask
    return decoded_texts, None, None, unusable_mask


def _summarize_dev_eval_judge_outputs(
    *,
    config,
    eval_state: DevEvalState,
    judge_engines: list,
    judge_prompt_token_ids_batch: list[list[int]],
    judge_prompt_token_ids_batch_swapped: list[list[int]],
    metric_prefix: str = "preference",
    candidate_label: str = "student",
    shorter_label: str | None = None,
    candidate_responses: list[str] | None = None,
    reference_responses: list[str] | None = None,
    auto_fail_mask: list[bool] | None = None,
) -> dict[str, float]:
    if len(judge_prompt_token_ids_batch) != len(judge_prompt_token_ids_batch_swapped):
        raise ValueError("Both preference judge prompt orderings must have the same number of prompts.")
    total_examples = len(judge_prompt_token_ids_batch)
    if auto_fail_mask is None:
        auto_fail_mask = [False] * total_examples
    if len(auto_fail_mask) != total_examples:
        raise ValueError("auto_fail_mask must align with judge prompt batches.")
    judge_indices = [index for index, failed in enumerate(auto_fail_mask) if not failed]
    selected_prompts = [judge_prompt_token_ids_batch[index] for index in judge_indices]
    selected_swapped_prompts = [judge_prompt_token_ids_batch_swapped[index] for index in judge_indices]
    judge_max_new_tokens = int(
        getattr(config, "dev_eval_judge_max_new_tokens", None)
        or config.max_completion_length
    )
    selected_outputs = generate_texts_with_vllm(
        args=config,
        rollout_builder=eval_state.judge_rollout_builder,
        vllm_engines=judge_engines,
        prompt_token_ids_batch=selected_prompts + selected_swapped_prompts,
        max_new_tokens=judge_max_new_tokens,
        top_p=config.judge_vllm_top_p,
        top_k=config.judge_vllm_top_k,
        presence_penalty=config.judge_vllm_presence_penalty,
    ) if judge_indices else []
    if len(selected_outputs) != 2 * len(judge_indices):
        raise ValueError(
            "Judge output count must match both pairwise orderings "
            f"({len(selected_outputs)} != {2 * len(judge_indices)})."
        )
    selected_candidate_first = selected_outputs[: len(judge_indices)]
    selected_reference_first = selected_outputs[len(judge_indices) :]
    judge_outputs = [PAIRWISE_NO_RESPONSE_CANDIDATE_FIRST_OUTPUT] * total_examples
    judge_outputs_swapped = [PAIRWISE_NO_RESPONSE_REFERENCE_FIRST_OUTPUT] * total_examples
    for offset, index in enumerate(judge_indices):
        judge_outputs[index] = selected_candidate_first[offset]
        judge_outputs_swapped[index] = selected_reference_first[offset]
    return summarize_pairwise_dev_eval_metrics(
        judge_outputs,
        judge_outputs_swapped,
        metric_prefix=metric_prefix,
        candidate_label=candidate_label,
        shorter_label=shorter_label,
        candidate_responses=candidate_responses,
        reference_responses=reference_responses,
    )


def _compute_dev_eval_loss(
    *,
    model_group: RayActorGroup,
    config,
    training_objective,
    policy_batch_builder,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
    completion_sequences: list[list[int]],
    completion_vllm_diagnostics: dict[str, torch.Tensor] | None,
    num_eval_generations: int,
    reference_responses: list[str] | None = None,
    usable_completion_mask: list[bool] | None = None,
) -> dict[str, float]:
    """Compute policy-objective eval loss via forward-only pass (no gradient)."""
    if policy_batch_builder is None:
        logger.warning("Skipping eval loss because no policy_batch_builder is available.")
        return {
            "eval_loss/no_response_skipped_count": 0.0,
            "eval_loss/usable_completion_count": 0.0,
        }
    num_eval_generations = int(num_eval_generations)
    if num_eval_generations <= 1:
        logger.warning(
            "Skipping eval loss because policy objectives require at least two eval generations "
            "per prompt to compute a prompt-group baseline; got dev_eval_num_generations=%s.",
            num_eval_generations,
        )
        return {
            "eval_loss/no_response_skipped_count": 0.0,
            "eval_loss/usable_completion_count": 0.0,
        }

    base_examples = [
        {**ex, "prompt": ex["context"]} if "prompt" not in ex else ex
        for ex in eval_state.examples
    ]
    if reference_responses is not None:
        base_examples = _attach_policy_reference_responses_for_eval_loss(
            base_examples,
            reference_responses=reference_responses,
            num_generations=num_eval_generations,
        )

    max_aligned_prompts = min(
        len(base_examples),
        len(eval_state.prompt_token_ids_batch),
        len(eval_state.prompt_texts),
        len(completion_sequences) // num_eval_generations,
    )
    if usable_completion_mask is None:
        usable_completion_mask = [True] * len(completion_sequences)
    kept_prompt_indices = [
        prompt_index
        for prompt_index in range(max_aligned_prompts)
        if any(
            sequence_index < len(usable_completion_mask)
            and bool(usable_completion_mask[sequence_index])
            for sequence_index in range(
                prompt_index * num_eval_generations,
                (prompt_index + 1) * num_eval_generations,
            )
        )
    ]
    total_aligned_sequences = max_aligned_prompts * num_eval_generations
    no_response_count = sum(
        1
        for index in range(total_aligned_sequences)
        if index < len(usable_completion_mask) and not bool(usable_completion_mask[index])
    )
    skipped_count = total_aligned_sequences - len(kept_prompt_indices) * num_eval_generations
    if skipped_count > 0:
        logger.info(
            "Skipping %s eval completions from prompt groups with no usable responses when computing eval loss.",
            skipped_count,
        )

    world_size = int(config.world_size)
    usable_prompt_count = (len(kept_prompt_indices) // world_size) * world_size
    if usable_prompt_count <= 0:
        logger.warning(
            "Not enough usable eval prompt groups (%s) for %s workers; skipping eval loss.",
            len(kept_prompt_indices),
            world_size,
        )
        return {
            "eval_loss/no_response_skipped_count": float(skipped_count),
            "eval_loss/no_response_count": float(no_response_count),
            "eval_loss/usable_completion_count": 0.0,
        }

    kept_prompt_indices = kept_prompt_indices[:usable_prompt_count]
    kept_sequence_indices = [
        sequence_index
        for prompt_index in kept_prompt_indices
        for sequence_index in range(
            prompt_index * num_eval_generations,
            (prompt_index + 1) * num_eval_generations,
        )
    ]
    examples = [base_examples[index] for index in kept_prompt_indices]
    prompt_token_ids_batch = [
        eval_state.prompt_token_ids_batch[index] for index in kept_prompt_indices
    ]
    prompt_texts = [eval_state.prompt_texts[index] for index in kept_prompt_indices]
    completion_sequences = [completion_sequences[index] for index in kept_sequence_indices]
    completion_vllm_diagnostics = _filter_completion_diagnostics_by_sequence_indices(
        completion_vllm_diagnostics,
        kept_sequence_indices,
    )

    prompt_tensors = rollout_builder._build_prompt_tensors_from_token_sequences(
        prompt_token_ids_batch,
    )
    package = rollout_builder.prepare_trajectory_batch(
        examples,
        completion_sequences=completion_sequences,
        completion_vllm_diagnostics=completion_vllm_diagnostics,
        update_token_count=False,
        prompt_texts=prompt_texts,
        prompt_tensors=prompt_tensors,
    )
    policy_batch_result = policy_batch_builder.build_batch(
        package.trajectory_batch,
        score_context=TrajectoryScoreContext(
            examples=examples,
            prompt_texts=prompt_texts,
        ),
    )

    prompts_per_worker = max(1, usable_prompt_count // config.world_size)
    worker_batches = shard_training_batch_for_workers(
        policy_batch_result.batch,
        layout=training_objective.training_batch_layout(),
        world_size=config.world_size,
        prompts_per_worker=prompts_per_worker,
        num_generations=num_eval_generations,
    )

    # Loss computation runs the trainer forward on the same GPUs that vLLM may
    # colocate with. Put rollout engines back to sleep before launching actor
    # calls; the next training rollout's wake_up handles bringing them online.
    if getattr(config, "vllm_enable_sleep_mode", False) and pool.engines:
        with pool.lock:
            batch_vllm_engine_call(pool.engines, "sleep")
    _release_cuda_cache_after_eval()

    eval_refs = [
        actor.eval_step.remote(worker_batch, num_eval_generations)
        for actor, worker_batch in zip(model_group.actor_handlers, worker_batches)
    ]
    eval_results = ray.get(eval_refs)

    rank_zero_result = next(result for result in eval_results if result["rank"] == 0)
    metrics = {f"eval_loss/{key}": value for key, value in rank_zero_result["metrics"].items()}
    metrics["eval_loss/no_response_skipped_count"] = float(skipped_count)
    metrics["eval_loss/no_response_count"] = float(no_response_count)
    metrics["eval_loss/usable_completion_count"] = float(usable_prompt_count * num_eval_generations)
    return metrics


def _attach_policy_reference_responses_for_eval_loss(
    examples: list[dict[str, Any]],
    *,
    reference_responses: list[str],
    num_generations: int,
) -> list[dict[str, Any]]:
    """Attach one anchor response per prompt for anchor-style policy scorers."""
    if num_generations <= 0:
        raise ValueError("num_generations must be > 0.")
    updated_examples: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        reference_index = index * num_generations
        if reference_index < len(reference_responses):
            updated_examples.append(
                {
                    **example,
                    "policy_reference_response": reference_responses[reference_index],
                }
            )
        else:
            updated_examples.append(dict(example))
    return updated_examples


def _run_privalign_pointwise_dev_eval(
    *,
    model_group: RayActorGroup,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
    training_objective=None,
    policy_batch_builder=None,
) -> dict[str, float]:
    """Dev-eval that scores each rollout with the leak/omit judge prompt."""
    eval_state.logged_eval_sample_texts = None
    eval_state.logged_eval_sample_texts_batch = None
    compute_eval_loss = training_objective is not None and bool(
        getattr(config, "dev_eval_loss", False)
    )
    num_eval_gens = _resolve_dev_eval_generation_count(
        config,
        compute_eval_loss=compute_eval_loss,
    )
    student_responses, completion_sequences, vllm_diagnostics, unusable_completion_mask = _generate_student_responses(
        config=config,
        rollout_builder=rollout_builder,
        eval_state=eval_state,
        pool=pool,
        num_generations=num_eval_gens,
        include_loss_diagnostics=compute_eval_loss,
    )
    expanded_examples = _expand_for_multi_sample(eval_state.examples, num_eval_gens)
    judge_prompt_token_ids_batch = build_privalign_rl_pointwise_judge_prompt_token_ids_batch(
        examples=expanded_examples,
        judge_template=eval_state.judge_template,
        rollout_builder=eval_state.judge_rollout_builder,
        response_batch=student_responses,
        max_response_words=getattr(config, "privalign_judge_max_response_words", 1000),
    )
    judge_prompt_texts = _decode_prompt_token_ids_batch(
        eval_state.judge_rollout_builder,
        judge_prompt_token_ids_batch,
    )
    if eval_state.prompt_texts:
        sample_index = eval_state.logged_sample_index if eval_state.logged_sample_index is not None else 0
        sample_expanded_index = sample_index * num_eval_gens
        sample_texts = _build_eval_sample_texts(
            prompt_text=eval_state.prompt_texts[sample_index],
            student_response=student_responses[sample_expanded_index],
            reference_response=None,
            judge_prompt=judge_prompt_texts[sample_expanded_index],
        )
        _log_eval_sample(sample_texts=sample_texts)
        eval_state.logged_eval_sample_texts = sample_texts
    eval_state.logged_eval_sample_texts_batch = _build_eval_sample_batch(
        rows=[
            _build_eval_sample_texts(
                prompt_text=eval_state.prompt_texts[index],
                student_response=student_responses[index * num_eval_gens],
                reference_response=None,
                judge_prompt=judge_prompt_texts[index * num_eval_gens],
            )
            for index in range(len(eval_state.prompt_texts))
        ]
    )

    judge_max_new_tokens = int(
        getattr(config, "dev_eval_judge_max_new_tokens", None)
        or config.max_completion_length
    )
    total_examples = len(judge_prompt_token_ids_batch)
    judge_indices = [
        index for index, failed in enumerate(unusable_completion_mask) if not failed
    ]
    selected_prompts = [judge_prompt_token_ids_batch[index] for index in judge_indices]
    judge_outputs: list[str] = [""] * total_examples
    if selected_prompts:
        with temporary_eval_vllm_engines(
            model_group=model_group,
            config=config,
            pool=pool,
            model_path=eval_state.judge_model_path,
            seed=build_eval_engine_seed(config.seed, purpose="judge"),
            max_model_len=_temporary_engine_max_model_len(
                selected_prompts,
                max_new_tokens=judge_max_new_tokens,
            ),
        ) as judge_engines:
            selected_outputs = generate_texts_with_vllm(
                args=config,
                rollout_builder=eval_state.judge_rollout_builder,
                vllm_engines=judge_engines,
                prompt_token_ids_batch=selected_prompts,
                max_new_tokens=judge_max_new_tokens,
                top_p=config.judge_vllm_top_p,
                top_k=config.judge_vllm_top_k,
                presence_penalty=config.judge_vllm_presence_penalty,
            )
        if len(selected_outputs) != len(selected_prompts):
            raise ValueError(
                "Judge output count must match selected pointwise prompt count "
                f"({len(selected_outputs)} != {len(selected_prompts)})."
            )
        for offset, index in enumerate(judge_indices):
            judge_outputs[index] = selected_outputs[offset]

    all_metrics = summarize_pointwise_dev_eval_metrics(
        judge_outputs,
        metric_prefix="privalign",
        auto_fail_mask=unusable_completion_mask,
    )
    all_metrics.update(
        _summarize_no_response_eval_metrics(
            unusable_completion_mask,
            metric_prefix="privalign",
        )
    )

    if compute_eval_loss and completion_sequences is not None:
        all_metrics.update(
            _compute_dev_eval_loss(
                model_group=model_group,
                config=config,
                training_objective=training_objective,
                policy_batch_builder=policy_batch_builder,
                rollout_builder=rollout_builder,
                eval_state=eval_state,
                pool=pool,
                completion_sequences=completion_sequences,
                completion_vllm_diagnostics=vllm_diagnostics,
                num_eval_generations=num_eval_gens,
                reference_responses=None,
                usable_completion_mask=[not value for value in unusable_completion_mask],
            )
        )
    return all_metrics


def _run_privalign_dev_eval(
    *,
    model_group: RayActorGroup,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
    training_objective=None,
    policy_batch_builder=None,
) -> dict[str, float]:
    eval_state.logged_eval_sample_texts = None
    eval_state.logged_eval_sample_texts_batch = None
    compute_eval_loss = training_objective is not None and bool(
        getattr(config, "dev_eval_loss", False)
    )
    num_eval_gens = _resolve_dev_eval_generation_count(
        config,
        compute_eval_loss=compute_eval_loss,
    )
    student_responses, completion_sequences, vllm_diagnostics, unusable_completion_mask = _generate_student_responses(
        config=config,
        rollout_builder=rollout_builder,
        eval_state=eval_state,
        pool=pool,
        num_generations=num_eval_gens,
        include_loss_diagnostics=compute_eval_loss,
    )
    reference_responses, _raw_reference_responses = _ensure_reference_responses(
        model_group=model_group,
        config=config,
        rollout_builder=rollout_builder,
        eval_state=eval_state,
        pool=pool,
        num_generations=num_eval_gens,
    )
    expanded_examples = _expand_for_multi_sample(eval_state.examples, num_eval_gens)
    judge_prompt_token_ids_batch = build_privalign_rl_judge_prompt_token_ids_batch(
        examples=expanded_examples,
        judge_template=eval_state.judge_template,
        rollout_builder=eval_state.judge_rollout_builder,
        response1_batch=student_responses,
        response2_batch=reference_responses,
        max_response_words=getattr(config, "privalign_judge_max_response_words", 1000),
    )
    judge_prompt_token_ids_batch_swapped = build_privalign_rl_judge_prompt_token_ids_batch(
        examples=expanded_examples,
        judge_template=eval_state.judge_template,
        rollout_builder=eval_state.judge_rollout_builder,
        response1_batch=reference_responses,
        response2_batch=student_responses,
        max_response_words=getattr(config, "privalign_judge_max_response_words", 1000),
    )
    judge_prompt_texts = _decode_prompt_token_ids_batch(
        eval_state.judge_rollout_builder,
        judge_prompt_token_ids_batch,
    )
    if eval_state.prompt_texts:
        sample_index = eval_state.logged_sample_index if eval_state.logged_sample_index is not None else 0
        sample_expanded_index = sample_index * num_eval_gens
        sample_texts = _build_eval_sample_texts(
            prompt_text=eval_state.prompt_texts[sample_index],
            student_response=student_responses[sample_expanded_index],
            reference_response=reference_responses[sample_expanded_index],
            judge_prompt=judge_prompt_texts[sample_expanded_index],
        )
        _log_eval_sample(sample_texts=sample_texts)
        eval_state.logged_eval_sample_texts = sample_texts
    eval_state.logged_eval_sample_texts_batch = _build_eval_sample_batch(
        rows=[
            _build_eval_sample_texts(
                prompt_text=eval_state.prompt_texts[index],
                student_response=student_responses[index * num_eval_gens],
                reference_response=reference_responses[index * num_eval_gens],
                judge_prompt=judge_prompt_texts[index * num_eval_gens],
            )
            for index in range(len(eval_state.prompt_texts))
        ]
    )

    judge_max_new_tokens = int(
        getattr(config, "dev_eval_judge_max_new_tokens", None)
        or config.max_completion_length
    )
    with temporary_eval_vllm_engines(
        model_group=model_group,
        config=config,
        pool=pool,
        model_path=eval_state.judge_model_path,
        seed=build_eval_engine_seed(config.seed, purpose="judge"),
        max_model_len=_temporary_engine_max_model_len(
            judge_prompt_token_ids_batch + judge_prompt_token_ids_batch_swapped,
            max_new_tokens=judge_max_new_tokens,
        ),
    ) as judge_engines:
        all_metrics = _summarize_dev_eval_judge_outputs(
            config=config,
            eval_state=eval_state,
            judge_engines=judge_engines,
            judge_prompt_token_ids_batch=judge_prompt_token_ids_batch,
            judge_prompt_token_ids_batch_swapped=judge_prompt_token_ids_batch_swapped,
            metric_prefix="privalign",
            candidate_label="student",
            shorter_label="student_response",
            candidate_responses=student_responses,
            reference_responses=reference_responses,
            auto_fail_mask=unusable_completion_mask,
        )
    all_metrics.update(_summarize_no_response_eval_metrics(unusable_completion_mask, metric_prefix="privalign"))
    all_metrics.update(summarize_reference_word_count_metrics(reference_responses, metric_prefix="privalign"))

    if compute_eval_loss and completion_sequences is not None:
        all_metrics.update(
            _compute_dev_eval_loss(
                model_group=model_group,
                config=config,
                training_objective=training_objective,
                policy_batch_builder=policy_batch_builder,
                rollout_builder=rollout_builder,
                eval_state=eval_state,
                pool=pool,
                completion_sequences=completion_sequences,
                completion_vllm_diagnostics=vllm_diagnostics,
                num_eval_generations=num_eval_gens,
                reference_responses=reference_responses,
                usable_completion_mask=[not value for value in unusable_completion_mask],
            )
        )
    return all_metrics


def run_dev_eval(
    *,
    model_group: RayActorGroup,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
    training_objective=None,
    policy_batch_builder=None,
) -> dict[str, float]:
    if getattr(eval_state, "eval_kind", None) == "pairwise_margin":
        return _run_pairwise_margin_dev_eval(
            config=config,
            rollout_builder=rollout_builder,
            eval_state=eval_state,
            pool=pool,
        )
    if getattr(eval_state, "eval_kind", None) == "privalign":
        return _run_privalign_dev_eval(
            model_group=model_group,
            config=config,
            rollout_builder=rollout_builder,
            eval_state=eval_state,
            pool=pool,
            training_objective=training_objective,
            policy_batch_builder=policy_batch_builder,
        )
    if getattr(eval_state, "eval_kind", None) == "privalign_pointwise":
        return _run_privalign_pointwise_dev_eval(
            model_group=model_group,
            config=config,
            rollout_builder=rollout_builder,
            eval_state=eval_state,
            pool=pool,
            training_objective=training_objective,
            policy_batch_builder=policy_batch_builder,
        )
    raise ValueError(f"Unsupported dev eval kind: {getattr(eval_state, 'eval_kind', None)!r}")


# ---------------------------------------------------------------------------
# Policy batch builder (wires scorer → batch builder when using judge scorer)
# ---------------------------------------------------------------------------


def build_policy_batch_builder(
    *,
    config,
    training_objective,
    rollout_builder: OfflineRolloutBatchBuilder,
    model_group: RayActorGroup,
    pool: VLLMEnginePool,
):
    if training_objective.name != "policy_optimization":
        return None
    if config.policy_scorer in {PairwiseMarginScorer.name, PrivalignPairwiseMarginScorer.name}:
        scorer_cls = (
            PrivalignPairwiseMarginScorer
            if config.policy_scorer == PrivalignPairwiseMarginScorer.name
            else PairwiseMarginScorer
        )
        scorer = scorer_cls(
            rollout_builder=rollout_builder,
            c1=getattr(config, "policy_pairwise_margin_c1", 100.0),
            c2=getattr(config, "policy_pairwise_margin_c2", 1.0),
            w_leak=getattr(config, "policy_pairwise_w_leak", 1.0),
            penalty_max_len=getattr(config, "policy_penalty_max_len", None),
            penalty_per_word=getattr(config, "policy_penalty_per_word", 0.0),
            penalty_max_value=getattr(config, "policy_penalty_max_value", 2.0),
            penalty_shape=getattr(config, "policy_penalty_shape", "linear"),
        )
        return build_policy_training_batch_builder(config, scorer=scorer)
    if config.policy_scorer in TRAINED_GENRM_POLICY_SCORERS:
        scorer = _build_trained_genrm_scorer(
            config=config,
            rollout_builder=rollout_builder,
            model_group=model_group,
            pool=pool,
        )
        return build_policy_training_batch_builder(config, scorer=scorer)
    if config.policy_scorer != PrivalignRLPairwiseJudgeScorer.name:
        raise ValueError(f"Unsupported policy_scorer for policy_optimization: {config.policy_scorer!r}")
    scorer = _build_judge_scorer(
        config=config,
        rollout_builder=rollout_builder,
        model_group=model_group,
        pool=pool,
    )
    return build_policy_training_batch_builder(config, scorer=scorer)


def _build_trained_genrm_scorer(
    *,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    model_group: RayActorGroup,
    pool: VLLMEnginePool,
) -> TrainedGenRMScorer:
    """Construct the TrainedGenRMScorer for a Phase B trained-gen-RM cell.

    Pulls the trained gen-RM checkpoint path and prompt-template name from
    ``config``, builds a model-specific rollout builder, wires a closure that
    wakes a temporary eval engine on the rm_model_path and writes ``Score:
    <int -2..2>`` for each pairwise rendering, and returns the configured
    scorer instance. Used by both the pure-genrm and composite dispatch
    branches in ``build_policy_batch_builder``.
    """
    rm_model_path = getattr(config, "policy_trained_genrm_path", None)
    if not rm_model_path:
        raise ValueError(
            "policy_trained_genrm_path must be set when policy_scorer="
            f"{config.policy_scorer!r}."
        )
    annotation_conditioning = bool(
        getattr(config, "policy_trained_genrm_annotation_conditioning", False)
    )
    prompt_template_name = resolve_trained_genrm_prompt_template(
        policy_scorer=config.policy_scorer,
        override=getattr(config, "policy_trained_genrm_prompt_template", None),
        annotation_conditioning=annotation_conditioning,
    )
    rm_rollout_builder = _resolve_rollout_builder_for_model(
        config=config,
        model_path=rm_model_path,
        fallback_rollout_builder=rollout_builder,
        enable_thinking=bool(getattr(config, "policy_trained_genrm_enable_thinking", True)),
    )

    def rm_text_generator(
        prompt_token_ids_batch: list[list[int]],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        requested_max_model_len = _temporary_engine_max_model_len(
            prompt_token_ids_batch,
            max_new_tokens=max_new_tokens,
        )

        def generate_with_engines(rm_engines: list) -> list[str]:
            return generate_texts_with_vllm(
                args=config,
                rollout_builder=rm_rollout_builder,
                vllm_engines=rm_engines,
                prompt_token_ids_batch=prompt_token_ids_batch,
                max_new_tokens=max_new_tokens,
                top_p=(
                    config.policy_judge_vllm_top_p
                    if config.policy_judge_vllm_top_p is not None
                    else config.judge_vllm_top_p
                ),
                top_k=(
                    config.policy_judge_vllm_top_k
                    if config.policy_judge_vllm_top_k is not None
                    else config.judge_vllm_top_k
                ),
                presence_penalty=(
                    config.policy_judge_vllm_presence_penalty
                    if config.policy_judge_vllm_presence_penalty is not None
                    else config.judge_vllm_presence_penalty
                ),
                repetition_penalty=config.policy_judge_vllm_repetition_penalty,
                temperature=config.policy_judge_vllm_temperature,
            )

        with pool.lock:
            rm_engines = ensure_trained_genrm_vllm_engines(
                pool=pool,
                config=config,
                model_path=rm_model_path,
                seed=build_eval_engine_seed(config.seed, purpose="trained_genrm"),
                max_model_len=requested_max_model_len,
            )
            if config.vllm_enable_sleep_mode:
                batch_vllm_engine_call(rm_engines, "wake_up")
            try:
                return generate_with_engines(rm_engines)
            finally:
                if config.vllm_enable_sleep_mode:
                    batch_vllm_engine_call(rm_engines, "sleep")

    rm_max_new_tokens = getattr(config, "policy_judge_max_new_tokens", None)
    if rm_max_new_tokens is None:
        rm_max_new_tokens = getattr(config, "policy_trained_genrm_max_new_tokens", 1024)

    return TrainedGenRMScorer(
        rollout_builder=rollout_builder,
        rm_rollout_builder=rm_rollout_builder,
        rm_text_generator=rm_text_generator,
        rm_max_new_tokens=int(rm_max_new_tokens),
        rm_dual_order=bool(getattr(config, "policy_judge_dual_order", False)),
        penalty_max_len=getattr(config, "policy_penalty_max_len", None),
        penalty_per_word=getattr(config, "policy_penalty_per_word", 0.0),
        penalty_max_value=getattr(config, "policy_penalty_max_value", 2.0),
        penalty_shape=getattr(config, "policy_penalty_shape", "linear"),
        retain_rm_outputs=False,
        prompt_template_name=prompt_template_name,
        annotation_conditioning=annotation_conditioning,
        privalign_judge_max_response_words=getattr(
            config, "privalign_judge_max_response_words", 1000
        ),
        undershort_penalty_max=getattr(config, "policy_undershort_penalty_max", 0.0),
        undershort_floor_ratio=getattr(config, "policy_undershort_floor_ratio", 0.5),
    )


def _build_judge_scorer(
    *,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    model_group: RayActorGroup,
    pool: VLLMEnginePool,
    scorer_name_override: str | None = None,
):
    """Construct the Privalign LLM-judge scorer.

    ``scorer_name_override`` lets the composite branch pick the judge variant
    that matches its dataset (the composite scorer names don't appear in the
    if/elif above; we just hand the helper the underlying pure-judge name to
    resolve scorer_cls + template + scoring_mode).
    """
    target_name = scorer_name_override or config.policy_scorer
    if target_name == PrivalignRLPairwiseJudgeScorer.name:
        scorer_cls = PrivalignRLPairwiseJudgeScorer
        judge_template_name = PRIVALIGN_RL_JUDGE_TEMPLATE_NAME
    else:
        raise ValueError(f"Unsupported LLM-judge scorer: {target_name!r}")
    judge_template = load_prompt(judge_template_name)
    judge_rollout_builder = _resolve_rollout_builder_for_model(
        config=config,
        model_path=config.policy_judge_model_name,
        fallback_rollout_builder=rollout_builder,
        enable_thinking=bool(getattr(config, "policy_judge_enable_thinking", True)),
    )

    def judge_text_generator(
        prompt_token_ids_batch: list[list[int]],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        requested_max_model_len = _temporary_engine_max_model_len(
            prompt_token_ids_batch,
            max_new_tokens=max_new_tokens,
        )

        def generate_with_engines(judge_engines: list) -> list[str]:
            return generate_texts_with_vllm(
                args=config,
                rollout_builder=judge_rollout_builder,
                vllm_engines=judge_engines,
                prompt_token_ids_batch=prompt_token_ids_batch,
                max_new_tokens=max_new_tokens,
                top_p=(
                    config.policy_judge_vllm_top_p
                    if config.policy_judge_vllm_top_p is not None
                    else config.judge_vllm_top_p
                ),
                top_k=(
                    config.policy_judge_vllm_top_k
                    if config.policy_judge_vllm_top_k is not None
                    else config.judge_vllm_top_k
                ),
                presence_penalty=(
                    config.policy_judge_vllm_presence_penalty
                    if config.policy_judge_vllm_presence_penalty is not None
                    else config.judge_vllm_presence_penalty
                ),
                repetition_penalty=config.policy_judge_vllm_repetition_penalty,
                temperature=config.policy_judge_vllm_temperature,
            )

        with pool.lock:
            judge_engines = ensure_policy_judge_vllm_engines(
                pool=pool,
                config=config,
                model_path=config.policy_judge_model_name,
                seed=build_eval_engine_seed(config.seed, purpose="judge"),
                max_model_len=requested_max_model_len,
            )
            if config.vllm_enable_sleep_mode:
                batch_vllm_engine_call(judge_engines, "wake_up")
            try:
                return generate_with_engines(judge_engines)
            finally:
                if config.vllm_enable_sleep_mode:
                    batch_vllm_engine_call(judge_engines, "sleep")

    scoring_mode = "peer"
    scorer_kwargs = {
        "rollout_builder": rollout_builder,
        "judge_rollout_builder": judge_rollout_builder,
        "judge_template": judge_template,
        "judge_text_generator": judge_text_generator,
        "judge_max_new_tokens": getattr(config, "policy_judge_max_new_tokens", None),
        "penalty_max_len": getattr(config, "policy_penalty_max_len", None),
        "penalty_per_word": getattr(config, "policy_penalty_per_word", 0.0),
        "penalty_max_value": getattr(config, "policy_penalty_max_value", 2.0),
        "penalty_shape": getattr(config, "policy_penalty_shape", "linear"),
        "scoring_mode": scoring_mode,
        "judge_dual_order": getattr(config, "policy_judge_dual_order", False),
        "retain_judge_outputs": False,
        "undershort_penalty_max": getattr(config, "policy_undershort_penalty_max", 0.0),
        "undershort_floor_ratio": getattr(config, "policy_undershort_floor_ratio", 0.5),
    }
    if scorer_cls is PrivalignRLPairwiseJudgeScorer:
        scorer_kwargs["privalign_judge_max_response_words"] = getattr(
            config, "privalign_judge_max_response_words", 1000
        )
    return scorer_cls(**scorer_kwargs)


def _run_pairwise_margin_dev_eval(
    *,
    config,
    rollout_builder: OfflineRolloutBatchBuilder,
    eval_state: DevEvalState,
    pool: VLLMEnginePool,
) -> dict[str, float]:
    """Inline dev eval for the Phase A gen-RM scorer.

    Runs the in-training gen-RM (current policy weights, served by the
    training-time vLLM engines in ``pool``) on the pairwise dev split,
    parses each rollout's `Score: <int -2..2>`, and computes:

        dev/pair_accuracy       fraction of all rollouts with signed_target > 0
                                (format violations count as wrong)
        dev/signed_target_mean  mean of (target_sign * predicted_score)
        dev/format_violation_rate
        dev/predicted_score_abs_mean
    """
    from experience.scorers import _parse_pairwise_leaks, _parse_signed_score, _resolve_target_score

    eval_state.logged_eval_sample_texts = None
    eval_state.logged_eval_sample_texts_batch = None

    sampling_kwargs = {
        "n": 1,
        "temperature": _resolve_eval_candidate_temperature(config),
        "top_p": _resolve_eval_candidate_top_p(config),
        "max_tokens": int(getattr(config, "max_completion_length", 1024)),
        "detokenize": False,
    }
    eval_top_k = _resolve_eval_candidate_top_k(config)
    if eval_top_k is not None:
        sampling_kwargs["top_k"] = int(eval_top_k)

    with pool.lock:
        if config.vllm_enable_sleep_mode:
            batch_vllm_engine_call(pool.engines, "wake_up")
        completion_sequences, _ = dispatch_prompts(
            args=config,
            vllm_engines=pool.engines,
            vllm_sampling_kwargs=sampling_kwargs,
            prompt_token_ids_batch=eval_state.prompt_token_ids_batch,
            include_logprobs=False,
            requested_logprob_count=None,
            expected_num_generations=1,
        )
        if config.vllm_enable_sleep_mode:
            batch_vllm_engine_call(pool.engines, "sleep")

    _release_cuda_cache_after_eval()

    completion_texts = _decode_completion_sequences(
        rollout_builder, completion_sequences, strip_thinking_traces=True
    )

    valid_count = 0
    correct_count = 0
    invalid_count = 0
    tie_total = 0
    tie_correct = 0
    nontie_total = 0
    nontie_correct = 0
    exact_match_count = 0
    abs_error_sum = 0.0
    signed_targets: list[float] = []
    predicted_values: list[int] = []
    leak_labeled_count = 0
    leak_correct_count = 0
    leak_abs_error_sum = 0.0
    leak_invalid_count = 0
    examples = eval_state.examples
    n = min(len(completion_texts), len(examples))
    for completion_text, example in zip(completion_texts[:n], examples[:n]):
        ranking_demo = example.get("ranking_demo") or example.get("judge_demo") or {}
        preferred_slot = int(ranking_demo.get("preferred_slot", 0))
        if preferred_slot not in (1, 2):
            invalid_count += 1
            continue
        predicted = _parse_signed_score(completion_text, valid_range=(-2, 2))
        pred_leak1, pred_leak2 = _parse_pairwise_leaks(completion_text)
        gold_leak1 = ranking_demo.get("gold_leak_response1")
        gold_leak2 = ranking_demo.get("gold_leak_response2")
        has_leak_targets = (
            isinstance(gold_leak1, (int, float))
            and isinstance(gold_leak2, (int, float))
            and 0.0 <= float(gold_leak1) <= 1.0
            and 0.0 <= float(gold_leak2) <= 1.0
        )
        if has_leak_targets:
            for pred_leak, gold_leak in (
                (pred_leak1, float(gold_leak1)),
                (pred_leak2, float(gold_leak2)),
            ):
                leak_labeled_count += 1
                if pred_leak is None:
                    leak_abs_error_sum += 1.0
                    continue
                pred_leak_f = 1.0 if pred_leak else 0.0
                abs_error = abs(pred_leak_f - gold_leak)
                leak_abs_error_sum += abs_error
                if (pred_leak_f >= 0.5) == (gold_leak >= 0.5):
                    leak_correct_count += 1

        target_margin = _resolve_target_score(example)
        if target_margin is not None:
            if target_margin == 0:
                tie_total += 1
            else:
                nontie_total += 1

        leak_invalid = has_leak_targets and (pred_leak1 is None or pred_leak2 is None)
        if predicted is None or leak_invalid:
            invalid_count += 1
            if leak_invalid:
                leak_invalid_count += 1
            continue
        valid_count += 1
        predicted_values.append(predicted)
        target_sign = -1.0 if preferred_slot == 1 else 1.0
        signed = target_sign * float(predicted)
        signed_targets.append(signed)
        # Target-score metrics: rely on the averaged target when present so tie
        # rows (target=0) are scored as "correct iff predicted==0" instead of
        # contributing arbitrary sign noise to pair_accuracy.
        if target_margin is not None:
            abs_error_sum += abs(predicted - target_margin)
            if target_margin == 0:
                is_correct = predicted == 0
            else:
                is_correct = (predicted > 0) == (target_margin > 0)
            if is_correct:
                correct_count += 1
            if predicted == target_margin:
                exact_match_count += 1
            if target_margin == 0:
                if predicted == 0:
                    tie_correct += 1
            else:
                # pair_accuracy on non-tie rows: sign agreement only
                if (predicted > 0) == (target_margin > 0):
                    nontie_correct += 1
        elif signed > 0:
            correct_count += 1

    if n > 0:
        sample_texts = _build_eval_sample_texts(
            prompt_text=eval_state.prompt_texts[0],
            student_response=completion_texts[0],
            reference_response="",
        )
        _log_eval_sample(sample_texts=sample_texts)
        eval_state.logged_eval_sample_texts = sample_texts
        eval_state.logged_sample_index = 0

    total = max(1, valid_count + invalid_count)
    exact_total = tie_total + nontie_total
    metrics = {
        "dev/pair_accuracy": float(correct_count) / total,
        "dev/signed_target_mean": (
            float(sum(signed_targets) / len(signed_targets)) if signed_targets else 0.0
        ),
        "dev/format_violation_rate": float(invalid_count) / total,
        "dev/predicted_score_abs_mean": (
            float(sum(abs(p) for p in predicted_values) / len(predicted_values))
            if predicted_values else 0.0
        ),
        "dev/num_valid": float(valid_count),
        "dev/num_invalid": float(invalid_count),
    }
    if leak_labeled_count > 0 or leak_invalid_count > 0:
        metrics["dev/leak_accuracy"] = (
            float(leak_correct_count) / leak_labeled_count if leak_labeled_count else 0.0
        )
        metrics["dev/leak_abs_error_mean"] = (
            float(leak_abs_error_sum) / leak_labeled_count if leak_labeled_count else 0.0
        )
        metrics["dev/leak_format_violation_rate"] = float(leak_invalid_count) / max(1, n)
        metrics["dev/num_leak_labels"] = float(leak_labeled_count)
    if nontie_total > 0 or tie_total > 0:
        metrics["dev/pair_accuracy_nontie"] = (
            float(nontie_correct) / nontie_total if nontie_total else 0.0
        )
        metrics["dev/tie_accuracy"] = (
            float(tie_correct) / tie_total if tie_total else 0.0
        )
        metrics["dev/exact_match_rate"] = (
            float(exact_match_count) / exact_total if exact_total else 0.0
        )
        metrics["dev/abs_error_mean"] = (
            float(abs_error_sum) / valid_count if valid_count else 0.0
        )
        metrics["dev/num_tie"] = float(tie_total)
        metrics["dev/num_nontie"] = float(nontie_total)
    logger.info(
        "Pairwise-margin dev eval: pair_acc=%.3f (nontie=%.3f tie=%.3f exact=%.3f) leak_acc=%.3f signed_mean=%+.3f invalid_rate=%.3f n_valid=%d n_tie=%d",
        metrics["dev/pair_accuracy"],
        metrics.get("dev/pair_accuracy_nontie", 0.0),
        metrics.get("dev/tie_accuracy", 0.0),
        metrics.get("dev/exact_match_rate", 0.0),
        metrics.get("dev/leak_accuracy", 0.0),
        metrics["dev/signed_target_mean"],
        metrics["dev/format_violation_rate"],
        valid_count,
        tie_total,
    )
    return metrics
