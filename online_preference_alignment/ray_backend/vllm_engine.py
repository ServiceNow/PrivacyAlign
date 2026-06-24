from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

import ray
import torch


def _patch_transformers_for_vllm() -> None:
    """Re-add ``all_special_tokens_extended`` for transformers 5.8+ tokenizers.

    transformers 5.8.0 removed the property but vLLM 0.10.2's
    ``get_cached_tokenizer`` still reads it. We fall back to the plain
    ``all_special_tokens`` list, which is what vLLM caches and serves back.
    Runs at module import inside every Ray actor before ``vllm`` itself is
    imported.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:  # pragma: no cover - transformers must be present, but keep harmless
        return
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )


_patch_transformers_for_vllm()

import vllm
from packaging import version
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ray_backend.ray_utils import wait_for_ray_refs
from ray_backend.vllm_utils import get_bundle_indices
from utils.hub_kernel_utils import force_local_transformers_hub_kernels_for_model


logger = logging.getLogger(__name__)


@ray.remote
class OfflineLLMRayActor:
    """Ray-hosted wrapper around offline vLLM `LLM`."""

    def __init__(self, *, bundle_indices: list[int] | None = None, num_gpus: float = 1.0, **kwargs) -> None:
        # Re-apply the transformers-5.8 compat patch inside the Ray worker.
        # Ray serializes the actor class via cloudpickle and may not re-run
        # module-level code on the worker, so this call guarantees the patch
        # is in place before vllm.LLM(...) triggers get_cached_tokenizer.
        _patch_transformers_for_vllm()
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._log_phase_progress = bool(kwargs.pop("log_phase_progress", False))
        vllm_port = kwargs.pop("vllm_port", None)
        logger.info(
            "Starting offline vLLM actor for model=%s tp=%s backend=%s bundle_indices=%s vllm_port=%s.",
            kwargs.get("model"),
            kwargs.get("tensor_parallel_size"),
            kwargs.get("distributed_executor_backend"),
            bundle_indices,
            vllm_port,
        )
        if vllm_port is not None:
            os.environ["VLLM_PORT"] = str(vllm_port)
        self._configure_device_env(
            backend=kwargs.get("distributed_executor_backend"),
            bundle_indices=bundle_indices,
            num_gpus=num_gpus,
        )
        self._configure_vllm_env(version, vllm, kwargs.pop("full_determinism", False))
        force_local_transformers_hub_kernels_for_model(kwargs.get("model"))
        llm_cls = vllm.LLM
        self.llm = llm_cls(**kwargs)
        # vLLM's documented RLHF recipe wakes `weights` and `kv_cache` in separate
        # steps; waking a tag that is already resident is outside that recipe and
        # has caused TP workers to desync in practice. Track per-tag state so
        # repeated wake_up calls for an already-awake tag become no-ops.
        self._awake_tags: set[str] = {"weights", "kv_cache"}
        self._sleep_level = 0
        logger.info("Offline vLLM actor is ready for model=%s.", kwargs.get("model"))

    def ping(self) -> bool:
        return True

    def _configure_device_env(self, backend, bundle_indices, num_gpus):
        if backend == "ray":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)

        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

    def _configure_vllm_env(self, version_module, vllm_module, full_determinism: bool):
        if version_module.parse(vllm_module.__version__) >= version_module.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if full_determinism:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
    ):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc(
            "update_weight",
            args=(name, dtype, shape, empty_cache),
        )

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    def sleep(self, level=1):
        level = int(level)
        if self._sleep_level >= level:
            logger.info("Offline vLLM actor already at sleep level %s; requested level %s.", self._sleep_level, level)
            return
        if self._sleep_level > 0:
            raise RuntimeError(
                "Cannot change vLLM sleep level while already asleep "
                f"(current={self._sleep_level}, requested={level}). "
                "Destroy/rebuild the engine instead."
            )
        self.llm.sleep(level=level)
        self._awake_tags.clear()
        self._sleep_level = level

    def wake_up(self, tags=("weights", "kv_cache")):
        if tags is None:
            if self._awake_tags >= {"weights", "kv_cache"}:
                return
            self.llm.wake_up()
            self._awake_tags = {"weights", "kv_cache"}
            self._sleep_level = 0
            return
        for tag in tags:
            if tag in self._awake_tags:
                continue
            self.llm.wake_up(tags=[tag])
            self._awake_tags.add(tag)
            self._sleep_level = 0

    def shutdown(self):
        """Gracefully tear down vLLM internals before Ray kills this actor.

        `ray.kill` terminates the top-level wrapper actor, but vLLM's Ray
        distributed executor owns child worker actors and NCCL/process-group
        state. Ask vLLM to shut those down first so temporary judge/RM engines
        don't leave stale workers behind for the next engine startup.
        """
        import gc

        llm_engine = getattr(self.llm, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        elif llm_engine is not None:
            model_executor = getattr(llm_engine, "model_executor", None)
            if model_executor is not None and hasattr(model_executor, "shutdown"):
                model_executor.shutdown()
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt_token_ids_batch: list[list[int]],
        sampling_kwargs: dict[str, Any],
        *,
        use_tqdm: bool = False,
        include_sampled_logprobs: bool = False,
    ):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sampling_params = SamplingParams(**sampling_kwargs)

        sequence_offsets = None
        sampled_logprobs_flat = None

        generate_start = time.monotonic()
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt_token_ids in prompt_token_ids_batch],
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )
        model_generate_duration = time.monotonic() - generate_start

        pack_start = time.monotonic()
        candidates = [
            candidate
            for request_output in outputs
            for candidate in request_output.outputs
        ]
        completion_sequences = [list(candidate.token_ids) for candidate in candidates]

        if include_sampled_logprobs:
            sequence_lengths = torch.tensor(
                [len(candidate.token_ids) for candidate in candidates],
                dtype=torch.int32,
            )
            sequence_offsets = torch.zeros(len(candidates) + 1, dtype=torch.int32)
            if sequence_lengths.numel() > 0:
                sequence_offsets[1:] = torch.cumsum(sequence_lengths, dim=0)

            sampled_logprobs_list: list[float] = []
            for candidate in candidates:
                logprob_entries = candidate.logprobs or ()
                num_logprob_entries = len(logprob_entries)
                for token_offset, token_id in enumerate(candidate.token_ids):
                    logprob_dict = logprob_entries[token_offset] if token_offset < num_logprob_entries else None
                    token_logprob = logprob_dict.get(token_id) if logprob_dict is not None else None
                    if token_logprob is None:
                        raise RuntimeError(
                            "vLLM did not return the sampled token logprob for every generated token."
                        )
                    sampled_logprobs_list.append(token_logprob.logprob)

            sampled_logprobs_flat = torch.tensor(sampled_logprobs_list, dtype=torch.float32)
        pack_duration = time.monotonic() - pack_start
        if self._log_phase_progress:
            total_completion_tokens = (
                int(sequence_offsets[-1].item())
                if torch.is_tensor(sequence_offsets)
                else sum(len(sequence) for sequence in completion_sequences)
            )
            logger.info(
                "stage=rollout phase=vllm_generate status=done prompts=%s completions=%s completion_tokens=%s "
                "include_sampled_logprobs=%s model_generate_seconds=%.4f "
                "pack_seconds=%.4f total_seconds=%.4f",
                len(prompt_token_ids_batch),
                len(completion_sequences),
                total_completion_tokens,
                include_sampled_logprobs,
                model_generate_duration,
                pack_duration,
                model_generate_duration + pack_duration,
            )

        return {
            "sequences": completion_sequences,
            "vllm_sequence_offsets": sequence_offsets,
            "vllm_logprobs_flat": sampled_logprobs_flat,
        }


def _is_address_in_use_error(exc: BaseException) -> bool:
    message = str(exc)
    return "EADDRINUSE" in message or "address already in use" in message


def _is_retryable_vllm_startup_error(exc: BaseException) -> bool:
    message = str(exc)
    return _is_address_in_use_error(exc) or "Engine core initialization failed" in message


def _read_int_env(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc


def _offline_vllm_port_job_key() -> str:
    explicit_key = os.environ.get("OFFLINE_VLLM_PORT_JOB_KEY")
    if explicit_key:
        return explicit_key
    return "|".join(
        value
        for value in (
            os.environ.get("EAI_JOB_ID"),
            os.environ.get("RAY_JOB_ID"),
            os.environ.get("SLURM_JOB_ID"),
            os.environ.get("JOB_ID"),
            os.environ.get("HOSTNAME"),
            os.environ.get("USER"),
            str(os.getpid()),
        )
        if value
    )


def _resolve_offline_vllm_port_base(
    *,
    num_engines: int,
    tensor_parallel_size: int,
    max_attempts: int,
    port_block_size: int,
) -> int | None:
    if tensor_parallel_size <= 1:
        return None

    explicit_base = _read_int_env("OFFLINE_VLLM_PORT_BASE")
    if explicit_base is None:
        explicit_base = _read_int_env("VLLM_PORT")
    if explicit_base is not None:
        return explicit_base

    range_start = _read_int_env("OFFLINE_VLLM_PORT_RANGE_START") or 20000
    range_end = _read_int_env("OFFLINE_VLLM_PORT_RANGE_END") or 60000
    if range_end <= range_start:
        raise ValueError("OFFLINE_VLLM_PORT_RANGE_END must be greater than OFFLINE_VLLM_PORT_RANGE_START.")

    attempt_span = max(1, max_attempts) * max(1, num_engines) * port_block_size
    available = range_end - range_start
    slots = available // attempt_span
    if slots <= 0:
        logger.warning(
            "Offline vLLM port range [%s, %s) is too small for %s engine(s), %s attempt(s), "
            "and block size %s; falling back to vLLM's default port selection.",
            range_start,
            range_end,
            num_engines,
            max_attempts,
            port_block_size,
        )
        return None

    digest = hashlib.sha1(_offline_vllm_port_job_key().encode("utf-8")).digest()
    slot = int.from_bytes(digest[:8], "big") % slots
    return range_start + slot * attempt_span


def create_offline_vllm_engine_bundle(
    *,
    num_engines: int,
    tensor_parallel_size: int,
    model_name: str,
    dtype: str | None,
    trust_remote_code: bool = False,
    seed: int,
    full_determinism: bool,
    max_model_len: int | None,
    max_logprobs: int | None,
    gpu_memory_utilization: float,
    vllm_enable_sleep: bool,
    vllm_enforce_eager: bool = False,
    log_phase_progress: bool = False,
    shared_pg=None,
) -> tuple[list[Any], Any | None]:
    max_attempts = max(1, int(os.environ.get("OFFLINE_VLLM_STARTUP_RETRIES", "3")))
    retry_wait_s = max(0.0, float(os.environ.get("OFFLINE_VLLM_STARTUP_RETRY_WAIT_S", "10")))
    port_block_size = max(1, int(os.environ.get("OFFLINE_VLLM_PORT_BLOCK_SIZE", "64")))
    vllm_port_base = _resolve_offline_vllm_port_base(
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        max_attempts=max_attempts,
        port_block_size=port_block_size,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            return _create_offline_vllm_engine_bundle_once(
                num_engines=num_engines,
                tensor_parallel_size=tensor_parallel_size,
                model_name=model_name,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                seed=seed,
                full_determinism=full_determinism,
                max_model_len=max_model_len,
                max_logprobs=max_logprobs,
                gpu_memory_utilization=gpu_memory_utilization,
                vllm_enable_sleep=vllm_enable_sleep,
                vllm_enforce_eager=vllm_enforce_eager,
                log_phase_progress=log_phase_progress,
                shared_pg=shared_pg,
                startup_attempt=attempt,
                vllm_port_base=vllm_port_base,
                vllm_port_block_size=port_block_size,
            )
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_vllm_startup_error(exc):
                raise
            logger.warning(
                "Offline vLLM startup hit a retryable EngineCore startup failure; retrying attempt %s/%s "
                "in %.1fs.",
                attempt + 1,
                max_attempts,
                retry_wait_s,
            )
            if retry_wait_s > 0:
                time.sleep(retry_wait_s)

    raise RuntimeError("unreachable")


def _create_offline_vllm_engine_bundle_once(
    *,
    num_engines: int,
    tensor_parallel_size: int,
    model_name: str,
    dtype: str | None,
    trust_remote_code: bool,
    seed: int,
    full_determinism: bool,
    max_model_len: int | None,
    max_logprobs: int | None,
    gpu_memory_utilization: float,
    vllm_enable_sleep: bool,
    vllm_enforce_eager: bool = False,
    log_phase_progress: bool = False,
    shared_pg=None,
    startup_attempt: int = 1,
    vllm_port_base: int | None = None,
    vllm_port_block_size: int = 64,
) -> tuple[list[Any], Any | None]:
    """Create offline vLLM Ray actors and return any dedicated placement group."""

    engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    dedicated_pg = None
    actor_num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        actor_num_gpus = 0.2
    # For tensor-parallel vLLM, the top-level Ray actor is only a coordinator.
    # The child workers consume the placement-group bundles, so reserving a full
    # CPU here can deadlock colocated placement groups before the actor starts.
    actor_num_cpus = actor_num_gpus

    if not use_hybrid_engine:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        logger.info(
            "Creating dedicated vLLM placement group with %s bundles for %s engine(s) at tp=%s.",
            len(bundles),
            num_engines,
            tensor_parallel_size,
        )
        shared_pg = placement_group(bundles, strategy="PACK")
        wait_for_ray_refs([shared_pg.ready()], description="vLLM placement group scheduling")
        dedicated_pg = shared_pg
    else:
        logger.info(
            "Using shared placement group for %s offline vLLM engine(s) at tp=%s.",
            num_engines,
            tensor_parallel_size,
        )

    try:
        startup_refs = []
        for index in range(num_engines):
            bundle_indices = None
            if tensor_parallel_size > 1:
                bundle_indices = get_bundle_indices(shared_pg, index, tensor_parallel_size)

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=shared_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_indices[0] if bundle_indices else index,
            )
            actor_kwargs = {
                "model": model_name,
                "worker_extension_cls": "ray_backend.vllm_worker_wrap.WorkerWrap",
                "tensor_parallel_size": tensor_parallel_size,
                "seed": seed + index,
                "distributed_executor_backend": distributed_executor_backend,
                "max_model_len": max_model_len,
                "trust_remote_code": trust_remote_code,
                "full_determinism": full_determinism,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": vllm_enforce_eager,
                "bundle_indices": bundle_indices,
                "num_gpus": 0.2 if use_hybrid_engine else 1,
                "enable_sleep_mode": vllm_enable_sleep,
                "log_phase_progress": log_phase_progress,
            }
            if vllm_port_base is not None:
                actor_kwargs["vllm_port"] = (
                    vllm_port_base
                    + ((max(1, startup_attempt) - 1) * num_engines + index) * vllm_port_block_size
                )
            if dtype is not None:
                actor_kwargs["dtype"] = dtype
            if max_logprobs is not None:
                actor_kwargs["max_logprobs"] = max_logprobs
            logger.info(
                "Scheduling offline vLLM engine %s/%s with bundle_indices=%s, ray_actor_num_cpus=%s, "
                "ray_actor_num_gpus=%s, vllm_worker_num_gpus=%s.",
                index + 1,
                num_engines,
                bundle_indices,
                actor_num_cpus,
                actor_num_gpus,
                actor_kwargs["num_gpus"],
            )
            engines.append(
                OfflineLLMRayActor.options(
                    num_cpus=actor_num_cpus,
                    num_gpus=actor_num_gpus,
                    scheduling_strategy=scheduling_strategy,
                ).remote(**actor_kwargs)
            )
            startup_refs.append(engines[-1].ping.remote())

        wait_for_ray_refs(
            startup_refs,
            description=f"offline vLLM engine bundle startup ({num_engines} engine(s))",
        )

        if vllm_enable_sleep:
            logger.info("Placing offline vLLM engines into sleep mode after startup.")
            batch_vllm_engine_call(engines, "sleep")
        return engines, dedicated_pg
    except Exception:
        if engines or dedicated_pg is not None:
            logger.exception("Offline vLLM engine startup failed; cleaning up partial bundle.")
            destroy_offline_vllm_engine_bundle(engines, dedicated_pg)
        raise


def _request_actor_termination(engine: Any, *, timeout_s: float = 30.0) -> bool:
    try:
        terminate = getattr(engine, "__ray_terminate__")
        terminate_ref = terminate.remote()
    except Exception:
        logger.exception("Failed to request graceful Ray actor termination.")
        return False
    try:
        ray.get(terminate_ref, timeout=timeout_s)
        return True
    except ray.exceptions.RayActorError:
        # Some Ray versions surface graceful actor exit as an actor-dead error
        # on the termination task. At this point the desired state is reached.
        return True
    except Exception:
        logger.exception("Graceful Ray actor termination did not complete before fallback kill.")
        return False


def destroy_offline_vllm_engine_bundle(engines: list[Any], placement_group_obj: Any | None) -> None:
    shutdown_refs: list[tuple[Any, Any]] = []
    fallback_kill_engines: list[Any] = []
    for engine in engines:
        try:
            shutdown_refs.append((engine, engine.shutdown.remote()))
        except Exception:
            logger.exception("Failed to request graceful vLLM actor shutdown before ray.kill.")
            fallback_kill_engines.append(engine)
    for engine, shutdown_ref in shutdown_refs:
        try:
            ray.get(shutdown_ref, timeout=60)
            if not _request_actor_termination(engine):
                fallback_kill_engines.append(engine)
        except Exception:
            logger.exception("Graceful vLLM actor shutdown did not complete before ray.kill.")
            fallback_kill_engines.append(engine)
    for engine in fallback_kill_engines:
        ray.kill(engine, no_restart=True)
    if placement_group_obj is not None:
        remove_placement_group(placement_group_obj)
    if fallback_kill_engines:
        # Hard kill can return before worker processes release their GPU pages,
        # which races with the next allocator call. Graceful termination has
        # already awaited actor exit, so only pause on the fallback path.
        release_wait_s = float(os.environ.get("OFFLINE_VLLM_RAY_KILL_WAIT_S", "5"))
        if release_wait_s > 0:
            logger.info("Waiting %.1fs after fallback ray.kill for GPU memory release.", release_wait_s)
            time.sleep(release_wait_s)


def batch_vllm_engine_call(
    engines: list[Any],
    method_name: str,
    *args,
    rank_0_only: bool = True,
    **kwargs,
):
    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        refs.append(getattr(engine, method_name).remote(*args, **kwargs))
    return wait_for_ray_refs(refs, description=f"offline vLLM `{method_name}`")
