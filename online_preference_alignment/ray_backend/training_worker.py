from __future__ import annotations

import logging
import math
import os
import socket
import time
from dataclasses import fields
from typing import Any

import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from cli import build_model_kwargs, initialize_distributed_runtime
from utils.deepspeed_utils import (
    build_hf_deepspeed_config,
    get_eval_ds_config,
    load_deepspeed_config_dict,
)
from utils.hub_kernel_utils import force_local_transformers_hub_kernels_for_model
from objectives import build_training_objective
from training.config import TrainingConfig
from training.trainer import Trainer
from ray_backend.launcher import BaseDistributedActor
from ray_backend.vllm_utils import get_physical_gpu_id, stateless_init_process_group
from utils.trainer_utils import (
    iter_deepspeed_weight_sync_params,
    normalize_vllm_param_name,
    torch_dist_barrier_and_cuda_sync,
)


logger = logging.getLogger(__name__)

_CUDA_CACHE_TRIM_MIN_IDLE_BYTES = 512 * 1024 * 1024
_CUDA_CACHE_TRIM_MIN_IDLE_RATIO = 0.25


@ray.remote(num_gpus=1)
class TrainingModelActor(BaseDistributedActor):
    """Distributed training worker for the Ray-backed training path."""

    def init_model_from_pretrained(
        self,
        *,
        config_kwargs: dict[str, Any],
        model_name: str,
        model_load_path: str | None = None,
        max_steps: int,
        vllm_engines=None,
        vllm_num_engines: int = 0,
        vllm_sync_backend: str = "nccl",
        ray_colocate_models: bool = False,
    ) -> dict[str, Any]:
        logger.info("[rank=%s] Starting Ray worker initialization for model %s.", self._rank, model_name)
        self.vllm_engines = vllm_engines
        self.vllm_num_engines = vllm_num_engines
        self.vllm_sync_backend = vllm_sync_backend
        self._shares_gpu_with_vllm = bool(ray_colocate_models)
        self.use_cuda_ipc = bool(ray_colocate_models and self.vllm_sync_backend == "nccl")
        self._model_update_group = None
        self._vllm_update_group_version = 0

        if self.vllm_sync_backend == "nccl":
            import vllm
            from packaging import version as pkg_version

            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        logger.info("[rank=%s] Initializing distributed runtime.", self._rank)
        initialize_distributed_runtime(self._world_size, local_rank=int(os.environ.get("LOCAL_RANK", "-1")))

        self.args = TrainingConfig(**config_kwargs)
        bf16_enabled = self.args.bf16
        runtime_deepspeed_config = load_deepspeed_config_dict(
            self.args.deepspeed,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            world_size=self.args.world_size,
            max_grad_norm=self.args.max_grad_norm,
            bf16=bf16_enabled,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            deepcompile=self.args.deepcompile,
            offload_optimizer=self.args.deepspeed_offload_optimizer,
            offload_param=self.args.deepspeed_offload_param,
        )

        model_kwargs = build_model_kwargs(
            self.args.attn_implementation,
            dtype=self.args.dtype,
            trust_remote_code=self.args.trust_remote_code,
        )
        # The student may be resumed from a checkpoint later, but the frozen reference
        # always anchors to the original pretrained snapshot — never to a resumed student.
        student_model_load_name = model_load_path or model_name
        frozen_reference_load_name = student_model_load_name
        training_objective = build_training_objective(self.args.training_objective)
        force_local_transformers_hub_kernels_for_model(model_name)
        force_local_transformers_hub_kernels_for_model(student_model_load_name)

        logger.info("[rank=%s] Loading student model weights.", self._rank)
        hf_deepspeed_config = build_hf_deepspeed_config(runtime_deepspeed_config)
        model = AutoModelForCausalLM.from_pretrained(student_model_load_name, **model_kwargs)
        hf_deepspeed_config = None

        ref_model = None
        if training_objective.requires_reference_model(self.args):
            logger.info("[rank=%s] Loading frozen reference model weights.", self._rank)
            ref_model_load_config = None
            if runtime_deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3:
                ref_model_load_config = get_eval_ds_config(
                    stage=3,
                    offload=self.args.ref_model_offload,
                    bf16=bf16_enabled,
                )
            ref_hf_deepspeed_config = (
                build_hf_deepspeed_config(ref_model_load_config)
                if ref_model_load_config is not None
                else None
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                frozen_reference_load_name,
                **model_kwargs,
            )
            ref_hf_deepspeed_config = None

        logger.info("[rank=%s] Loading tokenizer.", self._rank)
        tokenizer = AutoTokenizer.from_pretrained(
            student_model_load_name,
            trust_remote_code=self.args.trust_remote_code,
        )
        logger.info("[rank=%s] Building Trainer runtime.", self._rank)
        self.trainer = Trainer(
            model=model,
            ref_model=ref_model,
            args=self.args,
            processing_class=tokenizer,
        )
        self.trainer._initialize_runtime_engine()
        logger.info("[rank=%s] DeepSpeed runtime initialized.", self._rank)
        zero_stage = self.trainer._deepspeed_zero_stage()
        if self.use_cuda_ipc:
            logger.info("[rank=%s] Using CUDA IPC for vLLM weight sync (ZeRO stage=%s).", self._rank, zero_stage)
        else:
            logger.info("[rank=%s] Using %s communicator for vLLM weight sync.", self._rank, self.vllm_sync_backend)

        if self.args.warmup_steps is not None:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_steps = math.ceil(max_steps * self.args.warmup_ratio)
        scheduler_optimizer = self.trainer._scheduler_optimizer()
        self.trainer.scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=scheduler_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        if self.trainer.deepspeed_engine is not None:
            self.trainer.deepspeed_engine.lr_scheduler = self.trainer.scheduler

        if self.args.resume_from:
            # Resume only restores the student engine/optimizer state. The
            # frozen reference above remains on its separately-resolved source.
            self.trainer._load_checkpoint(self.args.resume_from)

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            logger.info("[rank=%s] Initializing vLLM weight-sync process group.", self._rank)
            self._init_vllm_update_group()

        logger.info("[rank=%s] Ray worker initialization finished.", self._rank)
        return self.get_training_state()

    def _maybe_trim_cuda_cache(self, *, force: bool = False) -> bool:
        device = getattr(self.trainer, "device", None)
        if device is None or device.type != "cuda":
            return False

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        idle_bytes = max(0, reserved - allocated)
        should_trim = force
        if not should_trim and reserved > 0:
            idle_ratio = idle_bytes / reserved
            should_trim = (
                idle_bytes >= _CUDA_CACHE_TRIM_MIN_IDLE_BYTES
                and idle_ratio >= _CUDA_CACHE_TRIM_MIN_IDLE_RATIO
            )
        if not should_trim:
            return False

        import gc

        gc.collect()
        torch.cuda.empty_cache()
        return True

    def _init_vllm_update_group(self) -> None:
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        world_size = self.vllm_num_engines * self.args.vllm_tensor_parallel_size + 1
        self._vllm_update_group_version += 1
        group_name = f"preference_rl_{self._vllm_update_group_version}"
        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                index * self.args.vllm_tensor_parallel_size + 1,
                world_size,
                group_name,
                backend=self.vllm_sync_backend,
            )
            for index, engine in enumerate(self.vllm_engines)
        ]
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            0,
            world_size,
            torch.cuda.current_device(),
        )
        logger.info(
            "[rank=%s] Waiting for %s vLLM engine(s) to join weight-sync group.",
            self._rank,
            len(refs),
        )
        ray.get(refs)
        logger.info("[rank=%s] vLLM weight-sync process group is ready.", self._rank)

    def set_vllm_engines(self, vllm_engines=None, *, vllm_num_engines: int | None = None) -> None:
        self.vllm_engines = vllm_engines
        if vllm_num_engines is not None:
            self.vllm_num_engines = int(vllm_num_engines)
        if (
            self.vllm_engines is not None
            and not self.use_cuda_ipc
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        ):
            logger.info("[rank=%s] Re-initializing vLLM weight-sync process group.", self._rank)
            self._init_vllm_update_group()

    def _broadcast_param(self, name: str, param: torch.Tensor, count: int, num_params: int) -> None:
        if torch.distributed.get_rank() != 0:
            return

        shape = param.shape if not hasattr(param, "ds_shape") else param.ds_shape
        refs = [
            engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
            for engine in self.vllm_engines
        ]
        self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
        ray.get(refs)

    def _broadcast_param_cuda_ipc(self, name: str, param: torch.Tensor, count: int, num_params: int) -> None:
        shape = param.shape if not hasattr(param, "ds_shape") else param.ds_shape
        expected_shape = tuple(int(dim) for dim in shape)
        weight = param.data if param.data.is_contiguous() else param.data.contiguous()
        local_metadata = {
            "rank": int(torch.distributed.get_rank()),
            "gpu": get_physical_gpu_id(),
            "shape": tuple(int(dim) for dim in weight.shape),
            "numel": int(weight.numel()),
        }
        metadata_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(metadata_list, local_metadata)
        invalid_metadata = [
            metadata
            for metadata in metadata_list
            if metadata is None or tuple(metadata["shape"]) != expected_shape or int(metadata["numel"]) <= 0
        ]
        if invalid_metadata:
            raise RuntimeError(
                "Cannot publish ZeRO-gathered parameter via CUDA IPC because at least one rank does not "
                f"hold the expected full tensor for {name}: expected_shape={expected_shape}, "
                f"invalid_rank_metadata={invalid_metadata}"
            )

        ipc_handle = reduce_tensor(weight)
        ipc_handle = {get_physical_gpu_id(): ipc_handle}
        ipc_handle_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

        if torch.distributed.get_rank() != 0:
            torch_dist_barrier_and_cuda_sync(torch.device("cuda", torch.cuda.current_device()))
            return

        ipc_handles = {}
        for handle_map in ipc_handle_list:
            ipc_handles.update(handle_map)
        refs = [
            engine.update_weight_cuda_ipc.remote(
                name,
                dtype=param.dtype,
                shape=shape,
                ipc_handles=ipc_handles,
                empty_cache=count == num_params,
            )
            for engine in self.vllm_engines
        ]
        ray.get(refs)
        torch_dist_barrier_and_cuda_sync(torch.device("cuda", torch.cuda.current_device()))

    def train_step(
        self,
        training_batch: dict[str, Any],
        *,
        num_input_tokens_seen: int | None = None,
    ) -> dict[str, Any]:
        self.trainer.model.train()
        self.trainer.base_model.train()
        if self.args.log_phase_progress:
            completion_mask = training_batch.get("completion_mask")
            completion_tokens = int(completion_mask.sum().item()) if torch.is_tensor(completion_mask) else 0
            packed_offsets = training_batch.get("vllm_sequence_offsets")
            packed_vllm_tokens = int(packed_offsets[-1].item()) if torch.is_tensor(packed_offsets) else 0
            logger.info(
                "[rank=%s] stage=train_step status=start objective=%s prompts=%s completion_tokens=%s packed_vllm_tokens=%s",
                self._rank,
                self.args.training_objective,
                training_batch.get("num_prompts"),
                completion_tokens,
                packed_vllm_tokens,
            )
        trim_start = time.monotonic()
        local_batch = self.trainer._trim_training_batch_tensors(training_batch)
        trim_duration = time.monotonic() - trim_start
        split_start = time.monotonic()
        micro_batches = self.trainer._split_into_micro_batches(local_batch)
        split_duration = time.monotonic() - split_start
        del training_batch, local_batch
        if self._shares_gpu_with_vllm:
            self._maybe_trim_cuda_cache(force=True)
        train_start = time.monotonic()
        metrics = self.trainer._run_training_step(micro_batches)
        train_duration = time.monotonic() - train_start
        if num_input_tokens_seen is not None:
            self.trainer.num_input_tokens_seen = int(num_input_tokens_seen)
        self.trainer.global_step += 1
        metrics["learning_rate"] = self.trainer.scheduler.get_last_lr()[0]
        if self._shares_gpu_with_vllm:
            self._maybe_trim_cuda_cache()
        if self.args.log_phase_progress:
            logger.info(
                "[rank=%s] stage=train_step status=done micro_batches=%s trim_seconds=%.4f split_seconds=%.4f train_seconds=%.4f total_seconds=%.4f",
                self._rank,
                len(micro_batches),
                trim_duration,
                split_duration,
                train_duration,
                trim_duration + split_duration + train_duration,
            )
        return {
            "global_step": self.trainer.global_step,
            "metrics": metrics,
            "rank": self.args.rank,
        }

    def eval_step(self, training_batch: dict[str, Any], num_generations: int = 1) -> dict[str, Any]:
        """Forward-only loss computation on validation data — no gradient, no optimizer."""
        if self._shares_gpu_with_vllm:
            self._maybe_trim_cuda_cache(force=True)
        metrics = self.trainer._run_eval_forward(training_batch, num_generations=num_generations)
        if self._shares_gpu_with_vllm:
            self._maybe_trim_cuda_cache()
        return {"metrics": metrics, "rank": self.args.rank}

    def broadcast_to_vllm(self) -> None:
        if not self.vllm_engines:
            return

        if self._shares_gpu_with_vllm:
            self._maybe_trim_cuda_cache(force=True)
        zero_stage = self.trainer._deepspeed_zero_stage()
        if zero_stage == 3:
            torch_dist_barrier_and_cuda_sync(self.trainer.device)
        model = self.trainer._unwrap_model(self.trainer.model)
        for count, num_params, name, param in iter_deepspeed_weight_sync_params(
            model,
            zero_stage=zero_stage,
        ):
            fixed_name = normalize_vllm_param_name(name)
            if self.use_cuda_ipc:
                self._broadcast_param_cuda_ipc(fixed_name, param, count, num_params)
                continue

            self._broadcast_param(fixed_name, param, count, num_params)

        if self._shares_gpu_with_vllm:
            self._maybe_trim_cuda_cache()
        torch_dist_barrier_and_cuda_sync(self.trainer.device)

    def save_checkpoint(self, name: str) -> None:
        self.trainer._save_checkpoint(name)

    def get_training_state(self) -> dict[str, Any]:
        return {
            "rank": self.args.rank,
            "global_step": self.trainer.global_step,
            "num_input_tokens_seen": self.trainer.num_input_tokens_seen,
        }


def build_training_config_init_kwargs(config: TrainingConfig) -> dict[str, Any]:
    return {
        field.name: getattr(config, field.name)
        for field in fields(TrainingConfig)
        if field.init
    }
