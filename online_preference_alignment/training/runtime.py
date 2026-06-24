"""Runtime and DeepSpeed lifecycle helpers for `Trainer`."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from transformers import PreTrainedModel

from utils.deepspeed_utils import (
    build_deepspeed_optimizer,
    get_eval_ds_config,
    load_deepspeed_config_dict,
    strip_optimizer_from_config,
)
from utils.trainer_utils import (
    disable_dropout_in_model,
    is_distributed_ready,
)


logger = logging.getLogger(__name__)


class TrainerRuntimeMixin:
    """Owns logging, process setup, and DeepSpeed lifecycle."""

    def _maybe_compile_deepspeed_engine(self, engine: Any, *, name: str) -> None:
        if not self.args.deepcompile:
            return
        compile_fn = getattr(engine, "compile", None)
        if not callable(compile_fn):
            logger.warning("DeepCompile requested for %s engine, but engine.compile() is unavailable.", name)
            return
        logger.info("Compiling %s DeepSpeed engine.", name)
        compile_fn()

    def _setup_logging(self) -> None:
        if self.args.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
            )
        else:
            logging.basicConfig(level=logging.WARNING)

    def _setup_distributed(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.args.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

    def _prepare_ref_model(
        self,
        ref_model: str | PreTrainedModel | None,
    ) -> PreTrainedModel | None:
        if ref_model is None:
            return None
        if isinstance(ref_model, str):
            raise TypeError("Pass an instantiated ref_model into Trainer.")
        ref_model.requires_grad_(False)
        ref_model.eval()
        if self.args.disable_dropout:
            disable_dropout_in_model(ref_model)
        return ref_model

    def _initialize_runtime_engine(self) -> None:
        if self.deepspeed_engine is not None:
            return

        import deepspeed

        if self.args.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        ds_config = self._load_deepspeed_config()
        optimizer = build_deepspeed_optimizer(self.base_model, ds_config)
        engine_config = strip_optimizer_from_config(ds_config) if optimizer is not None else ds_config

        initialize_kwargs: dict[str, Any] = {
            "model": self.base_model,
            "config": engine_config,
        }
        if optimizer is not None:
            initialize_kwargs["optimizer"] = optimizer
        else:
            initialize_kwargs["model_parameters"] = self.base_model.parameters()

        engine, optimizer, _, _ = deepspeed.initialize(**initialize_kwargs)
        self._maybe_compile_deepspeed_engine(engine, name="training")
        self.deepspeed_engine = engine
        self.model = engine
        self.optimizer = optimizer

        if self.ref_model is not None:
            self.ref_model = self._wrap_frozen_model_with_deepspeed(
                self.ref_model,
                offload=self.args.ref_model_offload,
            )

    def _scheduler_optimizer(self) -> Optimizer:
        optimizer = self.optimizer
        if optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")

        seen: set[int] = set()
        current: Any = optimizer
        while id(current) not in seen:
            if isinstance(current, Optimizer):
                return current
            seen.add(id(current))
            nested = getattr(current, "optimizer", None)
            if nested is None:
                break
            current = nested

        raise TypeError(f"{type(optimizer).__name__} does not wrap a torch.optim.Optimizer")

    def _wrap_frozen_model_with_deepspeed(
        self,
        model: nn.Module,
        *,
        offload: bool,
    ) -> nn.Module:
        """Wrap the frozen reference model with a DeepSpeed eval engine for ZeRO-3 sharding."""
        import deepspeed

        zero_stage = self._deepspeed_zero_stage()
        eval_ds_config = get_eval_ds_config(
            stage=zero_stage if zero_stage == 3 else 0,
            offload=offload,
            bf16=self.args.bf16,
        )
        engine, *_ = deepspeed.initialize(
            model=model,
            config=eval_ds_config,
        )
        return engine

    def _load_deepspeed_config(self) -> dict[str, Any]:
        if torch.distributed.is_initialized():
            ds_world_size = torch.distributed.get_world_size()
        else:
            ds_world_size = self.args.world_size

        config = load_deepspeed_config_dict(
            self.args.deepspeed,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            world_size=ds_world_size,
            max_grad_norm=self.args.max_grad_norm,
            bf16=self.args.bf16,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            deepcompile=self.args.deepcompile,
            offload_optimizer=self.args.deepspeed_offload_optimizer,
            offload_param=self.args.deepspeed_offload_param,
        )

        self.deepspeed_config = config
        return config

    def _deepspeed_zero_stage(self) -> int:
        if self.deepspeed_config is None:
            return 0
        return int(self.deepspeed_config.get("zero_optimization", {}).get("stage", 0))

    # ------------------------------------------------------------------
    # Distributed communication helpers
    # ------------------------------------------------------------------

    def _distributed_mean(self, value: torch.Tensor) -> torch.Tensor:
        tensor = value.detach().to(device=self.device, dtype=torch.float32)
        if is_distributed_ready():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        return tensor

    def _distributed_sum(self, value: torch.Tensor) -> torch.Tensor:
        tensor = value.detach().to(device=self.device, dtype=torch.float32)
        if is_distributed_ready():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _distributed_sum_int(self, value: int) -> int:
        tensor = torch.tensor(value, device=self.device, dtype=torch.long)
        if is_distributed_ready():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def _distributed_barrier(self) -> None:
        if is_distributed_ready():
            dist.barrier()
