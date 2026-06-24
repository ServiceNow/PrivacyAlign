"""Checkpoint save, load, and export helpers for the trainer stack."""

from __future__ import annotations

import copy
import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn


logger = logging.getLogger(__name__)

DEEPSPEED_CHECKPOINT_TAG = "deepspeed"


class TrainerCheckpointMixin:
    """Checkpoint I/O used by the Trainer and its DeepSpeed engine."""

    def _gather_zero3_16bit_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor] | None:
        gather_state_dict = getattr(model, "_consolidated_16bit_state_dict", None)
        if gather_state_dict is None:
            gather_state_dict = getattr(model, "_zero3_consolidated_16bit_state_dict", None)
        if not callable(gather_state_dict):
            raise RuntimeError(
                "DeepSpeed ZeRO-3 export requires a model engine exposing "
                "_consolidated_16bit_state_dict()."
            )
        return gather_state_dict(exclude_frozen_parameters=False)

    def _save_zero3_model_export(self, model: nn.Module, output_dir: Path) -> None:
        state_dict = self._gather_zero3_16bit_state_dict(model)
        try:
            if self.args.is_main_process:
                if state_dict is None:
                    raise RuntimeError("Main process did not receive a consolidated ZeRO-3 state dict.")
                torch.save(state_dict, output_dir / "pytorch_model.bin")
        finally:
            del state_dict
            gc.collect()

    def _write_export_metadata(self, output_dir: Path, state: dict[str, Any]) -> None:
        model_to_save = self._unwrap_model(self.model)
        model_to_save.config.save_pretrained(output_dir)
        generation_config = getattr(model_to_save, "generation_config", None)
        if generation_config is not None:
            generation_config = copy.deepcopy(generation_config)
            generation_config.do_sample = True
            generation_config.save_pretrained(output_dir)
        self.processing_class.save_pretrained(output_dir)
        with (output_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def _write_latest_checkpoint_metadata(self, output_dir: Path, state: dict[str, Any]) -> None:
        latest_state = {
            "checkpoint_name": output_dir.name,
            "checkpoint_path": str(output_dir.resolve()),
            **state,
        }
        latest_path = Path(self.args.output_dir) / "latest_checkpoint.json"
        with latest_path.open("w", encoding="utf-8") as handle:
            json.dump(latest_state, handle, indent=2)

    def _load_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        output_dir = Path(checkpoint_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {output_dir}")
        if self.deepspeed_engine is None:
            raise RuntimeError("DeepSpeed engine must be initialized before loading a checkpoint.")

        load_path, client_state = self.deepspeed_engine.load_checkpoint(
            str(output_dir),
            tag=DEEPSPEED_CHECKPOINT_TAG,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_only=False,
        )
        self._distributed_barrier()
        if load_path is None:
            raise RuntimeError(f"DeepSpeed failed to load checkpoint from {output_dir}.")

        if not client_state:
            state_path = output_dir / "trainer_state.json"
            if state_path.is_file():
                with state_path.open("r", encoding="utf-8") as handle:
                    client_state = json.load(handle)
            else:
                client_state = {}

        self.global_step = int(client_state.get("global_step", 0))
        self.num_input_tokens_seen = int(client_state.get("num_input_tokens_seen", 0))
        logger.info("Loaded checkpoint %s at global_step=%s", output_dir, self.global_step)
        return client_state

    def _save_checkpoint(self, name: str) -> None:
        output_dir = Path(self.args.output_dir) / name
        if self.args.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self._distributed_barrier()

        state = {
            "global_step": self.global_step,
            "num_input_tokens_seen": self.num_input_tokens_seen,
            "config": self.args.to_dict(),
            "deepspeed_tag": DEEPSPEED_CHECKPOINT_TAG,
        }

        if self.deepspeed_engine is not None:
            self.deepspeed_engine.save_checkpoint(
                str(output_dir),
                tag=DEEPSPEED_CHECKPOINT_TAG,
                client_state=state,
                save_latest=True,
            )
            self._distributed_barrier()

        if self.deepspeed_engine is not None and self._deepspeed_zero_stage() == 3:
            self._save_zero3_model_export(self.deepspeed_engine, output_dir)
            self._distributed_barrier()
            if self.args.is_main_process:
                self._write_export_metadata(output_dir, state)
                self._write_latest_checkpoint_metadata(output_dir, state)
                logger.info("Saved checkpoint %s to %s", name, output_dir)
            self._distributed_barrier()
            return

        if self.args.is_main_process:
            self._unwrap_model(self.model).save_pretrained(output_dir)
            self.processing_class.save_pretrained(output_dir)
            with (output_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
            self._write_latest_checkpoint_metadata(output_dir, state)
            logger.info("Saved checkpoint %s to %s", name, output_dir)
        self._distributed_barrier()
