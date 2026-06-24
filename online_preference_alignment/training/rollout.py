"""Prompt handling and rollout construction helpers."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any

import torch
from torch import nn

from .batch_types import TrajectoryBatch
from .tokenization import TokenizationHelper
from .batching import (
    move_training_batch_to_device,
    split_training_batch_into_micro_batches,
)
from utils.trainer_utils import (
    format_prompt_text,
    student_thinking_enabled,
    unwrap_model,
)
from utils.text_utils import decode_response_tokens


logger = logging.getLogger(__name__)


class TrainerRolloutMixin:
    """Builds prompt/completion batches and prepares rollout tensors."""

    def _training_batch_layout(self):
        objective = getattr(self, "training_objective", None)
        if objective is None:
            raise RuntimeError("TrainerRolloutMixin requires training_objective to be initialized.")
        return objective.training_batch_layout()

    def _resolve_stop_token_ids(self) -> list[int]:
        ordered_ids: list[int] = []
        seen_ids: set[int] = set()

        def add_token_ids(token_ids: Any) -> None:
            if token_ids is None:
                return
            if isinstance(token_ids, int):
                values = [token_ids]
            elif isinstance(token_ids, (list, tuple)):
                values = token_ids
            else:
                raise TypeError(
                    "Stop token ids must be an int, list[int], tuple[int, ...], or None."
                )

            for token_id in values:
                if not isinstance(token_id, int):
                    raise TypeError("Stop token ids must contain only ints.")
                if token_id in seen_ids:
                    continue
                seen_ids.add(token_id)
                ordered_ids.append(token_id)

        add_token_ids(getattr(self, "eos_token_id", None))
        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None:
            add_token_ids(getattr(processing_class, "eos_token_id", None))

        for model_attr in ("base_model", "model", "ref_model"):
            model = getattr(self, model_attr, None)
            if model is None:
                continue
            generation_config = getattr(model, "generation_config", None)
            if generation_config is None:
                generation_config = getattr(self._unwrap_model(model), "generation_config", None)
            if generation_config is None:
                continue
            add_token_ids(getattr(generation_config, "eos_token_id", None))
            add_token_ids(getattr(generation_config, "stop_token_ids", None))

        if not ordered_ids and not getattr(self, "_missing_stop_token_ids_warned", False):
            logger.warning(
                "Could not resolve any EOS or stop token ids for rollout generation; "
                "continuing without stop_token_ids."
            )
            self._missing_stop_token_ids_warned = True

        return ordered_ids

    def _get_tokenization_helper(self) -> TokenizationHelper:
        """Lazily build a TokenizationHelper from this mixin's instance state."""
        helper = getattr(self, "_tokenization_helper", None)
        if helper is not None:
            return helper
        processing_class = getattr(self, "processing_class", None)
        if processing_class is None:
            raise AttributeError("TrainerRolloutMixin requires processing_class.")
        helper = TokenizationHelper(
            processing_class,
            pad_token_id=getattr(self, "pad_token_id", None),
            eos_token_id=getattr(self, "eos_token_id", None),
            device=getattr(self, "device", torch.device("cpu")),
        )
        self._tokenization_helper = helper
        return helper

    def _tokenize_prompt_text_sequences(
        self,
        prompt_texts: list[str],
    ) -> list[list[int]]:
        return self._get_tokenization_helper().tokenize_prompt_text_sequences(prompt_texts)

    def _example_prompt_token_lengths(
        self,
        example: dict[str, Any],
    ) -> dict[str, int]:
        prompt_variants = {
            "prompt": example["prompt"],
        }

        prompt_names = list(prompt_variants.keys())
        prompt_texts = [
            self._format_student_prompt(prompt_variants[name])
            for name in prompt_names
        ]
        token_sequences = self._tokenize_prompt_text_sequences(prompt_texts)
        return {
            name: len(token_sequence)
            for name, token_sequence in zip(prompt_names, token_sequences)
        }

    def _build_prompt_tensors_from_token_sequences(
        self,
        token_sequences: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        return self._get_tokenization_helper().build_prompt_tensors(token_sequences)

    def _move_batch_to_device(
        self,
        batch: TrajectoryBatch,
    ) -> TrajectoryBatch:
        return move_training_batch_to_device(
            batch,
            device=getattr(self, "device", torch.device("cpu")),
            layout=self._training_batch_layout(),
        )

    def _format_prompt(self, prompt: Any) -> str:
        return format_prompt_text(
            prompt,
            self.processing_class,
            enable_thinking=self._student_thinking_enabled(),
        )

    def _student_thinking_enabled(self) -> bool:
        return student_thinking_enabled(self.args)

    def _format_student_prompt(self, prompt: Any) -> str:
        return format_prompt_text(
            prompt,
            self.processing_class,
            enable_thinking=self._student_thinking_enabled(),
            last_user_instruction=getattr(self.args, "student_last_user_instruction", None),
        )

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        return unwrap_model(model)

    def _autocast_context(self, device_type: str | None = None):
        resolved_device_type = self.device.type if device_type is None else device_type
        if resolved_device_type != "cuda":
            return nullcontext()
        if self.args.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _pad_completion_sequences(
        self,
        completion_sequences: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_tokenization_helper().pad_completion_sequences(completion_sequences)

    def _response_word_counts(
        self,
        response_texts: list[str],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Count visible response words after removing thinking traces."""
        counts = [float(len(response_text.split())) for response_text in response_texts]
        return torch.tensor(counts, dtype=torch.float32, device=device)

    def _decode_response_texts(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> list[str]:
        """Decode visible response text once for metrics and reward scorers."""
        completion_lengths = completion_mask.sum(dim=1).tolist()
        all_token_ids = completion_ids.tolist()
        return [
            decode_response_tokens(
                self.processing_class,
                token_ids[: int(valid_length)],
                strip_thinking_traces=True,
            )
            for token_ids, valid_length in zip(all_token_ids, completion_lengths)
        ]

    def _store_rollout_prompt_tensors(
        self,
        rollout_batch: TrajectoryBatch,
        *,
        prefix: str,
        prompt_tensors: dict[str, torch.Tensor],
    ) -> None:
        rollout_batch[f"{prefix}_ids"] = prompt_tensors["input_ids"]
        rollout_batch[f"{prefix}_mask"] = prompt_tensors["attention_mask"]

    def _store_rollout_completion_diagnostics(
        self,
        rollout_batch: TrajectoryBatch,
        *,
        completion_mask: torch.Tensor,
        completion_vllm_diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if completion_vllm_diagnostics is None:
            return
        sequence_offsets = completion_vllm_diagnostics.get("vllm_sequence_offsets")
        if not torch.is_tensor(sequence_offsets):
            raise TypeError("completion_vllm_diagnostics must include a tensor vllm_sequence_offsets.")
        if sequence_offsets.ndim != 1 or sequence_offsets.numel() != completion_mask.size(0) + 1:
            raise ValueError(
                "vllm_sequence_offsets must be a 1D tensor with one entry per completion sequence plus a trailing end offset."
            )
        total_tokens = int(sequence_offsets[-1].item())
        if total_tokens != int(completion_mask.sum().item()):
            raise ValueError(
                "Packed vLLM diagnostics must contain exactly one token entry per generated completion token."
            )
        rollout_batch["vllm_sequence_offsets"] = sequence_offsets

        vllm_logprobs_flat = completion_vllm_diagnostics.get("vllm_logprobs_flat")
        if torch.is_tensor(vllm_logprobs_flat):
            if vllm_logprobs_flat.ndim != 1 or vllm_logprobs_flat.size(0) != total_tokens:
                raise ValueError("vllm_logprobs_flat must have shape [total_completion_tokens].")
            rollout_batch["vllm_logprobs_flat"] = vllm_logprobs_flat

    def _trim_packed_vllm_diagnostics(
        self,
        rollout_batch: TrajectoryBatch,
        *,
        completion_mask: torch.Tensor,
    ) -> None:
        sequence_offsets = rollout_batch.get("vllm_sequence_offsets")
        if not torch.is_tensor(sequence_offsets):
            return
        if sequence_offsets.ndim != 1 or sequence_offsets.numel() != completion_mask.size(0) + 1:
            raise ValueError(
                "vllm_sequence_offsets must align with the number of completion sequences in completion_mask."
            )
        original_lengths = sequence_offsets[1:] - sequence_offsets[:-1]
        trimmed_lengths = completion_mask.sum(dim=1, dtype=sequence_offsets.dtype)
        if torch.equal(original_lengths, trimmed_lengths):
            return

        offsets_list = sequence_offsets.tolist()
        total_original_tokens = offsets_list[-1]
        trimmed_lengths_list = trimmed_lengths.tolist()
        new_offsets = torch.zeros_like(sequence_offsets)
        if trimmed_lengths.numel() > 0:
            new_offsets[1:] = torch.cumsum(trimmed_lengths, dim=0)

        key = "vllm_logprobs_flat"
        value = rollout_batch.get(key)
        if torch.is_tensor(value):
            if value.size(0) != total_original_tokens:
                raise ValueError(f"{key} does not match vllm_sequence_offsets.")
            chunks: list[torch.Tensor] = []
            for sequence_index, trimmed_length in enumerate(trimmed_lengths_list):
                source_start = offsets_list[sequence_index]
                source_end = offsets_list[sequence_index + 1]
                if source_end - source_start < trimmed_length:
                    raise ValueError(f"{key} is shorter than completion_mask for sequence {sequence_index}.")
                if trimmed_length > 0:
                    chunks.append(value[source_start : source_start + trimmed_length])
            trimmed_value = (
                torch.cat(chunks, dim=0)
                if chunks
                else value.new_empty((0, *value.shape[1:]))
            )
            rollout_batch[key] = trimmed_value
        rollout_batch["vllm_sequence_offsets"] = new_offsets

    def _build_trajectory_batch(
        self,
        *,
        prompt_batch: list[dict[str, Any]],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        prompt_tensors: dict[str, torch.Tensor],
        update_token_count: bool,
        completion_vllm_diagnostics: dict[str, torch.Tensor] | None = None,
        reduce_rollout_metrics: bool = True,
    ) -> tuple[TrajectoryBatch, dict[str, float]]:
        completion_texts = self._decode_response_texts(completion_ids, completion_mask)
        rollout_batch: TrajectoryBatch = {
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completion_texts": completion_texts,
            "num_prompts": len(prompt_batch),
        }
        self._store_rollout_completion_diagnostics(
            rollout_batch,
            completion_mask=completion_mask,
            completion_vllm_diagnostics=completion_vllm_diagnostics,
        )
        self._store_rollout_prompt_tensors(
            rollout_batch,
            prefix="prompt",
            prompt_tensors=prompt_tensors,
        )

        completion_lengths = completion_mask.sum(dim=1).float()
        response_word_counts = self._response_word_counts(
            completion_texts,
            device=completion_mask.device,
        )
        clipped_ratio = (completion_lengths == float(self.args.max_completion_length)).float().mean()
        prompt_attention_mask = prompt_tensors["attention_mask"]
        prompt_token_count = int(prompt_attention_mask.sum().item())
        if prompt_attention_mask.size(0) == len(prompt_batch):
            prompt_token_count *= self.args.num_generations
        batch_tokens_seen = prompt_token_count + int(completion_lengths.sum().item())
        if update_token_count:
            self.num_input_tokens_seen += self._distributed_sum_int(batch_tokens_seen)

        if reduce_rollout_metrics:
            mean_length = self._distributed_mean(completion_lengths.mean()).item()
            mean_response_word_count = self._distributed_mean(response_word_counts.mean()).item()
            clipped_ratio_value = self._distributed_mean(clipped_ratio).item()
        else:
            mean_length = completion_lengths.mean().item()
            mean_response_word_count = response_word_counts.mean().item()
            clipped_ratio_value = clipped_ratio.item()
        return rollout_batch, {
            "completions/mean_length": mean_length,
            "completions/response_word_count_mean": mean_response_word_count,
            "completions/clipped_ratio": clipped_ratio_value,
        }

    def _trim_sequence_padded_tensor(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        padding_side: str,
        max_length: int | None = None,
    ) -> torch.Tensor:
        if values.ndim < 2 or mask.ndim != 2 or values.size(1) == 0 or mask.size(1) == 0:
            return values

        active_length = int(mask.sum(dim=1).max().item())
        if max_length is not None:
            active_length = min(active_length, max_length)
        if active_length <= 0 or active_length >= values.size(1):
            return values

        slicer = [slice(None)] * values.ndim
        if padding_side == "left":
            slicer[1] = slice(values.size(1) - active_length, None)
        else:
            slicer[1] = slice(None, active_length)
        return values[tuple(slicer)]

    def _trim_training_batch_tensors(
        self,
        batch: TrajectoryBatch,
    ) -> TrajectoryBatch:
        trimmed_batch: TrajectoryBatch = dict(batch)
        layout = self._training_batch_layout()

        # Prompt tensors are left-padded and are trimmed with their own masks.
        for ids_key in layout.prompt_tensor_keys:
            if not ids_key.endswith("_ids"):
                continue
            mask_key = f"{ids_key[:-4]}_mask"
            if mask_key not in layout.prompt_tensor_keys:
                continue
            ids = trimmed_batch.get(ids_key)
            mask = trimmed_batch.get(mask_key)
            if not torch.is_tensor(ids) or not torch.is_tensor(mask):
                continue
            trimmed_batch[ids_key] = self._trim_sequence_padded_tensor(
                ids,
                mask,
                padding_side="left",
            )
            trimmed_batch[mask_key] = self._trim_sequence_padded_tensor(
                mask,
                mask,
                padding_side="left",
            )

        completion_mask = trimmed_batch.get("completion_mask")
        if torch.is_tensor(completion_mask):
            max_train_completion_length = self.args.max_train_completion_length
            for key in layout.sequence_tensor_keys:
                value = trimmed_batch.get(key)
                if not torch.is_tensor(value):
                    continue
                if (
                    value.ndim < 2
                    or value.size(0) != completion_mask.size(0)
                    or value.size(1) != completion_mask.size(1)
                ):
                    continue
                trimmed_batch[key] = self._trim_sequence_padded_tensor(
                    value,
                    completion_mask,
                    padding_side="right",
                    max_length=max_train_completion_length,
                )
            trimmed_completion_mask = trimmed_batch.get("completion_mask")
            if torch.is_tensor(trimmed_completion_mask):
                self._trim_packed_vllm_diagnostics(
                    trimmed_batch,
                    completion_mask=trimmed_completion_mask,
                )

        return trimmed_batch

    def _split_into_micro_batches(
        self,
        rollout_batch: TrajectoryBatch,
    ) -> list[TrajectoryBatch]:
        layout = self._training_batch_layout()
        micro_batches = split_training_batch_into_micro_batches(
            rollout_batch,
            layout=layout,
            num_generations=self.args.num_generations,
            prompts_per_micro_batch=self.args.per_device_train_batch_size,
            max_sequences_per_micro_batch=self.args.per_device_train_batch_size,
        )
        return [
            self._trim_training_batch_tensors(micro_batch)
            for micro_batch in micro_batches
        ]
