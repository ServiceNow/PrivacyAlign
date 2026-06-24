from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace
import time
from typing import Any

import torch

from training.batch_types import PolicyOptimizationBatch, TrajectoryBatch
from objectives import build_training_objective
from utils.sampling_utils import build_rollout_sampling_kwargs
from utils.text_utils import (
    build_response_token_mask,
)
from training.rollout import TrainerRolloutMixin


logger = logging.getLogger(__name__)


def _processing_class_model_name(processing_class: Any) -> str:
    name = getattr(processing_class, "name_or_path", None)
    if isinstance(name, str):
        return name.lower()
    init_kwargs = getattr(processing_class, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        name = init_kwargs.get("name_or_path")
        if isinstance(name, str):
            return name.lower()
    return ""


@dataclass
class TrainingBatchPackage:
    training_batch: TrajectoryBatch
    rollout_metrics: dict[str, float]
    num_input_tokens_seen: int = 0
    sample_texts: dict[str, str] | None = None


@dataclass
class TrajectoryPackage(TrainingBatchPackage):
    @property
    def trajectory_batch(self) -> TrajectoryBatch:
        return self.training_batch

    @trajectory_batch.setter
    def trajectory_batch(self, value: TrajectoryBatch) -> None:
        self.training_batch = value


@dataclass
class PolicyOptimizationPackage(TrainingBatchPackage):
    @property
    def policy_batch(self) -> PolicyOptimizationBatch:
        return self.training_batch

    @policy_batch.setter
    def policy_batch(self, value: PolicyOptimizationBatch) -> None:
        self.training_batch = value


class OfflineRolloutBatchBuilder(TrainerRolloutMixin):
    """CPU-side rollout builder shared by the Ray coordinator."""

    def __init__(
        self,
        args,
        processing_class,
        *,
        generation_config: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.args = args
        self.processing_class = processing_class
        self.device = torch.device("cpu") if device is None else device
        self.pad_token_id = processing_class.pad_token_id
        self.eos_token_id = processing_class.eos_token_id
        self.base_model = (
            SimpleNamespace(generation_config=generation_config)
            if generation_config is not None
            else None
        )
        self.training_objective = build_training_objective(args.training_objective)
        self.model = None
        self.ref_model = None
        self.deepspeed_engine = None
        self.num_input_tokens_seen = 0
        self._think_close_token_ids: list[list[int]] | None = None

    @staticmethod
    def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float()

    @staticmethod
    def _distributed_sum_int(value: int) -> int:
        return int(value)

    def _decode_completion_text(self, completion_sequence: list[int]) -> str:
        # Keep special tokens so gemma's <|channel>thought...<channel|> separator
        # survives for strip_thinking_trace to find.
        return self.processing_class.decode(
            completion_sequence,
            skip_special_tokens=False,
        )

    def _split_completion_sequence_for_sample(
        self,
        completion_sequence: list[int],
    ) -> tuple[list[int], list[int]]:
        token_mask = build_response_token_mask(
            completion_sequence,
            thinking_enabled=self._student_thinking_enabled(),
            think_close_token_ids=self._resolve_think_close_token_ids(),
        )
        prefix_tokens = [token for token, is_response in zip(completion_sequence, token_mask) if not is_response]
        response_tokens = [token for token, is_response in zip(completion_sequence, token_mask) if is_response]
        return prefix_tokens, response_tokens

    def _completion_has_response_tokens(self, completion_sequence: list[int]) -> bool:
        token_mask = build_response_token_mask(
            completion_sequence,
            thinking_enabled=self._student_thinking_enabled(),
            think_close_token_ids=self._resolve_think_close_token_ids(),
        )
        return any(token_mask)

    def _build_sample_texts(
        self,
        *,
        prompt_text: str,
        completion_sequence: list[int],
    ) -> dict[str, str]:
        student_prefix_tokens, student_response_tokens = self._split_completion_sequence_for_sample(
            completion_sequence,
        )
        student_prefix_with_reasoning = prompt_text
        if student_prefix_tokens:
            student_prefix_with_reasoning += self._decode_completion_text(student_prefix_tokens)
        student_response = self._decode_completion_text(student_response_tokens) if student_response_tokens else ""
        return {
            "prompt": prompt_text,
            "student_prefix_with_reasoning": student_prefix_with_reasoning,
            "student_response": student_response,
        }

    def _resolve_think_close_token_ids(self) -> list[list[int]]:
        if self._think_close_token_ids is None:
            model_name = _processing_class_model_name(self.processing_class)
            if "qwen" in model_name:
                close_markers = ["</think>"]
            elif "mistral" in model_name or "ministral" in model_name:
                close_markers = ["[/THINK]"]
            elif "gemma" in model_name:
                close_markers = ["<channel|>"]
            else:
                close_markers = ["</think>", "[/THINK]", "<channel|>"]
            encoded = self.processing_class(
                close_markers,
                truncation=False,
                add_special_tokens=False,
            )
            token_id_batches = encoded["input_ids"]
            marker_token_ids = [
                list(token_ids)
                for token_ids in token_id_batches
                if token_ids
            ]
            if not marker_token_ids:
                raise ValueError(
                    "Could not tokenize any thinking close markers into token ids."
                )
            self._think_close_token_ids = marker_token_ids
        return [list(token_ids) for token_ids in self._think_close_token_ids]

    @staticmethod
    def build_sampling_kwargs(args, *, stop_token_ids: list[int]) -> dict[str, Any]:
        generation_kwargs = dict(args.generation_kwargs or {})
        generation_kwargs["detokenize"] = False
        return build_rollout_sampling_kwargs(
            num_generations=args.num_generations,
            temperature=args.train_vllm_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_completion_length,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_token_ids,
            generation_kwargs=generation_kwargs,
        )

    def prepare_trajectory_batch(
        self,
        prompt_batch: list[dict[str, Any]],
        *,
        completion_sequences: list[list[int]],
        completion_vllm_diagnostics: dict[str, torch.Tensor] | None = None,
        update_token_count: bool = True,
        prompt_texts: list[str] | None = None,
        prompt_tensors: dict[str, torch.Tensor] | None = None,
    ) -> TrajectoryPackage:
        """Build a generic prompt/completion batch for RL objectives."""
        prepare_start = time.monotonic()
        if prompt_texts is None:
            prompts = [example["prompt"] for example in prompt_batch]
            prompt_texts = [self._format_student_prompt(prompt) for prompt in prompts]
        if prompt_tensors is None:
            prompt_token_ids_batch = self._tokenize_prompt_text_sequences(prompt_texts)
            prompt_tensors = self._build_prompt_tensors_from_token_sequences(prompt_token_ids_batch)
        pad_start = time.monotonic()
        completion_ids, completion_mask = self._pad_completion_sequences(completion_sequences)
        pad_duration = time.monotonic() - pad_start
        build_start = time.monotonic()
        trajectory_batch, rollout_metrics = self._build_trajectory_batch(
            prompt_batch=prompt_batch,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            prompt_tensors=prompt_tensors,
            completion_vllm_diagnostics=completion_vllm_diagnostics,
            update_token_count=update_token_count,
            reduce_rollout_metrics=False,
        )
        build_duration = time.monotonic() - build_start
        trim_start = time.monotonic()
        trimmed_batch: TrajectoryBatch = dict(trajectory_batch)
        completion_texts_stale = False
        for ids_key, mask_key in (("prompt_ids", "prompt_mask"), ("completion_ids", "completion_mask")):
            ids = trimmed_batch.get(ids_key)
            mask = trimmed_batch.get(mask_key)
            if not torch.is_tensor(ids) or not torch.is_tensor(mask):
                continue
            padding_side = "left" if ids_key == "prompt_ids" else "right"
            max_length = self.args.max_train_completion_length if ids_key == "completion_ids" else None
            trimmed_ids = self._trim_sequence_padded_tensor(
                ids,
                mask,
                padding_side=padding_side,
                max_length=max_length,
            )
            trimmed_mask = self._trim_sequence_padded_tensor(
                mask,
                mask,
                padding_side=padding_side,
                max_length=max_length,
            )
            trimmed_batch[ids_key] = trimmed_ids
            trimmed_batch[mask_key] = trimmed_mask
            if ids_key == "completion_ids" and (trimmed_ids is not ids or trimmed_mask is not mask):
                completion_texts_stale = True
        completion_mask = trimmed_batch.get("completion_mask")
        if torch.is_tensor(completion_mask):
            completion_ids = trimmed_batch.get("completion_ids")
            if completion_texts_stale and torch.is_tensor(completion_ids):
                trimmed_batch["completion_texts"] = self._decode_response_texts(
                    completion_ids,
                    completion_mask,
                )
            self._trim_packed_vllm_diagnostics(trimmed_batch, completion_mask=completion_mask)
        trim_duration = time.monotonic() - trim_start
        sample_texts = None
        if prompt_batch:
            sample_prompt_text = (
                prompt_texts[0]
                if prompt_texts is not None
                else self._format_student_prompt(prompt_batch[0]["prompt"])
            )
            sample_texts = self._build_sample_texts(
                prompt_text=sample_prompt_text,
                completion_sequence=completion_sequences[0],
            )
        if getattr(self.args, "log_phase_progress", False):
            logger.info(
                "stage=rollout phase=prepare_trajectory_batch status=done prompts=%s completions=%s completion_tokens=%s "
                "pad_seconds=%.4f build_seconds=%.4f trim_seconds=%.4f total_seconds=%.4f",
                len(prompt_batch),
                len(completion_sequences),
                int(completion_mask.sum().item()) if torch.is_tensor(completion_mask) else 0,
                pad_duration,
                build_duration,
                trim_duration,
                time.monotonic() - prepare_start,
            )
        return TrajectoryPackage(
            training_batch=trimmed_batch,
            rollout_metrics=rollout_metrics,
            num_input_tokens_seen=self.num_input_tokens_seen,
            sample_texts=sample_texts,
        )
