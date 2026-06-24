"""Shared tokenization helpers extracted from TrainerRolloutMixin.

Both the trainer mixin and the offline rollout builder delegate to
``TokenizationHelper`` for pure tokenization operations, avoiding fragile
mixin inheritance for non-training code paths.
"""

from __future__ import annotations

from typing import Any

import torch


class TokenizationHelper:
    """Stateless-ish tokenization helper used by both training and offline rollout paths."""

    def __init__(
        self,
        processing_class: Any,
        *,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.processing_class = processing_class
        self.pad_token_id = pad_token_id if pad_token_id is not None else getattr(processing_class, "pad_token_id", None)
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(processing_class, "eos_token_id", None)
        self.device = device if device is not None else torch.device("cpu")

    def tokenize_prompt_text_sequences(
        self,
        prompt_texts: list[str],
    ) -> list[list[int]]:
        encoded = self.processing_class(
            prompt_texts,
            truncation=False,
            add_special_tokens=False,
        )
        return [list(token_ids) for token_ids in encoded["input_ids"]]

    def pad_token_sequences(
        self,
        token_sequences: list[list[int]],
        *,
        padding_side: str,
    ) -> dict[str, torch.Tensor]:
        if not token_sequences:
            raise ValueError("token_sequences must not be empty.")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")

        max_length = max(1, max(len(sequence) for sequence in token_sequences))
        pad_token_id = self.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.eos_token_id or 0

        input_ids = torch.full(
            (len(token_sequences), max_length),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(token_sequences), max_length),
            dtype=torch.bool,
        )

        for index, sequence in enumerate(token_sequences):
            if not sequence:
                continue
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            sequence_length = sequence_tensor.numel()
            if padding_side == "left":
                input_ids[index, -sequence_length:] = sequence_tensor
                attention_mask[index, -sequence_length:] = True
            else:
                input_ids[index, :sequence_length] = sequence_tensor
                attention_mask[index, :sequence_length] = True

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def pad_completion_sequences(
        self,
        completion_sequences: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not completion_sequences:
            raise ValueError("Generation returned no completions.")
        padded = self.pad_token_sequences(completion_sequences, padding_side="right")
        return padded["input_ids"], padded["attention_mask"]

    def build_prompt_tensors(
        self,
        token_sequences: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        return self.pad_token_sequences(token_sequences, padding_side="left")
