"""Shared batch-layout helpers for rollout and training code paths.

The current trainer still moves plain dictionaries of tensors across Ray and
DeepSpeed boundaries. This module makes the alignment rules explicit so new
objectives can add reward-model or judge-derived tensors without rewriting the
sharding and micro-batching code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .batch_types import TrajectoryBatch


@dataclass(frozen=True)
class PackedSequenceFields:
    """One offsets tensor plus any packed token-aligned tensors it indexes."""

    offsets_key: str
    value_keys: tuple[str, ...] = ()

    @property
    def tensor_keys(self) -> tuple[str, ...]:
        return (self.offsets_key, *self.value_keys)


@dataclass(frozen=True)
class BatchFieldLayout:
    """Explicitly describes how a tensor batch is aligned.

    `prompt_tensor_keys` are indexed by prompt rows.
    `sequence_tensor_keys` are indexed by sampled completion rows.
    `packed_sequence_fields` use an offsets tensor plus flattened token rows.
    `host_tensor_keys` stay on CPU when the trainer moves a batch to the model
    device. This is useful for diagnostics that are only gathered lazily.
    """

    prompt_tensor_keys: tuple[str, ...]
    sequence_tensor_keys: tuple[str, ...]
    packed_sequence_fields: tuple[PackedSequenceFields, ...] = ()
    host_tensor_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tensor_keys = list(self.prompt_tensor_keys) + list(self.sequence_tensor_keys)
        for field_group in self.packed_sequence_fields:
            tensor_keys.extend(field_group.tensor_keys)

        seen: set[str] = set()
        duplicates: set[str] = set()
        for key in tensor_keys:
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            duplicate_text = ", ".join(sorted(duplicates))
            raise ValueError(f"BatchFieldLayout contains duplicate tensor keys: {duplicate_text}")

        unknown_host_keys = set(self.host_tensor_keys) - seen
        if unknown_host_keys:
            unknown_text = ", ".join(sorted(unknown_host_keys))
            raise ValueError(
                "host_tensor_keys must be a subset of the registered tensor keys: "
                f"{unknown_text}"
            )

    @property
    def packed_tensor_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for field_group in self.packed_sequence_fields:
            keys.extend(field_group.tensor_keys)
        return tuple(keys)

    def with_extensions(
        self,
        *,
        prompt_tensor_keys: tuple[str, ...] = (),
        sequence_tensor_keys: tuple[str, ...] = (),
        packed_sequence_fields: tuple[PackedSequenceFields, ...] = (),
        host_tensor_keys: tuple[str, ...] = (),
    ) -> BatchFieldLayout:
        """Return a new layout with extra registered fields.

        RL-style extensions should generally register advantage, score, and
        old-logprob tensors as `sequence_tensor_keys` so they shard with
        generated completions automatically.
        """
        return BatchFieldLayout(
            prompt_tensor_keys=_merge_ordered_unique(
                self.prompt_tensor_keys,
                prompt_tensor_keys,
            ),
            sequence_tensor_keys=_merge_ordered_unique(
                self.sequence_tensor_keys,
                sequence_tensor_keys,
            ),
            packed_sequence_fields=self.packed_sequence_fields + packed_sequence_fields,
            host_tensor_keys=_merge_ordered_unique(
                self.host_tensor_keys,
                host_tensor_keys,
            ),
        )


def _merge_ordered_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for group in groups for key in group))


ROLLOUT_BATCH_LAYOUT = BatchFieldLayout(
    prompt_tensor_keys=(
        "prompt_ids",
        "prompt_mask",
    ),
    sequence_tensor_keys=(
        "completion_ids",
        "completion_mask",
    ),
    packed_sequence_fields=(
        PackedSequenceFields(
            offsets_key="vllm_sequence_offsets",
            value_keys=("vllm_logprobs_flat",),
        ),
    ),
    host_tensor_keys=(
        "vllm_sequence_offsets",
        "vllm_logprobs_flat",
    ),
)


POLICY_OPTIMIZATION_BATCH_LAYOUT = ROLLOUT_BATCH_LAYOUT.with_extensions(
    sequence_tensor_keys=(
        "action_mask",
        "old_log_probs",
        "advantages",
        "sequence_scores",
        "valid_sequence_mask",
    ),
)


def materialize_packed_logprobs(
    flat_values: torch.Tensor,
    sequence_offsets: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Unpack a flat packed-sequence tensor into a right-padded [B, T] tensor.

    This is the single implementation used by both the trainer loss path and
    the coordinator-side policy pipeline to convert packed vLLM logprobs into
    the padded layout that the rest of the training stack expects.
    """
    if flat_values.ndim != 1:
        raise ValueError("flat_values must have shape [total_completion_tokens].")
    if sequence_offsets.ndim != 1 or sequence_offsets.numel() != completion_mask.size(0) + 1:
        raise ValueError(
            "sequence_offsets must contain one offset per completion sequence plus a trailing end offset."
        )
    completion_lengths = completion_mask.sum(dim=1, dtype=sequence_offsets.dtype).to(sequence_offsets.device)
    if not torch.equal(sequence_offsets[1:] - sequence_offsets[:-1], completion_lengths):
        raise ValueError("Packed values do not align with completion_mask.")

    offsets_list = sequence_offsets.tolist()
    if flat_values.size(0) != offsets_list[-1]:
        raise ValueError("flat_values does not match sequence_offsets.")

    padded = flat_values.new_zeros(completion_mask.shape)
    for i in range(completion_mask.size(0)):
        start = offsets_list[i]
        end = offsets_list[i + 1]
        if end > start:
            padded[i, : end - start] = flat_values[start:end]
    return padded


def move_training_batch_to_device(
    batch: TrajectoryBatch,
    *,
    device: torch.device,
    layout: BatchFieldLayout,
) -> TrajectoryBatch:
    """Move non-host tensors onto `device` while keeping packed diagnostics on CPU."""
    host_tensor_keys = set(layout.host_tensor_keys)
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value) and key not in host_tensor_keys
        else value
        for key, value in batch.items()
    }


def shard_training_batch_for_workers(
    batch: TrajectoryBatch,
    *,
    layout: BatchFieldLayout,
    world_size: int,
    prompts_per_worker: int,
    num_generations: int,
) -> list[TrajectoryBatch]:
    """Shard a rollout batch into per-worker prompt/sequence slices."""
    if world_size <= 0:
        raise ValueError("world_size must be > 0.")
    num_prompts = int(batch["num_prompts"])
    expected_prompts = prompts_per_worker * world_size
    if num_prompts != expected_prompts:
        raise ValueError(
            f"Expected {expected_prompts} prompts for {world_size} workers, but got {num_prompts}."
        )

    num_sequences = _resolve_num_sequences(batch, layout=layout, num_prompts=num_prompts)
    expected_sequences = num_prompts * num_generations
    if num_sequences != expected_sequences:
        raise ValueError(
            "Batch sequence count does not match num_prompts * num_generations "
            f"({num_sequences} != {expected_sequences})."
        )

    shards: list[TrajectoryBatch] = []
    for worker_index in range(world_size):
        prompt_start = worker_index * prompts_per_worker
        prompt_end = prompt_start + prompts_per_worker
        seq_start = prompt_start * num_generations
        seq_end = prompt_end * num_generations

        shard: TrajectoryBatch = {"num_prompts": prompts_per_worker}
        _slice_packed_sequence_fields_into_batch(
            shard,
            source_batch=batch,
            layout=layout,
            sequence_start=seq_start,
            sequence_end=seq_end,
        )
        _slice_registered_tensor_fields_into_batch(
            shard,
            source_batch=batch,
            layout=layout,
            num_prompts=num_prompts,
            num_sequences=num_sequences,
            prompt_start=prompt_start,
            prompt_end=prompt_end,
            sequence_start=seq_start,
            sequence_end=seq_end,
        )
        shards.append(shard)

    return shards


def split_training_batch_into_micro_batches(
    batch: TrajectoryBatch,
    *,
    layout: BatchFieldLayout,
    num_generations: int,
    prompts_per_micro_batch: int,
    max_sequences_per_micro_batch: int,
) -> list[TrajectoryBatch]:
    """Split one worker batch into prompt-aligned micro-batches.

    Prompt tensors are repeated to match the sampled sequence chunk they
    supervise.
    """
    if prompts_per_micro_batch <= 0:
        raise ValueError("prompts_per_micro_batch must be > 0.")
    if max_sequences_per_micro_batch <= 0:
        raise ValueError("max_sequences_per_micro_batch must be > 0.")

    num_prompts = int(batch["num_prompts"])
    num_sequences = _resolve_num_sequences(batch, layout=layout, num_prompts=num_prompts)
    expected_sequences = num_prompts * num_generations
    if num_sequences != expected_sequences:
        raise ValueError(
            "Batch sequence count does not match num_prompts * num_generations "
            f"({num_sequences} != {expected_sequences})."
        )

    micro_batches: list[TrajectoryBatch] = []
    for prompt_start in range(0, num_prompts, prompts_per_micro_batch):
        prompt_end = min(prompt_start + prompts_per_micro_batch, num_prompts)
        sequence_start = prompt_start * num_generations
        sequence_end = prompt_end * num_generations

        for chunk_start in range(sequence_start, sequence_end, max_sequences_per_micro_batch):
            chunk_end = min(chunk_start + max_sequences_per_micro_batch, sequence_end)
            chunk_prompt_indices = torch.div(
                torch.arange(chunk_start, chunk_end, dtype=torch.long),
                num_generations,
                rounding_mode="floor",
            )
            micro_batch: TrajectoryBatch = {
                "num_prompts": int(chunk_prompt_indices.unique().numel()),
            }
            _slice_packed_sequence_fields_into_batch(
                micro_batch,
                source_batch=batch,
                layout=layout,
                sequence_start=chunk_start,
                sequence_end=chunk_end,
            )
            _slice_registered_tensor_fields_into_batch(
                micro_batch,
                source_batch=batch,
                layout=layout,
                num_prompts=num_prompts,
                num_sequences=num_sequences,
                prompt_start=prompt_start,
                prompt_end=prompt_end,
                sequence_start=chunk_start,
                sequence_end=chunk_end,
                chunk_prompt_indices=chunk_prompt_indices,
            )
            micro_batches.append(micro_batch)

    return micro_batches


def _resolve_num_sequences(
    batch: TrajectoryBatch,
    *,
    layout: BatchFieldLayout,
    num_prompts: int,
) -> int:
    for key in layout.sequence_tensor_keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            return int(value.size(0))
    for key in layout.prompt_tensor_keys:
        value = batch.get(key)
        if torch.is_tensor(value) and int(value.size(0)) != num_prompts:
            return int(value.size(0))
    raise ValueError(
        "Could not resolve the sequence count from the batch. "
        "Expected at least one registered sequence tensor."
    )


def _slice_registered_tensor_fields_into_batch(
    destination_batch: TrajectoryBatch,
    *,
    source_batch: TrajectoryBatch,
    layout: BatchFieldLayout,
    num_prompts: int,
    num_sequences: int,
    prompt_start: int,
    prompt_end: int,
    sequence_start: int,
    sequence_end: int,
    chunk_prompt_indices: torch.Tensor | None = None,
) -> None:
    packed_tensor_keys = set(layout.packed_tensor_keys)
    prompt_tensor_keys = set(layout.prompt_tensor_keys)
    sequence_tensor_keys = set(layout.sequence_tensor_keys)

    for key, value in source_batch.items():
        if key == "num_prompts" or key in packed_tensor_keys or not torch.is_tensor(value):
            continue

        if key in prompt_tensor_keys:
            destination_batch[key] = _slice_prompt_tensor(
                value,
                key=key,
                num_prompts=num_prompts,
                num_sequences=num_sequences,
                prompt_start=prompt_start,
                prompt_end=prompt_end,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                chunk_prompt_indices=chunk_prompt_indices,
            )
            continue

        if key in sequence_tensor_keys:
            if value.size(0) != num_sequences:
                raise ValueError(
                    f"Sequence tensor {key} has {value.size(0)} rows, expected {num_sequences}."
                )
            destination_batch[key] = value[sequence_start:sequence_end]
            continue

        raise ValueError(
            f"Tensor batch key {key!r} is not registered in BatchFieldLayout. "
            "Extend the layout before sharding or micro-batching new tensors."
        )


def _slice_prompt_tensor(
    value: torch.Tensor,
    *,
    key: str,
    num_prompts: int,
    num_sequences: int,
    prompt_start: int,
    prompt_end: int,
    sequence_start: int,
    sequence_end: int,
    chunk_prompt_indices: torch.Tensor | None,
) -> torch.Tensor:
    if value.size(0) == num_prompts:
        if chunk_prompt_indices is None:
            return value[prompt_start:prompt_end]
        return value.index_select(0, chunk_prompt_indices.to(device=value.device))
    if value.size(0) == num_sequences:
        return value[sequence_start:sequence_end]
    raise ValueError(
        f"Prompt tensor {key} has {value.size(0)} rows, expected {num_prompts} prompts or {num_sequences} sequences."
    )


def _slice_packed_sequence_fields_into_batch(
    destination_batch: TrajectoryBatch,
    *,
    source_batch: TrajectoryBatch,
    layout: BatchFieldLayout,
    sequence_start: int,
    sequence_end: int,
) -> None:
    for field_group in layout.packed_sequence_fields:
        sequence_offsets = source_batch.get(field_group.offsets_key)
        if not torch.is_tensor(sequence_offsets):
            continue
        token_start = int(sequence_offsets[sequence_start].item())
        token_end = int(sequence_offsets[sequence_end].item())
        destination_batch[field_group.offsets_key] = (
            sequence_offsets[sequence_start : sequence_end + 1] - token_start
        )
        for value_key in field_group.value_keys:
            packed_value = source_batch.get(value_key)
            if not torch.is_tensor(packed_value):
                continue
            destination_batch[value_key] = packed_value[token_start:token_end]
