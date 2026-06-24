from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONTROLLED_SAMPLING_KWARGS = frozenset(
    {
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "presence_penalty",
        "repetition_penalty",
        "stop_token_ids",
        "logprobs",
    }
)


def validate_extra_sampling_kwargs(generation_kwargs: Mapping[str, Any] | None) -> None:
    if generation_kwargs is None:
        return
    if not isinstance(generation_kwargs, Mapping):
        raise TypeError("generation_kwargs must be a mapping when provided.")

    collisions = sorted(CONTROLLED_SAMPLING_KWARGS.intersection(generation_kwargs.keys()))
    if not collisions:
        return

    rendered = ", ".join(collisions)
    raise ValueError(
        "generation_kwargs must not override rollout-managed sampling keys: "
        f"{rendered}."
    )


def merge_sampling_kwargs(
    base_kwargs: dict[str, Any],
    generation_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validate_extra_sampling_kwargs(generation_kwargs)
    merged = dict(base_kwargs)
    if generation_kwargs is not None:
        merged.update(generation_kwargs)
    return {key: value for key, value in merged.items() if value is not None}


def build_rollout_sampling_kwargs(
    *,
    num_generations: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    max_tokens: int,
    presence_penalty: float | None,
    repetition_penalty: float,
    stop_token_ids: Sequence[int] | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sampling_kwargs: dict[str, Any] = {
        "n": num_generations,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
    }
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = list(stop_token_ids)
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if min_p is not None:
        sampling_kwargs["min_p"] = min_p
    if presence_penalty is not None:
        sampling_kwargs["presence_penalty"] = presence_penalty
    return merge_sampling_kwargs(sampling_kwargs, generation_kwargs)
