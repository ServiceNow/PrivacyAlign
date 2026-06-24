"""Sequence-level scoring interfaces for future RL objectives."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from training.batch_types import TrajectoryBatch
from judges.pairwise import (
    build_privalign_rl_judge_prompt_token_ids_batch,
    combine_pairwise_judge_outputs,
    render_privalign_rl_judge_prompt_texts,
    _truncate_judge_response,
)
from utils.text_utils import decode_response_tokens, strip_thinking_trace


# Strict `Score: <int>` parser shared by PairwiseMarginScorer (gen-RM training)
# and TrainedGenRMScorer (policy RL with the trained gen-RM). Matches the eval
# pipeline's pairwise judge format: a single integer Score on the [-2, +2]
# preference scale (negative = response 1 better, positive = response 2 better).
# We take the LAST occurrence so a truncated <think> that emitted an early
# guess still yields the final answer.
_SINGLE_SCORE_RE = re.compile(
    r"\bscore\b\s*[:=]\s*(?P<value>[-+]?\d+)\b",
    re.IGNORECASE,
)


def _parse_signed_score(
    raw: str,
    *,
    valid_range: tuple[int, int] = (-2, 2),
) -> int | None:
    """Return the parsed signed score in ``valid_range`` or None if missing/invalid."""
    if not isinstance(raw, str) or not raw:
        return None
    cleaned = strip_thinking_trace(raw)
    for source in (cleaned, raw):
        matches = _SINGLE_SCORE_RE.findall(source)
        if not matches:
            continue
        try:
            value = int(matches[-1])
        except ValueError:
            continue
        lo, hi = valid_range
        if value < lo or value > hi:
            continue
        return value
    return None


_PAIRWISE_LEAK_RE = re.compile(
    r"\bresponse\s*(?P<slot>[12])\s*leaks?\b\s*[:=]\s*(?P<value>yes|no|true|false|1|0)\b",
    re.IGNORECASE,
)
_BOOL_TRUE_TOKENS = frozenset({"yes", "true", "1"})


def _parse_pairwise_leaks(raw: str) -> tuple[bool | None, bool | None]:
    """Return parsed (response1_leaks, response2_leaks) booleans."""
    if not isinstance(raw, str) or not raw:
        return None, None
    parsed: dict[int, bool] = {}
    cleaned = strip_thinking_trace(raw)
    for source in (cleaned, raw):
        for match in _PAIRWISE_LEAK_RE.finditer(source):
            slot = int(match.group("slot"))
            value = match.group("value").strip().lower()
            parsed[slot] = value in _BOOL_TRUE_TOKENS
        if 1 in parsed and 2 in parsed:
            break
    return parsed.get(1), parsed.get(2)


@dataclass
class SequenceScoreOutputs:
    """One scalar score per sampled completion sequence.

    `metadata` is intentionally open-ended so scalar reward models can attach
    calibration details while LLM judges can attach rationales or paired scores.
    """

    scores: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, num_sequences: int) -> None:
        if not torch.is_tensor(self.scores):
            raise TypeError("scores must be a torch.Tensor.")
        if self.scores.ndim != 1:
            raise ValueError("scores must have shape [num_sequences].")
        if self.scores.size(0) != num_sequences:
            raise ValueError(
                "scores length must match the number of completion sequences "
                f"({self.scores.size(0)} != {num_sequences})."
            )


@dataclass(frozen=True)
class TrajectoryScoreContext:
    """Prompt-aligned metadata needed by richer scorers."""

    examples: list[dict[str, Any]] | None = None
    prompt_texts: list[str] | None = None


class TrajectoryScorer(Protocol):
    """Interface shared by scalar reward models and LLM judges."""

    name: str
    kind: str

    def score(
        self,
        batch: TrajectoryBatch,
        *,
        context: TrajectoryScoreContext | None = None,
    ) -> SequenceScoreOutputs:
        """Return one scalar score per generated completion sequence."""


class PairwiseRLJudgeScorerBase:
    """Shared pairwise-judge aggregation over sampled completions."""

    name = "pairwise_rl_judge"
    kind = "llm_judge"

    def __init__(
        self,
        *,
        rollout_builder: Any,
        judge_rollout_builder: Any | None = None,
        judge_template: str,
        judge_text_generator: Callable[..., list[str]],
        invalid_score: float = 0.0,
        judge_max_new_tokens: int | None = None,
        penalty_max_len: int | None = None,
        penalty_per_word: float = 0.0,
        penalty_max_value: float | None = 2.0,
        penalty_shape: str = "linear",
        scoring_mode: str = "anchor",
        judge_dual_order: bool = True,
        retain_judge_outputs: bool = True,
        undershort_penalty_max: float = 0.0,
        undershort_floor_ratio: float = 0.5,
    ) -> None:
        self.rollout_builder = rollout_builder
        self.judge_rollout_builder = rollout_builder if judge_rollout_builder is None else judge_rollout_builder
        self.judge_template = judge_template
        self.judge_text_generator = judge_text_generator
        self.invalid_score = float(invalid_score)
        # When None, fall back to the policy's max_completion_length (original behavior).
        self.judge_max_new_tokens = judge_max_new_tokens
        # Length penalty (None = disabled, default).
        self.penalty_max_len = penalty_max_len
        self.penalty_per_word = float(penalty_per_word)
        self.penalty_max_value = None if penalty_max_value is None else float(penalty_max_value)
        if penalty_shape not in {"linear"}:
            raise ValueError(f"penalty_shape must be 'linear'; got {penalty_shape!r}.")
        self.penalty_shape = penalty_shape
        self.undershort_penalty_max = float(undershort_penalty_max)
        self.undershort_floor_ratio = float(undershort_floor_ratio)
        # "anchor" (default, back-compat): each rollout judged vs cached base
        # anchor. "peer": pairwise-within-group; judge_dual_order controls
        # whether each comparison is double-swapped.
        if scoring_mode not in {"anchor", "peer"}:
            raise ValueError(f"scoring_mode must be 'anchor' or 'peer'; got {scoring_mode!r}.")
        self.scoring_mode = scoring_mode
        self.judge_dual_order = bool(judge_dual_order)
        self.retain_judge_outputs = bool(retain_judge_outputs)

    def score(
        self,
        batch: TrajectoryBatch,
        *,
        context: TrajectoryScoreContext | None = None,
    ) -> SequenceScoreOutputs:
        if context is None or context.examples is None:
            raise ValueError(f"{self.__class__.__name__} requires examples in TrajectoryScoreContext.")

        completion_ids = batch.get("completion_ids")
        completion_mask = batch.get("completion_mask")
        if not torch.is_tensor(completion_ids) or not torch.is_tensor(completion_mask):
            raise KeyError(f"{self.__class__.__name__} requires completion_ids and completion_mask.")

        num_prompts = batch.get("num_prompts")
        if not isinstance(num_prompts, int) or num_prompts <= 0:
            raise KeyError(f"{self.__class__.__name__} requires a positive integer num_prompts.")

        num_sequences = int(completion_ids.size(0))
        repeated_examples = _expand_prompt_aligned_items(
            context.examples,
            num_prompts=num_prompts,
            num_sequences=num_sequences,
        )
        completion_texts = _decode_completion_texts(
            self.rollout_builder, completion_ids, completion_mask, batch
        )

        if self.scoring_mode == "peer":
            return self._score_peer(
                completion_texts=completion_texts,
                num_prompts=num_prompts,
                num_sequences=num_sequences,
                context_examples=context.examples,
            )

        reference_response_batch = _resolve_reference_response_batch(repeated_examples)
        judge_prompt_texts = self._try_build_judge_prompt_text_batch(
            examples=repeated_examples,
            judge_template=self.judge_template,
            rollout_builder=self.judge_rollout_builder,
            response1_batch=reference_response_batch,
            response2_batch=completion_texts,
        )
        judge_prompt_token_ids_batch = self._build_judge_prompt_token_ids_batch(
            examples=repeated_examples,
            judge_template=self.judge_template,
            rollout_builder=self.judge_rollout_builder,
            response1_batch=reference_response_batch,
            response2_batch=completion_texts,
        )
        judge_prompt_texts_swapped: list[str] = []
        judge_prompt_token_ids_batch_swapped: list[list[int]] = []
        if self.judge_dual_order:
            judge_prompt_texts_swapped = self._try_build_judge_prompt_text_batch(
                examples=repeated_examples,
                judge_template=self.judge_template,
                rollout_builder=self.judge_rollout_builder,
                response1_batch=completion_texts,
                response2_batch=reference_response_batch,
            )
            judge_prompt_token_ids_batch_swapped = self._build_judge_prompt_token_ids_batch(
                examples=repeated_examples,
                judge_template=self.judge_template,
                rollout_builder=self.judge_rollout_builder,
                response1_batch=completion_texts,
                response2_batch=reference_response_batch,
            )
        judge_max_new_tokens = (
            int(self.judge_max_new_tokens)
            if self.judge_max_new_tokens is not None
            else int(self.rollout_builder.args.max_completion_length)
        )
        all_judge_outputs = self.judge_text_generator(
            judge_prompt_token_ids_batch + judge_prompt_token_ids_batch_swapped,
            max_new_tokens=judge_max_new_tokens,
        )
        expected_output_count = num_sequences * (2 if self.judge_dual_order else 1)
        if len(all_judge_outputs) != expected_output_count:
            raise ValueError(
                "Judge output count must match the configured pairwise orderings "
                f"({len(all_judge_outputs)} != {expected_output_count})."
            )
        judge_outputs = all_judge_outputs[:num_sequences]
        judge_outputs_swapped = (
            all_judge_outputs[num_sequences:]
            if self.judge_dual_order
            else [""] * num_sequences
        )

        parsed_scores: list[float] = []
        invalid_sequence_mask: list[bool] = []
        invalid_count = 0
        length_penalties: list[float] = []
        retain = self.retain_judge_outputs
        parsed_reference_first_reasonings: list[str | None] = []
        parsed_candidate_first_reasonings: list[str | None] = []
        parsed_reference_first_scores: list[int | None] = []
        parsed_candidate_first_scores: list[int | None] = []
        for completion_text, reference_first_output, candidate_first_output in zip(
            completion_texts,
            judge_outputs,
            judge_outputs_swapped,
        ):
            parsed_score, (reference_first_score, candidate_first_score), (
                reference_first_reasoning,
                candidate_first_reasoning,
            ) = combine_pairwise_judge_outputs(
                reference_first_output,
                candidate_first_output,
                require_both=self.judge_dual_order,
                processing_class=getattr(self.judge_rollout_builder, "processing_class", None),
            )
            if retain:
                parsed_reference_first_scores.append(reference_first_score)
                parsed_candidate_first_scores.append(candidate_first_score)
                parsed_reference_first_reasonings.append(reference_first_reasoning)
                parsed_candidate_first_reasonings.append(candidate_first_reasoning)
            length_penalty = 0.0
            if parsed_score is not None:
                length_penalty = _linear_length_penalty(
                    completion_text,
                    max_len=self.penalty_max_len,
                    per_word=self.penalty_per_word,
                    max_value=self.penalty_max_value,
                    shape=self.penalty_shape,
                )
            length_penalties.append(length_penalty)
            if parsed_score is None:
                invalid_count += 1
                invalid_sequence_mask.append(True)
                parsed_scores.append(self.invalid_score)
            else:
                invalid_sequence_mask.append(False)
                parsed_scores.append(float(parsed_score) - length_penalty)

        metadata: dict[str, Any] = {
            "score_name": self.name,
            "judge_dual_order": float(self.judge_dual_order),
            "invalid_sequence_count": float(invalid_count),
            "invalid_sequence_rate": float(invalid_count) / max(1, num_sequences),
            "invalid_sequence_mask": torch.tensor(invalid_sequence_mask, dtype=torch.bool),
            "length_penalties": torch.tensor(length_penalties, dtype=torch.float32),
        }
        if judge_prompt_texts:
            metadata["sample_policy_judge_prompt"] = judge_prompt_texts[0]
        if judge_prompt_texts_swapped:
            metadata["sample_policy_judge_prompt_swapped"] = judge_prompt_texts_swapped[0]
        if retain:
            metadata.update(
                {
                    "judge_outputs_reference_first": judge_outputs,
                    "judge_outputs_candidate_first": judge_outputs_swapped,
                    "judge_reference_first_reasonings": parsed_reference_first_reasonings,
                    "judge_candidate_first_reasonings": parsed_candidate_first_reasonings,
                    "judge_reference_first_scores": parsed_reference_first_scores,
                    "judge_candidate_first_scores": parsed_candidate_first_scores,
                }
            )
        return SequenceScoreOutputs(
            scores=torch.tensor(parsed_scores, dtype=torch.float32),
            metadata=metadata,
        )

    def _score_peer(
        self,
        *,
        completion_texts: list[str],
        num_prompts: int,
        num_sequences: int,
        context_examples: list[dict[str, Any]],
    ) -> SequenceScoreOutputs:
        """Pairwise-within-group LLM-judge scoring.

        For each prompt's group of G rollouts, builds both response orderings
        for every unordered pair. Per-rollout reward = mean signed margin
        across its (G-1) peer comparisons. When judge_dual_order=True, each
        pair's score is averaged across both response orderings.
        """
        if num_sequences % num_prompts != 0:
            raise ValueError(
                f"num_sequences ({num_sequences}) must be divisible by num_prompts ({num_prompts})."
            )
        group_size = num_sequences // num_prompts
        if group_size < 2:
            raise ValueError(
                f"peer scoring requires num_generations >= 2; got group_size={group_size}."
            )

        pair_examples: list[dict[str, Any]] = []
        response1_batch: list[str] = []
        response2_batch: list[str] = []
        response1_batch_swapped: list[str] = []
        response2_batch_swapped: list[str] = []
        pair_indices: list[tuple[int, int]] = []  # (slot1_seq, slot2_seq) global indices
        for p in range(num_prompts):
            base_idx = p * group_size
            prompt_example = context_examples[p]
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    pair_examples.append(prompt_example)
                    response1_batch.append(completion_texts[base_idx + i])
                    response2_batch.append(completion_texts[base_idx + j])
                    if self.judge_dual_order:
                        response1_batch_swapped.append(completion_texts[base_idx + j])
                        response2_batch_swapped.append(completion_texts[base_idx + i])
                    pair_indices.append((base_idx + i, base_idx + j))
        num_pairs = len(pair_examples)

        judge_prompt_texts = self._try_build_judge_prompt_text_batch(
            examples=pair_examples,
            judge_template=self.judge_template,
            rollout_builder=self.judge_rollout_builder,
            response1_batch=response1_batch,
            response2_batch=response2_batch,
        )
        judge_prompt_token_ids_batch = self._build_judge_prompt_token_ids_batch(
            examples=pair_examples,
            judge_template=self.judge_template,
            rollout_builder=self.judge_rollout_builder,
            response1_batch=response1_batch,
            response2_batch=response2_batch,
        )
        judge_prompt_texts_swapped: list[str] = []
        judge_prompt_token_ids_batch_swapped: list[list[int]] = []
        if self.judge_dual_order:
            judge_prompt_texts_swapped = self._try_build_judge_prompt_text_batch(
                examples=pair_examples,
                judge_template=self.judge_template,
                rollout_builder=self.judge_rollout_builder,
                response1_batch=response1_batch_swapped,
                response2_batch=response2_batch_swapped,
            )
            judge_prompt_token_ids_batch_swapped = self._build_judge_prompt_token_ids_batch(
                examples=pair_examples,
                judge_template=self.judge_template,
                rollout_builder=self.judge_rollout_builder,
                response1_batch=response1_batch_swapped,
                response2_batch=response2_batch_swapped,
            )
        judge_max_new_tokens = (
            int(self.judge_max_new_tokens)
            if self.judge_max_new_tokens is not None
            else int(self.rollout_builder.args.max_completion_length)
        )
        all_judge_outputs = self.judge_text_generator(
            judge_prompt_token_ids_batch + judge_prompt_token_ids_batch_swapped,
            max_new_tokens=judge_max_new_tokens,
        )
        expected_output_count = num_pairs * (2 if self.judge_dual_order else 1)
        if len(all_judge_outputs) != expected_output_count:
            raise ValueError(
                "Judge output count must match the configured peer pair orderings "
                f"({len(all_judge_outputs)} != {expected_output_count})."
            )
        judge_outputs = all_judge_outputs[:num_pairs]
        judge_outputs_swapped = (
            all_judge_outputs[num_pairs:]
            if self.judge_dual_order
            else [""] * num_pairs
        )

        processing_class = getattr(self.judge_rollout_builder, "processing_class", None)
        per_rollout_sum = [0.0] * num_sequences
        per_rollout_count = [0] * num_sequences
        pair_scores: list[float | None] = [None] * num_pairs
        pair_reference_first_scores: list[int | None] = [None] * num_pairs
        pair_candidate_first_scores: list[int | None] = [None] * num_pairs
        pair_invalid_count = 0
        for idx, ((slot1_seq, slot2_seq), output, swapped_output) in enumerate(
            zip(pair_indices, judge_outputs, judge_outputs_swapped)
        ):
            parsed_score, (reference_first_score, candidate_first_score), _reasonings = (
                combine_pairwise_judge_outputs(
                    output,
                    swapped_output,
                    require_both=self.judge_dual_order,
                    processing_class=processing_class,
                )
            )
            pair_reference_first_scores[idx] = reference_first_score
            pair_candidate_first_scores[idx] = candidate_first_score
            pair_scores[idx] = parsed_score
            if parsed_score is None:
                pair_invalid_count += 1
                continue
            # Positive combined margin = original slot 2 preferred.
            per_rollout_sum[slot2_seq] += float(parsed_score)
            per_rollout_sum[slot1_seq] -= float(parsed_score)
            per_rollout_count[slot2_seq] += 1
            per_rollout_count[slot1_seq] += 1

        scores: list[float] = []
        invalid_sequence_mask: list[bool] = []
        length_penalties: list[float] = []
        per_rollout_means: list[float] = []
        rollouts_no_valid_peer = 0
        for i in range(num_sequences):
            cnt = per_rollout_count[i]
            if cnt == 0:
                rollouts_no_valid_peer += 1
                invalid_sequence_mask.append(True)
                mean_margin = self.invalid_score
            else:
                invalid_sequence_mask.append(False)
                mean_margin = per_rollout_sum[i] / cnt
            per_rollout_means.append(mean_margin)
            length_penalty = 0.0
            if self.penalty_max_len is not None and self.penalty_per_word > 0.0:
                length_penalty = _linear_length_penalty(
                    completion_texts[i],
                    max_len=self.penalty_max_len,
                    per_word=self.penalty_per_word,
                    max_value=self.penalty_max_value,
                    shape=self.penalty_shape,
                )
            length_penalties.append(length_penalty)
            scores.append(mean_margin - length_penalty)

        undershort_penalties, undershort_metadata = _compute_privalign_group_undershort_penalties(
            completion_texts=completion_texts,
            num_prompts=num_prompts,
            examples=context_examples,
            penalty_max=self.undershort_penalty_max,
            floor_ratio=self.undershort_floor_ratio,
        )
        if undershort_penalties:
            scores = [
                score - undershort_penalty
                for score, undershort_penalty in zip(scores, undershort_penalties)
            ]

        metadata: dict[str, Any] = {
            "score_name": self.name,
            "scoring_mode": "peer",
            "judge_dual_order": float(self.judge_dual_order),
            "pair_scores": pair_scores,
            "pair_reference_first_scores": pair_reference_first_scores,
            "pair_candidate_first_scores": pair_candidate_first_scores,
            "pair_indices": pair_indices,
            "per_rollout_means": per_rollout_means,
            "num_pairs": float(num_pairs),
            "pair_invalid_count": float(pair_invalid_count),
            "pair_invalid_rate": float(pair_invalid_count) / max(1, num_pairs),
            "invalid_sequence_count": float(rollouts_no_valid_peer),
            "invalid_sequence_rate": float(rollouts_no_valid_peer) / max(1, num_sequences),
            "invalid_sequence_mask": torch.tensor(invalid_sequence_mask, dtype=torch.bool),
            "valid_pair_counts": torch.tensor(per_rollout_count, dtype=torch.float32),
            "length_penalties": torch.tensor(length_penalties, dtype=torch.float32),
        }
        if undershort_metadata:
            metadata.update(undershort_metadata)
        if judge_prompt_texts:
            metadata["sample_policy_judge_prompt"] = judge_prompt_texts[0]
        if judge_prompt_texts_swapped:
            metadata["sample_policy_judge_prompt_swapped"] = judge_prompt_texts_swapped[0]
        if self.retain_judge_outputs:
            metadata.update(
                {
                    "judge_outputs": judge_outputs,
                    "judge_outputs_swapped": judge_outputs_swapped,
                }
            )
        return SequenceScoreOutputs(
            scores=torch.tensor(scores, dtype=torch.float32),
            metadata=metadata,
        )

    def _build_judge_prompt_text_batch(
        self,
        *,
        examples: list[dict[str, Any]],
        judge_template: str,
        rollout_builder: Any,
        response1_batch: list[str],
        response2_batch: list[str],
    ) -> list[str]:
        raise NotImplementedError

    def _try_build_judge_prompt_text_batch(
        self,
        *,
        examples: list[dict[str, Any]],
        judge_template: str,
        rollout_builder: Any,
        response1_batch: list[str],
        response2_batch: list[str],
    ) -> list[str]:
        try:
            return self._build_judge_prompt_text_batch(
                examples=examples,
                judge_template=judge_template,
                rollout_builder=rollout_builder,
                response1_batch=response1_batch,
                response2_batch=response2_batch,
            )
        except Exception:
            # Prompt text is only a logging artifact. Keep scoring behavior
            # governed by the token-id builder, which existing tests and callers
            # may monkeypatch directly.
            return []

    def _build_judge_prompt_token_ids_batch(
        self,
        *,
        examples: list[dict[str, Any]],
        judge_template: str,
        rollout_builder: Any,
        response1_batch: list[str],
        response2_batch: list[str],
    ) -> list[list[int]]:
        raise NotImplementedError


class PrivalignRLPairwiseJudgeScorer(PairwiseRLJudgeScorerBase):
    """Privalign pairwise judge over all sampled completions in each prompt group.

    The prompt embeds the privacy task, the tool trajectory, both previously
    annotated reference responses, and the per-annotator leak/omit labels before
    comparing sampled policy completions. For a group of 4 completions this runs
    all 6 unordered pairs, with both response orderings for each pair.
    """

    name = "privalign_rl_pairwise_judge"
    format_penalty = 1.0

    def __init__(
        self,
        *,
        privalign_judge_max_response_words: int | None = 1000,
        **kwargs: Any,
    ) -> None:
        self.privalign_judge_max_response_words = privalign_judge_max_response_words
        kwargs.setdefault("scoring_mode", "peer")
        super().__init__(**kwargs)

    def score(
        self,
        batch: TrajectoryBatch,
        *,
        context: TrajectoryScoreContext | None = None,
    ) -> SequenceScoreOutputs:
        outputs = super().score(batch, context=context)
        if context is None or context.examples is None:
            return outputs

        completion_ids = batch.get("completion_ids")
        completion_mask = batch.get("completion_mask")
        num_prompts = batch.get("num_prompts")
        if (
            not torch.is_tensor(completion_ids)
            or not torch.is_tensor(completion_mask)
            or not isinstance(num_prompts, int)
        ):
            return outputs

        completion_texts = _decode_completion_texts(
            self.rollout_builder, completion_ids, completion_mask, batch
        )
        repeated_examples = _expand_prompt_aligned_items(
            context.examples,
            num_prompts=num_prompts,
            num_sequences=int(completion_ids.size(0)),
        )

        violation_reasons: list[str | None] = []
        penalties: list[float] = []
        for completion_text, example in zip(completion_texts, repeated_examples):
            expected_action = _resolve_privalign_expected_final_action(example)
            reason = _privalign_tool_call_format_violation_reason(
                completion_text,
                expected_action=expected_action,
            )
            violation_reasons.append(reason)
            penalties.append(self.format_penalty if reason is not None else 0.0)

        penalty_tensor = torch.tensor(
            penalties,
            dtype=outputs.scores.dtype,
            device=outputs.scores.device,
        )
        violation_mask = penalty_tensor > 0
        outputs.scores = outputs.scores - penalty_tensor
        outputs.metadata["format_penalties"] = penalty_tensor.detach().cpu()
        outputs.metadata["format_violation_mask"] = violation_mask.detach().cpu()
        outputs.metadata["format_violation_reasons"] = violation_reasons
        outputs.metadata["format_violation_count"] = float(violation_mask.sum().item())
        outputs.metadata["format_violation_rate"] = (
            float(violation_mask.float().mean().item()) if penalties else 0.0
        )
        return outputs

    def _build_judge_prompt_token_ids_batch(
        self,
        *,
        examples: list[dict[str, Any]],
        judge_template: str,
        rollout_builder: Any,
        response1_batch: list[str],
        response2_batch: list[str],
    ) -> list[list[int]]:
        return build_privalign_rl_judge_prompt_token_ids_batch(
            examples=examples,
            judge_template=judge_template,
            rollout_builder=rollout_builder,
            response1_batch=response1_batch,
            response2_batch=response2_batch,
            max_response_words=self.privalign_judge_max_response_words,
        )

    def _build_judge_prompt_text_batch(
        self,
        *,
        examples: list[dict[str, Any]],
        judge_template: str,
        rollout_builder: Any,
        response1_batch: list[str],
        response2_batch: list[str],
    ) -> list[str]:
        return render_privalign_rl_judge_prompt_texts(
            examples=examples,
            judge_template=judge_template,
            rollout_builder=rollout_builder,
            response1_batch=response1_batch,
            response2_batch=response2_batch,
            max_response_words=self.privalign_judge_max_response_words,
        )


class PairwiseMarginScorer:
    """Reward shaping used to train the generative reward model (Phase A).

    The policy here is the gen-RM. Each prompt embeds two candidate responses
    and asks the model to emit a single pairwise preference score in [-2, +2]
    matching the eval pipeline:

        -2 = Response 1 clearly better
        -1 = Response 1 slightly better
         0 = about equal
         1 = Response 2 slightly better
         2 = Response 2 clearly better

    The training dataset includes both orderings of each pair so the gen-RM
    doesn't pick up a slot-position bias ("both ways").

    Reward per rollout (sign-only path):

        R = -c1 * I_format + c2 * (target_sign * predicted_score) - length_penalty

    where ``target_sign = +1`` if ``preferred_slot == 2`` (consensus-preferred
    in Response 2 slot) else ``-1``. Positive R = correctly signed prediction.
    Max correct = +c2 * 2; max wrong = -c2 * 2; format violation = -c1.

    Reward per rollout (soft-target path):

    When ``ranking_demo.target_score`` is present, the scorer rewards matching
    the averaged per-annotator target score produced by the data builder rather
    than just the sign::

        R = -c1 * I_format + c2 * (2 - |predicted - target_score|)
            + w_leak * R_leak - length_penalty

    Older rows with ``ranking_demo.target_signed_margin`` still use the same
    distance reward, with the integer margin treated as the target score.

    When ``ranking_demo.gold_leak_response1`` and
    ``ranking_demo.gold_leak_response2`` are present, the scorer also expects
    ``Response 1 leaks: yes|no`` and ``Response 2 leaks: yes|no`` lines and
    adds an auxiliary per-response leak term::

        R_leak = sum_i(-|pred_leak_i - gold_leak_i|)

    Max correct (|pred - target| = 0) = +c2 * 2; worst valid (|pred - target| = 4)
    = -c2 * 2; leak reward ranges from -2 to 0 when both responses are
    labeled; format violation = -c1.
    """

    name = "pairwise_margin"
    kind = "pairwise_margin"

    _VALID_SCORE_RANGE = (-2, 2)

    def __init__(
        self,
        *,
        rollout_builder: Any,
        c1: float = 5.0,
        c2: float = 1.0,
        w_leak: float = 1.0,
        penalty_max_len: int | None = None,
        penalty_per_word: float = 0.0,
        penalty_max_value: float | None = 2.0,
        penalty_shape: str = "linear",
    ) -> None:
        self.rollout_builder = rollout_builder
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.w_leak = float(w_leak)
        self.penalty_max_len = penalty_max_len
        self.penalty_per_word = float(penalty_per_word)
        self.penalty_max_value = None if penalty_max_value is None else float(penalty_max_value)
        if penalty_shape not in {"linear"}:
            raise ValueError(f"penalty_shape must be 'linear'; got {penalty_shape!r}.")
        self.penalty_shape = penalty_shape

    def score(
        self,
        batch: TrajectoryBatch,
        *,
        context: TrajectoryScoreContext | None = None,
    ) -> SequenceScoreOutputs:
        if context is None or context.examples is None:
            raise ValueError("PairwiseMarginScorer requires examples in TrajectoryScoreContext.")

        completion_ids = batch.get("completion_ids")
        completion_mask = batch.get("completion_mask")
        if not torch.is_tensor(completion_ids) or not torch.is_tensor(completion_mask):
            raise KeyError("PairwiseMarginScorer requires completion_ids and completion_mask.")

        num_prompts = batch.get("num_prompts")
        if not isinstance(num_prompts, int) or num_prompts <= 0:
            raise KeyError("PairwiseMarginScorer requires a positive integer num_prompts.")

        num_sequences = int(completion_ids.size(0))
        repeated_examples = _expand_prompt_aligned_items(
            context.examples, num_prompts=num_prompts, num_sequences=num_sequences
        )
        completion_texts = _decode_completion_texts(
            self.rollout_builder, completion_ids, completion_mask, batch
        )

        scores: list[float] = []
        predicted_values: list[int | None] = []
        signed_targets: list[float | None] = []
        margin_l1: list[float | None] = []
        length_penalties: list[float] = []
        format_violations: list[bool] = []
        exact_margin_used: list[bool] = []
        predicted_leak1_values: list[bool | None] = []
        predicted_leak2_values: list[bool | None] = []
        leak_reward_values: list[float] = []
        format_violation_count = 0
        leak_format_violation_count = 0
        leak_labeled_count = 0
        leak_correct_count = 0
        leak_abs_error_sum = 0.0
        correct_count = 0
        valid_count = 0
        for completion_text, example in zip(completion_texts, repeated_examples):
            exact_target = _resolve_target_score(example)
            ranking_demo = example.get("ranking_demo") or example.get("judge_demo") or {}
            gold_leak1 = ranking_demo.get("gold_leak_response1")
            gold_leak2 = ranking_demo.get("gold_leak_response2")
            has_leak_targets = (
                isinstance(gold_leak1, (int, float))
                and isinstance(gold_leak2, (int, float))
                and 0.0 <= float(gold_leak1) <= 1.0
                and 0.0 <= float(gold_leak2) <= 1.0
            )
            if exact_target is None:
                preferred_slot = _resolve_preferred_slot(example)
                target_sign = 1.0 if preferred_slot == 2 else -1.0
            else:
                # Mirror the preferred-slot view so downstream sign metrics
                # ("pair_accuracy", "signed_target_mean") stay comparable.
                target_sign = 1.0 if exact_target > 0 else (-1.0 if exact_target < 0 else 0.0)
            predicted = _parse_signed_score(completion_text, valid_range=self._VALID_SCORE_RANGE)
            pred_leak1, pred_leak2 = _parse_pairwise_leaks(completion_text)
            predicted_leak1_values.append(pred_leak1)
            predicted_leak2_values.append(pred_leak2)
            predicted_values.append(predicted)
            leak_format_violation = has_leak_targets and (pred_leak1 is None or pred_leak2 is None)
            if leak_format_violation:
                leak_format_violation_count += 1
            format_violation = predicted is None or leak_format_violation
            format_violations.append(format_violation)
            exact_margin_used.append(exact_target is not None)
            if format_violation:
                format_violation_count += 1
            if predicted is None:
                signed_target: float | None = None
                signed_for_reward = 0.0
                margin_l1.append(None)
            else:
                signed_target = target_sign * float(predicted)
                valid_count += 1
                if exact_target is None:
                    signed_for_reward = signed_target  # sign-only path
                else:
                    abs_err = abs(float(predicted) - float(exact_target))
                    margin_l1.append(abs_err)
                    # soft-target reward, centered to match the sign-only [-2,+2] range
                    signed_for_reward = 2.0 - abs_err
                if exact_target is None:
                    margin_l1.append(None)
                if exact_target is None:
                    is_correct = signed_target > 0
                elif exact_target == 0:
                    is_correct = predicted == 0
                else:
                    is_correct = (predicted > 0) == (exact_target > 0)
                if is_correct and not format_violation:
                    correct_count += 1

            leak_reward = 0.0
            if has_leak_targets:
                per_response_rewards: list[float] = []
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
                    per_response_rewards.append(-abs_error)
                    leak_abs_error_sum += abs_error
                    if (pred_leak_f >= 0.5) == (gold_leak >= 0.5):
                        leak_correct_count += 1
                if per_response_rewards:
                    leak_reward = sum(per_response_rewards)
            leak_reward_values.append(leak_reward)
            signed_targets.append(signed_target)
            length_penalty = _linear_length_penalty(
                completion_text,
                max_len=self.penalty_max_len,
                per_word=self.penalty_per_word,
                max_value=self.penalty_max_value,
                shape=self.penalty_shape,
            )
            length_penalties.append(length_penalty)
            reward = (
                -self.c1 * (1.0 if format_violation else 0.0)
                + self.c2 * signed_for_reward
                + self.w_leak * leak_reward
                - length_penalty
            )
            scores.append(reward)

        valid_signed = [s for s in signed_targets if s is not None]
        valid_predicted = [p for p in predicted_values if p is not None]
        valid_l1 = [v for v in margin_l1 if v is not None]
        return SequenceScoreOutputs(
            scores=torch.tensor(scores, dtype=torch.float32),
            metadata={
                "score_name": self.name,
                "format_violation_count": float(format_violation_count),
                "format_violation_rate": float(format_violation_count) / max(1, num_sequences),
                "pair_accuracy": float(correct_count) / max(1, num_sequences),
                "signed_target_mean": (
                    float(sum(valid_signed) / len(valid_signed)) if valid_signed else 0.0
                ),
                "predicted_score_abs_mean": (
                    float(sum(abs(p) for p in valid_predicted) / len(valid_predicted))
                    if valid_predicted else 0.0
                ),
                "exact_margin_l1_mean": (
                    float(sum(valid_l1) / len(valid_l1)) if valid_l1 else 0.0
                ),
                "exact_margin_used_rate": float(sum(exact_margin_used)) / max(1, num_sequences),
                "leak_format_violation_count": float(leak_format_violation_count),
                "leak_format_violation_rate": float(leak_format_violation_count) / max(1, num_sequences),
                "leak_accuracy": (
                    float(leak_correct_count) / leak_labeled_count if leak_labeled_count else 0.0
                ),
                "leak_abs_error_mean": (
                    float(leak_abs_error_sum) / leak_labeled_count if leak_labeled_count else 0.0
                ),
                "leak_reward_mean": (
                    float(sum(leak_reward_values) / len(leak_reward_values))
                    if leak_reward_values else 0.0
                ),
                "predicted_scores": predicted_values,
                "predicted_leak_response1": predicted_leak1_values,
                "predicted_leak_response2": predicted_leak2_values,
                "signed_targets": signed_targets,
                "leak_rewards": torch.tensor(leak_reward_values, dtype=torch.float32),
                "length_penalties": torch.tensor(length_penalties, dtype=torch.float32),
                "format_violation_mask": torch.tensor(format_violations, dtype=torch.bool),
            },
        )


class PrivalignPairwiseMarginScorer(PairwiseMarginScorer):
    """Privalign-named pairwise GenRM scorer.

    Behavior is inherited from ``PairwiseMarginScorer``; Privalign rows carry
    soft preference targets plus optional per-response soft leak targets.
    """

    name = "privalign_pairwise_margin"


class TrainedGenRMScorer:
    """Use a trained generative reward model (Phase A output) as the policy reward source.

    Scoring is **pairwise-within-group**: each prompt's ``num_generations``
    rollouts are compared against each other (not against a fixed external
    anchor). For a group of size G this is ``C(G, 2)`` gen-RM calls. Each
    rollout's reward is the mean signed margin across its ``G-1`` peer
    comparisons. With ``num_generations=4`` that's 6 calls per prompt and 3
    contributions per rollout.

    Per-pair semantics: for pair ``(i, j)`` with ``i < j``, ``r_i`` is placed
    in slot 1 and ``r_j`` in slot 2. The gen-RM emits ``Score: <int -2..2>``
    on the eval-pipeline scale; positive = slot-2 preferred (= ``r_j`` wins).

    Per rollout reward (before length penalty):

        reward(r_k) = mean over peers p of "k beat p" signal
                    = mean of (+score when k is slot-2; -score when k is slot-1)

    Range: [-2, +2] when all peer comparisons parse. Length penalty applied
    per rollout as before. Rollouts that fail to receive any valid peer
    comparison fall back to ``invalid_score_scalar`` (default 0).

    Note: peer-relative scoring is naturally zero-sum across the group, so
    the REINFORCE++-baseline group-mean subtraction is mostly a no-op here
    (advantages closely track the per-rollout reward itself).
    """

    name = "trained_genrm"
    kind = "trained_genrm"

    _VALID_SCORE_RANGE = (-2, 2)
    _DEFAULT_PROMPT_TEMPLATE_NAME = "privalign_genrm_pairwise"
    _PRIVALIGN_TEMPLATE_NAME = "privalign_genrm_pairwise"
    _PRIVALIGN_TEMPLATE_NAMES = frozenset(
        {"privalign_genrm_pairwise", "privalign_rl_pairwise_judge"}
    )

    def __init__(
        self,
        *,
        rollout_builder: Any,
        rm_rollout_builder: Any,
        rm_text_generator: Callable[..., list[str]],
        rm_max_new_tokens: int = 1024,
        rm_dual_order: bool = False,
        invalid_score_scalar: float = 0.0,
        penalty_max_len: int | None = None,
        penalty_per_word: float = 0.0,
        penalty_max_value: float | None = 2.0,
        penalty_shape: str = "linear",
        retain_rm_outputs: bool = True,
        prompt_template_name: str | None = None,
        annotation_conditioning: bool = False,
        privalign_judge_max_response_words: int | None = 1000,
        undershort_penalty_max: float = 0.0,
        undershort_floor_ratio: float = 0.5,
    ) -> None:
        from data_loaders import load_prompt
        self.rollout_builder = rollout_builder
        self.rm_rollout_builder = rm_rollout_builder
        self.rm_text_generator = rm_text_generator
        self.rm_max_new_tokens = int(rm_max_new_tokens)
        self.rm_dual_order = bool(rm_dual_order)
        self.invalid_score_scalar = float(invalid_score_scalar)
        self.penalty_max_len = penalty_max_len
        self.penalty_per_word = float(penalty_per_word)
        self.penalty_max_value = None if penalty_max_value is None else float(penalty_max_value)
        if penalty_shape not in {"linear"}:
            raise ValueError(f"penalty_shape must be 'linear'; got {penalty_shape!r}.")
        self.penalty_shape = penalty_shape
        self.retain_rm_outputs = bool(retain_rm_outputs)
        self._prompt_template_name = (
            prompt_template_name or self._DEFAULT_PROMPT_TEMPLATE_NAME
        )
        self._prompt_template = load_prompt(self._prompt_template_name)
        self._is_privalign_template = (
            self._prompt_template_name in self._PRIVALIGN_TEMPLATE_NAMES
        )
        self._annotation_conditioning = bool(annotation_conditioning)
        if self._annotation_conditioning and not self._is_privalign_template:
            raise ValueError(
                "annotation_conditioning=True requires a Privalign prompt template; got "
                f"prompt_template_name={self._prompt_template_name!r}."
            )
        self.privalign_judge_max_response_words = privalign_judge_max_response_words
        self.undershort_penalty_max = float(undershort_penalty_max)
        self.undershort_floor_ratio = float(undershort_floor_ratio)

    def score(
        self,
        batch: TrajectoryBatch,
        *,
        context: TrajectoryScoreContext | None = None,
    ) -> SequenceScoreOutputs:
        if context is None or context.examples is None:
            raise ValueError("TrainedGenRMScorer requires examples in TrajectoryScoreContext.")

        completion_ids = batch.get("completion_ids")
        completion_mask = batch.get("completion_mask")
        if not torch.is_tensor(completion_ids) or not torch.is_tensor(completion_mask):
            raise KeyError("TrainedGenRMScorer requires completion_ids and completion_mask.")

        num_prompts = batch.get("num_prompts")
        if not isinstance(num_prompts, int) or num_prompts <= 0:
            raise KeyError("TrainedGenRMScorer requires a positive integer num_prompts.")

        num_sequences = int(completion_ids.size(0))
        if num_sequences % num_prompts != 0:
            raise ValueError(
                f"num_sequences ({num_sequences}) must be divisible by num_prompts ({num_prompts})."
            )
        group_size = num_sequences // num_prompts
        if group_size < 2:
            raise ValueError(
                "TrainedGenRMScorer requires num_generations >= 2 for pairwise-within-group scoring; "
                f"got group_size={group_size}."
            )
        repeated_examples = _expand_prompt_aligned_items(
            context.examples, num_prompts=num_prompts, num_sequences=num_sequences
        )
        completion_texts = _decode_completion_texts(
            self.rollout_builder, completion_ids, completion_mask, batch
        )

        # Build C(group_size, 2) pairs per prompt: (slot1_seq, slot2_seq) with
        # slot1_seq < slot2_seq, both inside the same prompt's group. When
        # dual-order scoring is enabled, add the swapped prompt for each pair
        # and later negate its score back into the original slot-2 convention.
        pair_indices: list[tuple[int, int]] = []  # global sequence indices
        rate_messages_batch: list[list[dict[str, str]]] = []
        rate_messages_batch_swapped: list[list[dict[str, str]]] = []
        for p in range(num_prompts):
            base_idx = p * group_size
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    slot1_seq = base_idx + i
                    slot2_seq = base_idx + j
                    msgs = self._build_rate_messages(
                        example=repeated_examples[base_idx],
                        response1_text=completion_texts[slot1_seq],
                        response2_text=completion_texts[slot2_seq],
                    )
                    rate_messages_batch.append(msgs)
                    if self.rm_dual_order:
                        swapped_msgs = self._build_rate_messages(
                            example=repeated_examples[base_idx],
                            response1_text=completion_texts[slot2_seq],
                            response2_text=completion_texts[slot1_seq],
                        )
                        rate_messages_batch_swapped.append(swapped_msgs)
                    pair_indices.append((slot1_seq, slot2_seq))
        num_pairs = len(rate_messages_batch)

        rate_prompt_texts = [
            self.rm_rollout_builder._format_student_prompt(msgs)
            for msgs in rate_messages_batch
        ]
        rate_prompt_texts_swapped = [
            self.rm_rollout_builder._format_student_prompt(msgs)
            for msgs in rate_messages_batch_swapped
        ]
        rate_prompt_token_ids_batch = self.rm_rollout_builder._tokenize_prompt_text_sequences(
            rate_prompt_texts
        )
        rate_prompt_token_ids_batch_swapped = (
            self.rm_rollout_builder._tokenize_prompt_text_sequences(
                rate_prompt_texts_swapped
            )
            if rate_prompt_texts_swapped
            else []
        )
        rm_outputs = self.rm_text_generator(
            rate_prompt_token_ids_batch + rate_prompt_token_ids_batch_swapped,
            max_new_tokens=self.rm_max_new_tokens,
        )
        expected_output_count = num_pairs * (2 if self.rm_dual_order else 1)
        if len(rm_outputs) != expected_output_count:
            raise ValueError(
                "Gen-RM output count must match the configured pair orderings "
                f"({len(rm_outputs)} != {expected_output_count})."
            )
        rm_outputs_primary = rm_outputs[:num_pairs]
        rm_outputs_swapped = (
            rm_outputs[num_pairs:]
            if self.rm_dual_order
            else [""] * num_pairs
        )

        # Accumulate signed peer-comparison contributions per rollout.
        per_rollout_sum = [0.0] * num_sequences
        per_rollout_count = [0] * num_sequences
        predicted_pair_scores: list[float | None] = [None] * num_pairs
        predicted_pair_primary_scores: list[int | None] = [None] * num_pairs
        predicted_pair_swapped_scores: list[int | None] = [None] * num_pairs
        pair_invalid_count = 0
        for idx, ((slot1_seq, slot2_seq), rm_output, swapped_output) in enumerate(
            zip(pair_indices, rm_outputs_primary, rm_outputs_swapped)
        ):
            predicted = _parse_signed_score(rm_output, valid_range=self._VALID_SCORE_RANGE)
            swapped_predicted = (
                _parse_signed_score(swapped_output, valid_range=self._VALID_SCORE_RANGE)
                if self.rm_dual_order
                else None
            )
            predicted_pair_primary_scores[idx] = predicted
            predicted_pair_swapped_scores[idx] = swapped_predicted

            normalized_scores: list[float] = []
            if predicted is not None:
                normalized_scores.append(float(predicted))
            if swapped_predicted is not None:
                normalized_scores.append(float(-swapped_predicted))

            if self.rm_dual_order and len(normalized_scores) != 2:
                combined_predicted = None
            elif normalized_scores:
                combined_predicted = sum(normalized_scores) / len(normalized_scores)
            else:
                combined_predicted = None

            predicted_pair_scores[idx] = combined_predicted
            if combined_predicted is None:
                pair_invalid_count += 1
                continue
            # Positive predicted = slot 2 (r_j) preferred.
            per_rollout_sum[slot2_seq] += float(combined_predicted)
            per_rollout_sum[slot1_seq] -= float(combined_predicted)
            per_rollout_count[slot2_seq] += 1
            per_rollout_count[slot1_seq] += 1

        scores: list[float] = []
        length_penalties: list[float] = []
        invalid_mask: list[bool] = []
        per_rollout_means: list[float] = []
        rollouts_with_no_valid_peer = 0
        for i in range(num_sequences):
            cnt = per_rollout_count[i]
            if cnt == 0:
                rollouts_with_no_valid_peer += 1
                invalid_mask.append(True)
                mean_margin = self.invalid_score_scalar
            else:
                invalid_mask.append(False)
                mean_margin = per_rollout_sum[i] / cnt
            per_rollout_means.append(mean_margin)
            length_penalty = _linear_length_penalty(
                completion_texts[i],
                max_len=self.penalty_max_len,
                per_word=self.penalty_per_word,
                max_value=self.penalty_max_value,
                shape=self.penalty_shape,
            )
            length_penalties.append(length_penalty)
            scores.append(mean_margin - length_penalty)

        undershort_penalties, undershort_metadata = _compute_privalign_group_undershort_penalties(
            completion_texts=completion_texts,
            num_prompts=num_prompts,
            examples=context.examples,
            penalty_max=(
                self.undershort_penalty_max if self._is_privalign_template else 0.0
            ),
            floor_ratio=self.undershort_floor_ratio,
        )
        if undershort_penalties:
            scores = [
                score - undershort_penalty
                for score, undershort_penalty in zip(scores, undershort_penalties)
            ]

        valid_pair_predicted = [p for p in predicted_pair_scores if p is not None]
        metadata: dict[str, Any] = {
            "score_name": self.name,
            "rm_dual_order": float(self.rm_dual_order),
            "rm_predicted_pair_scores": predicted_pair_scores,
            "rm_predicted_pair_primary_scores": predicted_pair_primary_scores,
            "rm_predicted_pair_swapped_scores": predicted_pair_swapped_scores,
            "rm_pair_indices": pair_indices,
            "rm_per_rollout_means": per_rollout_means,
            "rm_num_pairs": float(num_pairs),
            "rm_pair_invalid_count": float(pair_invalid_count),
            "rm_pair_invalid_rate": float(pair_invalid_count) / max(1, num_pairs),
            "rm_invalid_count": float(rollouts_with_no_valid_peer),
            "rm_invalid_rate": float(rollouts_with_no_valid_peer) / max(1, num_sequences),
            "invalid_sequence_count": float(rollouts_with_no_valid_peer),
            "invalid_sequence_rate": float(rollouts_with_no_valid_peer) / max(1, num_sequences),
            "rm_predicted_score_mean": (
                float(sum(valid_pair_predicted) / len(valid_pair_predicted))
                if valid_pair_predicted else 0.0
            ),
            "rm_predicted_score_abs_mean": (
                float(sum(abs(p) for p in valid_pair_predicted) / len(valid_pair_predicted))
                if valid_pair_predicted else 0.0
            ),
            "length_penalties": torch.tensor(length_penalties, dtype=torch.float32),
            "invalid_sequence_mask": torch.tensor(invalid_mask, dtype=torch.bool),
        }
        if undershort_metadata:
            metadata.update(undershort_metadata)
        if self.retain_rm_outputs:
            metadata["rm_outputs"] = rm_outputs_primary
            metadata["rm_outputs_swapped"] = rm_outputs_swapped
        return SequenceScoreOutputs(
            scores=torch.tensor(scores, dtype=torch.float32),
            metadata=metadata,
        )

    def _build_rate_messages(
        self,
        *,
        example: dict[str, Any],
        response1_text: str,
        response2_text: str,
    ) -> list[dict[str, str]]:
        if self._annotation_conditioning:
            rate_content = self._render_privalign_annotation_conditioned_rate_content(
                example=example,
                response1_text=_truncate_judge_response(
                    response1_text,
                    max_words=self.privalign_judge_max_response_words,
                ),
                response2_text=_truncate_judge_response(
                    response2_text,
                    max_words=self.privalign_judge_max_response_words,
                ),
            )
        else:
            rate_content = self._render_privalign_rate_content(
                example=example,
                response1_text=_truncate_judge_response(
                    response1_text,
                    max_words=self.privalign_judge_max_response_words,
                ),
                response2_text=_truncate_judge_response(
                    response2_text,
                    max_words=self.privalign_judge_max_response_words,
                ),
            )
        rate_content = re.sub(r"\n{3,}", "\n\n", rate_content)
        return [{"role": "user", "content": rate_content}]

    def _render_privalign_rate_content(
        self,
        *,
        example: dict[str, Any],
        response1_text: str,
        response2_text: str,
    ) -> str:
        """Render the Privalign rate-this prompt from ``example['judge_demo']``.

        The Privalign loader populates ``judge_demo`` with ``user_instruction``,
        ``memories`` (already formatted), and ``executable_trajectory``. Fall
        back to ``ranking_demo`` if a downstream caller has copied the demo
        there (e.g. JSONL builder output).
        """
        demo = example.get("judge_demo") or example.get("ranking_demo")
        if not isinstance(demo, dict):
            raise KeyError(
                "TrainedGenRMScorer requires example.judge_demo (or ranking_demo) "
                "with Privalign fields when prompt_template_name='privalign_genrm_pairwise'."
            )
        user_instruction = str(demo.get("user_instruction") or "").strip()
        memories = str(demo.get("memories") or "").strip() or "(no memories provided)"
        executable_trajectory = (
            str(demo.get("executable_trajectory") or "").strip() or "(no prior tool use)"
        )
        return self._prompt_template.format(
            user_instruction=user_instruction,
            memories=memories,
            executable_trajectory=executable_trajectory,
            eval_response1=response1_text,
            eval_response2=response2_text,
        )

    def _render_privalign_annotation_conditioned_rate_content(
        self,
        *,
        example: dict[str, Any],
        response1_text: str,
        response2_text: str,
    ) -> str:
        """Render a Privalign rate-this prompt with annotation conditioning.

        Delegates to ``render_privalign_template``, which fills
        ``{reference_response_a}``, ``{reference_response_b}``, and
        ``{annotator_block}`` from ``example['judge_demo']`` in addition to
        the user-instruction/memories/trajectory fields.
        """
        from data_loaders.preference import render_privalign_template

        demo = example.get("judge_demo") or example.get("ranking_demo")
        if not isinstance(demo, dict):
            raise KeyError(
                "TrainedGenRMScorer with annotation_conditioning=True requires "
                "example.judge_demo (or ranking_demo) with Privalign fields."
            )
        return render_privalign_template(
            self._prompt_template,
            judge_demo=demo,
            eval_response1=response1_text,
            eval_response2=response2_text,
        )


def _resolve_preferred_slot(example: dict[str, Any]) -> int:
    ranking_demo = example.get("ranking_demo") or example.get("judge_demo")
    if not isinstance(ranking_demo, dict):
        raise KeyError(
            "PairwiseMarginScorer requires example.ranking_demo (or judge_demo) with preferred_slot."
        )
    preferred_slot = ranking_demo.get("preferred_slot")
    if not isinstance(preferred_slot, int) or preferred_slot not in (1, 2):
        raise ValueError(
            "PairwiseMarginScorer requires ranking_demo.preferred_slot to be 1 or 2; "
            f"got {preferred_slot!r}."
        )
    return preferred_slot


def _resolve_target_score(example: dict[str, Any]) -> float | None:
    """Return the soft pairwise target in [-2, +2] if present.

    New Privalign rows carry ``ranking_demo.target_score`` as an averaged
    per-annotator score. Older rows carry integer ``target_signed_margin``;
    those remain valid fallback targets.
    """
    demo = example.get("ranking_demo") or example.get("judge_demo")
    if not isinstance(demo, dict):
        return None
    field_name = "target_score" if "target_score" in demo else "target_signed_margin"
    value = demo.get(field_name)
    if value is None:
        return None
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"ranking_demo.{field_name} must be numeric; got "
            f"{value!r}."
        )
    if not math.isfinite(fvalue) or fvalue < -2.0 or fvalue > 2.0:
        raise ValueError(
            f"ranking_demo.{field_name} must be in [-2, +2]; got "
            f"{fvalue}."
        )
    return fvalue


def _linear_length_penalty(
    completion_text: str,
    *,
    max_len: int | None,
    per_word: float,
    shape: str,
    max_value: float | None = 2.0,
) -> float:
    if max_len is None or per_word <= 0.0:
        return 0.0
    word_count = len(completion_text.split())
    extra_words = max(0, word_count - int(max_len))
    if shape == "linear":
        penalty = float(extra_words) * per_word
        if max_value is not None:
            penalty = min(penalty, float(max_value))
        return penalty
    raise ValueError(f"penalty_shape must be 'linear'; got {shape!r}.")


_PRIVALIGN_PAYLOAD_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_PRIVALIGN_PAYLOAD_EXCLUDED_KEYS = {
    "action",
    "bcc",
    "cc",
    "channel",
    "channel_id",
    "channel_name",
    "from",
    "id",
    "name",
    "parent_id",
    "recipient",
    "recipients",
    "subject",
    "team",
    "team_name",
    "title",
    "to",
    "tool",
    "tool_name",
    "type",
    "url",
    "urls",
}


def _compute_privalign_group_undershort_penalties(
    *,
    completion_texts: list[str],
    num_prompts: int,
    examples: list[dict[str, Any]] | None = None,
    penalty_max: float,
    floor_ratio: float,
) -> tuple[list[float], dict[str, Any]]:
    """Return per-sequence undershortness penalties for Privalign JSON actions.

    The floor is prompt-conditioned: ``floor_ratio`` times the average payload
    word count of the two dataset reference responses. Malformed rollouts do
    not receive this penalty; format-specific penalties remain responsible for
    those.
    """
    penalty_max = float(penalty_max)
    floor_ratio = float(floor_ratio)
    if penalty_max <= 0.0 or floor_ratio <= 0.0:
        return [], {}
    num_sequences = len(completion_texts)
    if num_prompts <= 0 or num_sequences == 0 or num_sequences % num_prompts != 0:
        return [], {}

    group_size = num_sequences // num_prompts
    penalties = [0.0] * num_sequences
    payload_word_counts = [0] * num_sequences
    valid_json_mask = [False] * num_sequences
    group_floors = [0.0] * num_sequences
    group_valid_counts = [0] * num_sequences
    reference_average_word_counts = [0.0] * num_sequences

    for group_idx in range(num_prompts):
        start = group_idx * group_size
        end = start + group_size
        valid_indices: list[int] = []
        for seq_idx in range(start, end):
            parsed = _parse_privalign_json_completion(completion_texts[seq_idx])
            if parsed is None:
                continue
            word_count = _privalign_payload_word_count(parsed)
            valid_json_mask[seq_idx] = True
            payload_word_counts[seq_idx] = word_count
            valid_indices.append(seq_idx)

        valid_count = len(valid_indices)
        for seq_idx in range(start, end):
            group_valid_counts[seq_idx] = valid_count
        reference_average_word_count = _resolve_privalign_reference_average_word_count(
            examples[group_idx] if examples is not None and group_idx < len(examples) else None
        )
        if reference_average_word_count <= 0.0:
            continue

        floor = floor_ratio * reference_average_word_count
        for seq_idx in range(start, end):
            group_floors[seq_idx] = floor
            reference_average_word_counts[seq_idx] = reference_average_word_count
        if floor <= 0.0:
            continue

        for seq_idx in valid_indices:
            word_count = payload_word_counts[seq_idx]
            shortfall_ratio = max(0.0, floor - float(word_count)) / floor
            penalties[seq_idx] = penalty_max * min(1.0, shortfall_ratio)

    penalty_count = sum(1 for value in penalties if value > 0.0)
    valid_count_total = sum(1 for value in valid_json_mask if value)
    metadata: dict[str, Any] = {
        "undershort_penalties": torch.tensor(penalties, dtype=torch.float32),
        "undershort_payload_word_counts": payload_word_counts,
        "undershort_valid_json_mask": torch.tensor(valid_json_mask, dtype=torch.bool),
        "undershort_group_floors": group_floors,
        "undershort_reference_average_word_counts": reference_average_word_counts,
        "undershort_group_valid_counts": group_valid_counts,
        "undershort_penalty_count": float(penalty_count),
        "undershort_penalty_rate": float(penalty_count) / max(1, num_sequences),
        "undershort_penalty_valid_json_rate": float(penalty_count) / max(1, valid_count_total),
        "undershort_penalty_mean": float(sum(penalties) / max(1, num_sequences)),
        "undershort_penalty_max": penalty_max,
        "undershort_floor_ratio": floor_ratio,
    }
    return penalties, metadata


def _parse_privalign_json_completion(completion_text: str) -> dict[str, Any] | None:
    stripped = _strip_single_markdown_json_fence(strip_thinking_trace(completion_text or ""))
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_privalign_reference_average_word_count(example: dict[str, Any] | None) -> float:
    if not isinstance(example, dict):
        return 0.0
    demo = example.get("judge_demo") or example.get("ranking_demo")
    if not isinstance(demo, dict):
        return 0.0
    counts: list[int] = []
    for field_name in ("reference_response_a", "reference_response_b"):
        value = demo.get(field_name)
        if isinstance(value, str) and value.strip():
            counts.append(_privalign_response_word_count(value))
    if not counts:
        return 0.0
    return float(sum(counts)) / float(len(counts))


def _privalign_response_word_count(text: str) -> int:
    parsed = _parse_privalign_json_completion(text)
    if parsed is not None:
        return _privalign_payload_word_count(parsed)
    return len(_PRIVALIGN_PAYLOAD_WORD_RE.findall(strip_thinking_trace(text)))


def _privalign_payload_word_count(parsed: dict[str, Any]) -> int:
    arguments = parsed.get("arguments")
    payload_root = arguments if isinstance(arguments, dict) else parsed
    payload_text = " ".join(_iter_privalign_payload_strings(payload_root))
    return len(_PRIVALIGN_PAYLOAD_WORD_RE.findall(payload_text))


def _iter_privalign_payload_strings(value: Any, *, key: str | None = None):
    if key is not None and key.lower() in _PRIVALIGN_PAYLOAD_EXCLUDED_KEYS:
        return
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if isinstance(child_key, str):
                yield from _iter_privalign_payload_strings(child_value, key=child_key)
            else:
                yield from _iter_privalign_payload_strings(child_value, key=None)
        return
    if isinstance(value, list):
        for child_value in value:
            yield from _iter_privalign_payload_strings(child_value, key=key)


def _resolve_privalign_expected_final_action(example: dict[str, Any]) -> str:
    judge_demo = example.get("judge_demo")
    if isinstance(judge_demo, dict):
        expected = judge_demo.get("expected_final_action")
        if isinstance(expected, str) and expected.strip():
            return expected.strip()
    expected = example.get("expected_final_action")
    if isinstance(expected, str) and expected.strip():
        return expected.strip()
    return ""


def _privalign_tool_call_format_violation_reason(
    completion_text: str,
    *,
    expected_action: str,
) -> str | None:
    stripped = _strip_single_markdown_json_fence(completion_text)
    if not stripped:
        return "empty"
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return "invalid_json"
    if not isinstance(parsed, dict):
        return "not_json_object"
    if parsed.get("type") != "tool_use":
        return "missing_tool_use_type"
    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip():
        return "missing_tool_name"
    if expected_action and name.strip() != expected_action:
        return "wrong_tool_name"
    if not isinstance(parsed.get("arguments"), dict):
        return "missing_arguments_object"
    return None


def _strip_single_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.fullmatch(
        r"```\s*(?:json|JSON)?[ \t]*\r?\n(?P<body>.*?)\r?\n```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match is None:
        return stripped
    return fence_match.group("body").strip()


def _decode_completion_texts(
    rollout_builder: Any,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    batch: TrajectoryBatch | None = None,
) -> list[str]:
    if batch is not None:
        cached_texts = batch.get("completion_texts")
        if isinstance(cached_texts, list) and len(cached_texts) == completion_ids.size(0):
            if all(isinstance(text, str) for text in cached_texts):
                return list(cached_texts)

    texts: list[str] = []
    for token_ids, token_mask in zip(completion_ids, completion_mask):
        valid_length = int(token_mask.sum().item())
        texts.append(decode_response_tokens(
            rollout_builder.processing_class,
            token_ids[:valid_length].tolist(),
        ))
    return texts


def _expand_prompt_aligned_items(
    items: list[Any],
    *,
    num_prompts: int,
    num_sequences: int,
) -> list[Any]:
    if len(items) != num_prompts:
        raise ValueError(f"Expected {num_prompts} prompt-aligned items, but got {len(items)}.")
    if num_sequences % num_prompts != 0:
        raise ValueError(
            "num_sequences must be divisible by num_prompts to expand prompt-aligned scorer context."
        )
    group_size = num_sequences // num_prompts
    expanded: list[Any] = []
    for item in items:
        expanded.extend([item] * group_size)
    return expanded


def _resolve_reference_response_batch(
    examples: list[dict[str, Any]],
    *,
    field_name: str = "policy_reference_response",
) -> list[str]:
    reference_responses: list[str] = []
    for index, example in enumerate(examples):
        reference_response = example.get(field_name)
        if not isinstance(reference_response, str) or not reference_response:
            raise KeyError(
                "Pairwise judge scorers require each example to include a non-empty "
                f"{field_name!r} string; missing or invalid value at index {index}."
            )
        reference_responses.append(reference_response)
    return reference_responses
