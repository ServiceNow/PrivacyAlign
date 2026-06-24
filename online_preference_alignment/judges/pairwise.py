"""General pairwise-judge prompt rendering, output parsing, and dual-order combination."""

from __future__ import annotations

import json
import re
from typing import Any

from utils.text_utils import parse_response_trace, strip_thinking_trace


_STANDALONE_SCORE_RE = re.compile(
    r"^\s*score\s*[:=]\s*(-?[0-2])\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_pairwise_judge_output(
    text: str,
    *,
    processing_class: Any | None = None,
) -> tuple[int | None, str | None]:
    """Parse a pairwise judge output containing a Reasoning line and a Score line.

    Returns (score, reasoning) where score is in {-2,-1,0,1,2} or None if
    no parseable score was found. Falls back to looking for a JSON object with
    `score` and `reasoning` fields.

    Pass `processing_class` (the judge tokenizer/processor) so the gemma
    `processor.parse_response` helper can be used to strip the thinking trace
    when available; otherwise the regex fallback handles Qwen and gemma.
    """
    cleaned_text = parse_response_trace(text, processing_class)
    extracted_reasoning = cleaned_text.strip() or None
    labeled_reasoning_match = re.search(
        r"^\s*reasoning\s*:\s*(.*?)(?=^\s*score\s*[:=]\s*-?[0-2]\s*$|\Z)",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if labeled_reasoning_match is not None:
        labeled_reasoning = labeled_reasoning_match.group(1).strip()
        if labeled_reasoning:
            extracted_reasoning = labeled_reasoning
    labeled_score_matches = list(_STANDALONE_SCORE_RE.finditer(cleaned_text))
    if labeled_score_matches:
        labeled_score_match = labeled_score_matches[-1]
        labeled_score = int(labeled_score_match.group(1))
        if labeled_score in {-2, -1, 0, 1, 2}:
            return labeled_score, extracted_reasoning
    match = re.search(r"\{.*?\}", cleaned_text, flags=re.DOTALL)
    if match is not None:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            raw_score = parsed.get("score")
            if raw_score is not None:
                try:
                    score = int(raw_score)
                except (TypeError, ValueError):
                    score = None
                if score in {-2, -1, 0, 1, 2}:
                    reasoning = parsed.get("reasoning")
                    if reasoning is not None:
                        extracted_reasoning = str(reasoning)
                    return score, extracted_reasoning

    # Defense-in-depth: if cleaned text yielded nothing, search raw text once.
    # Only standalone score lines are accepted so quoted/injected score text in
    # reasoning or candidate responses cannot be parsed as the verdict.
    raw_score_matches = list(_STANDALONE_SCORE_RE.finditer(text))
    if raw_score_matches:
        last_match = raw_score_matches[-1]
        labeled_score = int(last_match.group(1))
        if labeled_score in {-2, -1, 0, 1, 2}:
            return labeled_score, extracted_reasoning
    return None, extracted_reasoning


def combine_pairwise_judge_outputs(
    reference_first_output: str,
    candidate_first_output: str,
    *,
    require_both: bool = False,
    processing_class: Any | None = None,
) -> tuple[float | None, tuple[int | None, int | None], tuple[str | None, str | None]]:
    """Combine dual-order pairwise outputs into one candidate-centered score.

    Expected order:
    - `reference_first_output`: response1=reference, response2=candidate
    - `candidate_first_output`: response1=candidate, response2=reference

    The returned score is positive when the candidate is judged better than the
    reference and stays on the original pairwise -2..2 scale.
    """
    reference_first_score, reference_first_reasoning = parse_pairwise_judge_output(
        reference_first_output, processing_class=processing_class
    )
    candidate_first_score, candidate_first_reasoning = parse_pairwise_judge_output(
        candidate_first_output, processing_class=processing_class
    )

    normalized_scores: list[float] = []
    if reference_first_score is not None:
        normalized_scores.append(float(reference_first_score))
    if candidate_first_score is not None:
        normalized_scores.append(float(-candidate_first_score))

    if require_both and len(normalized_scores) != 2:
        combined_score = None
    elif normalized_scores:
        combined_score = sum(normalized_scores) / len(normalized_scores)
    else:
        combined_score = None

    return (
        combined_score,
        (reference_first_score, candidate_first_score),
        (reference_first_reasoning, candidate_first_reasoning),
    )


def build_privalign_rl_judge_prompt_token_ids_batch(
    *,
    examples: list[dict[str, Any]],
    judge_template: str,
    rollout_builder: Any,
    response1_batch: list[str],
    response2_batch: list[str],
    max_response_words: int | None = 1000,
) -> list[list[int]]:
    """Render and tokenize Privalign annotation-conditioned pairwise judge prompts."""
    judge_prompt_texts = render_privalign_rl_judge_prompt_texts(
        examples=examples,
        judge_template=judge_template,
        rollout_builder=rollout_builder,
        response1_batch=response1_batch,
        response2_batch=response2_batch,
        max_response_words=max_response_words,
    )
    return rollout_builder._tokenize_prompt_text_sequences(judge_prompt_texts)


def render_privalign_rl_judge_prompt_texts(
    *,
    examples: list[dict[str, Any]],
    judge_template: str,
    rollout_builder: Any,
    response1_batch: list[str],
    response2_batch: list[str],
    max_response_words: int | None = 1000,
) -> list[str]:
    """Render Privalign annotation-conditioned pairwise judge prompt text."""
    from data_loaders.preference import render_privalign_template

    if len(examples) != len(response1_batch) or len(examples) != len(response2_batch):
        raise ValueError("examples, response1_batch, and response2_batch must have the same length.")

    judge_messages: list[list[dict[str, str]]] = []
    for example, response1, response2 in zip(examples, response1_batch, response2_batch):
        demo = example.get("judge_demo")
        if not isinstance(demo, dict) or demo.get("demo_type") != "privalign_pairwise":
            raise ValueError(
                "render_privalign_rl_judge_prompt_texts requires each example "
                "to include a Privalign 'judge_demo' dict."
            )
        content = render_privalign_template(
            judge_template,
            judge_demo=demo,
            eval_response1=_truncate_judge_response(
                response1,
                max_words=max_response_words,
            ),
            eval_response2=_truncate_judge_response(
                response2,
                max_words=max_response_words,
            ),
        )
        judge_messages.append([{"role": "user", "content": content}])

    return [rollout_builder._format_prompt(messages) for messages in judge_messages]


def _truncate_judge_response(
    response: str,
    *,
    max_words: int | None,
) -> str:
    """Strip thinking traces and cap the response text shown to a pairwise judge."""
    stripped = strip_thinking_trace(response or "")
    if max_words is None or max_words <= 0:
        return stripped
    words = stripped.split()
    if len(words) <= max_words:
        return stripped
    omitted = len(words) - max_words
    return " ".join(words[:max_words]) + f"\n\n[Response truncated before judging; omitted {omitted} words.]"


def parse_pointwise_judge_output(
    text: str,
    *,
    processing_class: Any | None = None,
) -> tuple[bool | None, bool | None, str | None]:
    """Parse a pointwise judge output containing a leak/omit JSON object."""
    cleaned_text = parse_response_trace(text, processing_class)
    sources = [cleaned_text]
    if text and text != cleaned_text:
        sources.append(text)

    for source in sources:
        if not source or not source.strip():
            continue
        for candidate in _iter_pointwise_json_candidates(source):
            try:
                obj = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            if "leaks" not in obj or "omits" not in obj:
                continue
            leaks = _coerce_pointwise_bool(obj.get("leaks"))
            omits = _coerce_pointwise_bool(obj.get("omits"))
            if leaks is None or omits is None:
                continue
            reasoning = _coalesce_pointwise_explanations(obj)
            return leaks, omits, reasoning

    fallback_reasoning = cleaned_text.strip() or None
    return None, None, fallback_reasoning


_POINTWISE_BALANCED_JSON_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _iter_pointwise_json_candidates(text: str):
    """Yield JSON candidates, biased to the prompt's last-line contract."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        yield lines[-1]
    yield from reversed(_POINTWISE_BALANCED_JSON_RE.findall(text))


_POINTWISE_TRUE_TOKENS = frozenset({"yes", "true", "1"})
_POINTWISE_FALSE_TOKENS = frozenset({"no", "false", "0"})


def _coerce_pointwise_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _POINTWISE_TRUE_TOKENS:
            return True
        if token in _POINTWISE_FALSE_TOKENS:
            return False
    return None


def _coalesce_pointwise_explanations(obj: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for key in ("leaks_explanation", "omits_explanation", "reasoning"):
        value = obj.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{key}: {text}")
    return "; ".join(parts) if parts else None


def build_privalign_rl_pointwise_judge_prompt_token_ids_batch(
    *,
    examples: list[dict[str, Any]],
    judge_template: str,
    rollout_builder: Any,
    response_batch: list[str],
    max_response_words: int | None = 1000,
) -> list[list[int]]:
    """Render and tokenize Privalign single-response pointwise judge prompts."""
    judge_prompt_texts = render_privalign_rl_pointwise_judge_prompt_texts(
        examples=examples,
        judge_template=judge_template,
        rollout_builder=rollout_builder,
        response_batch=response_batch,
        max_response_words=max_response_words,
    )
    return rollout_builder._tokenize_prompt_text_sequences(judge_prompt_texts)


def render_privalign_rl_pointwise_judge_prompt_texts(
    *,
    examples: list[dict[str, Any]],
    judge_template: str,
    rollout_builder: Any,
    response_batch: list[str],
    max_response_words: int | None = 1000,
) -> list[str]:
    """Render Privalign single-response pointwise judge prompt text per row."""
    from data_loaders.preference import render_privalign_pointwise_eval_template

    if len(examples) != len(response_batch):
        raise ValueError("examples and response_batch must have the same length.")

    judge_messages: list[list[dict[str, str]]] = []
    for example, response in zip(examples, response_batch):
        demo = example.get("judge_demo")
        if not isinstance(demo, dict) or demo.get("demo_type") != "privalign_pairwise":
            raise ValueError(
                "render_privalign_rl_pointwise_judge_prompt_texts requires each example "
                "to include a Privalign 'judge_demo' dict."
            )
        content = render_privalign_pointwise_eval_template(
            judge_template,
            judge_demo=demo,
            new_response=_truncate_judge_response(response, max_words=max_response_words),
        )
        judge_messages.append([{"role": "user", "content": content}])

    return [rollout_builder._format_prompt(messages) for messages in judge_messages]
