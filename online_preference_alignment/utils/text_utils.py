"""Small text-normalization helpers shared across evaluation and rollout code."""

from __future__ import annotations

import re
from typing import Any


_GENERIC_TRAILING_SPECIAL_TOKEN_RE = re.compile(
    r"(?:"
    r"<\|[^<>]*\|>"
    r"|</s>"
    r"|<[/]?(?:s|eos|bos|pad|unk|sep|cls|mask)>"
    r"|<(?:end_of_turn|start_of_turn|turn|eot_id|im_end|endoftext|end_of_text)>"
    r"|<turn\|>"
    r"|\[/?(?:INST|AVAILABLE_TOOLS|TOOL_CALLS|TOOL_RESULTS)\]"
    r")\s*$",
    flags=re.IGNORECASE,
)


def parse_response_trace(text: str, processing_class: Any | None = None) -> str:
    """Strip a thinking trace from `text` using the official processor when possible.

    For gemma 4, `AutoProcessor.parse_response(text)` is the official way to
    separate the `<|channel>thought\\n...\\n<channel|>` reasoning from the answer
    (see https://huggingface.co/google/gemma-4-31B-it). We delegate to it when
    the processor exposes it, and fall back to the regex `strip_thinking_trace`
    otherwise (Qwen `<think>` blocks or processors without parse_response).

    Callers should decode with `skip_special_tokens=False` so the gemma
    separator tokens remain in `text` for the parser to find.
    """
    parser = getattr(processing_class, "parse_response", None) if processing_class is not None else None
    if callable(parser):
        try:
            parsed = parser(text)
        except Exception:
            parsed = None
        if isinstance(parsed, str):
            return strip_trailing_special_tokens(parsed, processing_class)
        if isinstance(parsed, dict):
            for key in ("response", "answer", "content"):
                value = parsed.get(key)
                if isinstance(value, str):
                    return strip_trailing_special_tokens(value, processing_class)
    return strip_thinking_trace(text, processing_class=processing_class)


def decode_response_tokens(
    processing_class: Any,
    token_ids: list[int],
    *,
    strip_thinking_traces: bool = True,
) -> str:
    """Decode generated response tokens into text for judges/logs/reward checks.

    The returned text should not contain tokenizer special tokens. When thinking
    stripping is enabled we do one raw internal decode to preserve model-specific
    thought delimiters, then remove any special-token strings from the extracted
    response. When thinking stripping is disabled we directly use
    ``skip_special_tokens=True``.
    """
    if not strip_thinking_traces:
        return processing_class.decode(token_ids, skip_special_tokens=True).strip()

    raw_text = processing_class.decode(token_ids, skip_special_tokens=False)
    response_text = parse_response_trace(raw_text, processing_class)
    raw_without_trailing_specials = strip_trailing_special_tokens(raw_text, processing_class)
    if response_text != raw_without_trailing_specials:
        return strip_special_tokens_from_text(response_text, processing_class)

    # No thinking trace was removed, so use the principled tokenizer decode for
    # the final response text instead of carrying special-token strings forward.
    special_free_text = processing_class.decode(token_ids, skip_special_tokens=True)
    return strip_trailing_special_tokens(special_free_text, processing_class)


def strip_thinking_trace(text: str, processing_class: Any | None = None) -> str:
    """Remove leading thinking traces emitted by thinking models.

    Handles three formats:
    - Qwen: ``<think>...</think>`` blocks (including partial/orphaned).
    - Ministral: ``[THINK]...[/THINK]`` blocks (including partial/orphaned).
    - Gemma 4: ``<|channel>thought\\n...\\n<channel|>`` blocks.

    Some chat templates absorb the opening token into the prompt, so
    the generated completion can begin mid-trace (for example
    ``"reasoning...</think>\\nAnswer"``). Treat those partial leading traces the
    same as a normal block.
    """
    stripped_text = text.strip()
    while True:
        # Qwen: full <think>...</think> block
        full_block_match = re.match(r"^<think>.*?</think>\s*", stripped_text, flags=re.DOTALL)
        if full_block_match is not None:
            stripped_text = stripped_text[full_block_match.end() :].lstrip()
            continue

        # Qwen: orphaned </think> at the start
        orphaned_close_match = re.match(r"^</think>\s*", stripped_text, flags=re.DOTALL)
        if orphaned_close_match is not None:
            stripped_text = stripped_text[orphaned_close_match.end() :].lstrip()
            continue

        # Qwen: partial leading trace ending with </think>
        partial_block_match = re.match(r"^(?:(?!</think>).)+</think>\s*", stripped_text, flags=re.DOTALL)
        if partial_block_match is not None:
            stripped_text = stripped_text[partial_block_match.end() :].lstrip()
            continue

        # Ministral: full [THINK]...[/THINK] block
        mistral_block_match = re.match(r"^\[THINK\].*?\[/THINK\]\s*", stripped_text, flags=re.DOTALL)
        if mistral_block_match is not None:
            stripped_text = stripped_text[mistral_block_match.end() :].lstrip()
            continue

        # Ministral: orphaned [/THINK] at the start
        mistral_orphan_match = re.match(r"^\[/THINK\]\s*", stripped_text, flags=re.DOTALL)
        if mistral_orphan_match is not None:
            stripped_text = stripped_text[mistral_orphan_match.end() :].lstrip()
            continue

        # Ministral: partial leading trace ending with [/THINK]
        mistral_partial_match = re.match(r"^(?:(?!\[/THINK\]).)+\[/THINK\]\s*", stripped_text, flags=re.DOTALL)
        if mistral_partial_match is not None:
            stripped_text = stripped_text[mistral_partial_match.end() :].lstrip()
            continue

        # Gemma 4: full <|channel>thought\n...\n<channel|> block
        gemma_block_match = re.match(
            r"^<\|channel>thought\n.*?\n<channel\|>\s*", stripped_text, flags=re.DOTALL,
        )
        if gemma_block_match is not None:
            stripped_text = stripped_text[gemma_block_match.end() :].lstrip()
            continue

        # Gemma 4: orphaned <channel|> at the start (partial trace)
        gemma_orphan_match = re.match(r"^<channel\|>\s*", stripped_text, flags=re.DOTALL)
        if gemma_orphan_match is not None:
            stripped_text = stripped_text[gemma_orphan_match.end() :].lstrip()
            continue

        # Gemma 4: partial leading trace ending with <channel|>
        gemma_partial_match = re.match(
            r"^(?:(?!<channel\|>).)+<channel\|>\s*", stripped_text, flags=re.DOTALL,
        )
        if gemma_partial_match is not None:
            stripped_text = stripped_text[gemma_partial_match.end() :].lstrip()
            continue

        # Teacher-emitted revised reasoning/response tags.
        # Can appear at start (before response) or end (after response).
        revised_thinking_match = re.match(
            r"^<revised_(?:thinking|response)>.*?</revised_(?:thinking|response)>\s*",
            stripped_text,
            flags=re.DOTALL,
        )
        if revised_thinking_match is not None:
            stripped_text = stripped_text[revised_thinking_match.end() :].lstrip()
            continue

        # Strip trailing <revised_thinking> or <revised_response> block.
        trailing_rt_match = re.search(
            r"\s*<revised_(?:thinking|response)>.*?</revised_(?:thinking|response)>\s*$",
            stripped_text,
            flags=re.DOTALL,
        )
        if trailing_rt_match is not None:
            stripped_text = stripped_text[: trailing_rt_match.start()].rstrip()
            continue

        return strip_trailing_special_tokens(stripped_text, processing_class)


def strip_trailing_special_tokens(
    text: str,
    processing_class: Any | None = None,
) -> str:
    """Remove trailing chat/eos special tokens from decoded text.

    Prefer model-declared tokenizer/processor specials when available, and use
    a conservative fallback for common chat-template sentinel forms.
    """
    stripped_text = text.rstrip()
    special_tokens = _processing_class_special_tokens(processing_class)

    while stripped_text:
        before = stripped_text
        for token in special_tokens:
            if stripped_text.endswith(token):
                stripped_text = stripped_text[: -len(token)].rstrip()
                break
        else:
            stripped_text = _GENERIC_TRAILING_SPECIAL_TOKEN_RE.sub("", stripped_text).rstrip()

        if stripped_text == before:
            return stripped_text
    return stripped_text


def strip_special_tokens_from_text(
    text: str,
    processing_class: Any | None = None,
) -> str:
    """Remove tokenizer-declared special token strings from already-decoded text."""
    stripped_text = text
    for token in _processing_class_special_tokens(processing_class):
        stripped_text = stripped_text.replace(token, "")
    return strip_trailing_special_tokens(stripped_text, processing_class)


def _processing_class_special_tokens(processing_class: Any | None) -> list[str]:
    if processing_class is None:
        return []
    tokens: list[str] = []
    for attr_name in (
        "all_special_tokens",
        "additional_special_tokens",
    ):
        value = getattr(processing_class, attr_name, None)
        if isinstance(value, (list, tuple)):
            tokens.extend(token for token in value if isinstance(token, str) and token)
    for attr_name in (
        "eos_token",
        "bos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
        "mask_token",
    ):
        value = getattr(processing_class, attr_name, None)
        if isinstance(value, str) and value:
            tokens.append(value)

    # Longest first so a composite token is removed before any prefix/suffix.
    return sorted(set(tokens), key=len, reverse=True)


def find_token_subsequence(
    sequence: list[int],
    subsequence: list[int],
) -> int | None:
    """Return the first start index of ``subsequence`` inside ``sequence``."""
    if not subsequence:
        return 0
    if len(subsequence) > len(sequence):
        return None
    last_start = len(sequence) - len(subsequence)
    for start in range(last_start + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None


def find_first_token_subsequence(
    sequence: list[int],
    subsequences: list[list[int]],
) -> tuple[int, int] | None:
    """Return ``(start, length)`` for the earliest matching subsequence."""
    best: tuple[int, int] | None = None
    for subsequence in subsequences:
        if not subsequence:
            continue
        start = find_token_subsequence(sequence, subsequence)
        if start is None:
            continue
        candidate = (start, len(subsequence))
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def normalize_thinking_close_token_ids(
    think_close_token_ids: list[int] | list[list[int]],
) -> list[list[int]]:
    """Normalize one or more thinking-close token-id sequences."""
    if not think_close_token_ids:
        return []
    first = think_close_token_ids[0]
    if isinstance(first, int):
        return [list(think_close_token_ids)]  # type: ignore[list-item]
    return [
        list(sequence)
        for sequence in think_close_token_ids  # type: ignore[assignment]
        if sequence
    ]


def build_response_token_mask(
    completion_sequence: list[int],
    *,
    thinking_enabled: bool,
    think_close_token_ids: list[int] | list[list[int]],
) -> list[bool]:
    """Return the response-phase mask for one completion sequence."""
    if not completion_sequence:
        return []
    if not thinking_enabled:
        return [True] * len(completion_sequence)
    close_match = find_first_token_subsequence(
        completion_sequence,
        normalize_thinking_close_token_ids(think_close_token_ids),
    )
    if close_match is None:
        return [False] * len(completion_sequence)
    close_index, close_length = close_match
    response_start = close_index + close_length
    return [position >= response_start for position in range(len(completion_sequence))]
