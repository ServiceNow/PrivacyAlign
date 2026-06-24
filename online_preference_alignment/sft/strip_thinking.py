"""Strip model thinking blocks from teacher generations.

The teacher prompt explicitly says "Output only the edited response", but Qwen3
in thinking mode still emits a leading <think>...</think> block. The SFT target
is the text *after* the last thinking close marker; if none is present, the row
is discarded so we don't poison the SFT dataset with mid-thought truncations.
"""

from __future__ import annotations

THINK_CLOSE_MARKERS = ("</think>", "[/THINK]")


def strip_thinking(raw_text: str) -> str | None:
    """Return the tail after the final thinking close marker.

    Returns None if no closing marker is present (caller should discard the row).
    """
    if not isinstance(raw_text, str):
        return None
    marker, idx = max(
        ((marker, raw_text.rfind(marker)) for marker in THINK_CLOSE_MARKERS),
        key=lambda item: item[1],
    )
    if idx < 0:
        return None
    tail = raw_text[idx + len(marker):].strip()
    if not tail:
        return None
    return tail
