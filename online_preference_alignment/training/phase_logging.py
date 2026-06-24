"""Phase-progress logging infrastructure for the trainer stack.

Extracted from ``training.loss`` to keep loss computation focused on
objective math.  All trainer subsystems inherit this mixin for structured
phase start/end/event logging behind the ``log_phase_progress`` flag.
"""

from __future__ import annotations

import logging
import time
from typing import Any


logger = logging.getLogger(__name__)


class TrainerPhaseLoggingMixin:
    """Structured phase-progress logging used by all trainer subsystems."""

    @staticmethod
    def _format_phase_details(details: dict[str, Any] | None) -> str:
        if not details:
            return ""
        items = []
        for key, value in details.items():
            if value is None:
                continue
            items.append(f"{key}={value}")
        return f" {' '.join(items)}" if items else ""

    def _phase_log_prefix(
        self,
        *,
        stage: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
    ) -> str:
        parts = [f"stage={stage}"]
        parts.append(f"step={self.global_step + 1}" if stage == "train" else f"step={self.global_step}")
        if micro_batch_index is not None and num_micro_batches is not None:
            parts.append(f"micro_batch={micro_batch_index}/{num_micro_batches}")
        return " ".join(parts)

    def _log_phase_start(
        self,
        *,
        stage: str,
        phase: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
        details: dict[str, Any] | None = None,
    ) -> float | None:
        if not self.args.log_phase_progress:
            return None
        logger.info(
            "%s phase=%s status=start%s",
            self._phase_log_prefix(
                stage=stage, micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            ),
            phase,
            self._format_phase_details(details),
        )
        return time.perf_counter()

    def _log_phase_end(
        self,
        *,
        stage: str,
        phase: str,
        micro_batch_index: int | None,
        num_micro_batches: int | None,
        start_time: float | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if start_time is None:
            return
        logger.info(
            "%s phase=%s status=done duration_seconds=%.4f%s",
            self._phase_log_prefix(
                stage=stage, micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            ),
            phase,
            time.perf_counter() - start_time,
            self._format_phase_details(details),
        )

    def _log_phase_event(
        self,
        *,
        stage: str,
        phase: str,
        status: str,
        micro_batch_index: int | None = None,
        num_micro_batches: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.args.log_phase_progress:
            return
        logger.info(
            "%s phase=%s status=%s%s",
            self._phase_log_prefix(
                stage=stage, micro_batch_index=micro_batch_index,
                num_micro_batches=num_micro_batches,
            ),
            phase,
            status,
            self._format_phase_details(details),
        )
