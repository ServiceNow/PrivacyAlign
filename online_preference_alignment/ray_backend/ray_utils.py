from __future__ import annotations

import logging
import time
from typing import Any

import ray


logger = logging.getLogger(__name__)


def wait_for_ray_refs(
    refs: list[Any],
    *,
    description: str,
    poll_interval_s: float = 300.0,
) -> list[Any]:
    if not refs:
        return []

    pending = list(refs)
    completed = 0
    total = len(pending)
    start_time = time.monotonic()
    results = []

    logger.info("Waiting for %s (%s task%s)...", description, total, "" if total == 1 else "s")
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=poll_interval_s)
        if not ready:
            logger.info(
                "Still waiting for %s (%s/%s completed, %.1fs elapsed)...",
                description,
                completed,
                total,
                time.monotonic() - start_time,
            )
            continue

        ref = ready[0]
        results.append(ray.get(ref))
        completed += 1
        logger.info(
            "%s progress: %s/%s completed (%.1fs elapsed)",
            description,
            completed,
            total,
            time.monotonic() - start_time,
        )

    return results


def wait_for_ray_ref(
    ref: Any,
    *,
    description: str,
    poll_interval_s: float = 300.0,
) -> Any:
    return wait_for_ray_refs(
        [ref],
        description=description,
        poll_interval_s=poll_interval_s,
    )[0]
