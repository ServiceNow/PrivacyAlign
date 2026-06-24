#!/usr/bin/env python3
"""
Post-generation rule-based filter for link-passthrough leakage.

Reads the JSON output of generate_data.py, removes samples whose leakage is
solely caused by forwarding a pre-existing document link, and writes accepted
samples back.  Rejected samples are logged to a JSONL file.

Usage (after generate_data.py):
    python filter_link_passthrough.py --input-path outputs/main_data_generated_gpt_oss_120b.json

By default the output is written to ``<input_stem>_link_filtered.<ext>``
alongside the input file.  Use --output-path to override.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.link_passthrough_filter import check_sample
from pipeline.utils import append_jsonl, write_json_atomic

LOGGER = logging.getLogger("link_passthrough_filter")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "outputs" / "main_data_generated.json"
DEFAULT_REJECTED_PATH = SCRIPT_DIR / "outputs" / "link_passthrough_rejected.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter generated samples to remove link-passthrough leakage.",
    )

    io = parser.add_argument_group("Input / output")
    io.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the generated JSON file (output of generate_data.py).",
    )
    io.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to write accepted samples (default: <input>_link_filtered.json).",
    )
    io.add_argument(
        "--rejected-path",
        type=str,
        default=str(DEFAULT_REJECTED_PATH),
        help="JSONL file to log rejected samples.",
    )
    io.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which samples would be rejected without writing any files.",
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        stem = input_path.stem + "_link_filtered"
        output_path = input_path.parent / (stem + input_path.suffix)
    rejected_path = Path(args.rejected_path)

    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as fh:
        samples: List[Dict[str, Any]] = json.load(fh)

    if not isinstance(samples, list):
        LOGGER.error("Expected a JSON array in %s", input_path)
        raise SystemExit(1)

    LOGGER.info("Loaded %d samples from %s", len(samples), input_path)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for sample in samples:
        if check_sample(sample):
            rejected.append(sample)
            LOGGER.warning("Rejected: %s", sample.get("name", "?"))
        else:
            accepted.append(sample)

    if args.dry_run:
        LOGGER.info(
            "Dry run complete. Would accept=%d  reject=%d",
            len(accepted), len(rejected),
        )
        for r in rejected:
            LOGGER.info("  would reject: %s", r.get("name", "?"))
        return

    for r in rejected:
        append_jsonl(rejected_path, {
            "name": r.get("name", "?"),
            "error": "Filtered by link-passthrough rule",
        })

    write_json_atomic(output_path, accepted)

    LOGGER.info(
        "Done. accepted=%d  rejected=%d  output=%s",
        len(accepted), len(rejected), output_path,
    )


if __name__ == "__main__":
    main()
