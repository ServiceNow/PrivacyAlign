#!/usr/bin/env python3
"""
Post-generation embedding diversity filter.

Reads the JSON output of generate_data.py, removes near-semantic duplicates
using cosine similarity, and writes accepted samples back.  Rejected samples
are logged to a JSONL file.

Usage (after generate_data.py):
    python filter_embedding_diversity.py --input-path outputs/main_data_generated.json

The script overwrites the input file with only the accepted samples by default.
Use --output-path to write to a different file instead.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.embedding_diversity import EmbeddingDiversityFilter
from pipeline.utils import append_jsonl, write_json_atomic

LOGGER = logging.getLogger("embedding_diversity_filter")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "outputs" / "main_data_generated.json"
DEFAULT_REJECTED_PATH = SCRIPT_DIR / "outputs" / "embedding_diversity_rejected.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter generated samples by removing near-semantic duplicates.",
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
        help="Path to write accepted samples. Defaults to overwriting --input-path.",
    )
    io.add_argument(
        "--rejected-path",
        type=str,
        default=str(DEFAULT_REJECTED_PATH),
        help="JSONL file to log rejected samples.",
    )

    emb = parser.add_argument_group("Embedding model")
    emb.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="HuggingFace embedding model for diversity scoring.",
    )
    emb.add_argument(
        "--embedding-device",
        type=str,
        default="cuda",
        help="Device for embedding model (cuda, cpu, etc.).",
    )
    emb.add_argument(
        "--embed-batch-size",
        type=int,
        default=1,
        help="Batch size for embedding model forward passes.",
    )
    emb.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Max token length for embedding input. Lower to reduce GPU memory (e.g. 8192).",
    )
    emb.add_argument(
        "--max-similarity",
        type=float,
        default=0.95,
        help="Max cosine similarity to any accepted sample before rejection (default 0.95). "
             "Lower = stricter deduplication, higher = more lenient.",
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
    output_path = Path(args.output_path) if args.output_path else input_path
    rejected_path = Path(args.rejected_path)

    # Load samples
    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as fh:
        samples: List[Dict[str, Any]] = json.load(fh)

    if not isinstance(samples, list):
        LOGGER.error("Expected a JSON array in %s", input_path)
        raise SystemExit(1)

    LOGGER.info("Loaded %d samples from %s", len(samples), input_path)

    if not samples:
        LOGGER.info("No samples to filter.")
        write_json_atomic(output_path, [])
        return

    # Run filter
    embedding_filter = EmbeddingDiversityFilter(
        model_name=args.embedding_model,
        device=args.embedding_device,
        embed_batch_size=args.embed_batch_size,
        max_length=args.max_length,
        max_similarity=args.max_similarity,
    )

    accepted, rejected = embedding_filter.filter_batch(samples)

    # Write rejected
    for r in rejected:
        append_jsonl(rejected_path, {"name": r.get("name", "?"), "error": "Filtered by embedding diversity"})

    # Write accepted
    write_json_atomic(output_path, accepted)

    LOGGER.info(
        "Done. accepted=%d  rejected=%d  output=%s",
        len(accepted), len(rejected), output_path,
    )
    LOGGER.info("Rejected samples logged to %s", rejected_path)


if __name__ == "__main__":
    main()
