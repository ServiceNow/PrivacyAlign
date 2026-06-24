from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DATASET_REGISTRY: dict[str, object] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template from prompts/{name}.txt."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text()


def register_dataset(name: str):
    """Decorator that registers a dataset loader function by name."""
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def get_hf_load_dataset():
    """Import and return ``datasets.load_dataset``."""
    from datasets import load_dataset

    return load_dataset


def get_dataset(name: str, **kwargs):
    """Look up a registered dataset loader by name and call it."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset type '{name}'. Available: {available}")
    return DATASET_REGISTRY[name](**kwargs)


# Register built-in dataset loaders by importing their modules.
import data_loaders.preference  # noqa: F401, E402
