from __future__ import annotations

import importlib
import logging
from types import ModuleType


logger = logging.getLogger(__name__)


_NEMOTRON_LOCAL_KERNEL_MODELS = (
    "nvidia/nvidia-nemotron-3-nano-4b-bf16",
)


def _is_nemotron_local_kernel_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    normalized = str(model_name).strip().lower()
    return any(token in normalized for token in _NEMOTRON_LOCAL_KERNEL_MODELS)


def _import_optional_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logger.warning(
            "Could not import local kernel module %s for Transformers hub-kernel override: %s",
            module_name,
            exc,
        )
        return None


def force_local_transformers_hub_kernels_for_model(model_name: str | None) -> None:
    """Make Nemotron 3 Nano BF16 use local Mamba kernels instead of Hub kernels.

    Some Transformers/Nemotron configurations resolve Mamba2 and causal-conv1d
    through ``transformers.integrations.hub_kernels``. In offline or restricted
    jobs that resolution can fail even when compatible local packages are
    installed, leaving Nemotron on its massive naive ``torch_forward`` fallback.
    """
    if not _is_nemotron_local_kernel_model(model_name):
        return

    try:
        from transformers.integrations.hub_kernels import _KERNEL_MODULE_MAPPING
    except Exception as exc:
        logger.info(
            "Transformers hub-kernel mapping is unavailable; local kernel override skipped: %s",
            exc,
        )
        return

    modules = {
        "mamba-ssm": _import_optional_module("mamba_ssm"),
        "causal-conv1d": _import_optional_module("causal_conv1d"),
    }
    applied = []
    for kernel_name, module in modules.items():
        if module is None:
            continue
        _KERNEL_MODULE_MAPPING[kernel_name] = module
        applied.append(kernel_name)

    if applied:
        logger.info(
            "Forced Transformers hub-kernel mapping to local modules for model=%s: %s",
            model_name,
            ", ".join(applied),
        )
