from __future__ import annotations

import copy
import json
from typing import Any


def _normalize_zero_optimization_config(config: dict[str, Any]) -> dict[str, Any]:
    zero_config = dict(config.get("zero_optimization", {}))
    try:
        stage = int(zero_config.get("stage", 0))
    except (TypeError, ValueError):
        stage = 0
    zero_config["stage"] = stage

    offload_param = dict(zero_config.get("offload_param", {}))
    offload_param.setdefault("device", "none")
    if offload_param.get("device", "none") == "cpu":
        offload_param.setdefault("pin_memory", True)
    zero_config["offload_param"] = offload_param

    offload_optimizer = dict(zero_config.get("offload_optimizer", {}))
    offload_optimizer.setdefault("device", "none")
    if offload_optimizer.get("device", "none") == "cpu":
        offload_optimizer.setdefault("pin_memory", True)
    zero_config["offload_optimizer"] = offload_optimizer

    if stage == 3:
        zero_config.setdefault("sub_group_size", "auto")
        zero_config.setdefault("stage3_max_live_parameters", "auto")
        zero_config.setdefault("stage3_max_reuse_distance", "auto")
        zero_config.setdefault("stage3_param_persistence_threshold", "auto")
        zero_config.setdefault("stage3_prefetch_bucket_size", "auto")
        zero_config.setdefault("reduce_bucket_size", "auto")
        zero_config.setdefault("reduce_scatter", True)
        zero_config["stage3_gather_16bit_weights_on_model_save"] = True

    return zero_config


def load_deepspeed_config_dict(
    config_path: str,
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    max_grad_norm: float,
    bf16: bool,
    learning_rate: float,
    weight_decay: float,
    deepcompile: bool = False,
    offload_optimizer: bool | None = None,
    offload_param: bool | None = None,
) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    config["train_batch_size"] = per_device_train_batch_size * gradient_accumulation_steps * world_size
    config["gradient_clipping"] = max_grad_norm
    config["zero_optimization"] = _normalize_zero_optimization_config(config)
    zero_config = config["zero_optimization"]

    if offload_optimizer is not None:
        zero_config["offload_optimizer"]["device"] = "cpu" if offload_optimizer else "none"
        if offload_optimizer:
            zero_config["offload_optimizer"]["pin_memory"] = True

    if offload_param is not None:
        zero_config["offload_param"]["device"] = "cpu" if offload_param else "none"
        if offload_param:
            zero_config["offload_param"]["pin_memory"] = True

    if "bf16" in config:
        config["bf16"]["enabled"] = bf16
    config.pop("fp16", None)

    if "optimizer" in config:
        opt_params = config["optimizer"].get("params", {})
        if opt_params.get("lr") == "auto":
            opt_params["lr"] = learning_rate
        if opt_params.get("weight_decay") == "auto":
            opt_params["weight_decay"] = weight_decay
        if zero_config["offload_optimizer"]["device"] == "cpu":
            opt_params.pop("torch_adam", None)

    compile_config = dict(config.get("compile", {}))
    compile_config["deepcompile"] = deepcompile
    config["compile"] = compile_config

    return config


def get_optimizer_grouped_parameters(
    model: Any,
    weight_decay: float,
    no_decay_name_list: list[str] | None = None,
) -> list[dict[str, Any]]:
    if no_decay_name_list is None:
        no_decay_name_list = [
            "bias",
            "layer_norm.weight",
            "layernorm.weight",
            "norm.weight",
            "ln_f.weight",
        ]

    return [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and not any(token in name for token in no_decay_name_list)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and any(token in name for token in no_decay_name_list)
            ],
            "weight_decay": 0.0,
        },
    ]


def build_deepspeed_optimizer(model: Any, config: dict[str, Any]) -> Any | None:
    optimizer_config = config.get("optimizer")
    if not optimizer_config:
        return None

    optimizer_params = dict(optimizer_config.get("params", {}))
    learning_rate = float(optimizer_params.get("lr", 0.0))
    betas = tuple(float(beta) for beta in optimizer_params.get("betas", (0.9, 0.999)))
    epsilon = float(optimizer_params.get("eps", 1e-8))
    weight_decay = float(optimizer_params.get("weight_decay", 0.0))
    grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)

    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

    optimizer_cls = (
        DeepSpeedCPUAdam
        if config.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none") == "cpu"
        else FusedAdam
    )
    return optimizer_cls(
        grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=epsilon,
        weight_decay=weight_decay,
    )


def strip_optimizer_from_config(config: dict[str, Any]) -> dict[str, Any]:
    stripped_config = copy.deepcopy(config)
    stripped_config.pop("optimizer", None)
    return stripped_config


def get_eval_ds_config(
    *,
    stage: int = 3,
    offload: bool = False,
    bf16: bool = True,
) -> dict[str, Any]:
    """Return a minimal DeepSpeed config for a frozen eval-only model."""
    config = {
        "steps_per_print": 100,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": stage,
            "stage3_max_live_parameters": "auto",
            "stage3_max_reuse_distance": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "offload_param": {
                "device": "cpu" if offload else "none",
                "pin_memory": True,
            },
        },
        "bf16": {"enabled": bf16},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    if stage == 3:
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
    return config


def build_hf_deepspeed_config(deepspeed_config: dict[str, Any]) -> Any | None:
    if deepspeed_config.get("zero_optimization", {}).get("stage", 0) != 3:
        return None

    from transformers.integrations import HfDeepSpeedConfig

    return HfDeepSpeedConfig(deepspeed_config)
