"""Stateless helpers shared across trainer subsystems."""

from __future__ import annotations

from collections.abc import Iterator
import random
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from training.full_vocab_kl import chunked_logsumexp


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_dropout_in_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DeepSpeed / DDP wrappers from a model."""
    if hasattr(model, "module"):
        return model.module
    return model


def selective_log_softmax(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute target-token log-probs with float32 normalization for numerical stability."""
    token_logits = torch.gather(logits, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).float()
    return token_logits - chunked_logsumexp(logits)


def is_distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def torch_dist_barrier_and_cuda_sync(device: torch.device, group: Any | None = None) -> None:
    if is_distributed_ready():
        if group is None:
            dist.barrier()
        else:
            dist.barrier(group=group)
    if device.type == "cuda":
        torch.cuda.synchronize()


def normalize_vllm_param_name(name: str) -> str:
    """Strip checkpoint wrapper and DDP module prefixes from parameter names."""
    return name.replace("_checkpoint_wrapped_module.", "").replace("module.", "")


def _synchronize_cuda_param(param: torch.Tensor) -> None:
    device = getattr(param, "device", None)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def iter_deepspeed_weight_sync_params(
    model: nn.Module,
    *,
    zero_stage: int,
) -> Iterator[tuple[int, int, str, torch.Tensor]]:
    named_parameters = list(model.named_parameters())
    num_params = len(named_parameters)
    if zero_stage == 3:
        import deepspeed

        gather_context = deepspeed.zero.GatheredParameters
        try:
            from deepspeed.utils import safe_get_full_fp32_param
        except Exception:
            safe_get_full_fp32_param = None
    else:
        gather_context = None
        safe_get_full_fp32_param = None

    for count, (name, param) in enumerate(named_parameters, start=1):
        if gather_context is None:
            yield count, num_params, name, param
            continue
        full_param = None
        if safe_get_full_fp32_param is not None:
            full_param = safe_get_full_fp32_param(param)
        if full_param is not None:
            target_device = getattr(param, "device", None)
            target_dtype = getattr(param, "dtype", None)
            if target_device is not None and full_param.device != target_device:
                full_param = full_param.to(device=target_device)
            if target_dtype is not None and full_param.dtype != target_dtype:
                full_param = full_param.to(dtype=target_dtype)
            if not full_param.is_contiguous():
                full_param = full_param.contiguous()
            try:
                yield count, num_params, name, full_param
            finally:
                # The consumer may hand the gathered tensor to CUDA/NCCL/IPC
                # work. Drain that work before releasing the temporary full
                # tensor assembled from DeepSpeed's fp32 master weights.
                _synchronize_cuda_param(full_param)
                del full_param
            continue
        with gather_context([param]):
            # ZeRO-3 all-gather may enqueue CUDA work before the parameter's
            # full storage is safe for CUDA IPC/NCCL consumers. Make the
            # gathered view material before yielding it to weight sync.
            _synchronize_cuda_param(param)
            try:
                yield count, num_params, name, param
            finally:
                # The consumer may hand the gathered tensor to CUDA/NCCL/IPC
                # work. Drain that work before ZeRO-3 repartitions the param.
                _synchronize_cuda_param(param)


def _tokenizer_model_name(tokenizer: Any) -> str | None:
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        return name
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        name = init_kwargs.get("name_or_path")
        if isinstance(name, str) and name:
            return name
    return None


def _chat_template_supports_enable_thinking(tokenizer: Any) -> bool:
    model_name = _tokenizer_model_name(tokenizer)
    if model_name is None:
        return False
    normalized_model_name = model_name.lower()
    return "qwen" in normalized_model_name or "gemma" in normalized_model_name


def student_thinking_enabled(config: Any) -> bool:
    """Return whether student-side thinking-mode formatting/parsing is enabled."""
    if bool(getattr(config, "disable_student_thinking", False)):
        return False
    return bool(getattr(config, "qwen_enable_thinking", False))


def append_last_user_instruction(
    prompt: Any,
    *,
    instruction: str | None,
) -> list[dict[str, Any]]:
    """Return a prompt copy with ``instruction`` appended to the last user turn."""
    if not isinstance(prompt, list):
        raise TypeError("Prompt must be a list of chat messages.")
    if not all(
        isinstance(message, dict) and "role" in message and "content" in message
        for message in prompt
    ):
        raise TypeError("Prompt must be a list of {'role', 'content'} chat messages.")

    prompt_messages = list(prompt)
    if not instruction:
        return prompt_messages

    for index in range(len(prompt_messages) - 1, -1, -1):
        message = prompt_messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        stripped_content = content.rstrip()
        if stripped_content.endswith(instruction):
            return prompt_messages
        updated_content = instruction if not stripped_content else f"{stripped_content}\n\n{instruction}"
        prompt_messages[index] = {
            **message,
            "content": updated_content,
        }
        return prompt_messages
    return prompt_messages


def format_prompt_text(
    prompt: Any,
    tokenizer: Any,
    *,
    enable_thinking: bool = False,
    last_user_instruction: str | None = None,
) -> str:
    """Render a chat prompt to text using the tokenizer's chat template."""
    prompt = append_last_user_instruction(prompt, instruction=last_user_instruction)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("Tokenizer must implement apply_chat_template().")

    chat_template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _chat_template_supports_enable_thinking(tokenizer):
        # Qwen/Gemma chat templates can default to thinking mode unless the
        # flag is explicitly overridden, so pass the resolved boolean through.
        chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
    return tokenizer.apply_chat_template(prompt, **chat_template_kwargs)
