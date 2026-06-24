"""Model forward-pass infrastructure for the trainer stack.

This mixin owns all low-level forward-pass helpers: backbone/lm-head splitting,
hidden-state extraction, chunked lm-head operations, and the higher-level
subbatched wrappers that policy objectives call.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from training.full_vocab_kl import compute_full_vocab_kl_and_logps_per_token
from utils.trainer_utils import selective_log_softmax


# ------------------------------------------------------------------
# Activation-checkpoint-friendly lm-head chunk functions
# ------------------------------------------------------------------

def _selective_log_softmax_chunk_fn(
    lm_head: nn.Module,
    hidden_chunk: torch.Tensor,
    token_ids_chunk: torch.Tensor,
) -> torch.Tensor:
    logits = lm_head(hidden_chunk)
    return selective_log_softmax(logits, token_ids_chunk)


def _full_vocab_reverse_kl_and_logps_chunk_fn(
    student_lm_head: nn.Module,
    teacher_lm_head: nn.Module,
    student_hidden_chunk: torch.Tensor,
    teacher_hidden_chunk: torch.Tensor,
    token_ids_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_logits = student_lm_head(student_hidden_chunk)
    with torch.no_grad():
        teacher_logits = teacher_lm_head(teacher_hidden_chunk)
    return compute_full_vocab_kl_and_logps_per_token(
        student_logits,
        teacher_logits,
        token_ids_chunk,
    )


class TrainerForwardMixin:
    """Model forward-pass infrastructure used by the loss and objective layers."""

    # ------------------------------------------------------------------
    # Config readers
    # ------------------------------------------------------------------

    def _lm_head_chunk_size(self) -> int | None:
        chunk_size = getattr(self.args, "lm_head_chunk_size", None)
        if chunk_size is None:
            return None
        return int(chunk_size)

    @staticmethod
    def _is_deepspeed_engine(model: nn.Module) -> bool:
        return (
            hasattr(model, "module")
            and callable(getattr(model, "backward", None))
            and callable(getattr(model, "step", None))
        )

    def _should_route_hidden_states_through_model_forward(self, model: nn.Module) -> bool:
        zero_stage_fn = getattr(self, "_deepspeed_zero_stage", None)
        if not callable(zero_stage_fn):
            return False
        return self._is_deepspeed_engine(model) and int(zero_stage_fn()) == 3

    @staticmethod
    def _extract_last_hidden_state(module_output: Any) -> torch.Tensor:
        if isinstance(module_output, torch.Tensor):
            return module_output
        last_hidden_state = getattr(module_output, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return last_hidden_state
        if isinstance(module_output, (tuple, list)) and module_output:
            first_item = module_output[0]
            if isinstance(first_item, torch.Tensor):
                return first_item
        raise TypeError("Expected a module output carrying a tensor last hidden state.")

    def _get_hidden_states_via_model_forward(
        self,
        model: nn.Module,
        backbone: nn.Module,
        forward_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        captured_hidden_states: list[torch.Tensor] = []

        def _capture_backbone_output(_module, _args, output):
            captured_hidden_states.append(self._extract_last_hidden_state(output))

        hook_handle = backbone.register_forward_hook(_capture_backbone_output)
        try:
            model(**forward_kwargs)
        finally:
            hook_handle.remove()

        if not captured_hidden_states:
            raise RuntimeError("Expected backbone forward hook to capture hidden states from model forward.")
        return captured_hidden_states[-1]

    # ------------------------------------------------------------------
    # Backbone / lm-head splitting
    # ------------------------------------------------------------------

    def _split_causal_lm_backbone_and_head(
        self,
        model: nn.Module,
    ) -> tuple[nn.Module, nn.Module]:
        causal_lm = self._unwrap_model(model)
        backbone_name = getattr(causal_lm, "base_model_prefix", None)
        backbone = getattr(causal_lm, backbone_name, None) if isinstance(backbone_name, str) else None
        get_output_embeddings = getattr(causal_lm, "get_output_embeddings", None)
        lm_head = get_output_embeddings() if callable(get_output_embeddings) else getattr(causal_lm, "lm_head", None)
        if not isinstance(backbone, nn.Module) or not isinstance(lm_head, nn.Module):
            raise TypeError(
                "lm_head chunking requires a causal LM with an accessible backbone module "
                "and output embedding head."
            )
        return backbone, lm_head

    # ------------------------------------------------------------------
    # Hidden-state and logit extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_right_padded_completion(
        attention_mask: torch.Tensor, completion_length: int,
    ) -> None:
        # `_get_completion_*` slices `[:, -completion_length:]` on the assumption that the
        # trailing `completion_length` columns are the right-padded completion section.
        # If completion padding ever flips to left-padded, that slice silently extracts
        # garbage. Verify the mask is non-increasing per row within the trailing window.
        if completion_length <= 1 or attention_mask.ndim != 2:
            return
        tail = attention_mask[:, -completion_length:]
        if not bool((tail[:, :-1] >= tail[:, 1:]).all()):
            raise ValueError(
                "_get_completion_* requires the completion section to be right-padded so the "
                "last `completion_length` columns align with completion positions; "
                "attention_mask is non-monotonic within the trailing window."
            )

    def _get_completion_hidden_states(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_length: int,
    ) -> torch.Tensor:
        self._assert_right_padded_completion(attention_mask, completion_length)
        route_via_model_forward = self._should_route_hidden_states_through_model_forward(model)
        backbone, _ = self._split_causal_lm_backbone_and_head(model)
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }

        # Under ZeRO-3, some architectures (for example Qwen 3.5 GatedDeltaNet)
        # read submodule parameters directly inside the backbone forward. Routing
        # through the full model preserves DeepSpeed's parameter coordination.
        if route_via_model_forward:
            forward_kwargs["output_hidden_states"] = False
            forward_kwargs["logits_to_keep"] = 1
            autocast_context = nullcontext()
        else:
            autocast_context = self._autocast_context()

        with autocast_context:
            if route_via_model_forward:
                hidden_states = self._get_hidden_states_via_model_forward(
                    model,
                    backbone,
                    forward_kwargs,
                )
            else:
                outputs = backbone(**forward_kwargs)
                hidden_states = self._extract_last_hidden_state(outputs)
            completion_hidden_states = hidden_states[:, -(completion_length + 1):, :]
            if completion_hidden_states.size(1) > completion_length:
                completion_hidden_states = completion_hidden_states[:, :-1, :]
        return completion_hidden_states[:, -completion_length:, :].contiguous()

    def _get_completion_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_length: int,
    ) -> torch.Tensor:
        self._assert_right_padded_completion(attention_mask, completion_length)
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "logits_to_keep": completion_length + 1,
        }

        autocast_context = (
            nullcontext() if self._is_deepspeed_engine(model) else self._autocast_context()
        )
        with autocast_context:
            outputs = model(**forward_kwargs)
            logits = outputs.logits
            if logits.size(1) > completion_length:
                logits = logits[:, :-1, :]
        return logits[:, -completion_length:, :]

    def _get_per_token_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_length: int,
    ) -> torch.Tensor:
        if self._lm_head_chunk_size() is None:
            logits = self._get_completion_logits(
                model, input_ids, attention_mask, completion_length,
            )
            logps = selective_log_softmax(logits, input_ids[:, -completion_length:])
            del logits
            return logps
        hidden_states = self._get_completion_hidden_states(
            model, input_ids, attention_mask, completion_length,
        )
        logps = self._selective_log_softmax_from_hidden_states(
            model, hidden_states, input_ids[:, -completion_length:],
        )
        del hidden_states
        return logps

    # ------------------------------------------------------------------
    # Sequence-dim padding / trimming / concatenation utilities
    # ------------------------------------------------------------------

    def _global_max_sequence_length(self, local_length: int) -> int:
        from utils.trainer_utils import is_distributed_ready
        import torch.distributed as dist
        max_length = torch.tensor(local_length, device=self.device, dtype=torch.long)
        if is_distributed_ready():
            dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
        return int(max_length.item())

    @staticmethod
    def _pad_sequence_dim(
        tensor: torch.Tensor,
        target_length: int,
        *,
        value: int | float | bool = 0,
    ) -> torch.Tensor:
        current_length = tensor.size(1)
        if current_length >= target_length:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[1] = target_length - current_length
        padding = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, padding), dim=1)

    # ------------------------------------------------------------------
    # Chunked lm-head operations
    # ------------------------------------------------------------------

    def _prepare_lm_head_chunk_inputs(
        self,
        hidden_states: torch.Tensor,
        *sequence_tensors: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor, tuple[torch.Tensor, ...]]:
        local_length = hidden_states.size(1)
        global_length = self._global_max_sequence_length(local_length)
        padded_hidden_states = self._pad_sequence_dim(hidden_states, global_length, value=0)
        padded_sequence_tensors = tuple(
            self._pad_sequence_dim(tensor, global_length, value=0)
            for tensor in sequence_tensors
        )
        return local_length, global_length, padded_hidden_states, padded_sequence_tensors

    def _selective_log_softmax_from_hidden_states(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, lm_head = self._split_causal_lm_backbone_and_head(model)
        chunk_size = self._lm_head_chunk_size()
        if chunk_size is None:
            raise RuntimeError("_selective_log_softmax_from_hidden_states requires lm_head_chunk_size.")
        local_length, global_length, hidden_states, padded = self._prepare_lm_head_chunk_inputs(
            hidden_states, token_ids,
        )
        padded_token_ids = padded[0]
        logps: torch.Tensor | None = None
        use_ac = torch.is_grad_enabled()
        with self._autocast_context():
            for start in range(0, global_length, chunk_size):
                end = min(start + chunk_size, global_length)
                h_chunk = hidden_states[:, start:end, :]
                ids_chunk = padded_token_ids[:, start:end]
                if use_ac:
                    part = activation_checkpoint(
                        _selective_log_softmax_chunk_fn,
                        lm_head, h_chunk, ids_chunk,
                        use_reentrant=False,
                    )
                else:
                    logits = lm_head(h_chunk)
                    part = selective_log_softmax(logits, ids_chunk)
                    del logits
                if logps is None:
                    batch_size = part.size(0)
                    logps = torch.empty(
                        (batch_size, global_length), dtype=part.dtype, device=part.device,
                    )
                logps[:, start:end].copy_(part)
        if logps is None:
            raise ValueError("Expected at least one logprob chunk.")
        return logps[:, :local_length]

    def _compute_full_vocab_kl_and_logps_from_hidden_states(
        self,
        student_model: nn.Module,
        student_hidden_states: torch.Tensor,
        teacher_model: nn.Module,
        teacher_hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute full-vocab reverse KL and sampled-token log-probs in one lm-head pass."""
        if student_hidden_states.ndim != 3 or teacher_hidden_states.ndim != 3:
            raise ValueError("hidden states must have shape [batch, tokens, hidden].")
        if student_hidden_states.shape[:2] != teacher_hidden_states.shape[:2]:
            raise ValueError(
                "student and teacher hidden states must agree on batch and seq dims."
            )
        if token_ids.shape != student_hidden_states.shape[:2]:
            raise ValueError("token_ids must have shape [batch, tokens] aligned with hidden states.")
        chunk_size = self._lm_head_chunk_size()
        if chunk_size is None:
            raise RuntimeError(
                "_compute_full_vocab_kl_and_logps_from_hidden_states requires lm_head_chunk_size."
            )
        _, student_lm_head = self._split_causal_lm_backbone_and_head(student_model)
        _, teacher_lm_head = self._split_causal_lm_backbone_and_head(teacher_model)
        local_length, global_length, student_hidden, padded = self._prepare_lm_head_chunk_inputs(
            student_hidden_states,
            teacher_hidden_states,
            token_ids,
        )
        teacher_hidden = padded[0]
        padded_token_ids = padded[1]
        kl_parts: list[torch.Tensor] = []
        logp_parts: list[torch.Tensor] = []
        use_ac = torch.is_grad_enabled()
        with self._autocast_context():
            for start in range(0, global_length, chunk_size):
                end = min(start + chunk_size, global_length)
                sh_chunk = student_hidden[:, start:end, :]
                th_chunk = teacher_hidden[:, start:end, :]
                ids_chunk = padded_token_ids[:, start:end]
                if use_ac:
                    kl_part, logp_part = activation_checkpoint(
                        _full_vocab_reverse_kl_and_logps_chunk_fn,
                        student_lm_head,
                        teacher_lm_head,
                        sh_chunk,
                        th_chunk,
                        ids_chunk,
                        use_reentrant=False,
                    )
                else:
                    kl_part, logp_part = _full_vocab_reverse_kl_and_logps_chunk_fn(
                        student_lm_head,
                        teacher_lm_head,
                        sh_chunk,
                        th_chunk,
                        ids_chunk,
                    )
                kl_parts.append(kl_part)
                logp_parts.append(logp_part)
        if not kl_parts:
            raise ValueError("Expected at least one full-vocab reverse-KL chunk.")
        reverse_kl = torch.cat(kl_parts, dim=1) if len(kl_parts) > 1 else kl_parts[0]
        logps = torch.cat(logp_parts, dim=1) if len(logp_parts) > 1 else logp_parts[0]
        del kl_parts, logp_parts
        return reverse_kl[:, :local_length], logps[:, :local_length]

    # ------------------------------------------------------------------
    # High-level subbatched forward wrappers
    # ------------------------------------------------------------------

    def _forward_student_completion_hidden_states_subbatched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_length: int,
    ) -> torch.Tensor:
        """Forward student over the completion and return its completion hidden states.

        The hidden states retain a grad path back through the student backbone and
        lm-head (when called under enable_grad), so downstream callers that close
        the loop with the full-vocab reverse-KL kernel see the same gradient as if
        student logits had been materialized.
        """
        if self._lm_head_chunk_size() is None:
            raise RuntimeError(
                "_forward_student_completion_hidden_states_subbatched requires lm_head_chunk_size."
            )
        return self._get_completion_hidden_states(
            self.model, input_ids, attention_mask, completion_length,
        )
