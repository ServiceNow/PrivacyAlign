"""Full-vocab reverse-KL kernel shared by the policy reference regularizer.

Streams the per-token reverse KL over the vocab dim in chunks of size
``_REVERSE_KL_VOCAB_CHUNK_SIZE`` so the peak fp32 footprint per call is bounded
by ``[batch, tokens, V_chunk]`` rather than ``[batch, tokens, V]``. The caller
is still expected to chunk over the seq dim (see
``TrainerForwardMixin._compute_full_vocab_kl_and_logps_from_hidden_states``)
to bound the total token count.
"""

from __future__ import annotations

import torch


_LOGSUMEXP_CHUNK_SIZE = 2048
_REVERSE_KL_VOCAB_CHUNK_SIZE = 2048


def chunked_logsumexp(logits: torch.Tensor) -> torch.Tensor:
    """Compute logsumexp in float32 without materializing a full float32 logits tensor."""
    max_logits_fp32 = logits.amax(dim=-1).float()
    sum_exp = torch.zeros_like(max_logits_fp32)
    effective_chunk_size = min(_LOGSUMEXP_CHUNK_SIZE, logits.size(-1))
    max_logits_unsq = max_logits_fp32.unsqueeze(-1)

    for start in range(0, logits.size(-1), effective_chunk_size):
        chunk = logits[..., start : start + effective_chunk_size]
        # .float() returns a fresh tensor for non-fp32 inputs; for fp32 it
        # returns the same view, so clone before the in-place ops below.
        chunk_fp32 = chunk.clone() if chunk.dtype == torch.float32 else chunk.float()
        chunk_fp32.sub_(max_logits_unsq)
        chunk_fp32.exp_()
        sum_exp = sum_exp + chunk_fp32.sum(dim=-1)

    tiny = torch.finfo(sum_exp.dtype).tiny
    return max_logits_fp32 + torch.log(sum_exp.clamp_min(tiny))


def compute_full_vocab_kl_and_logps_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token KL(student || teacher) and student log-probs at ``token_ids``.

    Fuses the student/teacher log-normalizer reduction with the divergence
    accumulator so each vocab chunk is loaded once per side instead of twice.
    Uses the identity ``KL = E_s[s - t] + (lse_t - lse_s)``: the loop
    accumulates ``sum_v exp(s_v - student_max) * (s_v - t_v)`` alongside the
    per-side ``sum_exp``, then closes the form after the loop. Teacher logits
    are read under ``no_grad`` so no gradient flows to the teacher even if its
    logits carry a grad_fn upstream.
    """
    if token_ids.shape != student_logits.shape[:-1]:
        raise ValueError("token_ids must have shape [batch, tokens] aligned with student_logits.")
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, tokens, vocab].")
    if teacher_logits.ndim != 3:
        raise ValueError("teacher_logits must have shape [batch, tokens, vocab].")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape.")

    per_token_shape = student_logits.shape[:-1]
    vocab_size = student_logits.size(-1)
    chunk_size = min(_REVERSE_KL_VOCAB_CHUNK_SIZE, vocab_size)
    device = student_logits.device

    # Per-token max for numerical stability. ``amax`` is a single-pass reduction
    # (memory-bandwidth bound) and lets the fused loop run its ``exp`` in fp32
    # without overflow.
    student_max = student_logits.amax(dim=-1).float()
    with torch.no_grad():
        teacher_max = teacher_logits.amax(dim=-1).float()

    token_logits = torch.gather(
        student_logits, dim=-1, index=token_ids.unsqueeze(-1),
    ).squeeze(-1).float()

    sum_exp_s = torch.zeros(per_token_shape, dtype=torch.float32, device=device)
    sum_exp_t = torch.zeros(per_token_shape, dtype=torch.float32, device=device)
    weighted_diff_sum = torch.zeros(per_token_shape, dtype=torch.float32, device=device)

    student_max_unsq = student_max.unsqueeze(-1)
    teacher_max_unsq = teacher_max.unsqueeze(-1)

    for v0 in range(0, vocab_size, chunk_size):
        v1 = min(v0 + chunk_size, vocab_size)
        s_chunk = student_logits[..., v0:v1].float()
        s_shifted_exp = (s_chunk - student_max_unsq).exp()
        sum_exp_s = sum_exp_s + s_shifted_exp.sum(dim=-1)

        with torch.no_grad():
            t_chunk = teacher_logits[..., v0:v1].float()
            sum_exp_t = sum_exp_t + (t_chunk - teacher_max_unsq).exp().sum(dim=-1)

        weighted_diff_sum = weighted_diff_sum + (s_shifted_exp * (s_chunk - t_chunk)).sum(dim=-1)

    tiny = torch.finfo(sum_exp_s.dtype).tiny
    safe_sum_exp_s = sum_exp_s.clamp_min(tiny)
    lse_s = student_max + torch.log(safe_sum_exp_s)
    with torch.no_grad():
        lse_t = teacher_max + torch.log(sum_exp_t.clamp_min(tiny))

    logps = token_logits - lse_s
    divergence = weighted_diff_sum / safe_sum_exp_s + (lse_t - lse_s)
    return divergence, logps
