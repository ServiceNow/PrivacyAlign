"""Shared training-runtime package.

This package holds the worker-side training runtime plus training-specific
support code such as batching, metrics, and tokenization helpers.

Module layout:
- ``trainer``          — entry-point ``Trainer`` class composing all mixins
- ``config``           — ``TrainingConfig`` dataclass
- ``runtime``          — distributed setup, DeepSpeed lifecycle
- ``rollout``          — prompt shaping and rollout batch construction
- ``forward``          — model forward-pass infrastructure (backbone/lm-head)
- ``loss``             — objective execution and policy math
- ``checkpoint``       — checkpoint save, load, and export
- ``phase_logging``    — structured phase-progress logging mixin
- ``validation``       — finite-value validation guards
- ``batching``         — batch-layout helpers and micro-batch splitting
- ``batch_types``      — TypedDict batch shapes
- ``metrics``          — step-level metric accumulators
- ``tokenization``     — tokenization helpers
- ``full_vocab_kl``    — fused full-vocab reverse-KL + sampled-token log-prob kernel
"""
