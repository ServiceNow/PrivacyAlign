"""Ray worker trainer entry point.

`Trainer` exists only as the worker-side runtime used by the Ray path.
Its implementation is split across focused mixins:
- `training/runtime.py`       — distributed setup, DeepSpeed lifecycle
- `training/rollout.py`       — prompt shaping and rollout batch construction
- `training/forward.py`       — model forward-pass infrastructure (backbone/lm-head)
- `training/checkpoint.py`    — checkpoint save, load, and export
- `training/loss.py`          — objective execution, policy math, metric
                                 aggregation (inherits forward + checkpoint)
- `training/phase_logging.py` — structured phase-progress logging
- `training/validation.py`    — finite-value validation guards
"""

from __future__ import annotations

import copy
from typing import Any

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import TrainingConfig
from objectives import build_training_objective
from .loss import TrainerLossMixin
from .rollout import TrainerRolloutMixin
from .runtime import TrainerRuntimeMixin
from utils.trainer_utils import (
    disable_dropout_in_model as _disable_dropout_in_model,
    set_seed as _set_seed,
)


class Trainer(
    TrainerRuntimeMixin,
    TrainerRolloutMixin,
    TrainerLossMixin,
):
    """Ray worker-side runtime for shared training objectives."""

    def __init__(
        self,
        model: str | PreTrainedModel,
        ref_model: str | PreTrainedModel | None = None,
        args: TrainingConfig | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
    ) -> None:
        if args is None:
            raise ValueError("Trainer requires an explicit TrainingConfig.")
        if isinstance(model, str):
            raise TypeError("Pass an instantiated model into Trainer.")

        # Inputs and process-local runtime.
        self.args = args
        self.training_objective = build_training_objective(self.args.training_objective)
        self._setup_logging()
        self._setup_distributed()
        _set_seed(args.seed + args.rank)

        # Tokenization state.
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path,
                trust_remote_code=self.args.trust_remote_code,
            )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        processing_class.padding_side = "left"
        self.processing_class = processing_class
        self.pad_token_id = processing_class.pad_token_id
        self.eos_token_id = processing_class.eos_token_id
        # Model ownership.
        needs_reference_model = self.training_objective.requires_reference_model(self.args)
        base_model = model
        if args.disable_dropout:
            _disable_dropout_in_model(base_model)
        if needs_reference_model and ref_model is None:
            ref_model = copy.deepcopy(base_model)
        if not needs_reference_model:
            ref_model = None

        self.base_model = base_model
        self.model = base_model
        self.ref_model = self._prepare_ref_model(ref_model)

        # Lazy-initialized runtime engines.
        self.deepspeed_engine: Any | None = None
        self.deepspeed_config: dict[str, Any] | None = None

        # Training bookkeeping.
        self.optimizer: Any | None = None
        self.scheduler: Any | None = None
        self.global_step = 0
        self.num_input_tokens_seen = 0
