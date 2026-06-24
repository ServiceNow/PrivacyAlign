"""
Embedding-based diversity filter using cosine similarity deduplication.

After each batch completes, computes embeddings for generated samples and rejects
near-semantic duplicates — samples whose max cosine similarity to any previously
accepted sample exceeds a configurable threshold.  Uses a local HuggingFace
embedding model (default: Qwen/Qwen3-Embedding-8B) — no API cost.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pipeline.trajectory_parsing import parse_steps

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter class
# ---------------------------------------------------------------------------

class EmbeddingDiversityFilter:
    """Post-generation semantic deduplication filter.

    Maintains a rolling history of L2-normalized embeddings for accepted
    samples.  A candidate is rejected if its maximum cosine similarity to
    any history vector exceeds ``max_similarity``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda",
        embed_batch_size: int = 1,
        max_length: int = 32768,
        max_similarity: float = 0.95,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.embed_batch_size = embed_batch_size
        self.max_length = max_length
        self.max_similarity = max_similarity

        # Lazy-loaded
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

        # History of accepted embeddings stored as a single (N, dim) matrix
        # for efficient batched cosine similarity.  Embeddings are already
        # L2-normalized so cosine similarity = dot product.
        self._history: Optional[np.ndarray] = None

    @staticmethod
    def _should_log_progress(current: int, total: int) -> bool:
        """Return True when a progress update should be emitted."""
        if total <= 20:
            return True
        interval = max(total // 20, 1)
        return current == 1 or current == total or current % interval == 0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the HuggingFace embedding model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer  # noqa: late import

        LOGGER.info("Loading embedding model: %s on device: %s", self.model_name, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

        use_cpu = self.device == "cpu"
        torch_dtype = torch.float32 if use_cpu else torch.float16

        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        if use_cpu:
            self._model = self._model.cpu()
        elif torch.cuda.is_available():
            self._model = self._model.to(self.device)

        self._model.eval()
        LOGGER.info("Loaded embedding model: %s", self.model_name)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returning an (N, dim) float32 numpy array.

        For Qwen embedding models, prefer the model's native ``encode`` path.
        Fall back to masked mean pooling + L2 normalization when ``encode`` is
        unavailable.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._load_model()
        batch_size = min(self.embed_batch_size, len(texts))
        total_batches = (len(texts) + batch_size - 1) // batch_size
        LOGGER.info(
            "Embedding %d texts with batch_size=%d (%d batches)",
            len(texts), batch_size, total_batches,
        )

        # Native encode path (used by Qwen embedding checkpoints with remote code).
        if hasattr(self._model, "encode"):
            try:
                all_batches: List[np.ndarray] = []
                for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
                    end = min(start + batch_size, len(texts))
                    if self._should_log_progress(batch_num, total_batches):
                        LOGGER.info(
                            "Embedding progress: batch %d/%d (%d/%d texts)",
                            batch_num, total_batches, end, len(texts),
                        )
                    encoded = self._model.encode(
                        texts[start:end],
                        instruction="",
                        max_length=self.max_length,
                        batch_size=min(batch_size, end - start),
                    )
                    if isinstance(encoded, torch.Tensor):
                        encoded = encoded.detach().cpu().numpy()
                    arr = np.asarray(encoded, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    all_batches.append(arr)
                arr = np.vstack(all_batches)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.clip(norms, 1e-8, None)
                LOGGER.info("Embedding complete: %d/%d texts", len(texts), len(texts))
                return arr.astype(np.float32)
            except TypeError:
                # Some model implementations expose encode() with a different signature.
                try:
                    all_batches = []
                    for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
                        end = min(start + batch_size, len(texts))
                        if self._should_log_progress(batch_num, total_batches):
                            LOGGER.info(
                                "Embedding progress: batch %d/%d (%d/%d texts)",
                                batch_num, total_batches, end, len(texts),
                            )
                        encoded = self._model.encode(texts[start:end])
                        if isinstance(encoded, torch.Tensor):
                            encoded = encoded.detach().cpu().numpy()
                        arr = np.asarray(encoded, dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        all_batches.append(arr)
                    arr = np.vstack(all_batches)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    arr = arr / np.clip(norms, 1e-8, None)
                    LOGGER.info("Embedding complete: %d/%d texts", len(texts), len(texts))
                    return arr.astype(np.float32)
                except Exception as exc:  # pragma: no cover - model specific
                    LOGGER.warning(
                        "Embedding model encode() failed; falling back to pooled embeddings: %s",
                        exc,
                    )
            except Exception as exc:  # pragma: no cover - model specific
                LOGGER.warning(
                    "Embedding model encode() failed; falling back to pooled embeddings: %s",
                    exc,
                )

        # Generic fallback path.
        device = next(self._model.parameters()).device
        embeddings: List[np.ndarray] = []

        for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
            end = min(start + batch_size, len(texts))
            if self._should_log_progress(batch_num, total_batches):
                LOGGER.info(
                    "Embedding progress: batch %d/%d (%d/%d texts)",
                    batch_num, total_batches, end, len(texts),
                )
            batch_texts = texts[start:start + batch_size]
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = self._model(**inputs)

                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs[0]

                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size())
                mask = mask.to(hidden.dtype)
                summed = torch.sum(hidden * mask, dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                embeddings.extend(pooled.cpu().float().numpy())

        LOGGER.info("Embedding complete: %d/%d texts", len(texts), len(texts))
        return np.asarray(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_observation_keys(observation: Any) -> List[str]:
        """Extract stable structural keys from a tool_result observation."""
        if isinstance(observation, str):
            text = observation.strip()
            if not text:
                return []
            try:
                observation = json.loads(text)
            except Exception:
                return []

        if isinstance(observation, dict):
            return sorted(str(key) for key in observation.keys() if isinstance(key, str))

        if (
            isinstance(observation, list)
            and observation
            and isinstance(observation[0], dict)
        ):
            return sorted(
                str(key)
                for key in observation[0].keys()
                if isinstance(key, str)
            )

        return []

    @staticmethod
    def _compress_trajectory(executable_trajectory: Any) -> str:
        """Build a compact, low-noise signature from trajectory structure."""
        steps = parse_steps(executable_trajectory)
        if not steps:
            return ""

        actions: List[str] = []
        observation_keys: Counter[str] = Counter()

        for step in steps:
            action = str(step.get("action", "")).strip()
            if action:
                actions.append(action)
            for key in EmbeddingDiversityFilter._extract_observation_keys(
                step.get("observation", "")
            ):
                observation_keys[key] += 1

        if not actions:
            return ""

        action_counts = Counter(actions)
        action_head = ">".join(actions[:10])
        top_actions = ",".join(
            f"{name}:{count}" for name, count in action_counts.most_common(8)
        )

        transition_counts: Counter[str] = Counter(
            f"{left}>{right}" for left, right in zip(actions, actions[1:])
        )
        top_transitions = ",".join(
            f"{transition}:{count}"
            for transition, count in transition_counts.most_common(8)
        )

        top_obs_keys = ",".join(
            key for key, _ in observation_keys.most_common(12)
        )

        return (
            f"steps={len(actions)}; "
            f"action_head={action_head}; "
            f"top_actions={top_actions}; "
            f"top_transitions={top_transitions}; "
            f"obs_keys={top_obs_keys}"
        )

    @staticmethod
    def _extract_text(sample: Dict[str, Any]) -> str:
        """Return the semantically rich text used for embedding."""
        seed = sample.get("seed", {})
        vignette = sample.get("vignette", {})
        trajectory = sample.get("trajectory", {})
        memories = sample.get("memories", [])

        data_subject = seed.get("data_subject", "")
        data_sender = seed.get("data_sender", "")
        data_recipient = seed.get("data_recipient", "")

        story = vignette.get("story", "")
        instruction = trajectory.get("user_instruction", "")

        sensitive_items = vignette.get("sensitive_info_items", [])
        sensitive_text = " | ".join(sensitive_items) if sensitive_items else ""
        relevant_items = vignette.get("relevant_info_items", [])
        relevant_text = " | ".join(relevant_items) if relevant_items else ""

        action = trajectory.get("final_action", "")
        toolkits = ", ".join(trajectory.get("toolkits", []))
        trajectory_signature = EmbeddingDiversityFilter._compress_trajectory(
            trajectory.get("executable_trajectory", "")
        )

        memories_text = " | ".join(memories) if memories else ""

        return (
            f"Subject: {data_subject}\n"
            f"Sender: {data_sender}\n"
            f"Recipient: {data_recipient}\n"
            f"{story}\n"
            f"{instruction}\n"
            f"Sensitive info: {sensitive_text}\n"
            f"Relevant info: {relevant_text}\n"
            f"Memories: {memories_text}\n"
            f"Action: {action}\n"
            f"Tools: {toolkits}\n"
            f"Trajectory signature: {trajectory_signature}"
        )

    # ------------------------------------------------------------------
    # Similarity check
    # ------------------------------------------------------------------

    def _max_cosine_similarity(self, embedding: np.ndarray) -> float:
        """Return the maximum cosine similarity between *embedding* and history.

        Embeddings are already L2-normalized, so cosine similarity is just a
        dot product.  Returns 0.0 when history is empty.
        """
        if self._history is None:
            return 0.0

        # (N, dim) @ (dim,) -> (N,)
        similarities = self._history @ embedding
        return float(similarities.max())

    # ------------------------------------------------------------------
    # Batch filter
    # ------------------------------------------------------------------

    def filter_batch(
        self, batch_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter a batch of generation results for near-duplicate removal.

        Embeds all samples in one forward pass, then processes them
        sequentially — rejecting a sample if its max cosine similarity to
        any previously accepted sample exceeds ``max_similarity``.

        Returns ``(accepted, rejected)``.
        """
        if not batch_results:
            return [], []

        texts = [self._extract_text(r) for r in batch_results]
        all_embeddings = self._embed_texts(texts)
        progress_interval = max(len(batch_results) // 20, 1)

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        accepted_sims: List[float] = []
        rejected_sims: List[float] = []

        for i, result in enumerate(batch_results):
            emb = all_embeddings[i]
            max_sim = self._max_cosine_similarity(emb)

            if max_sim <= self.max_similarity:
                accepted.append(result)
                accepted_sims.append(max_sim)
                # Append to history matrix
                if self._history is None:
                    self._history = emb.reshape(1, -1)
                else:
                    self._history = np.vstack([self._history, emb.reshape(1, -1)])
            else:
                rejected.append(result)
                rejected_sims.append(max_sim)

            processed = i + 1
            if (
                processed == 1
                or processed == len(batch_results)
                or processed % progress_interval == 0
            ):
                LOGGER.info(
                    "Similarity scan progress: %d/%d processed, accepted=%d, rejected=%d",
                    processed,
                    len(batch_results),
                    len(accepted),
                    len(rejected),
                )

        accepted_mean = float(np.mean(accepted_sims)) if accepted_sims else 0.0
        accepted_p95 = float(np.percentile(accepted_sims, 95)) if accepted_sims else 0.0
        rejected_mean = float(np.mean(rejected_sims)) if rejected_sims else 0.0
        rejected_p95 = float(np.percentile(rejected_sims, 95)) if rejected_sims else 0.0
        LOGGER.info(
            "Similarity scan complete: processed=%d accepted=%d rejected=%d accepted_mean_max_sim=%.4f accepted_p95_max_sim=%.4f rejected_mean_max_sim=%.4f rejected_p95_max_sim=%.4f",
            len(batch_results),
            len(accepted),
            len(rejected),
            accepted_mean,
            accepted_p95,
            rejected_mean,
            rejected_p95,
        )

        return accepted, rejected

    # ------------------------------------------------------------------
    # Warm-up (for resume)
    # ------------------------------------------------------------------

    def warm_from_samples(self, samples: List[Dict[str, Any]]) -> None:
        """Re-embed existing samples to warm the history on resume."""
        if not samples:
            return

        LOGGER.info("Warming embedding history from %d existing samples", len(samples))
        texts = [self._extract_text(s) for s in samples]
        embeddings = self._embed_texts(texts)

        if embeddings.shape[0] > 0:
            self._history = embeddings

        LOGGER.info("Embedding history warmed: %d entries", embeddings.shape[0])
