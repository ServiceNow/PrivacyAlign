#!/usr/bin/env python3
"""
Unified model client for multiple providers.

Supports:
- OpenRouter API
- vLLM Offline (in-process inference via vllm)

Model naming conventions:
- "openai/gpt-oss-120b" -> OpenRouter API
- "openrouter/openai/gpt-oss-120b" -> OpenRouter API (prefix stripped)
- Any model name + vllm_offline=True -> vLLM Offline
"""

import os
import logging
import asyncio
import re
import inspect
from typing import List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

# Set up logging
LOGGER = logging.getLogger(__name__)

_HARMONY_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>")


class ModelProvider(Enum):
    """Supported model providers."""
    OPENROUTER = "openrouter"
    VLLM_OFFLINE = "vllm-offline"  # vLLM running in-process (no server)


@dataclass
class ProviderConfig:
    """Configuration for an API provider."""
    provider: ModelProvider
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    model_name: Optional[str] = None  # The actual model name to use in API calls


# Provider detection patterns and configurations
PROVIDER_CONFIGS = {
    # Optional OpenRouter prefix, e.g. openrouter/openai/gpt-oss-120b
    "openrouter/": ProviderConfig(
        ModelProvider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
}


def detect_provider(model_path: str) -> Tuple[ProviderConfig, str]:
    """Detect the provider and return config with the actual model name.
    
    Args:
        model_path: Model name. Use prefixes to force specific providers:
            - "openrouter/" - Force OpenRouter API and strip this prefix
        
    Returns:
        Tuple of (ProviderConfig, actual_model_name)
    """
    # Check against known patterns
    for pattern, config in PROVIDER_CONFIGS.items():
        if model_path.lower().startswith(pattern.lower()):
            # For OpenRouter, strip the optional openrouter/ prefix for the actual API call
            actual_model = model_path
            if pattern == "openrouter/" and model_path.lower().startswith("openrouter/"):
                # openrouter/openai/gpt-oss-120b -> openai/gpt-oss-120b
                actual_model = model_path[len("openrouter/"):]
            return ProviderConfig(
                provider=config.provider,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                model_name=actual_model
            ), actual_model
    
    # Default to OpenRouter for unknown models
    return ProviderConfig(
        ModelProvider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_name=model_path,
    ), model_path


class UnifiedModelClient:
    """Unified client for multiple model providers.
    
    Automatically detects the provider based on model name and uses the
    appropriate client for generation.
    
    Example:
        >>> # OpenRouter API
        >>> client = UnifiedModelClient("openai/gpt-oss-120b")
        >>> responses = client.generate(["Hello, world!"])
    """
    
    def __init__(
        self,
        model_path: str,
        postprocess_fn: Optional[Callable[[str], str]] = None,
        vllm_offline: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        enforce_eager: bool = False,
        language_model_only: bool = False,
        enable_prefix_caching: bool = False,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        hf_cache_dir: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_exclude: Optional[bool] = None,
        verbosity: Optional[str] = None,
        enable_thinking: bool = True,
        speculative_config: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the model client.

        Args:
            model_path: Model name (API) or model name for vLLM offline
            postprocess_fn: Optional custom postprocessing function
            vllm_offline: If True, use in-process vLLM for local inference
            tensor_parallel_size: Tensor parallel size for offline vLLM
            pipeline_parallel_size: Pipeline parallel size for offline vLLM
            enable_expert_parallel: Enable expert parallelism for MoE models
            enforce_eager: Force eager execution in offline vLLM
            language_model_only: Skip multimodal modules, text-only mode
            enable_prefix_caching: Cache KV blocks for shared prompt prefixes
            presence_penalty: Penalize new tokens based on presence in text so far
            repetition_penalty: Multiplicative penalty for repeated tokens
            gpu_memory_utilization: Fraction of GPU memory for vLLM (default 0.9)
            max_model_len: Optional maximum model context length for vLLM
            kv_cache_dtype: Data type for KV cache storage ("auto", "fp8", "fp8_e4m3", "fp8_e5m2")
            hf_cache_dir: Optional HuggingFace cache directory
            reasoning_effort: Optional reasoning effort for GPT-OSS Harmony
            reasoning_enabled: Optional OpenRouter reasoning.enabled flag
            reasoning_max_tokens: Optional OpenRouter reasoning.max_tokens budget
            reasoning_exclude: Optional OpenRouter reasoning.exclude flag
            verbosity: Optional OpenRouter verbosity parameter
            enable_thinking: Enable thinking/reasoning for models that support it (default True)
            speculative_config: Optional dict for vLLM speculative decoding
                (e.g. {"method": "qwen3_next_mtp", "num_speculative_tokens": 2})
            **kwargs: Additional arguments (ignored, for backward compatibility)
        """
        # Log ignored kwargs for debugging
        if kwargs:
            LOGGER.debug("Ignoring unused parameters: %s", list(kwargs.keys()))
        self.model_path = model_path
        self._postprocess_fn = postprocess_fn
        self.vllm_offline = vllm_offline
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self.enforce_eager = enforce_eager
        self.language_model_only = language_model_only
        self.enable_prefix_caching = enable_prefix_caching
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.hf_cache_dir = hf_cache_dir
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning_exclude = reasoning_exclude
        self.verbosity = verbosity
        self._warned_structured_unsupported = False

        # If vllm_offline is requested, use in-process vLLM
        if vllm_offline:
            self.provider = ModelProvider.VLLM_OFFLINE
            self.actual_model_name = model_path
            self.config = ProviderConfig(ModelProvider.VLLM_OFFLINE)
            LOGGER.info(
                "Using offline vLLM for model: %s (tp=%s, pp=%s, ep=%s, enforce_eager=%s, hf_cache_dir=%s)",
                self.actual_model_name,
                tensor_parallel_size,
                pipeline_parallel_size,
                enable_expert_parallel,
                enforce_eager,
                hf_cache_dir,
            )
            self._init_vllm_offline(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                enforce_eager=enforce_eager,
                language_model_only=language_model_only,
                enable_prefix_caching=enable_prefix_caching,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                hf_cache_dir=hf_cache_dir,
                reasoning_effort=reasoning_effort,
                speculative_config=speculative_config,
            )
            return
        
        # Detect provider and get config
        self.config, self.actual_model_name = detect_provider(model_path)
        self.provider = self.config.provider
        
        LOGGER.info("Detected provider: %s for model: %s", self.provider.value, model_path)
        
        # Initialize based on provider
        if self.provider == ModelProvider.OPENROUTER:
            self._init_openrouter()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client via OpenAI-compatible SDK."""
        from openai import OpenAI, AsyncOpenAI
        
        api_key_env = self.config.api_key_env or "OPENROUTER_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required API key environment variable: {api_key_env}")

        base_url = self.config.base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

        # Optional OpenRouter attribution headers.
        default_headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title
        
        # Configure timeouts and retries for reliability
        # Default timeout is too short for some operations
        timeout_config = 120.0  # 2 minutes
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
            timeout=timeout_config,
            max_retries=3,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
            timeout=timeout_config,
            max_retries=3,
        )
        self.tokenizer = None
    
    @staticmethod
    def _is_gpt_oss_model(model_name: str) -> bool:
        return "gpt-oss" in (model_name or "").lower()

    @staticmethod
    def _strip_harmony_special_tokens(text: str) -> str:
        if not text:
            return ""
        return _HARMONY_SPECIAL_RE.sub("", text).strip()

    @staticmethod
    def _prompt_to_messages(prompt: Any) -> List[dict]:
        """Convert a prompt to a list of chat messages.

        Supports three formats:
        - A plain string: returned as a single user message.
        - A string containing ``\\n===\\n``: split into a system message
          (everything before the separator) and a user message (after).
        - A list of ``{"role": ..., "content": ...}`` dicts: returned as-is.
        """
        if isinstance(prompt, list):
            return prompt
        text = str(prompt)
        if "\n===\n" in text:
            system_part, user_part = text.split("\n===\n", 1)
            return [
                {"role": "system", "content": system_part.strip()},
                {"role": "user", "content": user_part.strip()},
            ]
        return [{"role": "user", "content": text}]

    def _init_vllm_offline(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        enable_expert_parallel: bool,
        enforce_eager: bool,
        language_model_only: bool,
        enable_prefix_caching: bool,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        kv_cache_dtype: str = "auto",
        hf_cache_dir: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        speculative_config: Optional[dict] = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # pylint: disable=import-error
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("vllm is required for offline inference. Install vllm to use vllm_offline mode.") from exc

        self._vllm_SamplingParams = SamplingParams
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "enable_expert_parallel": enable_expert_parallel,
            "enforce_eager": enforce_eager,
            "language_model_only": language_model_only,
            "enable_prefix_caching": enable_prefix_caching,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "download_dir": hf_cache_dir,
        }
        if kv_cache_dtype and kv_cache_dtype != "auto":
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if speculative_config is not None:
            llm_kwargs["speculative_config"] = speculative_config
            LOGGER.info("Speculative decoding config: %s", speculative_config)

        # Keep compatibility across vLLM versions by only forwarding
        # arguments supported by the installed LLM constructor / EngineArgs.
        # When LLM accepts **kwargs they are forwarded to EngineArgs, so we
        # need to check EngineArgs as well to avoid unsupported-arg errors.
        sig = inspect.signature(LLM)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_var_keyword:
            # LLM forwards unknown kwargs to EngineArgs — filter against EngineArgs too.
            try:
                from vllm.engine.arg_utils import EngineArgs  # pylint: disable=import-error
                engine_sig = inspect.signature(EngineArgs)
                explicit_llm = {
                    p.name for p in sig.parameters.values()
                    if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                }
                supported = explicit_llm | set(engine_sig.parameters.keys())
            except Exception:
                supported = None  # can't introspect — pass everything through
        else:
            supported = set(sig.parameters.keys())

        if supported is not None:
            filtered_kwargs = {k: v for k, v in llm_kwargs.items() if k in supported}
            dropped = sorted(set(llm_kwargs.keys()) - set(filtered_kwargs.keys()))
            if dropped:
                LOGGER.warning(
                    "Installed vLLM does not support these LLM/EngineArgs and they were ignored: %s",
                    ", ".join(dropped),
                )
        else:
            filtered_kwargs = llm_kwargs

        self._vllm_engine = LLM(**filtered_kwargs)

        self._harmony_encoding = None
        self._harmony_stop_token_ids: List[int] = []
        self._harmony = None

        if not self._is_gpt_oss_model(model_path):
            # Load tokenizer for chat template formatting (add_generation_prompt, enable_thinking)
            try:
                from transformers import AutoTokenizer  # pylint: disable=import-error
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=hf_cache_dir,
                )
                LOGGER.info("Loaded tokenizer for chat template formatting: %s", model_path)
            except Exception as exc:
                LOGGER.warning(
                    "Could not load tokenizer for %s: %s. "
                    "Prompts will be passed as raw strings without chat template formatting.",
                    model_path, exc,
                )
            return

        # GPT-OSS models require Harmony prompt formatting for best results.
        try:
            from openai_harmony import (  # pylint: disable=import-error
                HarmonyEncodingName,
                load_harmony_encoding,
                Conversation,
                Message as HarmonyMessage,
                Role as HarmonyRole,
                SystemContent,
                DeveloperContent,
                StreamableParser,
                ReasoningEffort,
            )
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "openai-harmony is required for GPT-OSS offline inference (Harmony prompt formatting). "
                "Install it with: pip install openai-harmony"
            ) from exc

        harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids: List[int] = []
        for attr in ("stop_tokens_for_assistant_actions", "stop_tokens_for_assistant_action"):
            if hasattr(harmony_encoding, attr):
                try:
                    stop_token_ids = list(getattr(harmony_encoding, attr)() or [])
                    break
                except Exception:
                    pass

        self._harmony_encoding = harmony_encoding
        self._harmony_stop_token_ids = stop_token_ids
        self._harmony = {
            "Conversation": Conversation,
            "Message": HarmonyMessage,
            "Role": HarmonyRole,
            "SystemContent": SystemContent,
            "DeveloperContent": DeveloperContent,
            "StreamableParser": StreamableParser,
            "ReasoningEffort": ReasoningEffort,
            "reasoning_effort": reasoning_effort,
        }
    
    def close(self):
        """Close the client and release resources.
        
        This should be called when done with the client to ensure proper cleanup
        of HTTP connection pools and background threads.
        """
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        if hasattr(self, 'async_client') and self.async_client is not None:
            try:
                self.async_client.close()
            except Exception:
                pass
        # Best-effort cleanup for offline vLLM.
        if hasattr(self, "_vllm_engine"):
            try:
                self._vllm_engine = None
            except Exception:
                pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def _default_postprocess(self, text: str) -> str:
        """Default post-processing for model output."""
        if text is None:
            return ""
        text = text.strip()
        
        # Handle thinking tokens (for models like Qwen3, DeepSeek-R1)
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        
        # Handle gpt-oss-120b style tokens (analysis/final/assistant without angle brackets)
        # These appear when special tokens are decoded without proper handling
        # Pattern: look for "final" marker and extract content after it
        # Handles both "<final>content</final>" and "finalcontent" (stripped brackets)
        if "</final>" in text:
            # Standard XML-style tags
            match = re.search(r'<final>(.*?)</final>', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        elif "final" in text.lower():
            # Avoid corrupting valid JSON or text that merely contains the word "final".
            # Only strip when the output looks like a special-token sequence (no prose).
            stripped = text.lstrip()
            looks_like_json = stripped.startswith("{") or stripped.startswith("[")
            if not looks_like_json:
                # Handle stripped special tokens: "analysis ... assistant final ...".
                patterns = [
                    r"(?<![A-Za-z0-9_])assistant\s*final(?![A-Za-z0-9_])\s*",
                    r"(?<![A-Za-z0-9_])final(?![A-Za-z0-9_])\s*",
                ]
                for pattern in patterns:
                    match = re.search(pattern, text, flags=re.IGNORECASE)
                    if not match:
                        continue
                    candidate = text[match.end():].strip()
                    # Only accept if the content after "final" looks like
                    # actual payload (starts with JSON or is non-trivial).
                    if candidate and (candidate.startswith("{") or candidate.startswith("[")):
                        text = candidate
                        break
                    # Only strip when the prefix before "final" consists
                    # entirely of known special-token words (e.g. "analysis",
                    # "assistant").  This prevents corrupting normal prose
                    # that happens to contain "final" (e.g. "The final
                    # answer is: Yes" from judge/filter stages).
                    prefix = text[:match.start()].strip()
                    if not prefix or re.fullmatch(
                        r'(?:(?:analysis|assistant)\s*)+', prefix, re.IGNORECASE,
                    ):
                        text = candidate
                        break
        
        # Also handle "</analysis>" style tags
        if "</analysis>" in text:
            text = text.split("</analysis>")[-1].strip()
        
        # Clean up any remaining assistant markers
        text = re.sub(r'^(assistant|</assistant>)\s*', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _postprocess_output(self, text: str) -> str:
        """Post-process model output."""
        text = self._default_postprocess(text)
        if self._postprocess_fn:
            text = self._postprocess_fn(text)
        return text

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 64000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            top_k: Top-k sampling threshold (-1 = disabled)
            enable_thinking: Override instance-level enable_thinking (None = use default)

        Returns:
            List of generated responses
        """
        if self.provider == ModelProvider.VLLM_OFFLINE:
            return self._generate_vllm_offline(prompts, max_tokens, temperature, top_p=top_p, top_k=top_k, enable_thinking=enable_thinking)
        elif self.provider == ModelProvider.OPENROUTER:
            return self._generate_openrouter(prompts, max_tokens, temperature, top_p=top_p)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_max_concurrent_requests(self) -> int:
        """Get the maximum number of concurrent requests based on provider.
        
        Cloud APIs can handle high concurrency, local models should be limited.
        """
        # Cloud API providers - high concurrency
        cloud_providers = {
            ModelProvider.OPENROUTER,
        }
        
        if self.provider in cloud_providers:
            return 128  # Cloud APIs can handle many concurrent requests
        else:
            # vLLM offline - limited concurrency
            return 12  # Conservative for local inference
    
    @staticmethod
    def _extract_response_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                elif hasattr(item, "text"):
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts).strip()
        if content is None:
            return ""
        return str(content)

    def _generate_openrouter(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: Optional[float],
        json_mode: bool = False,
        response_format: Optional[Any] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_exclude: Optional[bool] = None,
        verbosity: Optional[str] = None,
        top_p: Optional[float] = None,
        on_progress: Optional[Callable[[int, Any], None]] = None,
        request_timeout: Optional[float] = None,
    ) -> List[Any]:
        """Generate using OpenRouter API with async batching and concurrency limits.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature. If None, omit the parameter.
            json_mode: If True, request JSON object response
            response_format: Optional structured output schema (best effort)
            reasoning_effort: Optional OpenRouter reasoning effort
            reasoning_enabled: Optional OpenRouter reasoning.enabled flag
            reasoning_max_tokens: Optional OpenRouter reasoning.max_tokens budget
            reasoning_exclude: Whether OpenRouter should omit returned reasoning text
            verbosity: Optional OpenRouter verbosity parameter
        
        Returns:
            List of responses (strings or parsed Pydantic objects)
        """
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        
        # Check if we need structured output (Pydantic model)
        use_structured = response_format is not None and hasattr(response_format, 'model_fields')
        model_name = self.config.model_name or self.model_path
        
        # OpenRouter's Chat Completions schema uses max_tokens. For reasoning
        # models, OpenRouter derives the reasoning budget from max_tokens when
        # reasoning.effort is provided.
        common_params = {"max_tokens": max_tokens}
        if temperature is not None:
            common_params["temperature"] = temperature
        if top_p is not None:
            common_params["top_p"] = top_p

        extra_body: dict = {}
        if reasoning_effort or reasoning_enabled is not None or reasoning_max_tokens is not None:
            reasoning_config: dict = {}
            if reasoning_enabled is not None:
                reasoning_config["enabled"] = bool(reasoning_enabled)
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            if reasoning_max_tokens is not None:
                reasoning_config["max_tokens"] = int(reasoning_max_tokens)
            if reasoning_exclude is not None:
                reasoning_config["exclude"] = bool(reasoning_exclude)
            extra_body["reasoning"] = reasoning_config
        if verbosity:
            extra_body["verbosity"] = verbosity
        if extra_body:
            common_params["extra_body"] = extra_body
        
        # Limit concurrent requests based on provider type
        MAX_CONCURRENT = self._get_max_concurrent_requests()
        
        async def _generate_single_async(messages: List[dict], semaphore: asyncio.Semaphore, max_retries: int = 5):
            """Generate a single response with retry logic under semaphore."""
            async with semaphore:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        if use_structured:
                            coro = self.async_client.beta.chat.completions.parse(
                                model=model_name,
                                messages=messages,
                                response_format=response_format,
                                **common_params,
                            )
                        elif json_mode:
                            coro = self.async_client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                response_format={"type": "json_object"},
                                **common_params,
                            )
                        else:
                            coro = self.async_client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                **common_params,
                            )
                        if request_timeout is not None:
                            completion = await asyncio.wait_for(coro, timeout=request_timeout)
                        else:
                            completion = await coro
                        return completion
                    except asyncio.TimeoutError as e:
                        last_error = e
                        wait_time = min(2 ** attempt + 1, 60)
                        LOGGER.warning(
                            "Per-request timeout (>%.1fs) on attempt %d/%d. Retrying in %ds...",
                            request_timeout if request_timeout is not None else -1,
                            attempt + 1, max_retries, wait_time,
                        )
                        await asyncio.sleep(wait_time)
                    except (APIConnectionError, APITimeoutError) as e:
                        last_error = e
                        wait_time = min(2 ** attempt + 1, 60)
                        LOGGER.warning("Connection error on attempt %d/%d: %s. "
                                       "Retrying in %ds...", attempt + 1, max_retries, type(e).__name__, wait_time)
                        await asyncio.sleep(wait_time)
                    except APIStatusError as e:
                        if e.status_code in (429, 500, 502, 503, 504):  # Retryable errors
                            last_error = e
                            wait_time = min(2 ** attempt + 1, 60)
                            LOGGER.warning("Server error %s on attempt %d/%d. "
                                           "Retrying in %ds...", e.status_code, attempt + 1, max_retries, wait_time)
                            await asyncio.sleep(wait_time)
                        else:
                            LOGGER.error("API error: %s: %s", e.status_code, e)
                            return e
                    except Exception as e:
                        LOGGER.error("Unexpected error: %s: %s", type(e).__name__, e)
                        return e
                
                LOGGER.error("All %d retries exhausted. Last error: %s: %s", max_retries, type(last_error).__name__, last_error)
                return last_error
        
        try:
            from tqdm.auto import tqdm as _tqdm  # type: ignore
        except Exception:  # tqdm is a soft dependency
            _tqdm = None

        def _normalize(completion: Any) -> Any:
            """Map a raw completion (or exception) to its final response value."""
            if isinstance(completion, Exception):
                LOGGER.error("Error in OpenRouter API call: %s", completion)
                return {"error": str(completion)} if use_structured else ""
            if not getattr(completion, "choices", None):
                error = getattr(completion, "error", None)
                LOGGER.error(
                    "OpenRouter response contained no choices. error=%s completion=%s",
                    error, completion,
                )
                if use_structured:
                    return {"error": "OpenRouter response contained no choices", "details": str(error)}
                return ""
            if use_structured:
                parsed = completion.choices[0].message.parsed
                if parsed is not None:
                    return parsed
                refusal = completion.choices[0].message.refusal
                return {"error": "Couldn't parse output", "refusal": refusal}
            choice = completion.choices[0]
            choice_error = getattr(choice, "error", None)
            if choice_error:
                LOGGER.error("OpenRouter choice error: %s", choice_error)
                return ""
            message = getattr(choice, "message", None)
            if message is None:
                LOGGER.error("OpenRouter choice contained no message: %s", choice)
                return ""
            text = self._extract_response_text(message.content)
            return self._postprocess_output(text)

        async def _batch_generate_ordered():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            all_messages = [self._prompt_to_messages(p) for p in prompts]

            async def _indexed(i: int, msgs):
                r = await _generate_single_async(msgs, semaphore)
                return (i, r)

            tasks = [
                asyncio.create_task(_indexed(i, msgs))
                for i, msgs in enumerate(all_messages)
            ]
            results: List[Any] = [None] * len(tasks)
            iterator = asyncio.as_completed(tasks)
            if _tqdm is not None and len(tasks) > 1:
                iterator = _tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"OpenRouter {model_name}",
                    unit="req",
                )
            for fut in iterator:
                i, raw = await fut
                normalized = _normalize(raw)
                results[i] = normalized
                if on_progress is not None:
                    try:
                        on_progress(i, normalized)
                    except Exception:  # never let a callback break the batch
                        LOGGER.exception("on_progress callback raised; continuing.")
            return results

        # Run async batch (ordered via index wrapper so output matches input).
        # Each completion is normalized inside the loop so on_progress sees the
        # final response value (string for plain mode, dict for structured).
        return asyncio.run(_batch_generate_ordered())

    def _to_harmony_reasoning_effort(self, effort: Optional[str]):
        if not effort or effort == "none":
            return None
        if not self._harmony:
            return None
        reff = self._harmony["ReasoningEffort"]
        e = effort.strip().lower()
        if e in ("minimal", "low"):
            return reff.LOW
        if e == "medium":
            return reff.MEDIUM
        if e in ("high", "xhigh"):
            return reff.HIGH
        return None

    def _safe_set_reasoning_effort_on_system(self, system_content: Any, effort: Optional[str]) -> Any:
        reff = self._to_harmony_reasoning_effort(effort)
        if reff is None:
            return system_content
        for method_name in ("with_reasoning_effort", "with_reasoning"):
            if hasattr(system_content, method_name):
                try:
                    return getattr(system_content, method_name)(reff)
                except Exception:
                    pass
        return system_content

    def _to_harmony_conversation(self, prompt: Any, reasoning_effort_override: Optional[str] = None) -> Any:
        if not self._harmony or not self._harmony_encoding:
            raise RuntimeError("Harmony is not initialized.")

        HarmonyMessage = self._harmony["Message"]
        HarmonyRole = self._harmony["Role"]
        Conversation = self._harmony["Conversation"]
        SystemContent = self._harmony["SystemContent"]
        DeveloperContent = self._harmony["DeveloperContent"]
        reasoning_effort = (
            reasoning_effort_override
            if reasoning_effort_override is not None
            else self._harmony["reasoning_effort"]
        )

        messages: List[Any] = []

        system_content = SystemContent.new()
        system_content = self._safe_set_reasoning_effort_on_system(system_content, reasoning_effort)
        if hasattr(system_content, "with_required_channels"):
            try:
                system_content = system_content.with_required_channels(["final"])
            except Exception:
                pass
        messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, system_content))

        if isinstance(prompt, str):
            # Split on === separator to extract developer instructions vs user content
            chat_msgs = self._prompt_to_messages(prompt)
            for msg in chat_msgs:
                role = (msg.get("role") or "user").lower()
                content = msg.get("content") or ""
                if role == "system":
                    # Inject as a Harmony developer message (system slot is already taken)
                    try:
                        dc = DeveloperContent.new()
                        if content.strip() and hasattr(dc, "with_instructions"):
                            dc = dc.with_instructions(content)
                        messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, dc))
                    except Exception:
                        messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
                else:
                    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))
            return Conversation.from_messages(messages)

        # prompt can optionally be a chat-style list of dicts: [{"role": "...", "content": "..."}]
        for msg in prompt:
            role = (msg.get("role") or "user").lower()
            content = msg.get("content") or ""
            if role in ("system", "developer"):
                try:
                    dc = DeveloperContent.new()
                    if content.strip() and hasattr(dc, "with_instructions"):
                        dc = dc.with_instructions(content)
                    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, dc))
                    continue
                except Exception:
                    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
            elif role == "assistant":
                messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content))
            else:
                messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))

        return Conversation.from_messages(messages)

    @staticmethod
    def _harmony_dict_content_to_text(msg_dict: dict) -> str:
        content = msg_dict.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip()
        return ""

    def _extract_final_only_from_harmony_tokens(
        self,
        *,
        completion_token_ids: List[int],
        stop_token_ids: Optional[List[int]] = None,
    ) -> str:
        if not self._harmony_encoding or not self._harmony:
            return ""

        ids = list(completion_token_ids or [])

        if stop_token_ids:
            stop_set = set(stop_token_ids)
            while ids and ids[-1] in stop_set:
                ids.pop()

        StreamableParser = self._harmony["StreamableParser"]
        HarmonyRole = self._harmony["Role"]
        parser = StreamableParser(self._harmony_encoding, role=HarmonyRole.ASSISTANT)
        for tid in ids:
            parser.process(tid)

        finals: List[str] = []
        non_final_texts: List[str] = []

        for msg in getattr(parser, "messages", []) or []:
            try:
                msg_dict = msg.to_dict()
            except Exception:
                continue
            text = self._harmony_dict_content_to_text(msg_dict)
            if not text:
                continue
            if msg_dict.get("channel") == "final":
                finals.append(text)
            else:
                non_final_texts.append(text)

        if getattr(parser, "current_channel", None) == "final":
            current = getattr(parser, "current_content", None)
            if isinstance(current, str) and current.strip():
                finals.append(current.strip())
        else:
            current = getattr(parser, "current_content", None)
            if isinstance(current, str) and current.strip():
                non_final_texts.append(current.strip())

        if finals:
            return "\n".join(finals).strip()

        return ""

    def _generate_vllm_offline(
        self,
        prompts: List[Any],
        max_tokens: int,
        temperature: float,
        reasoning_effort_override: Optional[str] = None,
        top_p: float = 1.0,
        top_k: int = -1,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        if not hasattr(self, "_vllm_engine") or self._vllm_engine is None:
            raise RuntimeError("Offline vLLM engine is not initialized.")

        SamplingParams = getattr(self, "_vllm_SamplingParams", None)
        if SamplingParams is None:
            raise RuntimeError("Offline vLLM SamplingParams is not initialized.")

        stop_ids = self._harmony_stop_token_ids if self._harmony_stop_token_ids else []
        sp_kwargs = {
            "temperature": temperature if temperature is not None else 1.0,
            "max_tokens": max_tokens if max_tokens is not None else 64000,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "top_p": top_p,
            "top_k": top_k,
        }
        if stop_ids:
            sp_kwargs["stop_token_ids"] = stop_ids
        sp = SamplingParams(**sp_kwargs)

        if self._is_gpt_oss_model(self.model_path):
            if not self._harmony_encoding or not self._harmony:
                raise RuntimeError("Harmony is not initialized for GPT-OSS offline inference.")

            prompt_objs: List[dict] = []
            HarmonyRole = self._harmony["Role"]
            for p in prompts:
                convo = self._to_harmony_conversation(p, reasoning_effort_override=reasoning_effort_override)
                prefill_ids = self._harmony_encoding.render_conversation_for_completion(convo, HarmonyRole.ASSISTANT)
                prompt_objs.append({"prompt_token_ids": prefill_ids})

            outputs = self._vllm_engine.generate(prompt_objs, sampling_params=sp)

            results: List[str] = []
            for out in outputs:
                gen = out.outputs[0]
                completion_ids = getattr(gen, "token_ids", None) or []
                raw_text = (getattr(gen, "text", None) or "").strip()
                try:
                    final_text = self._extract_final_only_from_harmony_tokens(
                        completion_token_ids=completion_ids,
                        stop_token_ids=stop_ids,
                    )
                except Exception:
                    final_text = self._strip_harmony_special_tokens(raw_text)
                if not final_text.strip():
                    final_text = self._strip_harmony_special_tokens(raw_text)
                results.append(self._postprocess_output(final_text))
            return results

        # Apply chat template if tokenizer is available (non-GPT-OSS chat models)
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        if self.tokenizer is not None:
            formatted_prompts = []
            for p in prompts:
                messages = self._prompt_to_messages(p)
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        enable_thinking=thinking,
                        tokenize=False,
                    )
                except TypeError:
                    # Fallback: tokenizer's template doesn't accept enable_thinking
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                formatted_prompts.append(formatted)
            outputs = self._vllm_engine.generate(formatted_prompts, sampling_params=sp)
        else:
            outputs = self._vllm_engine.generate(prompts, sampling_params=sp)

        results: List[str] = []
        for out in outputs:
            text = (out.outputs[0].text or "").strip()
            results.append(self._postprocess_output(text))
        return results
    
    # Convenience methods
    def __call__(
        self, 
        prompts: List[str], 
        max_tokens: int = 64000,
        temperature: float = 0.0,
    ) -> List[str]:
        """Shorthand for generate()."""
        return self.generate(prompts, max_tokens=max_tokens, temperature=temperature)
    
    def interact(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: int = 64000,
        **kwargs
    ) -> Any:
        """Single prompt interaction.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments:
                - json_mode (bool): Request JSON object response (OpenRouter only)
                - response_format: Pydantic model for structured output (OpenRouter only)
                - history: Conversation history (not fully supported)
                - system_prompt: System prompt (not fully supported)
        
        Returns:
            Generated response (string or parsed Pydantic object for structured output)
        """
        json_mode = kwargs.get('json_mode', False)
        response_format = kwargs.get('response_format')
        reasoning_effort = kwargs.get('reasoning_effort', self.reasoning_effort)
        reasoning_enabled = kwargs.get('reasoning_enabled', self.reasoning_enabled)
        reasoning_max_tokens = kwargs.get('reasoning_max_tokens', self.reasoning_max_tokens)
        reasoning_exclude = kwargs.get('reasoning_exclude', self.reasoning_exclude)
        verbosity = kwargs.get('verbosity', self.verbosity)
        enable_thinking = kwargs.get('enable_thinking', None)

        top_p = kwargs.get('top_p', 1.0)
        top_k = kwargs.get('top_k', -1)

        # Log warning for unsupported kwargs
        supported_kwargs = {
            'history',
            'json_mode',
            'response_format',
            'system_prompt',
            'reasoning_effort',
            'reasoning_enabled',
            'reasoning_max_tokens',
            'reasoning_exclude',
            'verbosity',
            'enable_thinking',
            'top_p',
            'top_k',
        }
        unsupported = set(kwargs.keys()) - supported_kwargs
        if unsupported:
            LOGGER.warning("Unsupported arguments ignored: %s", unsupported)
        
        # Handle structured output for OpenRouter
        if (json_mode or response_format) and self.provider == ModelProvider.OPENROUTER:
            return self._generate_openrouter(
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                reasoning_enabled=reasoning_enabled,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_exclude=reasoning_exclude,
                verbosity=verbosity,
                top_p=top_p,
            )[0]
        
        # Warn if structured output requested for non-OpenRouter provider
        if json_mode or response_format:
            if self.provider != ModelProvider.OPENROUTER:
                if not self._warned_structured_unsupported:
                    LOGGER.warning(
                        "Structured output (json_mode/response_format) not supported for %s. Using standard generation.",
                        self.provider.value,
                    )
                    self._warned_structured_unsupported = True
        
        # Handle history (not fully supported, just log)
        if kwargs.get('history'):
            LOGGER.debug("History parameter not fully supported in UnifiedModelClient")

        if (
            self.provider == ModelProvider.OPENROUTER
            and (
                reasoning_effort is not None
                or reasoning_enabled is not None
                or reasoning_max_tokens is not None
                or verbosity is not None
            )
        ):
            return self._generate_openrouter(
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                reasoning_enabled=reasoning_enabled,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_exclude=reasoning_exclude,
                verbosity=verbosity,
                top_p=top_p,
            )[0]

        # Allow per-call reasoning effort override for offline GPT-OSS Harmony path.
        if self.provider == ModelProvider.VLLM_OFFLINE and reasoning_effort is not None:
            return self._generate_vllm_offline(
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort_override=reasoning_effort,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )[0]

        return self.generate([prompt], max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, enable_thinking=enable_thinking)[0]

    def batch_interact(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 64000,
        **kwargs
    ) -> List[str]:
        """Batch prompt interaction.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments:
                - json_mode (bool): Request JSON object response (OpenRouter only)
                - top_p (float): Nucleus sampling threshold (vLLM offline only)
                - top_k (int): Top-k sampling threshold (vLLM offline only)
                - reasoning_effort (str): Per-call reasoning effort override (vLLM offline GPT-OSS only)
                - enable_thinking (bool): Override instance-level enable_thinking

        Returns:
            List of generated responses
        """
        json_mode = kwargs.get('json_mode', False)
        enable_thinking = kwargs.get('enable_thinking', None)
        reasoning_effort = kwargs.get('reasoning_effort', self.reasoning_effort)
        reasoning_enabled = kwargs.get('reasoning_enabled', self.reasoning_enabled)
        reasoning_max_tokens = kwargs.get('reasoning_max_tokens', self.reasoning_max_tokens)
        reasoning_exclude = kwargs.get('reasoning_exclude', self.reasoning_exclude)
        verbosity = kwargs.get('verbosity', self.verbosity)
        top_p = kwargs.get('top_p', 1.0)
        top_k = kwargs.get('top_k', -1)
        on_progress = kwargs.get('on_progress', None)
        request_timeout = kwargs.get('request_timeout', None)

        if json_mode and self.provider == ModelProvider.OPENROUTER:
            return self._generate_openrouter(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
                reasoning_effort=reasoning_effort,
                reasoning_enabled=reasoning_enabled,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_exclude=reasoning_exclude,
                verbosity=verbosity,
                top_p=top_p,
                on_progress=on_progress,
                request_timeout=request_timeout,
            )

        if (
            self.provider == ModelProvider.OPENROUTER
            and (
                reasoning_effort is not None
                or reasoning_enabled is not None
                or reasoning_max_tokens is not None
                or verbosity is not None
            )
        ):
            return self._generate_openrouter(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                reasoning_enabled=reasoning_enabled,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_exclude=reasoning_exclude,
                verbosity=verbosity,
                top_p=top_p,
                on_progress=on_progress,
                request_timeout=request_timeout,
            )

        if self.provider == ModelProvider.VLLM_OFFLINE and reasoning_effort is not None:
            return self._generate_vllm_offline(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort_override=reasoning_effort,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )

        return self.generate(prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, enable_thinking=enable_thinking)



def load_model(
    model_name: str,
    **kwargs
) -> UnifiedModelClient:
    """Load a model by name (backward compatibility wrapper).

    Args:
        model_name: Model name
        **kwargs: Additional arguments passed to UnifiedModelClient

    Returns:
        UnifiedModelClient instance
    """
    return UnifiedModelClient(model_name, **kwargs)
