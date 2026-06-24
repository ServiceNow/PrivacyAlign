from __future__ import annotations


class WorkerWrap:
    """OpenRLHF-style worker extension for offline Ray-hosted vLLM engines."""

    @staticmethod
    def _should_trim_cuda_cache(device) -> bool:
        import torch

        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            return False
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        idle_bytes = max(0, reserved - allocated)
        if reserved <= 0:
            return False
        return idle_bytes >= 512 * 1024 * 1024 and idle_bytes / reserved >= 0.25

    def _normalize_incoming_weight_name(self, name: str) -> str:
        model = self.model_runner.model
        weights_mapper = getattr(model, "hf_to_vllm_mapper", None)
        if weights_mapper is not None:
            name = weights_mapper._map_name(name)

        # The training side loads text-only HF models, while vLLM may expose a
        # multimodal wrapper whose text backbone lives under `language_model.*`.
        if hasattr(model, "language_model") and not hasattr(model, "model"):
            text_prefix_map = (
                ("model.embed_tokens.", "language_model.model.embed_tokens."),
                ("model.layers.", "language_model.model.layers."),
                ("model.norm.", "language_model.model.norm."),
                ("lm_head.", "language_model.lm_head."),
            )
            for src_prefix, dst_prefix in text_prefix_map:
                if name.startswith(src_prefix):
                    return f"{dst_prefix}{name[len(src_prefix):]}"

        return name

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        import torch

        from ray_backend.vllm_utils import stateless_init_process_group

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        target_device = getattr(self, "device", None)
        if target_device is None:
            target_device = torch.device("cuda")
        if not isinstance(target_device, torch.device):
            target_device = torch.device(target_device)
        if target_device.type == "cuda":
            torch.cuda.set_device(target_device)

        # Try to broadcast directly into the model's existing parameter buffer
        # to avoid allocating a temporary tensor that doubles memory for large params.
        vllm_name = self._normalize_incoming_weight_name(name)
        existing_param = dict(self.model_runner.model.named_parameters()).get(vllm_name)
        if (
            existing_param is not None
            and existing_param.data.shape == tuple(shape)
            and existing_param.data.dtype == dtype
            and existing_param.data.is_contiguous()
        ):
            self._model_update_group.broadcast(existing_param.data, src=0, stream=torch.cuda.current_stream())
        else:
            weight = torch.empty(shape, dtype=dtype, device=target_device)
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())
            self.model_runner.model.load_weights(weights=[(vllm_name, weight)])
            del weight

        if empty_cache and self._should_trim_cuda_cache(target_device):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch

        from ray_backend.vllm_utils import get_physical_gpu_id

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        _STORAGE_DEVICE_INDEX = 6
        if not isinstance(list_args[_STORAGE_DEVICE_INDEX], int):
            raise RuntimeError(
                f"Expected an integer device index at position {_STORAGE_DEVICE_INDEX} in CUDA IPC "
                f"rebuild args, but got {type(list_args[_STORAGE_DEVICE_INDEX]).__name__}. "
                f"The PyTorch CUDA tensor serialization format may have changed."
            )
        list_args[_STORAGE_DEVICE_INDEX] = device_id
        weight = func(*list_args)
        try:
            self.model_runner.model.load_weights(
                weights=[(self._normalize_incoming_weight_name(name), weight)]
            )
            torch.cuda.synchronize()
        finally:
            del weight
        if empty_cache and self._should_trim_cuda_cache(self.device):
            torch.cuda.empty_cache()
