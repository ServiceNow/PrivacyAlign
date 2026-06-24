from __future__ import annotations

import logging
import os
import random
import socket

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ray_backend.ray_utils import wait_for_ray_ref


logger = logging.getLogger(__name__)


class BaseDistributedActor:
    """Minimal distributed actor bootstrap for Ray-managed model workers."""

    def __init__(self, world_size: int, rank: int, master_addr: str | None, master_port: int | None):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()

        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip() -> str:
        address = ray._private.services.get_node_ip_address()
        return address.strip("[]")

    @staticmethod
    def _get_free_port() -> int:
        # Pick from a high, less-trafficked range to avoid collisions with the OS
        # ephemeral port pool (typically 32768-60999 on Linux). Verify each candidate
        # is actually bindable, since the prior approach raced with concurrent processes.
        candidates = random.sample(range(20000, 30000), 100)
        for port in candidates:
            try:
                with socket.socket() as sock:
                    sock.bind(("", port))
                    return port
            except OSError:
                continue
        # Fall back to OS-assigned port if every candidate was taken.
        with socket.socket() as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def get_master_addr_port(self) -> tuple[str, int]:
        return self._master_addr, self._master_port


class RayActorGroup:
    """Small actor-group wrapper mirroring the OpenRLHF scheduling pattern."""

    def __init__(
        self,
        *,
        num_nodes: int,
        num_gpus_per_node: int,
        ray_actor_type: type,
        pg: PlacementGroup | None = None,
        num_gpus_per_actor: float = 1.0,
        resources: dict[str, float] | None = None,
        num_resources_per_node: int | None = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self._ray_actor_type = ray_actor_type
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node
        self._actor_handlers = self._initiate_actors(pg=pg, num_gpus_per_actor=num_gpus_per_actor)

    @property
    def actor_handlers(self) -> list:
        return self._actor_handlers

    def _initiate_actors(self, *, pg: PlacementGroup | None, num_gpus_per_actor: float) -> list:
        world_size = self._num_nodes * self._num_gpus_per_node
        logger.info(
            "Creating Ray actor group: nodes=%s, gpus_per_node=%s, world_size=%s, colocated=%s, actor_gpus=%s.",
            self._num_nodes,
            self._num_gpus_per_node,
            world_size,
            pg is not None,
            num_gpus_per_actor,
        )
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
            if self._resources:
                resource_name = next(iter(self._resources.keys()))
                for bundle in bundles:
                    bundle[resource_name] = self._num_resources_per_node
            pg = placement_group(bundles, strategy="PACK")
            wait_for_ray_ref(pg.ready(), description="training actor placement group scheduling")

        handlers = []
        actor_options = {
            "num_cpus": num_gpus_per_actor,
            "num_gpus": num_gpus_per_actor,
            "resources": self._resources,
        }
        if pg is not None:
            actor_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
            )
        logger.info("Scheduling master Ray training actor.")
        master_actor = self._ray_actor_type.options(**actor_options).remote(world_size, 0, None, None)
        handlers.append(master_actor)

        if world_size == 1:
            return handlers

        master_addr, master_port = wait_for_ray_ref(
            master_actor.get_master_addr_port.remote(),
            description="master Ray training actor bootstrap",
        )
        for rank in range(1, world_size):
            actor_options = {
                "num_cpus": num_gpus_per_actor,
                "num_gpus": num_gpus_per_actor,
                "resources": self._resources,
            }
            if pg is not None:
                actor_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=rank,
                )
            logger.info("Scheduling Ray training actor rank=%s.", rank)
            handlers.append(
                self._ray_actor_type.options(**actor_options).remote(world_size, rank, master_addr, master_port)
            )
        return handlers

    def async_init_model_from_pretrained(self, *args, **kwargs) -> list:
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_run_method(self, method_name: str, *args, **kwargs) -> list:
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs
