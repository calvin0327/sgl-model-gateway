"""Expose the initialized vLLM scheduler's layout, without running inference."""

import asyncio
import json
import logging
import uuid
from typing import Any

METADATA_PATH = "/v1/pre-kv-cache/metadata"
logger = logging.getLogger(__name__)


def get_pre_kv_cache_metadata(core: Any) -> dict[str, Any]:
    # Read the scheduler and its managers: model config and observed events do
    # not describe the complete, effective runtime cache layout.
    from vllm.v1.kv_cache_interface import (
        get_kv_cache_spec_kind,
        get_kv_cache_spec_sliding_window,
    )
    from vllm.version import __version__

    parallel = core.vllm_config.parallel_config
    if parallel.data_parallel_size != 1:
        raise ValueError("KV metadata currently requires data_parallel_size=1")
    scheduler = core.scheduler
    cache = scheduler.kv_cache_config
    cache_manager = scheduler.kv_cache_manager
    coordinator = cache_manager.coordinator
    groups = cache.kv_cache_groups
    managers = coordinator.single_type_managers
    if not groups or len(groups) != len(managers):
        raise ValueError("Initialized KV groups and managers are missing or inconsistent")
    # The hybrid scheduler's lookup order and equal-spec grouping influence
    # speculative tail drops, so preserve those rather than sorting groups.
    if hasattr(coordinator, "attention_groups"):
        lookup_groups = [list(group.group_ids) for group in coordinator.attention_groups]
    elif len(groups) == 1:
        lookup_groups = [[0]] if groups[0].kv_cache_spec.prefix_cacheable else []
    else:
        raise ValueError("Runtime hybrid KV lookup groups are unavailable")

    rows = []
    for group_idx, (group, manager) in enumerate(zip(groups, managers)):
        spec = group.kv_cache_spec
        rows.append(
            {
                "group_idx": group_idx,
                "kind": get_kv_cache_spec_kind(spec).value,
                "block_size": manager.block_size,
                "sliding_window": get_kv_cache_spec_sliding_window(spec),
                "manager": type(manager).__name__,
                "prefix_cacheable": spec.prefix_cacheable,
                "prefix_replay_tokens": spec.prefix_replay_tokens,
                "use_eagle": manager.use_eagle,
                "spec_block_size": spec.block_size,
                "spec": type(spec).__name__,
                "is_eagle_group": group.is_eagle_group,
            }
        )
    speculative = core.vllm_config.speculative_config
    result = {
        "schema_version": 1,
        "vllm_version": __version__,
        "coordinator": type(coordinator).__name__,
        "hash_block_size": coordinator.block_pool.hash_block_size,
        "alignment_tokens": getattr(
            coordinator, "_cache_hit_alignment_tokens", coordinator.scheduler_block_size
        ),
        "scheduler_block_size": coordinator.scheduler_block_size,
        "dcp_world_size": parallel.decode_context_parallel_size,
        "pcp_world_size": parallel.prefill_context_parallel_size,
        "data_parallel_size": parallel.data_parallel_size,
        "enable_partial_hash_hits": coordinator.enable_partial_hash_hits,
        "enable_prefix_caching": cache_manager.enable_caching,
        "enable_kv_cache_events": cache_manager.enable_kv_cache_events,
        "num_reprefillable_tokens": coordinator.num_reprefillable_tokens,
        "prefix_replay_tokens": scheduler.prefix_replay_tokens,
        "prefix_cache_retention_interval": cache.prefix_cache_retention_interval,
        "speculative_method": speculative.method if speculative else None,
        "num_speculative_tokens": speculative.num_speculative_tokens if speculative else 0,
        "lookup_groups": lookup_groups,
        "groups": rows,
    }

    # A new engine/scheduler/layout must invalidate the Router's previous
    # residency index. Keep the identity on the core, not in the API process.
    signature = json.dumps(result, sort_keys=True)
    previous = getattr(core, "_smg_kv_metadata_identity", None)
    if previous is None or previous[0] is not scheduler or previous[1] != signature:
        previous = (scheduler, signature, str(uuid.uuid4()))
        core._smg_kv_metadata_identity = previous
    result["engine_id"] = previous[2]
    return result


def register() -> None:
    # General plugins load in both API and engine processes. Add only this
    # read-only utility; vLLM's existing IPC transports the resulting JSON.
    from vllm.v1.engine.core import EngineCore

    current = getattr(EngineCore, "get_pre_kv_cache_metadata", None)
    if current is not None and current is not get_pre_kv_cache_metadata:
        raise RuntimeError("Another plugin registered get_pre_kv_cache_metadata")
    EngineCore.get_pre_kv_cache_metadata = get_pre_kv_cache_metadata


class KVMetadataMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Inference requests take the original ASGI path, without buffering,
        # an extra task, or a metadata RPC. Only intercept our dedicated route.
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if scope["type"] != "http" or path != METADATA_PATH:
            await self.app(scope, receive, send)
            return

        # Run the existing application first: its API-key/auth middleware must
        # see the request. Preserve all responses except the missing-route 404.
        missing = False

        async def intercept(message: Any) -> None:
            nonlocal missing
            if message["type"] == "http.response.start":
                missing = message["status"] == 404
            if not missing:
                await send(message)

        await self.app(scope, receive, intercept)
        if missing:
            response = await _metadata_response(scope["app"].state, scope["method"])
            await response(scope, receive, send)


async def _metadata_response(state: Any, method: str) -> Any:
    from starlette.responses import JSONResponse

    headers = {"Cache-Control": "no-store"}
    if method != "GET":
        return JSONResponse({"error": "Use GET"}, 405, headers={**headers, "Allow": "GET"})
    try:
        if state.vllm_config.parallel_config.data_parallel_size != 1:
            raise ValueError("KV metadata currently requires data_parallel_size=1")
        result = await asyncio.wait_for(
            state.engine_client.engine_core.call_utility_async("get_pre_kv_cache_metadata"),
            timeout=10,
        )
    except Exception as error:
        logger.warning("Unable to read runtime KV metadata: %s", error)
        return JSONResponse(
            {"error": "runtime_kv_metadata_unavailable", "detail": str(error)},
            503,
            headers=headers,
        )
    return JSONResponse(result, headers=headers)
