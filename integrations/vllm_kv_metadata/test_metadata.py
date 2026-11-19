"""The bridge must report runtime layout and preserve the server's auth result."""

import copy
import json
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import smg_vllm_kv_metadata as bridge


def runtime_core():
    # Runtime manager settings deliberately differ from the model's group spec.
    spec = NS(kind="full_attention", block_size=64, prefix_cacheable=True, prefix_replay_tokens=0)
    manager = type("FullAttentionManager", (), {})()
    manager.block_size = 128
    manager.use_eagle = True
    coordinator = NS(
        single_type_managers=[manager],
        block_pool=NS(hash_block_size=64),
        scheduler_block_size=256,
        _cache_hit_alignment_tokens=128,
        enable_partial_hash_hits=False,
        num_reprefillable_tokens=4,
    )
    scheduler = NS(
        prefix_replay_tokens=128,
        kv_cache_config=NS(
            kv_cache_groups=[NS(kv_cache_spec=spec, is_eagle_group=False)],
            prefix_cache_retention_interval=0,
        ),
        kv_cache_manager=NS(
            coordinator=coordinator, enable_caching=True, enable_kv_cache_events=True
        ),
    )
    config = NS(
        parallel_config=NS(
            data_parallel_size=1,
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
        ),
        speculative_config=NS(method="dspark", num_speculative_tokens=5),
    )
    return NS(scheduler=scheduler, vllm_config=config)


class RuntimeMetadataTests(unittest.TestCase):
    def setUp(self):
        # vLLM is intentionally imported only inside its plugin/runtime hooks.
        self.modules = patch.dict(
            "sys.modules",
            {
                "vllm.v1.kv_cache_interface": NS(
                    get_kv_cache_spec_kind=lambda spec: NS(value=spec.kind),
                    get_kv_cache_spec_sliding_window=lambda spec: getattr(spec, "sliding_window", None),
                ),
                "vllm.version": NS(__version__="0.31.0-test"),
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.core = runtime_core()

    def test_exports_effective_managers_not_model_defaults(self):
        result = bridge.get_pre_kv_cache_metadata(self.core)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["hash_block_size"], 64)
        self.assertEqual(result["alignment_tokens"], 128)
        self.assertEqual(result["scheduler_block_size"], 256)
        self.assertEqual(result["num_reprefillable_tokens"], 4)
        self.assertEqual(result["prefix_replay_tokens"], 128)
        self.assertEqual(result["dcp_world_size"], 2)
        self.assertEqual(result["speculative_method"], "dspark")
        self.assertEqual(result["num_speculative_tokens"], 5)
        self.assertEqual(result["lookup_groups"], [[0]])
        group = result["groups"][0]
        self.assertEqual(group["block_size"], 128)
        self.assertEqual(group["spec_block_size"], 64)
        self.assertEqual(group["manager"], "FullAttentionManager")
        self.assertTrue(group["use_eagle"])
        self.assertFalse(group["is_eagle_group"])
        json.dumps(result)

    def test_all_groups_exported_including_noncacheable(self):
        cache = self.core.scheduler.kv_cache_config
        coordinator = self.core.scheduler.kv_cache_manager.coordinator
        spec = NS(
            kind="sliding_window", block_size=64, prefix_cacheable=False,
            sliding_window=128, prefix_replay_tokens=128,
        )
        cache.kv_cache_groups.append(NS(kv_cache_spec=spec, is_eagle_group=False))
        coordinator.single_type_managers.append(NS(block_size=64, use_eagle=False))
        coordinator.attention_groups = [NS(group_ids=[0])]
        groups = bridge.get_pre_kv_cache_metadata(self.core)["groups"]
        self.assertEqual([g["group_idx"] for g in groups], [0, 1])
        self.assertFalse(groups[1]["prefix_cacheable"])
        self.assertEqual(groups[1]["sliding_window"], 128)

    def test_hybrid_lookup_order_is_preserved(self):
        cache = self.core.scheduler.kv_cache_config
        coordinator = self.core.scheduler.kv_cache_manager.coordinator
        cache.kv_cache_groups.extend(copy.deepcopy(cache.kv_cache_groups[0]) for _ in range(2))
        coordinator.single_type_managers.extend(copy.deepcopy(coordinator.single_type_managers[0]) for _ in range(2))
        coordinator.attention_groups = [NS(group_ids=[2, 1]), NS(group_ids=[0])]
        result = bridge.get_pre_kv_cache_metadata(self.core)
        self.assertEqual(result["lookup_groups"], [[2, 1], [0]])

    def test_wrapped_spec_window_uses_vllm_semantic_helper(self):
        # The helper can resolve a wrapper that has no sliding_window attribute.
        interface = sys.modules["vllm.v1.kv_cache_interface"]
        with patch.object(interface, "get_kv_cache_spec_sliding_window", return_value=128):
            result = bridge.get_pre_kv_cache_metadata(self.core)
        self.assertEqual(result["groups"][0]["sliding_window"], 128)

    def test_unknown_hybrid_lookup_layout_rejected(self):
        cache = self.core.scheduler.kv_cache_config
        coordinator = self.core.scheduler.kv_cache_manager.coordinator
        cache.kv_cache_groups.append(copy.deepcopy(cache.kv_cache_groups[0]))
        coordinator.single_type_managers.append(copy.deepcopy(coordinator.single_type_managers[0]))
        with self.assertRaisesRegex(ValueError, "lookup groups"):
            bridge.get_pre_kv_cache_metadata(self.core)

    def test_epoch_stable_until_core_scheduler_or_layout_changes(self):
        first = bridge.get_pre_kv_cache_metadata(self.core)["engine_id"]
        self.assertEqual(first, bridge.get_pre_kv_cache_metadata(self.core)["engine_id"])
        fresh = bridge.get_pre_kv_cache_metadata(runtime_core())["engine_id"]
        self.assertNotEqual(first, fresh)
        self.core.scheduler = copy.deepcopy(self.core.scheduler)
        second = bridge.get_pre_kv_cache_metadata(self.core)["engine_id"]
        self.assertNotEqual(first, second)
        self.core.scheduler.kv_cache_manager.coordinator.single_type_managers[0].use_eagle = False
        third = bridge.get_pre_kv_cache_metadata(self.core)["engine_id"]
        self.assertNotEqual(second, third)

    def test_unitary_alignment_and_no_speculation(self):
        del self.core.scheduler.kv_cache_manager.coordinator._cache_hit_alignment_tokens
        self.core.vllm_config.speculative_config = None
        result = bridge.get_pre_kv_cache_metadata(self.core)
        self.assertEqual(result["alignment_tokens"], 256)
        self.assertIsNone(result["speculative_method"])
        self.assertEqual(result["num_speculative_tokens"], 0)

    def test_incomplete_runtime_rejected(self):
        self.core.scheduler.kv_cache_manager.coordinator.single_type_managers.clear()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            bridge.get_pre_kv_cache_metadata(self.core)

    def test_multiple_data_parallel_engines_rejected(self):
        self.core.vllm_config.parallel_config.data_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "data_parallel_size"):
            bridge.get_pre_kv_cache_metadata(self.core)

    def test_plugin_registration_idempotent_and_collision_rejected(self):
        engine = type("EngineCore", (), {})
        with patch.dict("sys.modules", {"vllm.v1.engine.core": NS(EngineCore=engine)}):
            bridge.register()
            bridge.register()
            self.assertIs(engine.get_pre_kv_cache_metadata, bridge.get_pre_kv_cache_metadata)
            engine.get_pre_kv_cache_metadata = lambda self: {}
            with self.assertRaisesRegex(RuntimeError, "Another plugin"):
                bridge.register()


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        core = runtime_core()
        self.rpc = AsyncMock(return_value={"schema_version": 1, "engine_id": "test"})
        self.scope = {
            "type": "http", "method": "GET", "path": bridge.METADATA_PATH,
            "app": NS(state=NS(
                vllm_config=core.vllm_config,
                engine_client=NS(engine_core=NS(call_utility_async=self.rpc)),
            )),
        }
        self.status = 404
        self.sent = []
        self.calls = []

    async def invoke(self):
        async def application(scope, receive, send):
            self.calls.append("application")
            await send({"type": "http.response.start", "status": self.status, "headers": []})
            await send({"type": "http.response.body", "body": b"original"})

        async def send(message):
            self.sent.append(message)

        self.sent.clear()
        await bridge.KVMetadataMiddleware(application)(self.scope, AsyncMock(), send)
        return self.sent[0]["status"], self.sent[1]["body"]

    async def test_reads_utility_only_after_application_authentication(self):
        status, body = await self.invoke()
        self.assertEqual(self.calls, ["application"])
        self.rpc.assert_awaited_once_with("get_pre_kv_cache_metadata")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["engine_id"], "test")
        self.assertIn((b"cache-control", b"no-store"), self.sent[0]["headers"])

    async def test_authentication_denial_and_existing_routes_preserved(self):
        for status in (401, 403, 200, 503):
            with self.subTest(status=status):
                self.status = status
                actual_status, body = await self.invoke()
                self.assertEqual(actual_status, status)
                self.assertEqual(body, b"original")
        self.rpc.assert_not_awaited()

    async def test_unrelated_route_and_non_get_do_not_call_engine(self):
        self.scope["path"] = "/v1/chat/completions"
        status, body = await self.invoke()
        self.assertEqual(status, 404)
        self.assertEqual(body, b"original")
        self.scope["path"] = bridge.METADATA_PATH
        self.scope["method"] = "POST"
        status, _ = await self.invoke()
        self.assertEqual(status, 405)
        self.rpc.assert_not_awaited()

    async def test_inference_passes_original_asgi_channels(self):
        self.scope["path"] = "/v1/chat/completions"
        application, receive, send = AsyncMock(), AsyncMock(), AsyncMock()
        await bridge.KVMetadataMiddleware(application)(self.scope, receive, send)
        application.assert_awaited_once_with(self.scope, receive, send)
        self.rpc.assert_not_awaited()

    async def test_metadata_under_root_path(self):
        self.scope["root_path"] = "/worker"
        self.scope["path"] = "/worker" + bridge.METADATA_PATH
        status, _ = await self.invoke()
        self.assertEqual(status, 200)

    async def test_multi_engine_rejected_before_broadcast(self):
        self.scope["app"].state.vllm_config.parallel_config.data_parallel_size = 2
        with self.assertLogs(bridge.logger, level="WARNING"):
            status, _ = await self.invoke()
        self.assertEqual(status, 503)
        self.rpc.assert_not_awaited()

    async def test_engine_error_returns_diagnostic_without_fake_layout(self):
        self.rpc.side_effect = AttributeError("metadata plugin missing")
        with self.assertLogs(bridge.logger, level="WARNING"):
            status, body = await self.invoke()
        self.assertEqual(status, 503)
        data = json.loads(body)
        self.assertEqual(data["error"], "runtime_kv_metadata_unavailable")
        self.assertNotIn("groups", data)


if __name__ == "__main__":
    unittest.main()
