# vLLM runtime KV metadata

This small integration exposes the initialized scheduler's complete KV group layout.
It reads runtime managers, including effective block sizes and speculative-cache
rules, so users do not configure group IDs. It does not render prompts, tokenize,
run inference, or enable vLLM's development RPC endpoints.

Install in the **vLLM worker** environment before starting vLLM:

```bash
pip install /path/to/integrations/vllm_kv_metadata
```

Keep the existing worker command and add:

```bash
--middleware smg_vllm_kv_metadata.KVMetadataMiddleware
```

vLLM discovers the `smg_kv_metadata` general plugin automatically. If `VLLM_PLUGINS`
is explicitly set to an allowlist, add `smg_kv_metadata` to that list. Continue to
enable prefix caching and KV events. The Gateway reads
`GET /v1/pre-kv-cache/metadata` from each worker in the background; no group list
is needed. Existing vLLM API-key authentication also applies to this endpoint.

The bridge targets the vLLM 0.31 scheduler interfaces. It reports unsupported
interfaces as HTTP 503 instead of fabricating metadata. Each HTTP worker must
have `data_parallel_size=1`; tensor parallelism is supported. Multiple data
parallel engines behind one endpoint need separate event/rank tracking and are
explicitly rejected. Matcher support for the returned manager types is checked
separately by the Gateway.

The engine ID changes when the engine, scheduler, or layout changes. Responses
are never cached by the bridge. Metadata describes layout, not block residency;
the Gateway still needs the corresponding live KV events/replay stream.

Unit tests use fake engine state and the real Starlette response implementation:

```bash
python -m pip install starlette
python -m unittest discover -s integrations/vllm_kv_metadata -p 'test_*.py'
```
