# SGLang Model Gateway


## Reliability & Flow Control
- **HTTP Client**: Upstream HTTP client connection settings default to pool idle timeout 50s, connect timeout 10s, max idle connections per host 500, and TCP keepalive 30s. Configure via `--pool-idle-timeout-secs`, `--connect-timeout-secs`, `--pool-max-idle-per-host`, `--tcp-keepalive-secs`, or the corresponding `SMG_*` env vars.
High-performance model routing control and data plane for large-scale LLM deployments. The gateway orchestrates fleets of workers, balances traffic across HTTP and gRPC backends, and exposes OpenAI-compatible APIs with tool integrations—while remaining deeply optimized for the SGLang serving runtime.

## Overview
- Unified control plane for registering, monitoring, and orchestrating prefill, decode, and regular workers across heterogeneous model fleets.
- Data plane that routes requests across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability features.
- Industry-first gRPC pipeline with native Rust tokenization, reasoning, and tool-call execution for high-throughput OpenAI-compatible serving.
- Multi-model inference gateway mode (`--enable-igw`) that runs several routers at once and applies per-model policies.
- Built-in reliability primitives: retries with exponential backoff, circuit breakers, token-bucket rate limiting, and queuing.
- First-class observability with structured logging, OpenTelemetry trace and Prometheus metrics.

### Architecture at a Glance
**Control Plane**
- Worker Manager validates workers, discovers capabilities, and keeps the registry in sync.
- Job Queue serializes background operations (add/remove) and exposes status via `/workers/{worker_id}`.
- Background health checker and load monitor keep circuit breakers and policies informed.
- Optional Kubernetes service discovery keeps the registry aligned with pods.

**Data Plane**
- SGLang HTTP routers for regular and PD (prefill/decode) traffic with policy-aware selection.
- SGLang gRPC router and pipeline that stream tokenized requests through SRT gRPC workers with fully Rust tokenizer, reasoning parser, and tool parser implementations for maximal OpenAI API performance, supporting both single-stage and PD serving topologies.
- Router Manager coordinates multiple router implementations when IGW is enabled.
- Resilience layer delivers token-bucket rate limiting, request queuing, retry executor, and per-worker circuit breakers to keep traffic flowing through failures.
- Advanced load balancing with cache-aware request reuse, load-aware (power-of-two) selection, and per-model policy overrides.

## Feature Highlights
- Multiple load balancing strategies (`random`, `round_robin`, `cache_aware`, `power_of_two`, `bucket`) with DP-aware scheduling.
- Multi-model HTTP serving and inference gateway routing with model-specific policies.
- Prefill/decode disaggregation, including bootstrap port handling and cache-aware merging.
- gRPC routing with fully Rust tokenizer loading, reasoning parser selection, and tool parser integration for OpenAI-compatible endpoints—supporting streaming and non-streaming modes across DeepSeek, Llama, Kimi K2, Qwen, GPT-OSS, Mistral, Step-3, GLM4, GLM4.7 and other reasoning-capable models.
- OpenAI-compatible `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, and `/v1/classify` endpoints.
- **Tokenization APIs**: HTTP endpoints for tokenize (`/v1/tokenize`) and detokenize (`/v1/detokenize`) with batch support; tokenizer management APIs for dynamic registration.
- **Parser endpoints**: Reasoning parser (`/parse/reasoning`) and function call parser (`/parse/function_call`) for separating reasoning content and extracting tool calls.
- Reliability controls: retry with jitter, worker-scoped circuit breakers, token bucket limiter with optional queue, and cache flush APIs.
- Service discovery for regular and PD workloads with independent selectors.
- **Comprehensive observability**: Prometheus metrics across HTTP, router, worker, circuit breaker, retry, and discovery layers; OpenTelemetry tracing with OTLP export; structured logging with request ID propagation.

## Documentation
- **User Guide**: [docs.sglang.io/advanced_features/sgl_model_gateway.html](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
- Additional guides, API references, and deployment patterns are continuously updated alongside SGLang releases.

## Installation

### Docker
Pre-built Docker images are available on Docker Hub with multi-architecture support (x86_64 and ARM64):
```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### Prerequisites
- **Rust and Cargo**
  ```bash
  # Install rustup (Rust installer and version manager)
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # Reload shell environment
  source "$HOME/.cargo/env"

  # Verify installation
  rustc --version
  cargo --version
  ```
- **Python** with `pip` and virtualenv tooling available.

### Rust Binary
```bash
# Build release binary
cargo build --release
```

### Python Package
```bash
pip install maturin

# Fast development mode (debug build, no wheel, instant)
# Uses system OpenSSL (requires libssl-dev/openssl-devel)
cd bindings/python
maturin develop

# Production build (optimized, creates wheel)
# Uses vendored OpenSSL (cross-platform compatibility)
cd bindings/python
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl

# Development build with system OpenSSL (faster)
# Requires: apt install libssl-dev pkg-config (Ubuntu/Debian)
#       or: yum install openssl-devel (RHEL/CentOS)
cd bindings/python
maturin build --release --out dist
pip install --force-reinstall dist/*.whl
```
> **Note:** Python bindings are located in `bindings/python/` with their own Cargo.toml. Use `maturin develop` for fast iteration during development (builds in debug mode and installs directly). Use `maturin build --release --features vendored-openssl` for production wheels with full optimizations (opt-level="z", lto="fat") and cross-platform compatibility. The package uses abi3 support for Python 3.8+ compatibility.

## Checking Version

After installation, verify the installation and check version information:

```bash
# Simple version (Rust binary)
./target/release/sgl-model-gateway --version
# or use aliases
./target/release/smg --version
./target/release/amg --version

# Full version info with build details
./target/release/sgl-model-gateway --version-verbose

# Python CLI
amg --version
amg --version-verbose
python3 -m sglang_router --version
```

The `--version` (or `-V`) flag displays the version string. Use `--version-verbose` for comprehensive build information including Git commit, build time, compiler versions, and platform details.

## Quick Start
### Regular HTTP Routing
- **Rust binary**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```
  `cargo run --release -- …` provides the same behavior during development.
- **Python launcher**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```

### Priority Worker Reservations

Use `--priority-worker-count 1` to reserve one worker per eligible pool for
requests containing the `X-Request-Priority` header (any value, including empty).
The default is `0`, which disables reservations. Even when this option is omitted
or set to `0`, requests containing `X-Request-Priority` still prefer Regular workers
regardless of context length, falling back to PD if no Regular worker can serve
the request.

Reservations apply independently to Regular and Prefill pools for the
requested model and endpoint. Prefill pools are additionally separated by deployment
group, transport, and backend runtime so that pairing constraints remain intact.
When a pool has more healthy workers than the reserved count, priority requests
use the reserved workers and ordinary requests use the remainder. When its
healthy count is less than or equal to the reserved count, all requests share the
pool. Decode workers never count toward the reservation and are always shared
through the existing Decode policy and pairing rules. For example, with
`1P + 3D` and a reservation of `1`, Priority isolation is disabled for PD;
with `3P + 1D`, one Prefill is reserved and the Decode is shared by all requests.

The lexicographically first worker URLs in each eligible pool are reserved.
Assignments are recomputed as discovery membership or health changes; no fixed
worker count or URL configuration is required. Filtering happens before context
preferences and load balancing, so ordinary requests cannot fall back into a
reserved pool while isolation is active. Existing Regular/PD mode preferences
and fallback remain in effect within the allowed candidates. A single candidate
continues to bypass the balancing policy. Changes affect new scheduling attempts,
not requests already running on a worker.

### vLLM HTTP Routing
Add `--backend vllm` to mark URL-configured workers as vLLM workers:
```bash
./target/release/sgl-model-gateway \
  --backend vllm \
  --worker-urls http://vllm1:8000 http://vllm2:8000
```
vLLM workers advertise OpenAI-compatible chat, completions, responses, and tool capabilities, but not the SGLang-native `/generate` endpoint. Load-aware routing reads `vllm:num_requests_running` and `vllm:num_requests_waiting` from `/metrics`.

For vLLM PD, configure the same backend and connector:
```bash
./target/release/sgl-model-gateway \
  --backend vllm \
  --pd-disaggregation \
  --prefill http://prefill:30001 8998 \
  --decode http://decode:30011 \
  --kv-connector nixl
```
Use `--kv-connector mooncake --kv-engine-id <id>` for Mooncake deployments with a reported engine ID. The vLLM PD path dispatches prefill first and then decode with `kv_transfer_params`; requests with `n > 1` skip KV relay because the handoff is single-consumer.

### Prefill/Decode Disaggregation (PD)
- **Rust binary**
  ```bash
  ./target/release/sgl-model-gateway \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
  ```
- **Python launcher**
  ```bash
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware
  ```
Prefill entries accept an optional bootstrap port. PD mode merges prefill metadata with decode outputs and streams results back to the client.

### Multi-Model Inference Gateway
Enable IGW mode to route multiple models through a single router while applying per-model policies:
```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "mistral",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'

# Add another worker with a different model/policy hint
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-b:8000",
        "model_id": "llama3",
        "priority": 20,
        "labels": {"policy": "power_of_two", "tier": "silver"}
      }'

# Inspect registered workers
curl http://localhost:30000/workers
```
Sample response (http workers):
```json
{
  "workers": [
    {"id":"2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f","url":"http://0.0.0.0:31378","model_id":"mistral","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"},
    {"id":"9b0f6c2a-1c4f-4c2a-9f4a-1f2a6c0b9d3e","url":"http://0.0.0.0:34881","model_id":"llama3","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"}
  ],
  "total": 2,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 2
  }
}
```
Add more workers with the same API; include optional `labels` (for per-model policies) or `tokenizer_path` / `reasoning_parser` / `tool_parser` fields as needed. `/workers/{worker_id}` exposes queued job status while background jobs finalize registration.

### gRPC Routing
- **Rust binary**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls grpc://worker-grpc-0:31001 grpc://worker-grpc-1:31002 \
    --tokenizer-path /path/to/tokenizer.json \
    --reasoning-parser deepseek-r1 \
    --tool-call-parser json
  ```
- **Python router**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls grpc://127.0.0.1:20000 \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8080
  ```
The gRPC router tokenizes inputs locally, supports tool-call parsing, and streams responses. It supports both regular HTTP-equivalent serving and PD (prefill/decode) serving when the worker registry contains PD workers. Provide `--model-path` or `--tokenizer-path` (HuggingFace ID or local directory) whenever connection mode resolves to gRPC.
Use `--reasoning-parser` to select built-in reasoning pipelines (DeepSeek-R1, Qwen3, Step-3, GLM4, GLM4.7, etc.) and `--tool-call-parser` for JSON/Pythonic/XML tool contracts in streaming or non-streaming modes.

### OpenAI Backend Mode
Route requests to OpenAI or OpenAI-compatible endpoints:

```bash
# Route to OpenAI API
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \

# Route to custom OpenAI-compatible endpoint (Gemini, xAI, etc.)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls http://my-openai-compatible-service:8000 \
```

**Notes**
- OpenAI backend mode acts as a proxy to a single remote endpoint; load balancing is not applied.
- Provide exactly one `--worker-urls` entry per router instance.
- The Rust binary supports the same flags (`./target/release/sgl-model-gateway --backend openai ...`).

```bash
python3 -m sglang_router.launch_server --host 0.0.0.0
```
Add flags as needed for production deployments:
```bash
python3 -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode
```
Omit `--grpc-mode` to start HTTP workers; the router automatically configures worker URLs and schedules them based on the provided DP size.

### Mini Load Balancer (Debug)
```bash
python3 -m sglang_router.launch_router \
  --mini-lb \
  --pd-disaggregation \
  --prefill http://localhost:30001 \
  --decode http://localhost:30011
```
MiniLB forwards PD requests using simple random routing and is intended for local debugging only.

### Standalone Mock Worker (Debug)
The repository includes a model-free HTTP worker for testing gateway registration, health checks,
request forwarding, and Regular/PD routing. It returns deterministic mock responses and does not
require a GPU or a model download.

Start one Regular worker:
```bash
make mock-worker-regular
```

Start a paired PD worker (one process, two HTTP endpoints):
```bash
make mock-worker-pd
```

For two independent processes, use `--role` with `--mode pd`:
```bash
make mock-worker-pd-prefill
make mock-worker-pd-decode
```

The defaults can be overridden, for example:
```bash
make mock-worker-pd MOCK_WORKER_MODEL_ID=my-model MOCK_WORKER_PREFILL_PORT=18102 \
  MOCK_WORKER_DECODE_PORT=18103 MOCK_WORKER_BOOTSTRAP_PORT=19102
```

Useful fault-injection options are `--health unhealthy`, `--health flapping`, `--fail-rate 0.5`,
`--delay-ms 200`, and `--response-text "mock response"`. The worker exposes `/health`,
`/server_info`, `/model_info`, `/v1/models`, `/generate`, `/v1/chat/completions`,
`/v1/completions`, and `/flush_cache`.

### Running Worker Servers
Use upstream SGLang binaries to start dedicated worker processes.
- **Prefill worker server (gRPC mode)**:
  ```bash
  python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 20000 \
    --tp-size 1 \
    --grpc-mode
  ```
  Remove `--grpc-mode` for HTTP workers. Combine with the router commands above to register the worker via CLI flags or the control-plane API.

## Control Plane

### Worker Lifecycle & Job Queue
- `JobQueue` handles asynchronous add/remove operations to avoid blocking clients.
- `WorkerManager` inspects worker metadata (`/server_info`, `/get_model_info`), tracks load, and exposes `flush_cache` and `get_loads`.
- Per-worker circuit breakers and health probes keep the registry healthy; load monitor feeds metrics to cache-aware and power-of-two policies.

### Administrative & Worker APIs
| Method   | Path             | Description                                                                                                                                               |
|----------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `POST`   | `/workers`       | Queue worker registration (prefill/decode/regular). Body matches `WorkerConfigRequest`. Returns `202 Accepted` while the job queue processes the request. |
| `GET`    | `/workers`       | List workers with health, load, policy metadata, and queued job status.                                                                                   |
| `GET`    | `/workers/{worker_id}` | Inspect a specific worker or job queue entry (UUID).                                                                                                   |
| `PUT`    | `/workers/{worker_id}` | Queue worker update by UUID.                                                                                                                           |
| `DELETE` | `/workers/{worker_id}` | Queue worker removal by UUID.                                                                                                                          |
| `POST`   | `/flush_cache`   | Trigger cache flush across HTTP workers with success/failure breakdown.                                                                                   |
| `GET`    | `/get_loads`     | Sample current load reported by each worker.                                                                                                              |

All administrative routes inherit router API-key protection when `--api-key` is supplied. Job status includes `pending`, `processing`, and `failed` phases with timestamps.

### Service Discovery
Enable Kubernetes discovery to reconcile workers automatically:
```bash
./target/release/sgl-model-gateway \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace sglang-system \
  --service-discovery-port 8000
```
PD mode accepts dedicated selectors:
```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```
Prefill pods can expose bootstrap ports via the `sglang.ai/bootstrap-port` annotation. RBAC must allow `get`, `list`, and `watch` on pods.

Discovery tracks Pod namespace/name and UID independently of readiness or IP.
Deletion events, termination timestamps, selector removal, and completed relists
unregister discovery-owned workers. Add/remove jobs for a Pod finish in order;
failed operations retain state for retry. Existing static/API workers are not
adopted by discovery. Registry removal stops subsequent health-check rounds and
the automatic pre-KV subscription is removed on its next reconciliation tick.
An already-running probe may still finish after removal. Explicit KV event-source
mappings retain their configured lifecycle.

## Data Plane

### Router Capabilities (HTTP & gRPC)
Both router stacks:
- Share load-balancing policies (random, round-robin, cache-aware, power-of-two) with DP-aware scheduling, retries, circuit breakers, and rate limiting.
- Record metrics per request, track running load, and integrate with the router-wide policy registry.

The HTTP router exposes the full OpenAI-compatible surface area (`/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/rerank`, etc.). The gRPC router delivers blazing-fast `/generate` and `/v1/chat/completions` today, with the remaining endpoints returning `501 Not Implemented` until their pipelines are finalised.

#### HTTP Router specifics
- **Regular router** handles classic single-stage workers with per-model policy overrides.
- **Prefill/Decode router** coordinates disaggregated prefill and decode workers, merges metadata, and manages streaming fan-in.

#### gRPC Router specifics
- Industry-first fully Rust implementation of an OpenAI-compatible gRPC inference gateway, including tokenizer, reasoning parser, and tool parser execution in-process for maximum throughput.
- Supports both single-stage and PD (prefill/decode) worker topologies; the router automatically selects the appropriate pipeline per model.
- Provides the same `/v1/*` APIs as the HTTP router while streaming tokenized requests/responses directly to SRT gRPC workers.
- Built-in reasoning parsers for DeepSeek, Qwen, Llama, Mistral, GPT-OSS, Step-3, GLM4, GLM4.7, Kimi K2, and other structured-thought models.
- Tool-call parsers for JSON, Pythonic, XML, and custom schemas with streaming and non-streaming execution loops.
- Tokenizer factory supporting HuggingFace models, local tokenizer.json files, and chat template overrides (see `src/tokenizer`).
- Explore the code paths in `src/reasoning_parser`, `src/tool_parser`, and `src/tokenizer` for the end-to-end Rust implementations that power gRPC mode.

### OpenAI Router
- Proxies OpenAI-compatible chat completions and responses APIs, preserving headers and SSE streams end-to-end.
- Supports `/v1/responses` background jobs with cancellation, deletion, and listing input items—enabling agentic, multi-turn orchestration without persisting data at remote vendor endpoints.

### Request Endpoints
| Endpoint                                                                         | Notes                                                      |
|----------------------------------------------------------------------------------|------------------------------------------------------------|
| `POST /generate`                                                                 | SGLang generate API.                                       |
| `POST /v1/chat/completions`                                                      | OpenAI-compatible chat. Supports streaming and tool calls. |
| `POST /v1/completions`                                                           | OpenAI-compatible text completions.                        |
| `POST /v1/responses`                                                             | Create background responses, returns response IDs.         |
| `GET /v1/responses/{id}`                                                         | Retrieve stored responses.                                 |
| `POST /v1/embeddings`                                                            | Forward embedding requests (HTTP and gRPC).                |
| `POST /v1/rerank`, `POST /rerank`                                                | Ranking APIs.                                              |
| `POST /v1/classify`                                                              | Text classification endpoint.                              |

### Classification API

The `/v1/classify` endpoint provides text classification using sequence classification models (e.g., `Qwen2ForSequenceClassification`, `BertForSequenceClassification`).

**Request:**
```bash
curl http://localhost:30000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "I love this product!"
  }'
```

**Response:**
```json
{
  "id": "classify-a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "object": "list",
  "created": 1767034308,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "probs": [0.12, 0.88],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 0,
    "total_tokens": 6
  }
}
```

**Fields:**
- `label`: Predicted class label (from model's `id2label` config, or `LABEL_N` fallback)
- `probs`: Probability distribution over all classes (softmax of logits)
- `num_classes`: Number of classification classes

**Notes:**
- Classification reuses the embedding backend—the scheduler returns logits which are converted to probabilities via softmax
- Labels come from the model's HuggingFace config (`id2label` field); models without this mapping use generic labels (`LABEL_0`, `LABEL_1`, etc.)
- Both HTTP and gRPC routers support classification

Public health endpoints (`/liveness`, `/readiness`, `/health`, `/health_generate`) reflect registry state; readiness ensures PD workers are paired and IGW has at least one healthy route.

### Tokenization Endpoints

The gateway provides HTTP endpoints for text tokenization, designed to mirror the SGLang Python tokenization API with support for batch operations.

| Endpoint                      | Method   | Description                                           |
|-------------------------------|----------|-------------------------------------------------------|
| `POST /v1/tokenize`           | `POST`   | Tokenize text to token IDs (single or batch).         |
| `POST /v1/detokenize`         | `POST`   | Convert token IDs back to text (single or batch).     |
| `POST /v1/tokenizers`         | `POST`   | Register a new tokenizer (async, returns job status). |
| `GET /v1/tokenizers`          | `GET`    | List all registered tokenizers.                       |
| `GET /v1/tokenizers/{id}`     | `GET`    | Get tokenizer info by UUID.                           |
| `GET /v1/tokenizers/{id}/status` | `GET` | Check async tokenizer loading status.                 |
| `DELETE /v1/tokenizers/{id}`  | `DELETE` | Remove a tokenizer from the registry.                 |

**Tokenize Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

**Batch Tokenize Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

**Tokenize Response:**
```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

**Detokenize Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

**Add Tokenizer (async registration):**
```bash
# Register from HuggingFace
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'

# Check status
curl http://localhost:30000/v1/tokenizers/{tokenizer_id}/status
```

### Parser Endpoints

The gateway provides admin endpoints for parsing reasoning content and function calls from LLM outputs.

| Endpoint                 | Method | Description                                            |
|--------------------------|--------|--------------------------------------------------------|
| `POST /parse/reasoning`  | `POST` | Separate reasoning (`<think>`) from normal text.       |
| `POST /parse/function_call` | `POST` | Parse function/tool calls from text.                |

**Separate Reasoning Request:**
```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

**Response:**
```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

**Supported Reasoning Parsers:**
- `deepseek-r1` - DeepSeek-R1 (initial reasoning mode)
- `qwen3` - Qwen-3 models
- `qwen3-thinking` / `qwen-thinking` - Qwen thinking variant
- `kimi` - Kimi K2 with Unicode tokens
- `glm45` / `glm47` - GLM-4.5/4.6/4.7 models
- `step3` - Step-3 models
- `minimax` - MiniMax models

**Function Call Parsing:**
```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

Supported tool parsers: `json`, `python`, `xml`.
- **Retries**: Default max retries = 5 with exponential backoff (`--retry-max-retries`, `--retry-initial-backoff-ms`, `--retry-max-backoff-ms`, `--retry-backoff-multiplier`, `--retry-jitter-factor`). Retries trigger on 408/429/500/502/503/504.
- **Circuit Breakers**: Per worker thresholds (`--cb-failure-threshold`, `--cb-success-threshold`, `--cb-timeout-duration-secs`, `--cb-window-duration-secs`). Disable via `--disable-circuit-breaker`.
- **Rate Limiting**: Token bucket driven by `--max-concurrent-requests`. Set `--rate-limit-tokens-per-second` to override refill rate. Configure request queue via `--queue-size` and `--queue-timeout-secs`; queued requests observe FIFO order and respect cancellation.
- **Health Checks**: Runtime probes via `--health-check-interval-secs`, `--health-check-timeout-secs`, failure/success thresholds, and `--health-check-endpoint`. Use `--disable-health-check` to skip health checks entirely.
- **Cache Management**: `/flush_cache` ensures LRU eviction when redeploying PD workers.

## Load Balancing Policies

- `random`: uniform random worker selection.
- `round_robin`: sequential rotation with atomic counters.
- `cache_aware`: maintains a prefix tree of prompts to route repeat traffic and evens load with configurable thresholds (`--cache-threshold`, `--balance-abs-threshold`, `--balance-rel-threshold`, `--eviction-interval`, `--max-tree-size`).
- `pre_kv_cache`: uses vLLM KV events, exact token-block prefixes, and worker load. Configure a single discovered profile with `--pre-kv-cache-profile`, `--pre-kv-cache-model`, `--pre-kv-cache-block-size`, and optional `--pre-kv-cache-token-source`; advanced configurations can use `--pre-kv-cache-config`, `--pre-kv-cache-config-json`, or `PRE_KV_CACHE_CONFIG_JSON`.
- `power_of_two`: chooses the lighter worker among two random candidates; integrates with `LoadMonitor`.
  Per-model overrides are available in PD mode (`--prefill-policy`, `--decode-policy`) and IGW mode via the worker registry.

### Local tokenizer compatibility

The pinned `llm-tokenizer 1.9.0` recognizes `DeepseekV41ForCausalLM` and
`model_type: deepseek_v41`, including checkpoints without a Jinja chat template.
The Gateway adapts V4.1 options to vLLM 0.31 semantics: effort budgets
`low/high/xhigh/max = 50/75/75/100`, default `high`, top-level effort precedence,
OR of the thinking switches, `none` forcing thinking off, and history-message
`reasoning` overriding `reasoning_content` (including null). A null thinking
switch retains the default observed in `v0.31.0-cu129-fixed`. Detection uses the
loaded tokenizer's local `config.json`, never the arbitrary profile name, and
caches the result per loaded instance. Local tokenization still runs once and
serves both length routing and cache routing, without calling backend `/render`.
Unsupported options or unavailable native-model metadata produce WARN and a
safe routing fallback. The regression suite checks prompt construction; full
token-ID parity with the actual checkpoint must still be verified on deployment.

### V4.1 cache layout without a worker extension

For the measured DeepSeek-V4.1-Flash deployment, keep the existing profile,
served model, tokenizer path and event-source options. No new option is needed:

```bash
--pre-kv-cache-token-source local_tokenizer \
--pre-kv-cache-block-size 64
```

The background resolver uses the served model's registered tokenizer and its
local `config.json` to identify `DeepseekV41ForCausalLM` / `deepseek_v41`. With
`local_tokenizer`, block size `64`, and no explicit layout or manual groups, it
automatically selects `deepseek_v41_vllm031_dspark5` and logs the selection at
INFO. A tokenizer registered after worker startup is picked up on the next
background check (within five seconds). Profile names and served aliases are
not used as model-type heuristics; another model's sole registered tokenizer
cannot trigger this selection.

This mode needs no worker wheel, middleware, mount or metadata HTTP polling.
Recognition reuses cached model metadata, performs no encoding, and stays off
the inference request path. Request tokens still come from the local tokenizer;
the Router continues subscribing to live KV events and using replay to rebuild
residency. A layout describes how to interpret events, not which blocks are
currently resident.

This preset records the scheduler layout collected on 2026-10-12 from
`vllm-openai:v0.31.0-cu129-fixed`, with TP=8, DP/DCP/PCP=1, DSpark with five
speculative tokens, prefix retention interval 0, partial hash hits disabled,
and Mooncake disabled. Hash, scheduler and alignment blocks are all 64 tokens.
It contains all ten groups: groups 0–7 are non-prefix-cacheable SWA MLA with
32-token blocks and a 128-token window; group 8 is prefix-cacheable full MLA
with 64-token blocks; group 9 is a non-prefix-cacheable circular buffer with
8-token blocks. The hybrid coordinator looks up only `[[8]]`, its effective
`use_eagle` is false, and its prefix replay tail is 128 tokens. The tail is
backend SWA reconstruction, separate from the event publisher's replay port.

Automatic selection applies this known deployment contract; it does not discover
arbitrary V4.1 scheduler layouts. Before reporting hits it also requires a valid
local group-8 Store event with the expected block size and attention type. Empty
batches and auxiliary-group events alone leave the cache ratio unknown. Events
cannot verify all scheduler settings; a changed checkpoint, worker image, speculative-decoding
configuration, parallelism or cache configuration requires collecting and
checking the layout again. Unsupported event groups and incompatible cacheable
group metadata produce WARN and a safe fallback. Do not assume that every
change to a skipped auxiliary group is detectable through events. Repeated
background recognition preserves the index and any sticky compatibility error;
tokenizer removal/replacement or worker replacement clears automatic residency.

`--pre-kv-cache-layout deepseek_v41_vllm031_dspark5` remains an optional explicit
override (JSON: `"layout": "deepseek_v41_vllm031_dspark5"`), not a requirement.
It requires `local_tokenizer` and block size `64`, and cannot be combined with
manual `cache_groups` / `--pre-kv-cache-groups`. Explicit layouts and manual
groups are never overwritten by automatic model recognition.

Inspect the Router through its business address:

```bash
curl -fsS "$ROUTER_URL/v1/pre-kv-cache/status" | python3 -m json.tool
```

Automatically adapted workers report `layout_source: "automatic_builtin"`, the
selected `layout_preset`, and `layout_events_confirmed` for the observed group-8
signature. Explicit presets report `layout_source: "builtin_preset"`. Both
report `layout_engine_verified: false` and `layout_age_ms: null`.
Their layout's `engine_id` is empty because the Router did not obtain a live
engine identity. Event freshness, resident blocks and active warnings are
still reported and checked; a preset never makes a stale index valid. Verify
token-ID parity and predicted cache hits against the actual backend before
interpreting performance results.

### Optional automatic runtime cache groups

Deployments without a recognized built-in adapter retain automatic metadata
discovery. The following optional worker bridge exposes actual runtime layouts;
the recognized V4.1 configuration above does not require or poll it. Do not configure group IDs
by hand or infer a complete layout from observed events. The vLLM 0.31 engine
has an internal `get_kv_cache_group_metadata` method, but its standard HTTP API
does not expose the complete lookup contract (including DSpark's effective
last-block-drop rules).

Install the small metadata bridge on each vLLM worker, using its existing Python
environment, and append one option to its existing `vllm serve` command:

```bash
pip install --no-deps /path/to/smg_vllm_kv_metadata-0.1.0-py3-none-any.whl
# Append to the existing vllm serve command:
--middleware smg_vllm_kv_metadata.KVMetadataMiddleware
```

The wheel is built from `integrations/vllm_kv_metadata`. If `VLLM_PLUGINS` is set,
include `smg_kv_metadata` in that allowlist. The bridge reads actual scheduler
metadata through vLLM's internal engine IPC; it does not enable the development
RPC API. Its read-only `/v1/pre-kv-cache/metadata` endpoint preserves existing
API-key authentication. Normal inference requests pass through directly.

In this dynamic mode the Router fetches metadata in the background every five seconds; no
metadata HTTP request or backend `/render` runs in the inference request path.
`--pre-kv-cache-groups` is unnecessary. Each engine's complete layout includes
cacheable groups, physical/hash block sizes, lookup order, sliding windows and
effective speculative-decoding flags. Layout or engine changes clear residency
and reconnect the event stream; missing or incompatible metadata produces WARN
and `cache_hit_ratio=unknown`, never an invented layout. The profile block size
must match the backend hash block size.

Matching follows vLLM's common-prefix search across full-attention/MLA, RSWA and
sliding-window groups, including DSpark/EAGLE block drops and mixed physical
block sizes. Only local GPU residency counts; Mooncake remote/CPU/storage hits
are not included. Unsupported managers, partial-block lookup and DCP/PCP/DP > 1
produce an explicit compatibility warning. TP > 1 is supported. Sparse events
resolve physical hashes through verified content ancestry; missing history
never becomes a positive hit. The unit tests cover the upstream search rules;
the actual V4.1 checkpoint's token IDs and custom worker build must still be
verified in deployment.

The same status endpoint reports `layout_source: "vllm_engine"`, the complete `runtime_layout`,
`layout_error`, event freshness, resident blocks and active warnings. In contrast,
`observed_groups` only describes sampled events; `observed_groups_are_complete`
remains false. A status query alone cannot expose engine metadata on an old
worker that has no bridge. Older single-group models retain their existing
configuration without requiring this extension.

A subscription log only confirms socket setup. `pre_kv_cache event stream active`
requires accepted events. If none have been received since binding the worker,
`no_kv_events` emits WARN after `event_stale_ms` (default 30 seconds). The legacy
configuration name remains accepted, but this interval is only an initial
diagnostic threshold: it never expires confirmed cache residency. An idle worker
may have no cache changes to publish. Use a new prompt prefix when checking a
fresh worker's event publisher.

Confirmed cache residency stays usable during silence, matching llm-d's event
index model. Only cache events or actual stream/worker failures change it.
Disconnects, worker/layout replacement and sequence gaps invalidate serving
state. For a forward gap, a verified checkpoint is replayed from its next missing
sequence, including removals through the live boundary. Missing parents also
trigger a background rebuild, even when event sequence numbers are continuous.

Subscriptions start with live events; startup and automatic layout detection
do not trigger replay. Valid new root chains work immediately. Existing cached
prefixes that never emit another event remain unknown until observed or recovered.
Replay starts for a sequence gap, missing ancestry, or the request-driven silent
stream check described below. Gap/ancestry recovery performs one recovery per
worker at a time, with at most eight across the Router, a two-minute
budget, a two-second idle limit per attempt and three consecutive no-progress
attempts. Interior gaps and interrupted responses resume from the missing sequence
using a fresh DEALER identity; the vLLM protocol accepts a start sequence only,
not a range or batch limit. Receive buffering is bounded to 10,000 messages and
never substitutes for sequence validation. Transient failures retain private
progress and retry on a timer after 30/60/120/240/300 seconds, capped at 300.
Replay runs outside the request path and index lock; rebuilt state is committed
only after validating continuity, worker/layout identity and the connection.
The vLLM replay end marker (empty topic, signed sequence -1, empty payload) is
handled as a control frame before checking event topics. An end marker never
makes a gapped history complete; event frames still require the configured topic.
A failed rebuild leaves only verified live branches available for routing.
If the first replay event starts after the requested sequence, the old checkpoint
is discarded and `history_truncated_at` records the start of a partial suffix.
This is evidence of unavailable history, not proof of eviction rather than packet
loss. Roots in that suffix can be indexed; missing ancestors remain misses and
`history_complete` remains false until an authoritative clear or complete rebuild.
Successful partial recovery does not repeatedly request the missing beginning.
A bounded replay buffer is not a cache snapshot. Do not clear the worker cache
merely to silence a warning during a benchmark.

A selected request with exact tokens and zero/unknown cache match also records
a background progress hint. If no live frame arrives for two seconds, the
subscriber renews its silent SUB connection without discarding the index, waits
two seconds for subscription propagation, then checks replay from the next
missing sequence (zero if no events have been indexed). This requires no later
live frame to reveal a gap. The renewed SUB stays open during and after replay.
Progress checks run separately from live receipt, with an eight-second total
budget, two-second receive-idle limit, 10,000-frame cap, and aggregate payload
limit of `max_response_bytes`. They share the eight global replay slots. A reply
is applied only if its sequence is continuous and the worker/layout, connection,
live receipt counter and indexed sequence have not changed while it was read.
An empty reply or a replay transport failure preserves existing cache evidence;
live events can still update the index while replay is unavailable. Invalid
event contracts and confirmed truncated histories still follow the conservative
validation rules above. No cache residency is inferred from a successful request.

Progress checks are at least 30 seconds apart. Failures back off through
30/60/120/240/300 seconds; checks stop five minutes after the last selected miss,
or when valid live progress resolves the hint. Normal idle workers and known
positive matches do not start checks. Without a replay endpoint, only subscription
renewal is attempted. An unresolved progress failure logs one deduplicated
`event_progress_unavailable` warning; it does not disable the live subscriber.
Status exposes `progress_check_in_progress`, `progress_checks`,
`progress_recovered_frames`, `last_progress_check_age_ms`, and
`last_progress_error`. Live frame counters remain live-only, so a successful
history repair can advance `last_sequence` without increasing
`received_event_frames`. If the publisher supplies neither usable live events
nor retained replay history, the Router still cannot reconstruct missing keys.

INFO logs `pre_kv_cache replay started` and `pre_kv_cache replay completed` report
the sequence range and rebuilt block count. The status field `history_complete`
distinguishes verified complete history from a usable subset. `event_state_usable`
reports whether the bound stream/layout can supply cache evidence; request-specific
token/key validation still applies. `event_state_fresh` and
`last_valid_event_age_ms` remain observational: an idle index can be usable even
when `event_state_fresh` is false. Replay failures and `missing_parent` remain
visible until recovery is verified. These rules require no new startup options
and preserve local tokenization and the KV/load scoring formula.

Unknown Store identities (for example unsupported extra-key structures or LoRA)
are isolated at the first affected block, including descendants across groups.
They do not disable the worker or prevent unrelated Stores, Removes and Clears
from being applied. Dense Stores retain only a verified prefix; sparse spans with
uncertain alignment are skipped. Extra keys are never stripped to manufacture a
plain-text key. Conflicting hash mappings, unknown deletion formats and incompatible
layouts still invalidate unsafe residency. This follows llm-d's per-Store failure
isolation while retaining stricter identity validation for exact text matching.

The status endpoint adds `skipped_stores`, `skipped_blocks`, `last_skipped_store`
and `isolated_hashes`. The last sample contains bounded type/length information,
not prompt tokens, salt values or binary contents. Counters describe the current
index generation; recovery or worker replacement can reset them. The exact
isolation dictionary is bounded to at most 65,536 hashes (also limited by the
configured index capacity). When full, evicted rejections enter a fixed 1 MiB
Bloom filter and `isolation_overflow` becomes true. False positives can only
reduce coverage; old rejected identities are never silently accepted. A genuine
index reset clears both structures. `history_complete` describes event sequence
coverage, not support for every request identity. `unsupported_cache_identity`
warns once while partial coverage persists; it is not a worker-wide failure.

For receive/index stalls, compare `last_received_sequence` and
`last_received_event_age_ms` with `last_sequence` and `last_valid_event_age_ms`.
`received_event_frames`, `duplicate_event_frames`, `event_apply_failures`,
`subscription_running`, `connection_generation`, `last_connection_event` and
`replay_in_progress` distinguish receipt, filtering, indexing and recovery.
Receipt counters are cumulative within the subscription; received frames include
malformed frames. `connecting` means socket setup, not a confirmed connection.
Live and replay sequence watermarks are separate: a live regression invalidates
the old incarnation before replay-overlap filtering. Failed subscriber tasks are
supervised and restarted after invalidating old residency.

Index capacity is reclaimed entry by entry. `max_blocks_per_worker` bounds
resident `(group, hash)` entries and a separate cache of nonresident ancestors
(each defaults to 500,000 entries). Resident identities are protected from
historical churn; unused ancestors use LRU eviction. If residency itself exceeds
its budget, the least recently stored local entry is dropped, causing a
conservative miss rather than a stream reset. `known_blocks` is bounded by twice
the configured limit; `ancestry_evictions` and `residency_evictions` expose local
capacity reclamation. These are entry limits, not byte limits.

`event_apply_failures` counts live batches with an actual processing failure.
A previously skipped orphan does not fail later valid batches. `last_error`
describes the latest apply result; `history_missing_ancestry` and
`history_complete` retain the separate coverage information. Incompatible
protocol/layout state lasts for the current publisher epoch: reconnect, sequence
restart or an authoritative clear rebuilds from empty state and revalidates
events. A merely well-formed Store on the same incompatible stream cannot revive
old residency. Missing optional group fields use current-epoch observations or
an explicit runtime/manual contract; explicit contradictions still fail closed.

`pre_kv_cache_selected_cached_tokens` records measured matches only, including
real zeroes. Unknown selections are counted separately by
`pre_kv_cache_selected_cache_state_total{state="unknown"}` (and `state="known"`
for measured selections). These selection metrics cover the cost-scoring path;
single-candidate scheduling continues to bypass scoring. Engine running/waiting
counts are summed, but KV occupancy ratios use the maximum engine value. When
load metrics are unavailable, the fallback uses the largest last-observed peer
cost plus local in-flight load; it is an estimate, not an upper bound on direct
backend traffic. Fresh load measurements and the KV/load formula are unchanged.

TCP keepalive uses idle=60 seconds, interval=15 seconds and count=3 on SUB/DEALER
sockets. It probes connections without expiring quiet cache entries or starting
replay during normal silence. It cannot detect a stuck application publisher on
an otherwise healthy connection. No new startup flags or Worker installation are
required. Requests sent directly to a shared Worker still affect its events,
cache residency and backend load. A Router inference test competes for that same
capacity; status-only inspection does not submit an inference request.

### Pre-KV-cache importance and failure visibility

For a single discovered profile, set these options alongside the existing
`--pre-kv-cache-profile`, model, block size and event-source options:

```bash
--pre-kv-cache-token-source local_tokenizer \
--pre-kv-cache-kv-importance 1 \
--pre-kv-cache-load-importance 1 \
--pre-kv-cache-log-decisions
```

Both importance values default to `1`, preserving the existing scoring formula.
They accept numbers from `0` to `1000`, including fractions, and cannot both be
zero. The upper bound prevents extreme multipliers from overflowing ordinary
cost estimates. Higher `kv_importance` gives more importance to avoiding
uncached work; higher `load_importance` gives more importance to avoiding busy
workers. They are relative multipliers and need not sum to one or 100. Scaling
both equally preserves ordering within a profile. The lowest estimated cost
wins inside the existing healthy, eligible worker pool:

```text
score = kv_importance * uncached_prefill_ms
      + load_importance * (backend_or_fallback_load_ms + reservation_ms)
```

Adjust both values atomically at runtime through the same authenticated admin
routes as worker management. The URL uses the profile name, not the served model:

```bash
curl -sS "$ROUTER_URL/v1/pre-kv-cache/weights/deepseek_v41"
curl -sS -X PUT "$ROUTER_URL/v1/pre-kv-cache/weights/deepseek_v41" \
  -H 'Content-Type: application/json' \
  -d '{"kv_importance":2,"load_importance":1}'
```

Include the usual Authorization header when authentication is enabled. Each
selection uses one consistent snapshot for every candidate. Updates take effect
on subsequent selections without re-tokenization or resetting subscriptions,
indices or load reservations. They are process-local and in memory: configure
each Gateway replica separately, and set startup CLI/JSON values to persist
across restarts. Invalid values leave the existing pair unchanged (HTTP 400);
unknown profiles return 404 and an unconfigured runtime returns 503.

The KV term estimates the work remaining after the cached prefix. The load term
includes backend waiting/running requests, KV-cache occupancy pressure, and
Router reservations for dispatches not yet reflected in a load poll. When
backend metrics are missing or stale, the fallback uses Router in-flight
requests plus the largest last-observed peer load estimate (zero if none).
`kv_importance=0` ignores the current request's cache savings in scoring;
`load_importance=0` ignores both observed load and reservations. Health checks,
pool eligibility and tokenization behavior are unchanged.

`--pre-kv-cache-log-decisions` is optional and off by default. It emits one INFO
`pre_kv_cache candidate score` event per candidate, correlated through the
request span. Fields include `selected`, `cache_state_valid`, `kv_importance`,
`load_importance`, `load_source`, `waiting_requests`, `running_requests`,
`kv_usage_ratio`, `load_sample_age_ms`, `local_in_flight`, `pending_tokens`,
`uncached_prefill_ms`, `backend_or_fallback_load_ms`, `reservation_ms`,
`weighted_kv_ms`, `weighted_load_ms`, and `estimated_cost_ms`.
`load_source=local_fallback` means backend load gauges are unavailable; backend
gauge fields are then absent. If only the occupancy gauge is missing or invalid,
the queue/running gauges still apply, the pressure term retains its existing
zero default, and `kv_usage_ratio` is omitted rather than reporting a measured
zero. KV occupancy is not GPU compute utilization.
One eligible worker logs `pre_kv_cache scoring bypassed reason=single_candidate`
and keeps the fast path without a reservation or additional tokenization.

Compare candidate differences to understand a load-induced routing change:
if worker A saves 200 estimated ms of prefill but adds 300 weighted ms of load,
worker B can win despite less cache. `weighted_load_ms / estimated_cost_ms`
describes one candidate's score composition, not a load limit or traffic share.
These costs use initial coefficients, not measured TTFT; start with `1/1`, then
hold KV importance at `1` and compare load importance `0.5`, `1`, and `2` on identical
workloads. Validate TTFT, TPOT, throughput, waiting requests and actual backend
cache hits. Turn off detailed decision logs for final performance comparisons
because log volume grows with requests times candidates.

For JSON/file configuration, set `profiles.<name>.cost.kv_importance` and
`profiles.<name>.cost.load_importance`, and the top-level `log_decisions` boolean.
Structured CLI options cannot be mixed with JSON/file configuration.
The previous CLI names `--pre-kv-cache-kv-weight` and
`--pre-kv-cache-load-weight`, and JSON fields `kv_weight` and `load_weight`,
remain accepted aliases with the same range validation. Use only one spelling
per setting. Startup and decision logs use the new importance names.

Runtime degradation emits WARN at the ordinary `info` log level, with `scope`,
`reason` and `detail`; no debug switch or decision-log switch is required.
The same unresolved scope/reason warns once and remains visible in the status
API; detected event/load recovery emits INFO and allows a later outage to warn
again. Invalid startup configuration
still fails validation. Transient problems preserve request forwarding with
the existing conservative estimates:

| Warning reason | Meaning / next check |
| --- | --- |
| `profile_model_mismatch`, `subscription_unavailable` | Match the served model and profile; check discovery or explicit worker mappings. |
| `no_kv_events` | No frame within `event_stale_ms` (default 30 s) after worker binding. Check worker `--kv-events-config`, event endpoint, topic and network; an idle worker may also produce no events. Confirmed residency does not expire during silence. |
| `kv_events_unusable` | Frames arrived but none established usable state; compare receive/application progress and the specific error. |
| `unsupported_cache_identity` | A Store identity and its descendants are isolated; other valid branches remain eligible. Inspect the structure-only `last_skipped_store` sample. |
| `subscriber_task_stopped` | The subscriber exited unexpectedly; old residency was invalidated and the supervisor will restart it. |
| `event_state_unavailable` | No usable event state for this worker: inspect initialization, stream errors and worker/layout identity. Idle time alone never disables matching. |
| `event_connect`, `event_receive`, `disconnected`, `event_frame` | ZMQ connection, receive or protocol failure; inspect the endpoint in the detail. |
| `unsupported_group`, `unsupported_attention`, `block_size_mismatch` | The event format/cache kind or block size is incompatible. Verify the backend; weight changes cannot fix this. |
| `replay_*` | Recovery failed; transient failures retry on a bounded timer. A first-frame history truncation is recorded at INFO and in `history_truncated_at`, not as a repeating replay failure. |
| `missing_parent`, `sparse_hash_unknown` | Some ancestry is missing. Verified branches remain usable; recovery runs once for known truncated history. Compare `history_complete`, `history_truncated_at` and replay logs before benchmarking weights. |
| `local_tokenizer_unavailable`, `local_chat_template`, `local_tokenize` | Exact local request tokens could not be derived. Check tokenizer files, served model name and chat-template support. Local mode does not fall back to HTTP render. |
| `load_metrics_unavailable`, `load_state_unavailable` | Backend load metrics failed, are missing or stale; scoring uses local fallback estimates. |
| `kv_usage_unavailable` | Occupancy gauge missing/invalid; the pressure term contributes zero while queue/running gauges remain usable. |

The startup message `pre_kv_cache subscription started; awaiting valid KV events`
only means a subscriber task was created, not that the publisher is reachable.
Warnings distinguish observed silence from a confirmed configuration error;
the Router cannot infer a missing backend flag from silence alone. These
diagnostics do not add support for previously unsupported cache-event formats
or model chat templates.

The KV index describes local GPU residency. CPU/storage and remote connector
events do not add or remove local GPU blocks; unknown media remain unsupported.
A shared store may affect backend cache counters and transfer costs, so backend
`cached_tokens` is not automatically equivalent to the Router's indexed hits.

## Observability

### Logging
Structured tracing through `tracing` with optional file sink (`--log-dir`) and `--log-level` (`debug`, `info`, `warn`, `error`). Text logs use a compact format with colored but non-italic field names; scheduler logs include model, endpoint, context length, runtime, and the selected load-balancing policy. Use `--json-log` for machine-readable logs.

The INFO scheduling event includes `cache_hit_ratio`, formatted to four decimal places (for example, `0.7500` means 75%). For `pre_kv_cache`, it reports the selected worker's indexed cached tokens divided by the existing `context_tokens`; in PD mode it describes the selected Prefill worker. It is a routing-time estimate, not the backend's final cache usage. Missing tokens, unavailable event state, unmeasured policies, or cached tokens exceeding `context_tokens` produce `unknown`. Local tokenization is reused, including on the single-worker path, without additional tokenization or render calls for logging; retries and pool fallback discard the preceding selection's cache count.

HTTP completion logs include `error_code` for failures and `latency_us` in microseconds; the existing `latency` field is retained with the same unit. HTTP 4xx/5xx responses also emit an `HTTP error response details` event as the response body finishes or is dropped, using the original request span. Search by `request_id` to correlate the completion and details events.

Error details include `error_message`, `error_code`, `error_body_truncated`, and `error_body_complete`. Capture is limited to the first 8 KiB of the error response, with diagnostic fields limited to 1,024 characters. JSON summaries extract message/code/type fields or validation messages, excluding unrelated fields such as validation `input`; plain-text errors use a bounded preview. Unknown or incomplete JSON produces a diagnostic placeholder instead of a raw JSON dump. This observes the body during normal transmission without buffering the entire response or delaying streaming headers. A dropped or failed body can produce incomplete details, and successful HTTP responses (including errors embedded in HTTP 200 SSE streams) are not inspected.

### Prometheus Metrics
Enable with `--prometheus-host`/`--prometheus-port` (defaults to `0.0.0.0:29000`).

**Metric Categories (40+ metrics):**

| Layer | Metric Prefix | Description |
|-------|---------------|-------------|
| HTTP | `smg_http_*` | Request counts, duration, active connections, rate limiting |
| Router | `smg_router_*` | Requests by model/endpoint, latency, errors, upstream responses |
| Inference | `smg_router_ttft/tpot/tokens_*` | Time to first token, time per output token, token counts (gRPC) |
| Worker | `smg_worker_*` | Pool size, active connections, health checks, selection events |
| Circuit Breaker | `smg_worker_cb_*` | State (closed/open/half-open), transitions, outcomes |
| Retry | `smg_worker_retries_*` | Retry attempts, exhausted retries, backoff duration |
| Discovery | `smg_discovery_*` | K8s registrations, sync duration, workers discovered |
| Database | `smg_db_*` | Operations, duration, connections, items stored |

**Key Metrics:**
- `smg_router_ttft_seconds` - Time to first token histogram (gRPC mode)
- `smg_router_tpot_seconds` - Time per output token histogram (gRPC mode)
- `smg_router_tokens_total` - Total input/output tokens by model
- `smg_router_generation_duration_seconds` - End-to-end generation time
- `smg_worker_cb_state` - Circuit breaker state gauge (0=closed, 1=open, 2=half-open)

**Duration Buckets:**
1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 15s, 30s, 45s, 60s, 90s, 120s, 180s, 240s

### OpenTelemetry Tracing
Enable distributed tracing with OTLP export:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

**Features:**
- OTLP/gRPC exporter (default port 4317)
- W3C Trace Context propagation for HTTP and gRPC
- Batch span processing (500ms delay, 64 span batch size)
- Custom filtering to reduce noise (only exports relevant spans)
- Trace context injection into upstream worker requests

**Configuration:**
- `--enable-trace` - Enable OpenTelemetry tracing
- `--otlp-traces-endpoint <host:port>` - OTLP collector endpoint

### Request ID Propagation
Configure headers for request ID extraction:
```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```
Responses include `x-request-id` header for correlation.

### CORS
Set `--cors-allowed-origins` for browser access.

## Security

### Router and Worker API Keys
- **Router API key (`--api-key`)** protects client access to router endpoints; all protected routes expect `Authorization: Bearer <key>`.
- Workers listed in `--worker-urls` inherit the router API key automatically.
- When adding workers dynamically, provide explicit API keys via payload or query string; they do **not** inherit automatically.

```bash
# Router and initial workers share the same key
python3 -m sglang_router.launch_router \
  --api-key "shared-api-key" \
  --worker-urls http://worker1:8000 http://worker2:8000

# Adding a worker without key while router has one triggers a warning and leaves the worker unprotected
curl -X POST http://localhost:8080/add_worker?url=http://worker3:8000

# Add worker with explicit key
curl -X POST "http://localhost:8080/add_worker?url=http://worker3:8000&api_key=worker3-specific-key"
```

### Security Configurations
1. **No Authentication** (default): Router and workers accept requests without keys—use only in trusted environments.
2. **Router-only Authentication**: Provide `--api-key`; clients must present the key, router accesses workers without credentials.
3. **Worker-only Authentication**: Router open to clients; each worker requires its own key. Supply keys when calling `/workers` or `/add_worker`.
4. **Full Authentication**: Set router API key and provide per-worker keys. Example:
   ```bash
   python3 -m sglang_router.launch_router --api-key "router-key"
   curl -H "Authorization: Bearer router-key" \
     -X POST http://localhost:8080/add_worker?url=http://worker:8000&api_key=worker-key
   ```

### Important Notes
- Initial workers declared via CLI inherit the router key; dynamic workers must supply keys explicitly.
- Router logs a warning when a worker is registered without a key while the router expects authentication.
- When router and workers share the same key, still include the key when invoking dynamic registration APIs.

### TLS (HTTPS) for Gateway Server

Enable TLS to serve the gateway over HTTPS:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

| Parameter | Description |
|-----------|-------------|
| `--tls-cert-path` | Path to server certificate (PEM format) |
| `--tls-key-path` | Path to server private key (PEM format) |

Both parameters must be provided together. The gateway uses rustls with the ring crypto provider for TLS termination. If TLS is not configured, the gateway falls back to plain HTTP.

### mTLS for Worker Communication

Enable mutual TLS (mTLS) for secure communication with workers in HTTP mode:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| Parameter | Description |
|-----------|-------------|
| `--client-cert-path` | Path to client certificate for mTLS (PEM format) |
| `--client-key-path` | Path to client private key for mTLS (PEM format) |
| `--ca-cert-path` | Path to CA certificate for verifying worker TLS (PEM format) |

**Key Points:**
- Client certificate and key must be provided together
- Multiple CA certificates can be added with multiple `--ca-cert-path` flags
- Uses rustls backend when TLS is configured
- Single HTTP client is created for all workers (assumes single security domain)
- TCP keepalive (30 seconds) is enabled for long-lived connections

**Full TLS Example (Gateway HTTPS + Worker mTLS):**
```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key"
```

### Control Plane Authentication

The gateway supports role-based access control (RBAC) for control plane APIs (worker management, tokenizer registration, cache operations). Two authentication methods are available:

#### Authentication Methods

| Method | Use Case | Configuration |
|--------|----------|---------------|
| **API Keys** | Service accounts, internal services | `--control-plane-api-keys` |
| **JWT/OIDC** | User authentication via Identity Provider | `--jwt-issuer`, `--jwt-audience` |

Both methods can be used together. Requests are authenticated in order: API key → JWT token.

#### Roles

| Role | Access |
|------|--------|
| `admin` | Full access to all control plane APIs (workers, tokenizers, cache, etc.) |
| `user` | Inference/data plane APIs only (chat completions, embeddings, etc.) |

#### API Key Authentication

Static API keys for service accounts and automation:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --control-plane-api-keys 'svc1:CI Pipeline:admin:secret-key-123' \
                           'svc2:Monitoring:user:readonly-key-456' \
  --control-plane-audit-enabled
```

**Format:** `id:name:role:key`
- `id` - Unique identifier for the key
- `name` - Human-readable description
- `role` - Either `admin` or `user`
- `key` - The secret key (stored as SHA-256 hash internally)

**Usage:**
```bash
curl -H "Authorization: Bearer secret-key-123" \
  http://localhost:30000/workers
```

#### JWT/OIDC Authentication

Authenticate users via an external Identity Provider (Azure AD, Okta, Auth0, Keycloak, etc.):

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
  --jwt-audience "api://my-gateway-client-id" \
  --jwt-jwks-uri "https://login.microsoftonline.com/{tenant-id}/discovery/v2.0/keys" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user' \
  --control-plane-audit-enabled
```

| Parameter | Description |
|-----------|-------------|
| `--jwt-issuer` | OIDC issuer URL. Used to validate the `iss` claim and discover JWKS endpoint via `.well-known/openid-configuration`. |
| `--jwt-audience` | Expected audience (`aud` claim). Typically your application's client ID or API identifier (e.g., `api://client-id`). |
| `--jwt-jwks-uri` | (Optional) Explicit JWKS URI. If omitted, discovered automatically from the issuer's OIDC configuration. |
| `--jwt-role-mapping` | Map IDP group/role names to gateway roles. Format: `idp_role=gateway_role`. |

**How it works:**
1. User authenticates with Identity Provider (OAuth2/OIDC flow)
2. IDP issues a JWT token
3. User sends token to gateway: `Authorization: Bearer <jwt-token>`
4. Gateway validates the JWT:
   - Verifies signature against JWKS
   - Checks `iss` matches `--jwt-issuer`
   - Checks `aud` matches `--jwt-audience`
   - Validates expiration and other standard claims
   - Extracts role from `roles` claim (or `groups` as fallback)
   - Maps IDP role to gateway role via `--jwt-role-mapping`

**Example Azure AD Configuration:**
```bash
# Azure AD issues tokens with:
#   iss: https://login.microsoftonline.com/{tenant}/v2.0
#   aud: api://your-client-id (or the client ID itself)
#   roles: ["Gateway.Admins"] or groups: ["group-id"]

python3 -m sglang_router.launch_router \
  --jwt-issuer "https://login.microsoftonline.com/your-tenant-id/v2.0" \
  --jwt-audience "api://your-client-id" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user'
```

#### Audit Logging

Enable `--control-plane-audit-enabled` to log all control plane operations with:
- Timestamp
- Principal (API key ID or JWT subject)
- Role
- Action performed
- Success/failure status

#### Combined Authentication Example

Use both API keys and JWT for different use cases:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  # API keys for service accounts
  --control-plane-api-keys 'ci:CI/CD Pipeline:admin:ci-secret' \
  # JWT for human users via Azure AD
  --jwt-issuer "https://login.microsoftonline.com/{tenant}/v2.0" \
  --jwt-audience "api://gateway" \
  --jwt-role-mapping 'Platform.Admins=admin' 'Platform.Users=user' \
  # Enable audit logging
  --control-plane-audit-enabled
```

## Development & Testing
```bash
# Build Rust components (debug mode, fast)
cargo build

# Run Rust tests
cargo test

# Fast Python development (rebuilds and installs in debug mode)
cd bindings/python && maturin develop

# Run Python tests
cd ../..  # Back to sgl-model-gateway root
pytest e2e_test/
```
For production builds, use `maturin build --release --out dist` from the `bindings/python/` directory to create optimized wheels. During development, `maturin develop` rebuilds and installs instantly without creating wheel files. Use `python -m sglang_router.launch_server` to co-launch router and SGLang workers in small clusters for local validation.

### Build Caching

**Local development** uses incremental compilation by default (configured in `.cargo/config.toml`), which is optimal for the edit-compile-test cycle.

**For release builds or CI**, you can optionally use [sccache](https://github.com/mozilla/sccache) to cache compilation artifacts:

```bash
# Install sccache
cargo install sccache

# Option 1: Set environment variable (per-session)
export RUSTC_WRAPPER=sccache
cargo build --release

# Option 2: Add to your global cargo config (~/.cargo/config.toml)
# [build]
# rustc-wrapper = "sccache"
```

> **Note:** sccache and incremental compilation are mutually exclusive—sccache cannot cache incrementally compiled crates. The project defaults to incremental compilation for faster local iteration. Use sccache for clean/release builds where caching across builds matters more. CI workflows use sccache with GitHub Actions cache backend for cross-job compilation caching.

---

## Release Management

### Creating Gateway Releases

Create releases for the Gateway/Router component with filtered commits:

```bash
# Using make
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0

# Save to file
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 OUTPUT=RELEASE_NOTES.md

# Create draft release (requires gh CLI, DEFAULT behavior)
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

# Publish release immediately (requires gh CLI)
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
```

**Tag Naming**: Use `gateway-*` or `router-*` prefixes to avoid triggering unrelated CI workflows.

### Release Workflow

1. **Create and push tag**:
   ```bash
   git tag -a gateway-v1.0.0 <commit-hash> -m "Gateway release v1.0.0"
   git push origin gateway-v1.0.0
   ```

2. **Generate release notes** (automatically filters gateway-related commits):
   ```bash
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0
   ```

3. **Create GitHub release**:
   ```bash
   # Create draft (DEFAULT - review before publishing)
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

   # Or publish immediately (skip draft)
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
   ```

### Filtered Paths

Release notes only include commits touching:
- `sgl-model-gateway/` - Router codebase
- `python/sglang/srt/grpc/` - gRPC protocol
- `python/sglang/srt/entrypoints/grpc_server.py` - gRPC server

The script automatically extracts author attribution, PR links, and identifies new contributors.

---

SGLang Model Gateway continues to evolve alongside the core SGLang runtime. Contributions should keep CLI flags, documentation, and Python bindings in sync with the Rust implementation.
