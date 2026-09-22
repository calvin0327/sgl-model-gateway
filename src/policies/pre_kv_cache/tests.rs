//! Acceptance tests exercise real HTTP rendering, native ZMQ and existing routers.

use super::*;
use crate::{
    app_context::AppContext,
    config::{GatewayConfig, PolicyConfig},
    core::{model_card::ModelCard, BasicWorkerBuilder},
    protocols::chat::ChatCompletionRequest,
    routers::{router_manager::RouterManager, RouterTrait},
    tokenizer::{
        chat_template::ChatTemplateParams,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
        TokenizerRegistry,
    },
};
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::atomic::AtomicUsize;

#[derive(Default)]
struct CountingTokenizer {
    encode_calls: AtomicUsize,
    special_tokens: SpecialTokens,
}

impl Encoder for CountingTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        assert!(!add_special_tokens);
        self.encode_calls.fetch_add(1, Ordering::Relaxed);
        let tokens = if input == "rendered prompt" {
            vec![1, 2, 3, 4, 5]
        } else {
            vec![1]
        };
        Ok(Encoding::Plain(tokens))
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl Decoder for CountingTokenizer {
    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        Ok("decoded".into())
    }
}

impl Tokenizer for CountingTokenizer {
    fn vocab_size(&self) -> usize {
        1024
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn apply_chat_template(
        &self,
        _messages: &[Value],
        _params: ChatTemplateParams,
    ) -> anyhow::Result<String> {
        Ok("rendered prompt".into())
    }
}

struct Mock {
    url: String,
    waiting: Arc<AtomicUsize>,
    renders: Arc<AtomicUsize>,
    render_bodies: Arc<Mutex<Vec<Value>>>,
    server: tokio::task::JoinHandle<()>,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn mock(name: &'static str) -> Mock {
    let waiting = Arc::new(AtomicUsize::new(0));
    let renders = Arc::new(AtomicUsize::new(0));
    let render_bodies = Arc::new(Mutex::new(Vec::new()));
    let w = waiting.clone();
    let r = renders.clone();
    let bodies = render_bodies.clone();
    let render = move |Json(body): Json<Value>| {
        r.fetch_add(1, Ordering::Relaxed);
        bodies.lock().push(body);
        async { Json(json!({"token_ids":[1,2,3,4,5]})) }
    };
    let app = Router::new()
        .route("/metrics",get(move || { let w=w.load(Ordering::Relaxed); async move {
            format!("vllm:num_requests_running 0\nvllm:num_requests_waiting {w}\nvllm:kv_cache_usage_perc 0.1\n")
        }}))
        .route("/v1/chat/completions/render",post(render))
        .route("/v1/completions/render",post(|| async { Json(json!([{"token_ids":[1,2,3,4,5]}])) }))
        .route("/v1/chat/completions",post(move |Json(_): Json<Value>| async move {
            Json(json!({"id":name,"object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":name},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}))
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Mock {
        url,
        waiting,
        renders,
        render_bodies,
        server,
    }
}

fn publisher() -> (zmq::Context, zmq::Socket, String) {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::XPUB).unwrap();
    socket.set_linger(0).unwrap();
    socket.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = socket.get_last_endpoint().unwrap().unwrap();
    (ctx, socket, endpoint)
}

async fn wait_until(mut ready: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(8), async {
        while !ready() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("condition did not become true");
}

fn source(url: &str, endpoint: &str) -> EventSource {
    EventSource {
        worker_url: url.into(),
        endpoint: endpoint.into(),
        topic: "kv".into(),
        profile: "m".into(),
        replay_endpoint: None,
        speed: 1.0,
    }
}

fn config(render: &str, sources: Vec<EventSource>) -> Config {
    let mut cfg = events::tests::config();
    cfg.profiles.get_mut("m").unwrap().render_url = render.into();
    cfg.profiles.get_mut("m").unwrap().cost.prefill_ms_per_token = 100.0;
    cfg.workers = sources;
    cfg.load_poll_ms = 50;
    cfg.load_stale_ms = 1000;
    cfg
}

fn worker(url: &str, role: WorkerType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .runtime_type(BackendType::Vllm)
            .worker_type(role)
            .model(ModelCard::new("m"))
            .build(),
    )
}

fn publish(socket: &zmq::Socket, seq: u64, event: rmpv::Value) {
    let payload = events::tests::payload(vec![event]);
    socket
        .send_multipart(
            [
                b"kv".as_slice(),
                seq.to_be_bytes().as_slice(),
                payload.as_slice(),
            ],
            0,
        )
        .unwrap();
}

fn two_blocks() -> rmpv::Value {
    use rmpv::Value as V;
    V::Array(vec![
        "BlockStored".into(),
        V::Array(vec![1.into(), 2.into()]),
        V::Nil,
        V::Array(vec![1.into(), 2.into(), 3.into(), 4.into()]),
        2.into(),
    ])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_gateway_uses_events_render_and_load_without_changing_forwarding() {
    let a = mock("a").await;
    let b = mock("b").await;
    let (_ca, pa, ea) = publisher();
    let (_cb, pb, eb) = publisher();
    let cfg = config(&a.url, vec![source(&a.url, &ea), source(&b.url, &eb)]);
    let mut gateway = GatewayConfig::new(PolicyConfig::PreKvCache);
    gateway.workers.backend = "vllm".into();
    gateway.routing.pre_kv_cache = Some(cfg.clone());
    let app = Arc::new(AppContext::from_config(gateway.clone()).await.unwrap());
    let wa = worker(&a.url, WorkerType::Regular);
    let wb = worker(&b.url, WorkerType::Regular);
    app.worker_registry.register(wa);
    app.worker_registry.register(wb);
    let runtime = Runtime::new(cfg, app.client.clone()).unwrap();
    app.policy_registry.install_pre_kv_cache(runtime.clone());
    runtime.start(&app.worker_registry);
    wait_until(|| pa.recv_bytes(zmq::DONTWAIT).is_ok()).await;
    wait_until(|| pb.recv_bytes(zmq::DONTWAIT).is_ok()).await;
    publish(&pa, 0, two_blocks());
    wait_until(|| {
        runtime.subscriptions.read()[&a.url]
            .state
            .lock()
            .stream
            .index
            .len()
            == 2
            && runtime.loads.lock().len() == 2
    })
    .await;
    let manager = RouterManager::from_config(&gateway, &app).unwrap();
    let body:ChatCompletionRequest=serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"enable_thinking":false}})).unwrap();
    let response = manager.route_chat(None, &body, Some("m")).await;
    assert!(response.status().is_success());
    let bytes = axum::body::to_bytes(response.into_body(), 1_000_000)
        .await
        .unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["id"], "a");
    assert_eq!(a.renders.load(Ordering::Relaxed), 1);
    assert_eq!(
        a.render_bodies.lock()[0]["chat_template_kwargs"]["enable_thinking"],
        false
    );

    // A cache hit must lose to a sufficiently large queue.
    a.waiting.store(100, Ordering::Relaxed);
    wait_until(|| {
        runtime
            .loads
            .lock()
            .get(&a.url)
            .and_then(|s| s.snapshot.as_ref())
            .is_some_and(|s| s.waiting == 100.0)
    })
    .await;
    let response = manager.route_chat(None, &body, Some("m")).await;
    let bytes = axum::body::to_bytes(response.into_body(), 1_000_000)
        .await
        .unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["id"], "b");

    // Native deletion invalidates the prefix even while a later block remains.
    publish(
        &pa,
        1,
        rmpv::Value::Array(vec![
            "BlockRemoved".into(),
            rmpv::Value::Array(vec![1.into()]),
            "gpu".into(),
        ]),
    );
    wait_until(|| {
        runtime.subscriptions.read()[&a.url]
            .state
            .lock()
            .stream
            .index
            .len()
            == 1
    })
    .await;
    assert_eq!(
        runtime.subscriptions.read()[&a.url]
            .state
            .lock()
            .stream
            .index
            .matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""),
        0
    );
    runtime.stop();
}

#[tokio::test]
async fn retry_reuses_render_and_failed_attempt_releases_credit() {
    let render = mock("render").await;
    let cfg = config(
        &render.url,
        vec![
            source("http://w1", "tcp://127.0.0.1:30001"),
            source("http://w2", "tcp://127.0.0.1:30002"),
        ],
    );
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    runtime.select(&workers, &info).await.unwrap();
    assert_eq!(
        runtime
            .loads
            .lock()
            .values()
            .map(|s| s.credits.len())
            .sum::<usize>(),
        1
    );
    request.finish_attempt(false);
    assert_eq!(
        runtime
            .loads
            .lock()
            .values()
            .map(|s| s.credits.len())
            .sum::<usize>(),
        0
    );
    runtime.select(&workers, &info).await.unwrap();
    assert_eq!(render.renders.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn burst_reservations_spread_equal_workers() {
    let render = mock("render").await;
    let cfg = config(
        &render.url,
        vec![
            source("http://w1", "tcp://127.0.0.1:30001"),
            source("http://w2", "tcp://127.0.0.1:30002"),
        ],
    );
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let requests: Vec<_> = (0..20)
        .map(|_| {
            KvRequest::new(
                "/v1/chat/completions/render",
                json!({"model":"m","messages":[]}),
            )
        })
        .collect();
    let selections = futures::future::join_all(requests.iter().map(|r| async {
        runtime
            .select(
                &workers,
                &SelectWorkerInfo {
                    kv_request: Some(r),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    }))
    .await;
    assert_eq!(selections.iter().filter(|i| **i == 0).count(), 10);
}

#[tokio::test]
async fn render_failure_and_unhealthy_workers_fall_back_to_load() {
    let cfg = config(
        "http://127.0.0.1:1",
        vec![
            source("http://w1", "tcp://127.0.0.1:30001"),
            source("http://w2", "tcp://127.0.0.1:30002"),
        ],
    );
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    workers[0].increment_load();
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
    request.finish_attempt(false);
    workers[1].set_healthy(false);
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
}

#[tokio::test]
async fn missing_local_tokenizer_falls_back_to_load_without_render() {
    let render = mock("render").await;
    let mut cfg = config(
        &render.url,
        vec![
            source("http://w1", "tcp://127.0.0.1:30001"),
            source("http://w2", "tcp://127.0.0.1:30002"),
        ],
    );
    cfg.profiles.get_mut("m").unwrap().token_source = TokenSource::LocalTokenizer;
    let runtime = Runtime::new_with_tokenizer_registry(
        cfg,
        reqwest::Client::new(),
        Arc::new(TokenizerRegistry::new()),
    )
    .unwrap();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    workers[0].increment_load();
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[{"role":"user","content":"hello"}]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };

    assert_eq!(runtime.select(&workers, &info).await, Some(1));
    assert_eq!(render.renders.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn scheduler_and_policy_share_one_local_tokenization() {
    // The exact prompt tokens decide the Regular/PD pool and then feed the KV
    // lookup. Scheduling the same request twice must still call encode once.
    use crate::core::{
        model_type::Endpoint,
        scheduler::{Scheduler, SchedulerInput, WorkerSelection},
    };
    use crate::policies::PolicyRegistry;

    let tokenizer = Arc::new(CountingTokenizer::default());
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load("id", "m", "m", || {
            let tokenizer = tokenizer.clone();
            async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
        })
        .await
        .unwrap();

    let mut cfg = config(
        "",
        vec![
            source("http://w1", "tcp://127.0.0.1:30001"),
            source("http://p1", "tcp://127.0.0.1:30002"),
            source("http://p2", "tcp://127.0.0.1:30003"),
        ],
    );
    cfg.profiles.get_mut("m").unwrap().token_source = TokenSource::LocalTokenizer;
    let runtime = Runtime::new_with_tokenizer_registry(
        cfg,
        reqwest::Client::new(),
        tokenizer_registry.clone(),
    )
    .unwrap();
    let worker_registry = Arc::new(WorkerRegistry::new());
    worker_registry.register(Arc::new(
        BasicWorkerBuilder::new("http://w1")
            .runtime_type(BackendType::Vllm)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .preferred_context_tokens(3)
            .build(),
    ));
    for (url, role) in [
        (
            "http://p1",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ),
        (
            "http://p2",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ),
        ("http://d1", WorkerType::Decode),
        ("http://d2", WorkerType::Decode),
    ] {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .runtime_type(BackendType::Vllm)
                .worker_type(role)
                .model(ModelCard::new("m"))
                .label("pd_group_id", "g")
                .build(),
        );
        if url == "http://p2" {
            seed(&runtime, &worker);
        }
        worker_registry.register(worker);
    }
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::PreKvCache));
    policies.install_pre_kv_cache(runtime);
    let scheduler =
        Scheduler::new_with_tokenizer_registry(worker_registry, policies, Some(tokenizer_registry));
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[{"role":"user","content":"hello"}]}),
    );
    let input = SchedulerInput {
        kv_request: Some(&request),
        model_id: Some("m"),
        endpoint: Endpoint::Chat,
        request_text: Some("hello"),
        token_ids: None,
        headers: None,
    };

    for _ in 0..2 {
        match scheduler.schedule(&input).await.unwrap().worker_selection {
            WorkerSelection::PrefillDecode { prefill, .. } => {
                assert_eq!(prefill.url(), "http://p2");
            }
            other => panic!("expected PD, got {other:?}"),
        }
        request.finish_attempt(false);
    }

    assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn scheduler_never_calls_render_to_measure_context_length() {
    // Render remains a policy-only token source. Length-based pool selection
    // must keep its local text path instead of adding an upstream HTTP call.
    let render = mock("render").await;
    let cfg = config(
        &render.url,
        vec![source("http://w1", "tcp://127.0.0.1:30001")],
    );
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[{"role":"user","content":"hello"}]}),
    );

    assert_eq!(runtime.local_request_token_count("m", &request).await, None);
    assert_eq!(render.renders.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn completion_render_accepts_array_response_and_rejects_batch() {
    let render = mock("render").await;
    let cfg = config(&render.url, vec![]);
    let request = KvRequest::new(
        "/v1/completions/render",
        json!({"model":"m","prompt":"hello"}),
    );
    assert_eq!(
        request
            .tokens(
                "m",
                &cfg.profiles["m"],
                &cfg,
                &reqwest::Client::new(),
                &TokenizerRegistry::new(),
                None,
            )
            .await
            .unwrap()
            .tokens
            .len(),
        5
    );
    let request = KvRequest::new(
        "/v1/completions/render",
        json!({"model":"m","prompt":["a","b"]}),
    );
    assert_eq!(
        request
            .tokens(
                "m",
                &cfg.profiles["m"],
                &cfg,
                &reqwest::Client::new(),
                &TokenizerRegistry::new(),
                None,
            )
            .await
            .unwrap_err(),
        "unsupported_batch"
    );
}

fn seed(runtime: &Runtime, worker: &Arc<dyn Worker>) {
    let subscription = runtime.subscriptions.read()[worker.url()].clone();
    let mut state = subscription.state.lock();
    state.binding = Some(worker.clone());
    state
        .stream
        .apply(
            0,
            &events::tests::payload(vec![two_blocks()]),
            "m",
            &runtime.config,
        )
        .unwrap();
}

#[tokio::test]
async fn stale_events_and_worker_replacement_cannot_reuse_old_hits() {
    let render = mock("render").await;
    let cfg = config(
        &render.url,
        vec![
            source("http://w1", "tcp://w1:1"),
            source("http://w2", "tcp://w2:1"),
        ],
    );
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let mut workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    workers[0].increment_load();
    seed(&runtime, &workers[0]);
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m", "messages":[]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    request.finish_attempt(false);
    runtime.subscriptions.read()["http://w1"]
        .state
        .lock()
        .stream
        .last_event = Some(Instant::now() - Duration::from_secs(60));
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
    request.finish_attempt(false);
    seed(&runtime, &workers[0]);
    workers[0] = worker("http://w1", WorkerType::Regular);
    workers[0].increment_load();
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publisher_disconnect_invalidates_residency_before_ttl() {
    let render = mock("render").await;
    let (_context, publisher, endpoint) = publisher();
    let cfg = config(&render.url, vec![source(&render.url, &endpoint)]);
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    registry.register(worker(&render.url, WorkerType::Regular));
    runtime.start(&registry);
    wait_until(|| publisher.recv_bytes(zmq::DONTWAIT).is_ok()).await;
    publish(&publisher, 0, two_blocks());
    wait_until(|| {
        runtime.subscriptions.read()[&render.url]
            .state
            .lock()
            .stream
            .index
            .len()
            == 2
    })
    .await;
    drop(publisher);
    wait_until(|| {
        runtime.subscriptions.read()[&render.url]
            .state
            .lock()
            .stream
            .last_event
            .is_none()
    })
    .await;
    assert_eq!(
        runtime.subscriptions.read()[&render.url]
            .state
            .lock()
            .stream
            .index
            .len(),
        0
    );
    runtime.stop();
}

#[tokio::test]
async fn prefill_policy_preserves_length_priority_and_decode_selection() {
    // Exercise the existing scheduler, including its mode and PD-group selection.
    use crate::core::{
        model_type::Endpoint,
        scheduler::{Scheduler, SchedulerInput, WorkerSelection},
    };
    use crate::policies::{PolicyRegistry, RoundRobinPolicy};
    let render = mock("render").await;
    let cfg = config(
        &render.url,
        vec![
            source("http://p1", "tcp://p1:1"),
            source("http://p2", "tcp://p2:1"),
        ],
    );
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    let regular: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://regular")
            .model(ModelCard::new("m"))
            .runtime_type(BackendType::Vllm)
            .preferred_context_tokens(3)
            .build(),
    );
    registry.register(regular);
    for (url, role) in [
        (
            "http://p1",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ),
        (
            "http://p2",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ),
        ("http://d1", WorkerType::Decode),
        ("http://d2", WorkerType::Decode),
    ] {
        let w: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(role)
                .runtime_type(BackendType::Vllm)
                .model(ModelCard::new("m"))
                .label("pd_group_id", "g")
                .build(),
        );
        if url == "http://p2" {
            seed(&runtime, &w);
        }
        registry.register(w);
    }
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    policies.set_prefill_policy(Arc::new(PreKvCachePolicy::default()));
    policies.set_decode_policy(Arc::new(RoundRobinPolicy::new()));
    policies.install_pre_kv_cache(runtime);
    let scheduler = Scheduler::new(registry, policies);
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m", "messages":[]}),
    );
    let mut input = SchedulerInput {
        kv_request: Some(&request),
        model_id: Some("m"),
        endpoint: Endpoint::Chat,
        request_text: None,
        token_ids: Some(&[99]),
        headers: None,
    };
    assert!(matches!(
        scheduler.schedule(&input).await.unwrap().worker_selection,
        WorkerSelection::Regular { .. }
    ));
    assert_eq!(render.renders.load(Ordering::Relaxed), 0);
    input.token_ids = Some(&[99; 10]);
    let mut decodes = Vec::new();
    for _ in 0..2 {
        match scheduler.schedule(&input).await.unwrap().worker_selection {
            WorkerSelection::PrefillDecode {
                prefill,
                decode,
                pd_group_id,
            } => {
                assert_eq!(prefill.url(), "http://p2");
                assert_eq!(pd_group_id, "g");
                decodes.push(decode.url().to_string());
            }
            other => panic!("expected PD, got {other:?}"),
        }
        request.finish_attempt(false);
    }
    assert_ne!(decodes[0], decodes[1]);
    assert_eq!(render.renders.load(Ordering::Relaxed), 1);
    // Rendered length is five, but the legacy short length still selects Regular.
    input.token_ids = Some(&[99]);
    assert!(matches!(
        scheduler.schedule(&input).await.unwrap().worker_selection,
        WorkerSelection::Regular { .. }
    ));
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-priority", "high".parse().unwrap());
    input.token_ids = Some(&[99; 10]);
    input.headers = Some(&headers);
    assert!(matches!(
        scheduler.schedule(&input).await.unwrap().worker_selection,
        WorkerSelection::Regular { .. }
    ));
}

#[test]
fn cli_config_is_opt_in_and_rejects_unsupported_combinations() {
    use crate::config::cli::Cli;
    use clap::Parser;
    let file = tempfile::NamedTempFile::new().unwrap();
    let cfg = config(
        "http://render",
        vec![source("http://worker", "tcp://worker:5557")],
    );
    std::fs::write(file.path(), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let args = [
        "gateway",
        "--backend",
        "vllm",
        "--worker-urls",
        "http://worker",
        "--policy",
        "pre_kv_cache",
        "--pre-kv-cache-config",
        file.path().to_str().unwrap(),
    ];
    let gateway = Cli::try_parse_from(args)
        .unwrap()
        .try_into_config()
        .unwrap();
    gateway.validate().unwrap();
    let mut bad = gateway.clone();
    bad.routing.pre_kv_cache = None;
    assert!(bad.validate().is_err());
    let mut bad = gateway.clone();
    bad.routing.decode_policy = Some(PolicyConfig::PreKvCache);
    assert!(bad.validate().is_err());
    let mut bad = gateway;
    bad.routing.dp_aware = true;
    assert!(bad.validate().is_err());
    let old = Cli::try_parse_from([
        "gateway",
        "--backend",
        "sglang",
        "--worker-urls",
        "http://worker",
        "--policy",
        "round_robin",
    ])
    .unwrap()
    .try_into_config()
    .unwrap();
    assert!(old.routing.pre_kv_cache.is_none());
    old.validate().unwrap();
}

#[test]
fn cli_inline_json_supports_startup_without_files() {
    // The existing direct startup command needs only model and discovery settings.
    use crate::config::cli::Cli;
    use clap::Parser;
    let json = r#"{"profiles":{"qwen":{"model":"qwen3-0.6b","block_size":16}},"discovery":{}}"#;
    let gateway = Cli::try_parse_from([
        "gateway",
        "--backend",
        "vllm",
        "--service-discovery",
        "--policy",
        "pre_kv_cache",
        "--pre-kv-cache-config-json",
        json,
    ])
    .unwrap()
    .try_into_config()
    .unwrap();
    let config = gateway.routing.pre_kv_cache.unwrap();
    assert_eq!(config.profiles["qwen"].model, "qwen3-0.6b");
    assert_eq!(config.profiles["qwen"].block_size, 16);
    assert_eq!(config.discovery.unwrap().event_port, 5557);
    assert!(config.workers.is_empty());
}

#[test]
fn cli_structured_config_matches_inline_json() {
    // The readable single-profile flags must build the same canonical config as JSON.
    use crate::config::cli::Cli;
    use clap::Parser;

    let common = [
        "gateway",
        "--backend",
        "vllm",
        "--service-discovery",
        "--policy",
        "pre_kv_cache",
    ];
    let json = r#"{"profiles":{"qwen3":{"model":"/models/Qwen3-0.6B","block_size":128,"token_source":"local_tokenizer"}},"discovery":{"event_port":5557,"replay_port":5558,"topic":"kv","refresh_ms":1000}}"#;
    let from_json = Cli::try_parse_from(
        common
            .into_iter()
            .chain(["--pre-kv-cache-config-json", json]),
    )
    .unwrap()
    .try_into_config()
    .unwrap()
    .routing
    .pre_kv_cache
    .unwrap();
    let from_flags = Cli::try_parse_from(common.into_iter().chain([
        "--pre-kv-cache-profile",
        "qwen3",
        "--pre-kv-cache-model",
        "/models/Qwen3-0.6B",
        "--pre-kv-cache-block-size",
        "128",
        "--pre-kv-cache-token-source",
        "local_tokenizer",
        "--pre-kv-cache-event-port",
        "5557",
        "--pre-kv-cache-replay-port",
        "5558",
        "--pre-kv-cache-topic",
        "kv",
        "--pre-kv-cache-refresh-ms",
        "1000",
    ]))
    .unwrap()
    .try_into_config()
    .unwrap()
    .routing
    .pre_kv_cache
    .unwrap();

    assert_eq!(
        serde_json::to_value(from_flags).unwrap(),
        serde_json::to_value(from_json).unwrap()
    );
}

#[test]
fn cli_structured_config_uses_existing_defaults() {
    // Optional flags stay aligned with Config, Discovery and Profile defaults.
    use crate::config::cli::Cli;
    use clap::Parser;

    let config = Cli::try_parse_from([
        "gateway",
        "--backend",
        "vllm",
        "--service-discovery",
        "--policy",
        "pre_kv_cache",
        "--pre-kv-cache-profile",
        "qwen3",
        "--pre-kv-cache-model",
        "/models/Qwen3-0.6B",
        "--pre-kv-cache-block-size",
        "128",
    ])
    .unwrap()
    .try_into_config()
    .unwrap()
    .routing
    .pre_kv_cache
    .unwrap();

    assert_eq!(config.profiles["qwen3"].token_source, TokenSource::Render);
    let discovery = config.discovery.unwrap();
    assert_eq!(discovery.event_port, 5557);
    assert_eq!(discovery.replay_port, Some(5558));
    assert_eq!(discovery.topic, "kv");
    assert_eq!(discovery.refresh_ms, 1000);
}

#[test]
fn cli_structured_config_preserves_round_robin_pd_startup() {
    // Config presence still enables shared local tokenization in the RR control setup.
    use crate::config::cli::Cli;
    use clap::Parser;

    let gateway = Cli::try_parse_from([
        "gateway",
        "--backend",
        "vllm",
        "--policy",
        "round_robin",
        "--prefill-policy",
        "round_robin",
        "--decode-policy",
        "round_robin",
        "--pre-kv-cache-profile",
        "qwen3",
        "--pre-kv-cache-model",
        "/var/gfs/public-models/Qwen/Qwen3-0.6B",
        "--pre-kv-cache-block-size",
        "128",
        "--pre-kv-cache-token-source",
        "local_tokenizer",
        "--pd-disaggregation",
        "--service-discovery",
        "--selector",
        "app=regular",
        "--prefill-selector",
        "app=prefill",
        "--decode-selector",
        "app=decode",
        "--kv-connector",
        "nixl",
        "--regular-preferred-context-tokens",
        "16000",
        "--tokenizer-path",
        "/var/gfs/public-models/Qwen/Qwen3-0.6B",
    ])
    .unwrap()
    .try_into_config()
    .unwrap();

    assert!(matches!(gateway.routing.policy, PolicyConfig::RoundRobin));
    assert!(matches!(
        gateway.routing.prefill_policy,
        Some(PolicyConfig::RoundRobin)
    ));
    assert!(matches!(
        gateway.routing.decode_policy,
        Some(PolicyConfig::RoundRobin)
    ));
    assert_eq!(
        gateway.workers.regular_preferred_context_tokens,
        Some(16000)
    );
    assert_eq!(
        gateway.routing.pre_kv_cache.unwrap().profiles["qwen3"].token_source,
        TokenSource::LocalTokenizer
    );
}

#[test]
fn cli_structured_config_requires_complete_identity_and_rejects_legacy_sources() {
    // Partial or ambiguous startup settings must fail before the Gateway starts.
    use crate::config::cli::Cli;
    use clap::{error::ErrorKind, Parser};

    let base = ["gateway", "--backend", "vllm", "--policy", "pre_kv_cache"];
    for tail in [
        vec!["--pre-kv-cache-model", "/models/Qwen3-0.6B"],
        vec!["--pre-kv-cache-block-size", "128"],
        vec!["--pre-kv-cache-topic", "kv"],
    ] {
        assert_eq!(
            Cli::try_parse_from(base.into_iter().chain(tail))
                .unwrap_err()
                .kind(),
            ErrorKind::MissingRequiredArgument
        );
    }
    let error = Cli::try_parse_from(base.into_iter().chain(["--pre-kv-cache-profile", "qwen3"]))
        .unwrap()
        .try_into_config()
        .unwrap_err();
    assert!(format!("{error:#}").contains("--pre-kv-cache-model is required"));
    let error = Cli::try_parse_from(base.into_iter().chain([
        "--pre-kv-cache-profile",
        "qwen3",
        "--pre-kv-cache-model",
        "/models/Qwen3-0.6B",
    ]))
    .unwrap()
    .try_into_config()
    .unwrap_err();
    assert!(format!("{error:#}").contains("--pre-kv-cache-block-size is required"));

    let structured = [
        "--pre-kv-cache-profile",
        "qwen3",
        "--pre-kv-cache-model",
        "/models/Qwen3-0.6B",
        "--pre-kv-cache-block-size",
        "128",
    ];
    for legacy in [
        ["--pre-kv-cache-config-json", "{}"],
        ["--pre-kv-cache-config", "unused.json"],
    ] {
        assert_eq!(
            Cli::try_parse_from(base.into_iter().chain(structured).chain(legacy))
                .unwrap_err()
                .kind(),
            ErrorKind::ArgumentConflict
        );
    }
}

#[test]
fn cli_structured_config_reuses_policy_validation() {
    // Invalid discovery values must fail through the same validator as JSON config.
    use crate::config::cli::Cli;
    use clap::Parser;

    let base = [
        "gateway",
        "--backend",
        "vllm",
        "--service-discovery",
        "--policy",
        "pre_kv_cache",
        "--pre-kv-cache-profile",
        "qwen3",
        "--pre-kv-cache-model",
        "/models/Qwen3-0.6B",
        "--pre-kv-cache-block-size",
        "128",
    ];
    for invalid in [
        vec!["--pre-kv-cache-event-port", "0"],
        vec![
            "--pre-kv-cache-event-port",
            "5557",
            "--pre-kv-cache-replay-port",
            "5557",
        ],
        vec!["--pre-kv-cache-refresh-ms", "0"],
    ] {
        let error = Cli::try_parse_from(base.into_iter().chain(invalid))
            .unwrap()
            .try_into_config()
            .unwrap_err();
        assert!(format!("{error:#}").contains("invalid pre_kv_cache discovery"));
    }
}

#[test]
fn cli_inline_json_rejects_invalid_config_and_conflicting_file() {
    // Bad input must fail at startup rather than silently disable cache routing.
    use crate::config::cli::Cli;
    use clap::Parser;
    let base = ["gateway", "--backend", "vllm", "--policy", "pre_kv_cache"];
    for json in [
        "",
        "{",
        "{}",
        r#"{"profiles":{"q":{"model":"q","block_size":0}},"discovery":{}}"#,
    ] {
        let args = base.into_iter().chain(["--pre-kv-cache-config-json", json]);
        let error = Cli::try_parse_from(args)
            .unwrap()
            .try_into_config()
            .unwrap_err();
        assert!(format!("{error:#}").contains("pre_kv_cache"));
    }
    let args = base.into_iter().chain([
        "--pre-kv-cache-config-json",
        "{}",
        "--pre-kv-cache-config",
        "unused.json",
    ]);
    assert_eq!(
        Cli::try_parse_from(args).unwrap_err().kind(),
        clap::error::ErrorKind::ArgumentConflict
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discovery_tracks_late_replicas_removal_and_replacement_without_urls() {
    // Start with no workers, then follow the same registry updates used by K8s.
    let backend = mock("replica").await;
    let (_context, publisher, endpoint) = publisher();
    // Report every subscriber while old and replacement sockets briefly overlap.
    publisher.set_xpub_verbose(true).unwrap();
    let event_port = url::Url::parse(&endpoint).unwrap().port().unwrap();
    let mut cfg = config("", vec![]);
    cfg.discovery = Some(Discovery {
        event_port,
        replay_port: None,
        refresh_ms: 20,
        ..Discovery::default()
    });
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    runtime.start(&registry);
    assert!(runtime.subscriptions.read().is_empty());

    let regular = worker(&backend.url, WorkerType::Regular);
    let id = registry.register(regular.clone());
    // Decode and other models must not accidentally subscribe to this profile.
    registry.register(worker("http://127.0.0.1:39990", WorkerType::Decode));
    registry.register(Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:39991")
            .runtime_type(BackendType::Vllm)
            .model(ModelCard::new("other"))
            .build(),
    ));
    wait_until(|| {
        publisher
            .recv_bytes(zmq::DONTWAIT)
            .is_ok_and(|v| v == b"\x01kv")
    })
    .await;
    publish(&publisher, 0, two_blocks());
    wait_until(|| {
        runtime
            .subscriptions
            .read()
            .get(&backend.url)
            .is_some_and(|s| s.state.lock().stream.index.len() == 2)
    })
    .await;
    assert_eq!(runtime.subscriptions.read().len(), 1);
    let retired = runtime.subscriptions.read()[&backend.url].clone();
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    assert_eq!(
        runtime
            .select(
                &[regular],
                &SelectWorkerInfo {
                    kv_request: Some(&request),
                    ..Default::default()
                }
            )
            .await,
        Some(0)
    );
    assert_eq!(backend.renders.load(Ordering::Relaxed), 1);
    request.finish_attempt(false);

    // Scale down releases connections, residency and load; next replica starts cold.
    registry.remove(&id);
    wait_until(|| {
        runtime.subscriptions.read().is_empty() && !runtime.loads.lock().contains_key(&backend.url)
    })
    .await;
    assert!(retired.stop.load(Ordering::Acquire));
    assert!(retired.tasks.lock().is_empty());
    assert_eq!(retired.state.lock().stream.index.len(), 0);
    let prefill = worker(
        &backend.url,
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    );
    registry.register(prefill);
    wait_until(|| {
        publisher
            .recv_bytes(zmq::DONTWAIT)
            .is_ok_and(|v| v == b"\x01kv")
    })
    .await;
    let replacement = runtime.subscriptions.read()[&backend.url].clone();
    assert!(!Arc::ptr_eq(&retired, &replacement));
    assert_eq!(replacement.state.lock().stream.index.len(), 0);
    publish(&publisher, 0, two_blocks());
    wait_until(|| replacement.state.lock().stream.index.len() == 2).await;

    // A registration replaced in place also gets a fresh subscription, even if
    // the registry never appears empty between the two Pod generations.
    registry.register(worker(&backend.url, WorkerType::Regular));
    wait_until(|| replacement.stop.load(Ordering::Acquire)).await;
    let newest = runtime.subscriptions.read()[&backend.url].clone();
    assert!(!Arc::ptr_eq(&replacement, &newest));
    assert_eq!(newest.state.lock().stream.index.len(), 0);
    runtime.stop();
    assert!(newest.stop.load(Ordering::Acquire));
}

#[test]
fn discovery_config_requires_no_worker_or_render_url_but_rejects_ambiguity() {
    let cfg: Config = serde_json::from_value(json!({
        "profiles":{"qwen":{"model":"qwen3-0.6b","block_size":16}},
        "discovery":{"event_port":5557,"replay_port":5558,"topic":"kv"}
    }))
    .unwrap();
    cfg.validate().unwrap();
    assert!(cfg.workers.is_empty());
    assert!(cfg.profiles["qwen"].render_url.is_empty());
    let discovery = cfg.discovery.as_ref().unwrap();
    let v4 = discovery.source("http://10.2.3.4:8000", "qwen").unwrap();
    assert_eq!(v4.endpoint, "tcp://10.2.3.4:5557");
    assert_eq!(v4.replay_endpoint.as_deref(), Some("tcp://10.2.3.4:5558"));
    let v6 = discovery.source("http://[fd00::123]:8000", "qwen").unwrap();
    assert_eq!(v6.endpoint, "tcp://[fd00::123]:5557");
    assert!(discovery.source("grpc://worker:8000", "qwen").is_none());
    let mut bad = cfg.clone();
    bad.profiles
        .insert("ambiguous".into(), cfg.profiles["qwen"].clone());
    assert!(bad.validate().is_err());
    let mut bad = cfg.clone();
    bad.discovery.as_mut().unwrap().replay_port = Some(5557);
    assert!(bad.validate().is_err());

    // The existing service-discovery CLI accepts an initially empty worker pool.
    use clap::Parser;
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let gateway = crate::config::cli::Cli::try_parse_from([
        "gateway",
        "--backend",
        "vllm",
        "--service-discovery",
        "--selector",
        "app=qwen",
        "role=leader",
        "--policy",
        "pre_kv_cache",
        "--pre-kv-cache-config",
        file.path().to_str().unwrap(),
    ])
    .unwrap()
    .try_into_config()
    .unwrap();
    gateway.validate().unwrap();
}

#[tokio::test]
async fn discovery_scales_without_duplicate_tasks_and_preserves_explicit_overrides() {
    // Existing static bindings win; additional replicas derive their own endpoints.
    let explicit = source("http://127.0.0.1:18081", "tcp://127.0.0.1:15559");
    let mut cfg = config("", vec![explicit]);
    cfg.discovery = Some(Discovery::default());
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    registry.register(worker("http://127.0.0.1:18081", WorkerType::Regular));
    runtime.reconcile(&registry);
    let original = runtime.subscriptions.read()["http://127.0.0.1:18081"].clone();
    assert_eq!(original.source.endpoint, "tcp://127.0.0.1:15559");

    let added = registry.register(worker(
        "http://[::1]:18082",
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    ));
    runtime.reconcile(&registry);
    let replica = runtime.subscriptions.read()["http://[::1]:18082"].clone();
    assert_eq!(replica.source.endpoint, "tcp://[::1]:5557");
    assert_eq!(replica.tasks.lock().len(), 2);
    runtime.reconcile(&registry);
    assert_eq!(runtime.subscriptions.read().len(), 2);
    assert!(Arc::ptr_eq(
        &original,
        &runtime.subscriptions.read()["http://127.0.0.1:18081"]
    ));
    assert!(Arc::ptr_eq(
        &replica,
        &runtime.subscriptions.read()["http://[::1]:18082"]
    ));
    assert_eq!(replica.tasks.lock().len(), 2);
    registry.remove(&added);
    runtime.reconcile(&registry);
    assert_eq!(runtime.subscriptions.read().len(), 1);
    assert!(replica.stop.load(Ordering::Acquire));
    runtime.stop();
}
