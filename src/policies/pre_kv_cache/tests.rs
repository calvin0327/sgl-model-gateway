//! Acceptance tests exercise real HTTP rendering, native ZMQ and existing routers.

use super::*;
use crate::{
    app_context::AppContext,
    config::{GatewayConfig, PolicyConfig},
    core::{
        model_card::ModelCard,
        model_type::Endpoint,
        scheduler::{Scheduler, SchedulerInput, WorkerSelection},
        BasicWorkerBuilder,
    },
    policies::PolicyRegistry,
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
use tracing::instrument::WithSubscriber;

// Capture the ordinary INFO event, so the contract covers actual emitted fields.
#[derive(Clone, Default)]
struct LogWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct CountingTokenizer {
    encode_calls: AtomicUsize,
    template_calls: AtomicUsize,
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
        self.template_calls.fetch_add(1, Ordering::Relaxed);
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
        // Decode policy selection must preserve the chosen Prefill's coverage.
        assert_eq!(request.selected_cached_tokens(), Some(4));
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

    // With one candidate, logging must not add a render call or a reservation.
    let workers = Arc::new(WorkerRegistry::new());
    workers.register(worker("http://w1", WorkerType::Regular));
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::PreKvCache));
    policies.install_pre_kv_cache(runtime.clone());
    let scheduler = Scheduler::new(workers, policies);
    let input = SchedulerInput {
        kv_request: Some(&request),
        model_id: Some("m"),
        endpoint: Endpoint::Chat,
        request_text: Some("hello"),
        token_ids: None,
        headers: None,
    };
    assert!(matches!(
        scheduler.schedule(&input).await.unwrap().worker_selection,
        WorkerSelection::Regular { .. }
    ));
    assert_eq!(request.selected_cached_tokens(), None);
    assert!(request.reservations.lock().is_empty());
    assert!(runtime.loads.lock().is_empty());
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

// Exercise policy decisions with exact local tokens and explicit load snapshots,
// without polling or HTTP render races changing the candidate costs mid-test.
async fn local_runtime(mut cfg: Config) -> Arc<Runtime> {
    cfg.profiles.get_mut("m").unwrap().token_source = TokenSource::LocalTokenizer;
    let registry = Arc::new(TokenizerRegistry::new());
    registry
        .load("id", "m", "m", || async {
            Ok(Arc::new(CountingTokenizer::default()) as Arc<dyn Tokenizer>)
        })
        .await
        .unwrap();
    Runtime::new_with_tokenizer_registry(cfg, reqwest::Client::new(), registry).unwrap()
}

fn set_load_snapshot(runtime: &Runtime, worker: &Arc<dyn Worker>, waiting: f64) {
    let mut loads = runtime.loads.lock();
    let state = loads.entry(worker.url().to_owned()).or_default();
    state.bind(worker);
    state.snapshot = Some(load::Snapshot {
        running: 0.0,
        waiting,
        kv_usage: Some(0.0),
        observed: Instant::now(),
    });
}

fn v41_config(sources: Vec<EventSource>) -> Config {
    let mut cfg = config("", sources);
    let profile = cfg.profiles.get_mut("m").unwrap();
    profile.token_source = TokenSource::LocalTokenizer;
    profile.block_size = 64;
    profile.layout = Some(LayoutPreset::DeepseekV41Vllm031Dspark5);
    cfg
}

#[tokio::test]
async fn builtin_v41_preserves_hit_reporting_and_charges_replay_work() {
    // Exercise the captured deployment through the real selector: the cached
    // length remains 256, but only 128 tokens of computation can be skipped.
    let runtime = local_runtime(v41_config(vec![
        source("http://w1", "tcp://w1:1"),
        source("http://w2", "tcp://w2:1"),
    ]))
    .await;
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let tokens: Vec<u32> = (1..=257).collect();
    for (i, worker) in workers.iter().enumerate() {
        let subscription = runtime.subscriptions.read()[worker.url()].clone();
        let mut state = subscription.state.lock();
        state.bind(Some(worker.clone()), &runtime.config.profiles["m"]);
        let events = if i == 0 {
            vec![events::tests::v41_store(8, &[1, 2, 3, 4], &tokens[..256])]
        } else {
            vec![]
        };
        state
            .stream
            .apply(0, &events::tests::payload(events), "m", &runtime.config)
            .unwrap();
        set_load_snapshot(&runtime, worker, 0.0);
    }
    let request = KvRequest::new(
        "/v1/completions/render",
        json!({"model":"m", "prompt":tokens}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    assert_eq!(request.selected_cached_tokens(), Some(256));
    assert_eq!(
        runtime.loads.lock()[workers[0].url()]
            .credits
            .values()
            .next()
            .unwrap()
            .tokens,
        129
    );
    request.finish_attempt(false);

    // An idle cache remains usable with a static contract. A new worker at the
    // same URL must still rebuild residency even though its preset is unchanged.
    let subscription = runtime.subscriptions.read()[workers[0].url()].clone();
    let input = request.cached_input("m").unwrap();
    let mut state = subscription.state.lock();
    state.layout_checked = Some(Instant::now() - Duration::from_secs(60));
    assert!(state.layout_is_current());
    // The same old timestamp must remain unusable in automatic discovery mode.
    state.layout_preset = None;
    assert!(!state.layout_is_current());
    state.layout_preset = runtime.config.profiles["m"].layout;
    state.layout_checked = None;
    state.stream.last_event =
        Some(Instant::now() - Duration::from_millis(runtime.config.event_stale_ms + 1));
    drop(state);
    assert_eq!(
        runtime.cached_tokens(&subscription, &workers[0], &input),
        Some(256)
    );
    let replacement = worker(workers[0].url(), WorkerType::Regular);
    let mut state = subscription.state.lock();
    let previous_revision = state.layout_revision;
    assert!(state.bind(Some(replacement.clone()), &runtime.config.profiles["m"]));
    assert!(state.layout_is_current());
    assert!(state.layout_revision > previous_revision);
    assert_eq!(state.stream.index.len(), 0);
    assert!(state.stream.last_event.is_none());
    assert!(!state.bind(Some(replacement.clone()), &runtime.config.profiles["m"]));
    drop(state);
    assert_eq!(
        runtime.cached_tokens(&subscription, &replacement, &input),
        None
    );
    assert_eq!(
        runtime.cached_tokens(&subscription, &workers[0], &input),
        None
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn builtin_v41_subscribes_without_worker_metadata_extension() {
    // A real HTTP worker returns 404 for metadata. The fixed layout must start
    // from native ZMQ events without making that request, even across a poll period.
    let metadata_requests = Arc::new(AtomicUsize::new(0));
    let calls = metadata_requests.clone();
    let app = Router::new()
        .route("/metrics", get(|| async { "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0\n" }))
        .route("/v1/pre-kv-cache/metadata", get(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            async { http::StatusCode::NOT_FOUND }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (_context, publisher, endpoint) = publisher();
    let runtime = local_runtime(v41_config(vec![source(&url, &endpoint)])).await;
    let registry = Arc::new(WorkerRegistry::new());
    let worker = worker(&url, WorkerType::Regular);
    registry.register(worker.clone());
    runtime.start(&registry);
    wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
    let tokens: Vec<u32> = (1..=257).collect();
    publish(
        &publisher,
        0,
        events::tests::v41_store(8, &[1, 2, 3, 4], &tokens[..256]),
    );
    let subscription = runtime.subscriptions.read()[&url].clone();
    wait_until(|| subscription.state.lock().stream.index.len() == 4).await;
    let input = render::Rendered {
        keys: index::prefix_keys(&tokens, 64, "m", ""),
        tokens,
        requires_group_layout: true,
    };
    assert_eq!(
        runtime.cached_tokens(&subscription, &worker, &input),
        Some(256)
    );
    tokio::time::sleep(Duration::from_millis(5200)).await;
    assert_eq!(metadata_requests.load(Ordering::Relaxed), 0);
    assert_eq!(
        runtime.cached_tokens(&subscription, &worker, &input),
        Some(256)
    );
    let status = runtime.status();
    let entry = &status["workers"][0];
    assert_eq!(entry["layout_source"], "builtin_preset");
    assert_eq!(entry["layout_preset"], "deepseek_v41_vllm031_dspark5");
    assert_eq!(entry["layout_engine_verified"], false);
    assert_eq!(entry["layout_age_ms"], Value::Null);
    assert_eq!(entry["runtime_layout"]["engine_id"], "");
    assert_eq!(entry["runtime_layout"]["lookup_groups"], json!([[8]]));
    assert!(entry["warnings"].as_array().unwrap().is_empty());
    runtime.stop();
    server.abort();
}

// Keep the metadata file alive with its registry entry. Detection must inspect
// the model configuration without tokenizing or applying a chat template.
async fn register_layout_tokenizer(
    registry: &TokenizerRegistry,
    name: &str,
    model_type: &str,
) -> (tempfile::TempDir, Arc<CountingTokenizer>) {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("config.json"),
        serde_json::to_vec(&json!({"model_type":model_type})).unwrap(),
    )
    .unwrap();
    let tokenizer = Arc::new(CountingTokenizer::default());
    registry
        .load(
            &TokenizerRegistry::generate_id(),
            name,
            directory.path().to_str().unwrap(),
            || {
                let tokenizer = tokenizer.clone();
                async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
            },
        )
        .await
        .unwrap();
    (directory, tokenizer)
}

#[tokio::test]
async fn automatic_v41_selection_respects_model_identity_and_explicit_configuration() {
    // The profile name is arbitrary. Only an exact model registration with
    // V4.1 metadata can supply the automatic default, with no sole-entry fallback.
    let mut profile = v41_config(vec![]).profiles.remove("m").unwrap();
    profile.layout = None;
    let registry = TokenizerRegistry::new();
    let (_unrelated_dir, unrelated) =
        register_layout_tokenizer(&registry, "another-model", "deepseek_v41").await;
    assert_eq!(
        metadata::automatic_tokenizer(&profile, &registry).err(),
        Some("local_tokenizer_unavailable")
    );
    let (_generic_dir, generic) = register_layout_tokenizer(&registry, "m", "qwen3").await;
    assert!(metadata::automatic_tokenizer(&profile, &registry)
        .unwrap()
        .is_none());
    registry.remove("m").unwrap();
    let (_v41_dir, v41) = register_layout_tokenizer(&registry, "m", "deepseek_v41").await;
    let entry = metadata::automatic_tokenizer(&profile, &registry)
        .unwrap()
        .unwrap();
    assert_eq!(entry.name, "m");
    // Explicit IDs remain supported without guessing from the profile label.
    let mut by_id = profile.clone();
    by_id.model = entry.id.clone();
    assert!(metadata::automatic_tokenizer(&by_id, &registry)
        .unwrap()
        .is_some());

    let mut wrong_block = profile.clone();
    wrong_block.block_size = 32;
    assert!(metadata::automatic_tokenizer(&wrong_block, &registry)
        .unwrap()
        .is_none());
    let mut explicit = profile.clone();
    explicit.layout = Some(LayoutPreset::DeepseekV41Vllm031Dspark5);
    let mut manual = profile.clone();
    manual.cache_groups = vec![CacheGroup {
        group_idx: 8,
        block_size: 64,
        kind: CacheKind::FullAttention,
        sliding_window: None,
    }];
    let mut remote = profile;
    remote.token_source = TokenSource::Render;
    for profile in [explicit, manual, remote] {
        assert!(metadata::automatic_tokenizer(&profile, &registry)
            .unwrap()
            .is_none());
    }
    for tokenizer in [unrelated, generic, v41] {
        assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 0);
        assert_eq!(tokenizer.template_calls.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn automatic_v41_model_replacement_and_unloading_discard_old_residency() {
    // Repeated detection preserves state, while changed tokenizer/worker
    // identity and explicit removal start a clean event-index generation.
    let registry = TokenizerRegistry::new();
    let (_first_dir, _first_tokenizer) =
        register_layout_tokenizer(&registry, "m", "deepseek_v41").await;
    let mut cfg = v41_config(vec![]);
    cfg.profiles.get_mut("m").unwrap().layout = None;
    let first = metadata::automatic_tokenizer(&cfg.profiles["m"], &registry)
        .unwrap()
        .unwrap();
    let mut state = WorkerState::default();
    state.bind(
        Some(worker("http://w1", WorkerType::Regular)),
        &cfg.profiles["m"],
    );
    assert!(state.set_automatic_layout(Some(&first)));
    let tokens: Vec<u32> = (1..=256).collect();
    let batch = events::tests::payload(vec![events::tests::v41_store(8, &[1, 2, 3, 4], &tokens)]);
    state.stream.apply(0, &batch, "m", &cfg).unwrap();
    assert!(state.automatic_events_confirmed());
    state.stream.unsupported = true;
    let revision = state.layout_revision;
    assert!(!state.set_automatic_layout(Some(&first)));
    assert_eq!(state.stream.index.len(), 4);
    assert!(state.stream.unsupported);
    assert_eq!(state.layout_revision, revision);

    registry.remove("m").unwrap();
    let (_second_dir, _second_tokenizer) =
        register_layout_tokenizer(&registry, "m", "deepseek_v41").await;
    let second = metadata::automatic_tokenizer(&cfg.profiles["m"], &registry)
        .unwrap()
        .unwrap();
    // The first entry is deliberately still held alive: routing must reject
    // its old instance immediately, before the next background detection tick.
    assert!(!state.automatic_tokenizer_is_current(&registry, "m"));
    assert!(state.set_automatic_layout(Some(&second)));
    assert!(state.automatic_tokenizer_is_current(&registry, "m"));
    assert_eq!(state.layout_revision, revision + 1);
    assert_eq!(state.stream.index.len(), 0);
    assert!(!state.stream.unsupported);
    assert!(!state.automatic_events_confirmed());
    assert!(state.layout_is_current());

    state.stream.apply(0, &batch, "m", &cfg).unwrap();
    assert!(state.set_automatic_layout(None));
    assert_eq!(state.stream.index.len(), 0);
    assert!(state.stream.layout.is_none());
    assert!(state.layout_preset.is_none());
    assert!(state.layout_tokenizer.is_none());
    assert!(!state.layout_is_current());
    assert!(!state.set_automatic_layout(None));

    assert!(state.set_automatic_layout(Some(&second)));
    state.stream.apply(0, &batch, "m", &cfg).unwrap();
    assert!(state.bind(
        Some(worker("http://w1", WorkerType::Regular)),
        &cfg.profiles["m"]
    ));
    assert_eq!(state.stream.index.len(), 0);
    assert!(state.stream.layout.is_none());
    assert!(state.layout_tokenizer.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_v41_waits_for_its_tokenizer_and_keeps_event_failures_sticky() {
    // Discover the layout from this model's local registration, even when it
    // arrives after ZMQ. No request rendering or Worker extension is involved.
    let metadata_requests = Arc::new(AtomicUsize::new(0));
    let render_requests = Arc::new(AtomicUsize::new(0));
    let metadata_calls = metadata_requests.clone();
    let render_calls = render_requests.clone();
    let app = Router::new()
        .route("/metrics", get(|| async { "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0\n" }))
        .route("/v1/pre-kv-cache/metadata", get(move || {
            metadata_calls.fetch_add(1, Ordering::Relaxed);
            async { http::StatusCode::NOT_FOUND }
        }))
        .route("/v1/chat/completions/render", post(move || {
            render_calls.fetch_add(1, Ordering::Relaxed);
            async { http::StatusCode::NOT_FOUND }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (_context, publisher, endpoint) = publisher();
    // A layout switch reconnects while the old SUB may still be closing. Ask
    // XPUB for each subscription, not only the first subscriber to the topic.
    publisher.set_xpub_verbose(true).unwrap();
    let mut cfg = v41_config(vec![source(&url, &endpoint)]);
    cfg.profiles.get_mut("m").unwrap().layout = None;
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let runtime = Runtime::new_with_tokenizer_registry(
        cfg,
        reqwest::Client::new(),
        tokenizer_registry.clone(),
    )
    .unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    let worker = worker(&url, WorkerType::Regular);
    registry.register(worker.clone());
    runtime.start(&registry);
    wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
    let subscription = runtime.subscriptions.read()[&url].clone();

    // A sole unrelated V4.1 tokenizer must not become this worker's contract.
    // Neither the arbitrary served-model name "m" nor the profile selects it.
    let model_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        model_dir.path().join("config.json"),
        r#"{"model_type":"deepseek_v41","architectures":["DeepseekV41ForCausalLM"]}"#,
    )
    .unwrap();
    tokenizer_registry
        .load(
            "unrelated-v41",
            "another-model",
            model_dir.path().to_str().unwrap(),
            || async { Ok(Arc::new(CountingTokenizer::default()) as Arc<dyn Tokenizer>) },
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(5200)).await;
    assert!(subscription.state.lock().stream.layout.is_none());
    assert_eq!(metadata_requests.load(Ordering::Relaxed), 0);

    let tokenizer = Arc::new(CountingTokenizer::default());
    tokenizer_registry
        .load(
            "matching-v41",
            "m",
            model_dir.path().to_str().unwrap(),
            || {
                let tokenizer = tokenizer.clone();
                async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
            },
        )
        .await
        .unwrap();
    wait_until(|| subscription.state.lock().stream.layout.is_some()).await;
    // Installing the new contract replaces the subscription. Wait for its
    // positive XPUB notification before sending events to the fresh socket.
    wait_until(|| {
        publisher
            .recv_bytes(zmq::DONTWAIT)
            .is_ok_and(|frame| frame.first() == Some(&1))
    })
    .await;
    let empty_batch = events::tests::payload(vec![]);
    publisher
        .send_multipart(
            [
                b"kv".as_slice(),
                0u64.to_be_bytes().as_slice(),
                empty_batch.as_slice(),
            ],
            0,
        )
        .unwrap();
    wait_until(|| subscription.state.lock().stream.last_seq == Some(0)).await;
    let tokens: Vec<u32> = (1..=257).collect();
    let input = render::Rendered {
        keys: index::prefix_keys(&tokens, 64, "m", ""),
        tokens: tokens.clone(),
        requires_group_layout: true,
    };
    assert_eq!(runtime.cached_tokens(&subscription, &worker, &input), None);
    let status = runtime.status();
    let entry = &status["workers"][0];
    assert_eq!(entry["layout_source"], "automatic_builtin");
    assert_eq!(entry["layout_preset"], "deepseek_v41_vllm031_dspark5");
    assert_eq!(entry["layout_engine_verified"], false);
    assert_eq!(entry["layout_events_confirmed"], false);
    assert_eq!(entry["runtime_layout"]["lookup_groups"], json!([[8]]));

    // Only a matching lookup-group Store proves the live stream is usable.
    publish(
        &publisher,
        1,
        events::tests::v41_store(8, &[1, 2, 3, 4], &tokens[..256]),
    );
    wait_until(|| subscription.state.lock().stream.index.len() == 4).await;
    assert_eq!(
        runtime.cached_tokens(&subscription, &worker, &input),
        Some(256)
    );
    assert_eq!(
        runtime.status()["workers"][0]["layout_events_confirmed"],
        true
    );
    let revision = subscription.state.lock().layout_revision;
    tokio::time::sleep(Duration::from_millis(5200)).await;
    assert_eq!(subscription.state.lock().layout_revision, revision);
    assert_eq!(
        runtime.cached_tokens(&subscription, &worker, &input),
        Some(256)
    );

    // A mismatching event invalidates the contract's residency. Re-identifying
    // the same tokenizer on later polls must never revive that failed stream.
    let mut mismatch = events::tests::v41_store(8, &[5], &tokens[..64]);
    if let rmpv::Value::Array(fields) = &mut mismatch {
        fields[4] = 32.into();
    }
    publish(&publisher, 2, mismatch);
    wait_until(|| subscription.state.lock().stream.unsupported).await;
    assert_eq!(runtime.cached_tokens(&subscription, &worker, &input), None);
    tokio::time::sleep(Duration::from_millis(5200)).await;
    assert!(subscription.state.lock().stream.unsupported);
    assert_eq!(subscription.state.lock().layout_revision, revision);
    assert_eq!(runtime.cached_tokens(&subscription, &worker, &input), None);
    assert_eq!(metadata_requests.load(Ordering::Relaxed), 0);
    assert_eq!(render_requests.load(Ordering::Relaxed), 0);
    assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(tokenizer.template_calls.load(Ordering::Relaxed), 0);
    runtime.stop();
    server.abort();
}

fn log_records(output: &LogWriter) -> Vec<Value> {
    String::from_utf8(output.0.lock().clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn bounded_replay_is_charged_without_changing_reported_cache_hits() {
    // Equal cache residency need not save equal work: the first worker must
    // recompute a tail. Selection and pending-token reservations include it.
    let runtime = local_runtime(config(
        "",
        vec![
            source("http://w1", "tcp://w1:1"),
            source("http://w2", "tcp://w2:1"),
        ],
    ))
    .await;
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    for (i, worker) in workers.iter().enumerate() {
        seed(&runtime, worker);
        set_load_snapshot(&runtime, worker, 0.0);
        let subscription = runtime.subscriptions.read()[worker.url()].clone();
        let mut state = subscription.state.lock();
        let mut layout = layout::fixture();
        layout.prefix_replay_tokens = if i == 0 { 2 } else { 0 };
        state.stream.layout = Some(Arc::new(layout));
        state.layout_checked = Some(Instant::now());
    }
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
    assert_eq!(request.selected_cached_tokens(), Some(4));
    assert_eq!(
        runtime.loads.lock()[workers[1].url()]
            .credits
            .values()
            .next()
            .unwrap()
            .tokens,
        1
    );
    request.finish_attempt(false);
    workers[1].set_healthy(false);
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    assert_eq!(request.selected_cached_tokens(), Some(4));
    assert_eq!(
        runtime.loads.lock()[workers[0].url()]
            .credits
            .values()
            .next()
            .unwrap()
            .tokens,
        3
    );
    request.finish_attempt(false);
}

#[tokio::test]
async fn grouped_local_routing_uses_common_residency_and_reports_diagnostics() {
    // Drive the real runtime from grouped events. Only a worker with every
    // required group may advertise a hit; eviction changes the next decision.
    use rmpv::Value as V;
    let cfg = config(
        "",
        vec![
            source("http://w1", "tcp://w1:1"),
            source("http://w2", "tcp://w2:1"),
        ],
    );
    let runtime = local_runtime(cfg).await;
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let store = |group: u64, tokens: &[u32]| {
        let mut fields = two_blocks().as_array().unwrap().clone();
        fields[3] = V::Array(tokens.iter().map(|n| (*n).into()).collect());
        fields.resize(12, V::Nil);
        fields[9] = group.into();
        fields[10] = if group == 0 {
            "FullAttentionSpec"
        } else {
            "SlidingWindowSpec"
        }
        .into();
        if group == 1 {
            fields[11] = 3.into();
        }
        V::Array(fields)
    };
    let apply = |worker: &Arc<dyn Worker>, seq, events| {
        let subscription = runtime.subscriptions.read()[worker.url()].clone();
        let mut state = subscription.state.lock();
        state.binding = Some(worker.clone());
        // A complete engine response replaces manual group configuration.
        let mut layout = layout::fixture();
        let mut window = layout.groups[0].clone();
        window.group_idx = 1;
        window.kind = "sliding_window".into();
        window.manager = "SlidingWindowManager".into();
        window.sliding_window = Some(3);
        layout.groups.push(window);
        layout.coordinator = "HybridKVCacheCoordinator".into();
        layout.lookup_groups = vec![vec![0], vec![1]];
        state.stream.layout = Some(Arc::new(layout));
        state.layout_checked = Some(Instant::now());
        state
            .stream
            .apply(seq, &events::tests::payload(events), "m", &runtime.config)
            .unwrap();
    };
    let before = runtime.status();
    assert_eq!(before["workers"][0]["bound"], false);
    assert_eq!(before["workers"][0]["last_valid_event_age_ms"], Value::Null);
    apply(&workers[0], 0, vec![store(0, &[1, 2, 3, 4])]);
    let input = render::Rendered {
        tokens: vec![1, 2, 3, 4, 5],
        keys: index::prefix_keys(&[1, 2, 3, 4, 5], 2, "m", ""),
        requires_group_layout: true,
    };
    let first = runtime.subscriptions.read()[workers[0].url()].clone();
    assert_eq!(runtime.cached_tokens(&first, &workers[0], &input), Some(0));
    assert!(!runtime
        .diagnostics
        .reasons(workers[0].url())
        .contains(&"event_state_unavailable"));
    apply(&workers[0], 1, vec![store(1, &[1, 2, 3, 4])]);
    assert_eq!(runtime.cached_tokens(&first, &workers[0], &input), Some(4));
    assert!(!runtime
        .diagnostics
        .reasons(workers[0].url())
        .contains(&"event_state_unavailable"));
    apply(
        &workers[1],
        0,
        vec![store(0, &[9, 8, 7, 6]), store(1, &[9, 8, 7, 6])],
    );
    set_load_snapshot(&runtime, &workers[0], 1.0);
    set_load_snapshot(&runtime, &workers[1], 0.0);
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    assert_eq!(request.selected_cached_tokens(), Some(4));
    request.finish_attempt(false);
    apply(
        &workers[0],
        2,
        vec![V::Array(vec![
            "BlockRemoved".into(),
            V::Array(vec![2.into()]),
            "GPU".into(),
            1.into(),
        ])],
    );
    assert_eq!(runtime.cached_tokens(&first, &workers[0], &input), Some(2));
    apply(
        &workers[0],
        3,
        vec![V::Array(vec![
            "BlockRemoved".into(),
            V::Array(vec![1.into()]),
            "GPU".into(),
            0.into(),
        ])],
    );
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
    assert_eq!(request.selected_cached_tokens(), Some(0));
    let status = runtime.status();
    assert_eq!(
        status["workers"][0]["observed_groups"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(status["workers"][0]["resident_blocks"], 2);
    assert_eq!(status["workers"][0]["observed_groups_are_complete"], false);
    assert_eq!(status["workers"][0]["layout_source"], "vllm_engine");
    assert_eq!(
        status["workers"][0]["runtime_layout"]["groups"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let entry = runtime.tokenizer_registry.get_by_name("m").unwrap();
    let tokenizer = entry
        .tokenizer
        .as_any()
        .downcast_ref::<CountingTokenizer>()
        .unwrap();
    assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn v41_layout_is_required_and_rejected_metadata_remains_visible() {
    // A valid legacy index does not establish V4.1's complete group layout.
    let runtime = local_runtime(config("", vec![source("http://w1", "tcp://w1:1")])).await;
    let worker = worker("http://w1", WorkerType::Regular);
    seed(&runtime, &worker);
    let input = render::Rendered {
        tokens: vec![1, 2, 3, 4, 5],
        keys: index::prefix_keys(&[1, 2, 3, 4, 5], 2, "m", ""),
        requires_group_layout: true,
    };
    let subscription = runtime.subscriptions.read()[worker.url()].clone();
    assert_eq!(runtime.cached_tokens(&subscription, &worker, &input), None);
    assert!(runtime
        .diagnostics
        .reasons(worker.url())
        .contains(&"cache_group_layout_missing"));
    let mut fields = two_blocks().as_array().unwrap().clone();
    fields.resize(12, rmpv::Value::Nil);
    fields[9] = 1.into();
    fields[10] = "SlidingWindowSpec".into();
    fields[11] = 3.into();
    assert_eq!(
        subscription.state.lock().stream.apply(
            1,
            &events::tests::payload(vec![rmpv::Value::Array(fields)]),
            "m",
            &runtime.config
        ),
        Err("unsupported_group")
    );
    let status = runtime.status();
    assert_eq!(status["workers"][0]["unsupported"], true);
    assert_eq!(status["workers"][0]["last_error"], "unsupported_group");
    assert_eq!(status["workers"][0]["observed_groups"][0]["group_idx"], 1);
    assert_eq!(status["workers"][0]["resident_blocks"], 0);
    assert_eq!(status["workers"][0]["event_state_fresh"], false);
    let unmatched = Arc::new(
        BasicWorkerBuilder::new("http://unmatched")
            .model(ModelCard::new("other"))
            .build(),
    ) as Arc<dyn Worker>;
    runtime.warn_missing_subscription(&unmatched);
    assert_eq!(
        runtime.status()["diagnostics_by_scope"]["http://unmatched"],
        json!(["profile_model_mismatch"])
    );
}

#[tokio::test]
async fn runtime_weight_updates_change_next_selection_without_resetting_cache() {
    // Reuse one live runtime: tuning must move traffic without rebuilding the
    // index, retokenizing a request, or discarding backend load observations.
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let runtime = local_runtime(config(
        "",
        vec![
            source("http://w1", "tcp://w1:1"),
            source("http://w2", "tcp://w2:1"),
        ],
    ))
    .await;
    for worker in &workers {
        seed(&runtime, worker);
    }
    runtime.subscriptions.read()[workers[1].url()]
        .state
        .lock()
        .stream
        .index
        .clear();
    set_load_snapshot(&runtime, &workers[0], 5.0);
    set_load_snapshot(&runtime, &workers[1], 0.0);
    let cached_subscription = runtime.subscriptions.read()[workers[0].url()].clone();
    let event_time = cached_subscription.state.lock().stream.last_event;
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    for (kv_importance, load_importance, expected) in [(1.0, 1.0, 1), (2.0, 1.0, 0), (2.0, 2.0, 1)]
    {
        runtime
            .update_weights(
                "m",
                settings::Weights {
                    kv_importance,
                    load_importance,
                },
            )
            .unwrap();
        assert_eq!(
            runtime
                .select(
                    &workers,
                    &SelectWorkerInfo {
                        kv_request: Some(&request),
                        ..Default::default()
                    },
                )
                .await,
            Some(expected),
        );
        assert_eq!(
            request.selected_cached_tokens(),
            Some(if expected == 0 { 4 } else { 0 }),
        );
        request.finish_attempt(false);
        assert!(Arc::ptr_eq(
            &cached_subscription,
            &runtime.subscriptions.read()[workers[0].url()],
        ));
        let state = cached_subscription.state.lock();
        assert_eq!(state.stream.index.len(), 2);
        assert_eq!(state.stream.last_event, event_time);
    }
    assert_eq!(runtime.config.profiles["m"].cost.kv_importance, 1.0);
    let tokenizer = runtime.tokenizer_registry.get("m").unwrap();
    assert_eq!(
        tokenizer
            .as_any()
            .downcast_ref::<CountingTokenizer>()
            .unwrap()
            .encode_calls
            .load(Ordering::Relaxed),
        1,
    );
}

#[tokio::test]
async fn higher_importance_favors_cache_savings_or_load_avoidance() {
    // Raising KV importance favors the cached worker. Holding KV at 2 and
    // raising load importance from 1 to 2 must then favor the idle worker.
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    for (kv_importance, load_importance, expected) in [
        (1.0, 1.0, 1),
        (2.0, 1.0, 0),
        (2.0, 2.0, 1),
        (1.0, 0.5, 0),
        (1.0, 0.0, 0),
        (0.0, 1.0, 1),
    ] {
        let mut cfg = config(
            "",
            vec![
                source("http://w1", "tcp://w1:1"),
                source("http://w2", "tcp://w2:1"),
            ],
        );
        let cost = &mut cfg.profiles.get_mut("m").unwrap().cost;
        cost.kv_importance = kv_importance;
        cost.load_importance = load_importance;
        let runtime = local_runtime(cfg).await;
        for worker in &workers {
            seed(&runtime, worker);
        }
        runtime.subscriptions.read()[workers[1].url()]
            .state
            .lock()
            .stream
            .index
            .clear();
        set_load_snapshot(&runtime, &workers[0], 5.0);
        set_load_snapshot(&runtime, &workers[1], 0.0);
        let request = KvRequest::new(
            "/v1/chat/completions/render",
            json!({"model":"m","messages":[]}),
        );
        let selected = runtime
            .select(
                &workers,
                &SelectWorkerInfo {
                    kv_request: Some(&request),
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(
            selected,
            Some(expected),
            "kv_importance={kv_importance}, load_importance={load_importance}"
        );
        assert_eq!(
            request.selected_cached_tokens(),
            Some(if expected == 0 { 4 } else { 0 })
        );
        request.finish_attempt(false);
    }
}

#[tokio::test]
async fn missing_subscription_preserves_model_weights_and_unknown_kv_occupancy() {
    // A newly registered model can be selected before event reconciliation.
    // Its configured weights and fallback size must still apply during that gap.
    let mut cfg = config("", vec![]);
    cfg.discovery = Some(Discovery::default());
    cfg.log_decisions = true;
    let cost = &mut cfg.profiles.get_mut("m").unwrap().cost;
    cost.kv_importance = 2.0;
    cost.load_importance = 0.0;
    cost.mean_uncached_tokens = 7;
    cost.prefill_ms_per_token = 3.0;
    let runtime = local_runtime(cfg).await;
    assert!(runtime.subscriptions.read().is_empty());
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    set_load_snapshot(&runtime, &workers[0], 5.0);
    runtime
        .loads
        .lock()
        .get_mut(workers[0].url())
        .unwrap()
        .snapshot
        .as_mut()
        .unwrap()
        .kv_usage = None;
    workers[1].increment_load();
    workers[1].increment_load();
    let request = KvRequest::new(
        "/v1/chat/completions/render",
        json!({"model":"m","messages":[]}),
    );
    let output = LogWriter::default();
    let writer = output.clone();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .finish();
    let selected = runtime
        .select(
            &workers,
            &SelectWorkerInfo {
                kv_request: Some(&request),
                ..Default::default()
            },
        )
        .with_subscriber(subscriber)
        .await
        .unwrap();
    assert_eq!(request.selected_cached_tokens(), None);
    let scores: Vec<_> = log_records(&output)
        .into_iter()
        .filter(|row| row["fields"]["message"] == "pre_kv_cache candidate score")
        .collect();
    assert_eq!(scores.len(), 2);
    for row in &scores {
        let fields = &row["fields"];
        let backend = fields["worker_url"] == workers[0].url();
        assert_eq!(fields["kv_importance"], 2.0);
        assert_eq!(fields["load_importance"], 0.0);
        assert_eq!(
            fields["selected"],
            fields["worker_url"] == workers[selected].url()
        );
        assert_eq!(fields["cache_state_valid"], false);
        assert_eq!(
            fields["load_source"],
            if backend { "backend" } else { "local_fallback" }
        );
        assert_eq!(
            fields["backend_or_fallback_load_ms"],
            if backend { 500.0 } else { 700.0 }
        );
        assert!(fields.get("kv_usage_ratio").is_none());
        let prefill = fields["uncached_prefill_ms"].as_f64().unwrap();
        assert!((21.0..21.1).contains(&prefill));
        assert_eq!(fields["weighted_kv_ms"], 2.0 * prefill);
        assert_eq!(fields["weighted_load_ms"], 0.0);
        assert_eq!(fields["estimated_cost_ms"], 2.0 * prefill);
    }
    request.finish_attempt(false);
}

#[tokio::test]
async fn decision_info_logs_show_raw_costs_reservations_and_weighted_fallback() {
    // A missing load sample inherits 300 raw ms, then applies load_importance=2 once.
    // Applying that weight twice would incorrectly select the uncached worker.
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let mut cfg = config(
        "",
        vec![
            source("http://w1", "tcp://w1:1"),
            source("http://w2", "tcp://w2:1"),
        ],
    );
    cfg.log_decisions = true;
    cfg.profiles.get_mut("m").unwrap().cost.load_importance = 2.0;
    let runtime = local_runtime(cfg).await;
    for worker in &workers {
        seed(&runtime, worker);
    }
    runtime.subscriptions.read()[workers[0].url()]
        .state
        .lock()
        .stream
        .index
        .clear();
    set_load_snapshot(&runtime, &workers[0], 3.0);
    let requests: Vec<_> = (0..2)
        .map(|_| {
            KvRequest::new(
                "/v1/chat/completions/render",
                json!({"model":"m","messages":[]}),
            )
        })
        .collect();
    let output = LogWriter::default();
    let writer = output.clone();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .finish();
    async {
        for request in &requests {
            assert_eq!(
                runtime
                    .select(
                        &workers,
                        &SelectWorkerInfo {
                            kv_request: Some(request),
                            ..Default::default()
                        }
                    )
                    .await,
                Some(1)
            );
        }
    }
    .with_subscriber(subscriber)
    .await;
    let scores: Vec<_> = log_records(&output)
        .into_iter()
        .filter(|row| row["fields"]["message"] == "pre_kv_cache candidate score")
        .collect();
    assert_eq!(scores.len(), 4);
    for row in &scores {
        assert_eq!(row["level"], "INFO");
        let fields = &row["fields"];
        let selected = fields["worker_url"] == "http://w2";
        assert_eq!(fields["selected"], selected);
        assert_eq!(fields["candidate_count"], 2);
        assert_eq!(fields["cache_state_valid"], true);
        assert_eq!(fields["cached_tokens"], if selected { 4 } else { 0 });
        assert_eq!(fields["kv_importance"], 1.0);
        assert_eq!(fields["load_importance"], 2.0);
        assert_eq!(
            fields["load_source"],
            if selected {
                "local_fallback"
            } else {
                "backend"
            }
        );
        assert_eq!(fields["backend_or_fallback_load_ms"], 300.0);
        let prefill = fields["uncached_prefill_ms"].as_f64().unwrap();
        let reservation = fields["reservation_ms"].as_f64().unwrap();
        let weighted_load = fields["weighted_load_ms"].as_f64().unwrap();
        assert_eq!(fields["weighted_kv_ms"], prefill);
        assert!((weighted_load - 2.0 * (300.0 + reservation)).abs() < 1e-9);
        assert!(
            (fields["estimated_cost_ms"].as_f64().unwrap() - prefill - weighted_load).abs() < 1e-9
        );
    }
    let selected_scores: Vec<_> = scores
        .iter()
        .filter(|row| row["fields"]["selected"] == true)
        .collect();
    assert_eq!(selected_scores[0]["fields"]["reservation_ms"], 0.0);
    assert_eq!(selected_scores[1]["fields"]["reservation_ms"], 100.0);
    assert_eq!(selected_scores[1]["fields"]["pending_tokens"], 1);
    for request in &requests {
        request.finish_attempt(false);
    }
}

#[tokio::test]
async fn default_decision_logging_is_quiet_and_singletons_skip_scoring() {
    for enabled in [false, true] {
        let mut cfg = config(
            "",
            vec![
                source("http://w1", "tcp://w1:1"),
                source("http://w2", "tcp://w2:1"),
            ],
        );
        if enabled {
            cfg.log_decisions = true;
        }
        let runtime = local_runtime(cfg).await;
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        for worker in &workers {
            seed(&runtime, worker);
            set_load_snapshot(&runtime, worker, 0.0);
        }
        let request = KvRequest::new(
            "/v1/chat/completions/render",
            json!({"model":"m","messages":[]}),
        );
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(workers[0].clone());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::PreKvCache));
        policies.install_pre_kv_cache(runtime.clone());
        let scheduler = Scheduler::new(registry, policies);
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .finish();
        async {
            runtime
                .select(
                    &workers,
                    &SelectWorkerInfo {
                        kv_request: Some(&request),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            request.finish_attempt(false);
            scheduler
                .schedule(&SchedulerInput {
                    kv_request: Some(&request),
                    model_id: Some("m"),
                    endpoint: Endpoint::Chat,
                    request_text: None,
                    token_ids: None,
                    headers: None,
                })
                .await
                .unwrap();
        }
        .with_subscriber(subscriber)
        .await;
        let records = log_records(&output);
        let scores = records
            .iter()
            .filter(|row| row["fields"]["message"] == "pre_kv_cache candidate score")
            .count();
        let bypasses: Vec<_> = records
            .iter()
            .filter(|row| row["fields"]["message"] == "pre_kv_cache scoring bypassed")
            .collect();
        assert_eq!(scores, if enabled { 2 } else { 0 });
        assert_eq!(bypasses.len(), usize::from(enabled));
        if enabled {
            assert_eq!(bypasses[0]["level"], "INFO");
            assert_eq!(bypasses[0]["fields"]["reason"], "single_candidate");
            assert_eq!(bypasses[0]["fields"]["candidate_count"], 1);
        }
        assert_eq!(request.selected_cached_tokens(), Some(4));
        assert!(request.reservations.lock().is_empty());
        assert!(runtime
            .loads
            .lock()
            .values()
            .all(|load| load.credits.is_empty()));
    }
}

#[tokio::test]
async fn missing_profile_and_tokenizer_warn_once_per_scope_across_requests() {
    // Both failures must be visible at ordinary INFO verbosity without repeating
    // the same warning for every request or attempting HTTP render.
    for mismatch in [false, true] {
        let mut cfg = config(
            "",
            vec![
                source("http://w1", "tcp://w1:1"),
                source("http://w2", "tcp://w2:1"),
            ],
        );
        cfg.profiles.get_mut("m").unwrap().token_source = TokenSource::LocalTokenizer;
        if mismatch {
            cfg.workers.clear();
            cfg.discovery = Some(Discovery::default());
            cfg.profiles.get_mut("m").unwrap().model = "other-model".into();
        }
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
        let registry = Arc::new(WorkerRegistry::new());
        for worker in &workers {
            registry.register(worker.clone());
            set_load_snapshot(&runtime, worker, 0.0);
        }
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .finish();
        async {
            for _ in 0..3 {
                runtime.reconcile(&registry);
                let request = KvRequest::new(
                    "/v1/chat/completions/render",
                    json!({"model":"m","messages":[]}),
                );
                assert!(runtime
                    .select(
                        &workers,
                        &SelectWorkerInfo {
                            kv_request: Some(&request),
                            ..Default::default()
                        }
                    )
                    .await
                    .is_some());
                request.finish_attempt(false);
            }
        }
        .with_subscriber(subscriber)
        .await;
        let reason = if mismatch {
            "profile_model_mismatch"
        } else {
            "local_tokenizer_unavailable"
        };
        let warnings: Vec<_> = log_records(&output)
            .into_iter()
            .filter(|row| row["fields"]["reason"] == reason)
            .collect();
        assert_eq!(warnings.len(), if mismatch { 2 } else { 1 });
        assert!(warnings.iter().all(|row| row["level"] == "WARN"));
        if mismatch {
            assert!(runtime.subscriptions.read().is_empty());
        }
    }
}

#[tokio::test]
async fn missing_load_does_not_attract_traffic_as_the_idlest_peer() {
    // Real users may bypass the Router. Losing that backend's metrics must
    // not turn its external queue into a measured zero or minimum peer load.
    let runtime = local_runtime(config(
        "",
        vec![
            source("http://idle", "tcp://idle:1"),
            source("http://busy", "tcp://busy:1"),
            source("http://unknown", "tcp://unknown:1"),
        ],
    ))
    .await;
    let workers: Vec<_> = ["http://idle", "http://busy", "http://unknown"]
        .into_iter()
        .map(|url| worker(url, WorkerType::Regular))
        .collect();
    for worker in &workers {
        seed(&runtime, worker);
    }
    // Unknown has a cache advantage, but its missing queue sample is not free.
    runtime.subscriptions.read()[workers[0].url()]
        .state
        .lock()
        .stream
        .index
        .clear();
    set_load_snapshot(&runtime, &workers[0], 1.0);
    set_load_snapshot(&runtime, &workers[1], 10.0);
    let request = KvRequest::new(
        "/v1/completions/render",
        json!({"model":"m","prompt":[1,2,3,4,5]}),
    );
    let info = SelectWorkerInfo {
        kv_request: Some(&request),
        ..Default::default()
    };
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    request.finish_attempt(false);
    // Once fresh idle telemetry arrives, the cache advantage is usable again.
    set_load_snapshot(&runtime, &workers[2], 0.0);
    assert_eq!(runtime.select(&workers, &info).await, Some(2));
    request.finish_attempt(false);
    // Retain the last high sample as an estimate if it expires; it is never
    // presented as a fresh measurement or allowed to override healthy peers.
    set_load_snapshot(&runtime, &workers[2], 20.0);
    runtime
        .loads
        .lock()
        .get_mut(workers[2].url())
        .unwrap()
        .snapshot
        .as_mut()
        .unwrap()
        .observed -= Duration::from_millis(runtime.config.load_stale_ms + 1);
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    request.finish_attempt(false);
}

#[tokio::test]
async fn scheduler_info_logs_preserve_idle_hits_and_distinguish_invalid_retries() {
    // A retry must replace old coverage, while a valid empty index is a real
    // zero. INFO logging must reuse the one local tokenization across retries.
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _recorder_guard = metrics::set_default_local_recorder(&recorder);
    let tokenizer = Arc::new(CountingTokenizer::default());
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load("id", "m", "m", || {
            let tokenizer = tokenizer.clone();
            async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
        })
        .await
        .unwrap();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let mut cfg = config(
        "",
        vec![
            source("http://w1", "tcp://w1:1"),
            source("http://w2", "tcp://w2:1"),
        ],
    );
    cfg.profiles.get_mut("m").unwrap().token_source = TokenSource::LocalTokenizer;
    let runtime =
        Runtime::new_with_tokenizer_registry(cfg, reqwest::Client::new(), tokenizer_registry)
            .unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    for worker in &workers {
        registry.register(worker.clone());
        seed(&runtime, worker);
    }
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::PreKvCache));
    policies.install_pre_kv_cache(runtime.clone());
    let scheduler = Scheduler::new(registry, policies.clone());
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
    let output = LogWriter::default();
    let writer = output.clone();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .finish();

    async {
        for (step, expected) in [Some(4), Some(4), None, Some(0)].into_iter().enumerate() {
            for subscription in runtime.subscriptions.read().values() {
                let mut state = subscription.state.lock();
                if step == 1 {
                    state.stream.last_event = Some(Instant::now() - Duration::from_secs(60));
                } else if expected.is_none() {
                    state.stream.invalidate();
                } else if expected == Some(0) {
                    state.stream.index.clear();
                    state.stream.last_event = Some(Instant::now());
                }
            }
            scheduler.schedule(&input).await.unwrap();
            assert_eq!(request.selected_cached_tokens(), expected);
            request.finish_attempt(false);
        }

        // Scaling down to one worker still reports coverage from the existing
        // tokens without running load scoring or creating a reservation.
        let singleton = Arc::new(WorkerRegistry::new());
        singleton.register(workers[0].clone());
        seed(&runtime, &workers[0]);
        Scheduler::new(singleton, policies)
            .schedule(&input)
            .await
            .unwrap();
        assert_eq!(request.selected_cached_tokens(), Some(4));
        assert!(request.reservations.lock().is_empty());
        assert!(runtime
            .loads
            .lock()
            .values()
            .all(|load| load.credits.is_empty()));

        // The logging helper alone cannot initialize a fresh tokenization.
        let fresh = KvRequest::new(
            "/v1/chat/completions/render",
            json!({"model":"m","messages":[]}),
        );
        runtime.record_single_worker_cache(&workers[0], &fresh);
        assert_eq!(fresh.selected_cached_tokens(), None);
    }
    .with_subscriber(subscriber)
    .await;

    let logs = String::from_utf8(output.0.lock().clone()).unwrap();
    let scheduled: Vec<Value> = logs
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|entry| entry["fields"]["message"] == "Request scheduled to worker")
        .collect();
    assert_eq!(scheduled.len(), 5);
    for (entry, ratio) in scheduled
        .iter()
        .zip(["0.8000", "0.8000", "unknown", "0.0000", "0.8000"])
    {
        assert_eq!(entry["level"], "INFO");
        assert_eq!(entry["fields"]["context_tokens"], "5");
        assert_eq!(entry["fields"]["cache_hit_ratio"], ratio);
        assert!(entry["fields"].get("input_tokens").is_none());
    }
    assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
    let metrics = handle.render();
    assert!(
        metrics
            .lines()
            .any(|line| line == "pre_kv_cache_selected_cached_tokens_count 3"),
        "{metrics}"
    );
    assert!(
        metrics
            .lines()
            .any(|line| line == "pre_kv_cache_selected_cached_tokens_sum 8"),
        "{metrics}"
    );
    assert!(metrics.contains("pre_kv_cache_selected_cache_state_total{state=\"known\"} 3"));
    assert!(metrics.contains("pre_kv_cache_selected_cache_state_total{state=\"unknown\"} 1"));
}

#[tokio::test]
async fn idle_residency_remains_usable_but_invalidation_and_replacement_do_not() {
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
    assert_eq!(runtime.select(&workers, &info).await, Some(0));
    assert_eq!(request.selected_cached_tokens(), Some(4));
    request.finish_attempt(false);
    let status = runtime.status();
    assert_eq!(status["workers"][0]["event_state_fresh"], false);
    assert_eq!(status["workers"][0]["event_state_usable"], true);
    runtime.subscriptions.read()["http://w1"]
        .state
        .lock()
        .stream
        .invalidate();
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
    request.finish_attempt(false);
    seed(&runtime, &workers[0]);
    workers[0] = worker("http://w1", WorkerType::Regular);
    workers[0].increment_load();
    assert_eq!(runtime.select(&workers, &info).await, Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incompatible_publisher_recovers_on_same_address_reconnect() {
    // Use real loopback sockets: compatibility and old residency belong to one
    // connection epoch, even if a replacement publisher keeps its address.
    let render = mock("render").await;
    let (context, publisher, endpoint) = publisher();
    let cfg = config(&render.url, vec![source(&render.url, &endpoint)]);
    let runtime = Runtime::new(cfg, reqwest::Client::new()).unwrap();
    let registry = Arc::new(WorkerRegistry::new());
    registry.register(worker(&render.url, WorkerType::Regular));
    runtime.start(&registry);
    let subscription = runtime.subscriptions.read()[&render.url].clone();
    wait_until(|| publisher.recv_bytes(zmq::DONTWAIT).is_ok()).await;
    publish(&publisher, 100, two_blocks());
    wait_until(|| subscription.state.lock().stream.index.len() == 2).await;
    publish(
        &publisher,
        101,
        rmpv::Value::Array(vec!["FutureRemoval".into()]),
    );
    wait_until(|| subscription.state.lock().stream.unsupported).await;
    publish(&publisher, 102, two_blocks());
    wait_until(|| subscription.state.lock().progress.last_received_sequence == Some(102)).await;
    assert!(subscription.state.lock().stream.unsupported);
    let generation = subscription.state.lock().progress.connection_generation;
    drop(publisher);
    wait_until(|| subscription.state.lock().progress.connection_generation > generation).await;
    assert!(!subscription.state.lock().stream.unsupported);
    assert_eq!(subscription.state.lock().stream.index.len(), 0);
    let replacement = context.socket(zmq::XPUB).unwrap();
    replacement.set_linger(0).unwrap();
    replacement.bind(&endpoint).unwrap();
    wait_until(|| replacement.recv_bytes(zmq::DONTWAIT).is_ok()).await;
    publish(
        &replacement,
        104,
        events::tests::stored(9, rmpv::Value::Nil, &[9, 10]),
    );
    wait_until(|| subscription.state.lock().stream.last_seq == Some(104)).await;
    {
        let state = subscription.state.lock();
        assert!(state.event_state_usable());
        assert_eq!(
            state
                .stream
                .index
                .matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""),
            0
        );
        assert_eq!(
            state.stream.index.matched_tokens(&[9, 10, 11], 2, "m", ""),
            2
        );
    }
    assert!(!runtime
        .diagnostics
        .is_active(&render.url, "unsupported_event"));
    assert!(!runtime
        .diagnostics
        .is_active(&render.url, "unsupported_stream"));
    runtime.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publisher_disconnect_invalidates_residency_without_an_idle_ttl() {
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
    assert_eq!(replica.tasks.lock().len(), 3);
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
    assert_eq!(replica.tasks.lock().len(), 3);
    registry.remove(&added);
    runtime.reconcile(&registry);
    assert_eq!(runtime.subscriptions.read().len(), 1);
    assert!(replica.stop.load(Ordering::Acquire));
    runtime.stop();
}
