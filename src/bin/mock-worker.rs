//! Standalone HTTP worker for local gateway routing tests.
//!
//! This process does not run a model. It implements the small HTTP surface
//! needed by the gateway, and makes the selected mode/role visible in every
//! generation response.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::Event, sse::KeepAlive, IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
};
use clap::{Parser, ValueEnum};
use futures_util::stream;
use rand::Rng;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use uuid::Uuid;

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Regular,
    Pd,
}

#[derive(Debug, Clone, ValueEnum)]
enum Role {
    Regular,
    Prefill,
    Decode,
}

#[derive(Debug, Clone, ValueEnum)]
enum HealthMode {
    Healthy,
    Unhealthy,
    Flapping,
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "mock-worker",
    about = "Standalone HTTP mock worker for gateway debugging"
)]
struct Args {
    /// Run one regular worker or a paired prefill/decode worker.
    #[arg(long, value_enum, default_value_t = Mode::Regular)]
    mode: Mode,

    /// Role when running a single endpoint. Normally inferred from --mode.
    #[arg(long, value_enum)]
    role: Option<Role>,

    #[arg(long, default_value = "deepseek-v4-flash")]
    model_id: String,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port for regular mode or single-role mode.
    #[arg(long, default_value_t = 18001)]
    port: u16,

    /// Prefill endpoint port in --mode pd.
    #[arg(long, default_value_t = 18002)]
    prefill_port: u16,

    /// Decode endpoint port in --mode pd.
    #[arg(long, default_value_t = 18003)]
    decode_port: u16,

    /// Advertised bootstrap port for the prefill worker.
    #[arg(long, default_value_t = 19002)]
    bootstrap_port: u16,

    #[arg(long, value_enum, default_value_t = HealthMode::Healthy)]
    health: HealthMode,

    /// Random request failure probability, from 0.0 to 1.0.
    #[arg(long, default_value_t = 0.0)]
    fail_rate: f32,

    #[arg(long, default_value_t = 0)]
    delay_ms: u64,

    /// Text returned by generation endpoints. Defaults to a role-specific value.
    #[arg(long)]
    response_text: Option<String>,
}

#[derive(Debug, Clone)]
struct WorkerState {
    model_id: String,
    mode: &'static str,
    role: &'static str,
    port: u16,
    bootstrap_port: Option<u16>,
    health: HealthMode,
    fail_rate: f32,
    delay_ms: u64,
    response_text: String,
}

impl WorkerState {
    fn worker_id(&self) -> String {
        format!("mock-{}-{}", self.role, self.port)
    }

    fn should_fail(&self) -> bool {
        self.fail_rate > 0.0 && rand::rng().random::<f32>() < self.fail_rate
    }

    fn is_healthy(&self) -> bool {
        match self.health {
            HealthMode::Healthy => true,
            HealthMode::Unhealthy => false,
            HealthMode::Flapping => self.port % 2 == 0,
        }
    }
}

fn app(state: WorkerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/server_info", get(server_info))
        .route("/get_server_info", get(server_info))
        .route("/model_info", get(model_info))
        .route("/get_model_info", get(model_info))
        .route("/v1/models", get(models))
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/flush_cache", post(flush_cache))
        .with_state(Arc::new(state))
}

async fn health(State(state): State<Arc<WorkerState>>) -> Response {
    if state.is_healthy() {
        Json(json!({
            "status": "healthy",
            "model_id": state.model_id,
            "mock_mode": state.mode,
            "mock_role": state.role,
            "port": state.port,
        }))
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unhealthy", "mock_role": state.role})),
        )
            .into_response()
    }
}

async fn health_generate(State(state): State<Arc<WorkerState>>) -> Response {
    if state.is_healthy() {
        Json(json!({"status": "ok", "mock_worker_id": state.worker_id()})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable", "mock_worker_id": state.worker_id()})),
        )
            .into_response()
    }
}

async fn server_info(State(state): State<Arc<WorkerState>>) -> Response {
    Json(json!({
        "model_path": state.model_id,
        "tokenizer_path": "mock-tokenizer",
        "version": "mock-worker-0.1.0",
        "tp_size": 1,
        "dp_size": 1,
        "mock_mode": state.mode,
        "mock_role": state.role,
        "port": state.port,
        "bootstrap_port": state.bootstrap_port,
    }))
    .into_response()
}

async fn model_info(State(state): State<Arc<WorkerState>>) -> Response {
    Json(json!({
        "model_path": state.model_id,
        "tokenizer_path": "mock-tokenizer",
        "is_generation": true,
    }))
    .into_response()
}

async fn models(State(state): State<Arc<WorkerState>>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "mock-worker"
        }]
    }))
    .into_response()
}

fn failure(state: &WorkerState) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        [("x-worker-id", state.worker_id())],
        Json(json!({
            "error": {
                "message": "mock failure",
                "type": "mock_error",
                "mock_mode": state.mode,
                "mock_role": state.role
            }
        })),
    )
        .into_response()
}

fn is_streaming(payload: &Value) -> bool {
    payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

async fn generate(State(state): State<Arc<WorkerState>>, Json(payload): Json<Value>) -> Response {
    if !state.is_healthy() || state.should_fail() {
        return failure(&state);
    }
    if state.delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(state.delay_ms)).await;
    }
    if is_streaming(&payload) {
        let body = json!({
            "text": state.response_text,
            "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop", "reason": "eos"}},
            "stage": "mid",
            "mock_mode": state.mode,
            "mock_role": state.role,
        });
        let events = stream::iter(vec![
            Ok::<Event, std::io::Error>(Event::default().data(body.to_string())),
            Ok(Event::default().data("[DONE]")),
        ]);
        return (
            [("x-worker-id", state.worker_id())],
            Sse::new(events).keep_alive(KeepAlive::default()),
        )
            .into_response();
    }
    (
        [("x-worker-id", state.worker_id())],
        Json(json!({
            "text": state.response_text,
            "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop", "reason": "eos"}},
            "mock_mode": state.mode,
            "mock_role": state.role,
            "mock_worker_id": state.worker_id(),
        })),
    )
        .into_response()
}

async fn chat_completions(
    State(state): State<Arc<WorkerState>>,
    Json(payload): Json<Value>,
) -> Response {
    if !state.is_healthy() || state.should_fail() {
        return failure(&state);
    }
    if state.delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(state.delay_ms)).await;
    }
    let id = format!("chatcmpl-mock-{}", Uuid::new_v4());
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_id);
    if is_streaming(&payload) {
        let chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": state.response_text}, "finish_reason": null}],
            "mock_mode": state.mode,
            "mock_role": state.role,
        });
        let events = stream::iter(vec![
            Ok::<Event, std::io::Error>(Event::default().data(chunk.to_string())),
            Ok(Event::default().data("[DONE]")),
        ]);
        return (
            [("x-worker-id", state.worker_id())],
            Sse::new(events).keep_alive(KeepAlive::default()),
        )
            .into_response();
    }
    let response = json!({
        "id": id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": state.response_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "mock_mode": state.mode,
        "mock_role": state.role,
        "mock_worker_id": state.worker_id(),
    });
    Json(response).into_response()
}

async fn completions(
    State(state): State<Arc<WorkerState>>,
    Json(payload): Json<Value>,
) -> Response {
    generate(State(state), Json(payload)).await
}

async fn flush_cache(State(state): State<Arc<WorkerState>>) -> Response {
    Json(json!({"success": true, "mock_worker_id": state.worker_id()})).into_response()
}

async fn serve(state: WorkerState, host: String) -> anyhow::Result<()> {
    let address: SocketAddr = format!("{}:{}", host, state.port).parse()?;
    let listener = TcpListener::bind(address).await?;
    println!(
        "mock-worker listening on http://{} (mode={}, role={}, model_id={})",
        address, state.mode, state.role, state.model_id
    );
    axum::serve(listener, app(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn build_state(
    args: &Args,
    mode: &'static str,
    role: &'static str,
    port: u16,
    bootstrap: Option<u16>,
) -> WorkerState {
    let default_text = format!("mock response from {}-{}", mode, role);
    WorkerState {
        model_id: args.model_id.clone(),
        mode,
        role,
        port,
        bootstrap_port: bootstrap,
        health: args.health.clone(),
        fail_rate: args.fail_rate.clamp(0.0, 1.0),
        delay_ms: args.delay_ms,
        response_text: args.response_text.clone().unwrap_or(default_text),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.mode {
        Mode::Regular => {
            let role = match args.role {
                Some(Role::Prefill) => "prefill",
                Some(Role::Decode) => "decode",
                Some(Role::Regular) | None => "regular",
            };
            serve(
                build_state(&args, "regular", role, args.port, None),
                args.host,
            )
            .await
        }
        Mode::Pd => {
            if let Some(ref role) = args.role {
                let (role_name, port, bootstrap) = match role {
                    Role::Prefill => ("prefill", args.port, Some(args.bootstrap_port)),
                    Role::Decode => ("decode", args.port, None),
                    Role::Regular => anyhow::bail!("--role regular is invalid with --mode pd"),
                };
                return serve(
                    build_state(&args, "pd", role_name, port, bootstrap),
                    args.host,
                )
                .await;
            }

            let prefill = build_state(
                &args,
                "pd",
                "prefill",
                args.prefill_port,
                Some(args.bootstrap_port),
            );
            let decode = build_state(&args, "pd", "decode", args.decode_port, None);
            let host = args.host.clone();
            tokio::try_join!(serve(prefill, host.clone()), serve(decode, host))?;
            Ok(())
        }
    }
}
