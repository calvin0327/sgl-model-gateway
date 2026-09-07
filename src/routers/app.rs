use std::{sync::Arc, time::Duration};

use axum::{
    extract::{Path, Query, Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;
use validator::Validate;

use crate::{
    app_state::AppState,
    core::{worker_manager::WorkerManager, UNKNOWN_MODEL_ID},
    middleware::{self, AuthConfig},
    protocol_compat::{ResponsesGetParams, WorkerConfigRequest, WorkerUpdateRequest},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
        rerank::{RerankRequest, V1RerankReqInput},
        responses::ResponsesRequest,
        tokenize::{AddTokenizerRequest, DetokenizeRequest, TokenizeRequest},
        validated::{Normalizable, ValidatedJson},
    },
    routers::{parse, tokenize},
};

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([http::Method::GET, http::Method::POST, http::Method::OPTIONS])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> Response {
    let model_id = Some(body.model.as_str());
    state
        .router
        .route_generate(Some(&headers), &body, model_id)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    state
        .router
        .route_chat(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<RerankRequestInput>,
) -> Response {
    // V1RerankReqInput predates model-aware routing and only contains
    // `query` and `documents`. Keep that compatibility shape while accepting
    // an optional model so RouterManager can select the correct model pool.
    let model = body.model.clone();
    let mut rerank_body: RerankRequest = V1RerankReqInput {
        query: body.query,
        documents: body.documents,
    }
    .into();

    rerank_body.model = model.unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

    state
        .router
        .route_rerank(Some(&headers), &rerank_body, Some(&rerank_body.model))
        .await
}

/// Request body for the OpenAI-compatible `/v1/rerank` endpoint.
///
/// The upstream protocol's legacy `V1RerankReqInput` does not include a model
/// field, but model-aware routing requires preserving it at the gateway edge.
#[derive(Debug, Deserialize)]
struct RerankRequestInput {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    model: Option<String>,
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<ResponsesRequest>,
) -> Response {
    let store_was_specified = body.store.is_some();
    body.normalize();
    if let Err(errors) = body.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": errors.to_string(),
                    "type": "invalid_request_error",
                    "code": 400
                }
            })),
        )
            .into_response();
    }
    if !store_was_specified {
        body.store = Some(false);
    }

    state
        .router
        .route_responses(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    state
        .router
        .route_classify(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
    Query(params): Query<ResponsesGetParams>,
) -> Response {
    state
        .router
        .get_response(Some(&headers), &response_id, &params)
        .await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .delete_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .list_response_input_items(Some(&headers), &response_id)
        .await
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers = workers.iter().filter(|worker| worker.is_healthy()).count();
    let available_workers: Vec<_> = workers
        .iter()
        .filter(|worker| worker.is_available())
        .collect();

    // TODO:
    // Use the same route-level readiness semantics as RouterManager. The
    // fallback keeps lightweight tests and embedded users working when they
    // provide a RouterTrait directly instead of a RouterManager.
    let is_ready = state
        .router_manager
        .as_ref()
        .map(|manager| manager.has_available_route(None))
        .unwrap_or(!available_workers.is_empty());

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers,
                "available_workers": available_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "no complete available route"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    WorkerManager::get_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

// TODO: multi model
async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

// TODO: multi model
async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

// TODO: multi model
async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::get_all_worker_loads(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    match state.context.worker_service.create_worker(config).await {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    state.context.worker_service.list_workers().into_response()
}

async fn get_worker(State(state): State<Arc<AppState>>, Path(worker_id): Path<String>) -> Response {
    match state.context.worker_service.get_worker(&worker_id) {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Response {
    match state.context.worker_service.delete_worker(&worker_id).await {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn update_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
    Json(update): Json<WorkerUpdateRequest>,
) -> Response {
    match state
        .context
        .worker_service
        .update_worker(&worker_id, update)
        .await
    {
        Ok(result) => result.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn parse_function_call_http(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ParseFunctionCallRequest>,
) -> Response {
    parse::parse_function_call(&state.context, &request).await
}

async fn parse_reasoning_http(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SeparateReasoningRequest>,
) -> Response {
    parse::parse_reasoning(&state.context, &request).await
}

async fn v1_tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    tokenize::tokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    tokenize::detokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_tokenizers_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddTokenizerRequest>,
) -> Response {
    tokenize::add_tokenizer(&state.context, request).await
}

async fn v1_tokenizers_list(State(state): State<Arc<AppState>>) -> Response {
    tokenize::list_tokenizers(&state.context.tokenizer_registry).await
}

async fn v1_tokenizers_get(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_info(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_status(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_status(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_remove(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::remove_tokenizer(&state.context, &tokenizer_id).await
}

pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    control_plane_auth_state: Option<crate::auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Router {
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/classify", post(v1_classify))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input_items",
            get(v1_responses_list_input_items),
        )
        // Tokenize / Detokenize endpoints
        .route("/v1/tokenize", post(v1_tokenize))
        .route("/v1/detokenize", post(v1_detokenize))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/model_info", get(get_model_info))
        // TODO: Remove `/get_model_info` alias after one release-cycle deprecation window.
        .route("/get_model_info", get(get_model_info))
        .route("/server_info", get(get_server_info))
        // TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
        .route("/get_server_info", get(get_server_info));

    // Build admin routes with control plane auth if configured, otherwise use simple API key auth
    let admin_routes = Router::new()
        .route("/flush_cache", post(flush_cache))
        .route("/v1/loads", get(get_loads))
        // TODO: Remove `/get_loads` alias after one release-cycle deprecation window.
        .route("/get_loads", get(get_loads))
        .route("/parse/function_call", post(parse_function_call_http))
        .route("/parse/reasoning", post(parse_reasoning_http))
        // Tokenizer management endpoints
        .route(
            "/v1/tokenizers",
            post(v1_tokenizers_add).get(v1_tokenizers_list),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}",
            get(v1_tokenizers_get).delete(v1_tokenizers_remove),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}/status",
            get(v1_tokenizers_status),
        );

    // Build worker routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker).get(list_workers_rest))
        .route(
            "/workers/{worker_id}",
            get(get_worker).put(update_worker).delete(delete_worker),
        );

    // Apply authentication middleware to control plane routes
    let apply_control_plane_auth = |routes: Router<Arc<AppState>>| {
        if let Some(ref cp_state) = control_plane_auth_state {
            routes.route_layer(axum::middleware::from_fn_with_state(
                cp_state.clone(),
                crate::auth::control_plane_auth_middleware,
            ))
        } else {
            routes.route_layer(axum::middleware::from_fn_with_state(
                auth_config.clone(),
                middleware::auth_middleware,
            ))
        }
    };
    let admin_routes = apply_control_plane_auth(admin_routes);
    let worker_routes = apply_control_plane_auth(worker_routes);

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state)
}
