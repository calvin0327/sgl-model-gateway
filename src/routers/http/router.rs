use std::{sync::Arc, time::Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{stream, StreamExt};
use reqwest::Client;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error};

use crate::{
    app_context::AppContext,
    core::{
        scheduler::WorkerSelection as ScheduledWorkerSelection, AttachedBody, ConnectionMode,
        Worker, WorkerLoadGuard, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    protocol_compat::ResponsesGetParams,
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::GenerationRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        messages::CreateMessageRequest,
        rerank::{RerankRequest, RerankResponse, RerankResult},
        responses::ResponsesRequest,
    },
    routers::{
        current_scheduled_selection, error, header_utils,
        http::{error_type_from_status, route_to_endpoint},
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
};

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    client: Client,
    dp_aware: bool,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("client", &self.client)
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

const SGLANG_DEFAULT_FIELDS: &[&str] = &[
    "request_id",
    "priority",
    "top_k",
    "min_p",
    "min_tokens",
    "regex",
    "ebnf",
    "json_schema",
    "stop_token_ids",
    "no_stop_trim",
    "ignore_eos",
    "continue_final_message",
    "skip_special_tokens",
    "lora_path",
    "session_params",
    "separate_reasoning",
    "stream_reasoning",
    "chat_template",
    "chat_template_kwargs",
    "return_hidden_states",
    "repetition_penalty",
    "sampling_seed",
    "backend_url",
];

pub(crate) fn strip_default_sglang_fields(payload: &mut serde_json::Value) {
    let Some(object) = payload.as_object_mut() else {
        return;
    };

    for field in SGLANG_DEFAULT_FIELDS {
        let remove = object.get(*field).is_some_and(|value| {
            value.is_null()
                || value == &serde_json::Value::Bool(false)
                || (matches!(*field, "separate_reasoning" | "stream_reasoning")
                    && value == &serde_json::Value::Bool(true))
        });
        if remove {
            object.remove(*field);
        }
    }
}

fn is_pre_response_transport_error(error: &reqwest::Error) -> bool {
    error.status().is_none()
        && !error.is_timeout()
        && !error.is_body()
        && !error.is_decode()
        && !error.is_builder()
        && !error.is_redirect()
}

pub(crate) async fn send_with_stale_conn_retry(
    builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, reqwest::Error> {
    let retry = builder.try_clone();
    match builder.send().await {
        Err(error) if is_pre_response_transport_error(&error) => match retry {
            Some(retry) => retry.send().await,
            None => Err(error),
        },
        other => other,
    }
}

impl Router {
    fn scheduled_regular_worker() -> Result<Arc<dyn Worker>, Response> {
        let Some(selection) = current_scheduled_selection() else {
            return Err(error::service_unavailable(
                "scheduler_selection_required",
                "Request must be scheduled before entering the HTTP router",
            ));
        };

        match selection.worker_selection {
            ScheduledWorkerSelection::Regular { worker } => Ok(worker),
            _ => Err(error::bad_request(
                "invalid_worker_selection",
                "Expected a Regular worker",
            )),
        }
    }

    fn serialize_vllm_request<T: serde::Serialize>(
        typed_req: &T,
    ) -> Result<serde_json::Value, Response> {
        let mut json_value = serde_json::to_value(typed_req).map_err(|error| {
            error::bad_request(
                "serialization_failed",
                format!("Convert into serde_json::Value failed: {}", error),
            )
        })?;
        strip_default_sglang_fields(&mut json_value);
        Ok(json_value)
    }

    /// Create a regular HTTP execution router.
    pub fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            client: ctx.client.clone(),
            dp_aware: ctx.config.routing.dp_aware,
        })
    }

    fn management_worker_url(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,
            true,
        );

        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.management_worker_url() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    async fn route_detached_typed_request_with_worker<
        T: GenerationRequest + serde::Serialize + Clone,
    >(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        worker: Arc<dyn Worker>,
    ) -> Response {
        self.route_typed_request_internal_with_detach(
            headers, typed_req, route, model_id, worker, true,
        )
        .await
    }

    async fn route_typed_request_internal<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        selected_worker: Arc<dyn Worker>,
    ) -> Response {
        self.route_typed_request_internal_with_detach(
            headers,
            typed_req,
            route,
            model_id,
            selected_worker,
            false,
        )
        .await
    }

    async fn route_typed_request_internal_with_detach<
        T: GenerationRequest + serde::Serialize + Clone,
    >(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        selected_worker: Arc<dyn Worker>,
        detach_stream: bool,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        // Worker selection and retry ownership belong to RouterManager. This
        // router executes exactly the worker selected for this attempt.
        let response = self
            .route_typed_request_once(
                headers,
                typed_req,
                route,
                model_id,
                is_stream,
                selected_worker,
                detach_stream,
            )
            .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_typed_request_once<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        _model_id: Option<&str>,
        is_stream: bool,
        worker: Arc<dyn Worker>,
        detach_stream: bool,
    ) -> Response {
        // Load is execution state, not a routing decision. Keep it attached for
        // every scheduled worker so all Scheduler policies observe real load.
        let load_guard = Some(WorkerLoadGuard::new(worker.clone(), headers));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();

        let mut headers_with_trace = headers.cloned().unwrap_or_default();

        inject_trace_context_http(&mut headers_with_trace);

        let headers = Some(&headers_with_trace);

        let response = self
            .send_typed_request(
                headers,
                typed_req,
                route,
                &worker,
                is_stream,
                detach_stream,
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        // Keep circuit-breaker accounting separate from routing selection.
        // Scheduler uses HealthCheck state for candidate filtering.
        if !is_stream {
            worker.record_outcome(status.is_success());
        }

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    // Helper: return base worker URL (strips DP suffix when enabled)
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.dp_aware {
            if let Ok((prefix, _)) = Self::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| header_utils::should_forward_request_header(name.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let worker_url = worker.url();
                let base = self.worker_base_url(worker_url);
                let url = format!("{}/{}", base, endpoint);
                let client = self.client.clone();
                let method = method.clone();

                let headers = filtered_headers.clone();

                let api_key = worker.api_key().clone();

                async move {
                    let mut request_builder = match method {
                        Method::GET => client.get(url),
                        Method::POST => client.post(url),
                        _ => {
                            return Err(error::method_not_allowed(
                                "unsupported_method",
                                "Unsupported method for simple routing",
                            ))
                        }
                    };

                    if let Some(key) = api_key {
                        let mut auth_header = String::with_capacity(7 + key.len());
                        auth_header.push_str("Bearer ");
                        auth_header.push_str(&key);
                        request_builder = request_builder.header("Authorization", auth_header);
                    }

                    for (name, value) in headers {
                        request_builder = request_builder.header(name.clone(), value.clone());
                    }

                    request_builder.send().await.map_err(convert_reqwest_error)
                }
            })
            .collect();

        // Now execute the collected futures concurrently
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut last_response: Option<Response> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;

                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(e);
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // TODO (rui): Better accommodate to the Worker abstraction
    fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
        let parts: Vec<&str> = worker_url.split('@').collect();
        if parts.len() != 2 {
            return Err(format!("invalid worker_url format: {}", worker_url));
        }

        // Parse the second part (dp_rank) into an integer
        match parts[1].parse::<usize>() {
            Ok(dp_rank) => Ok((parts[0], dp_rank)),
            Err(_) => Err(format!(
                "failed to parse dp_rank from worker_url: {}",
                worker_url
            )),
        }
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker: &Arc<dyn Worker>,
        is_stream: bool,
        detach_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        let worker_url = worker.url();
        let api_key = worker.api_key().clone();

        // Static key string to avoid per-request allocations
        const DP_RANK_KEY: &str = "data_parallel_rank";

        let is_vllm = matches!(
            worker.metadata().runtime_type,
            crate::core::worker::BackendType::Vllm
        );

        let mut request_builder = if self.dp_aware {
            let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return error::internal_error(
                        "dp_rank_extraction_failed",
                        format!("Failed to extract dp_rank: {}", e),
                    );
                }
            };

            let mut json_val = match serde_json::to_value(typed_req) {
                Ok(j) => j,
                Err(e) => {
                    return error::bad_request(
                        "serialization_failed",
                        format!("Convert into serde_json::Value failed: {}", e),
                    );
                }
            };

            if let Some(map) = json_val.as_object_mut() {
                // Use static key string to avoid allocation
                map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
                // Only serialize if debug logging is enabled to avoid CPU overhead
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!(
                        "Modified request body: {}",
                        serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
                    );
                }
            } else {
                return error::bad_request(
                    "dp_rank_insertion_failed",
                    "Failed to insert the data_parallel_rank field into the request body",
                );
            }

            if is_vllm {
                strip_default_sglang_fields(&mut json_val);
            }

            self.client
                .post(format!("{}{}", worker_url_prefix, route))
                .json(&json_val)
        } else if is_vllm {
            match Self::serialize_vllm_request(typed_req) {
                Ok(json_value) => self
                    .client
                    .post(format!("{}{}", worker_url, route))
                    .json(&json_value),
                Err(response) => return response,
            }
        } else {
            self.client
                .post(format!("{}{}", worker_url, route))
                .json(typed_req) // Use json() directly with typed request
        };

        if let Some(key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match send_with_stale_conn_retry(request_builder).await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                if is_stream {
                    worker.record_outcome(false);
                }

                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        } else {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            // Pass the reqwest byte stream straight through as the response body.
            // Dropping the response body drops this stream, which closes the
            // upstream HTTP connection and lets the engine abort generation —
            // no spawned task or channel needed.
            let mut tracked = BreakerTrackedStream::new(
                res.bytes_stream(),
                worker.clone(),
                worker_url.to_string(),
            );
            if !status.is_success() {
                tracked.mark_errored();
            }

            if detach_stream {
                let (tx, rx) = mpsc::channel(16);
                tokio::spawn(async move {
                    while let Some(chunk) = tracked.next().await {
                        if tx.send(chunk).await.is_err() {
                            while tracked.next().await.is_some() {}
                            break;
                        }
                    }
                    drop(load_guard);
                });

                let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
                *response.status_mut() = status;
                *response.headers_mut() = response_headers;
                response
            } else {
                let body = Body::from_stream(tracked);
                let mut response = Response::new(body);
                *response.status_mut() = status;
                *response.headers_mut() = response_headers;

                if let Some(guard) = load_guard {
                    response = AttachedBody::wrap_response(response, guard);
                }
                response
            }
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        self.route_typed_request_internal(headers, body, "/generate", model_id, worker.clone())
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        self.route_typed_request_internal(
            headers,
            body,
            "/v1/chat/completions",
            model_id,
            worker.clone(),
        )
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        self.route_typed_request_internal(
            headers,
            body,
            "/v1/completions",
            model_id,
            worker.clone(),
        )
        .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        body: &CreateMessageRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        self.route_typed_request_internal(headers, body, "/v1/messages", model_id, worker.clone())
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        if body.store == Some(true) && body.is_stream() {
            self.route_detached_typed_request_with_worker(
                headers,
                body,
                "/v1/responses",
                model_id,
                worker.clone(),
            )
            .await
        } else {
            self.route_typed_request_internal(
                headers,
                body,
                "/v1/responses",
                model_id,
                worker.clone(),
            )
            .await
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        self.route_typed_request_internal(headers, body, "/v1/embeddings", model_id, worker.clone())
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        self.route_typed_request_internal(headers, body, "/v1/classify", model_id, worker.clone())
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let worker = match Self::scheduled_regular_worker() {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        let response = self
            .route_typed_request_internal(headers, body, "/v1/rerank", model_id, worker.clone())
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(response) => response,
                Err(_) => error::internal_error(
                    "rerank_response_build_failed",
                    "Failed to build rerank response",
                ),
            }
        } else {
            response
        }
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        model_type::ModelType, worker::BackendType, BasicWorkerBuilder, ModelCard, WorkerType,
    };

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .build();
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        Router {
            worker_registry: worker_registry.clone(),
            dp_aware: false,
            client: Client::new(),
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_healthy(false);
        router
    }

    async fn spawn_recording_chat_worker() -> (String, Arc<std::sync::Mutex<Vec<serde_json::Value>>>)
    {
        use axum::{extract::State, routing::post, Json, Router as AxumRouter};
        use tokio::net::TcpListener;

        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let state = requests.clone();
        let app = AxumRouter::new()
            .route(
                "/v1/chat/completions",
                post(
                    |State(requests): State<Arc<std::sync::Mutex<Vec<serde_json::Value>>>>,
                     Json(body): Json<serde_json::Value>| async move {
                        requests
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(body);
                        Json(serde_json::json!({"object": "chat.completion"}))
                    },
                ),
            )
            .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{}", address), requests)
    }

    #[test]
    fn test_management_worker_selection_skips_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.management_worker_url();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }

    #[tokio::test]
    async fn regular_router_requires_scheduler_selection() {
        let router = create_test_regular_router();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let response = router.route_chat(None, &request, None).await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("scheduler_selection_required"));
    }

    #[test]
    fn strip_default_sglang_fields_removes_vllm_defaults() {
        let mut payload = serde_json::json!({
            "model": "test-model",
            "skip_special_tokens": false,
            "separate_reasoning": false,
            "stream_reasoning": false,
            "request_id": null,
            "top_p": 0.9,
        });

        strip_default_sglang_fields(&mut payload);

        assert!(payload.get("skip_special_tokens").is_none());
        assert!(payload.get("separate_reasoning").is_none());
        assert!(payload.get("stream_reasoning").is_none());
        assert!(payload.get("request_id").is_none());
        assert_eq!(payload["top_p"], 0.9);
    }

    #[tokio::test]
    async fn vllm_regular_requests_strip_sglang_defaults_while_sglang_preserves_them() {
        let (sglang_url, sglang_requests) = spawn_recording_chat_worker().await;
        let (vllm_url, vllm_requests) = spawn_recording_chat_worker().await;
        let router = create_test_regular_router();

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "skip_special_tokens": false,
            "separate_reasoning": false,
            "stream_reasoning": false,
        }))
        .unwrap();
        let sglang_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(sglang_url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("test-model").with_model_type(ModelType::SGLANG_LLM))
                .build(),
        );
        let vllm_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(vllm_url)
                .worker_type(WorkerType::Regular)
                .runtime_type(BackendType::Vllm)
                .model(ModelCard::new("test-model").with_model_type(ModelType::LLM))
                .build(),
        );

        let sglang_response = router
            .send_typed_request(
                None,
                &request,
                "/v1/chat/completions",
                &sglang_worker,
                false,
                false,
                None,
            )
            .await;
        let vllm_response = router
            .send_typed_request(
                None,
                &request,
                "/v1/chat/completions",
                &vllm_worker,
                false,
                false,
                None,
            )
            .await;

        assert_eq!(sglang_response.status(), StatusCode::OK);
        assert_eq!(vllm_response.status(), StatusCode::OK);

        let sglang_requests = sglang_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let vllm_requests = vllm_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(sglang_requests.len(), 1);
        assert_eq!(vllm_requests.len(), 1);
        assert_eq!(sglang_requests[0]["skip_special_tokens"], false);
        assert_eq!(sglang_requests[0]["separate_reasoning"], false);
        assert_eq!(sglang_requests[0]["stream_reasoning"], false);
        assert!(vllm_requests[0].get("skip_special_tokens").is_none());
        assert!(vllm_requests[0].get("separate_reasoning").is_none());
        assert!(vllm_requests[0].get("stream_reasoning").is_none());
    }
}
