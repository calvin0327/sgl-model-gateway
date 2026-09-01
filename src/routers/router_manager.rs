use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use serde_json::Value;
use tracing::{debug, info};

use crate::{
    app_context::AppContext,
    config::GatewayConfig,
    core::{pd_group_id, worker::Worker, ConnectionMode, RuntimeType, WorkerRegistry, WorkerType},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(&'static str);

impl RouterId {
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        self.0
    }
}

/// Static router ID constants to avoid heap allocations in hot paths
pub mod router_ids {
    use super::RouterId;

    pub const HTTP_REGULAR: RouterId = RouterId::new("http-regular");
    pub const HTTP_PD: RouterId = RouterId::new("http-pd");
    pub const HTTP_OPENAI: RouterId = RouterId::new("http-openai");
    pub const GRPC_REGULAR: RouterId = RouterId::new("grpc-regular");
    pub const GRPC_PD: RouterId = RouterId::new("grpc-pd");
}

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    router_init_lock: Arc<tokio::sync::Mutex<()>>,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            routers_snapshot: ArcSwap::from_pointee(Vec::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            router_init_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub async fn from_config(
        _config: &GatewayConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        let manager = Arc::new(Self::new(app_context.worker_registry.clone()));

        info!("Initializing RouterManager with lazy router creation");

        Ok(manager)
    }

    fn router_id_for_worker(worker: &dyn Worker) -> Option<RouterId> {
        if matches!(worker.metadata().runtime_type, RuntimeType::External) {
            return matches!(worker.connection_mode(), ConnectionMode::Http)
                .then_some(router_ids::HTTP_OPENAI);
        }

        match (worker.worker_type(), worker.connection_mode()) {
            (WorkerType::Regular, ConnectionMode::Http) => Some(router_ids::HTTP_REGULAR),
            (WorkerType::Regular, ConnectionMode::Grpc { .. }) => Some(router_ids::GRPC_REGULAR),
            (WorkerType::Prefill { .. }, ConnectionMode::Http)
            | (WorkerType::Decode, ConnectionMode::Http) => Some(router_ids::HTTP_PD),
            (WorkerType::Prefill { .. }, ConnectionMode::Grpc { .. })
            | (WorkerType::Decode, ConnectionMode::Grpc { .. }) => Some(router_ids::GRPC_PD),
        }
    }

    /// Create the router required by a newly registered worker, if necessary.
    pub async fn ensure_router_for_worker(
        &self,
        app_context: &Arc<AppContext>,
        worker: Arc<dyn Worker>,
    ) -> Result<(), String> {
        let Some(router_id) = Self::router_id_for_worker(worker.as_ref()) else {
            return Ok(());
        };

        if self.routers.contains_key(&router_id) {
            return Ok(());
        }

        let _guard = self.router_init_lock.lock().await;

        if self.routers.contains_key(&router_id) {
            return Ok(());
        }

        use crate::routers::RouterFactory;
        let router = match router_id.as_str() {
            "http-regular" => RouterFactory::create_regular_router(app_context).await?,
            "grpc-regular" => RouterFactory::create_grpc_router(app_context).await?,
            "http-pd" => RouterFactory::create_pd_router(app_context).await?,
            "grpc-pd" => RouterFactory::create_grpc_pd_router(app_context).await?,
            "http-openai" => RouterFactory::create_openai_router(app_context).await?,
            _ => return Ok(()),
        };

        self.register_router(router_id.clone(), Arc::from(router));
        info!("Created router on demand: {}", router_id.as_str());

        Ok(())
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // Update the lock-free snapshot for fast per-request iteration
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Resolve model_id for a request, inferring from available workers if not specified.
    ///
    /// Behavior in IGW mode (must fail fast if model not resolvable):
    /// - If model_id is provided, use it directly
    /// - If not provided and only one model exists, use it as implicit default
    /// - If not provided and multiple models exist, return error requiring specification
    /// - If no models exist, return service unavailable error
    fn resolve_model_id(&self, model_id: Option<&str>) -> Result<String, Box<Response>> {
        // If model_id is provided, use it
        if let Some(id) = model_id {
            return Ok(id.to_string());
        }

        // Get all available models from worker registry
        let available_models = self.worker_registry.get_models();

        match available_models.len() {
            0 => Err(Box::new(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No models available - no workers registered",
                )
                    .into_response(),
            )),
            1 => {
                // Single model: use it as implicit default
                debug!(
                    "Model not specified, using implicit default: {}",
                    available_models[0]
                );
                Ok(available_models[0].clone())
            }
            _ => {
                // Multiple models: require explicit model specification
                Err(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Model must be specified. Available models: {}",
                            available_models.join(", ")
                        ),
                    )
                        .into_response(),
                ))
            }
        }
    }

    /// Return workers visible for a model, or all workers for an unresolved model.
    fn workers_for_model(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        match model_id {
            Some(model) => self.worker_registry.get_by_model(model).to_vec(),
            None => self.worker_registry.get_all(),
        }
    }

    /// Check whether a router has the healthy workers required to serve a request.
    ///
    /// This check deliberately includes connection mode. A HTTP router must never
    /// be selected merely because a matching gRPC worker exists, and vice versa.
    fn is_router_available(&self, router_id: &RouterId, model_id: Option<&str>) -> bool {
        let workers = self.workers_for_model(model_id);
        let available_workers = workers.iter().filter(|worker| worker.is_available());

        match router_id.as_str() {
            "http-regular" => available_workers.clone().any(|worker| {
                matches!(worker.worker_type(), WorkerType::Regular)
                    && matches!(worker.connection_mode(), ConnectionMode::Http)
                    && !matches!(worker.metadata().runtime_type, RuntimeType::External)
            }),
            "grpc-regular" => available_workers.clone().any(|worker| {
                matches!(worker.worker_type(), WorkerType::Regular)
                    && matches!(worker.connection_mode(), ConnectionMode::Grpc { .. })
                    && !matches!(worker.metadata().runtime_type, RuntimeType::External)
            }),
            "http-pd" => Self::has_available_pd_pair(&workers, &ConnectionMode::Http),
            "grpc-pd" => {
                Self::has_available_pd_pair(&workers, &ConnectionMode::Grpc { port: None })
            }
            "http-openai" => available_workers.clone().any(|worker| {
                matches!(worker.metadata().runtime_type, RuntimeType::External)
                    && matches!(worker.connection_mode(), ConnectionMode::Http)
            }),
            _ => false,
        }
    }

    /// Check whether one model has an available Prefill/Decode pair in the
    /// same connection mode and PD deployment group.
    fn has_available_pd_pair(workers: &[Arc<dyn Worker>], mode: &ConnectionMode) -> bool {
        let prefill_groups: Vec<&str> = workers
            .iter()
            .filter(|worker| {
                worker.is_available()
                    && matches!(worker.worker_type(), WorkerType::Prefill { .. })
                    && worker.connection_mode().matches(mode)
            })
            .map(|worker| pd_group_id(worker.as_ref()))
            .collect();

        workers.iter().any(|worker| {
            worker.is_available()
                && matches!(worker.worker_type(), WorkerType::Decode)
                && worker.connection_mode().matches(mode)
                && prefill_groups.contains(&pd_group_id(worker.as_ref()))
        })
    }

    /// Return whether at least one registered Router can currently serve the
    /// supplied model. With no model supplied, every registered model is
    /// checked independently so PD workers from different models cannot make
    /// the gateway appear ready as a false combined pair.
    pub fn has_available_route(&self, model_id: Option<&str>) -> bool {
        match model_id {
            Some(model) => Self::router_candidates(false).iter().any(|router_id| {
                self.routers.contains_key(router_id)
                    && self.is_router_available(router_id, Some(model))
            }),
            None => self.worker_registry.get_models().iter().any(|model| {
                Self::router_candidates(false).iter().any(|router_id| {
                    self.routers.contains_key(router_id)
                        && self.is_router_available(router_id, Some(model))
                })
            }),
        }
    }

    fn router_candidates(prefer_pd: bool) -> [RouterId; 5] {
        if prefer_pd {
            [
                router_ids::HTTP_PD,
                router_ids::GRPC_PD,
                router_ids::HTTP_REGULAR,
                router_ids::GRPC_REGULAR,
                router_ids::HTTP_OPENAI,
            ]
        } else {
            [
                router_ids::HTTP_REGULAR,
                router_ids::GRPC_REGULAR,
                router_ids::HTTP_PD,
                router_ids::GRPC_PD,
                router_ids::HTTP_OPENAI,
            ]
        }
    }

    pub fn select_router_for_request(
        &self,
        headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        let prefer_pd = headers
            .and_then(|h| {
                h.get("x-prefer-pd")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == "true" || s == "1")
            })
            .unwrap_or(false);

        Self::router_candidates(prefer_pd)
            .iter()
            .find(|router_id| self.is_router_available(router_id, model_id))
            .and_then(|router_id| self.routers.get(router_id).map(|router| router.clone()))
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // A gateway is healthy only when at least one complete route is
        // available. In particular, a lone PD Prefill or Decode worker is not
        // sufficient, while a healthy Regular route is sufficient even when
        // the PD route is incomplete.
        if self.has_available_route(None) {
            (StatusCode::OK, "At least one router has an available route").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No available route").into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
        (
            StatusCode::OK,
            serde_json::json!({
                "router_manager": true,
                "routers_count": self.routers.len(),
                "workers_count": self.worker_registry.get_all().len()
            })
            .to_string(),
        )
            .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let model_names = self.worker_registry.get_models();

        if model_names.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            // Convert model names to OpenAI-compatible model objects
            let models: Vec<Value> = model_names
                .iter()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "object": "model",
                        "owned_by": "local"
                    })
                })
                .collect();

            (
                StatusCode::OK,
                serde_json::json!({
                    "object": "list",
                    "data": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    // TODO: multi model
    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Route to default router or first available router
        let router_id = {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            default_router.clone()
        };

        let router = if let Some(id) = router_id {
            self.routers.get(&id).map(|r| r.clone())
        } else {
            // If no default, use first available router
            self.routers.iter().next().map(|r| r.value().clone())
        };

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let effective_model_id = match self.resolve_model_id(model_id) {
            Ok(id) => Some(id),
            Err(err_response) => return *err_response,
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_generate(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Use provided model_id or fall back to body.model
        let model = model_id.or(Some(&body.model));
        let effective_model_id = match self.resolve_model_id(model) {
            Ok(id) => Some(id),
            Err(err_response) => return *err_response,
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_chat(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Use provided model_id or fall back to body.model
        let model = model_id.or(Some(&body.model));
        let effective_model_id = match self.resolve_model_id(model) {
            Ok(id) => Some(id),
            Err(err_response) => return *err_response,
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_completion(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Use provided model_id or fall back to body.model
        let model = model_id.or(Some(&body.model));
        let effective_model_id = match self.resolve_model_id(model) {
            Ok(id) => Some(id),
            Err(err_response) => return *err_response,
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_responses(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id, params).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to get response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        // Delegate to the default router (typically http-regular)
        // Response storage is shared across all routers via AppContext
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.list_response_input_items(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to list response input items",
            )
                .into_response()
        }
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_embeddings(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_classify(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_rerank(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*default_router)
            .finish()
    }
}
