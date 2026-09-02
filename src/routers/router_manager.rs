use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use data_connector::{ResponseId, ResponseStorage};
use serde_json::Value;
use tracing::info;

use crate::{
    app_context::AppContext,
    config::{GatewayConfig, RoutingConfig},
    core::{
        model_type::Endpoint,
        scheduler::{Scheduler, SchedulerError, SchedulerInput, SchedulerOutput, WorkerSelection},
        worker::Worker,
        ConnectionMode, WorkerRegistry,
    },
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::GenerationRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{RouterFactory, RouterTrait},
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
    response_storage: Arc<dyn ResponseStorage>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    scheduler: Arc<Scheduler>,
}

impl RouterManager {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        routing_config: &RoutingConfig,
        response_storage: Arc<dyn ResponseStorage>,
    ) -> Self {
        Self {
            scheduler: Arc::new(Scheduler::new(worker_registry.clone(), routing_config)),
            worker_registry,
            response_storage,
            routers: Arc::new(DashMap::new()),
        }
    }

    pub fn from_config(
        config: &GatewayConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        let manager = Arc::new(Self::new(
            app_context.worker_registry.clone(),
            &config.routing,
            app_context.response_storage.clone(),
        ));

        info!("Initializing all router implementations");

        // Routers are protocol/mode handlers shared by all models. They do not
        // need a worker to be present during construction; worker availability
        // is checked from WorkerRegistry for every request.
        let router_specs = [
            (
                router_ids::HTTP_REGULAR,
                RouterFactory::create_regular_router(app_context)?,
            ),
            (
                router_ids::HTTP_PD,
                RouterFactory::create_pd_router(app_context)?,
            ),
            (
                router_ids::GRPC_REGULAR,
                RouterFactory::create_grpc_router(app_context)?,
            ),
            (
                router_ids::GRPC_PD,
                RouterFactory::create_grpc_pd_router(app_context)?,
            ),
            (
                router_ids::HTTP_OPENAI,
                RouterFactory::create_openai_router(app_context)?,
            ),
        ];

        for (router_id, router) in router_specs {
            manager.register_router(router_id, Arc::from(router));
        }

        Ok(manager)
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);
        info!("Registered router: {}", id.as_str());
    }

    /// Return workers visible for a model, or all workers for an unresolved model.
    fn workers_for_model(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        match model_id {
            Some(model) => self.worker_registry.get_by_model(model).to_vec(),
            None => self.worker_registry.get_all(),
        }
    }

    /// Return whether at least one healthy worker is available for the model.
    ///
    /// This is a lightweight readiness check. Detailed protocol, worker-mode,
    /// and PD-pair validation is performed by the Scheduler for actual
    /// requests.
    pub fn has_available_route(&self, model_id: Option<&str>) -> bool {
        self.workers_for_model(model_id)
            .iter()
            .any(|worker| worker.is_healthy())
    }

    fn get_router(&self, router_id: &RouterId) -> Option<Arc<dyn RouterTrait>> {
        self.routers
            .get(router_id)
            .map(|entry| entry.value().clone())
    }

    async fn response_is_local(&self, response_id: &str) -> Result<bool, Response> {
        match self
            .response_storage
            .get_response(&ResponseId::from(response_id))
            .await
        {
            Ok(response) => Ok(response.is_some()),
            Err(error) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to locate response '{}': {}", response_id, error),
            )
                .into_response()),
        }
    }

    async fn select_response_router(
        &self,
        response_id: &str,
        local_router_id: RouterId,
    ) -> Result<Option<Arc<dyn RouterTrait>>, Response> {
        let router_id = if self.response_is_local(response_id).await? {
            local_router_id
        } else {
            router_ids::HTTP_REGULAR
        };

        Ok(self.get_router(&router_id))
    }

    fn select_router(&self, selection: &SchedulerOutput) -> Option<Arc<dyn RouterTrait>> {
        let route_id = match &selection.worker_selection {
            WorkerSelection::Regular { worker } => match worker.connection_mode() {
                ConnectionMode::Http => Some(router_ids::HTTP_REGULAR),
                ConnectionMode::Grpc { .. } => Some(router_ids::GRPC_REGULAR),
            },
            WorkerSelection::PrefillDecode { prefill, .. } => match prefill.connection_mode() {
                ConnectionMode::Http => Some(router_ids::HTTP_PD),
                ConnectionMode::Grpc { .. } => Some(router_ids::GRPC_PD),
            },
            WorkerSelection::External { worker } => match worker.connection_mode() {
                ConnectionMode::Http => Some(router_ids::HTTP_OPENAI),
                ConnectionMode::Grpc { .. } => None,
            },
        };

        route_id.and_then(|route_id| {
            self.routers
                .get(&route_id)
                .map(|router| router.value().clone())
        })
    }

    async fn schedule_request(
        &self,
        model_id: Option<&str>,
        endpoint: Endpoint,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<SchedulerOutput, Response> {
        self.scheduler
            .schedule(&SchedulerInput {
                model_id,
                endpoint,
                request_text,
                token_ids: None,
                headers,
            })
            .await
            .map_err(|error| {
                let status = match error {
                    SchedulerError::ModelRequired { .. } => StatusCode::BAD_REQUEST,
                    SchedulerError::NoModelsAvailable
                    | SchedulerError::NoAvailableWorker { .. }
                    | SchedulerError::NoAvailablePdPair { .. } => StatusCode::SERVICE_UNAVAILABLE,
                };
                (status, error.to_string()).into_response()
            })
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Keep the health check simple: the gateway is ready when at least one
        // registered worker is healthy. Detailed worker selection is handled by
        // the Scheduler when serving an actual request.
        if self
            .worker_registry
            .get_all()
            .iter()
            .any(|worker| worker.is_healthy())
        {
            (StatusCode::OK, "At least one router has healthy workers").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
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
        // Keep the existing no-model behavior: use the first registered
        // router. Model-aware request APIs use Scheduler selections.
        let router = self.routers.iter().next().map(|r| r.value().clone());

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
        let model_id = model_id.or_else(|| body.get_model());
        let selection = match self
            .schedule_request(model_id, Endpoint::Generate, body.text.as_deref(), headers)
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };

        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_generate_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
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
        let model_id = model_id.or_else(|| body.get_model());
        let request_text = body.extract_text_for_routing();

        let selection = match self
            .schedule_request(
                model_id,
                Endpoint::Chat,
                (!request_text.is_empty()).then_some(request_text.as_str()),
                headers,
            )
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };

        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_chat_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
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
        let model_id = model_id.or_else(|| body.get_model());
        let request_text = body.extract_text_for_routing();

        let selection = match self
            .schedule_request(
                model_id,
                Endpoint::Completions,
                (!request_text.is_empty()).then_some(request_text.as_str()),
                headers,
            )
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };

        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_completion_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
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
        let model_id = model_id.or_else(|| body.get_model());
        let selection = match self
            .schedule_request(model_id, Endpoint::Responses, None, headers)
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };

        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_responses_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
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
        let router = match self
            .select_response_router(response_id, router_ids::HTTP_OPENAI)
            .await
        {
            Ok(router) => router,
            Err(response) => return response,
        };
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
        let router = match self
            .select_response_router(response_id, router_ids::GRPC_REGULAR)
            .await
        {
            Ok(router) => router,
            Err(response) => return response,
        };
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
        let router = match self
            .select_response_router(response_id, router_ids::HTTP_OPENAI)
            .await
        {
            Ok(router) => router,
            Err(response) => return response,
        };
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
        let model_id = model_id.or_else(|| body.get_model());
        let selection = match self
            .schedule_request(model_id, Endpoint::Embeddings, None, headers)
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };

        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_embeddings_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
                .await
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
        let model_id = model_id.or_else(|| body.get_model());
        let selection = match self
            .schedule_request(model_id, Endpoint::Models, None, headers)
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };
        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_classify_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
                .await
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
        let model_id = model_id.or_else(|| body.get_model());
        let selection = match self
            .schedule_request(
                model_id,
                Endpoint::Rerank,
                Some(body.query.as_str()),
                headers,
            )
            .await
        {
            Ok(selection) => selection,
            Err(response) => return response,
        };
        let router = self.select_router(&selection);

        if let Some(router) = router {
            router
                .route_rerank_with_selection(
                    headers,
                    body,
                    Some(selection.model_id.as_str()),
                    &selection,
                )
                .await
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
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .finish()
    }
}
