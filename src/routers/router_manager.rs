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
use futures_util::future::BoxFuture;
use tracing::info;

use crate::{
    app_context::AppContext,
    config::{GatewayConfig, RetryConfig},
    core::{
        is_retryable_status,
        model_type::Endpoint,
        scheduler::{Scheduler, SchedulerError, SchedulerInput, SchedulerOutput, WorkerSelection},
        worker::Worker,
        ConnectionMode, RetryExecutor, WorkerRegistry,
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
    tokenizer::registry::TokenizerRegistry,
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

type ScheduledExecution =
    Arc<dyn Fn(Arc<dyn RouterTrait>) -> BoxFuture<'static, Response> + Send + Sync>;

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    response_storage: Arc<dyn ResponseStorage>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    scheduler: Arc<Scheduler>,
    retry_config: RetryConfig,
}

impl RouterManager {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<crate::policies::PolicyRegistry>,
        response_storage: Arc<dyn ResponseStorage>,
    ) -> Self {
        Self::new_with_retry_config(
            worker_registry,
            policy_registry,
            response_storage,
            RetryConfig::default(),
            None,
        )
    }

    fn new_with_retry_config(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<crate::policies::PolicyRegistry>,
        response_storage: Arc<dyn ResponseStorage>,
        retry_config: RetryConfig,
        tokenizer_registry: Option<Arc<TokenizerRegistry>>,
    ) -> Self {
        Self {
            scheduler: Arc::new(Scheduler::new_with_tokenizer_registry(
                worker_registry.clone(),
                policy_registry,
                tokenizer_registry,
            )),
            worker_registry,
            response_storage,
            routers: Arc::new(DashMap::new()),
            retry_config,
        }
    }

    pub fn from_config(
        config: &GatewayConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        let manager = Arc::new(Self::new_with_retry_config(
            app_context.worker_registry.clone(),
            app_context.policy_registry.clone(),
            app_context.response_storage.clone(),
            config.workers.effective_retry_config(),
            Some(app_context.tokenizer_registry.clone()),
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

    /// Select the backend used by APIs that do not carry a model ID.
    /// These endpoints retain their historical direct-backend behavior.
    fn default_router(&self) -> Option<Arc<dyn RouterTrait>> {
        [
            router_ids::HTTP_REGULAR,
            router_ids::HTTP_OPENAI,
            router_ids::GRPC_REGULAR,
            router_ids::HTTP_PD,
            router_ids::GRPC_PD,
        ]
        .iter()
        .find_map(|router_id| self.get_router(router_id))
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

    /// Schedule and execute one request attempt.
    ///
    /// Retry ownership deliberately lives here rather than in Scheduler or a
    /// concrete Router. Every retry gets a fresh Scheduler snapshot and the
    /// resulting selection is passed to exactly one matching Router.
    async fn execute_with_retry(
        &self,
        model_id: Option<&str>,
        endpoint: Endpoint,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        execute: ScheduledExecution,
    ) -> Response {
        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let execute = Arc::clone(&execute);
                async move {
                    let selection = match self
                        .schedule_request(model_id, endpoint, request_text, headers)
                        .await
                    {
                        Ok(selection) => selection,
                        Err(response) => return response,
                    };

                    let Some(router) = self.select_router(&selection) else {
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "No router available for the scheduled worker",
                        )
                            .into_response();
                    };

                    crate::routers::SCHEDULED_SELECTION
                        .scope(Some(selection), execute(router))
                        .await
                }
            },
            |response, _attempt| is_retryable_status(response.status()),
            |_delay, _next_attempt| {},
            || {},
        )
        .await
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        if let Some(router) = self.default_router() {
            router.health_generate(Request::new(Body::empty())).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        match self.default_router() {
            Some(router) => router.get_server_info(req).await,
            None => (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response(),
        }
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        match self.default_router() {
            Some(router) => router.get_models(req).await,
            None => (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response(),
        }
    }

    // TODO: multi model
    async fn get_model_info(&self, req: Request<Body>) -> Response {
        if let Some(router) = self.default_router() {
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
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();

        self.execute_with_retry(
            model_id,
            Endpoint::Generate,
            body.text.as_deref(),
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move {
                    router
                        .route_generate(headers.as_ref(), &request, None)
                        .await
                })
            }),
        )
        .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let model_id = model_id.or_else(|| body.get_model());
        let request_text = body.extract_text_for_routing();
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();

        self.execute_with_retry(
            model_id,
            Endpoint::Chat,
            (!request_text.is_empty()).then_some(request_text.as_str()),
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move { router.route_chat(headers.as_ref(), &request, None).await })
            }),
        )
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let model_id = model_id.or_else(|| body.get_model());
        let request_text = body.extract_text_for_routing();
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();

        self.execute_with_retry(
            model_id,
            Endpoint::Completions,
            (!request_text.is_empty()).then_some(request_text.as_str()),
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move {
                    router
                        .route_completion(headers.as_ref(), &request, None)
                        .await
                })
            }),
        )
        .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let model_id = model_id.or_else(|| body.get_model());
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();

        self.execute_with_retry(
            model_id,
            Endpoint::Responses,
            None,
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move {
                    router
                        .route_responses(headers.as_ref(), &request, None)
                        .await
                })
            }),
        )
        .await
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
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();
        self.execute_with_retry(
            model_id,
            Endpoint::Embeddings,
            None,
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move {
                    router
                        .route_embeddings(headers.as_ref(), &request, None)
                        .await
                })
            }),
        )
        .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let model_id = model_id.or_else(|| body.get_model());
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();

        self.execute_with_retry(
            model_id,
            Endpoint::Models,
            None,
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move {
                    router
                        .route_classify(headers.as_ref(), &request, None)
                        .await
                })
            }),
        )
        .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let model_id = model_id.or_else(|| body.get_model());
        let request = Arc::new(body.clone());
        let request_headers = headers.cloned();

        self.execute_with_retry(
            model_id,
            Endpoint::Rerank,
            Some(body.query.as_str()),
            headers,
            Arc::new(move |router| {
                let request = Arc::clone(&request);
                let headers = request_headers.clone();
                Box::pin(async move { router.route_rerank(headers.as_ref(), &request, None).await })
            }),
        )
        .await
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
