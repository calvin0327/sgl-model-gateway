//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::{collections::HashMap, sync::Arc};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tracing::warn;

use crate::{
    config::GatewayConfig,
    core::{worker::worker_to_info, worker_registry::WorkerId, Job, JobQueue, WorkerRegistry},
    protocol_compat::{WorkerConfigRequest, WorkerInfo, WorkerUpdateRequest},
    protocols::worker::WorkerErrorResponse,
};

/// Error types for worker service operations
#[derive(Debug)]
pub enum WorkerServiceError {
    /// Worker with given ID was not found
    NotFound { worker_id: String },
    /// Invalid worker ID format (expected UUID)
    InvalidId { raw: String, message: String },
    /// Invalid worker registration payload
    InvalidConfiguration { message: String },
    /// Job queue not initialized
    QueueNotInitialized,
    /// Failed to submit job to queue
    QueueSubmitFailed { message: String },
}

impl WorkerServiceError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "WORKER_NOT_FOUND",
            Self::InvalidId { .. } => "BAD_REQUEST",
            Self::InvalidConfiguration { .. } => "BAD_REQUEST",
            Self::QueueNotInitialized => "INTERNAL_SERVER_ERROR",
            Self::QueueSubmitFailed { .. } => "INTERNAL_SERVER_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId { .. } => StatusCode::BAD_REQUEST,
            Self::InvalidConfiguration { .. } => StatusCode::BAD_REQUEST,
            Self::QueueNotInitialized => StatusCode::INTERNAL_SERVER_ERROR,
            Self::QueueSubmitFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for WorkerServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { worker_id } => write!(f, "Worker {} not found", worker_id),
            Self::InvalidId { raw, message } => {
                write!(
                    f,
                    "Invalid worker_id '{}' (expected UUID). Error: {}",
                    raw, message
                )
            }
            Self::InvalidConfiguration { message } => {
                write!(f, "Invalid worker configuration: {}", message)
            }
            Self::QueueNotInitialized => write!(f, "Job queue not initialized"),
            Self::QueueSubmitFailed { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for WorkerServiceError {}

impl IntoResponse for WorkerServiceError {
    fn into_response(self) -> Response {
        let error = WorkerErrorResponse {
            error: self.to_string(),
            code: self.error_code().to_string(),
        };
        (self.status_code(), Json(error)).into_response()
    }
}

/// Result of creating a worker (async job submission)
#[derive(Debug)]
pub struct CreateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
    pub location: String,
}

impl IntoResponse for CreateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "url": self.url,
            "location": self.location,
            "message": "Worker addition queued for background processing"
        });
        (
            StatusCode::ACCEPTED,
            [(http::header::LOCATION, self.location)],
            Json(response),
        )
            .into_response()
    }
}

/// Result of deleting a worker (async job submission)
#[derive(Debug)]
pub struct DeleteWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for DeleteWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker removal queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of updating a worker (async job submission)
#[derive(Debug)]
pub struct UpdateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for UpdateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker update queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of listing workers
#[derive(Debug)]
pub struct ListWorkersResult {
    pub workers: Vec<WorkerInfoResponse>,
    pub total: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub regular_count: usize,
}

/// Worker information returned by the gateway control-plane API.
///
/// `WorkerInfo::metadata` already contains the worker labels, but exposing
/// them as a dedicated top-level field makes the API easier to consume while
/// preserving the existing metadata field for backward compatibility.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkerInfoResponse {
    #[serde(flatten)]
    pub worker: WorkerInfo,

    /// Labels configured for or discovered from this worker.
    pub labels: HashMap<String, String>,
}

impl WorkerInfoResponse {
    fn from_worker_info(worker: WorkerInfo) -> Self {
        Self {
            labels: worker.metadata.clone(),
            worker,
        }
    }
}

impl IntoResponse for ListWorkersResult {
    fn into_response(self) -> Response {
        let response = json!({
            "workers": self.workers,
            "total": self.total,
            "stats": {
                "prefill_count": self.prefill_count,
                "decode_count": self.decode_count,
                "regular_count": self.regular_count,
            }
        });
        Json(response).into_response()
    }
}

/// Wrapper for WorkerInfo to implement IntoResponse
pub struct GetWorkerResponse(pub WorkerInfoResponse);

impl IntoResponse for GetWorkerResponse {
    fn into_response(self) -> Response {
        Json(self.0).into_response()
    }
}

/// Worker Service - Orchestrates worker business logic
///
/// This service provides a clean API for worker operations, separating
/// business logic from HTTP concerns. Handlers in server.rs become thin
/// wrappers that translate between HTTP and this service.
pub struct WorkerService {
    worker_registry: Arc<WorkerRegistry>,
    job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
    config: GatewayConfig,
}

impl WorkerService {
    /// Create a new WorkerService
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
        gateway_config: GatewayConfig,
    ) -> Self {
        Self {
            worker_registry,
            job_queue,
            config: gateway_config,
        }
    }

    /// Parse and validate a worker ID string
    pub fn parse_worker_id(&self, raw: &str) -> Result<WorkerId, WorkerServiceError> {
        uuid::Uuid::parse_str(raw)
            .map(|_| WorkerId::from_string(raw.to_string()))
            .map_err(|e| WorkerServiceError::InvalidId {
                raw: raw.to_string(),
                message: e.to_string(),
            })
    }

    /// Get the job queue, returning an error if not initialized
    fn get_job_queue(&self) -> Result<&Arc<JobQueue>, WorkerServiceError> {
        self.job_queue
            .get()
            .ok_or(WorkerServiceError::QueueNotInitialized)
    }

    pub async fn create_worker(
        &self,
        mut config: WorkerConfigRequest,
    ) -> Result<CreateWorkerResult, WorkerServiceError> {
        if !matches!(config.runtime.as_str(), "sglang" | "vllm") {
            return Err(WorkerServiceError::InvalidConfiguration {
                message: "runtime must be 'sglang' or 'vllm'".to_string(),
            });
        }
        if let Some(connector) = config.kv_connector.as_deref() {
            if !matches!(connector, "mooncake" | "nixl") {
                return Err(WorkerServiceError::InvalidConfiguration {
                    message: "kv_connector must be 'mooncake' or 'nixl'".to_string(),
                });
            }
        }
        if config.kv_engine_id.as_ref().is_some_and(String::is_empty) {
            return Err(WorkerServiceError::InvalidConfiguration {
                message: "kv_engine_id cannot be empty when specified".to_string(),
            });
        }

        if self.config.server.api_key.is_some() && config.api_key.is_none() {
            warn!(
                "Adding worker {} without API key while router has API key configured. \
                Worker will be accessible without authentication. \
                If the worker requires the same API key as the router, please specify it explicitly.",
                config.url
            );
        }

        config.dp_aware = self.config.routing.dp_aware;

        let worker_url = config.url.clone();
        let worker_id = self.worker_registry.reserve_id_for_url(&worker_url);

        let job = Job::AddWorker {
            config: Box::new(config),
        };

        self.get_job_queue()?
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        let location = format!("/workers/{}", worker_id.as_str());

        Ok(CreateWorkerResult {
            worker_id,
            url: worker_url,
            location,
        })
    }

    /// List all workers with their info
    pub fn list_workers(&self) -> ListWorkersResult {
        let workers = self.worker_registry.get_all_with_ids();
        let worker_infos: Vec<WorkerInfoResponse> = workers
            .iter()
            .map(|(worker_id, worker)| {
                let mut info = worker_to_info(worker);
                info.id = worker_id.as_str().to_string();
                WorkerInfoResponse::from_worker_info(info)
            })
            .collect();

        let stats = self.worker_registry.stats();

        ListWorkersResult {
            workers: worker_infos,
            total: stats.total_workers,
            prefill_count: stats.prefill_workers,
            decode_count: stats.decode_workers,
            regular_count: stats.regular_workers,
        }
    }

    pub fn get_worker(&self, worker_id_raw: &str) -> Result<GetWorkerResponse, WorkerServiceError> {
        let worker_id = self.parse_worker_id(worker_id_raw)?;
        let job_queue = self.get_job_queue()?;

        if let Some(worker) = self.worker_registry.get(&worker_id) {
            let worker_url = worker.url().to_string();
            let mut worker_info = worker_to_info(&worker);
            worker_info.id = worker_id.as_str().to_string();
            if let Some(status) = job_queue.get_status(&worker_url) {
                worker_info.job_status = Some(status);
            }
            return Ok(GetWorkerResponse(WorkerInfoResponse::from_worker_info(
                worker_info,
            )));
        }

        if let Some(worker_url) = self.worker_registry.get_url_by_id(&worker_id) {
            if let Some(status) = job_queue.get_status(&worker_url) {
                return Ok(GetWorkerResponse(WorkerInfoResponse::from_worker_info(
                    WorkerInfo::pending(worker_id.as_str(), worker_url, Some(status)),
                )));
            }
        }

        Err(WorkerServiceError::NotFound {
            worker_id: worker_id_raw.to_string(),
        })
    }

    /// Delete a worker by ID (submits async job)
    pub async fn delete_worker(
        &self,
        worker_id_raw: &str,
    ) -> Result<DeleteWorkerResult, WorkerServiceError> {
        let worker_id = self.parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::RemoveWorker { url: url.clone() };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(DeleteWorkerResult { worker_id, url })
    }

    /// Update a worker by ID (submits async job)
    pub async fn update_worker(
        &self,
        worker_id_raw: &str,
        update: WorkerUpdateRequest,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = self.parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::UpdateWorker {
            url: url.clone(),
            update: Box::new(update),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }
}
