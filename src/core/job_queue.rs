//! Async job queue for control plane operations
//!
//! Provides non-blocking worker management by queuing operations and processing
//! them asynchronously in background worker tasks.

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};
use wfaas::WorkflowId;

use crate::{
    app_context::AppContext,
    config::GatewayConfig,
    core::{
        steps::{
            create_local_worker_workflow_data, create_tokenizer_workflow_data,
            create_worker_removal_workflow_data, create_worker_update_workflow_data,
            TokenizerConfigRequest, TokenizerRemovalRequest,
        },
        PD_GROUP_ID_LABEL, PREFERRED_CONTEXT_TOKENS_LABEL,
    },
    protocol_compat::{WorkerConfigRequest, WorkerUpdateRequest},
    protocols::worker::JobStatus,
    routers::router_manager::RouterManager,
};

/// Job types for control plane operations
#[derive(Debug, Clone)]
pub enum Job {
    AddWorker {
        config: Box<WorkerConfigRequest>,
    },
    UpdateWorker {
        url: String,
        update: Box<WorkerUpdateRequest>,
    },
    RemoveWorker {
        url: String,
    },
    InitializeWorkersFromConfig {
        gateway_config: Box<GatewayConfig>,
    },
    AddTokenizer {
        config: Box<TokenizerConfigRequest>,
    },
    RemoveTokenizer {
        request: Box<TokenizerRemovalRequest>,
    },
}

impl Job {
    /// Get job type as string for logging
    pub fn job_type(&self) -> &'static str {
        match self {
            Job::AddWorker { .. } => "AddWorker",
            Job::UpdateWorker { .. } => "UpdateWorker",
            Job::RemoveWorker { .. } => "RemoveWorker",
            Job::InitializeWorkersFromConfig { .. } => "InitializeWorkersFromConfig",
            Job::AddTokenizer { .. } => "AddTokenizer",
            Job::RemoveTokenizer { .. } => "RemoveTokenizer",
        }
    }

    /// Get worker URL or tokenizer identifier for logging and status tracking
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config } => &config.url,
            Job::UpdateWorker { url, .. } => url,
            Job::RemoveWorker { url } => url,
            Job::InitializeWorkersFromConfig { .. } => "startup",
            Job::AddTokenizer { config } => &config.id,
            Job::RemoveTokenizer { request } => &request.id,
        }
    }
}

/// Job queue configuration
#[derive(Clone, Debug)]
pub struct JobQueueConfig {
    /// Maximum pending jobs in queue
    pub queue_capacity: usize,
    /// Maximum number of jobs executing concurrently
    pub max_concurrent_jobs: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1000,
            max_concurrent_jobs: 200,
        }
    }
}

/// Job queue manager for worker validation and removal operations
pub struct JobQueue {
    /// Channel for submitting jobs
    tx: mpsc::Sender<Job>,
    /// Weak reference to AppContext to avoid circular dependencies
    context: Weak<AppContext>,
    /// Job status tracking by worker URL
    status_map: Arc<DashMap<String, JobStatus>>,
    /// Semaphore to limit concurrent job execution
    concurrency_limit: Arc<Semaphore>,
    /// Router manager injected after application startup.
    router_manager: Arc<OnceLock<Arc<RouterManager>>>,
}

impl std::fmt::Debug for JobQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobQueue")
            .field("status_count", &self.status_map.len())
            .finish()
    }
}

impl JobQueue {
    /// Create a new job queue with semaphore-based concurrency control
    ///
    /// Takes a Weak reference to AppContext to avoid circular strong references.
    /// Spawns a single dispatcher task that spawns individual job tasks with semaphore control.
    pub fn new(config: JobQueueConfig, context: Weak<AppContext>) -> Arc<Self> {
        let (tx, mut rx) = mpsc::channel(config.queue_capacity);

        debug!(
            "Initializing job queue: capacity={}, max_concurrent={}",
            config.queue_capacity, config.max_concurrent_jobs
        );

        let status_map = Arc::new(DashMap::new());
        let concurrency_limit = Arc::new(Semaphore::new(config.max_concurrent_jobs));

        let queue = Arc::new(Self {
            tx,
            context: context.clone(),
            status_map: status_map.clone(),
            concurrency_limit: concurrency_limit.clone(),
            router_manager: Arc::new(OnceLock::new()),
        });

        // Single dispatcher task
        let ctx = context.clone();
        let status = status_map.clone();
        let sem = concurrency_limit.clone();

        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                // Acquire permit (blocks if at concurrency limit)
                let Ok(permit) = sem.clone().acquire_owned().await else {
                    error!("Semaphore closed, stopping dispatcher");
                    break;
                };

                let ctx_clone = ctx.clone();
                let status_clone = status.clone();

                tokio::spawn(async move {
                    Self::process_job(job, ctx_clone, status_clone, permit).await;
                });
            }

            debug!("Job dispatcher stopped");
        });

        // Spawn cleanup task for old job statuses (TTL 5 minutes)
        let cleanup_status_map = status_map.clone();
        tokio::spawn(async move {
            Self::cleanup_old_statuses(cleanup_status_map).await;
        });

        queue
    }

    /// Get current queue and concurrency status
    pub fn get_load_info(&self) -> (usize, usize) {
        let queue_depth = self.tx.max_capacity() - self.tx.capacity();
        let available_permits = self.concurrency_limit.available_permits();
        (queue_depth, available_permits)
    }

    pub fn set_router_manager(&self, router_manager: Arc<RouterManager>) -> Result<(), String> {
        self.router_manager
            .set(router_manager)
            .map_err(|_| "RouterManager already initialized".to_string())
    }

    pub fn router_manager(&self) -> Option<Arc<RouterManager>> {
        self.router_manager.get().cloned()
    }

    /// Submit a job with detailed queue status
    pub async fn submit(&self, job: Job) -> Result<(), String> {
        // Check if context is still alive before accepting jobs
        if self.context.upgrade().is_none() {
            return Err("Job queue shutting down: AppContext dropped".to_string());
        }

        // Extract values before moving job
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();

        // Record pending status
        self.status_map.insert(
            worker_url.clone(),
            JobStatus::pending(job_type, &worker_url),
        );

        match self.tx.send(job).await {
            Ok(_) => {
                let (queue_depth, available_permits) = self.get_load_info();
                debug!(
                    "Job submitted: type={}, worker={}, queue_depth={}, available_slots={}",
                    job_type, worker_url, queue_depth, available_permits
                );
                Ok(())
            }
            Err(_) => {
                self.status_map.remove(&worker_url);
                let (queue_depth, _) = self.get_load_info();
                Err(format!(
                    "Job queue full: {} jobs pending (capacity: {})",
                    queue_depth,
                    self.tx.max_capacity()
                ))
            }
        }
    }

    /// Get job status by worker URL
    pub fn get_status(&self, worker_url: &str) -> Option<JobStatus> {
        self.status_map.get(worker_url).map(|entry| entry.clone())
    }

    /// Remove job status (called when worker is deleted)
    pub fn remove_status(&self, worker_url: &str) {
        self.status_map.remove(worker_url);
    }

    /// Process a single job with status tracking and error handling
    async fn process_job(
        job: Job,
        context: Weak<AppContext>,
        status_map: Arc<DashMap<String, JobStatus>>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();
        let start = std::time::Instant::now();

        // Update to processing
        status_map.insert(
            worker_url.clone(),
            JobStatus::processing(job_type, &worker_url),
        );

        debug!(job_type, resource = %worker_url, "processing job");

        // Execute job
        match context.upgrade() {
            Some(ctx) => {
                let result = Self::execute_job(&job, &ctx).await;
                let duration = start.elapsed();
                Self::record_job_completion(job_type, &worker_url, duration, &result, &status_map);
            }
            None => {
                let error_msg = "AppContext dropped".to_string();
                status_map.insert(
                    worker_url.clone(),
                    JobStatus::failed(job_type, &worker_url, error_msg),
                );
                error!(
                    job_type,
                    resource = %worker_url,
                    "AppContext dropped; cannot process job"
                );
            }
        }

        // Permit automatically released when dropped
    }

    /// Execute a specific job
    async fn execute_job(job: &Job, context: &Arc<AppContext>) -> Result<String, String> {
        let router_manager = context
            .worker_job_queue
            .get()
            .and_then(|queue| queue.router_manager());

        match job {
            Job::AddWorker { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let timeout_duration =
                    Duration::from_secs(context.config.workers.startup_timeout_secs + 30);

                let workflow_data = create_local_worker_workflow_data(
                    (**config).clone(),
                    Arc::clone(context),
                    router_manager,
                );
                let instance_id = engines
                    .local_worker
                    .start_workflow(WorkflowId::new("local_worker_registration"), workflow_data)
                    .await
                    .map_err(|e| {
                        format!(
                            "Failed to start local worker registration workflow: {:?}",
                            e
                        )
                    })?;

                debug!(
                    "Started local worker registration workflow for {} (instance: {})",
                    config.url, instance_id
                );

                engines
                    .local_worker
                    .wait_for_completion(instance_id, &config.url, timeout_duration)
                    .await
            }
            Job::UpdateWorker { url, update } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_update_workflow_data(
                    url.to_string(),
                    (**update).clone(),
                    Arc::clone(context),
                );

                let instance_id = engines
                    .worker_update
                    .start_workflow(WorkflowId::new("worker_update"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker update workflow: {:?}", e))?;

                debug!(
                    "Started worker update workflow for {} (instance: {})",
                    url, instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                engines
                    .worker_update
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await
            }
            Job::RemoveWorker { url } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_removal_workflow_data(
                    url.to_string(),
                    context.config.routing.dp_aware,
                    Arc::clone(context),
                );

                let instance_id = engines
                    .worker_removal
                    .start_workflow(WorkflowId::new("worker_removal"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker removal workflow: {:?}", e))?;

                debug!(
                    "Started worker removal workflow for {} (instance: {})",
                    url, instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                let result = engines
                    .worker_removal
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await;

                // Clean up job status when removing worker
                if let Some(queue) = context.worker_job_queue.get() {
                    queue.remove_status(url);
                }

                result
            }
            Job::InitializeWorkersFromConfig {
                gateway_config: config,
            } => {
                let api_key = config.server.api_key.clone();

                let mut worker_specs: Vec<(String, &'static str, Option<u16>, String)> = Vec::new();

                // Regular workers
                worker_specs.extend(
                    config
                        .workers
                        .regular_urls
                        .iter()
                        .cloned()
                        .map(|url| (url, "regular", None, config.workers.backend.clone())),
                );

                // Prefill workers
                worker_specs.extend(config.workers.prefill_urls.iter().map(
                    |(url, bootstrap_port)| {
                        (
                            url.clone(),
                            "prefill",
                            *bootstrap_port,
                            config.workers.backend.clone(),
                        )
                    },
                ));

                // Decode workers
                worker_specs.extend(
                    config
                        .workers
                        .decode_urls
                        .iter()
                        .cloned()
                        .map(|url| (url, "decode", None, config.workers.backend.clone())),
                );

                debug!(
                    "Creating AddWorker jobs for {} workers from config",
                    worker_specs.len()
                );

                let mut submitted_count = 0;

                for (url, worker_type, bootstrap_port, runtime) in worker_specs {
                    let url_for_error = url.clone();
                    let health_check = &config.workers.health_check;
                    let mut labels = HashMap::new();
                    if matches!(worker_type, "prefill" | "decode") {
                        if let Some(group_id) = config.workers.pd_group_id.as_ref() {
                            labels.insert(PD_GROUP_ID_LABEL.to_string(), group_id.clone());
                        }
                    }

                    let preferred_context_tokens = match worker_type {
                        "regular" => config.workers.regular_preferred_context_tokens,
                        "prefill" => config.workers.pd_preferred_context_tokens,
                        // Decode workers do not participate in context-length routing.
                        _ => None,
                    };
                    if let Some(tokens) = preferred_context_tokens {
                        labels.insert(
                            PREFERRED_CONTEXT_TOKENS_LABEL.to_string(),
                            tokens.to_string(),
                        );
                    }
                    let worker_config = WorkerConfigRequest {
                        url,
                        api_key: api_key.clone(),

                        worker_type: Some(worker_type.to_string()),
                        runtime,
                        kv_connector: config.workers.kv_connector.clone(),
                        kv_engine_id: config.workers.kv_engine_id.clone(),

                        labels,
                        model_id: None,
                        priority: None,
                        cost: None,

                        tokenizer_path: None,
                        reasoning_parser: None,
                        tool_parser: None,
                        chat_template: config.model.chat_template.clone(),

                        bootstrap_port,

                        health_check_timeout_secs: health_check.timeout_secs,
                        health_check_interval_secs: health_check.check_interval_secs,
                        health_success_threshold: health_check.success_threshold,
                        health_failure_threshold: health_check.failure_threshold,
                        disable_health_check: health_check.disable_health_check,

                        max_connection_attempts: health_check.success_threshold.saturating_mul(10),

                        dp_aware: config.routing.dp_aware,
                    };

                    let job = Job::AddWorker {
                        config: Box::new(worker_config),
                    };

                    let queue = context
                        .worker_job_queue
                        .get()
                        .ok_or_else(|| "JobQueue not available".to_string())?;

                    queue.submit(job).await.map_err(|error| {
                        format!(
                            "Failed to submit AddWorker job for {} worker {}: {}",
                            worker_type, url_for_error, error
                        )
                    })?;

                    submitted_count += 1;
                }

                Ok(format!("Submitted {} AddWorker jobs", submitted_count))
            }
            Job::AddTokenizer { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_tokenizer_workflow_data(*config.clone(), Arc::clone(context));

                let instance_id = engines
                    .tokenizer
                    .start_workflow(WorkflowId::new("tokenizer_registration"), workflow_data)
                    .await
                    .map_err(|e| {
                        format!("Failed to start tokenizer registration workflow: {:?}", e)
                    })?;

                debug!(
                    "Started tokenizer registration workflow for '{}' id={} (instance: {})",
                    config.name, config.id, instance_id
                );

                // Allow up to 10 minutes for HuggingFace downloads
                let timeout_duration = Duration::from_secs(600);

                engines
                    .tokenizer
                    .wait_for_completion(instance_id, &config.id, timeout_duration)
                    .await
            }
            Job::RemoveTokenizer { request } => {
                // Tokenizer removal is synchronous and fast
                if let Some(entry) = context.tokenizer_registry.remove_by_id(&request.id) {
                    info!(
                        "Successfully removed tokenizer '{}' (id: {})",
                        entry.name, entry.id
                    );
                    Ok(format!("Tokenizer '{}' removed successfully", entry.name))
                } else {
                    Err(format!("Tokenizer with id '{}' not found", request.id))
                }
            }
        }
    }

    /// Update job status on completion
    fn record_job_completion(
        job_type: &'static str,
        worker_url: &str,
        duration: Duration,
        result: &Result<String, String>,
        status_map: &Arc<DashMap<String, JobStatus>>,
    ) {
        match result {
            Ok(message) => {
                status_map.remove(worker_url);
                debug!(
                    job_type,
                    resource = %worker_url,
                    duration_ms = duration.as_millis() as u64,
                    result_message = %message,
                    "job completed"
                );
            }
            Err(error) => {
                status_map.insert(
                    worker_url.to_string(),
                    JobStatus::failed(job_type, worker_url, error.clone()),
                );
                warn!(
                    job_type,
                    resource = %worker_url,
                    duration_ms = duration.as_millis() as u64,
                    error = %error,
                    "job failed"
                );
            }
        }
    }

    /// Cleanup old job statuses (TTL 5 minutes)
    async fn cleanup_old_statuses(status_map: Arc<DashMap<String, JobStatus>>) {
        const CLEANUP_INTERVAL: Duration = Duration::from_secs(60); // Run every minute
        const STATUS_TTL: u64 = 300; // 5 minutes in seconds

        loop {
            tokio::time::sleep(CLEANUP_INTERVAL).await;

            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Remove statuses older than TTL
            status_map.retain(|_key, value| now - value.timestamp < STATUS_TTL);

            debug!(
                "Cleaned up old job statuses, remaining: {}",
                status_map.len()
            );
        }
    }
}
