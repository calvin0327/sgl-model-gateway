use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        watcher::{watcher, Config},
        WatchStreamExt,
    },
    Client,
};
use rustls;
use tokio::{task, time};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    core::{Job, PREFERRED_CONTEXT_TOKENS_LABEL},
    observability::metrics::{metrics_labels, Metrics},
    protocol_compat::WorkerConfigRequest,
};

#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub enabled: bool,
    pub selector: HashMap<String, String>,
    pub check_interval: Duration,
    pub port: u16,
    pub namespace: Option<String>,

    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    // Bootstrap port annotation specific to mooncake implementation
    pub bootstrap_port_annotation: String,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8000,
            namespace: None,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Prefill,
    Decode,
    Regular,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodInfo {
    pub name: String,
    pub ip: String,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    pub bootstrap_port: Option<u16>,
}

impl PodInfo {
    fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata
            .labels
            .as_ref()
            .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
    }

    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.selector.is_empty()
            && config.prefill_selector.is_empty()
            && config.decode_selector.is_empty()
        {
            warn!("Worker discovery enabled but all worker selectors are empty");
            return false;
        }

        Self::matches_selector(pod, &config.selector)
            || Self::matches_selector(pod, &config.prefill_selector)
            || Self::matches_selector(pod, &config.decode_selector)
    }

    pub fn from_pod(pod: &Pod, config: Option<&ServiceDiscoveryConfig>) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let status = pod.status.clone()?;
        let pod_ip = status.pod_ip?;

        let is_ready = if let Some(conditions) = &status.conditions {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        } else {
            false
        };

        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        let pod_type = config.and_then(|config| {
            if Self::matches_selector(pod, &config.prefill_selector) {
                Some(PodType::Prefill)
            } else if Self::matches_selector(pod, &config.decode_selector) {
                Some(PodType::Decode)
            } else if Self::matches_selector(pod, &config.selector) {
                Some(PodType::Regular)
            } else {
                None
            }
        });

        let bootstrap_port = if matches!(pod_type, Some(PodType::Prefill)) {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.bootstrap_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        Some(PodInfo {
            name,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            bootstrap_port,
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }

    pub fn worker_url(&self, port: u16) -> String {
        // Default to http:// prefix; workflow will detect actual protocol (HTTP vs gRPC)
        format!("http://{}:{}", self.ip, port)
    }
}

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.enabled {
        return Err(kube::Error::Api(kube::error::ErrorResponse {
            status: "Disabled".to_string(),
            message: "Service discovery is disabled".to_string(),
            reason: "ConfigurationError".to_string(),
            code: 400,
        }));
    }

    let _ = rustls::crypto::ring::default_provider().install_default();

    let client = Client::try_default().await?;

    let format_selector = |selector: &HashMap<String, String>| {
        selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    };
    info!(
        "Starting K8s service discovery | regular: '{}' | prefill: '{}' | decode: '{}'",
        format_selector(&config.selector),
        format_selector(&config.prefill_selector),
        format_selector(&config.decode_selector),
    );

    let handle = task::spawn(async move {
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));

        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("K8s service discovery initialized");

        let config_arc = Arc::new(config.clone());
        let port = config.port;

        let mut retry_delay = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

        loop {
            let watcher_config = Config::default();
            let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

            let config_clone = Arc::clone(&config_arc);
            let tracked_pods_clone = Arc::clone(&tracked_pods);

            let filtered_stream = watcher_stream.filter_map(move |obj_res| {
                let config_inner = Arc::clone(&config_clone);

                async move {
                    match obj_res {
                        Ok(pod) => {
                            if PodInfo::should_include(&pod, &config_inner) {
                                Some(Ok(pod))
                            } else {
                                None
                            }
                        }
                        Err(e) => Some(Err(e)),
                    }
                }
            });

            let tracked_pods_clone2 = Arc::clone(&tracked_pods_clone);
            let app_context_clone = Arc::clone(&app_context);
            let config_clone2 = Arc::clone(&config_arc);

            match filtered_stream
                .try_for_each(move |pod| {
                    let tracked_pods_inner = Arc::clone(&tracked_pods_clone2);
                    let app_context_inner = Arc::clone(&app_context_clone);
                    let config_inner = Arc::clone(&config_clone2);

                    async move {
                        let pod_info = PodInfo::from_pod(&pod, Some(&config_inner));

                        if let Some(pod_info) = pod_info {
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(
                                    &pod_info,
                                    tracked_pods_inner,
                                    app_context_inner,
                                    port,
                                )
                                .await;
                            } else {
                                handle_pod_event(
                                    &pod_info,
                                    tracked_pods_inner,
                                    app_context_inner,
                                    port,
                                )
                                .await;
                            }
                        }
                        Ok(())
                    }
                })
                .await
            {
                Ok(_) => {
                    retry_delay = Duration::from_secs(1);
                }
                Err(err) => {
                    error!("Error in Kubernetes watcher: {}", err);
                    warn!(
                        "Retrying in {} seconds with exponential backoff",
                        retry_delay.as_secs()
                    );
                    time::sleep(retry_delay).await;

                    retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                }
            }

            warn!(
                "Kubernetes watcher exited, restarting in {} seconds",
                config_arc.check_interval.as_secs()
            );
            time::sleep(config_arc.check_interval).await;
        }
    });

    Ok(handle)
}

async fn handle_pod_event(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    app_context: Arc<AppContext>,
    port: u16,
) {
    let worker_url = pod_info.worker_url(port);

    if pod_info.is_healthy() {
        // Track whether to add and get count in single lock acquisition
        let (should_add, tracked_count) = {
            let mut tracker = match tracked_pods.lock() {
                Ok(tracker) => tracker,
                Err(e) => {
                    error!("Failed to acquire tracked_pods lock: {}", e);
                    return;
                }
            };

            if tracker.contains(pod_info) {
                (false, tracker.len())
            } else {
                tracker.insert(pod_info.clone());
                (true, tracker.len())
            }
        };

        if should_add {
            info!(
                "Adding pod: {} | type: {:?} | url: {}",
                pod_info.name, pod_info.pod_type, worker_url
            );

            let worker_type = match &pod_info.pod_type {
                Some(PodType::Prefill) => Some("prefill".to_string()),
                Some(PodType::Decode) => Some("decode".to_string()),
                Some(PodType::Regular) => Some("regular".to_string()),
                None => return,
            };

            let bootstrap_port = match &pod_info.pod_type {
                Some(PodType::Prefill) => pod_info.bootstrap_port,
                _ => None,
            };

            let preferred_context_tokens = match &pod_info.pod_type {
                Some(PodType::Regular) => {
                    app_context.config.workers.regular_preferred_context_tokens
                }
                Some(PodType::Prefill) => app_context.config.workers.pd_preferred_context_tokens,
                _ => None,
            };
            let mut labels = HashMap::new();
            if let Some(tokens) = preferred_context_tokens {
                labels.insert(
                    PREFERRED_CONTEXT_TOKENS_LABEL.to_string(),
                    tokens.to_string(),
                );
            }

            let config = WorkerConfigRequest {
                url: worker_url.clone(),
                model_id: None,
                worker_type,
                priority: None,
                cost: None,
                runtime: app_context.config.workers.backend.clone(),
                kv_connector: app_context.config.workers.kv_connector.clone(),
                kv_engine_id: app_context.config.workers.kv_engine_id.clone(),
                labels,
                bootstrap_port,
                tokenizer_path: None,
                reasoning_parser: None,
                tool_parser: None,
                chat_template: None,
                api_key: app_context.config.server.api_key.clone(),
                health_check_timeout_secs: app_context.config.workers.health_check.timeout_secs,
                health_check_interval_secs: app_context
                    .config
                    .workers
                    .health_check
                    .check_interval_secs,
                health_success_threshold: app_context.config.workers.health_check.success_threshold,
                health_failure_threshold: app_context.config.workers.health_check.failure_threshold,
                disable_health_check: app_context.config.workers.health_check.disable_health_check,
                max_connection_attempts: app_context.config.workers.health_check.success_threshold
                    * 20,
                dp_aware: app_context.config.routing.dp_aware,
            };

            let job = Job::AddWorker {
                config: Box::new(config.clone()),
            };

            if let Some(job_queue) = app_context.worker_job_queue.get() {
                match job_queue.submit(job).await {
                    Ok(_) => {
                        debug!("Worker addition job submitted for: {}", worker_url);

                        // Layer 4: Record successful registration from K8s discovery
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_SUCCESS,
                        );

                        // Update workers discovered gauge (using count from initial lock)
                        Metrics::set_discovery_workers_discovered(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            tracked_count,
                        );
                    }
                    Err(e) => {
                        error!(
                            "Failed to submit worker addition job for {}: {}",
                            worker_url, e
                        );

                        // Layer 4: Record failed registration
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_FAILED,
                        );

                        if let Ok(mut tracker) = tracked_pods.lock() {
                            tracker.remove(pod_info);
                        }
                    }
                }
            } else {
                debug!(
                    "JobQueue not initialized, skipping async worker addition for: {}",
                    worker_url
                );
            }
        } else {
            // Pod already tracked - this is a duplicate event
            Metrics::record_discovery_registration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::REGISTRATION_DUPLICATE,
            );
        }
    }
}

async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    app_context: Arc<AppContext>,
    port: u16,
) {
    let worker_url = pod_info.worker_url(port);

    // Remove pod and get remaining count in single lock acquisition
    let (was_tracked, remaining_count) = {
        let mut tracked = match tracked_pods.lock() {
            Ok(tracked) => tracked,
            Err(e) => {
                error!("Failed to acquire tracked_pods lock during deletion: {}", e);
                return;
            }
        };
        let removed = tracked.remove(pod_info);
        (removed, tracked.len())
    };

    if was_tracked {
        info!(
            "Removing pod: {} | type: {:?} | url: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );

        let job = Job::RemoveWorker {
            url: worker_url.clone(),
        };

        if let Some(job_queue) = app_context.worker_job_queue.get() {
            if let Err(e) = job_queue.submit(job).await {
                error!(
                    "Failed to submit worker removal job for {}: {}",
                    worker_url, e
                );
            } else {
                debug!("Submitted worker removal job for {}", worker_url);

                // Layer 4: Record deregistration from K8s pod deletion
                Metrics::record_discovery_deregistration(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    metrics_labels::DEREGISTRATION_POD_DELETED,
                );

                // Update workers discovered gauge (using count from initial lock)
                Metrics::set_discovery_workers_discovered(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    remaining_count,
                );
            }
        } else {
            error!(
                "JobQueue not initialized, cannot remove worker {}",
                worker_url
            );
        }
    } else {
        debug!(
            "Pod deletion event for untracked/already removed pod: {} (type: {:?}). Worker URL: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );
    }
}
