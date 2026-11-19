use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        watcher::{watcher, Config, Event},
        WatchStreamExt,
    },
    Client,
};
use rustls;
use tokio::{sync::oneshot, task, time};
use tracing::{error, info, warn};

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

const DISCOVERY_OWNER_LABEL: &str = "sgl-model-gateway/discovery-pod";
type PodKey = (String, String);

// Readiness and status are observations, not registration identity. Keep the
// last registered endpoint so a deletion tombstone needs only namespace/name/UID.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Registration {
    uid: String,
    ip: String,
    pod_type: Option<PodType>,
    bootstrap_port: Option<u16>,
}

impl Registration {
    fn url(&self, port: u16) -> String {
        format!("http://{}:{}", self.ip, port)
    }

    fn pod_info(&self, name: &str) -> PodInfo {
        PodInfo {
            name: name.to_string(),
            ip: self.ip.clone(),
            status: "Running".into(),
            is_ready: true,
            pod_type: self.pod_type.clone(),
            bootstrap_port: self.bootstrap_port,
        }
    }
}

struct PendingChange {
    target: Option<Registration>,
    completion: oneshot::Receiver<Result<String, String>>,
}

#[derive(Default)]
struct DiscoveredPod {
    uid: String,
    desired: Option<Registration>,
    registered: Option<Registration>,
    pending: Option<PendingChange>,
    retry_at: Option<Instant>,
}

impl DiscoveredPod {
    // Only one operation per Pod may run at a time. Replacement always removes
    // the old registration before adding the new one, even at the same URL.
    fn next_change(&self, now: Instant) -> Option<Option<Registration>> {
        if self.pending.is_some()
            || self.retry_at.is_some_and(|deadline| now < deadline)
            || self.registered == self.desired
        {
            return None;
        }
        Some(if self.registered.is_some() {
            None
        } else {
            self.desired.clone()
        })
    }
}

#[derive(Default)]
struct DiscoveryState {
    pods: HashMap<PodKey, DiscoveredPod>,
    listed: HashSet<PodKey>,
}

impl DiscoveryState {
    fn key(pod: &Pod) -> Option<PodKey> {
        Some((
            pod.metadata.namespace.clone().unwrap_or_default(),
            pod.metadata.name.clone()?,
        ))
    }

    fn observe(&mut self, event: Event<Pod>, config: &ServiceDiscoveryConfig) {
        match event {
            Event::Init => self.listed.clear(),
            Event::InitApply(pod) => {
                if let Some(key) = Self::key(&pod) {
                    self.listed.insert(key);
                }
                self.apply(pod, config);
            }
            Event::InitDone => {
                // A complete relist repairs deletes missed during disconnection.
                // Never remove on a partial/failed list, or touch manual workers.
                for (key, state) in &mut self.pods {
                    if !self.listed.contains(key) {
                        state.desired = None;
                    }
                }
            }
            Event::Apply(pod) => self.apply(pod, config),
            Event::Delete(pod) => {
                let Some(key) = Self::key(&pod) else { return };
                if let Some(state) = self.pods.get_mut(&key) {
                    // A delayed delete for the old incarnation must not delete
                    // a replacement with the same namespace/name.
                    if pod
                        .metadata
                        .uid
                        .as_ref()
                        .is_none_or(|uid| uid == &state.uid)
                    {
                        state.desired = None;
                    }
                }
            }
        }
    }

    fn apply(&mut self, pod: Pod, config: &ServiceDiscoveryConfig) {
        let Some(key) = Self::key(&pod) else { return };
        let selected = PodInfo::should_include(&pod, config);
        if !selected && !self.pods.contains_key(&key) {
            return;
        }
        let uid = pod
            .metadata
            .uid
            .clone()
            .unwrap_or_else(|| format!("{}/{}", key.0, key.1));
        let state = self.pods.entry(key).or_default();
        if uid != state.uid {
            state.uid = uid.clone();
            state.desired = None;
        }
        if !selected || pod.metadata.deletion_timestamp.is_some() {
            state.desired = None;
            return;
        }
        // Keep the existing readiness semantics: transient unready updates do
        // not unregister a live Pod; health checks control its eligibility.
        if let Some(info) = PodInfo::from_pod(&pod, Some(config)).filter(PodInfo::is_healthy) {
            state.desired = Some(Registration {
                uid,
                ip: info.ip,
                pod_type: info.pod_type,
                bootstrap_port: info.bootstrap_port,
            });
        }
    }

    async fn reconcile(&mut self, context: &Arc<AppContext>, config: &ServiceDiscoveryConfig) {
        let Some(queue) = context.worker_job_queue.get() else {
            return;
        };
        let retry_delay = config.check_interval.max(Duration::from_secs(1));
        for (key, state) in &mut self.pods {
            if let Some(pending) = &mut state.pending {
                let result = match pending.completion.try_recv() {
                    Ok(result) => Some(result),
                    Err(oneshot::error::TryRecvError::Empty) => None,
                    Err(oneshot::error::TryRecvError::Closed) => {
                        Some(Err("job completion channel closed".into()))
                    }
                };
                if let Some(result) = result {
                    let pending = state.pending.take().unwrap();
                    match result {
                        Ok(_) => {
                            state.registered = pending.target;
                            state.retry_at = None;
                        }
                        Err(error) => {
                            // Registration may fail after inserting the worker.
                            // Retain ownership so a subsequent delete cleans it up.
                            if let Some(target) = pending.target {
                                if has_owned_worker(context, &target, config.port) {
                                    state.registered = Some(target);
                                }
                            }
                            warn!(pod = key.1, %error, "Discovery job failed; retaining state for reconciliation");
                            state.retry_at = Some(Instant::now() + retry_delay);
                        }
                    }
                }
            }
            let Some(target) = state.next_change(Instant::now()) else {
                continue;
            };
            let job = if let Some(registration) = &target {
                let url = registration.url(config.port);
                // Discovery must not adopt/delete an existing static or API
                // registration, or overlap a retiring Pod using the same URL.
                if has_worker_at_url(context, &url) {
                    state.retry_at = Some(Instant::now() + retry_delay);
                    continue;
                }
                info!(pod = key.1, worker_url = url, "Adding discovered pod");
                let info = registration.pod_info(&key.1);
                let mut worker_config = worker_config(&info, context, config.port);
                worker_config
                    .labels
                    .insert(DISCOVERY_OWNER_LABEL.into(), registration.uid.clone());
                Job::AddWorker {
                    config: Box::new(worker_config),
                }
            } else {
                let registered = state.registered.as_ref().unwrap();
                if !has_owned_worker(context, registered, config.port) {
                    state.registered = None;
                    continue;
                }
                let url = registered.url(config.port);
                info!(pod = key.1, worker_url = url, "Removing discovered pod");
                Job::RemoveWorker { url }
            };
            match queue.submit_with_completion(job).await {
                Ok(completion) => {
                    if target.is_some() {
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_SUCCESS,
                        );
                    } else {
                        Metrics::record_discovery_deregistration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::DEREGISTRATION_POD_DELETED,
                        );
                    }
                    state.pending = Some(PendingChange { target, completion });
                }
                Err(error) => {
                    warn!(pod = key.1, %error, "Discovery submission failed; will retry");
                    state.retry_at = Some(Instant::now() + retry_delay);
                }
            }
        }
        self.pods.retain(|_, state| {
            state.desired.is_some() || state.registered.is_some() || state.pending.is_some()
        });
        Metrics::set_discovery_workers_discovered(
            metrics_labels::DISCOVERY_KUBERNETES,
            self.pods
                .values()
                .filter(|state| state.registered.is_some())
                .count(),
        );
    }
}

fn matches_worker_url(actual: &str, base: &str) -> bool {
    actual == base
        || actual
            .strip_prefix(base)
            .is_some_and(|suffix| suffix.starts_with('@'))
}

fn has_worker_at_url(context: &AppContext, url: &str) -> bool {
    context
        .worker_registry
        .get_all()
        .iter()
        .any(|worker| matches_worker_url(worker.url(), url))
}

fn has_owned_worker(context: &AppContext, registration: &Registration, port: u16) -> bool {
    let url = registration.url(port);
    context.worker_registry.get_all().iter().any(|worker| {
        matches_worker_url(worker.url(), &url)
            && worker.metadata().labels.get(DISCOVERY_OWNER_LABEL) == Some(&registration.uid)
    })
}

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.enabled {
        return Err(kube::Error::Api(kube::error::ErrorResponse {
            status: "Disabled".into(),
            message: "Service discovery is disabled".into(),
            reason: "ConfigurationError".into(),
            code: 400,
        }));
    }
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = Client::try_default().await?;
    let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
        Api::namespaced(client, namespace)
    } else {
        Api::all(client)
    };
    info!(regular = ?config.selector, prefill = ?config.prefill_selector,
        decode = ?config.decode_selector, "Starting K8s service discovery");
    Ok(task::spawn(async move {
        let mut state = DiscoveryState::default();
        let mut tick = time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        // kube-runtime handles reconnect/backoff and emits InitDone only after
        // a complete list. Keep processing pending removals during watch errors.
        loop {
            let stream = watcher(pods.clone(), Config::default()).default_backoff();
            futures::pin_mut!(stream);
            loop {
                tokio::select! {
                    event = stream.next() => match event {
                        Some(Ok(event)) => state.observe(event, &config),
                        Some(Err(error)) => warn!(%error, "Kubernetes watch error; awaiting recovery"),
                        None => break,
                    },
                    _ = tick.tick() => {},
                }
                state.reconcile(&app_context, &config).await;
            }
            error!("Kubernetes watcher ended; restarting");
            time::sleep(Duration::from_secs(1)).await;
        }
    }))
}

fn worker_config(pod_info: &PodInfo, app_context: &AppContext, port: u16) -> WorkerConfigRequest {
    let worker_url = pod_info.worker_url(port);
    let worker_type = match &pod_info.pod_type {
        Some(PodType::Prefill) => Some("prefill".to_string()),
        Some(PodType::Decode) => Some("decode".to_string()),
        Some(PodType::Regular) => Some("regular".to_string()),
        None => unreachable!("only selected pods are registered"),
    };

    let bootstrap_port = match &pod_info.pod_type {
        Some(PodType::Prefill) => pod_info.bootstrap_port,
        _ => None,
    };

    let preferred_context_tokens = match &pod_info.pod_type {
        Some(PodType::Regular) => app_context.config.workers.regular_preferred_context_tokens,
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

    WorkerConfigRequest {
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
        health_check_interval_secs: app_context.config.workers.health_check.check_interval_secs,
        health_success_threshold: app_context.config.workers.health_check.success_threshold,
        health_failure_threshold: app_context.config.workers.health_check.failure_threshold,
        disable_health_check: app_context.config.workers.health_check.disable_health_check,
        max_connection_attempts: app_context.config.workers.health_check.success_threshold * 20,
        dp_aware: app_context.config.routing.dp_aware,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> ServiceDiscoveryConfig {
        ServiceDiscoveryConfig {
            selector: HashMap::from([("app".into(), "worker".into())]),
            ..Default::default()
        }
    }

    fn pod(namespace: &str, uid: &str, ip: &str) -> Pod {
        serde_json::from_value(json!({
            "metadata": {"namespace": namespace, "name": "worker", "uid": uid,
                         "labels": {"app": "worker"}},
            "status": {"phase": "Running", "podIP": ip,
                       "conditions": [{"type": "Ready", "status": "True"}]}
        }))
        .unwrap()
    }

    fn key() -> PodKey {
        ("default".into(), "worker".into())
    }

    fn registered() -> (DiscoveryState, Pod) {
        let pod = pod("default", "old", "10.0.0.1");
        let mut state = DiscoveryState::default();
        state.observe(Event::Apply(pod.clone()), &config());
        let worker = state.pods.get_mut(&key()).unwrap();
        worker.registered = worker.desired.clone();
        (state, pod)
    }

    #[test]
    fn delete_without_labels_status_or_ip_removes_saved_endpoint() {
        let (mut state, mut pod) = registered();
        pod.metadata.labels = None;
        pod.status = None;
        state.observe(Event::Delete(pod), &config());
        let worker = &state.pods[&key()];
        assert_eq!(worker.next_change(Instant::now()), Some(None));
        assert_eq!(
            worker.registered.as_ref().unwrap().url(8000),
            "http://10.0.0.1:8000"
        );
    }

    #[test]
    fn terminating_unready_pod_is_removed_before_final_delete() {
        let (mut state, mut pod) = registered();
        pod.status = None;
        pod.metadata.deletion_timestamp =
            Some(serde_json::from_value(json!("2026-10-15T00:00:00Z")).unwrap());
        state.observe(Event::Apply(pod), &config());
        assert_eq!(state.pods[&key()].next_change(Instant::now()), Some(None));
    }

    #[test]
    fn readiness_change_and_duplicate_apply_do_not_reregister() {
        let (mut state, mut pod) = registered();
        state.observe(Event::Apply(pod.clone()), &config());
        pod.status.as_mut().unwrap().conditions = None;
        state.observe(Event::Apply(pod), &config());
        assert_eq!(state.pods[&key()].next_change(Instant::now()), None);
    }

    #[test]
    fn selector_removal_unregisters_previously_selected_pod() {
        let (mut state, mut pod) = registered();
        pod.metadata.labels = None;
        state.observe(Event::Apply(pod), &config());
        assert_eq!(state.pods[&key()].next_change(Instant::now()), Some(None));
    }

    #[test]
    fn completed_relist_repairs_missed_delete_but_partial_list_does_not() {
        let (mut state, pod) = registered();
        state.observe(Event::Init, &config());
        assert!(state.pods[&key()].desired.is_some());
        state.observe(Event::InitApply(pod), &config());
        state.observe(Event::InitDone, &config());
        assert_eq!(state.pods[&key()].next_change(Instant::now()), None);
        state.observe(Event::Init, &config());
        assert!(state.pods[&key()].desired.is_some());
        state.observe(Event::InitDone, &config());
        assert_eq!(state.pods[&key()].next_change(Instant::now()), Some(None));
    }

    #[test]
    fn namespaced_identity_and_delayed_old_uid_delete_preserve_replacement() {
        let (mut state, old) = registered();
        state.observe(Event::Apply(pod("other", "other", "10.0.0.3")), &config());
        state.observe(Event::Apply(pod("default", "new", "10.0.0.2")), &config());
        state.observe(Event::Delete(old), &config());
        assert_eq!(state.pods.len(), 2);
        let worker = state.pods.get_mut(&key()).unwrap();
        assert_eq!(worker.desired.as_ref().unwrap().uid, "new");
        assert_eq!(worker.next_change(Instant::now()), Some(None));
        worker.registered = None; // Successful removal permits the new incarnation.
        assert_eq!(
            worker.next_change(Instant::now()),
            Some(worker.desired.clone())
        );
    }

    #[test]
    fn delete_during_add_waits_for_completion_then_removes() {
        let (mut state, pod) = registered();
        let worker = state.pods.get_mut(&key()).unwrap();
        let target = worker.registered.take();
        let (tx, completion) = oneshot::channel();
        worker.pending = Some(PendingChange { target, completion });
        state.observe(Event::Delete(pod), &config());
        let worker = state.pods.get_mut(&key()).unwrap();
        assert_eq!(worker.next_change(Instant::now()), None);
        tx.send(Ok("added".into())).unwrap();
        let mut pending = worker.pending.take().unwrap();
        assert!(pending.completion.try_recv().unwrap().is_ok());
        worker.registered = pending.target;
        assert_eq!(worker.next_change(Instant::now()), Some(None));
    }

    #[test]
    fn failed_removal_is_retained_for_retry_without_another_pod_event() {
        let (mut state, pod) = registered();
        state.observe(Event::Delete(pod), &config());
        let worker = state.pods.get_mut(&key()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        worker.retry_at = Some(deadline);
        assert_eq!(worker.next_change(Instant::now()), None);
        assert_eq!(worker.next_change(deadline), Some(None));
    }

    #[test]
    fn regular_prefill_decode_selectors_and_bootstrap_are_preserved() {
        let mut config = config();
        config
            .prefill_selector
            .insert("role".into(), "prefill".into());
        config
            .decode_selector
            .insert("role".into(), "decode".into());
        for (role, expected) in [
            ("regular", PodType::Regular),
            ("prefill", PodType::Prefill),
            ("decode", PodType::Decode),
        ] {
            let mut pod = pod("default", role, "10.0.0.1");
            pod.metadata
                .labels
                .as_mut()
                .unwrap()
                .insert("role".into(), role.into());
            pod.metadata.annotations =
                Some([(config.bootstrap_port_annotation.clone(), "8998".into())].into());
            let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
            assert_eq!(
                info.bootstrap_port,
                if role == "prefill" { Some(8998) } else { None }
            );
            assert_eq!(info.pod_type, Some(expected));
        }
    }

    #[test]
    fn dp_worker_url_matching_does_not_match_neighbor_addresses() {
        assert!(matches_worker_url(
            "http://10.0.0.1:8000@0",
            "http://10.0.0.1:8000"
        ));
        assert!(!matches_worker_url(
            "http://10.0.0.1:80001",
            "http://10.0.0.1:8000"
        ));
    }
}
