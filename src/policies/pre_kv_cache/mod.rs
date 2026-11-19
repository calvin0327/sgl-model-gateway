//! vLLM event residency + exact request tokens + load-aware cost selection.
//! Background work is shared by all policy instances through PolicyRegistry.

mod config;
mod diagnostics;
mod events;
mod index;
mod layout;
mod load;
mod metadata;
mod render;
pub(crate) mod settings;
#[cfg(test)]
mod tests;
mod transport;

pub use config::{
    CacheGroup, CacheKind, Config, CostModel, Discovery, EventSource, LayoutPreset, Profile,
    TokenSource,
};
pub use render::KvRequest;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use rand::Rng;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock, Weak,
    },
    time::{Duration, Instant},
};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, PowerOfTwoPolicy, SelectWorkerInfo};
use crate::{
    core::{BackendType, Worker, WorkerRegistry, WorkerType},
    tokenizer::{registry::TokenizerEntry, traits::Tokenizer, TokenizerRegistry},
};
use diagnostics::Diagnostics;
use load::{Credit, Loads, Reservation};

#[derive(Debug, Default)]
pub(super) struct WorkerState {
    binding: Option<Arc<dyn Worker>>,
    stream: events::StreamIndex,
    progress: transport::EventProgress,
    // Metadata belongs to one engine incarnation. A changed layout forces a
    // fresh event connection, so queued events cannot seed the replacement.
    layout_revision: u64,
    layout_checked: Option<Instant>,
    layout_error: Option<String>,
    layout_preset: Option<LayoutPreset>,
    // Automatic defaults belong to the detected tokenizer instance. They must
    // be reconsidered after unloading/replacing that model, even at the same URL.
    layout_tokenizer: Option<Weak<dyn Tokenizer>>,
    requires_layout: bool,
}

impl WorkerState {
    // A preset fixes the layout, never residency or engine identity. Rebinding
    // must start with an empty index even when the URL and preset are unchanged.
    fn bind(&mut self, worker: Option<Arc<dyn Worker>>, profile: &Profile) -> bool {
        let same = match (&self.binding, &worker) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if same {
            return false;
        }
        self.stream = events::StreamIndex::default();
        self.layout_preset = worker.as_ref().and(profile.layout);
        self.stream.layout = self
            .layout_preset
            .map(|preset| Arc::new(preset.runtime_layout()));
        self.layout_checked = None;
        self.layout_error = None;
        self.layout_tokenizer = None;
        self.layout_revision = self.layout_revision.wrapping_add(1);
        self.binding = worker;
        true
    }

    // Repeated detection must preserve residency and sticky event errors. Only
    // a changed model instance/contract starts a new event-index generation.
    fn set_automatic_layout(&mut self, entry: Option<&TokenizerEntry>) -> bool {
        let same = match (&self.layout_tokenizer, entry) {
            (Some(previous), Some(entry)) => previous
                .upgrade()
                .is_some_and(|tokenizer| Arc::ptr_eq(&tokenizer, &entry.tokenizer)),
            (None, None) => true,
            _ => false,
        };
        if same {
            return false;
        }
        self.stream = events::StreamIndex::default();
        self.layout_preset = entry.map(|_| LayoutPreset::DeepseekV41Vllm031Dspark5);
        self.stream.layout = self
            .layout_preset
            .map(|preset| Arc::new(preset.runtime_layout()));
        self.layout_tokenizer = entry.map(|entry| Arc::downgrade(&entry.tokenizer));
        self.layout_checked = None;
        self.layout_error = None;
        self.layout_revision = self.layout_revision.wrapping_add(1);
        true
    }

    // Model recognition selects a known contract; a compatible local Store is
    // additional evidence before reporting any automatic cache estimate.
    fn automatic_events_confirmed(&self) -> bool {
        self.stream.observed_groups.get(&8).is_some_and(|group| {
            group.kind == CacheKind::FullAttention
                && group.block_size == 64
                && group.sliding_window.is_none()
        })
    }

    // A removed tokenizer can still be alive through another alias or an
    // in-flight request. Compare the current registry entry, not just liveness.
    fn automatic_tokenizer_is_current(&self, registry: &TokenizerRegistry, model: &str) -> bool {
        self.layout_tokenizer.as_ref().is_none_or(|previous| {
            let current = registry
                .get_by_name(model)
                .or_else(|| registry.get_by_id(model));
            previous
                .upgrade()
                .zip(current)
                .is_some_and(|(previous, current)| Arc::ptr_eq(&previous, &current.tokenizer))
        })
    }

    // Only dynamic metadata expires. Residency follows events, not idle time;
    // all layouts still require valid events and a matching worker binding.
    fn layout_is_current(&self) -> bool {
        self.stream.layout.is_some()
            && self
                .layout_tokenizer
                .as_ref()
                .is_none_or(|tokenizer| tokenizer.upgrade().is_some())
            && (self.layout_preset.is_some()
                || self
                    .layout_checked
                    .is_some_and(|at| at.elapsed() <= metadata::MAX_AGE))
    }

    // Absence of cache changes does not invalidate confirmed residency. Actual
    // stream failures clear last_event; unsupported formats remain unusable.
    fn event_state_usable(&self) -> bool {
        self.binding.is_some() && !self.stream.unsupported && self.stream.last_event.is_some()
    }
}

/// A discovered registration owns its subscriptions; replacement never inherits KV.
struct Subscription {
    source: EventSource,
    worker: Option<Arc<dyn Worker>>,
    state: Arc<Mutex<WorkerState>>,
    stop: Arc<AtomicBool>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl Subscription {
    // A selected miss is only a recovery hint. Never infer residency from a
    // request or start network work while the scheduler holds its hot-path locks.
    fn request_progress_check(&self) {
        let now = Instant::now();
        let mut state = self.state.lock();
        state.progress.first_cache_miss.get_or_insert(now);
        state.progress.last_cache_miss = Some(now);
    }

    fn new(source: EventSource, worker: Option<Arc<dyn Worker>>) -> Arc<Self> {
        Arc::new(Self {
            source,
            worker,
            state: Arc::new(Mutex::new(WorkerState::default())),
            stop: Arc::new(AtomicBool::new(false)),
            tasks: Mutex::new(Vec::new()),
        })
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        for task in self.tasks.lock().drain(..) {
            task.abort();
        }
        self.state.lock().stream.invalidate();
        metrics::gauge!("pre_kv_cache_index_blocks", "worker" => self.source.worker_url.clone())
            .set(0.0);
    }

    // Use the same residency validity checks for selection and logging.
    fn cached_tokens(
        &self,
        worker: &Arc<dyn Worker>,
        input: &render::Rendered,
        profile: &Profile,
    ) -> Option<usize> {
        let state = self.state.lock();
        let current = state
            .binding
            .as_ref()
            .is_some_and(|bound| Arc::ptr_eq(bound, worker));
        if current
            && !self.stop.load(Ordering::Acquire)
            && state.event_state_usable()
            && (state.stream.layout.is_none() || state.layout_is_current())
            && (!input.requires_group_layout || state.layout_is_current())
            && (state.layout_tokenizer.is_none()
                || (input.requires_group_layout && state.automatic_events_confirmed()))
        {
            if let Some(layout) = &state.stream.layout {
                state.stream.index.matched_keys_runtime(&input.keys, layout)
            } else {
                state.stream.index.matched_keys_layout(
                    &input.keys,
                    profile.block_size,
                    &profile.cache_groups,
                )
            }
        } else {
            None
        }
    }
}

pub struct Runtime {
    pub config: Arc<Config>,
    client: reqwest::Client,
    tokenizer_registry: Arc<TokenizerRegistry>,
    subscriptions: RwLock<HashMap<String, Arc<Subscription>>>,
    weights: RwLock<HashMap<String, settings::Weights>>,
    loads: Loads,
    next_credit: AtomicU64,
    stop: Arc<AtomicBool>,
    started: AtomicBool,
    diagnostics: Arc<Diagnostics>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreKvCacheRuntime")
            .field("workers", &self.subscriptions.read().len())
            .finish_non_exhaustive()
    }
}

impl Runtime {
    /// Report current subscriber evidence without contacting workers or encoding
    /// a request. Observed groups are samples, never proof of a complete layout.
    pub fn status(&self) -> serde_json::Value {
        let subscriptions = self.subscriptions.read();
        let mut workers: Vec<_> = subscriptions.values().map(|subscription| {
            let source = &subscription.source;
            let state = subscription.state.lock();
            let profile = &self.config.profiles[&source.profile];
            let age = state.stream.last_event.map(|at| at.elapsed());
            let mut entry = serde_json::json!({
                "worker_url": source.worker_url,
                "profile": source.profile,
                "model": profile.model,
                "endpoint": source.endpoint,
                "replay_endpoint": source.replay_endpoint,
                "topic": source.topic,
                "bound": state.binding.is_some(),
                "unsupported": state.stream.unsupported,
                "last_error": state.stream.last_error,
                "history_complete": state.stream.history_complete,
                "history_truncated_at": state.stream.history_truncated_at,
                "last_sequence": state.stream.last_seq,
                "last_valid_event_age_ms": age.map(|age| age.as_millis()),
                "event_state_fresh": age.is_some_and(|age| age <= Duration::from_millis(self.config.event_stale_ms)),
                "event_state_usable": !subscription.stop.load(Ordering::Acquire)
                    && state.event_state_usable()
                    && (state.stream.layout.is_none() || state.layout_is_current())
                    && (!state.requires_layout || state.layout_is_current())
                    && state.automatic_tokenizer_is_current(&self.tokenizer_registry, &profile.model)
                    && (state.layout_tokenizer.is_none() || state.automatic_events_confirmed()),
                "resident_blocks": state.stream.index.len(),
                "isolated_hashes": state.stream.index.opaque_len(),
                "isolation_overflow": state.stream.index.isolation_overflow(),
                "skipped_stores": state.stream.skipped_stores,
                "skipped_blocks": state.stream.skipped_blocks,
                "last_skipped_store": state.stream.last_skip,
                "subscription_running": state.progress.running,
                "connection_generation": state.progress.connection_generation,
                "last_connection_event": state.progress.connection_event,
                "last_received_sequence": state.progress.last_received_sequence,
                "last_received_event_age_ms": state.progress.last_received.map(|at| at.elapsed().as_millis()),
                "received_event_frames": state.progress.received_frames,
                "duplicate_event_frames": state.progress.duplicate_frames,
                "event_apply_failures": state.progress.apply_failures,
                "replay_in_progress": state.progress.replay_in_progress,
                "configured_groups": profile.cache_groups,
                "runtime_layout": state.stream.layout.as_deref(),
                "layout_source": state.stream.layout.as_ref().map(|_| if state.layout_tokenizer.is_some() { "automatic_builtin" } else if state.layout_preset.is_some() { "builtin_preset" } else { "vllm_engine" }),
                "layout_preset": state.layout_preset,
                "layout_events_confirmed": state.layout_tokenizer.as_ref().map(|_| state.automatic_events_confirmed()),
                "layout_engine_verified": state.layout_preset.is_none() && state.layout_is_current(),
                "layout_age_ms": state.layout_checked.map(|at| at.elapsed().as_millis()),
                "layout_error": state.layout_error,
                "observed_groups": state.stream.observed_groups.values().collect::<Vec<_>>(),
                "observed_groups_are_complete": false,
                "warnings": self.diagnostics.reasons(&source.worker_url),
            });
            entry["progress_check_in_progress"] = state.progress.progress_check_in_progress.into();
            entry["progress_checks"] = state.progress.progress_checks.into();
            entry["progress_recovered_frames"] = state.progress.progress_recovered_frames.into();
            entry["last_progress_check_age_ms"] = serde_json::json!(state.progress.last_progress_check.map(|at| at.elapsed().as_millis()));
            entry["last_progress_error"] = state.progress.last_progress_error.into();
            entry["history_missing_ancestry"] = state.stream.incomplete().into();
            entry["known_blocks"] = state.stream.index.known_len().into();
            entry["ancestry_evictions"] = state.stream.index.ancestry_evictions.into();
            entry["residency_evictions"] = state.stream.index.residency_evictions.into();
            entry
        }).collect();
        workers.sort_by(|a, b| a["worker_url"].as_str().cmp(&b["worker_url"].as_str()));
        let profiles: std::collections::BTreeMap<_, _> = self
            .config
            .profiles
            .iter()
            .map(|(name, profile)| {
                (
                    name,
                    serde_json::json!({
                        "model": profile.model,
                        "block_size": profile.block_size,
                        "cache_groups": profile.cache_groups,
                        "layout": profile.layout,
                        "token_source": profile.token_source,
                        "weights": self.weights(name),
                        "recent_fallback_reasons": self.diagnostics.reasons(name),
                    }),
                )
            })
            .collect();
        serde_json::json!({
            "profiles": profiles,
            "workers": workers,
            "diagnostics_by_scope": self.diagnostics.by_scope(),
        })
    }

    pub fn new(config: Config, client: reqwest::Client) -> Result<Arc<Self>, String> {
        Self::new_with_tokenizer_registry(config, client, Arc::new(TokenizerRegistry::new()))
    }

    pub fn new_with_tokenizer_registry(
        config: Config,
        client: reqwest::Client,
        tokenizer_registry: Arc<TokenizerRegistry>,
    ) -> Result<Arc<Self>, String> {
        config.validate()?;
        let subscriptions = config
            .workers
            .iter()
            .map(|s| (s.worker_url.clone(), Subscription::new(s.clone(), None)))
            .collect();
        let weights = config
            .profiles
            .iter()
            .map(|(name, profile)| (name.clone(), settings::Weights::from(&profile.cost)))
            .collect();
        Ok(Arc::new(Self {
            config: Arc::new(config),
            client,
            tokenizer_registry,
            subscriptions: RwLock::new(subscriptions),
            weights: RwLock::new(weights),
            loads: Loads::default(),
            next_credit: AtomicU64::new(1),
            stop: Arc::new(AtomicBool::new(false)),
            started: AtomicBool::new(false),
            diagnostics: Arc::new(Diagnostics::default()),
        }))
    }

    pub fn start(self: &Arc<Self>, registry: &Arc<WorkerRegistry>) {
        if self.started.swap(true, Ordering::AcqRel) {
            return;
        }
        for (name, profile) in &self.config.profiles {
            tracing::info!(
                profile = name,
                model = profile.model,
                token_source = ?profile.token_source,
                kv_importance = profile.cost.kv_importance,
                load_importance = profile.cost.load_importance,
                log_decisions = self.config.log_decisions,
                "pre_kv_cache scoring configured"
            );
        }
        for subscription in self.subscriptions.read().values() {
            self.start_subscription(subscription, registry);
        }
        if let Some(discovery) = &self.config.discovery {
            let period = Duration::from_millis(discovery.refresh_ms);
            let runtime = Arc::downgrade(self);
            let registry = Arc::downgrade(registry);
            // Reuse successful worker registration from any source, including K8s.
            // No additional Pod watcher or K8s permissions are needed.
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(period);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    let (Some(runtime), Some(registry)) = (runtime.upgrade(), registry.upgrade())
                    else {
                        break;
                    };
                    if runtime.stop.load(Ordering::Acquire) {
                        break;
                    }
                    runtime.reconcile(&registry);
                }
            });
        }
    }

    /// Reuse the policy's exact local tokenization for length-based pool selection.
    /// Render profiles keep the existing lightweight scheduler path so choosing a
    /// Regular/PD pool never adds a remote HTTP request.
    pub async fn local_request_token_count(
        &self,
        model: &str,
        request: &KvRequest,
    ) -> Option<usize> {
        let mut profiles = self
            .config
            .profiles
            .iter()
            .filter(|(_, profile)| profile.model == model);
        let (name, profile) = profiles.next()?;

        // Automatic discovery already enforces one profile per model. Explicit
        // mappings may be more complex; keep their previous scheduler behavior
        // instead of guessing which profile defines the request tokenization.
        if profile.token_source != TokenSource::LocalTokenizer || profiles.next().is_some() {
            return None;
        }

        let result = request
            .tokens(
                name,
                profile,
                &self.config,
                &self.client,
                &self.tokenizer_registry,
                None,
            )
            .await;
        if let Err(reason) = result.as_ref() {
            self.warn_tokens(name, reason);
        } else {
            self.tokens_available(name);
        }
        result.ok().map(|rendered| rendered.tokens.len())
    }

    // Surface fallback reasons at INFO-level deployments without logging prompts
    // or redoing tokenization. Background event errors use this same throttle.
    fn warn_tokens(&self, profile: &str, reason: &'static str) {
        self.diagnostics.warn(
            profile,
            reason,
            "Exact request tokens unavailable; cache matching falls back to estimates. Check model/profile, tokenizer and chat-template compatibility; local_tokenizer does not fall back to HTTP render",
        );
    }

    // Success proves tokenizer registration and metadata are usable again.
    // Input-specific failures remain recent observations, not model-wide health.
    fn tokens_available(&self, profile: &str) {
        for reason in [
            "local_tokenizer_unavailable",
            "local_tokenizer_metadata",
            "request_tokens_unavailable",
        ] {
            self.diagnostics.recovered(profile, reason);
        }
    }

    fn warn_missing_subscription(&self, worker: &Arc<dyn Worker>) {
        let reason = if self
            .config
            .profiles
            .values()
            .any(|p| p.model == worker.model_id())
        {
            "subscription_unavailable"
        } else {
            "profile_model_mismatch"
        };
        self.diagnostics.warn(
            worker.url(),
            reason,
            &format!(
                "No KV subscription for model={}; check --pre-kv-cache-model matches --served-model-name and event discovery or explicit worker mappings; cache matching is unavailable",
                worker.model_id(),
            ),
        );
    }

    fn cached_tokens(
        &self,
        subscription: &Subscription,
        worker: &Arc<dyn Worker>,
        input: &render::Rendered,
    ) -> Option<usize> {
        let missing_layout = {
            let mut state = subscription.state.lock();
            state.requires_layout |= input.requires_group_layout;
            let profile = &self.config.profiles[&subscription.source.profile];
            !state.automatic_tokenizer_is_current(&self.tokenizer_registry, &profile.model)
                || (input.requires_group_layout && !state.layout_is_current())
        };
        if missing_layout {
            self.diagnostics.warn(
                worker.url(),
                "cache_group_layout_missing",
                "V4.1 cache layout is unavailable or stale; automatic adaptation requires the served model's registered local tokenizer and a supported cache configuration. Inspect /v1/pre-kv-cache/status for layout_error and event warnings",
            );
            return None;
        }
        self.diagnostics
            .recovered(worker.url(), "cache_group_layout_missing");
        let cached = subscription.cached_tokens(
            worker,
            input,
            &self.config.profiles[&subscription.source.profile],
        );
        if cached.is_none() {
            self.diagnostics.warn(
                worker.url(),
                "event_state_unavailable",
                "No valid current KV index for this worker; cache matching falls back to estimates. Check no_kv_events, event/replay errors and subscription startup logs",
            );
        } else {
            self.diagnostics
                .recovered(worker.url(), "event_state_unavailable");
        }
        cached
    }

    /// Observe a single local-tokenizer worker using the existing request tokens.
    /// Keep the fast path free of tokenization, HTTP calls and load reservations.
    pub(crate) fn record_single_worker_cache(&self, worker: &Arc<dyn Worker>, request: &KvRequest) {
        if worker.metadata().runtime_type != BackendType::Vllm
            || matches!(worker.worker_type(), WorkerType::Decode)
        {
            return;
        }
        let subscription = self.subscriptions.read().get(worker.url()).cloned();
        if subscription.is_none() {
            self.warn_missing_subscription(worker);
        }
        let cached = subscription.and_then(|subscription| {
            let name = &subscription.source.profile;
            let profile = &self.config.profiles[name];
            if profile.token_source != TokenSource::LocalTokenizer {
                return None;
            }
            let input = request.cached_input(name).or_else(|| {
                self.warn_tokens(name, "request_tokens_unavailable");
                None
            })?;
            self.tokens_available(name);
            let cached = self.cached_tokens(&subscription, worker, &input);
            if cached.is_none_or(|tokens| tokens == 0) {
                subscription.request_progress_check();
            }
            cached
        });
        request.set_selected_cached_tokens(cached);
        if self.config.log_decisions {
            tracing::info!(
                worker_url = worker.url(),
                reason = "single_candidate",
                candidate_count = 1,
                "pre_kv_cache scoring bypassed"
            );
        }
    }

    fn start_subscription(&self, subscription: &Subscription, registry: &Arc<WorkerRegistry>) {
        let mut tasks = subscription.tasks.lock();
        // The explicit preset replaces metadata discovery, including its HTTP
        // polling. Worker plugins are unnecessary for this fixed deployment.
        if let Some(preset) = self.config.profiles[&subscription.source.profile].layout {
            tracing::info!(
                worker_url = %subscription.source.worker_url,
                layout_preset = preset.name(),
                "pre_kv_cache using configured layout preset; worker engine configuration is not auto-verified; awaiting valid KV events"
            );
        } else {
            tasks.push(metadata::poll(
                subscription.source.clone(),
                self.config.clone(),
                Arc::downgrade(registry),
                self.client.clone(),
                subscription.state.clone(),
                subscription.stop.clone(),
                self.diagnostics.clone(),
                self.tokenizer_registry.clone(),
            ));
        }
        tasks.push(transport::subscribe(
            subscription.source.clone(),
            self.config.clone(),
            Arc::downgrade(registry),
            subscription.state.clone(),
            subscription.stop.clone(),
            self.diagnostics.clone(),
        ));
        tasks.push(transport::poll_load(
            subscription.source.clone(),
            self.config.clone(),
            Arc::downgrade(registry),
            self.client.clone(),
            self.loads.clone(),
            subscription.stop.clone(),
            self.diagnostics.clone(),
        ));
    }

    fn reconcile(&self, registry: &Arc<WorkerRegistry>) {
        let Some(discovery) = &self.config.discovery else {
            return;
        };
        let desired: HashMap<_, _> = registry
            .get_all()
            .into_iter()
            .filter_map(|worker| {
                if !matches!(worker.metadata().runtime_type, BackendType::Vllm)
                    || matches!(worker.worker_type(), WorkerType::Decode)
                {
                    return None;
                }
                let Some((profile, _)) = self
                    .config
                    .profiles
                    .iter()
                    .find(|(_, p)| p.model == worker.model_id()) else {
                        self.warn_missing_subscription(&worker);
                        return None;
                    };
                self.diagnostics.recovered(worker.url(), "profile_model_mismatch");
                let source = discovery.source(worker.url(), profile).or_else(|| {
                    self.diagnostics.warn(worker.url(), "event_address_unavailable", "Cannot derive a ZMQ address from this worker URL; configure an explicit event source");
                    None
                })?;
                self.diagnostics.recovered(worker.url(), "event_address_unavailable");
                Some((worker.url().to_string(), (worker, source)))
            })
            .collect();
        let mut subscriptions = self.subscriptions.write();
        if self.stop.load(Ordering::Acquire) {
            return;
        }
        subscriptions.retain(|url, subscription| {
            // Explicit mappings retain their existing behavior and override discovery.
            let Some(bound) = &subscription.worker else {
                return true;
            };
            let keep = desired
                .get(url)
                .is_some_and(|(worker, _)| Arc::ptr_eq(bound, worker));
            if !keep {
                subscription.stop();
                self.loads.lock().remove(url);
                tracing::info!(
                    worker_url = url,
                    "pre_kv_cache removed discovered subscription"
                );
            }
            keep
        });
        for (url, (worker, source)) in desired {
            self.diagnostics.recovered(&url, "subscription_unavailable");
            subscriptions.entry(url.clone()).or_insert_with(|| {
                let subscription = Subscription::new(source, Some(worker));
                self.start_subscription(&subscription, registry);
                tracing::info!(
                    worker_url = url,
                    endpoint = subscription.source.endpoint,
                    topic = subscription.source.topic,
                    "pre_kv_cache subscription started; awaiting valid KV events"
                );
                subscription
            });
        }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        for subscription in self.subscriptions.write().values() {
            subscription.stop();
        }
    }

    async fn select(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let request = info.kv_request;
        // Hold no registry/subscription lock across render HTTP I/O.
        let subscriptions: HashMap<_, _> = {
            let sources = self.subscriptions.read();
            healthy
                .iter()
                .filter_map(|&i| sources.get(workers[i].url()).map(|s| (i, s.clone())))
                .collect()
        };
        // Derive exact request tokens only when this policy is reached, and reuse
        // the result on retries of the same request.
        let mut rendered = HashMap::new();
        if let Some(request) = request {
            for &i in &healthy {
                if !matches!(workers[i].metadata().runtime_type, BackendType::Vllm)
                    || matches!(workers[i].worker_type(), WorkerType::Decode)
                {
                    continue;
                }
                if let Some(subscription) = subscriptions.get(&i) {
                    let source = &subscription.source;
                    if !rendered.contains_key(&source.profile) {
                        let mut profile = self.config.profiles[&source.profile].clone();
                        // Reuse backend authentication only when rendering on that backend.
                        let api_key = if profile.render_url.is_empty() {
                            profile.render_url = workers[i].url().to_string();
                            workers[i].api_key().as_deref()
                        } else {
                            None
                        };
                        let result = request
                            .tokens(
                                &source.profile,
                                &profile,
                                &self.config,
                                &self.client,
                                &self.tokenizer_registry,
                                api_key,
                            )
                            .await;
                        if let Err(reason) = result.as_ref() {
                            self.warn_tokens(&source.profile, reason);
                        } else {
                            self.tokens_available(&source.profile);
                        }
                        rendered.insert(source.profile.clone(), result.ok());
                    }
                } else {
                    self.warn_missing_subscription(&workers[i]);
                }
            }
        } else {
            self.diagnostics.warn(
                "request",
                "request_tokens_unavailable",
                "No request payload available to pre_kv_cache; using estimated uncached work",
            );
        }
        let started = Instant::now();
        // One snapshot per selection keeps every candidate on the same tuning,
        // even when the admin endpoint updates both weights during a request.
        let weights = self.weights.read().clone();
        let mut choices = Vec::new();
        for &i in &healthy {
            let worker = &workers[i];
            // Health may have changed while the render HTTP call was in flight.
            if !worker.is_healthy() || !worker.circuit_breaker().can_execute() {
                continue;
            }
            let subscription = subscriptions.get(&i);
            let source = subscription.map(|s| &s.source);
            // Discovery can lag registration. Honor the model's configured
            // weights even while its cache subscription is still unavailable;
            // ambiguous explicit profiles keep the conservative default.
            let profile = source
                .map(|s| (&s.profile, &self.config.profiles[&s.profile]))
                .or_else(|| {
                    let mut matching = self
                        .config
                        .profiles
                        .iter()
                        .filter(|(_, p)| p.model == worker.model_id());
                    let profile = matching.next()?;
                    matching.next().is_none().then_some(profile)
                });
            let mut cost = profile.map(|(_, p)| p.cost.clone()).unwrap_or_default();
            if let Some((name, _)) = profile {
                if let Some(weights) = weights.get(name) {
                    weights.apply(&mut cost);
                }
            }
            let input = source
                .and_then(|s| rendered.get(&s.profile))
                .and_then(|r| r.as_ref());
            let mut cache_state_valid = false;
            let total = input
                .map(|r| r.tokens.len())
                .unwrap_or(cost.mean_uncached_tokens);
            let mut cached = 0;
            let mut saved = 0;
            if let (Some(subscription), Some(input)) = (subscription, input) {
                if let Some(tokens) = self.cached_tokens(subscription, worker, input) {
                    cached = tokens;
                    // Bounded SWA replay reports a cache hit but recomputes its
                    // tail. Charge that work in both the score and reservation.
                    saved = cached.saturating_sub(
                        subscription
                            .state
                            .lock()
                            .stream
                            .layout
                            .as_ref()
                            .map_or(0, |layout| layout.prefix_replay_tokens),
                    );
                    cache_state_valid = true;
                } else {
                    metrics::counter!("pre_kv_cache_fallback_total", "reason" => "event_state_unavailable").increment(1);
                }
            }
            choices.push((
                i,
                cost,
                total,
                cached,
                saved,
                source.map(|s| s.speed).unwrap_or(1.0),
                cache_state_valid,
            ));
        }
        // Hold the load lock through selection and credit: concurrent requests
        // must see prior picks even before the next backend metrics sample.
        let mut loads = self.loads.lock();
        for (i, ..) in &choices {
            loads
                .entry(workers[*i].url().to_string())
                .or_default()
                .bind(&workers[*i]);
        }
        let now = Instant::now();
        let stale = Duration::from_millis(self.config.load_stale_ms);
        for state in loads.values_mut() {
            state
                .credits
                .retain(|_, c| now.duration_since(c.created) <= stale);
        }
        // Missing telemetry must not make a directly loaded backend look as
        // idle as the least-busy peer. Use the largest last-observed cost as a
        // fallback estimate; fresh snapshots still use their measured values.
        let baseline = choices
            .iter()
            .filter_map(|(i, c, ..)| {
                let s = loads.get(workers[*i].url())?.snapshot.as_ref()?;
                Some(c.backend_load_ms(Some(s), 0, 0.0))
            })
            .reduce(f64::max)
            .unwrap_or(0.0);
        let mut best: Option<(usize, f64, usize, usize, usize, bool)> = None;
        let mut tied = 0;
        // Copy diagnostic values while holding the load snapshot stable, then
        // emit logs after releasing the selection/reservation lock.
        let mut decision_logs = Vec::new();
        let mut load_availability = Vec::new();
        for (i, cost, total, cached, saved, speed, cache_state_valid) in choices {
            let state = loads.entry(workers[i].url().to_string()).or_default();
            let snapshot = state
                .snapshot
                .as_ref()
                .filter(|s| now.duration_since(s.observed) <= stale);
            let pending = state
                .credits
                .values()
                .fold(0usize, |n, c| n.saturating_add(c.tokens));
            let local_load = workers[i].load();
            let breakdown =
                cost.cost_breakdown(total, saved, speed, snapshot, local_load, pending, baseline);
            let score = breakdown.total_ms;
            load_availability.push((i, snapshot.is_some()));
            if self.config.log_decisions {
                decision_logs.push((
                    i,
                    cost,
                    breakdown,
                    snapshot.cloned(),
                    local_load,
                    pending,
                    cache_state_valid,
                    cached,
                ));
            }
            if best.is_none_or(|b| score < b.1) {
                best = Some((i, score, total, cached, saved, cache_state_valid));
                tied = 1;
            } else if best.is_some_and(|b| (score - b.1).abs() < 1e-9) {
                tied += 1;
                if rand::rng().random_range(0..tied) == 0 {
                    best = Some((i, score, total, cached, saved, cache_state_valid));
                }
            }
        }
        let (selected, score, total, cached, saved, cache_state_valid) = best?;
        if let Some(request) = request {
            let id = self.next_credit.fetch_add(1, Ordering::Relaxed);
            let url = workers[selected].url().to_string();
            loads.entry(url.clone()).or_default().credits.insert(
                id,
                Credit {
                    tokens: total.saturating_sub(saved).max(1),
                    created: now,
                },
            );
            // The request owns cleanup for failed attempts and cancellation.
            request
                .reservations
                .lock()
                .push(Reservation::new(&self.loads, url, id));
        }
        drop(loads);
        for (i, available) in load_availability {
            if available {
                self.diagnostics
                    .recovered(workers[i].url(), "load_state_unavailable");
            } else {
                self.diagnostics.warn(workers[i].url(), "load_state_unavailable", "Backend load sample missing or stale; using Router in-flight requests and the largest last-observed peer cost; direct backend traffic remains unmeasured");
            }
        }
        let candidate_count = decision_logs.len();
        for (i, cost, score, snapshot, local_load, pending, cache_state_valid, cached) in
            decision_logs
        {
            tracing::info!(
                worker_url = workers[i].url(),
                selected = i == selected,
                candidate_count,
                cache_state_valid,
                cached_tokens = cached,
                kv_importance = cost.kv_importance,
                load_importance = cost.load_importance,
                load_source = if snapshot.is_some() {
                    "backend"
                } else {
                    "local_fallback"
                },
                waiting_requests = snapshot.as_ref().map(|s| s.waiting),
                running_requests = snapshot.as_ref().map(|s| s.running),
                kv_usage_ratio = snapshot.as_ref().and_then(|s| s.kv_usage),
                load_sample_age_ms = snapshot
                    .as_ref()
                    .map(|s| now.duration_since(s.observed).as_secs_f64() * 1000.0),
                local_in_flight = local_load,
                pending_tokens = pending,
                uncached_prefill_ms = score.prefill_ms,
                backend_or_fallback_load_ms = score.load_ms,
                reservation_ms = score.reservation_ms,
                weighted_kv_ms = score.weighted_kv_ms,
                weighted_load_ms = score.weighted_load_ms,
                estimated_cost_ms = score.total_ms,
                "pre_kv_cache candidate score"
            );
        }
        workers[selected].increment_processed();
        if (!cache_state_valid || cached == 0)
            && subscriptions
                .get(&selected)
                .is_some_and(|s| rendered.get(&s.source.profile).is_some_and(Option::is_some))
        {
            subscriptions[&selected].request_progress_check();
        }
        let lookup_elapsed = started.elapsed();
        metrics::histogram!("pre_kv_cache_lookup_seconds").record(lookup_elapsed.as_secs_f64());
        // Unknown measurements must not depress the measured cache histogram.
        // Preserve an explicit denominator for availability and coverage charts.
        metrics::counter!("pre_kv_cache_selected_cache_state_total", "state" => if cache_state_valid { "known" } else { "unknown" }).increment(1);
        if cache_state_valid {
            metrics::histogram!("pre_kv_cache_selected_cached_tokens").record(cached as f64);
        }
        metrics::histogram!("pre_kv_cache_selected_cost_ms").record(score);
        metrics::counter!("pre_kv_cache_selections_total", "worker" => workers[selected].url().to_string()).increment(1);
        // Record only coverage measured with exact tokens and a valid event
        // index. A fallback is unknown, not a measured zero-hit request.
        if let Some(request) = request {
            request.set_selected_cached_tokens(cache_state_valid.then_some(cached));
        }
        tracing::debug!(
            worker_url = workers[selected].url(),
            cached_tokens = cached,
            estimated_cost_ms = score,
            "pre_kv_cache selected worker"
        );
        Some(selected)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Debug, Default)]
pub struct PreKvCachePolicy {
    runtime: OnceLock<Arc<Runtime>>,
    fallback: PowerOfTwoPolicy,
}

impl PreKvCachePolicy {
    pub fn bind(&self, runtime: Arc<Runtime>) {
        let _ = self.runtime.set(runtime);
    }
}

#[async_trait]
impl LoadBalancingPolicy for PreKvCachePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        if workers
            .iter()
            .all(|w| matches!(w.worker_type(), WorkerType::Decode))
        {
            return self.fallback.select_worker(workers, info).await;
        }
        match self.runtime.get() {
            Some(runtime) => runtime.select(workers, info).await,
            None => self.fallback.select_worker(workers, info).await,
        }
    }
    fn name(&self) -> &'static str {
        "pre_kv_cache"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
