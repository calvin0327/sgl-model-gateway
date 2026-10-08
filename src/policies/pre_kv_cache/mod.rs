//! vLLM event residency + vLLM-rendered tokens + load-aware cost selection.
//! Background work is shared by all policy instances through PolicyRegistry.

mod config;
mod events;
mod index;
mod load;
mod render;
#[cfg(test)]
mod tests;
mod transport;

pub use config::{Config, CostModel, Discovery, EventSource, Profile, TokenSource};
pub use render::KvRequest;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use rand::Rng;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, PowerOfTwoPolicy, SelectWorkerInfo};
use crate::{
    core::{BackendType, Worker, WorkerRegistry, WorkerType},
    tokenizer::TokenizerRegistry,
};
use load::{Credit, Loads, Reservation};

#[derive(Debug, Default)]
pub(super) struct WorkerState {
    binding: Option<Arc<dyn Worker>>,
    stream: events::StreamIndex,
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
        block_size: usize,
        event_stale_ms: u64,
    ) -> Option<usize> {
        let state = self.state.lock();
        let current = state
            .binding
            .as_ref()
            .is_some_and(|bound| Arc::ptr_eq(bound, worker));
        if current
            && !self.stop.load(Ordering::Acquire)
            && !state.stream.unsupported
            && state
                .stream
                .last_event
                .is_some_and(|t| t.elapsed() <= Duration::from_millis(event_stale_ms))
        {
            Some(state.stream.index.matched_keys(&input.keys, block_size))
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
    loads: Loads,
    next_credit: AtomicU64,
    stop: Arc<AtomicBool>,
    started: AtomicBool,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreKvCacheRuntime")
            .field("workers", &self.subscriptions.read().len())
            .finish_non_exhaustive()
    }
}

impl Runtime {
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
        Ok(Arc::new(Self {
            config: Arc::new(config),
            client,
            tokenizer_registry,
            subscriptions: RwLock::new(subscriptions),
            loads: Loads::default(),
            next_credit: AtomicU64::new(1),
            stop: Arc::new(AtomicBool::new(false)),
            started: AtomicBool::new(false),
        }))
    }

    pub fn start(self: &Arc<Self>, registry: &Arc<WorkerRegistry>) {
        if self.started.swap(true, Ordering::AcqRel) {
            return;
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

        request
            .tokens(
                name,
                profile,
                &self.config,
                &self.client,
                &self.tokenizer_registry,
                None,
            )
            .await
            .ok()
            .map(|rendered| rendered.tokens.len())
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
        let cached = subscription.and_then(|subscription| {
            let name = &subscription.source.profile;
            let profile = &self.config.profiles[name];
            if profile.token_source != TokenSource::LocalTokenizer {
                return None;
            }
            let input = request.cached_input(name)?;
            subscription.cached_tokens(
                worker,
                &input,
                profile.block_size,
                self.config.event_stale_ms,
            )
        });
        request.set_selected_cached_tokens(cached);
    }

    fn start_subscription(&self, subscription: &Subscription, registry: &Arc<WorkerRegistry>) {
        let mut tasks = subscription.tasks.lock();
        tasks.push(transport::subscribe(
            subscription.source.clone(),
            self.config.clone(),
            Arc::downgrade(registry),
            subscription.state.clone(),
            subscription.stop.clone(),
        ));
        tasks.push(transport::poll_load(
            subscription.source.clone(),
            self.config.clone(),
            Arc::downgrade(registry),
            self.client.clone(),
            self.loads.clone(),
            subscription.stop.clone(),
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
                let (profile, _) = self
                    .config
                    .profiles
                    .iter()
                    .find(|(_, p)| p.model == worker.model_id())?;
                let source = discovery.source(worker.url(), profile)?;
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
            subscriptions.entry(url.clone()).or_insert_with(|| {
                let subscription = Subscription::new(source, Some(worker));
                self.start_subscription(&subscription, registry);
                tracing::info!(
                    worker_url = url,
                    "pre_kv_cache subscribed discovered worker"
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
                        rendered.insert(source.profile.clone(), result.ok());
                    }
                }
            }
        }
        let started = Instant::now();
        let mut choices = Vec::new();
        for &i in &healthy {
            let worker = &workers[i];
            // Health may have changed while the render HTTP call was in flight.
            if !worker.is_healthy() || !worker.circuit_breaker().can_execute() {
                continue;
            }
            let subscription = subscriptions.get(&i);
            let source = subscription.map(|s| &s.source);
            let profile = source.map(|s| &self.config.profiles[&s.profile]);
            let cost = profile.map(|p| p.cost.clone()).unwrap_or_default();
            let input = source
                .and_then(|s| rendered.get(&s.profile))
                .and_then(|r| r.as_ref());
            let mut cache_state_valid = false;
            let total = input
                .map(|r| r.tokens.len())
                .unwrap_or(cost.mean_uncached_tokens);
            let mut cached = 0;
            if let (Some(_), Some(input), Some(profile)) = (source, input, profile) {
                let subscription = subscription.unwrap();
                if let Some(tokens) = subscription.cached_tokens(
                    worker,
                    input,
                    profile.block_size,
                    self.config.event_stale_ms,
                ) {
                    cached = tokens;
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
        let baseline = choices
            .iter()
            .filter_map(|(i, c, _, _, speed, _)| {
                let s = loads
                    .get(workers[*i].url())?
                    .snapshot
                    .as_ref()
                    .filter(|s| now.duration_since(s.observed) <= stale)?;
                Some(c.cost_ms(0, 0, *speed, Some(s), 0, 0, 0.0))
            })
            .reduce(f64::min)
            .unwrap_or(0.0);
        let mut best: Option<(usize, f64, usize, usize, bool)> = None;
        let mut tied = 0;
        for (i, cost, total, cached, speed, cache_state_valid) in choices {
            let state = loads.entry(workers[i].url().to_string()).or_default();
            let snapshot = state
                .snapshot
                .as_ref()
                .filter(|s| now.duration_since(s.observed) <= stale);
            let pending = state
                .credits
                .values()
                .fold(0usize, |n, c| n.saturating_add(c.tokens));
            let score = cost.cost_ms(
                total,
                cached,
                speed,
                snapshot,
                workers[i].load(),
                pending,
                baseline,
            );
            if best.is_none_or(|b| score < b.1) {
                best = Some((i, score, total, cached, cache_state_valid));
                tied = 1;
            } else if best.is_some_and(|b| (score - b.1).abs() < 1e-9) {
                tied += 1;
                if rand::rng().random_range(0..tied) == 0 {
                    best = Some((i, score, total, cached, cache_state_valid));
                }
            }
        }
        let (selected, score, total, cached, cache_state_valid) = best?;
        if let Some(request) = request {
            let id = self.next_credit.fetch_add(1, Ordering::Relaxed);
            let url = workers[selected].url().to_string();
            loads.entry(url.clone()).or_default().credits.insert(
                id,
                Credit {
                    tokens: total.saturating_sub(cached).max(1),
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
        workers[selected].increment_processed();
        let lookup_elapsed = started.elapsed();
        metrics::histogram!("pre_kv_cache_lookup_seconds").record(lookup_elapsed.as_secs_f64());
        metrics::histogram!("pre_kv_cache_selected_cached_tokens").record(cached as f64);
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
