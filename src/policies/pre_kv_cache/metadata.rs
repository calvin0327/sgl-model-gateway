//! Resolve cache layouts off the request path. Recognized V4.1 deployments use
//! the measured built-in contract; other engines retain metadata discovery.
//! Automatic selection and observed events do not prove full engine settings.

use parking_lot::Mutex;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use super::{
    config::{Config, EventSource, Profile, TokenSource},
    diagnostics::Diagnostics,
    events::StreamIndex,
    layout::RuntimeLayout,
    render::limited_body,
    WorkerState,
};
use crate::{
    core::{BackendType, WorkerRegistry},
    tokenizer::{registry::TokenizerEntry, TokenizerRegistry},
};

pub(super) const MAX_AGE: Duration = Duration::from_secs(30);

// Use the served model's own tokenizer, never an arbitrary profile name or the
// only tokenizer currently loaded in a multi-model gateway. Late registration
// stays pending and is retried without probing an unavailable Worker extension.
pub(super) fn automatic_tokenizer(
    profile: &Profile,
    registry: &TokenizerRegistry,
) -> Result<Option<TokenizerEntry>, &'static str> {
    if profile.layout.is_some()
        || !profile.cache_groups.is_empty()
        || profile.token_source != TokenSource::LocalTokenizer
    {
        return Ok(None);
    }
    let entry = registry
        .get_by_name(&profile.model)
        .or_else(|| registry.get_by_id(&profile.model))
        .ok_or("local_tokenizer_unavailable")?;
    if !super::render::is_deepseek_v41(&entry)? {
        return Ok(None);
    }
    // A different block size has no built-in adapter. Preserve authoritative
    // engine discovery instead of preventing an otherwise supported layout.
    if profile.block_size != 64 {
        return Ok(None);
    }
    Ok(Some(entry))
}

// Install metadata atomically with an empty index when its engine or contract
// changes. In-flight replay checks this revision before committing any blocks.
fn install(state: &mut WorkerState, layout: RuntimeLayout) -> bool {
    let changed = state.stream.layout.as_deref() != Some(&layout) || state.layout_checked.is_none();
    if changed {
        state.stream = StreamIndex::default();
        state.stream.layout = Some(Arc::new(layout));
        state.layout_revision = state.layout_revision.wrapping_add(1);
    }
    state.layout_checked = Some(Instant::now());
    state.layout_preset = None;
    state.layout_tokenizer = None;
    state.layout_error = None;
    changed
}

pub(super) fn poll(
    source: EventSource,
    cfg: Arc<Config>,
    registry: Weak<WorkerRegistry>,
    client: reqwest::Client,
    state: Arc<Mutex<WorkerState>>,
    stop: Arc<AtomicBool>,
    diagnostics: Arc<Diagnostics>,
    tokenizer_registry: Arc<TokenizerRegistry>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        while !stop.load(Ordering::Acquire) {
            interval.tick().await;
            let Some(registry) = registry.upgrade() else {
                break;
            };
            let Some(worker) = registry.get_by_url(&source.worker_url) else {
                continue;
            };
            if stop.load(Ordering::Acquire) || worker.metadata().runtime_type != BackendType::Vllm {
                continue;
            }
            let profile = &cfg.profiles[&source.profile];
            let automatic = automatic_tokenizer(profile, &tokenizer_registry);
            {
                let mut state = state.lock();
                // Detection may read model metadata. Check the registration again
                // before changing the index, just as for an HTTP metadata reply.
                if stop.load(Ordering::Acquire)
                    || !registry
                        .get_by_url(&source.worker_url)
                        .is_some_and(|current| Arc::ptr_eq(&worker, &current))
                {
                    continue;
                }
                state.bind(Some(worker.clone()), profile);
                match automatic {
                    Ok(Some(entry)) => {
                        if state.set_automatic_layout(Some(&entry)) {
                            state.requires_layout = true;
                            metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone()).set(0.0);
                            tracing::info!(
                                worker_url = %source.worker_url,
                                model = %profile.model,
                                layout_preset = "deepseek_v41_vllm031_dspark5",
                                lookup_groups = "[[8]]",
                                prefix_replay_tokens = 128,
                                "pre_kv_cache automatically selected known V4.1 layout from local model configuration; awaiting compatible KV events; full worker engine settings are not verified"
                            );
                        }
                        for reason in [
                            "cache_layout_auto_unavailable",
                            "cache_group_metadata_unavailable",
                            "cache_group_layout_missing",
                        ] {
                            diagnostics.recovered(&source.worker_url, reason);
                        }
                        // Keep checking local tokenizer identity on later ticks,
                        // but never poll a metadata bridge for this built-in mode.
                        continue;
                    }
                    Ok(None) => {
                        if state.set_automatic_layout(None) {
                            metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone()).set(0.0);
                            tracing::info!(worker_url = %source.worker_url, "pre_kv_cache automatic layout withdrawn after tokenizer change; discovering engine metadata");
                        }
                        diagnostics.recovered(&source.worker_url, "cache_layout_auto_unavailable");
                    }
                    Err(reason) => {
                        if state.set_automatic_layout(None) {
                            metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone()).set(0.0);
                        }
                        state.layout_error = Some(reason.into());
                        if state.requires_layout || reason != "local_tokenizer_unavailable" {
                            diagnostics.warn(&source.worker_url, "cache_layout_auto_unavailable", &format!(
                                "Automatic cache layout unavailable: {reason}; model={} block_size={}; waiting for a compatible registered tokenizer and cache configuration",
                                profile.model, profile.block_size,
                            ));
                        }
                        continue;
                    }
                }
            }
            let mut request = client.get(format!(
                "{}/v1/pre-kv-cache/metadata",
                source.worker_url.trim_end_matches('/')
            ));
            if let Some(key) = worker.api_key().as_deref() {
                request = request.bearer_auth(key);
            }
            let result = tokio::time::timeout(Duration::from_secs(2), async {
                let response = request.send().await.map_err(|err| err.to_string())?;
                if !response.status().is_success() {
                    return Err(format!("metadata endpoint returned HTTP {}; install/enable the vLLM KV metadata bridge", response.status()));
                }
                let body = limited_body(response, cfg.max_response_bytes.min(256 * 1024)).await.map_err(str::to_owned)?;
                let layout: RuntimeLayout = serde_json::from_slice(&body).map_err(|err| err.to_string())?;
                layout.validate().map_err(str::to_owned)?;
                if layout.hash_block_size != cfg.profiles[&source.profile].block_size {
                    return Err(format!("hash block size mismatch: engine={} profile={}", layout.hash_block_size, cfg.profiles[&source.profile].block_size));
                }
                Ok(layout)
            }).await.unwrap_or_else(|_| Err("metadata request timed out".into()));
            let current = registry
                .get_by_url(&source.worker_url)
                .is_some_and(|current| Arc::ptr_eq(&worker, &current));
            let mut state = state.lock();
            if stop.load(Ordering::Acquire)
                || !current
                || !state
                    .binding
                    .as_ref()
                    .is_some_and(|bound| Arc::ptr_eq(bound, &worker))
            {
                continue;
            }
            match result {
                Ok(layout) => {
                    let group_count = layout.groups.len();
                    if install(&mut state, layout) {
                        metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone()).set(0.0);
                        tracing::info!(worker_url = %source.worker_url, group_count, "pre_kv_cache discovered complete runtime cache groups");
                    }
                    diagnostics.recovered(&source.worker_url, "cache_group_metadata_unavailable");
                    diagnostics.recovered(&source.worker_url, "cache_group_layout_missing");
                }
                Err(detail) => {
                    state.layout_error = Some(detail.clone());
                    // A failing response may come from a changed backend. Do
                    // not keep advertising the previous contract as current.
                    state.layout_checked = None;
                    // Older single-group engines need no bridge. A V4.1 request
                    // or an existing runtime layout makes its absence actionable.
                    if state.requires_layout || state.stream.layout.is_some() {
                        diagnostics.warn(
                            &source.worker_url,
                            "cache_group_metadata_unavailable",
                            &detail,
                        );
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, Worker};

    #[test]
    fn changing_engine_metadata_discards_residency_but_refresh_does_not() {
        // Refreshing the same engine preserves its index. A new engine/layout
        // discards even a valid index before accepting replacement events.
        let mut state = WorkerState::default();
        let layout = super::super::layout::fixture();
        assert!(install(&mut state, layout.clone()));
        state
            .stream
            .index
            .store(&[vec![1]], None, &[1, 2], 2, "m", "", 100)
            .unwrap();
        assert!(!install(&mut state, layout.clone()));
        assert_eq!(state.stream.index.len(), 1);
        let mut replacement = layout;
        replacement.engine_id = "engine-b".into();
        assert!(install(&mut state, replacement));
        assert_eq!(state.stream.index.len(), 0);
        assert_eq!(state.layout_revision, 2);
    }

    #[test]
    fn recovery_rebuilds_even_when_the_layout_returns_to_its_previous_value() {
        // Events received during an unavailable/incompatible metadata interval
        // cannot survive recovery merely because the returned JSON is equal.
        let mut state = WorkerState::default();
        let layout = super::super::layout::fixture();
        install(&mut state, layout.clone());
        state.layout_checked = None;
        state.stream.unsupported = true;
        assert!(install(&mut state, layout));
        assert!(!state.stream.unsupported);
        assert_eq!(state.layout_revision, 2);
    }

    #[tokio::test]
    async fn discovers_complete_groups_over_http_without_manual_groups_or_render() {
        use axum::{routing::get, Json, Router};
        let layout = super::super::layout::fixture();
        let reply = layout.clone();
        let app = Router::new().route(
            "/v1/pre-kv-cache/metadata",
            get(move || {
                let reply = reply.clone();
                async move { Json(reply) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let registry = Arc::new(WorkerRegistry::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(&url)
                .runtime_type(BackendType::Vllm)
                .build(),
        );
        registry.register(worker.clone());
        let state = Arc::new(Mutex::new(WorkerState::default()));
        state.lock().binding = Some(worker);
        state.lock().requires_layout = true;
        let cfg = super::super::events::tests::config();
        assert!(cfg.profiles["m"].cache_groups.is_empty());
        let source = EventSource {
            worker_url: url,
            profile: "m".into(),
            endpoint: "tcp://unused:5557".into(),
            replay_endpoint: None,
            topic: "kv".into(),
            speed: 1.0,
        };
        let stop = Arc::new(AtomicBool::new(false));
        let task = poll(
            source,
            Arc::new(cfg),
            Arc::downgrade(&registry),
            reqwest::Client::new(),
            state.clone(),
            stop.clone(),
            Arc::new(Diagnostics::default()),
            Arc::new(TokenizerRegistry::new()),
        );
        tokio::time::timeout(Duration::from_secs(3), async {
            while state.lock().stream.layout.is_none() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(state.lock().stream.layout.as_deref(), Some(&layout));
        assert!(state.lock().layout_checked.is_some());
        stop.store(true, Ordering::Release);
        task.abort();
        server.abort();
    }
}
