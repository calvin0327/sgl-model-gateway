//! Explicit model and event-source contracts for vLLM prefix routing.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub profiles: HashMap<String, Profile>,
    pub workers: Vec<EventSource>,
    /// Derive per-Pod event addresses from the existing worker registry.
    pub discovery: Option<Discovery>,
    pub render_timeout_ms: u64,
    pub load_poll_ms: u64,
    pub load_stale_ms: u64,
    pub event_stale_ms: u64,
    pub max_blocks_per_worker: usize,
    pub max_render_tokens: usize,
    pub max_response_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            profiles: HashMap::new(),
            workers: Vec::new(),
            discovery: None,
            render_timeout_ms: 1000,
            load_poll_ms: 1000,
            load_stale_ms: 5000,
            event_stale_ms: 30000,
            max_blocks_per_worker: 500_000,
            max_render_tokens: 1_000_000,
            max_response_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Workers sharing a profile must use identical weights, tokenizer and template.
/// Cost coefficients are initial estimates in milliseconds, to be calibrated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub model: String,
    /// Choose whether request token IDs come from vLLM or the Gateway tokenizer.
    #[serde(default)]
    pub token_source: TokenSource,
    /// Empty means render on a healthy worker in this profile's candidate pool.
    #[serde(default)]
    pub render_url: String,
    /// Optional environment variable containing the render service bearer token.
    #[serde(default)]
    pub render_api_key_env: Option<String>,
    pub block_size: usize,
    #[serde(default)]
    pub cost: CostModel,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenSource {
    #[default]
    Render,
    LocalTokenizer,
}

/// Replicas can share these ports because each Pod has its own IP address.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Discovery {
    pub event_port: u16,
    pub replay_port: Option<u16>,
    pub topic: String,
    pub refresh_ms: u64,
}

impl Default for Discovery {
    fn default() -> Self {
        Self {
            event_port: 5557,
            replay_port: Some(5558),
            topic: "kv".into(),
            refresh_ms: 1000,
        }
    }
}

impl Discovery {
    /// Keep the discovered host (including IPv6 brackets), replacing only port.
    pub fn source(&self, worker_url: &str, profile: &str) -> Option<EventSource> {
        let url = url::Url::parse(worker_url).ok()?;
        if !matches!(url.scheme(), "http" | "https") {
            return None;
        }
        let host = url.host()?;
        Some(EventSource {
            worker_url: worker_url.into(),
            profile: profile.into(),
            endpoint: format!("tcp://{host}:{}", self.event_port),
            replay_endpoint: self.replay_port.map(|port| format!("tcp://{host}:{port}")),
            topic: self.topic.clone(),
            speed: 1.0,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CostModel {
    pub prefill_ms_per_token: f64,
    pub context_ms_per_million_token_pairs: f64,
    pub waiting_request_ms: f64,
    pub running_request_ms: f64,
    pub kv_pressure_ms: f64,
    pub mean_uncached_tokens: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            prefill_ms_per_token: 0.05,
            context_ms_per_million_token_pairs: 0.1,
            waiting_request_ms: 100.0,
            running_request_ms: 10.0,
            kv_pressure_ms: 10.0,
            mean_uncached_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventSource {
    pub worker_url: String,
    pub profile: String,
    pub endpoint: String,
    pub topic: String,
    #[serde(default)]
    pub replay_endpoint: Option<String>,
    /// Relative prefill speed; useful for heterogeneous workers.
    #[serde(default = "one")]
    pub speed: f64,
}

fn one() -> f64 {
    1.0
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.profiles.is_empty() || (self.workers.is_empty() && self.discovery.is_none()) {
            return Err(
                "pre_kv_cache requires profiles and discovery or explicit event sources".into(),
            );
        }
        if let Some(discovery) = &self.discovery {
            if discovery.event_port == 0
                || discovery.replay_port == Some(0)
                || discovery.replay_port == Some(discovery.event_port)
                || discovery.topic.is_empty()
                || discovery.refresh_ms == 0
            {
                return Err("invalid pre_kv_cache discovery ports, topic or refresh_ms".into());
            }
            let mut models = HashSet::new();
            if self.profiles.values().any(|p| !models.insert(&p.model)) {
                return Err("discovery requires one unambiguous profile per served model".into());
            }
        }
        if [
            self.render_timeout_ms,
            self.load_poll_ms,
            self.load_stale_ms,
            self.event_stale_ms,
        ]
        .contains(&0)
            || self.load_stale_ms < self.load_poll_ms
            || self.max_blocks_per_worker == 0
            || self.max_render_tokens == 0
            || self.max_response_bytes == 0
        {
            return Err(
                "pre_kv_cache limits must be positive; load_stale_ms must cover load_poll_ms"
                    .into(),
            );
        }
        for (name, profile) in &self.profiles {
            let c = &profile.cost;
            if !profile.render_url.is_empty() {
                let url = url::Url::parse(&profile.render_url).map_err(|_| "invalid render_url")?;
                if !matches!(url.scheme(), "http" | "https")
                    || url.host_str().is_none()
                    || !url.username().is_empty()
                    || url.password().is_some()
                {
                    return Err("invalid render_url".into());
                }
            }
            if name.is_empty()
                || profile.model.is_empty()
                || profile.block_size == 0
                || profile
                    .render_api_key_env
                    .as_ref()
                    .is_some_and(|s| s.is_empty())
                || [
                    c.prefill_ms_per_token,
                    c.context_ms_per_million_token_pairs,
                    c.waiting_request_ms,
                    c.running_request_ms,
                    c.kv_pressure_ms,
                ]
                .iter()
                .any(|n| !n.is_finite() || *n < 0.0)
                || c.prefill_ms_per_token == 0.0
                || c.mean_uncached_tokens == 0
            {
                return Err(format!("invalid pre_kv_cache profile {name}"));
            }
        }
        let mut urls = HashSet::new();
        let mut endpoints = HashSet::new();
        for source in &self.workers {
            if !self.profiles.contains_key(&source.profile)
                || !urls.insert(&source.worker_url)
                || !endpoints.insert(&source.endpoint)
                || source.topic.is_empty()
                || !source.endpoint.starts_with("tcp://")
                || source
                    .replay_endpoint
                    .as_ref()
                    .is_some_and(|e| !e.starts_with("tcp://"))
                || !source.speed.is_finite()
                || source.speed <= 0.0
            {
                return Err("invalid/duplicate pre_kv_cache event source or speed".into());
            }
            let url = url::Url::parse(&source.worker_url).map_err(|_| "invalid worker_url")?;
            if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                return Err("pre_kv_cache supports HTTP vLLM workers".into());
            }
            if !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err("worker_url must not contain credentials, query or fragment".into());
            }
        }
        Ok(())
    }
}
