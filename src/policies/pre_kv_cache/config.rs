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
    /// Legacy name: warn if the initial subscription receives no events within
    /// this interval. Confirmed residency never expires merely due to silence.
    pub event_stale_ms: u64,
    /// Emit per-candidate cost components at INFO when tuning routing weights.
    pub log_decisions: bool,
    /// Bound resident entries and, separately, nonresident ancestry per worker.
    /// Normal capacity pressure evicts individual entries, never the stream.
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
            log_decisions: false,
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
    /// Complete cacheable group layout. Empty keeps the legacy group-zero contract.
    /// Events alone cannot establish that every required group has been observed.
    #[serde(default)]
    pub cache_groups: Vec<CacheGroup>,
    /// Optional override; omission selects a recognized built-in layout or
    /// discovers engine metadata for deployments without a built-in adapter.
    #[serde(default)]
    pub layout: Option<LayoutPreset>,
    #[serde(default)]
    pub cost: CostModel,
}

/// A measured contract selected explicitly or for a recognized deployment.
/// Neither selection method verifies every setting of a live engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum LayoutPreset {
    #[serde(rename = "deepseek_v41_vllm031_dspark5")]
    #[value(name = "deepseek_v41_vllm031_dspark5")]
    DeepseekV41Vllm031Dspark5,
}

impl LayoutPreset {
    pub fn name(self) -> &'static str {
        match self {
            Self::DeepseekV41Vllm031Dspark5 => "deepseek_v41_vllm031_dspark5",
        }
    }
}

/// Group IDs describe backend cache groups, not tensor-parallel GPU ranks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheGroup {
    pub group_idx: u64,
    pub block_size: usize,
    pub kind: CacheKind,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    FullAttention,
    SlidingWindow,
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
    /// Higher importance favors avoiding uncached work; 1.0 preserves scoring.
    #[serde(alias = "kv_weight")]
    pub kv_importance: f64,
    /// Higher importance favors avoiding load; 0.0 ignores load and reservations.
    #[serde(alias = "load_weight")]
    pub load_importance: f64,
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
            kv_importance: 1.0,
            load_importance: 1.0,
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
            // A measured preset has one request-key contract and cannot be
            // combined with a second, manually supplied group inventory.
            if profile.layout.is_some()
                && (!profile.cache_groups.is_empty()
                    || profile.block_size != 64
                    || profile.token_source != TokenSource::LocalTokenizer)
            {
                return Err(format!(
                    "invalid pre_kv_cache profile {name}: layout requires local_tokenizer, block_size 64 and empty cache_groups"
                ));
            }
            // One request key chain serves every group. Reject mixed physical
            // block sizes rather than interpreting their hashes at the wrong size.
            let mut group_ids = HashSet::new();
            if profile.cache_groups.len() > 64
                || profile.cache_groups.iter().any(|group| {
                    !group_ids.insert(group.group_idx)
                        || group.block_size == 0
                        || group.block_size != profile.block_size
                        || match group.kind {
                            CacheKind::FullAttention => group.sliding_window.is_some(),
                            CacheKind::SlidingWindow => {
                                !group.sliding_window.is_some_and(|window| window > 1)
                            }
                        }
                })
            {
                return Err(format!(
                    "invalid pre_kv_cache profile {name}: cache_groups must contain at most 64 unique groups with the profile block_size; sliding_window groups require a window greater than 1"
                ));
            }
            let c = &profile.cost;
            // Keep relative importance within a practical range; accepting huge
            // finite multipliers can overflow ordinary routing costs.
            if [c.kv_importance, c.load_importance]
                .iter()
                .any(|importance| !(0.0..=1000.0).contains(importance))
                || (c.kv_importance == 0.0 && c.load_importance == 0.0)
            {
                return Err(format!(
                    "invalid pre_kv_cache profile {name}: kv_importance and load_importance must be between 0 and 1000, with at least one positive"
                ));
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_config() -> Config {
        serde_json::from_str(
            r#"{"profiles":{"model":{"model":"served","block_size":64}},"discovery":{}}"#,
        )
        .unwrap()
    }

    #[test]
    fn legacy_json_keeps_existing_weights_and_logging() {
        let config = legacy_config();
        config.validate().unwrap();
        let cost = &config.profiles["model"].cost;
        assert_eq!((cost.kv_importance, cost.load_importance), (1.0, 1.0));
        assert!(!config.log_decisions);
        assert!(config.profiles["model"].cache_groups.is_empty());
        assert!(config.profiles["model"].layout.is_none());

        // Existing partial cost overrides retain both new defaults too.
        let cost: CostModel = serde_json::from_str(r#"{"waiting_request_ms":25.0}"#).unwrap();
        assert_eq!((cost.kv_importance, cost.load_importance), (1.0, 1.0));
        assert_eq!(cost.waiting_request_ms, 25.0);
    }

    #[test]
    fn preset_json_requires_the_measured_contract_and_no_manual_groups() {
        let mut config: Config = serde_json::from_str(
            r#"{"profiles":{"model":{"model":"arbitrary-served-name","block_size":64,
                "token_source":"local_tokenizer","layout":"deepseek_v41_vllm031_dspark5"}},
                "discovery":{}}"#,
        )
        .unwrap();
        config.validate().unwrap();
        let selected = config.profiles["model"].layout.unwrap();
        assert_eq!(selected.name(), "deepseek_v41_vllm031_dspark5");
        assert_eq!(
            serde_json::to_value(selected).unwrap(),
            "deepseek_v41_vllm031_dspark5"
        );
        config.profiles.get_mut("model").unwrap().block_size = 32;
        assert!(config.validate().unwrap_err().contains("block_size 64"));
        config.profiles.get_mut("model").unwrap().block_size = 64;
        config.profiles.get_mut("model").unwrap().token_source = TokenSource::Render;
        assert!(config.validate().unwrap_err().contains("local_tokenizer"));
        let profile = config.profiles.get_mut("model").unwrap();
        profile.token_source = TokenSource::LocalTokenizer;
        profile.cache_groups = vec![CacheGroup {
            group_idx: 8,
            block_size: 64,
            kind: CacheKind::FullAttention,
            sliding_window: None,
        }];
        assert!(config
            .validate()
            .unwrap_err()
            .contains("empty cache_groups"));
        assert!(serde_json::from_str::<LayoutPreset>(r#""deepseek_v41""#).is_err());
    }

    #[test]
    fn group_layout_rejects_ambiguous_or_incompatible_contracts() {
        let mut config = legacy_config();
        let full = CacheGroup {
            group_idx: 0,
            block_size: 64,
            kind: CacheKind::FullAttention,
            sliding_window: None,
        };
        let swa = CacheGroup {
            group_idx: 1,
            block_size: 64,
            kind: CacheKind::SlidingWindow,
            sliding_window: Some(128),
        };
        config.profiles.get_mut("model").unwrap().cache_groups = vec![full.clone(), swa.clone()];
        config.validate().unwrap();
        for groups in [
            vec![full.clone(), full.clone()],
            vec![CacheGroup {
                block_size: 128,
                ..full.clone()
            }],
            vec![CacheGroup {
                sliding_window: Some(128),
                ..full.clone()
            }],
            vec![CacheGroup {
                sliding_window: None,
                ..swa.clone()
            }],
            vec![CacheGroup {
                sliding_window: Some(1),
                ..swa
            }],
            (0..65)
                .map(|group_idx| CacheGroup {
                    group_idx,
                    ..full.clone()
                })
                .collect(),
        ] {
            config.profiles.get_mut("model").unwrap().cache_groups = groups;
            assert!(config.validate().unwrap_err().contains("cache_groups"));
        }
        let unknown = r#"{"group_idx":0,"block_size":64,"kind":"mamba"}"#;
        assert!(serde_json::from_str::<CacheGroup>(unknown).is_err());
    }

    #[test]
    fn json_can_configure_scoring_weights_and_decision_logging() {
        let config: Config = serde_json::from_str(
            r#"{
                "profiles":{"ds":{"model":"served","block_size":64,
                    "cost":{"kv_importance":0.5,"load_importance":2.0}}},
                "discovery":{},
                "log_decisions":true
            }"#,
        )
        .unwrap();
        config.validate().unwrap();
        let cost = &config.profiles["ds"].cost;
        assert_eq!((cost.kv_importance, cost.load_importance), (0.5, 2.0));
        assert_eq!(cost.waiting_request_ms, 100.0);
        assert!(config.log_decisions);
    }

    #[test]
    fn old_weight_names_remain_aliases_for_importance() {
        // Existing JSON keeps its meaning, while serialization uses clear names.
        let old: CostModel = serde_json::from_str(r#"{"kv_weight":4,"load_weight":0.5}"#).unwrap();
        let new: CostModel =
            serde_json::from_str(r#"{"kv_importance":4,"load_importance":0.5}"#).unwrap();
        let value = serde_json::to_value(&old).unwrap();
        assert_eq!(value, serde_json::to_value(new).unwrap());
        assert_eq!(value["kv_importance"], 4.0);
        assert!(value.get("kv_weight").is_none());
        assert!(value.get("load_weight").is_none());
        assert!(serde_json::from_str::<CostModel>(r#"{"kv_weight":1,"kv_importance":2}"#).is_err());
    }

    #[test]
    fn importance_must_be_in_range_and_not_both_zero() {
        for (kv_importance, load_importance) in [
            (-1.0, 1.0),
            (1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (1.0, f64::NEG_INFINITY),
            (1000.01, 1.0),
            (1.0, 1000.01),
            (1e308, 1.0),
            (1.0, 1e308),
            (0.0, 0.0),
        ] {
            let mut config = legacy_config();
            let cost = &mut config.profiles.get_mut("model").unwrap().cost;
            cost.kv_importance = kv_importance;
            cost.load_importance = load_importance;
            assert!(config.validate().unwrap_err().contains("kv_importance"));
        }
        for (kv_importance, load_importance) in
            [(0.0, 1.0), (1.0, 0.0), (0.3, 2.0), (1000.0, 1000.0)]
        {
            let mut config = legacy_config();
            let cost = &mut config.profiles.get_mut("model").unwrap().cost;
            cost.kv_importance = kv_importance;
            cost.load_importance = load_importance;
            config.validate().unwrap();
        }
    }
}
