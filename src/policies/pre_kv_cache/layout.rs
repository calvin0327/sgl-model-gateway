//! Cache contracts come from the backend scheduler or a measured built-in
//! layout, never inferred solely from the groups seen in sampled events.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::config::LayoutPreset;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RuntimeLayout {
    pub schema_version: u32,
    pub engine_id: String,
    pub hash_block_size: usize,
    pub alignment_tokens: usize,
    pub dcp_world_size: usize,
    pub pcp_world_size: usize,
    pub data_parallel_size: usize,
    pub enable_prefix_caching: bool,
    pub enable_kv_cache_events: bool,
    pub enable_partial_hash_hits: bool,
    /// vLLM rebuilds non-cacheable SWA state from this tail after a cache hit.
    pub prefix_replay_tokens: usize,
    pub coordinator: String,
    /// Preserve backend equal-spec grouping and lookup order for EAGLE drops.
    pub lookup_groups: Vec<Vec<u64>>,
    pub groups: Vec<RuntimeGroup>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RuntimeGroup {
    pub group_idx: u64,
    pub block_size: usize,
    pub kind: String,
    pub sliding_window: Option<usize>,
    pub manager: String,
    pub prefix_cacheable: bool,
    /// The effective manager flag includes vLLM's OR across equal-spec groups.
    pub use_eagle: bool,
}

impl RuntimeGroup {
    pub fn cache_kind(&self) -> Option<super::config::CacheKind> {
        use super::config::CacheKind;
        match self.kind.as_str() {
            "full_attention" | "mla_attention" => Some(CacheKind::FullAttention),
            "sliding_window" | "sliding_window_mla" => Some(CacheKind::SlidingWindow),
            _ => None,
        }
    }

    pub fn is_window(&self) -> bool {
        matches!(self.kind.as_str(), "sliding_window" | "sliding_window_mla")
    }
}

impl RuntimeLayout {
    /// Dynamic metadata must identify its live engine as well as its contract.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.engine_id.is_empty() {
            return Err("cache_metadata_schema");
        }
        self.validate_contract()
    }

    /// Validate lookup semantics separately from live-engine identity. A preset
    /// has no discovered engine ID and must never claim that it verified one.
    pub fn validate_contract(&self) -> Result<(), &'static str> {
        if self.schema_version != 1 {
            return Err("cache_metadata_schema");
        }
        if !self.enable_prefix_caching || !self.enable_kv_cache_events {
            return Err("cache_metadata_disabled");
        }
        if self.dcp_world_size != 1 || self.pcp_world_size != 1 || self.data_parallel_size != 1 {
            return Err("cache_metadata_parallelism");
        }
        if self.enable_partial_hash_hits {
            return Err("cache_metadata_partial_hits");
        }
        if self.hash_block_size == 0
            || self.alignment_tokens == 0
            || self.alignment_tokens % self.hash_block_size != 0
            || self.groups.is_empty()
            || self.groups.len() > 256
        {
            return Err("cache_metadata_shape");
        }
        let mut ids = HashSet::new();
        let mut cacheable = 0;
        for group in &self.groups {
            if !ids.insert(group.group_idx) {
                return Err("cache_metadata_duplicate_group");
            }
            if !group.prefix_cacheable {
                continue;
            }
            cacheable += 1;
            if group.block_size == 0
                || group.block_size % self.hash_block_size != 0
                || self.alignment_tokens % group.block_size != 0
            {
                return Err("cache_metadata_block_size");
            }
            match (group.manager.as_str(), group.kind.as_str()) {
                ("FullAttentionManager" | "RSWAManager", "full_attention" | "mla_attention") => {
                    if group.sliding_window.is_some() {
                        return Err("cache_metadata_window");
                    }
                }
                ("SlidingWindowManager", "sliding_window" | "sliding_window_mla") => {
                    if group.sliding_window.is_none_or(|window| window <= 1) {
                        return Err("cache_metadata_window");
                    }
                }
                _ => return Err("cache_metadata_manager"),
            }
        }
        if cacheable == 0 {
            return Err("cache_metadata_no_cacheable_groups");
        }
        if !matches!(
            self.coordinator.as_str(),
            "UnitaryKVCacheCoordinator" | "HybridKVCacheCoordinator"
        ) {
            return Err("cache_metadata_coordinator");
        }
        let mut lookup_ids = HashSet::new();
        for ids in &self.lookup_groups {
            let first = self
                .group(*ids.first().ok_or("cache_metadata_lookup_groups")?)
                .ok_or("cache_metadata_lookup_groups")?;
            for id in ids {
                let group = self.group(*id).ok_or("cache_metadata_lookup_groups")?;
                if !lookup_ids.insert(*id)
                    || !group.prefix_cacheable
                    || group.block_size != first.block_size
                    || group.cache_kind() != first.cache_kind()
                    || group.sliding_window != first.sliding_window
                    || group.manager != first.manager
                    || group.use_eagle != first.use_eagle
                {
                    return Err("cache_metadata_lookup_groups");
                }
            }
        }
        if lookup_ids.len() != cacheable
            || (self.coordinator == "UnitaryKVCacheCoordinator" && self.groups.len() != 1)
        {
            return Err("cache_metadata_lookup_groups");
        }
        Ok(())
    }

    pub fn group(&self, id: u64) -> Option<&RuntimeGroup> {
        self.groups.iter().find(|group| group.group_idx == id)
    }

    pub fn reusable_hit(&self, hit: usize) -> usize {
        if hit <= self.prefix_replay_tokens {
            0
        } else {
            hit
        }
    }
}

impl LayoutPreset {
    /// Preserve the complete scheduler layout measured on the V4.1 deployment.
    /// Only group 8 participates in prefix lookup; SWA state is replayed locally.
    pub fn runtime_layout(self) -> RuntimeLayout {
        match self {
            Self::DeepseekV41Vllm031Dspark5 => {
                let mut groups: Vec<_> = (0..8)
                    .map(|group_idx| RuntimeGroup {
                        group_idx,
                        block_size: 32,
                        kind: "sliding_window_mla".into(),
                        sliding_window: Some(128),
                        manager: "SlidingWindowManager".into(),
                        prefix_cacheable: false,
                        use_eagle: false,
                    })
                    .collect();
                groups.push(RuntimeGroup {
                    group_idx: 8,
                    block_size: 64,
                    kind: "mla_attention".into(),
                    sliding_window: None,
                    manager: "FullAttentionManager".into(),
                    prefix_cacheable: true,
                    use_eagle: false,
                });
                groups.push(RuntimeGroup {
                    group_idx: 9,
                    block_size: 8,
                    kind: "unknown".into(),
                    sliding_window: None,
                    manager: "CircularBufferManager".into(),
                    prefix_cacheable: false,
                    use_eagle: false,
                });
                RuntimeLayout {
                    schema_version: 1,
                    engine_id: String::new(),
                    hash_block_size: 64,
                    alignment_tokens: 64,
                    dcp_world_size: 1,
                    pcp_world_size: 1,
                    data_parallel_size: 1,
                    enable_prefix_caching: true,
                    enable_kv_cache_events: true,
                    enable_partial_hash_hits: false,
                    prefix_replay_tokens: 128,
                    coordinator: "HybridKVCacheCoordinator".into(),
                    lookup_groups: vec![vec![8]],
                    groups,
                }
            }
        }
    }
}

#[cfg(test)]
pub(super) fn fixture() -> RuntimeLayout {
    RuntimeLayout {
        schema_version: 1,
        engine_id: "engine-a".into(),
        hash_block_size: 2,
        alignment_tokens: 2,
        dcp_world_size: 1,
        pcp_world_size: 1,
        data_parallel_size: 1,
        enable_prefix_caching: true,
        enable_kv_cache_events: true,
        enable_partial_hash_hits: false,
        prefix_replay_tokens: 0,
        coordinator: "UnitaryKVCacheCoordinator".into(),
        lookup_groups: vec![vec![0]],
        groups: vec![RuntimeGroup {
            group_idx: 0,
            block_size: 2,
            kind: "full_attention".into(),
            sliding_window: None,
            manager: "FullAttentionManager".into(),
            prefix_cacheable: true,
            use_eagle: false,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_preset_is_a_complete_contract_without_live_engine_identity() {
        let layout = LayoutPreset::DeepseekV41Vllm031Dspark5.runtime_layout();
        assert_eq!(layout.validate_contract(), Ok(()));
        assert_eq!(layout.validate(), Err("cache_metadata_schema"));
        assert!(layout.engine_id.is_empty());
        assert_eq!(layout.groups.len(), 10);
        assert_eq!(layout.lookup_groups, vec![vec![8]]);
        assert_eq!(layout.hash_block_size, 64);
        assert_eq!(layout.alignment_tokens, 64);
        assert_eq!(layout.prefix_replay_tokens, 128);
        assert_eq!(layout.reusable_hit(128), 0);
        assert_eq!(layout.reusable_hit(192), 192);
        assert_eq!(
            layout
                .groups
                .iter()
                .filter(|g| g.prefix_cacheable)
                .map(|g| g.group_idx)
                .collect::<Vec<_>>(),
            vec![8]
        );
        for id in 0..8 {
            let group = layout.group(id).unwrap();
            assert_eq!(group.block_size, 32);
            assert_eq!(group.sliding_window, Some(128));
            assert_eq!(group.manager, "SlidingWindowManager");
            assert!(!group.use_eagle);
        }
        assert_eq!(layout.group(8).unwrap().block_size, 64);
        assert_eq!(layout.group(9).unwrap().manager, "CircularBufferManager");
        assert_eq!(layout.group(9).unwrap().block_size, 8);
    }

    #[test]
    fn metadata_rejects_unimplemented_lookup_semantics() {
        let mut layout = fixture();
        assert_eq!(layout.validate(), Ok(()));
        layout.groups[0].manager = "CustomAttentionManager".into();
        assert_eq!(layout.validate(), Err("cache_metadata_manager"));
        layout.groups[0].prefix_cacheable = false;
        assert_eq!(layout.validate(), Err("cache_metadata_no_cacheable_groups"));
        layout = fixture();
        layout.enable_partial_hash_hits = true;
        assert_eq!(layout.validate(), Err("cache_metadata_partial_hits"));
        layout = fixture();
        layout.data_parallel_size = 2;
        assert_eq!(layout.validate(), Err("cache_metadata_parallelism"));
    }

    #[test]
    fn metadata_accepts_mixed_blocks_only_at_a_common_alignment() {
        let mut layout = fixture();
        let mut group = layout.groups[0].clone();
        group.group_idx = 1;
        group.block_size = 4;
        layout.groups.push(group);
        layout.coordinator = "HybridKVCacheCoordinator".into();
        layout.lookup_groups.push(vec![1]);
        assert_eq!(layout.validate(), Err("cache_metadata_block_size"));
        layout.alignment_tokens = 4;
        assert_eq!(layout.validate(), Ok(()));
    }
}
