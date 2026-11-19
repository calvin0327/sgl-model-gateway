//! Event residency and request tokens share one prefix-key function.
//! Engine hashes remain opaque identifiers, scoped to one worker incarnation.

use std::collections::{HashMap, HashSet, VecDeque};

use lru::LruCache;

pub type Key = [u8; 32];
pub type EngineHash = Vec<u8>;

pub fn root(profile: &str, salt: &str) -> Key {
    let mut h = blake3::Hasher::new();
    h.update(b"pre_kv_cache/v1");
    h.update(&(profile.len() as u64).to_le_bytes());
    h.update(profile.as_bytes());
    h.update(salt.as_bytes());
    *h.finalize().as_bytes()
}

pub fn block_key(parent: Key, tokens: &[u32]) -> Key {
    let mut h = blake3::Hasher::new();
    h.update(&parent);
    for token in tokens {
        h.update(&token.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// Hash each request once per profile, then reuse its chain across all workers.
pub fn prefix_keys(tokens: &[u32], block_size: usize, profile: &str, salt: &str) -> Vec<Key> {
    let mut key = root(profile, salt);
    tokens[..tokens.len().saturating_sub(1)]
        .chunks_exact(block_size)
        .map(|chunk| {
            key = block_key(key, chunk);
            key
        })
        .collect()
}

#[derive(Debug)]
pub struct Index {
    // Engine hashes identify content independently of cache group. Keep this
    // bounded ancestry after eviction, but keep physical residency group-local.
    known: HashMap<EngineHash, KnownPrefix>,
    resident_hashes: LruCache<(u64, EngineHash), Key>,
    resident: HashMap<(u64, Key), usize>,
    // Protect resident identities from historical churn. Retain at most one
    // additional capacity of unused ancestors, evicting the oldest first.
    resident_refs: HashMap<EngineHash, usize>,
    unused: LruCache<EngineHash, ()>,
    limit: usize,
    pub ancestry_evictions: u64,
    pub residency_evictions: u64,
    seen_groups: HashSet<u64>,
    // Unsupported identities never enter the request-key dictionary. Bounded
    // tombstones distinguish their descendants from recoverable missing history.
    opaque: HashSet<EngineHash>,
    opaque_order: VecDeque<EngineHash>,
    // Evicted tombstones must not become ordinary text identities. A fixed
    // Bloom filter may add conservative misses, but never forgets a rejection.
    opaque_overflow: Vec<u64>,
}

impl Default for Index {
    fn default() -> Self {
        Self {
            known: HashMap::new(),
            resident_hashes: LruCache::unbounded(),
            resident: HashMap::new(),
            resident_refs: HashMap::new(),
            unused: LruCache::unbounded(),
            limit: 1,
            ancestry_evictions: 0,
            residency_evictions: 0,
            seen_groups: HashSet::new(),
            opaque: HashSet::new(),
            opaque_order: VecDeque::new(),
            opaque_overflow: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KnownPrefix {
    key: Key,
    namespace: Key,
}

impl Index {
    pub fn known_len(&self) -> usize {
        self.known.len()
    }

    fn trim_ancestry(&mut self) {
        while self.unused.len() > self.limit {
            if let Some((hash, ())) = self.unused.pop_lru() {
                self.known.remove(&hash);
                self.ancestry_evictions = self.ancestry_evictions.saturating_add(1);
                metrics::counter!("pre_kv_cache_index_evictions_total", "kind" => "ancestry")
                    .increment(1);
            }
        }
    }

    fn remember(&mut self, hash: &EngineHash, prefix: KnownPrefix) {
        self.known.insert(hash.clone(), prefix);
        if !self.resident_refs.contains_key(hash) {
            self.unused.put(hash.clone(), ());
        }
        self.trim_ancestry();
    }

    fn prefix(&mut self, hash: &EngineHash) -> Option<KnownPrefix> {
        // A parent still used by new Stores deserves to outlive cold ancestry.
        self.unused.get(hash);
        self.known.get(hash).copied()
    }

    fn release_resident(&mut self, group: u64, hash: &EngineHash, key: Key) {
        let entry = (group, key);
        if let Some(count) = self.resident.get_mut(&entry) {
            *count -= 1;
            if *count == 0 {
                self.resident.remove(&entry);
            }
        }
        if let Some(count) = self.resident_refs.get_mut(hash) {
            *count -= 1;
            if *count == 0 {
                self.resident_refs.remove(hash);
                self.unused.put(hash.clone(), ());
            }
        }
        self.trim_ancestry();
    }

    pub fn is_opaque(&self, hash: &EngineHash) -> bool {
        self.opaque.contains(hash)
            || (!self.opaque_overflow.is_empty()
                && Self::opaque_bits(hash)
                    .iter()
                    .all(|&bit| self.opaque_overflow[bit / 64] & (1u64 << (bit % 64)) != 0))
    }

    fn opaque_bits(hash: &EngineHash) -> [usize; 3] {
        let digest = blake3::hash(hash);
        std::array::from_fn(|i| {
            let word = u32::from_le_bytes(digest.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap());
            word as usize % (1 << 23)
        })
    }

    pub fn isolation_overflow(&self) -> bool {
        !self.opaque_overflow.is_empty()
    }

    pub fn opaque_len(&self) -> usize {
        self.opaque.len()
    }

    pub fn quarantine(&mut self, hashes: &[EngineHash], limit: usize) -> Result<(), &'static str> {
        // Contradicting an already learned identity cannot leave old hits alive.
        if hashes.iter().any(|hash| self.known.contains_key(hash)) {
            return Err("engine_hash_conflict");
        }
        let limit = limit.clamp(1, 65_536);
        for hash in hashes {
            if self.opaque.contains(hash) {
                continue;
            }
            if self.opaque.len() >= limit {
                if let Some(oldest) = self.opaque_order.pop_front() {
                    self.opaque.remove(&oldest);
                    if self.opaque_overflow.is_empty() {
                        self.opaque_overflow.resize(1 << 17, 0);
                    }
                    for bit in Self::opaque_bits(&oldest) {
                        self.opaque_overflow[bit / 64] |= 1u64 << (bit % 64);
                    }
                }
            }
            self.opaque.insert(hash.clone());
            self.opaque_order.push_back(hash.clone());
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn store(
        &mut self,
        hashes: &[EngineHash],
        parent: Option<&EngineHash>,
        tokens: &[u32],
        block_size: usize,
        profile: &str,
        salt: &str,
        limit: usize,
    ) -> Result<(), &'static str> {
        self.store_group(0, hashes, parent, tokens, block_size, profile, salt, limit)
    }

    /// Learning content never marks it resident. This permits sparse groups
    /// earlier in a batch to resolve hashes from a later dense group without
    /// reordering Store/Remove events or resurrecting evicted blocks.
    #[cfg(test)]
    pub fn learn_dense(
        &mut self,
        hashes: &[EngineHash],
        parent: Option<&EngineHash>,
        tokens: &[u32],
        block_size: usize,
        profile: &str,
        salt: &str,
        limit: usize,
    ) -> Result<(), &'static str> {
        self.learn_dense_runtime(
            hashes, parent, tokens, block_size, block_size, profile, salt, limit,
        )
    }

    /// Normalize physical blocks to the runtime hash granularity so different
    /// cache group block sizes still share the same content ancestry.
    pub fn learn_dense_runtime(
        &mut self,
        hashes: &[EngineHash],
        parent: Option<&EngineHash>,
        tokens: &[u32],
        block_size: usize,
        hash_block_size: usize,
        profile: &str,
        salt: &str,
        limit: usize,
    ) -> Result<(), &'static str> {
        if block_size == 0
            || hash_block_size == 0
            || block_size % hash_block_size != 0
            || hashes.len().checked_mul(block_size) != Some(tokens.len())
        {
            return Err("block_shape");
        }
        self.limit = limit.max(1);
        let start = match parent {
            Some(parent) => self.prefix(parent).ok_or("missing_parent")?,
            None => {
                let namespace = root(profile, salt);
                KnownPrefix {
                    key: namespace,
                    namespace,
                }
            }
        };
        let mut key = start.key;
        for (hash, chunk) in hashes.iter().zip(tokens.chunks_exact(block_size)) {
            for part in chunk.chunks_exact(hash_block_size) {
                key = block_key(key, part);
            }
            let prefix = KnownPrefix {
                key,
                namespace: start.namespace,
            };
            if self
                .known
                .get(hash)
                .is_some_and(|previous| previous != &prefix)
            {
                return Err("engine_hash_conflict");
            }
            self.remember(hash, prefix);
        }
        Ok(())
    }

    /// A sparse Store carries the entire token range but only resident hashes.
    /// Resolve those hashes from previously verified dense content mappings;
    /// never zip a sparse hash list with the beginning of the token range.
    #[cfg(test)]
    pub fn store_group(
        &mut self,
        group: u64,
        hashes: &[EngineHash],
        parent: Option<&EngineHash>,
        tokens: &[u32],
        block_size: usize,
        profile: &str,
        salt: &str,
        limit: usize,
    ) -> Result<(), &'static str> {
        self.store_group_runtime(
            group, hashes, parent, tokens, block_size, block_size, profile, salt, limit,
        )
    }

    pub fn store_group_runtime(
        &mut self,
        group: u64,
        hashes: &[EngineHash],
        parent: Option<&EngineHash>,
        tokens: &[u32],
        block_size: usize,
        hash_block_size: usize,
        profile: &str,
        salt: &str,
        limit: usize,
    ) -> Result<(), &'static str> {
        if block_size == 0
            || hash_block_size == 0
            || block_size % hash_block_size != 0
            || tokens.len() % block_size != 0
            || hashes.len() > tokens.len() / block_size
        {
            return Err("block_shape");
        }
        let sparse = hashes.len() != tokens.len() / block_size;
        self.limit = limit.max(1);
        let mut start = match parent {
            Some(parent) => self.prefix(parent).ok_or("missing_parent")?,
            None => {
                let namespace = root(profile, salt);
                KnownPrefix {
                    key: namespace,
                    namespace,
                }
            }
        };
        // Sparse root events can omit block zero and therefore its cache_salt.
        // Recover that namespace only from a hash verified by a dense event;
        // every sparse hash below must belong to the same namespace.
        if sparse && parent.is_none() {
            if let Some(hash) = hashes.first() {
                let known = self.prefix(hash).ok_or("sparse_hash_unknown")?;
                if !salt.is_empty() && known.namespace != start.namespace {
                    return Err("cache_salt_mismatch");
                }
                start = KnownPrefix {
                    key: known.namespace,
                    namespace: known.namespace,
                };
            }
        }
        let mut key = start.key;
        let keys: Vec<_> = tokens
            .chunks_exact(block_size)
            .map(|chunk| {
                for part in chunk.chunks_exact(hash_block_size) {
                    key = block_key(key, part);
                }
                key
            })
            .collect();
        let mappings: Vec<Key> = if hashes.len() == keys.len() {
            keys
        } else {
            let mut offset = 0;
            let mut mappings = Vec::with_capacity(hashes.len());
            for hash in hashes {
                let known = self.prefix(hash).ok_or("sparse_hash_unknown")?;
                if known.namespace != start.namespace {
                    return Err("sparse_namespace_mismatch");
                }
                let position = keys[offset..]
                    .iter()
                    .position(|key| key == &known.key)
                    .ok_or("sparse_hash_mismatch")?;
                offset += position + 1;
                mappings.push(known.key);
            }
            mappings
        };
        // Validate identity before changing residency. Capacity pressure only
        // evicts local entries; it is not evidence of a broken event stream.
        let mut batch_keys = HashMap::new();
        for (hash, key) in hashes.iter().zip(&mappings) {
            if batch_keys
                .insert(hash, key)
                .is_some_and(|previous| previous != key)
            {
                return Err("engine_hash_conflict");
            }
            if self.known.get(hash).is_some_and(|previous| {
                previous.key != *key || previous.namespace != start.namespace
            }) {
                return Err("engine_hash_conflict");
            }
        }
        for (hash, key) in hashes.iter().zip(mappings) {
            self.remember(
                hash,
                KnownPrefix {
                    key,
                    namespace: start.namespace,
                },
            );
            if self
                .resident_hashes
                .put((group, hash.clone()), key)
                .is_none()
            {
                *self.resident.entry((group, key)).or_default() += 1;
                *self.resident_refs.entry(hash.clone()).or_default() += 1;
            }
            self.unused.pop(hash);
            while self.resident_hashes.len() > self.limit {
                if let Some(((old_group, old_hash), old_key)) = self.resident_hashes.pop_lru() {
                    self.release_resident(old_group, &old_hash, old_key);
                    self.residency_evictions = self.residency_evictions.saturating_add(1);
                    metrics::counter!("pre_kv_cache_index_evictions_total", "kind" => "residency")
                        .increment(1);
                }
            }
        }
        self.seen_groups.insert(group);
        Ok(())
    }

    #[cfg(test)]
    pub fn remove(&mut self, hashes: &[EngineHash]) {
        self.remove_group(0, hashes);
    }

    pub fn remove_group(&mut self, group: u64, hashes: &[EngineHash]) {
        for hash in hashes {
            if let Some(key) = self.resident_hashes.pop(&(group, hash.clone())) {
                self.release_resident(group, hash, key);
            }
        }
    }

    pub fn clear(&mut self) {
        self.known.clear();
        self.resident_hashes.clear();
        self.resident.clear();
        self.resident_refs.clear();
        self.unused.clear();
        self.seen_groups.clear();
        self.opaque.clear();
        self.opaque_order.clear();
        self.opaque_overflow.clear();
    }

    pub fn len(&self) -> usize {
        self.resident_hashes.len()
    }

    pub fn matched_keys(&self, keys: &[Key], block_size: usize) -> usize {
        keys.iter()
            .take_while(|key| self.resident.contains_key(&(0, **key)))
            .count()
            * block_size
    }

    /// Check one common boundary across every required group. A sliding window
    /// may reuse a tail even after its older blocks were evicted. Descending
    /// candidates are rechecked across all groups; minimum individual maxima
    /// alone is insufficient when a window has holes.
    pub fn matched_keys_layout(
        &self,
        keys: &[Key],
        block_size: usize,
        groups: &[super::config::CacheGroup],
    ) -> Option<usize> {
        use super::config::CacheKind;
        if groups.is_empty() {
            return Some(self.matched_keys(keys, block_size));
        }
        if block_size == 0
            || groups.iter().any(|group| {
                group.block_size != block_size || !self.seen_groups.contains(&group.group_idx)
            })
        {
            return None;
        }
        // For each group, record valid endpoints in linear time. Window prefix
        // shortfalls are allowed only at the request's actual root.
        let mut common = vec![true; keys.len() + 1];
        for group in groups {
            let required = match group.kind {
                CacheKind::FullAttention => usize::MAX,
                CacheKind::SlidingWindow => group
                    .sliding_window?
                    .checked_sub(1)?
                    .div_ceil(block_size)
                    .max(1),
            };
            let mut contiguous = 0;
            for (i, key) in keys.iter().enumerate() {
                contiguous = if self.resident.contains_key(&(group.group_idx, *key)) {
                    contiguous + 1
                } else {
                    0
                };
                common[i + 1] &= contiguous >= required.min(i + 1);
            }
        }
        common
            .iter()
            .rposition(|valid| *valid)
            .map(|blocks| blocks * block_size)
    }

    /// Follow the backend coordinator's lookup order and EAGLE drop-once
    /// state. A later group can shrink the candidate boundary; vLLM then
    /// rechecks prior windows without applying an already-verified drop twice.
    pub fn matched_keys_runtime(
        &self,
        keys: &[Key],
        layout: &super::layout::RuntimeLayout,
    ) -> Option<usize> {
        layout.validate_contract().ok()?;
        let max_length = keys.len().checked_mul(layout.hash_block_size)?;
        let groups: Vec<_> = layout
            .lookup_groups
            .iter()
            .map(|ids| (ids, layout.group(ids[0]).unwrap()))
            .collect();
        if layout.coordinator == "UnitaryKVCacheCoordinator" {
            let (ids, group) = groups[0];
            return Some(layout.reusable_hit(self.runtime_group_hit(
                keys,
                layout,
                ids,
                max_length,
                group.use_eagle,
            )));
        }
        let mut hit_length = max_length;
        let mut full_hits = vec![None; groups.len()];
        let mut eagle_verified = HashSet::new();
        let simple_hybrid = groups.len() == 2 && !groups[0].1.is_window();
        loop {
            let mut current = hit_length;
            for (i, (ids, group)) in groups.iter().enumerate() {
                if let Some(previous) = full_hits[i] {
                    current = current.min(previous);
                    continue;
                }
                let drop = group.use_eagle && !eagle_verified.contains(&i);
                let cap = if drop {
                    current.saturating_add(group.block_size).min(max_length)
                } else {
                    current
                };
                let next = self.runtime_group_hit(keys, layout, ids, cap, drop);
                if drop {
                    eagle_verified.insert(i);
                } else if next < current {
                    eagle_verified.clear();
                }
                if !group.is_window() {
                    full_hits[i] = Some(next);
                }
                current = next;
            }
            if current >= hit_length {
                return Some(layout.reusable_hit(hit_length));
            }
            hit_length = current;
            if simple_hybrid {
                return Some(layout.reusable_hit(hit_length));
            }
        }
    }

    /// Full attention scans from the root; SWA scans backwards for a resident
    /// tail. Physical block keys are endpoints in the shared hash-size chain.
    fn runtime_group_hit(
        &self,
        keys: &[Key],
        layout: &super::layout::RuntimeLayout,
        ids: &[u64],
        max_length: usize,
        drop: bool,
    ) -> usize {
        let group = layout.group(ids[0]).unwrap();
        let block_size = group.block_size;
        let stride = block_size / layout.hash_block_size;
        let max_blocks = max_length / block_size;
        let present = |block: usize| {
            let key = keys[(block + 1) * stride - 1];
            ids.iter().all(|id| self.resident.contains_key(&(*id, key)))
        };
        let mut hit_blocks = 0;
        if group.is_window() {
            let required =
                (group.sliding_window.unwrap() - 1).div_ceil(block_size) + usize::from(drop);
            let mut contiguous = 0;
            let mut found = false;
            for block in (0..max_blocks).rev() {
                if present(block) {
                    // A new window run must end at an aligned post-drop boundary.
                    let after_drop = block + 1 - usize::from(drop);
                    if contiguous == 0 && after_drop * block_size % layout.alignment_tokens != 0 {
                        continue;
                    }
                    contiguous += 1;
                    if contiguous >= required {
                        hit_blocks = block + contiguous;
                        found = true;
                        break;
                    }
                } else {
                    contiguous = 0;
                }
            }
            if !found {
                // A short resident prefix at the true root needs no older tail.
                hit_blocks = contiguous;
                while hit_blocks * block_size % layout.alignment_tokens != 0 {
                    hit_blocks -= 1;
                }
            }
        } else {
            while hit_blocks < max_blocks && present(hit_blocks) {
                hit_blocks += 1;
            }
        }
        if drop {
            hit_blocks = hit_blocks.saturating_sub(1);
        }
        let tokens = hit_blocks * block_size;
        tokens - tokens % layout.alignment_tokens
    }

    #[cfg(test)]
    pub fn matched_tokens(
        &self,
        tokens: &[u32],
        block_size: usize,
        profile: &str,
        salt: &str,
    ) -> usize {
        self.matched_keys(&prefix_keys(tokens, block_size, profile, salt), block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_dspark_window_requires_the_extra_block_and_drops_it_once() {
        let mut layout = super::super::layout::fixture();
        let mut group = layout.groups[0].clone();
        group.group_idx = 1;
        group.kind = "sliding_window".into();
        group.manager = "SlidingWindowManager".into();
        group.sliding_window = Some(4);
        group.use_eagle = true;
        layout.groups.push(group);
        layout.coordinator = "HybridKVCacheCoordinator".into();
        layout.lookup_groups.push(vec![1]);
        let tokens: Vec<_> = (0..13).collect();
        let keys = prefix_keys(&tokens, 2, "m", "");
        let hashes: Vec<_> = (0..6).map(|n| vec![n]).collect();
        let mut index = Index::default();
        index
            .store_group(0, &hashes, None, &tokens[..12], 2, "m", "", 20)
            .unwrap();
        // The metadata enumerates a required window that has no residents yet.
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(0));
        index
            .store_group(1, &hashes[3..], None, &tokens[..12], 2, "m", "", 20)
            .unwrap();
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(10));
        index.remove_group(1, &hashes[3..4]);
        // Two tail blocks suffice for ordinary SWA, but not for DSpark's drop.
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(0));
        layout.groups[1].use_eagle = false;
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(12));
    }

    #[test]
    fn runtime_mixed_blocks_share_hash_ancestry_but_keep_physical_residency() {
        let mut layout = super::super::layout::fixture();
        layout.alignment_tokens = 4;
        let mut group = layout.groups[0].clone();
        group.group_idx = 1;
        group.block_size = 4;
        group.use_eagle = true;
        layout.groups.push(group);
        layout.coordinator = "HybridKVCacheCoordinator".into();
        layout.lookup_groups.push(vec![1]);
        let tokens: Vec<_> = (0..13).collect();
        let keys = prefix_keys(&tokens, 2, "m", "");
        let hashes: Vec<_> = (0..6).map(|n| vec![n]).collect();
        let wide_hashes: Vec<_> = hashes.iter().skip(1).step_by(2).cloned().collect();
        let mut index = Index::default();
        index
            .store_group_runtime(0, &hashes, None, &tokens[..12], 2, 2, "m", "", 20)
            .unwrap();
        index
            .store_group_runtime(1, &wide_hashes, None, &tokens[..12], 4, 2, "m", "", 20)
            .unwrap();
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(8));
        index.remove_group(1, &wide_hashes[1..2]);
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(0));
        // Evicting a wide group's physical block never evicts narrow residents.
        assert_eq!(index.matched_keys(&keys, 2), 12);
    }

    #[test]
    fn runtime_hybrid_preserves_backend_eagle_verification_and_lookup_partition() {
        let mut layout = super::super::layout::fixture();
        layout.coordinator = "HybridKVCacheCoordinator".into();
        let group = &mut layout.groups[0];
        group.kind = "sliding_window".into();
        group.manager = "SlidingWindowManager".into();
        group.sliding_window = Some(4);
        group.use_eagle = true;
        let mut other = group.clone();
        other.group_idx = 1;
        other.sliding_window = Some(6);
        layout.groups.push(other);
        layout.lookup_groups.push(vec![1]);
        let keys = prefix_keys(&(0..13).collect::<Vec<_>>(), 2, "m", "");
        let mut index = Index::default();
        for (group, mask) in [(0, 29), (1, 3)] {
            for (i, key) in keys.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    index.resident.insert((group, *key), 1);
                }
            }
        }
        // First iteration drops each group once: 12 -> 8 -> 2. At 2 the
        // already-verified groups each accept their resident root block;
        // independently requiring another lookahead block would wrongly give 0.
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(2));
        // Equal specs can instead be combined by the backend into one lookup.
        // Both groups must then contain each block before a single drop occurs.
        layout.groups[1].sliding_window = Some(4);
        layout.lookup_groups = vec![vec![0, 1]];
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(0));
    }

    #[test]
    fn runtime_bounded_replay_discards_short_hits_but_reports_long_raw_hits() {
        let mut layout = super::super::layout::fixture();
        layout.prefix_replay_tokens = 4;
        let tokens: Vec<_> = (0..7).collect();
        let keys = prefix_keys(&tokens, 2, "m", "");
        let mut index = Index::default();
        index
            .store_group(0, &[vec![1], vec![2]], None, &tokens[..4], 2, "m", "", 20)
            .unwrap();
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(0));
        index
            .store_group(0, &[vec![3]], Some(&vec![2]), &tokens[4..6], 2, "m", "", 20)
            .unwrap();
        // Backend usage reports six cached tokens; only two avoid computation
        // because the four-token SWA tail is replayed after prefix lookup.
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(6));
    }

    #[test]
    fn v41_preset_preserves_raw_hits_and_the_last_token_boundary() {
        let layout = super::super::config::LayoutPreset::DeepseekV41Vllm031Dspark5.runtime_layout();
        for (length, expected) in [
            (128, 0),
            (129, 0),
            (192, 0),
            (193, 192),
            (256, 192),
            (257, 256),
            (12846, 12800),
        ] {
            let tokens: Vec<_> = (0..length).collect();
            let blocks = tokens.len() / 64;
            let hashes: Vec<_> = (0..blocks as u64)
                .map(|hash| hash.to_be_bytes().to_vec())
                .collect();
            let mut index = Index::default();
            index
                .store_group_runtime(
                    8,
                    &hashes,
                    None,
                    &tokens[..blocks * 64],
                    64,
                    64,
                    "m",
                    "",
                    1000,
                )
                .unwrap();
            // No auxiliary group has residents. Only group 8 participates;
            // DSpark=5 causes neither a five-token nor an extra-block drop.
            let keys = prefix_keys(&tokens, 64, "m", "");
            assert_eq!(
                index.matched_keys_runtime(&keys, &layout),
                Some(expected),
                "prompt length={length}"
            );
        }
    }

    #[test]
    fn v41_dense_eviction_stops_at_the_first_missing_block() {
        let layout = super::super::config::LayoutPreset::DeepseekV41Vllm031Dspark5.runtime_layout();
        let tokens: Vec<_> = (0..513).collect();
        let hashes: Vec<_> = (0..8u64).map(|hash| hash.to_be_bytes().to_vec()).collect();
        let keys = prefix_keys(&tokens, 64, "m", "");
        let mut index = Index::default();
        index
            .store_group_runtime(8, &hashes, None, &tokens[..512], 64, 64, "m", "", 1000)
            .unwrap();
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(512));
        // A hole in the dense group breaks the prefix even when later blocks
        // remain resident. A shorter hit inside the replay window saves nothing.
        index.remove_group(8, &hashes[3..4]);
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(192));
        index.remove_group(8, &hashes[2..3]);
        assert_eq!(index.matched_keys_runtime(&keys, &layout), Some(0));
    }

    /// Reference traversal follows vLLM v0.31 FullAttentionManager and
    /// SlidingWindowManager: dense left scan or right-to-left tail scan,
    /// followed by one EAGLE pop and scheduler alignment.
    fn backend_single_group_hit(
        resident: &[bool],
        block_size: usize,
        alignment: usize,
        window: Option<usize>,
        eagle: bool,
    ) -> usize {
        let mut blocks = resident.iter().take_while(|present| **present).count();
        if let Some(window) = window {
            let required = (window - 1).div_ceil(block_size) + usize::from(eagle);
            let mut run = 0;
            let mut matched = None;
            for i in (0..resident.len()).rev() {
                if resident[i] {
                    let after_pop = i + 1 - usize::from(eagle);
                    if run == 0 && after_pop * block_size % alignment != 0 {
                        continue;
                    }
                    run += 1;
                    if run >= required {
                        matched = Some(i + run);
                        break;
                    }
                } else {
                    run = 0;
                }
            }
            blocks = matched.unwrap_or(run);
            if matched.is_none() {
                while blocks * block_size % alignment != 0 {
                    blocks -= 1;
                }
            }
        }
        if eagle {
            blocks = blocks.saturating_sub(1);
        }
        let tokens = blocks * block_size;
        tokens - tokens % alignment
    }

    #[test]
    fn runtime_lookup_matches_backend_traversal_for_all_small_residency_patterns() {
        for block_size in [2, 4] {
            for alignment in [block_size, block_size * 2] {
                for window in [None, Some(block_size * 2), Some(block_size * 3)] {
                    for eagle in [false, true] {
                        let mut layout = super::super::layout::fixture();
                        layout.alignment_tokens = alignment;
                        layout.groups[0].block_size = block_size;
                        layout.groups[0].use_eagle = eagle;
                        layout.groups[0].sliding_window = window;
                        if window.is_some() {
                            layout.groups[0].kind = "sliding_window".into();
                            layout.groups[0].manager = "SlidingWindowManager".into();
                        }
                        let tokens: Vec<_> = (0..(8 * block_size + 1) as u32).collect();
                        let keys = prefix_keys(&tokens, 2, "m", "");
                        let stride = block_size / 2;
                        for mask in 0..256 {
                            let mut index = Index::default();
                            let resident: Vec<_> = (0..8)
                                .map(|i| {
                                    let present = mask & (1 << i) != 0;
                                    if present {
                                        index.resident.insert((0, keys[(i + 1) * stride - 1]), 1);
                                    }
                                    present
                                })
                                .collect();
                            assert_eq!(
                                index.matched_keys_runtime(&keys, &layout),
                                Some(backend_single_group_hit(&resident, block_size, alignment, window, eagle)),
                                "mask={mask}, block={block_size}, alignment={alignment}, window={window:?}, eagle={eagle}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn residency_respects_eviction_parent_and_salt() {
        let mut index = Index::default();
        let hashes = vec![vec![1], vec![2], vec![3]];
        let tokens: Vec<u32> = (0..7).collect();
        index
            .store(&hashes, None, &tokens[..6], 2, "model", "tenant-a", 100)
            .unwrap();
        assert_eq!(index.matched_tokens(&tokens, 2, "model", "tenant-a"), 6);
        assert_eq!(index.matched_tokens(&tokens, 2, "model", "tenant-b"), 0);
        assert_eq!(
            index.matched_tokens(&tokens, 2, "other-model", "tenant-a"),
            0
        );
        index.remove(&hashes[1..2]);
        assert_eq!(index.matched_tokens(&tokens, 2, "model", "tenant-a"), 2);
        assert_eq!(
            index.store(&[vec![4]], Some(&vec![99]), &[8, 9], 2, "model", "", 100),
            Err("missing_parent")
        );
        assert_eq!(index.matched_tokens(&[8, 9, 10], 2, "model", ""), 0);
        index.clear();
        assert_eq!(index.matched_tokens(&tokens, 2, "model", "tenant-a"), 0);
    }

    #[test]
    fn chunked_events_and_repeated_stores_match_the_request_chain() {
        let mut index = Index::default();
        index
            .store(&[vec![1]], None, &[10, 11], 2, "m", "s", 100)
            .unwrap();
        for _ in 0..2 {
            index
                .store(&[vec![2]], Some(&vec![1]), &[12, 13], 2, "m", "", 100)
                .unwrap();
        }
        assert_eq!(index.matched_tokens(&[10, 11, 12, 13, 14], 2, "m", "s"), 4);
        assert_eq!(index.matched_tokens(&[10, 11, 12, 13], 2, "m", "s"), 2);
        index.remove(&[vec![2]]);
        assert_eq!(index.matched_tokens(&[10, 11, 12, 13, 14], 2, "m", "s"), 2);
    }

    fn groups() -> Vec<super::super::config::CacheGroup> {
        use super::super::config::{CacheGroup, CacheKind};
        vec![
            CacheGroup {
                group_idx: 0,
                block_size: 2,
                kind: CacheKind::FullAttention,
                sliding_window: None,
            },
            CacheGroup {
                group_idx: 1,
                block_size: 2,
                kind: CacheKind::SlidingWindow,
                sliding_window: Some(4),
            },
        ]
    }

    #[test]
    fn groups_isolate_residency_and_reuse_swa_tail_after_eviction() {
        let mut index = Index::default();
        let tokens: Vec<_> = (0..9).collect();
        let hashes: Vec<_> = (0..4).map(|n| vec![n]).collect();
        let keys = prefix_keys(&tokens, 2, "m", "");
        index
            .store_group(0, &hashes, None, &tokens[..8], 2, "m", "", 20)
            .unwrap();
        assert_eq!(index.matched_keys_layout(&keys, 2, &groups()), None);
        index
            .store_group(1, &hashes[2..], None, &tokens[..8], 2, "m", "", 20)
            .unwrap();
        assert_eq!(index.matched_keys_layout(&keys, 2, &groups()), Some(8));
        index.remove_group(1, &hashes[3..]);
        assert_eq!(index.matched_keys_layout(&keys, 2, &groups()), Some(0));
        assert_eq!(index.matched_keys(&keys, 2), 8);
        // Repeating a Store must not create an extra residency reference.
        for _ in 0..2 {
            index
                .store_group(1, &hashes[2..], None, &tokens[..8], 2, "m", "", 20)
                .unwrap();
        }
        index.remove_group(1, &hashes[3..]);
        assert_eq!(index.matched_keys_layout(&keys, 2, &groups()), Some(0));
    }

    #[test]
    fn multiple_windows_need_a_common_endpoint_not_minimum_maxima() {
        use super::super::config::{CacheGroup, CacheKind};
        let groups: Vec<_> = (1..=2)
            .map(|group_idx| CacheGroup {
                group_idx,
                block_size: 2,
                kind: CacheKind::SlidingWindow,
                sliding_window: Some(4),
            })
            .collect();
        let tokens: Vec<_> = (0..17).collect();
        let hashes: Vec<_> = (0..8).map(|n| vec![n]).collect();
        let keys = prefix_keys(&tokens, 2, "m", "");
        let mut index = Index::default();
        index
            .learn_dense(&hashes, None, &tokens[..16], 2, "m", "", 30)
            .unwrap();
        for (group, selected) in [
            (
                1,
                vec![
                    hashes[2].clone(),
                    hashes[3].clone(),
                    hashes[6].clone(),
                    hashes[7].clone(),
                ],
            ),
            (2, hashes[2..6].to_vec()),
        ] {
            index
                .store_group(group, &selected, None, &tokens[..16], 2, "m", "", 30)
                .unwrap();
        }
        // Group 1's maximum is 16 tokens and group 2's is 12; 12 is
        // invalid in group 1. Their actual common boundary is 8 tokens.
        assert_eq!(index.matched_keys_layout(&keys, 2, &groups), Some(8));
    }

    #[test]
    fn residency_pressure_is_local_and_keeps_group_reference_counts() {
        // Exceeding the local residency budget drops only the oldest entry.
        // A different group still holding that hash must keep its identity.
        let mut index = Index::default();
        index
            .store_group(0, &[vec![1]], None, &[1, 2], 2, "m", "", 2)
            .unwrap();
        index
            .store_group(1, &[vec![1]], None, &[1, 2], 2, "m", "", 2)
            .unwrap();
        index
            .store_group(0, &[vec![2]], None, &[3, 4], 2, "m", "", 2)
            .unwrap();
        assert_eq!(index.residency_evictions, 1);
        assert_eq!(index.len(), 2);
        assert_eq!(index.matched_tokens(&[1, 2, 5], 2, "m", ""), 0);
        assert_eq!(index.matched_tokens(&[3, 4, 5], 2, "m", ""), 2);
        assert_eq!(index.resident_refs[&vec![1]], 1);
        index.remove_group(0, &[vec![1]]);
        assert_eq!(index.resident_refs[&vec![1]], 1);
        index.remove_group(1, &[vec![1]]);
        assert!(!index.resident_refs.contains_key(&vec![1]));
        index
            .store(&[vec![3]], Some(&vec![1]), &[5, 6], 2, "m", "", 2)
            .unwrap();
        assert_eq!(index.len(), 2);
        // An evicted root never becomes a false prefix hit via a live child.
        assert_eq!(index.matched_tokens(&[1, 2, 5, 6, 7], 2, "m", ""), 0);
        index.clear();
        assert!(index.resident_refs.is_empty());
        assert!(index.unused.is_empty());
    }

    #[test]
    fn eviction_retains_bounded_ancestry_but_clear_discards_it() {
        let mut index = Index::default();
        index
            .store(&[vec![1]], None, &[1, 2], 2, "m", "", 2)
            .unwrap();
        index.remove(&[vec![1]]);
        assert_eq!(index.len(), 0);
        index
            .store(&[vec![2]], Some(&vec![1]), &[3, 4], 2, "m", "", 2)
            .unwrap();
        assert_eq!(index.matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""), 0);
        index
            .store(&[vec![3]], Some(&vec![2]), &[5, 6], 2, "m", "", 2)
            .unwrap();
        assert_eq!(index.len(), 2);
        index.clear();
        assert_eq!(
            index.store(&[vec![2]], Some(&vec![1]), &[3, 4], 2, "m", "", 2),
            Err("missing_parent")
        );
    }

    #[test]
    fn sparse_hashes_are_never_assigned_to_first_token_blocks() {
        let mut index = Index::default();
        assert_eq!(
            index.store_group(1, &[vec![3]], None, &[1, 2, 3, 4, 5, 6], 2, "m", "", 20),
            Err("sparse_hash_unknown")
        );
        assert_eq!(index.len(), 0);
        index
            .learn_dense(
                &[vec![1], vec![2], vec![3]],
                None,
                &[1, 2, 3, 4, 5, 6],
                2,
                "m",
                "",
                20,
            )
            .unwrap();
        index
            .store_group(1, &[vec![3]], None, &[1, 2, 3, 4, 5, 6], 2, "m", "", 20)
            .unwrap();
        let keys = prefix_keys(&[1, 2, 3, 4, 5, 6, 7], 2, "m", "");
        assert!(!index.resident.contains_key(&(1, keys[0])));
        assert!(index.resident.contains_key(&(1, keys[2])));
        assert_eq!(
            index.store_group(1, &[vec![3]], None, &[9, 9, 9, 9], 2, "m", "", 20),
            Err("sparse_hash_mismatch")
        );
    }

    #[test]
    fn sparse_root_recovers_salt_only_from_verified_dense_hashes() {
        let mut index = Index::default();
        let tokens = [1, 2, 3, 4, 5, 6, 7];
        let hashes = [vec![1], vec![2], vec![3]];
        index
            .store_group(0, &hashes, None, &tokens[..6], 2, "m", "tenant-a", 20)
            .unwrap();
        // The SWA event omits block zero, so its extra_keys has no cache_salt.
        index
            .store_group(1, &hashes[1..], None, &tokens[..6], 2, "m", "", 20)
            .unwrap();
        assert_eq!(
            index.matched_keys_layout(&prefix_keys(&tokens, 2, "m", "tenant-a"), 2, &groups()),
            Some(6)
        );
        for salt in ["", "tenant-b"] {
            assert_eq!(
                index.matched_keys_layout(&prefix_keys(&tokens, 2, "m", salt), 2, &groups()),
                Some(0)
            );
        }
        // Eviction retains namespace ancestry, but never retains residency.
        index.remove_group(1, &hashes[1..]);
        assert_eq!(
            index.matched_keys_layout(&prefix_keys(&tokens, 2, "m", "tenant-a"), 2, &groups()),
            Some(0)
        );
        index
            .store_group(1, &hashes[1..], None, &tokens[..6], 2, "m", "tenant-a", 20)
            .unwrap();
        index.clear();
        assert_eq!(
            index.store_group(1, &hashes[1..], None, &tokens[..6], 2, "m", "", 20),
            Err("sparse_hash_unknown")
        );
    }

    #[test]
    fn sparse_salt_conflicts_or_mixed_namespaces_never_create_residency() {
        let mut index = Index::default();
        let tokens = [1, 2, 3, 4, 5, 6];
        let a = [vec![1], vec![2], vec![3]];
        let b = [vec![4], vec![5], vec![6]];
        for (hashes, salt) in [(&a, "tenant-a"), (&b, "tenant-b")] {
            index
                .learn_dense(hashes, None, &tokens, 2, "m", salt, 20)
                .unwrap();
        }
        assert_eq!(
            index.store_group(1, &a[1..], None, &tokens, 2, "m", "tenant-b", 20),
            Err("cache_salt_mismatch")
        );
        assert_eq!(
            index.store_group(
                1,
                &[a[1].clone(), b[2].clone()],
                None,
                &tokens,
                2,
                "m",
                "",
                20
            ),
            Err("sparse_namespace_mismatch")
        );
        assert_eq!(
            index.store_group(1, &[a[1].clone(), vec![99]], None, &tokens, 2, "m", "", 20),
            Err("sparse_hash_unknown")
        );
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn short_swa_prefix_still_requires_root_contiguity() {
        let mut index = Index::default();
        let tokens = [1, 2, 3];
        index
            .store_group(0, &[vec![1]], None, &tokens[..2], 2, "m", "", 20)
            .unwrap();
        index
            .store_group(1, &[vec![1]], None, &tokens[..2], 2, "m", "", 20)
            .unwrap();
        assert_eq!(
            index.matched_keys_layout(&prefix_keys(&tokens, 2, "m", ""), 2, &groups()),
            Some(2)
        );
    }
}
