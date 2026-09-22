//! Event residency and request tokens share one prefix-key function.
//! Engine hashes remain opaque identifiers, scoped to one worker incarnation.

use std::collections::HashMap;

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

#[derive(Debug, Default)]
pub struct Index {
    engine: HashMap<EngineHash, Key>,
    resident: HashMap<Key, usize>,
}

impl Index {
    /// Missing parents are skipped, never reinterpreted as root prefixes.
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
        if block_size == 0 || hashes.len().checked_mul(block_size) != Some(tokens.len()) {
            return Err("block_shape");
        }
        let mut key = match parent {
            Some(parent) => *self.engine.get(parent).ok_or("missing_parent")?,
            None => root(profile, salt),
        };
        let new_hashes = hashes
            .iter()
            .filter(|h| !self.engine.contains_key(*h))
            .count();
        if self.engine.len().saturating_add(new_hashes) > limit {
            return Err("index_capacity");
        }
        for (hash, chunk) in hashes.iter().zip(tokens.chunks_exact(block_size)) {
            key = block_key(key, chunk);
            if let Some(previous) = self.engine.insert(hash.clone(), key) {
                self.remove_membership(&previous);
            }
            *self.resident.entry(key).or_default() += 1;
        }
        Ok(())
    }

    fn remove_membership(&mut self, key: &Key) {
        if let Some(count) = self.resident.get_mut(key) {
            *count -= 1;
            if *count == 0 {
                self.resident.remove(key);
            }
        }
    }

    pub fn remove(&mut self, hashes: &[EngineHash]) {
        for hash in hashes {
            if let Some(key) = self.engine.remove(hash) {
                self.remove_membership(&key);
            }
        }
    }

    pub fn clear(&mut self) {
        self.engine.clear();
        self.resident.clear();
    }
    pub fn len(&self) -> usize {
        self.engine.len()
    }

    /// Check every full block: an evicted middle block terminates the match.
    /// vLLM must execute at least the last prompt token to obtain logits.
    pub fn matched_keys(&self, keys: &[Key], block_size: usize) -> usize {
        keys.iter()
            .take_while(|key| self.resident.contains_key(*key))
            .count()
            * block_size
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
}
