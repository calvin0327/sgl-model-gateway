//! Decode native vLLM MessagePack events without converting engine hashes to text.
//! Only GPU, single-group full-attention text caches are scored in this version.

use rmpv::Value;
use std::{io::Cursor, time::Instant};

use super::{
    config::Config,
    index::{EngineHash, Index},
};

#[derive(Debug)]
pub enum Event {
    Store {
        hashes: Vec<EngineHash>,
        parent: Option<EngineHash>,
        tokens: Vec<u32>,
        block_size: usize,
        salt: String,
    },
    Remove(Vec<EngineHash>),
    Clear,
    Ignore,
}

fn field<'a>(value: &'a Value, name: &str, position: usize) -> Option<&'a Value> {
    match value {
        Value::Array(items) => items.get(position),
        Value::Map(items) => items
            .iter()
            .find(|(key, _)| key.as_str() == Some(name))
            .map(|(_, value)| value),
        _ => None,
    }
    .filter(|v| !v.is_nil())
}

fn hash(value: &Value) -> Result<EngineHash, &'static str> {
    match value {
        Value::Binary(bytes) if !bytes.is_empty() => {
            let mut id = vec![1];
            id.extend_from_slice(bytes);
            Ok(id)
        }
        Value::Integer(n) => {
            let mut id = vec![0];
            let n = n
                .as_u64()
                .or_else(|| n.as_i64().map(|v| v as u64))
                .ok_or("hash_integer")?;
            id.extend_from_slice(&n.to_be_bytes());
            Ok(id)
        }
        _ => Err("hash_encoding"),
    }
}

fn hashes(value: Option<&Value>) -> Result<Vec<EngineHash>, &'static str> {
    value
        .and_then(Value::as_array)
        .ok_or("hash_array")?
        .iter()
        .map(hash)
        .collect()
}

pub fn decode(payload: &[u8]) -> Result<Vec<Event>, &'static str> {
    let mut cursor = Cursor::new(payload);
    let batch = rmpv::decode::read_value(&mut cursor).map_err(|_| "msgpack")?;
    if cursor.position() != payload.len() as u64 {
        return Err("trailing_bytes");
    }
    if field(&batch, "data_parallel_rank", 2).is_some_and(|v| v.as_u64() != Some(0)) {
        return Err("unsupported_rank");
    }
    let events = field(&batch, "events", 1)
        .and_then(Value::as_array)
        .ok_or("event_batch")?;
    events.iter().map(decode_event).collect()
}

fn decode_event(value: &Value) -> Result<Event, &'static str> {
    let tag = field(value, "type", 0)
        .and_then(Value::as_str)
        .ok_or("event_tag")?;
    if tag == "AllBlocksCleared" {
        return Ok(Event::Clear);
    }
    let (medium_pos, group_pos, locality_pos) = match tag {
        "BlockStored" => (6, 9, 12),
        "BlockRemoved" => (2, 3, 4),
        _ => return Err("unsupported_event"),
    };
    // vLLM 0.28 can publish remote connector events alongside local GPU events.
    // Neither a remote Store nor a remote Remove changes this engine's GPU KV.
    if let Some(locality) = field(value, "locality", locality_pos) {
        match locality.as_str() {
            Some("REMOTE") => return Ok(Event::Ignore),
            Some("LOCAL") => {}
            _ => return Err("unsupported_locality"),
        }
    }
    let medium = field(value, "medium", medium_pos)
        .map(|v| v.as_str().ok_or("unsupported_medium"))
        .transpose()?
        .unwrap_or("gpu");
    if medium.eq_ignore_ascii_case("cpu") || medium.eq_ignore_ascii_case("storage") {
        return Ok(Event::Ignore);
    }
    if !medium.eq_ignore_ascii_case("gpu") {
        return Err("unsupported_medium");
    }
    if let Some(group) = field(value, "group_idx", group_pos) {
        if group.as_u64() != Some(0) {
            return Err("unsupported_group");
        }
    }
    if let Some(kind) = field(value, "kv_cache_spec_kind", 10) {
        if !matches!(kind.as_str(), Some("full_attention" | "FullAttentionSpec")) {
            return Err("unsupported_attention");
        }
    }
    if field(value, "kv_cache_spec_sliding_window", 11).is_some() {
        return Err("unsupported_attention");
    }
    let hashes = hashes(field(value, "block_hashes", 1))?;
    if tag == "BlockRemoved" {
        return Ok(Event::Remove(hashes));
    }
    if field(value, "lora_id", 5).is_some_and(|v| v.as_u64() != Some(0))
        || field(value, "lora_name", 7).is_some_and(|v| v.as_str() != Some(""))
    {
        return Err("unsupported_lora");
    }
    let parent = field(value, "parent_block_hash", 2).map(hash).transpose()?;
    let tokens = field(value, "token_ids", 3)
        .and_then(Value::as_array)
        .ok_or("event_tokens")?
        .iter()
        .map(|n| {
            n.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("token_range")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let block_size = field(value, "block_size", 4)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or("block_size")?;
    let mut salt = String::new();
    if let Some(extra) = field(value, "extra_keys", 8) {
        let entries = extra.as_array().ok_or("extra_keys")?;
        if entries.len() != hashes.len() {
            return Err("extra_keys_shape");
        }
        for (i, entry) in entries.iter().enumerate() {
            if entry.is_nil() {
                continue;
            }
            let keys = entry.as_array().ok_or("extra_keys")?;
            if keys.is_empty() {
                continue;
            }
            // In text-only vLLM, cache_salt is the sole extra key of block zero.
            if i != 0 || parent.is_some() || keys.len() != 1 {
                return Err("unsupported_extra_keys");
            }
            salt = keys[0]
                .as_str()
                .ok_or("unsupported_extra_keys")?
                .to_string();
        }
    }
    Ok(Event::Store {
        hashes,
        parent,
        tokens,
        block_size,
        salt,
    })
}

#[derive(Debug, Default)]
pub struct StreamIndex {
    pub index: Index,
    pub last_seq: Option<u64>,
    pub last_event: Option<Instant>,
    pub unsupported: bool,
}

impl StreamIndex {
    pub fn invalidate(&mut self) {
        self.index.clear();
        self.last_seq = None;
        self.last_event = None;
    }

    /// After a gap, rebuild only positively observed root chains. Old residency
    /// is discarded even when replay is unavailable; unknown parents stay misses.
    pub fn apply(
        &mut self,
        seq: u64,
        payload: &[u8],
        profile: &str,
        cfg: &Config,
    ) -> Result<(), &'static str> {
        if self.last_seq == Some(seq) && seq != 0 {
            return Ok(());
        }
        if self
            .last_seq
            .is_some_and(|last| last.checked_add(1) != Some(seq))
        {
            self.invalidate();
            metrics::counter!("pre_kv_cache_event_resets_total", "reason" => "sequence")
                .increment(1);
        }
        let events = match decode(payload) {
            Ok(events) => events,
            Err(reason) => {
                self.invalidate();
                if reason.starts_with("unsupported_") {
                    self.unsupported = true;
                }
                return Err(reason);
            }
        };
        if self.unsupported {
            return Err("unsupported_stream");
        }
        for event in events {
            match event {
                Event::Store {
                    hashes,
                    parent,
                    tokens,
                    block_size,
                    salt,
                } => {
                    if block_size != cfg.profiles[profile].block_size {
                        self.invalidate();
                        self.unsupported = true;
                        return Err("block_size_mismatch");
                    }
                    if let Err(reason) = self.index.store(
                        &hashes,
                        parent.as_ref(),
                        &tokens,
                        block_size,
                        profile,
                        &salt,
                        cfg.max_blocks_per_worker,
                    ) {
                        if reason != "missing_parent" {
                            self.invalidate();
                            return Err(reason);
                        }
                        metrics::counter!("pre_kv_cache_event_errors_total", "reason" => reason)
                            .increment(1);
                    }
                }
                Event::Remove(hashes) => self.index.remove(&hashes),
                Event::Clear => self.index.clear(),
                Event::Ignore => {}
            }
        }
        self.last_seq = Some(seq);
        self.last_event = Some(Instant::now());
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::config::Profile;
    use super::*;

    pub fn payload(events: Vec<Value>) -> Vec<u8> {
        let value = Value::Array(vec![0.0.into(), Value::Array(events), 0.into()]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &value).unwrap();
        bytes
    }
    pub fn stored(hash: u64, parent: Value, tokens: &[u32]) -> Value {
        Value::Array(vec![
            "BlockStored".into(),
            Value::Array(vec![hash.into()]),
            parent,
            Value::Array(tokens.iter().map(|n| (*n).into()).collect()),
            (tokens.len() as u64).into(),
        ])
    }
    pub fn config() -> Config {
        let mut cfg = Config::default();
        cfg.profiles.insert(
            "m".into(),
            Profile {
                model: "m".into(),
                token_source: Default::default(),
                render_url: "http://localhost:8000".into(),
                render_api_key_env: None,
                block_size: 2,
                cost: Default::default(),
            },
        );
        cfg
    }
    #[test]
    fn gap_clears_old_residency_and_orphans_do_not_become_roots() {
        let cfg = config();
        let mut state = StreamIndex::default();
        state
            .apply(0, &payload(vec![stored(1, Value::Nil, &[1, 2])]), "m", &cfg)
            .unwrap();
        assert_eq!(state.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 2);
        state
            .apply(2, &payload(vec![stored(2, 1.into(), &[3, 4])]), "m", &cfg)
            .unwrap();
        assert_eq!(state.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 0);
        assert_eq!(state.index.matched_tokens(&[3, 4, 5], 2, "m", ""), 0);
        state
            .apply(3, &payload(vec![stored(3, Value::Nil, &[5, 6])]), "m", &cfg)
            .unwrap();
        state
            .apply(3, &payload(vec![stored(3, Value::Nil, &[5, 6])]), "m", &cfg)
            .unwrap();
        assert_eq!(state.index.len(), 1);
        state
            .apply(
                4,
                &payload(vec![Value::Array(vec!["AllBlocksCleared".into()])]),
                "m",
                &cfg,
            )
            .unwrap();
        assert_eq!(state.index.len(), 0);
    }
    #[test]
    fn map_encoding_and_binary_hashes_are_supported() {
        let event = Value::Map(vec![
            ("type".into(), "BlockStored".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![1; 32])]),
            ),
            ("token_ids".into(), Value::Array(vec![1.into(), 2.into()])),
            ("block_size".into(), 2.into()),
        ]);
        assert!(
            matches!(&decode(&payload(vec![event])).unwrap()[0], Event::Store {hashes,..} if hashes[0].len()==33)
        );
    }

    #[test]
    fn vllm_028_remote_events_do_not_create_or_evict_local_gpu_blocks() {
        let mut local = stored(1, Value::Nil, &[1, 2]).as_array().unwrap().clone();
        local.resize(13, Value::Nil);
        local[6] = "GPU".into();
        local[9] = 0.into();
        local[10] = "FullAttentionSpec".into();
        local[12] = "LOCAL".into();
        let mut state = StreamIndex::default();
        state
            .apply(
                0,
                &payload(vec![Value::Array(local.clone())]),
                "m",
                &config(),
            )
            .unwrap();

        let mut remote = local;
        remote[1] = Value::Array(vec![2.into()]);
        remote[3] = Value::Array(vec![3.into(), 4.into()]);
        remote[12] = "REMOTE".into();
        let remove = Value::Array(vec![
            "BlockRemoved".into(),
            Value::Array(vec![1.into()]),
            "GPU".into(),
            0.into(),
            "REMOTE".into(),
        ]);
        state
            .apply(
                1,
                &payload(vec![Value::Array(remote), remove]),
                "m",
                &config(),
            )
            .unwrap();
        assert_eq!(state.index.matched_tokens(&[1, 2, 9], 2, "m", ""), 2);
        assert_eq!(state.index.matched_tokens(&[3, 4, 9], 2, "m", ""), 0);
        assert!(!state.unsupported);
    }

    #[test]
    fn vllm_028_storage_and_remote_map_events_are_ignored() {
        for (medium, locality) in [("STORAGE", "LOCAL"), ("GPU", "REMOTE"), ("CPU", "LOCAL")] {
            for tag in ["BlockStored", "BlockRemoved"] {
                let event = Value::Map(vec![
                    ("type".into(), tag.into()),
                    ("medium".into(), medium.into()),
                    ("locality".into(), locality.into()),
                ]);
                assert!(matches!(
                    decode(&payload(vec![event])).unwrap()[0],
                    Event::Ignore
                ));
            }
        }
        let event = Value::Map(vec![
            ("type".into(), "BlockStored".into()),
            ("locality".into(), "unknown".into()),
        ]);
        assert_eq!(
            decode(&payload(vec![event])).unwrap_err(),
            "unsupported_locality"
        );
    }

    #[test]
    fn unsupported_rank_or_group_cannot_be_treated_as_rank_zero() {
        for rank in [(-1).into(), 1.into(), "zero".into()] {
            let batch = Value::Array(vec![0.into(), Value::Array(vec![]), rank]);
            let mut bytes = Vec::new();
            rmpv::encode::write_value(&mut bytes, &batch).unwrap();
            assert_eq!(decode(&bytes).unwrap_err(), "unsupported_rank");
        }
        let mut state = StreamIndex::default();
        let mut event = stored(1, Value::Nil, &[1, 2]).as_array().unwrap().clone();
        event.resize(10, Value::Nil);
        event[9] = 1.into();
        assert_eq!(
            state.apply(0, &payload(vec![Value::Array(event)]), "m", &config()),
            Err("unsupported_group")
        );
        assert!(state.unsupported);
        assert!(state
            .apply(
                1,
                &payload(vec![stored(2, Value::Nil, &[1, 2])]),
                "m",
                &config()
            )
            .is_err());
        assert_eq!(state.index.len(), 0);
    }

    #[test]
    fn native_cache_salt_matches_only_its_request_namespace() {
        let mut event = stored(1, Value::Nil, &[1, 2]).as_array().unwrap().clone();
        event.resize(9, Value::Nil);
        event[8] = Value::Array(vec![Value::Array(vec!["tenant-a".into()])]);
        let mut state = StreamIndex::default();
        state
            .apply(0, &payload(vec![Value::Array(event)]), "m", &config())
            .unwrap();
        assert_eq!(
            state.index.matched_tokens(&[1, 2, 3], 2, "m", "tenant-a"),
            2
        );
        assert_eq!(
            state.index.matched_tokens(&[1, 2, 3], 2, "m", "tenant-b"),
            0
        );
        assert_eq!(state.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 0);
    }
}
