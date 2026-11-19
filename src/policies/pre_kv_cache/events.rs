//! Decode native vLLM MessagePack events without converting engine hashes to text.
//! Residency is scoped to declared GPU cache groups; unknown layouts fail closed.

use rmpv::Value;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io::Cursor,
    time::Instant,
};

use super::{
    config::{CacheGroup, CacheKind, Config},
    index::{EngineHash, Index},
};

#[derive(Debug)]
pub enum Event {
    Store {
        group: u64,
        kind: Option<CacheKind>,
        sliding_window: Option<usize>,
        hashes: Vec<EngineHash>,
        parent: Option<EngineHash>,
        tokens: Vec<u32>,
        block_size: usize,
        salt: String,
        skip: Option<StoreSkip>,
    },
    Remove {
        group: u64,
        hashes: Vec<EngineHash>,
    },
    Clear,
    Ignore,
}

#[derive(Debug)]
pub struct StoreSkip {
    from: usize,
    reason: &'static str,
    structure: String,
}

#[derive(Debug, serde::Serialize)]
pub struct SkipSummary {
    pub sequence: u64,
    pub group: u64,
    pub reason: &'static str,
    pub first_skipped_block: usize,
    pub skipped_blocks: usize,
    pub has_parent: bool,
    pub hash_count: usize,
    pub token_count: usize,
    pub medium: &'static str,
    pub extra_structure: String,
}

// Report only bounded schema information: never tokens, salt values or binary
// contents. The actual fixed vLLM image may differ from the upstream release.
fn value_shape(value: &Value, depth: usize) -> String {
    match value {
        Value::Nil => "null".into(),
        Value::String(value) => format!("string({})", value.as_bytes().len()),
        Value::Binary(value) => format!("bytes({})", value.len()),
        Value::Array(values) if depth < 2 => format!(
            "array({})[{}]",
            values.len(),
            values
                .iter()
                .take(4)
                .map(|v| value_shape(v, depth + 1))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Array(values) => format!("array({})", values.len()),
        Value::Map(values) => format!("map({})", values.len()),
        Value::Integer(_) => "integer".into(),
        Value::Boolean(_) => "boolean".into(),
        _ => "other".into(),
    }
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

#[cfg(test)]
pub fn decode(payload: &[u8]) -> Result<Vec<Event>, &'static str> {
    decode_with_layout(payload, None, false)
}

fn decode_with_layout(
    payload: &[u8],
    layout: Option<&super::layout::RuntimeLayout>,
    check_auxiliary_layout: bool,
) -> Result<Vec<Event>, &'static str> {
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
    events
        .iter()
        .map(|event| {
            // Auxiliary groups never supply prefix hits. A fixed preset still
            // checks their published shape before ignoring them, so a changed
            // backend layout cannot silently reuse the old routing contract.
            let position = match field(event, "type", 0).and_then(Value::as_str) {
                Some("BlockStored") => Some(9),
                Some("BlockRemoved") => Some(3),
                _ => None,
            };
            if let Some(group) = position
                .and_then(|pos| match field(event, "group_idx", pos) {
                    Some(group) => group.as_u64(),
                    // Match decode_event's default group without letting an
                    // omitted optional ID bypass preset shape validation.
                    None if check_auxiliary_layout => Some(0),
                    None => None,
                })
                .and_then(|id| layout.and_then(|layout| layout.group(id)))
                .filter(|group| !group.prefix_cacheable)
            {
                if check_auxiliary_layout && position == Some(9) {
                    // Remote/storage events do not describe this worker's GPU
                    // layout. Nullable optional spec fields are unspecified;
                    // block_size is required by the BlockStored protocol.
                    let remote =
                        field(event, "locality", 12).and_then(Value::as_str) == Some("REMOTE");
                    let external = field(event, "medium", 6)
                        .and_then(Value::as_str)
                        .is_some_and(|medium| {
                            medium.eq_ignore_ascii_case("cpu")
                                || medium.eq_ignore_ascii_case("storage")
                        });
                    if !remote && !external {
                        let block_size = field(event, "block_size", 4).and_then(Value::as_u64);
                        let kind_matches = field(event, "kv_cache_spec_kind", 10)
                            .is_none_or(|kind| kind.as_str() == Some(group.kind.as_str()));
                        let window_matches = field(event, "kv_cache_spec_sliding_window", 11)
                            .is_none_or(|window| {
                                window.as_u64().is_some_and(|window| {
                                    Some(window) == group.sliding_window.map(|value| value as u64)
                                })
                            });
                        if block_size != Some(group.block_size as u64)
                            || !kind_matches
                            || !window_matches
                        {
                            return Err("cache_group_mismatch");
                        }
                    }
                }
                Ok(Event::Ignore)
            } else {
                decode_event(event)
            }
        })
        .collect()
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
    let group = field(value, "group_idx", group_pos)
        .map(|value| value.as_u64().ok_or("unsupported_group"))
        .transpose()?
        .unwrap_or(0);
    let hashes = hashes(field(value, "block_hashes", 1))?;
    if tag == "BlockRemoved" {
        return Ok(Event::Remove { group, hashes });
    }
    let kind = field(value, "kv_cache_spec_kind", 10)
        .map(|value| match value.as_str() {
            Some("full_attention" | "mla_attention" | "FullAttentionSpec") => {
                Ok(CacheKind::FullAttention)
            }
            Some("sliding_window" | "sliding_window_mla" | "SlidingWindowSpec") => {
                Ok(CacheKind::SlidingWindow)
            }
            _ => Err("unsupported_attention"),
        })
        .transpose()?;
    let sliding_window = field(value, "kv_cache_spec_sliding_window", 11)
        .map(|value| {
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0)
                .ok_or("unsupported_attention")
        })
        .transpose()?;
    // Optional metadata can be completed from a validated group contract. An
    // explicit full-attention/window contradiction is still invalid.
    if kind == Some(CacheKind::FullAttention) && sliding_window.is_some() {
        return Err("unsupported_attention");
    }
    let lora = field(value, "lora_id", 5).is_some_and(|v| v.as_u64() != Some(0))
        || field(value, "lora_name", 7).is_some_and(|v| v.as_str() != Some(""));
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
    let mut skip = lora.then(|| StoreSkip {
        from: 0,
        reason: "unsupported_lora",
        structure: "lora_identity".into(),
    });
    if let Some(extra) = field(value, "extra_keys", 8) {
        // A known Store with unknown identity is safe to omit, unlike an
        // unknown action that might hide a deletion. Keep the batch decodable.
        let unsupported =
            if let Some(entries) = extra.as_array().filter(|v| v.len() == hashes.len()) {
                entries.iter().enumerate().find_map(|(i, entry)| {
                    if entry.is_nil() || entry.as_array().is_some_and(Vec::is_empty) {
                        return None;
                    }
                    if i == 0 && parent.is_none() {
                        if let Some(text) = entry
                            .as_array()
                            .filter(|keys| keys.len() == 1)
                            .and_then(|keys| keys[0].as_str())
                        {
                            salt = text.to_owned();
                            return None;
                        }
                    }
                    Some((i, value_shape(entry, 0)))
                })
            } else {
                Some((0, value_shape(extra, 0)))
            };
        if skip.is_none() {
            skip = unsupported.map(|(from, structure)| StoreSkip {
                from,
                reason: "unsupported_extra_keys",
                structure,
            });
        }
    }
    Ok(Event::Store {
        group,
        kind,
        sliding_window,
        hashes,
        parent,
        tokens,
        block_size,
        salt,
        skip,
    })
}

#[derive(Debug, Default)]
pub struct StreamIndex {
    pub layout: Option<std::sync::Arc<super::layout::RuntimeLayout>>,
    pub index: Index,
    pub last_seq: Option<u64>,
    pub last_event: Option<Instant>,
    pub unsupported: bool,
    pub observed_groups: BTreeMap<u64, CacheGroup>,
    pub last_error: Option<&'static str>,
    // Missing history permits only rooted, positively observed branches. A
    // partial checkpoint can be continued, but must never be called complete.
    pub history_complete: bool,
    pub history_truncated_at: Option<u64>,
    incomplete_groups: BTreeMap<u64, &'static str>,
    pub skipped_stores: u64,
    pub skipped_blocks: u64,
    pub last_skip: Option<SkipSummary>,
}

impl StreamIndex {
    // Preserve only a provably aligned prefix; an opaque parent taints every
    // descendant. Sparse spans cannot be cut by simply zipping hashes/tokens.
    fn isolate_store(
        &mut self,
        event: &mut Event,
        seq: u64,
        limit: usize,
    ) -> Result<Vec<EngineHash>, &'static str> {
        let Event::Store {
            group,
            hashes,
            parent,
            tokens,
            block_size,
            skip,
            ..
        } = event
        else {
            return Ok(Vec::new());
        };
        let inherited = parent
            .as_ref()
            .is_some_and(|hash| self.index.is_opaque(hash));
        let from = if inherited {
            Some(0)
        } else {
            hashes
                .iter()
                .position(|hash| self.index.is_opaque(hash))
                .into_iter()
                .chain(skip.as_ref().map(|skip| skip.from))
                .min()
        };
        let Some(from) = from else {
            return Ok(Vec::new());
        };
        let from = from.min(hashes.len());
        let isolated = hashes[from..].to_vec();
        self.index.quarantine(&hashes[from..], limit)?;
        let retained = if !inherited && hashes.len().checked_mul(*block_size) == Some(tokens.len())
        {
            from
        } else {
            0
        };
        let reason = skip
            .as_ref()
            .map_or("unsupported_parent", |skip| skip.reason);
        self.skipped_stores = self.skipped_stores.saturating_add(1);
        self.skipped_blocks = self
            .skipped_blocks
            .saturating_add((hashes.len() - retained) as u64);
        // Keep the original schema sample when subsequent descendants are skipped.
        if skip.is_some() || self.last_skip.is_none() {
            self.last_skip = Some(SkipSummary {
                sequence: seq,
                group: *group,
                reason,
                first_skipped_block: from,
                skipped_blocks: hashes.len() - retained,
                has_parent: parent.is_some(),
                hash_count: hashes.len(),
                token_count: tokens.len(),
                medium: "GPU",
                extra_structure: skip.as_ref().map_or_else(
                    || "inherited_identity".into(),
                    |skip| skip.structure.clone(),
                ),
            });
        }
        if retained == 0 {
            *event = Event::Ignore;
        } else {
            hashes.truncate(retained);
            tokens.truncate(retained * *block_size);
            *skip = None;
        }
        Ok(isolated)
    }

    // Rebuild against the same validated layout, without inheriting residency
    // or claiming that an empty dictionary is complete history.
    pub fn empty_with_layout(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            ..Default::default()
        }
    }

    pub fn incomplete(&self) -> bool {
        !self.incomplete_groups.is_empty()
    }

    pub fn invalidate(&mut self) {
        self.index.clear();
        self.last_seq = None;
        self.last_event = None;
        self.incomplete_groups.clear();
        self.history_complete = false;
        self.history_truncated_at = None;
    }

    /// A new publisher epoch must revalidate its format and observed groups.
    /// Ordinary malformed frames only invalidate residency, not compatibility.
    pub fn reset_epoch(&mut self) {
        *self = self.empty_with_layout();
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
        let result = self.apply_inner(seq, payload, profile, cfg);
        if let Err(reason) = result {
            if reason != "unsupported_stream" || self.last_error.is_none() {
                self.last_error = Some(reason);
            }
        } else {
            self.last_error = None;
        }
        result
    }

    fn apply_inner(
        &mut self,
        seq: u64,
        payload: &[u8],
        profile: &str,
        cfg: &Config,
    ) -> Result<(), &'static str> {
        if self.last_seq == Some(seq) && seq != 0 {
            // Duplicate delivery neither repairs history nor fails a new batch.
            return Ok(());
        }
        if seq == 0 || self.last_seq.is_some_and(|last| seq < last) {
            self.reset_epoch();
        }
        if self
            .last_seq
            .is_some_and(|last| last.checked_add(1) != Some(seq))
        {
            self.invalidate();
            metrics::counter!("pre_kv_cache_event_resets_total", "reason" => "sequence")
                .increment(1);
        }
        // Built-in layouts have no live engine identity, whether explicitly
        // selected or automatically recognized. Keep their shape checks with
        // the runtime layout so a replay inherits the same validation.
        let check_auxiliary_layout = self
            .layout
            .as_ref()
            .is_some_and(|layout| layout.engine_id.is_empty());
        let mut events =
            match decode_with_layout(payload, self.layout.as_deref(), check_auxiliary_layout) {
                Ok(events) => events,
                Err(reason) => {
                    self.invalidate();
                    if reason.starts_with("unsupported_") || reason == "cache_group_mismatch" {
                        self.unsupported = true;
                    }
                    return Err(reason);
                }
            };
        if self.unsupported {
            // A valid authoritative clear discards the incompatible epoch. Do
            // not apply any pre-clear Stores or revive their old residency.
            if let Some(clear) = events
                .iter()
                .rposition(|event| matches!(event, Event::Clear))
            {
                self.reset_epoch();
                events.drain(..clear);
            } else {
                return Err("unsupported_stream");
            }
        }
        let profile_config = &cfg.profiles[profile];
        // Complete optional fields in publication order, including later Stores
        // in the same batch. Explicit metadata remains visible on rejection;
        // the entire contract is validated below before learning any keys.
        for event in &mut events {
            let Event::Store {
                group,
                kind,
                sliding_window,
                block_size,
                ..
            } = event
            else {
                continue;
            };
            // Built-in geometry is a hypothesis until this epoch publishes an
            // explicit kind. Runtime metadata and manual groups are contracts.
            let contract = self
                .observed_groups
                .get(group)
                .map(|g| (g.kind, g.sliding_window))
                .or_else(|| {
                    self.layout
                        .as_ref()
                        .filter(|layout| !layout.engine_id.is_empty())
                        .and_then(|layout| layout.group(*group))
                        .and_then(|g| g.cache_kind().map(|kind| (kind, g.sliding_window)))
                })
                .or_else(|| {
                    profile_config
                        .cache_groups
                        .iter()
                        .find(|g| g.group_idx == *group)
                        .map(|g| (g.kind, g.sliding_window))
                });
            if let Some((known_kind, known_window)) = contract {
                *kind = kind.or(Some(known_kind));
                if *kind == Some(CacheKind::SlidingWindow) && sliding_window.is_none() {
                    *sliding_window = known_window;
                }
            }
            if let Some(kind) = *kind {
                if self.observed_groups.len() < 64 || self.observed_groups.contains_key(group) {
                    self.observed_groups.insert(
                        *group,
                        CacheGroup {
                            group_idx: *group,
                            block_size: *block_size,
                            kind,
                            sliding_window: *sliding_window,
                        },
                    );
                }
            }
        }
        if self.last_seq.is_none() && seq == 0 {
            self.history_complete = true;
        }
        // Validate every group before sharing its content keys. Undeclared or
        // differently-sized groups must never seed another group's dictionary.
        for event in &events {
            let failure = match event {
                Event::Store {
                    group,
                    kind,
                    sliding_window,
                    block_size,
                    ..
                } => {
                    if let Some(layout) = &self.layout {
                        match layout.groups.iter().find(|entry| entry.group_idx == *group) {
                            Some(entry) if !entry.prefix_cacheable => continue,
                            Some(entry) => {
                                if kind.is_none() {
                                    Some("cache_group_metadata_missing")
                                } else if entry.cache_kind() != *kind
                                    || entry.block_size != *block_size
                                    || entry.sliding_window != *sliding_window
                                {
                                    Some("cache_group_mismatch")
                                } else {
                                    None
                                }
                            }
                            None => Some("unsupported_group"),
                        }
                    } else {
                        let declared = profile_config
                            .cache_groups
                            .iter()
                            .find(|entry| entry.group_idx == *group);
                        if *block_size != profile_config.block_size {
                            Some("block_size_mismatch")
                        } else if profile_config.cache_groups.is_empty() {
                            if *group != 0 {
                                Some("unsupported_group")
                            } else if *kind == Some(CacheKind::SlidingWindow)
                                || sliding_window.is_some()
                            {
                                Some("unsupported_attention")
                            } else {
                                None
                            }
                        } else if let Some(entry) = declared {
                            (Some(entry.kind) != *kind
                                || entry.block_size != *block_size
                                || entry.sliding_window != *sliding_window)
                                .then_some("cache_group_mismatch")
                        } else {
                            Some("unsupported_group")
                        }
                    }
                }
                Event::Remove { group, .. } => {
                    let declared = if let Some(layout) = &self.layout {
                        layout.groups.iter().any(|entry| entry.group_idx == *group)
                    } else if profile_config.cache_groups.is_empty() {
                        *group == 0
                    } else {
                        profile_config
                            .cache_groups
                            .iter()
                            .any(|entry| entry.group_idx == *group)
                    };
                    (!declared).then_some("unsupported_group")
                }
                _ => None,
            };
            if let Some(reason) = failure {
                self.invalidate();
                self.unsupported = reason != "cache_group_metadata_missing";
                return Err(reason);
            }
        }
        // A clear separates content dictionaries as well as residency. Learn
        // ahead only inside each segment so an old hash cannot bridge resets.
        let mut batch_error = None;
        for segment in events.split_inclusive_mut(|event| matches!(event, Event::Clear)) {
            // Propagate identity rejection before any cross-group learning.
            // Wake only affected Stores, so reversed descendant chains do not
            // require repeated scans of the whole batch or spuriously replay.
            let mut dependents: HashMap<EngineHash, Vec<usize>> = HashMap::new();
            for (position, event) in segment.iter().enumerate() {
                if let Event::Store { hashes, parent, .. } = event {
                    for hash in hashes.iter().chain(parent.iter()) {
                        dependents.entry(hash.clone()).or_default().push(position);
                    }
                }
            }
            let mut pending: VecDeque<_> = (0..segment.len()).collect();
            while let Some(position) = pending.pop_front() {
                let isolated = match self.isolate_store(
                    &mut segment[position],
                    seq,
                    cfg.max_blocks_per_worker,
                ) {
                    Ok(hashes) => hashes,
                    Err(reason) => {
                        self.invalidate();
                        return Err(reason);
                    }
                };
                for hash in isolated {
                    if let Some(positions) = dependents.remove(&hash) {
                        pending.extend(positions);
                    }
                }
            }
            for event in segment.iter_mut() {
                if let Err(reason) = self.isolate_store(event, seq, cfg.max_blocks_per_worker) {
                    self.invalidate();
                    return Err(reason);
                }
                if let Event::Store {
                    group,
                    hashes,
                    parent,
                    tokens,
                    block_size,
                    salt,
                    ..
                } = event
                {
                    if self.layout.as_ref().is_some_and(|layout| {
                        layout
                            .groups
                            .iter()
                            .any(|entry| entry.group_idx == *group && !entry.prefix_cacheable)
                    }) {
                        continue;
                    }
                    if hashes.len().checked_mul(*block_size) == Some(tokens.len()) {
                        if let Err(reason) = self.index.learn_dense_runtime(
                            hashes,
                            parent.as_ref(),
                            tokens,
                            *block_size,
                            self.layout
                                .as_ref()
                                .map_or(*block_size, |layout| layout.hash_block_size),
                            profile,
                            salt,
                            cfg.max_blocks_per_worker,
                        ) {
                            if reason != "missing_parent" {
                                self.invalidate();
                                return Err(reason);
                            }
                        }
                    }
                }
            }
            // Content learning above does not mark residency. Apply all physical
            // mutations in their original order, including removes after stores.
            for event in segment.iter_mut() {
                // A later dense event may have resolved or quarantined a parent.
                if let Err(reason) = self.isolate_store(event, seq, cfg.max_blocks_per_worker) {
                    self.invalidate();
                    return Err(reason);
                }
                match event {
                    Event::Store {
                        group,
                        hashes,
                        parent,
                        tokens,
                        block_size,
                        salt,
                        ..
                    } => {
                        if self.layout.as_ref().is_some_and(|layout| {
                            layout
                                .groups
                                .iter()
                                .any(|entry| entry.group_idx == *group && !entry.prefix_cacheable)
                        }) {
                            continue;
                        }
                        let previous_evictions = self.index.residency_evictions;
                        if let Err(reason) = self.index.store_group_runtime(
                            *group,
                            hashes,
                            parent.as_ref(),
                            tokens,
                            *block_size,
                            self.layout
                                .as_ref()
                                .map_or(*block_size, |layout| layout.hash_block_size),
                            profile,
                            salt,
                            cfg.max_blocks_per_worker,
                        ) {
                            if matches!(reason, "missing_parent" | "sparse_hash_unknown") {
                                self.incomplete_groups.insert(*group, reason);
                                self.history_complete = false;
                                batch_error.get_or_insert(reason);
                            } else {
                                self.invalidate();
                                return Err(reason);
                            }
                        }
                        if self.index.residency_evictions != previous_evictions {
                            self.history_complete = false;
                        }
                    }
                    Event::Remove { group, hashes } => self.index.remove_group(*group, hashes),
                    Event::Clear => {
                        self.index.clear();
                        self.incomplete_groups.clear();
                        self.history_complete = true;
                        self.history_truncated_at = None;
                        self.last_skip = None;
                    }
                    Event::Ignore => {}
                }
            }
        }
        self.last_seq = Some(seq);
        self.last_event = Some(Instant::now());
        // Coverage can remain partial after a successful independent Store.
        // Only this batch's actual failures contribute to apply failure counts.
        batch_error.map_or(Ok(()), Err)
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

    /// Reproduce the measured V4.1 group metadata without a backend process.
    pub fn v41_store(group: u64, hashes: &[u64], tokens: &[u32]) -> Value {
        let layout = super::super::config::LayoutPreset::DeepseekV41Vllm031Dspark5.runtime_layout();
        let spec = layout.group(group).unwrap();
        Value::Array(vec![
            "BlockStored".into(),
            Value::Array(hashes.iter().map(|hash| (*hash).into()).collect()),
            Value::Nil,
            Value::Array(tokens.iter().map(|token| (*token).into()).collect()),
            (spec.block_size as u64).into(),
            Value::Nil,
            "GPU".into(),
            Value::Nil,
            Value::Nil,
            group.into(),
            spec.kind.as_str().into(),
            spec.sliding_window
                .map(|window| Value::from(window as u64))
                .unwrap_or(Value::Nil),
            "LOCAL".into(),
        ])
    }

    fn v41_config() -> Config {
        let mut cfg = config();
        let profile = cfg.profiles.get_mut("m").unwrap();
        profile.block_size = 64;
        profile.layout = Some(super::super::config::LayoutPreset::DeepseekV41Vllm031Dspark5);
        cfg
    }

    fn v41_stream() -> StreamIndex {
        StreamIndex {
            layout: Some(std::sync::Arc::new(
                super::super::config::LayoutPreset::DeepseekV41Vllm031Dspark5.runtime_layout(),
            )),
            ..Default::default()
        }
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
                cache_groups: Vec::new(),
                layout: None,
            },
        );
        cfg
    }

    #[test]
    fn v41_preset_indexes_only_dense_group_and_ignores_auxiliary_removals() {
        let tokens: Vec<_> = (0..257).collect();
        let cfg = v41_config();
        let mut stream = v41_stream();
        let mut events = vec![v41_store(8, &[1, 2, 3, 4], &tokens[..256])];
        for group in (0..8).chain(std::iter::once(9)) {
            events.push(v41_store(group, &[1], &tokens[..32]));
            events.push(Value::Array(vec![
                "BlockRemoved".into(),
                Value::Array(vec![1.into()]),
                "GPU".into(),
                group.into(),
                "LOCAL".into(),
            ]));
        }
        stream.apply(0, &payload(events), "m", &cfg).unwrap();
        let keys = super::super::index::prefix_keys(&tokens, 64, "m", "");
        assert_eq!(stream.index.len(), 4);
        assert_eq!(
            stream
                .index
                .matched_keys_runtime(&keys, stream.layout.as_ref().unwrap()),
            Some(256)
        );
    }

    #[test]
    fn v41_preset_rejects_changed_auxiliary_shape_before_indexing_the_batch() {
        // These groups do not contribute hits, but a changed shape is evidence
        // that this worker no longer matches the explicitly selected preset.
        for (group, position, value) in [
            (0, 4, 64.into()),
            (0, 4, Value::Nil),
            (0, 10, "mla_attention".into()),
            (0, 11, 256.into()),
            (9, 4, 32.into()),
            (9, 10, "mla_attention".into()),
            (9, 11, 128.into()),
            (9, 11, "invalid".into()),
        ] {
            let tokens: Vec<_> = (0..256).collect();
            let mut auxiliary = v41_store(group, &[1], &tokens[..32])
                .as_array()
                .unwrap()
                .clone();
            auxiliary[position] = value;
            let batch = payload(vec![
                v41_store(8, &[1, 2, 3, 4], &tokens),
                Value::Array(auxiliary),
            ]);
            let mut stream = v41_stream();
            assert_eq!(
                stream.apply(0, &batch, "m", &v41_config()),
                Err("cache_group_mismatch"),
                "group={group}, position={position}"
            );
            assert!(stream.unsupported);
            assert_eq!(stream.index.len(), 0);
            assert_eq!(
                stream.apply(1, &payload(vec![]), "m", &v41_config()),
                Err("unsupported_stream")
            );

            // Authoritative runtime metadata already determines the ignored
            // groups; its existing decode behavior does not use preset checks.
            let mut cfg = v41_config();
            cfg.profiles.get_mut("m").unwrap().layout = None;
            let mut dynamic = v41_stream();
            std::sync::Arc::make_mut(dynamic.layout.as_mut().unwrap()).engine_id =
                "worker-engine".into();
            dynamic.layout.as_ref().unwrap().validate().unwrap();
            dynamic.apply(0, &batch, "m", &cfg).unwrap();
            assert_eq!(dynamic.index.len(), 4);
        }
    }

    #[test]
    fn v41_automatic_layout_and_replay_accept_matching_groups_without_cli_preset() {
        // Automatic selection is stored in the stream, not profile.layout.
        // Copying only that layout into replay must preserve usable dense hits.
        let mut cfg = v41_config();
        cfg.profiles.get_mut("m").unwrap().layout = None;
        let tokens: Vec<_> = (0..257).collect();
        let mut events = vec![v41_store(8, &[1, 2, 3, 4], &tokens[..256])];
        for group in (0..8).chain(std::iter::once(9)) {
            events.push(v41_store(group, &[1], &tokens[..32]));
        }
        let batch = payload(events);
        let keys = super::super::index::prefix_keys(&tokens, 64, "m", "");
        let mut live = v41_stream();
        live.apply(0, &batch, "m", &cfg).unwrap();
        let mut replay = StreamIndex {
            layout: live.layout.clone(),
            ..Default::default()
        };
        replay.apply(0, &batch, "m", &cfg).unwrap();
        for stream in [&live, &replay] {
            assert!(!stream.unsupported);
            assert_eq!(stream.index.len(), 4);
            assert_eq!(
                stream
                    .index
                    .matched_keys_runtime(&keys, stream.layout.as_ref().unwrap()),
                Some(256)
            );
        }
    }

    #[test]
    fn v41_automatic_layout_rejects_auxiliary_unknown_and_dense_mismatches() {
        // A missing CLI flag must not weaken the automatic layout's guards,
        // including when replay copies the layout into a fresh stream.
        let mut cfg = v41_config();
        cfg.profiles.get_mut("m").unwrap().layout = None;
        for (group, position, value, reason) in [
            (0, 4, 64.into(), "cache_group_mismatch"),
            (0, 10, "mla_attention".into(), "cache_group_mismatch"),
            (0, 11, 256.into(), "cache_group_mismatch"),
            (9, 4, 32.into(), "cache_group_mismatch"),
            (8, 9, 10.into(), "unsupported_group"),
            (8, 4, 32.into(), "cache_group_mismatch"),
            (8, 10, "sliding_window_mla".into(), "cache_group_mismatch"),
        ] {
            let tokens: Vec<_> = (0..256).collect();
            let valid = payload(vec![v41_store(8, &[1, 2, 3, 4], &tokens)]);
            let mut changed = v41_store(group, &[1], &tokens[..64])
                .as_array()
                .unwrap()
                .clone();
            changed[position] = value;
            let rejected = payload(vec![Value::Array(changed)]);
            let mut live = v41_stream();
            live.apply(0, &valid, "m", &cfg).unwrap();
            let mut replay = StreamIndex {
                layout: live.layout.clone(),
                ..Default::default()
            };
            for stream in [&mut live, &mut replay] {
                assert_eq!(
                    stream.apply(1, &rejected, "m", &cfg),
                    Err(reason),
                    "group={group}, position={position}"
                );
                assert!(stream.unsupported);
                assert_eq!(stream.index.len(), 0);
                assert_eq!(
                    stream.apply(2, &payload(vec![]), "m", &cfg),
                    Err("unsupported_stream")
                );
            }
        }
    }

    #[test]
    fn v41_auxiliary_optional_metadata_can_be_nil_or_omitted() {
        // vLLM declares kind/window optional and omits default values on the
        // wire. Their absence cannot establish a different auxiliary layout.
        let mut omitted = v41_store(0, &[1], &[0; 32]).as_array().unwrap().clone();
        omitted.truncate(10);
        let mut nil = v41_store(1, &[1], &[0; 32]).as_array().unwrap().clone();
        nil[10] = Value::Nil;
        nil[11] = Value::Nil;
        let map = Value::Map(vec![
            ("type".into(), "BlockStored".into()),
            ("group_idx".into(), 9.into()),
            ("block_size".into(), 8.into()),
        ]);
        let mut stream = v41_stream();
        stream
            .apply(
                0,
                &payload(vec![Value::Array(omitted), Value::Array(nil), map]),
                "m",
                &v41_config(),
            )
            .unwrap();
        assert!(!stream.unsupported);
        assert_eq!(stream.index.len(), 0);

        // A missing optional group ID defaults to group 0, whose required
        // block size must still match the preset before the event is ignored.
        for group in [Value::Nil, 0.into()] {
            let mut invalid = v41_store(0, &[1], &[0; 32]).as_array().unwrap().clone();
            invalid[9] = group;
            invalid[4] = 64.into();
            assert_eq!(
                v41_stream().apply(0, &payload(vec![Value::Array(invalid)]), "m", &v41_config()),
                Err("cache_group_mismatch")
            );
        }
    }

    #[test]
    fn v41_preset_does_not_validate_remote_auxiliary_storage_as_local_gpu_layout() {
        for (position, value) in [(12, "REMOTE"), (6, "CPU"), (6, "STORAGE")] {
            let mut event = v41_store(0, &[1], &[0; 32]).as_array().unwrap().clone();
            event[position] = value.into();
            event[4] = 999.into();
            event[10] = "external".into();
            let mut stream = v41_stream();
            stream
                .apply(0, &payload(vec![Value::Array(event)]), "m", &v41_config())
                .unwrap();
            assert!(!stream.unsupported);
        }
    }

    #[test]
    fn v41_preset_rejects_unknown_groups_and_changed_dense_metadata() {
        for (position, value, reason) in [
            (9, 10.into(), "unsupported_group"),
            (4, 32.into(), "cache_group_mismatch"),
            (10, "sliding_window_mla".into(), "cache_group_mismatch"),
        ] {
            let mut event = v41_store(8, &[1], &[0; 64]).as_array().unwrap().clone();
            event[position] = value;
            let mut stream = v41_stream();
            assert_eq!(
                stream.apply(0, &payload(vec![Value::Array(event)]), "m", &v41_config()),
                Err(reason)
            );
            assert!(stream.unsupported);
            assert_eq!(stream.index.len(), 0);
        }
    }
    #[test]
    fn gap_clears_old_residency_and_orphans_do_not_become_roots() {
        let cfg = config();
        let mut state = StreamIndex::default();
        state
            .apply(0, &payload(vec![stored(1, Value::Nil, &[1, 2])]), "m", &cfg)
            .unwrap();
        assert_eq!(state.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 2);
        assert_eq!(
            state.apply(2, &payload(vec![stored(2, 1.into(), &[3, 4])]), "m", &cfg),
            Err("missing_parent")
        );
        assert_eq!(state.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 0);
        assert_eq!(state.index.matched_tokens(&[3, 4, 5], 2, "m", ""), 0);
        // A new root can be indexed, but it does not prove that the skipped
        // orphan was recovered. Keep coverage partial, without failing good batches.
        assert_eq!(
            state.apply(3, &payload(vec![stored(3, Value::Nil, &[5, 6])]), "m", &cfg),
            Ok(())
        );
        assert_eq!(
            state.apply(3, &payload(vec![stored(3, Value::Nil, &[5, 6])]), "m", &cfg),
            Ok(())
        );
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

    // Production traffic can bypass the gateway. Unsupported request identity
    // must never poison other roots or make a plain-text false positive.
    pub fn with_extras(event: Value, extras: Value) -> Value {
        let mut fields = event.as_array().unwrap().clone();
        fields.resize(fields.len().max(9), Value::Nil);
        fields[8] = extras;
        Value::Array(fields)
    }

    #[test]
    fn unsupported_identity_isolated_without_losing_removals_or_new_roots() {
        for extra in [
            Value::Array(vec![Value::Array(vec!["image-hash".into(), 0.into()])]),
            Value::Array(vec![Value::Binary(vec![7; 32])]),
            Value::Array(vec!["salt".into(), "other".into()]),
            123.into(),
        ] {
            let mut stream = StreamIndex::default();
            let bad = with_extras(stored(2, Value::Nil, &[3, 4]), Value::Array(vec![extra]));
            stream
                .apply(
                    0,
                    &payload(vec![stored(1, Value::Nil, &[1, 2]), bad]),
                    "m",
                    &config(),
                )
                .unwrap();
            assert!(!stream.unsupported);
            assert_eq!(stream.index.matched_tokens(&[1, 2, 9], 2, "m", ""), 2);
            assert_eq!(stream.index.matched_tokens(&[3, 4, 9], 2, "m", ""), 0);
            let remove = Value::Array(vec!["BlockRemoved".into(), Value::Array(vec![1.into()])]);
            stream
                .apply(
                    1,
                    &payload(vec![
                        stored(3, 2.into(), &[5, 6]),
                        remove,
                        stored(4, Value::Nil, &[7, 8]),
                    ]),
                    "m",
                    &config(),
                )
                .unwrap();
            assert_eq!(stream.index.matched_tokens(&[1, 2, 9], 2, "m", ""), 0);
            assert_eq!(stream.index.matched_tokens(&[3, 4, 5, 6, 9], 2, "m", ""), 0);
            assert_eq!(stream.index.matched_tokens(&[7, 8, 9], 2, "m", ""), 2);
            assert!(stream.history_complete);
            assert!(!stream.incomplete()); // intentionally skipped parent is not lost history
            assert_eq!(stream.skipped_stores, 2);
        }
    }

    #[test]
    fn dense_store_preserves_only_prefix_before_unknown_identity() {
        let event = grouped_store(0, &[1, 2, 3], &[1, 2, 3, 4, 5, 6], None);
        let event = with_extras(
            event,
            Value::Array(vec![
                Value::Nil,
                Value::Array(vec![Value::Binary(vec![1])]),
                Value::Nil,
            ]),
        );
        let mut stream = StreamIndex::default();
        stream
            .apply(0, &payload(vec![event]), "m", &config())
            .unwrap();
        assert_eq!(
            stream
                .index
                .matched_tokens(&[1, 2, 3, 4, 5, 6, 7], 2, "m", ""),
            2
        );
        assert_eq!(stream.index.len(), 1);
        stream
            .apply(
                1,
                &payload(vec![stored(4, 3.into(), &[7, 8])]),
                "m",
                &config(),
            )
            .unwrap();
        assert_eq!(stream.index.len(), 1);
        // Clear is a genuine reset, including the bounded quarantine dictionary.
        stream
            .apply(
                2,
                &payload(vec![
                    Value::Array(vec!["AllBlocksCleared".into()]),
                    stored(3, Value::Nil, &[5, 6]),
                ]),
                "m",
                &config(),
            )
            .unwrap();
        assert_eq!(stream.index.matched_tokens(&[5, 6, 7], 2, "m", ""), 2);
        assert!(stream.last_skip.is_none());
    }

    #[test]
    fn lora_and_misaligned_extras_skip_store_but_not_same_batch_remove() {
        for position in [5, 7, 8] {
            let mut stream = StreamIndex::default();
            stream
                .apply(
                    0,
                    &payload(vec![stored(1, Value::Nil, &[1, 2])]),
                    "m",
                    &config(),
                )
                .unwrap();
            let mut bad = stored(2, Value::Nil, &[3, 4]).as_array().unwrap().clone();
            bad.resize(9, Value::Nil);
            bad[position] = match position {
                5 => 7.into(),
                7 => "adapter".into(),
                _ => Value::Array(vec![]),
            };
            let remove = Value::Array(vec!["BlockRemoved".into(), Value::Array(vec![1.into()])]);
            stream
                .apply(1, &payload(vec![Value::Array(bad), remove]), "m", &config())
                .unwrap();
            assert_eq!(stream.index.len(), 0);
            assert!(!stream.unsupported);
            assert_eq!(stream.last_seq, Some(1));
        }
    }

    #[test]
    fn sparse_store_cannot_learn_through_quarantined_dense_hash() {
        let sparse = grouped_store(1, &[2], &[1, 2, 3, 4], Some(4));
        let dense = with_extras(
            grouped_store(0, &[1, 2], &[1, 2, 3, 4], None),
            Value::Array(vec![Value::Nil, Value::Array(vec![Value::Binary(vec![8])])]),
        );
        let mut stream = StreamIndex::default();
        stream
            .apply(0, &payload(vec![sparse, dense]), "m", &grouped_config())
            .unwrap();
        assert_eq!(stream.index.len(), 1);
        assert!(!stream.incomplete());
        assert!(!stream.unsupported);
    }

    #[test]
    fn contradictory_identity_clears_old_positive_hits() {
        let mut stream = StreamIndex::default();
        stream
            .apply(
                0,
                &payload(vec![stored(1, Value::Nil, &[1, 2])]),
                "m",
                &config(),
            )
            .unwrap();
        let bad = with_extras(
            stored(1, Value::Nil, &[1, 2]),
            Value::Array(vec![Value::Array(vec![Value::Binary(vec![8])])]),
        );
        assert_eq!(
            stream.apply(1, &payload(vec![bad]), "m", &config()),
            Err("engine_hash_conflict")
        );
        assert_eq!(stream.index.len(), 0);
    }

    #[test]
    fn reversed_opaque_descendants_do_not_trigger_missing_history() {
        // Cross-group batches can put descendants before their rejected root.
        // Propagate rejection before learning, without blocking unrelated roots.
        let mut events: Vec<_> = (2..40)
            .rev()
            .map(|id| stored(id, (id - 1).into(), &[3, 4]))
            .collect();
        events.push(with_extras(
            stored(1, Value::Nil, &[1, 2]),
            Value::Array(vec![Value::Binary(vec![9])]),
        ));
        events.push(stored(50, Value::Nil, &[7, 8]));
        let mut stream = StreamIndex::default();
        stream.apply(0, &payload(events), "m", &config()).unwrap();
        assert!(!stream.incomplete());
        assert_eq!(stream.index.len(), 1);
        assert_eq!(stream.index.matched_tokens(&[7, 8, 9], 2, "m", ""), 2);
        assert_eq!(stream.skipped_stores, 39);
    }

    #[test]
    fn bounded_isolation_never_forgets_evicted_identities() {
        // At capacity, compact old rejections instead of turning them into
        // clean hashes. This may add misses, but must never create a hit.
        let mut cfg = config();
        cfg.max_blocks_per_worker = 2;
        let mut stream = StreamIndex::default();
        let events = (1..10)
            .map(|id| {
                with_extras(
                    stored(id, Value::Nil, &[1, 2]),
                    Value::Array(vec![Value::Binary(vec![9])]),
                )
            })
            .collect();
        stream.apply(0, &payload(events), "m", &cfg).unwrap();
        assert_eq!(stream.index.opaque_len(), 2);
        assert!(stream.index.isolation_overflow());
        stream
            .apply(
                1,
                &payload(vec![
                    stored(1, Value::Nil, &[1, 2]),
                    stored(10, 1.into(), &[3, 4]),
                    stored(50, Value::Nil, &[7, 8]),
                ]),
                "m",
                &cfg,
            )
            .unwrap();
        assert_eq!(stream.index.len(), 1);
        assert_eq!(stream.index.matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""), 0);
        assert!(!stream.incomplete());
    }

    fn grouped_store(group: u64, hashes: &[u64], tokens: &[u32], window: Option<u64>) -> Value {
        let mut event = stored(0, Value::Nil, tokens).as_array().unwrap().clone();
        event.resize(13, Value::Nil);
        event[1] = Value::Array(hashes.iter().map(|n| (*n).into()).collect());
        event[4] = 2.into();
        event[6] = "GPU".into();
        event[9] = group.into();
        event[10] = if window.is_some() {
            "SlidingWindowSpec"
        } else {
            "FullAttentionSpec"
        }
        .into();
        event[11] = window.map(Value::from).unwrap_or(Value::Nil);
        event[12] = "LOCAL".into();
        Value::Array(event)
    }

    fn grouped_config() -> Config {
        let mut cfg = config();
        cfg.profiles.get_mut("m").unwrap().cache_groups = vec![
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
        ];
        cfg
    }

    #[test]
    fn sparse_before_dense_resolves_content_without_reordering_remove() {
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        let tokens = [1, 2, 3, 4, 5, 6];
        let sparse = grouped_store(1, &[2, 3], &tokens, Some(4));
        let remove = Value::Array(vec![
            "BlockRemoved".into(),
            Value::Array(vec![3.into()]),
            "GPU".into(),
            1.into(),
            "LOCAL".into(),
        ]);
        let dense = grouped_store(0, &[1, 2, 3], &tokens, None);
        stream
            .apply(0, &payload(vec![sparse, remove, dense]), "m", &cfg)
            .unwrap();
        let keys = super::super::index::prefix_keys(&[1, 2, 3, 4, 5, 6, 7], 2, "m", "");
        assert_eq!(
            stream
                .index
                .matched_keys_layout(&keys, 2, &cfg.profiles["m"].cache_groups),
            Some(0)
        );
        assert_eq!(stream.index.matched_keys(&keys, 2), 6);
        assert_eq!(stream.observed_groups.len(), 2);
        stream
            .apply(
                1,
                &payload(vec![grouped_store(1, &[2, 3], &tokens, Some(4))]),
                "m",
                &cfg,
            )
            .unwrap();
        assert_eq!(
            stream
                .index
                .matched_keys_layout(&keys, 2, &cfg.profiles["m"].cache_groups),
            Some(6)
        );
    }

    #[test]
    fn sparse_unknown_is_visible_without_discarding_valid_dense_group() {
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        stream
            .apply(
                0,
                &payload(vec![grouped_store(0, &[1], &[1, 2], None)]),
                "m",
                &cfg,
            )
            .unwrap();
        assert_eq!(
            stream.apply(
                1,
                &payload(vec![grouped_store(1, &[3], &[1, 2, 3, 4, 5, 6], Some(4))]),
                "m",
                &cfg
            ),
            Err("sparse_hash_unknown")
        );
        assert_eq!(stream.last_seq, Some(1));
        assert_eq!(stream.last_error, Some("sparse_hash_unknown"));
        assert_eq!(stream.index.len(), 1);
        assert!(!stream.unsupported);
    }

    #[test]
    fn declared_group_metadata_must_match_and_rejected_metadata_remains_visible() {
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        assert_eq!(
            stream.apply(
                0,
                &payload(vec![grouped_store(1, &[1], &[1, 2], Some(8))]),
                "m",
                &cfg
            ),
            Err("cache_group_mismatch")
        );
        assert!(stream.unsupported);
        assert_eq!(stream.observed_groups[&1].sliding_window, Some(8));
        assert_eq!(
            stream.apply(
                1,
                &payload(vec![grouped_store(0, &[1], &[1, 2], None)]),
                "m",
                &cfg
            ),
            Err("unsupported_stream")
        );
        assert_eq!(stream.last_error, Some("cache_group_mismatch"));
        assert_eq!(stream.observed_groups.len(), 1);
        assert_eq!(stream.index.len(), 0);
    }

    #[test]
    fn sequence_gap_discards_group_ancestry_and_residency() {
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        stream
            .apply(
                0,
                &payload(vec![
                    grouped_store(0, &[1, 2], &[1, 2, 3, 4], None),
                    grouped_store(1, &[1, 2], &[1, 2, 3, 4], Some(4)),
                ]),
                "m",
                &cfg,
            )
            .unwrap();
        let keys = super::super::index::prefix_keys(&[1, 2, 3, 4, 5], 2, "m", "");
        assert_eq!(
            stream
                .index
                .matched_keys_layout(&keys, 2, &cfg.profiles["m"].cache_groups),
            Some(4)
        );
        assert_eq!(
            stream.apply(
                2,
                &payload(vec![grouped_store(1, &[2], &[1, 2, 3, 4], Some(4))]),
                "m",
                &cfg
            ),
            Err("sparse_hash_unknown")
        );
        assert_eq!(
            stream
                .index
                .matched_keys_layout(&keys, 2, &cfg.profiles["m"].cache_groups),
            None
        );
    }

    #[test]
    fn clear_inside_batch_cannot_reuse_preclear_ancestry() {
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        let clear = Value::Array(vec!["AllBlocksCleared".into()]);
        let old = grouped_store(0, &[1, 2], &[1, 2, 3, 4], None);
        let sparse = grouped_store(1, &[2], &[1, 2, 3, 4], Some(4));
        assert_eq!(
            stream.apply(0, &payload(vec![old, clear, sparse]), "m", &cfg),
            Err("sparse_hash_unknown")
        );
        assert_eq!(stream.index.len(), 0);
        assert_eq!(stream.last_error, Some("sparse_hash_unknown"));
        stream.apply(1, &payload(vec![]), "m", &cfg).unwrap();
        assert_eq!(stream.last_error, None);
        assert!(stream.incomplete());
        stream.apply(1, &payload(vec![]), "m", &cfg).unwrap();
        assert_eq!(stream.last_error, None);
        assert!(stream.incomplete());
    }

    #[test]
    fn long_running_churn_keeps_hot_residency_and_bounded_ancestry() {
        // More distinct historical blocks than the capacity must not reset a
        // still-resident prefix. Backend removes remain authoritative.
        let mut cfg = config();
        cfg.max_blocks_per_worker = 2;
        let mut stream = StreamIndex::default();
        stream
            .apply(0, &payload(vec![stored(1, Value::Nil, &[1, 2])]), "m", &cfg)
            .unwrap();
        for n in 2..=10_000u64 {
            stream
                .apply(
                    2 * n - 3,
                    &payload(vec![stored(n, Value::Nil, &[n as u32, 0])]),
                    "m",
                    &cfg,
                )
                .unwrap();
            assert_eq!(stream.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 2);
            let remove = Value::Array(vec!["BlockRemoved".into(), Value::Array(vec![n.into()])]);
            stream
                .apply(2 * n - 2, &payload(vec![remove]), "m", &cfg)
                .unwrap();
            assert_eq!(
                stream.index.matched_tokens(&[n as u32, 0, 3], 2, "m", ""),
                0
            );
            assert!(stream.index.known_len() <= 2 * cfg.max_blocks_per_worker);
            assert_eq!(stream.index.len(), 1);
            assert!(stream.last_event.is_some());
            assert!(stream.history_complete);
        }
        assert!(stream.index.ancestry_evictions > 9000);
        assert_eq!(stream.index.residency_evictions, 0);
    }

    #[test]
    fn historical_orphan_does_not_fail_later_valid_batches() {
        // Partial coverage is persistent evidence, not a new processing error
        // for each subsequent independent root or harmless duplicate.
        let cfg = config();
        let mut stream = StreamIndex::default();
        assert_eq!(
            stream.apply(5, &payload(vec![stored(2, 1.into(), &[3, 4])]), "m", &cfg),
            Err("missing_parent")
        );
        for n in 6..106 {
            stream
                .apply(
                    n,
                    &payload(vec![stored(n, Value::Nil, &[n as u32, 0])]),
                    "m",
                    &cfg,
                )
                .unwrap();
            assert_eq!(stream.last_error, None);
            assert!(stream.incomplete());
            assert!(!stream.history_complete);
            assert_eq!(
                stream.index.matched_tokens(&[n as u32, 0, 1], 2, "m", ""),
                2
            );
        }
        assert_eq!(stream.index.len(), 100);
        let at = stream.last_event;
        stream.apply(105, &payload(vec![]), "m", &cfg).unwrap();
        assert_eq!(stream.last_event, at);
        stream
            .apply(
                106,
                &payload(vec![Value::Array(vec!["AllBlocksCleared".into()])]),
                "m",
                &cfg,
            )
            .unwrap();
        assert!(!stream.incomplete());
        assert!(stream.history_complete);
    }

    #[test]
    fn incompatible_epoch_recovers_only_after_reset_restart_or_clear() {
        // A good Store alone cannot excuse an unknown deletion contract. New
        // epochs and authoritative Clears rebuild from empty, revalidated state.
        let cfg = config();
        for boundary in ["disconnect", "restart", "clear"] {
            let mut stream = StreamIndex::default();
            stream
                .apply(
                    10,
                    &payload(vec![stored(1, Value::Nil, &[1, 2])]),
                    "m",
                    &cfg,
                )
                .unwrap();
            assert_eq!(
                stream.apply(
                    11,
                    &payload(vec![Value::Array(vec!["FutureRemoval".into()])]),
                    "m",
                    &cfg
                ),
                Err("unsupported_event")
            );
            assert_eq!(
                stream.apply(
                    12,
                    &payload(vec![stored(2, Value::Nil, &[3, 4])]),
                    "m",
                    &cfg
                ),
                Err("unsupported_stream")
            );
            stream.invalidate();
            assert!(stream.unsupported);
            match boundary {
                "disconnect" => stream.reset_epoch(),
                "clear" => stream
                    .apply(
                        13,
                        &payload(vec![Value::Array(vec!["AllBlocksCleared".into()])]),
                        "m",
                        &cfg,
                    )
                    .unwrap(),
                _ => {}
            }
            stream
                .apply(
                    if boundary == "restart" { 0 } else { 14 },
                    &payload(vec![stored(3, Value::Nil, &[5, 6])]),
                    "m",
                    &cfg,
                )
                .unwrap();
            assert!(!stream.unsupported);
            assert_eq!(stream.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 0);
            assert_eq!(stream.index.matched_tokens(&[5, 6, 7], 2, "m", ""), 2);
        }
    }

    #[test]
    fn optional_group_fields_use_only_current_epoch_evidence() {
        let cfg = v41_config();
        let mut stream = v41_stream();
        let mut omitted = v41_store(8, &[2], &[2; 64]);
        // MessagePack arrays permit nil fields independently of map omission.
        if let Value::Array(fields) = &mut omitted {
            fields[10] = Value::Nil;
        }
        assert_eq!(
            stream.apply(1, &payload(vec![omitted.clone()]), "m", &cfg),
            Err("cache_group_metadata_missing")
        );
        assert!(!stream.unsupported);
        stream
            .apply(2, &payload(vec![v41_store(8, &[1], &[1; 64])]), "m", &cfg)
            .unwrap();
        stream
            .apply(3, &payload(vec![omitted.clone()]), "m", &cfg)
            .unwrap();
        assert_eq!(stream.index.len(), 2);
        stream.reset_epoch();
        stream
            .apply(
                3,
                &payload(vec![v41_store(8, &[1], &[1; 64]), omitted.clone()]),
                "m",
                &cfg,
            )
            .unwrap();
        assert_eq!(stream.index.len(), 2);
        stream.reset_epoch();
        assert!(stream.observed_groups.is_empty());
        assert_eq!(
            stream.apply(4, &payload(vec![omitted]), "m", &cfg),
            Err("cache_group_metadata_missing")
        );
        assert_eq!(stream.index.len(), 0);

        // A manual SWA contract can supply omitted optional metadata, but an
        // explicit contradictory window never becomes a guessed match.
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        let mut window = grouped_store(1, &[1], &[1, 2], Some(4));
        if let Value::Array(fields) = &mut window {
            fields[10] = Value::Nil;
            fields[11] = Value::Nil;
        }
        stream.apply(0, &payload(vec![window]), "m", &cfg).unwrap();
        assert_eq!(stream.observed_groups[&1].sliding_window, Some(4));
        assert_eq!(
            stream.apply(
                1,
                &payload(vec![grouped_store(1, &[2], &[3, 4], Some(8))]),
                "m",
                &cfg
            ),
            Err("cache_group_mismatch")
        );
    }

    #[test]
    fn incompatible_layout_does_not_seed_content_dictionary() {
        let cfg = grouped_config();
        let mut stream = StreamIndex::default();
        let sparse = grouped_store(1, &[2], &[1, 2, 3, 4], Some(4));
        let invalid = grouped_store(9, &[1, 2], &[1, 2, 3, 4], None);
        assert_eq!(
            stream.apply(0, &payload(vec![sparse, invalid]), "m", &cfg),
            Err("unsupported_group")
        );
        assert_eq!(stream.index.len(), 0);
        assert_eq!(stream.observed_groups.len(), 2);
    }
}
