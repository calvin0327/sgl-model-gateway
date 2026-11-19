//! Derive the exact request token IDs used to query the worker KV index.

use parking_lot::Mutex;
use serde_json::Value;
use std::{
    borrow::Cow,
    collections::HashMap,
    path::Path,
    sync::{Arc, LazyLock, Weak},
    time::{Duration, Instant},
};
use tokio::sync::OnceCell;

use super::{
    config::{Config, Profile, TokenSource},
    load::Reservation,
};
use crate::tokenizer::{
    chat_template::ChatTemplateParams, registry::TokenizerEntry, traits::Tokenizer,
    TokenizerRegistry,
};

#[derive(Debug)]
pub struct Rendered {
    pub tokens: Vec<u32>,
    pub keys: Vec<super::index::Key>,
    pub requires_group_layout: bool,
}

type RenderResult = Result<Arc<Rendered>, &'static str>;

// Model metadata belongs to the loaded tokenizer, not the arbitrary KV profile
// name. Keep only weak references so unloading/reloading cannot reuse old mode
// metadata, and bound the cache when many tokenizer registrations come and go.
struct CachedTokenizerMode {
    tokenizer: Weak<dyn Tokenizer>,
    deepseek_v41: Result<bool, &'static str>,
}

static TOKENIZER_MODES: LazyLock<Mutex<HashMap<String, CachedTokenizerMode>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(super) fn is_deepseek_v41(entry: &TokenizerEntry) -> Result<bool, &'static str> {
    let mut cache = TOKENIZER_MODES.lock();
    if let Some(cached) = cache.get(&entry.id) {
        if cached
            .tokenizer
            .upgrade()
            .is_some_and(|tokenizer| Arc::ptr_eq(&tokenizer, &entry.tokenizer))
        {
            return cached.deepseek_v41;
        }
    }
    let source = Path::new(&entry.source);
    let directory = if source.is_file() {
        source.parent().unwrap_or(source)
    } else {
        source
    };
    let model_type = std::fs::read(directory.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .map(|config| {
            config.get("model_type").and_then(Value::as_str) == Some("deepseek_v41")
                || config
                    .get("architectures")
                    .and_then(Value::as_array)
                    .is_some_and(|architectures| {
                        architectures
                            .iter()
                            .any(|value| value.as_str() == Some("DeepseekV41ForCausalLM"))
                    })
        });
    // A native renderer loaded from a downloaded HF snapshot may have a remote
    // registry source, or its config may become unreadable after loading. Its
    // capabilities reveal that generic kwargs are unsafe, but do not establish
    // model identity. Report missing metadata instead of guessing V4.1.
    let capabilities = entry.tokenizer.renderer_capabilities();
    let native_v41_like = capabilities.enable_thinking_alias
        && capabilities.native_assistant_continuation
        && capabilities.raw_tool_call_arguments;
    let deepseek_v41 = match model_type {
        Some(true) => Ok(true),
        _ if native_v41_like => Err("local_tokenizer_metadata"),
        _ => Ok(false),
    };
    if cache.len() >= 256 {
        cache.retain(|_, cached| cached.tokenizer.strong_count() > 0);
        if cache.len() >= 256 {
            cache.clear();
        }
    }
    cache.insert(
        entry.id.clone(),
        CachedTokenizerMode {
            tokenizer: Arc::downgrade(&entry.tokenizer),
            deepseek_v41,
        },
    );
    deepseek_v41
}

pub struct KvRequest {
    path: &'static str,
    payload: Value,
    inputs: Mutex<HashMap<String, Arc<OnceCell<RenderResult>>>>,
    // Keep the selected worker's cached prefix for the scheduler log without
    // re-tokenizing the request or adding another token-count field.
    selected_cached_tokens: Mutex<Option<usize>>,
    pub(super) reservations: Mutex<Vec<Reservation>>,
}

impl std::fmt::Debug for KvRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvRequest")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl KvRequest {
    pub fn new(path: &'static str, mut payload: Value) -> Self {
        crate::routers::http::router::strip_default_sglang_fields(&mut payload);
        Self {
            path,
            payload,
            inputs: Mutex::new(HashMap::new()),
            selected_cached_tokens: Mutex::new(None),
            reservations: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn clear_selected_cached_tokens(&self) {
        *self.selected_cached_tokens.lock() = None;
    }

    pub(crate) fn set_selected_cached_tokens(&self, cached_tokens: Option<usize>) {
        *self.selected_cached_tokens.lock() = cached_tokens;
    }

    pub(crate) fn selected_cached_tokens(&self) -> Option<usize> {
        *self.selected_cached_tokens.lock()
    }

    // Logging may reuse tokens already computed for length routing, but must
    // never initialize tokenization or render solely to observe a single worker.
    pub(super) fn cached_input(&self, profile: &str) -> Option<Arc<Rendered>> {
        self.inputs
            .lock()
            .get(profile)?
            .get()?
            .as_ref()
            .ok()
            .cloned()
    }

    /// A successful dispatch remains credited until a newer backend poll.
    /// Errors/cancellation drop the guard and undo only this attempt's credit.
    pub fn finish_attempt(&self, success: bool) {
        // Drop guards outside the request lock, avoiding an inverted load lock.
        let reservations = std::mem::take(&mut *self.reservations.lock());
        for mut reservation in reservations {
            if success {
                reservation.commit();
            }
        }
    }

    /// Compute one exact token sequence per request/profile and reuse it on retry.
    pub(super) async fn tokens(
        &self,
        name: &str,
        profile: &Profile,
        cfg: &Config,
        client: &reqwest::Client,
        tokenizer_registry: &TokenizerRegistry,
        worker_api_key: Option<&str>,
    ) -> RenderResult {
        let cell = self
            .inputs
            .lock()
            .entry(name.to_string())
            .or_default()
            .clone();
        cell.get_or_init(|| async {
            let started = Instant::now();
            let result = match profile.token_source {
                TokenSource::Render => {
                    let result = tokio::time::timeout(
                        Duration::from_millis(cfg.render_timeout_ms),
                        self.fetch_render(name, profile, cfg, client, worker_api_key),
                    )
                    .await
                    .unwrap_or(Err("render_timeout"));
                    metrics::histogram!("pre_kv_cache_render_seconds")
                        .record(started.elapsed().as_secs_f64());
                    result
                }
                TokenSource::LocalTokenizer => {
                    let result = self.local_tokens(name, profile, cfg, tokenizer_registry);
                    metrics::histogram!("pre_kv_cache_local_tokenize_seconds")
                        .record(started.elapsed().as_secs_f64());
                    result
                }
            };
            if let Err(reason) = &result {
                metrics::counter!("pre_kv_cache_fallback_total", "reason" => *reason).increment(1);
            }
            result
        })
        .await
        .clone()
    }

    /// Ask vLLM to apply its own tokenizer and chat template.
    async fn fetch_render(
        &self,
        name: &str,
        profile: &Profile,
        cfg: &Config,
        client: &reqwest::Client,
        worker_api_key: Option<&str>,
    ) -> RenderResult {
        validate_text_request(&self.payload)?;
        validate_model(&self.payload, profile)?;
        let salt = cache_salt(&self.payload)?;
        let mut payload = self.payload.clone();
        payload["model"] = profile.model.clone().into();
        let mut request = client
            .post(format!(
                "{}{}",
                profile.render_url.trim_end_matches('/'),
                self.path
            ))
            .json(&payload);
        if let Some(name) = &profile.render_api_key_env {
            let key = std::env::var(name).map_err(|_| "render_credentials")?;
            if key.is_empty() {
                return Err("render_credentials");
            }
            request = request.bearer_auth(key);
        } else if let Some(key) = worker_api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|_| "render_transport")?;
        if !response.status().is_success() {
            return Err("render_status");
        }
        let bytes = limited_body(response, cfg.max_response_bytes).await?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| "render_json")?;
        let prompt = if self.path == "/v1/completions/render" {
            let prompts = value.as_array().ok_or("render_shape")?;
            if prompts.len() != 1 {
                return Err("unsupported_batch");
            }
            &prompts[0]
        } else {
            &value
        };
        if prompt.get("features").is_some_and(|features| {
            ["mm_hashes", "mm_placeholders"].iter().any(|key| {
                features.get(key).is_some_and(|value| {
                    !value.is_null() && value.as_object().is_none_or(|map| !map.is_empty())
                })
            })
        }) {
            return Err("unsupported_multimodal");
        }
        let tokens = parse_token_ids(prompt.get("token_ids"), cfg.max_render_tokens)?;
        build_rendered(tokens, profile.block_size, name, &salt, false)
    }

    /// Reproduce vLLM text rendering inside the Gateway without an HTTP round trip.
    fn local_tokens(
        &self,
        name: &str,
        profile: &Profile,
        cfg: &Config,
        tokenizer_registry: &TokenizerRegistry,
    ) -> RenderResult {
        validate_text_request(&self.payload)?;
        validate_model(&self.payload, profile)?;
        let salt = cache_salt(&self.payload)?;
        // Startup registration uses --tokenizer-path as its initial name, while
        // worker discovery later registers the same tokenizer by served model.
        // A single-model gateway can safely use the sole loaded tokenizer during
        // that interval; a multi-model gateway must wait for an exact model name.
        let entry = tokenizer_registry
            .get_by_name(&profile.model)
            .or_else(|| tokenizer_registry.get_by_id(&profile.model))
            .or_else(|| {
                let mut entries = tokenizer_registry.list();
                (entries.len() == 1).then(|| entries.remove(0))
            });
        let entry = entry.ok_or_else(|| {
            tracing::debug!(
                profile = name,
                model = profile.model,
                registered_tokenizers = tokenizer_registry.len(),
                "pre_kv_cache local tokenizer unavailable"
            );
            "local_tokenizer_unavailable"
        })?;
        let tokenizer = &entry.tokenizer;
        let deepseek_v41 = is_deepseek_v41(&entry)?;

        let tokens = match self.path {
            "/v1/chat/completions/render" => {
                let messages = self
                    .payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .ok_or("local_chat_messages")?;
                let messages = local_messages(messages, deepseek_v41);
                let tools = optional_array(&self.payload, "tools")?;
                let documents = optional_array(&self.payload, "documents")?;
                let template_kwargs = if deepseek_v41 {
                    Some(v41_template_kwargs(&self.payload)?)
                } else {
                    local_template_kwargs(&self.payload)?
                };
                let continue_final_message =
                    optional_bool(&self.payload, "continue_final_message")?.unwrap_or(false);
                // vLLM's V4.1 native encoder ignores continuation flags. The
                // library's false flag would instead add wo_eos to the last
                // assistant message and derive a different prefix.
                let add_generation_prompt = deepseek_v41 || !continue_final_message;
                let rendered = tokenizer
                    .apply_chat_template(
                        &messages,
                        ChatTemplateParams {
                            add_generation_prompt,
                            tools,
                            documents,
                            template_kwargs: template_kwargs.as_ref(),
                            special_tokens: Some(tokenizer.get_special_tokens()),
                            ..Default::default()
                        },
                    )
                    .map_err(|error| {
                        tracing::debug!(
                            profile = name,
                            error = %error,
                            "pre_kv_cache local chat template failed"
                        );
                        "local_chat_template"
                    })?;
                tokenizer
                    .encode(&rendered, false)
                    .map_err(|error| {
                        tracing::debug!(
                            profile = name,
                            error = %error,
                            "pre_kv_cache local tokenization failed"
                        );
                        "local_tokenize"
                    })?
                    .token_ids()
                    .to_vec()
            }
            "/v1/completions/render" => local_completion_tokens(&self.payload, &**tokenizer)?,
            _ => return Err("unsupported_endpoint"),
        };
        if tokens.is_empty() || tokens.len() > cfg.max_render_tokens {
            return Err("render_token_limit");
        }
        build_rendered(tokens, profile.block_size, name, &salt, deepseek_v41)
    }
}

fn validate_model(body: &Value, profile: &Profile) -> Result<(), &'static str> {
    if body
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|model| !model.is_empty() && model != profile.model)
    {
        return Err("render_model_mismatch");
    }
    Ok(())
}

fn cache_salt(body: &Value) -> Result<String, &'static str> {
    body.get("cache_salt")
        .filter(|value| !value.is_null())
        .map(|value| value.as_str().ok_or("cache_salt_type"))
        .transpose()
        .map(|salt| salt.unwrap_or("").to_string())
}

fn parse_token_ids(value: Option<&Value>, limit: usize) -> Result<Vec<u32>, &'static str> {
    let tokens = value.and_then(Value::as_array).ok_or("render_tokens")?;
    if tokens.is_empty() || tokens.len() > limit {
        return Err("render_token_limit");
    }
    tokens
        .iter()
        .map(|token| {
            token
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("render_token_range")
        })
        .collect()
}

fn build_rendered(
    tokens: Vec<u32>,
    block_size: usize,
    profile: &str,
    salt: &str,
    requires_group_layout: bool,
) -> RenderResult {
    let keys = super::index::prefix_keys(&tokens, block_size, profile, salt);
    Ok(Arc::new(Rendered {
        tokens,
        keys,
        requires_group_layout,
    }))
}

fn optional_array<'a>(body: &'a Value, key: &str) -> Result<Option<&'a [Value]>, &'static str> {
    match body.get(key).filter(|value| !value.is_null()) {
        Some(value) => value
            .as_array()
            .map(|items| Some(items.as_slice()))
            .ok_or("local_request_shape"),
        None => Ok(None),
    }
}

fn optional_object(
    body: &Value,
    key: &str,
) -> Result<Option<HashMap<String, Value>>, &'static str> {
    match body.get(key).filter(|value| !value.is_null()) {
        Some(value) => value
            .as_object()
            .map(|object| {
                Some(
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                )
            })
            .ok_or("local_request_shape"),
        None => Ok(None),
    }
}

fn local_template_kwargs(body: &Value) -> Result<Option<HashMap<String, Value>>, &'static str> {
    let mut kwargs = optional_object(body, "chat_template_kwargs")?.unwrap_or_default();

    // vLLM forwards the OpenAI top-level reasoning effort into the model's
    // native chat encoder. Keep that value in the one local rendering pass.
    if !kwargs.contains_key("reasoning_effort") {
        if let Some(effort) = body
            .get("reasoning_effort")
            .filter(|value| !value.is_null())
        {
            kwargs.insert("reasoning_effort".into(), effort.clone());
        }
    }

    Ok((!kwargs.is_empty()).then_some(kwargs))
}

fn local_messages(messages: &[Value], deepseek_v41: bool) -> Cow<'_, [Value]> {
    // vLLM's V4.1 wrapper copies a present reasoning alias over reasoning_content,
    // including null. Preserve that precedence without changing the forwarded
    // request, other models, or allocating for the usual no-alias request.
    if !deepseek_v41
        || !messages
            .iter()
            .any(|message| message.get("reasoning").is_some())
    {
        return Cow::Borrowed(messages);
    }
    let mut normalized = messages.to_vec();
    for message in &mut normalized {
        if let Some(reasoning) = message.get("reasoning").cloned() {
            message["reasoning_content"] = reasoning;
        }
    }
    Cow::Owned(normalized)
}

fn v41_template_kwargs(body: &Value) -> Result<HashMap<String, Value>, &'static str> {
    let mut kwargs = optional_object(body, "chat_template_kwargs")?.unwrap_or_default();

    // Normalize the HTTP backend's precedence before calling llm-tokenizer.
    // The library targets SMG semantics: its defaults, effort mapping, and
    // contradictory toggle handling differ from vLLM's V4.1 encoder.
    if let Some(effort) = body
        .get("reasoning_effort")
        .filter(|value| !value.is_null())
    {
        kwargs.insert("reasoning_effort".into(), effort.clone());
        kwargs
            .entry("enable_thinking".into())
            .or_insert_with(|| Value::Bool(effort.as_str() != Some("none")));
    }

    let effort = kwargs.get("reasoning_effort");
    let thinking_off = effort.and_then(Value::as_str) == Some("none");
    let budget = match effort {
        None | Some(Value::Null) => 75,
        Some(Value::String(name)) => match name.as_str() {
            "none" | "high" | "xhigh" => 75,
            "low" => 50,
            "max" => 100,
            _ => return Err("local_v41_reasoning_effort"),
        },
        Some(value) => value
            .as_u64()
            .filter(|value| (1..=100).contains(value))
            .ok_or("local_v41_reasoning_effort")?,
    };

    // Actual v0.31.0-cu129-fixed responses establish default/null => on,
    // conflicting booleans => OR, and effort none => off even with true.
    // Reject non-boolean switches rather than silently inventing a prefix.
    let toggle = |key| match kwargs.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        _ => Err("local_v41_thinking"),
    };
    let thinking = match (toggle("thinking")?, toggle("enable_thinking")?) {
        (None, None) => true,
        (thinking, alias) => thinking.unwrap_or(false) || alias.unwrap_or(false),
    } && !thinking_off;
    kwargs.insert("thinking".into(), thinking.into());
    kwargs.insert("enable_thinking".into(), thinking.into());
    kwargs.insert("reasoning_effort".into(), budget.into());
    Ok(kwargs)
}

fn optional_bool(body: &Value, key: &str) -> Result<Option<bool>, &'static str> {
    match body.get(key).filter(|value| !value.is_null()) {
        Some(value) => value.as_bool().map(Some).ok_or("local_request_shape"),
        None => Ok(None),
    }
}

fn local_completion_tokens(
    body: &Value,
    tokenizer: &dyn Tokenizer,
) -> Result<Vec<u32>, &'static str> {
    let prompt = body.get("prompt").ok_or("local_completion_prompt")?;
    if let Some(text) = prompt.as_str() {
        return tokenizer
            .encode(text, false)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|_| "local_tokenize");
    }
    let values = prompt.as_array().ok_or("local_completion_prompt")?;
    if values.iter().all(Value::is_u64) {
        return values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .ok_or("render_token_range")
            })
            .collect();
    }
    Err("unsupported_batch")
}

fn validate_text_request(body: &Value) -> Result<(), &'static str> {
    if [
        "lora_path",
        "lora_request",
        "prompt_embeds",
        "multi_modal_data",
        "mm_processor_kwargs",
        "chat_template",
        "tokenizer_kwargs",
        "truncate_prompt_tokens",
        "add_special_tokens",
    ]
    .iter()
    .any(|key| body.get(key).is_some_and(|value| !value.is_null()))
    {
        return Err("unsupported_request");
    }
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages {
            if let Some(parts) = message.get("content").and_then(Value::as_array) {
                if parts
                    .iter()
                    .any(|part| part.get("type").and_then(Value::as_str) != Some("text"))
                {
                    return Err("unsupported_multimodal");
                }
            }
        }
    }
    if let Some(prompts) = body.get("prompt").and_then(Value::as_array) {
        if prompts.iter().any(|prompt| !prompt.is_u64()) {
            return Err("unsupported_batch");
        }
    }
    Ok(())
}

pub(super) async fn limited_body(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, &'static str> {
    if response
        .content_length()
        .is_some_and(|size| size > limit as u64)
    {
        return Err("response_size");
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| "response_body")? {
        if bytes.len().saturating_add(chunk.len()) > limit {
            return Err("response_size");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{
        huggingface::HuggingFaceTokenizer,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
    };
    use serde_json::json;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    const MIN_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": { "hello": 0, "<s>": 1, "</s>": 2 },
            "merges": []
        }
    }"#;

    #[derive(Default)]
    struct TestTokenizer {
        encode_calls: AtomicUsize,
        template_calls: AtomicUsize,
        generation_prompts: Mutex<Vec<bool>>,
        kwargs: Mutex<Vec<Option<HashMap<String, Value>>>>,
        special_tokens: SpecialTokens,
    }

    impl Encoder for TestTokenizer {
        fn encode(&self, _input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
            assert!(!add_special_tokens);
            self.encode_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Encoding::Plain(vec![1, 2, 3, 4, 5]))
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|input| self.encode(input, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for TestTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok("decoded".into())
        }
    }

    impl Tokenizer for TestTokenizer {
        fn vocab_size(&self) -> usize {
            1024
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special_tokens
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn apply_chat_template(
            &self,
            messages: &[Value],
            params: ChatTemplateParams,
        ) -> anyhow::Result<String> {
            assert_eq!(messages[0]["role"], "user");
            assert_eq!(params.tools.map(<[Value]>::len), Some(1));
            assert_eq!(params.documents.map(<[Value]>::len), Some(1));
            assert!(params.special_tokens.is_some());
            self.template_calls.fetch_add(1, Ordering::Relaxed);
            self.generation_prompts
                .lock()
                .push(params.add_generation_prompt);
            self.kwargs.lock().push(params.template_kwargs.cloned());
            Ok("rendered prompt".into())
        }
    }

    #[test]
    fn unsupported_input_is_not_treated_as_plain_text() {
        assert!(
            validate_text_request(&json!({"messages":[{"content":[{"type":"image_url"}]}]}))
                .is_err()
        );
        assert!(validate_text_request(&json!({"prompt":["a","b"]})).is_err());
        assert!(validate_text_request(&json!({"prompt":[1,2,3]})).is_ok());
        assert!(validate_text_request(&json!({"messages":[],"chat_template":"custom"})).is_err());
        assert!(validate_text_request(
            &json!({"messages":[{"content":"hello"}],"tools":[],"cache_salt":"a"})
        )
        .is_ok());
    }

    #[test]
    fn token_source_defaults_to_render_and_rejects_unknown_values() {
        let profile: Profile =
            serde_json::from_value(json!({"model":"m","block_size":128})).unwrap();
        assert_eq!(profile.token_source, TokenSource::Render);
        let local: Profile = serde_json::from_value(
            json!({"model":"m","block_size":128,"token_source":"local_tokenizer"}),
        )
        .unwrap();
        assert_eq!(local.token_source, TokenSource::LocalTokenizer);
        assert!(serde_json::from_value::<Profile>(
            json!({"model":"m","block_size":128,"token_source":"local"})
        )
        .is_err());
    }

    #[test]
    fn deepseek_v4_model_uses_its_native_chat_encoder() {
        // A V4 checkpoint has no ordinary Jinja chat template. The local
        // tokenizer must select its native encoder from the model architecture.
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join("DeepSeek-V4-Flash-0731");
        fs::create_dir(&model_dir).unwrap();
        let tokenizer_path = model_dir.join("tokenizer.json");
        fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"architectures":["DeepseekV4ForCausalLM"]}"#,
        )
        .unwrap();

        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
        let messages = vec![json!({"role":"user","content":"Hello"})];
        let prompt = tokenizer
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();

        assert!(prompt.contains("<｜begin▁of▁sentence｜>"));
        assert!(prompt.contains("<｜User｜>Hello<｜Assistant｜>"));
        assert!(prompt.ends_with("</think>"));
    }

    #[test]
    fn deepseek_v41_without_jinja_uses_native_chat_encoding() {
        // Both metadata forms must select V4.1 without a Jinja template.
        // An explicit budget avoids conflating encoder support with the
        // different default effort mappings in tokenizer 1.9 and vLLM 0.31.
        for config in [
            json!({"architectures":["DeepseekV41ForCausalLM"]}),
            json!({"model_type":"deepseek_v41"}),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("tokenizer.json");
            fs::write(&path, MIN_TOKENIZER_JSON).unwrap();
            fs::write(temp.path().join("config.json"), config.to_string()).unwrap();
            fs::write(
                temp.path().join("tokenizer_config.json"),
                r#"{"tokenizer_class":"PreTrainedTokenizerFast"}"#,
            )
            .unwrap();
            let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
            let messages = vec![json!({"role":"user","content":"Hello"})];
            for (body, header, tail) in [
                (
                    json!({"chat_template_kwargs":{"reasoning_effort":75}}),
                    "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
                    "<think>",
                ),
                (
                    json!({"chat_template_kwargs":{"thinking":false}}),
                    "",
                    "</think>",
                ),
                (json!({"reasoning_effort":"none"}), "", "</think>"),
            ] {
                let kwargs = local_template_kwargs(&body).unwrap();
                let prompt = tokenizer
                    .apply_chat_template(
                        &messages,
                        ChatTemplateParams {
                            template_kwargs: kwargs.as_ref(),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    prompt,
                    format!("<｜begin▁of▁sentence｜>{header}<｜User｜>Hello<｜Assistant｜>{tail}")
                );
            }
        }
    }

    #[test]
    fn v41_http_options_match_observed_backend_prompt_modes() {
        // These eight HTTP shapes were captured from the deployed backend.
        // Assert complete rendered prompts, not only token counts: low/high
        // differ near the beginning and therefore change every prefix key.
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        fs::write(&path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            temp.path().join("config.json"),
            r#"{"model_type":"deepseek_v41"}"#,
        )
        .unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        let messages = vec![
            json!({"role":"system","content":"You are a helpful assistant."}),
            json!({"role":"user","content":"Reply only OK."}),
        ];
        for (body, budget) in [
            (json!({}), Some(75)),
            (json!({"reasoning_effort":"low"}), Some(50)),
            (json!({"reasoning_effort":"high"}), Some(75)),
            (json!({"reasoning_effort":"none"}), None),
            (
                json!({"reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"low"}}),
                Some(75),
            ),
            (
                json!({"chat_template_kwargs":{"thinking":false,"enable_thinking":true}}),
                Some(75),
            ),
            (
                json!({"reasoning_effort":"none","chat_template_kwargs":{"thinking":true}}),
                None,
            ),
            (json!({"chat_template_kwargs":{"thinking":null}}), Some(75)),
        ] {
            let kwargs = v41_template_kwargs(&body).unwrap();
            let prompt = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        template_kwargs: Some(&kwargs),
                        ..Default::default()
                    },
                )
                .unwrap();
            let (effort, tail) = match budget {
                Some(budget) => (
                    format!("Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"),
                    "<think>",
                ),
                None => (String::new(), "</think>"),
            };
            assert_eq!(
                prompt,
                format!("<｜begin▁of▁sentence｜><｜System｜>{effort}You are a helpful assistant.<｜User｜>Reply only OK.<｜Assistant｜>{tail}"),
                "request: {body}"
            );
        }
    }

    #[test]
    fn v41_invalid_encoder_options_do_not_produce_guessed_prefixes() {
        for body in [
            json!({"reasoning_effort":"minimal"}),
            json!({"chat_template_kwargs":{"reasoning_effort":"42"}}),
            json!({"chat_template_kwargs":{"reasoning_effort":0}}),
            json!({"chat_template_kwargs":{"reasoning_effort":101}}),
            json!({"chat_template_kwargs":{"thinking":"false"}}),
        ] {
            assert!(v41_template_kwargs(&body).is_err(), "request: {body}");
        }
    }

    #[test]
    fn v41_reasoning_alias_overrides_content_including_null() {
        let messages = vec![
            json!({"role":"assistant","content":"A","reasoning":"alias","reasoning_content":"old"}),
            json!({"role":"assistant","content":"B","reasoning":null,"reasoning_content":"old"}),
            json!({"role":"assistant","content":"C","reasoning_content":"retained"}),
        ];
        let normalized = local_messages(&messages, true);
        assert_eq!(normalized[0]["reasoning_content"], "alias");
        assert_eq!(normalized[1]["reasoning_content"], Value::Null);
        assert_eq!(normalized[2]["reasoning_content"], "retained");
        assert_eq!(messages[0]["reasoning_content"], "old");
        assert_eq!(messages[1]["reasoning_content"], "old");
        assert!(matches!(local_messages(&messages, false), Cow::Borrowed(_)));
        assert_eq!(
            local_messages(&messages, false)[0]["reasoning_content"],
            "old"
        );
        assert!(matches!(
            local_messages(&messages[2..], true),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn v41_reasoning_alias_history_matches_native_content_history() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        fs::write(&path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            temp.path().join("config.json"),
            r#"{"model_type":"deepseek_v41"}"#,
        )
        .unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        let kwargs = v41_template_kwargs(&json!({
            "chat_template_kwargs":{"drop_thinking":false}
        }))
        .unwrap();
        for (reasoning, expected_text) in [(json!("alias"), "alias"), (Value::Null, "")] {
            let history = vec![
                json!({"role":"user","content":"Q"}),
                json!({"role":"assistant","content":"A","reasoning":reasoning,"reasoning_content":"old"}),
                json!({"role":"user","content":"Follow up"}),
            ];
            let expected = vec![
                json!({"role":"user","content":"Q"}),
                json!({"role":"assistant","content":"A","reasoning_content":reasoning}),
                json!({"role":"user","content":"Follow up"}),
            ];
            let render = |messages: &[Value]| {
                tokenizer
                    .apply_chat_template(
                        messages,
                        ChatTemplateParams {
                            template_kwargs: Some(&kwargs),
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let actual = render(&local_messages(&history, true));
            assert_eq!(actual, render(&expected));
            assert!(actual.contains(&format!("<think>{expected_text}</think>A")));
            assert!(!actual.contains("old"));
        }
    }

    #[tokio::test]
    async fn v41_metadata_selects_http_normalization_and_reuses_tokens() {
        // A profile/model alias is not a tokenizer implementation selector.
        // The actual loaded config controls V4.1 normalization, even when the
        // registry source names tokenizer.json rather than its directory.
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        fs::write(&path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            temp.path().join("config.json"),
            r#"{"architectures":["DeepseekV41ForCausalLM"]}"#,
        )
        .unwrap();
        let tokenizer = Arc::new(TestTokenizer::default());
        let registry = TokenizerRegistry::new();
        registry
            .load(
                "v41-normalization-test",
                "unrelated-alias",
                path.to_str().unwrap(),
                || {
                    let tokenizer = tokenizer.clone();
                    async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
                },
            )
            .await
            .unwrap();
        let profile: Profile = serde_json::from_value(json!({
            "model":"unrelated-alias", "block_size":2,
            "token_source":"local_tokenizer", "render_url":"http://127.0.0.1:1"
        }))
        .unwrap();
        let request = KvRequest::new(
            "/v1/chat/completions/render",
            json!({
                "model":"unrelated-alias",
                "messages":[{"role":"user","content":"hello"}],
                "tools":[{"type":"function","function":{"name":"lookup"}}],
                "documents":[{"title":"memo"}],
                "reasoning_effort":"high",
                "chat_template_kwargs":{"reasoning_effort":"low"},
                "continue_final_message":true
            }),
        );
        let client = reqwest::Client::new();
        let cfg = Config::default();
        let first = request
            .tokens("arbitrary", &profile, &cfg, &client, &registry, None)
            .await
            .unwrap();
        let second = request
            .tokens("arbitrary", &profile, &cfg, &client, &registry, None)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(first.requires_group_layout);
        assert_eq!(tokenizer.template_calls.load(Ordering::Relaxed), 1);
        assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
        assert_eq!(*tokenizer.generation_prompts.lock(), vec![true]);
        assert_eq!(
            tokenizer.kwargs.lock()[0].as_ref().unwrap()["reasoning_effort"],
            75
        );

        // The same loaded instance keeps its metadata without re-reading disk.
        let entry = registry.get_by_name("unrelated-alias").unwrap();
        fs::write(temp.path().join("config.json"), "{}").unwrap();
        assert_eq!(is_deepseek_v41(&entry), Ok(true));
        // A replacement tokenizer with the same ID must not inherit V4.1 mode.
        let replacement = TokenizerEntry {
            tokenizer: Arc::new(TestTokenizer::default()),
            ..entry
        };
        assert_eq!(is_deepseek_v41(&replacement), Ok(false));
    }

    #[test]
    fn v41_native_encoder_does_not_guess_without_model_metadata() {
        // The tokenizer can already be loaded from a local HF snapshot while
        // registry.source still names a remote model. Native capabilities then
        // require visible fallback until local model metadata is available.
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        fs::write(&path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            temp.path().join("config.json"),
            r#"{"model_type":"deepseek_v41"}"#,
        )
        .unwrap();
        let tokenizer = Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap());
        let entry = TokenizerEntry {
            id: "missing-v41-metadata".into(),
            name: "arbitrary".into(),
            source: temp
                .path()
                .join("missing-remote-source")
                .to_str()
                .unwrap()
                .into(),
            tokenizer: tokenizer.clone(),
        };
        assert_eq!(is_deepseek_v41(&entry), Err("local_tokenizer_metadata"));
        fs::write(temp.path().join("config.json"), "broken JSON").unwrap();
        let unreadable = TokenizerEntry {
            id: "invalid-v41-metadata".into(),
            source: temp.path().to_str().unwrap().into(),
            ..entry
        };
        assert_eq!(
            is_deepseek_v41(&unreadable),
            Err("local_tokenizer_metadata")
        );

        // A normal Jinja tokenizer remains compatible without architecture
        // metadata; the new guard only applies to native V4.1-like renderers.
        let generic = TokenizerEntry {
            tokenizer: Arc::new(TestTokenizer::default()),
            ..unreadable
        };
        assert_eq!(is_deepseek_v41(&generic), Ok(false));
    }

    #[test]
    fn qwen_jinja_template_still_receives_thinking_kwargs() {
        // Native DeepSeek detection must not replace a model's Jinja template.
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        fs::write(&path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            temp.path().join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"]}"#,
        )
        .unwrap();
        fs::write(
            temp.path().join("tokenizer_config.json"),
            json!({"chat_template": "{{ messages[0].content }}{% if enable_thinking %}<think>{% else %}</think>{% endif %}"}).to_string(),
        )
        .unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        for (thinking, suffix) in [(true, "<think>"), (false, "</think>")] {
            let kwargs = local_template_kwargs(&json!({
                "chat_template_kwargs":{"enable_thinking":thinking}
            }))
            .unwrap();
            let prompt = tokenizer
                .apply_chat_template(
                    &[json!({"role":"user","content":"Hello"})],
                    ChatTemplateParams {
                        template_kwargs: kwargs.as_ref(),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(prompt, format!("Hello{suffix}"));
        }
    }

    #[tokio::test]
    async fn local_chat_applies_template_once_and_never_requires_render() {
        let tokenizer = Arc::new(TestTokenizer::default());
        let registry = TokenizerRegistry::new();
        registry
            .load("id", "/models/test", "/models/test", || {
                let tokenizer = tokenizer.clone();
                async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
            })
            .await
            .unwrap();
        let profile: Profile = serde_json::from_value(json!({
            "model":"m",
            "block_size":2,
            "token_source":"local_tokenizer",
            "render_url":"http://127.0.0.1:1"
        }))
        .unwrap();
        let request = KvRequest::new(
            "/v1/chat/completions/render",
            json!({
                "model":"m",
                "messages":[{"role":"user","content":"hello"}],
                "tools":[{"type":"function","function":{"name":"lookup"}}],
                "documents":[{"title":"memo"}],
                "chat_template_kwargs":{"enable_thinking":false},
                "reasoning_effort":"high",
                "continue_final_message":true
            }),
        );
        let cfg = Config::default();
        let first = request
            .tokens(
                "m",
                &profile,
                &cfg,
                &reqwest::Client::new(),
                &registry,
                None,
            )
            .await
            .unwrap();
        let second = request
            .tokens(
                "m",
                &profile,
                &cfg,
                &reqwest::Client::new(),
                &registry,
                None,
            )
            .await
            .unwrap();

        assert_eq!(first.tokens, vec![1, 2, 3, 4, 5]);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(tokenizer.template_calls.load(Ordering::Relaxed), 1);
        assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
        assert_eq!(*tokenizer.generation_prompts.lock(), vec![false]);
        assert_eq!(
            tokenizer.kwargs.lock()[0].as_ref().unwrap()["enable_thinking"],
            false
        );
        assert_eq!(
            tokenizer.kwargs.lock()[0].as_ref().unwrap()["reasoning_effort"],
            "high"
        );
    }

    #[tokio::test]
    async fn local_tokenizer_never_guesses_between_multiple_models() {
        let registry = TokenizerRegistry::new();
        for (id, name) in [("id-a", "model-a"), ("id-b", "model-b")] {
            registry
                .load(id, name, name, || async {
                    Ok(Arc::new(TestTokenizer::default()) as Arc<dyn Tokenizer>)
                })
                .await
                .unwrap();
        }
        let profile: Profile = serde_json::from_value(json!({
            "model":"missing-model",
            "block_size":2,
            "token_source":"local_tokenizer"
        }))
        .unwrap();
        let request = KvRequest::new(
            "/v1/chat/completions/render",
            json!({
                "model":"missing-model",
                "messages":[{"role":"user","content":"hello"}]
            }),
        );

        let error = request
            .tokens(
                "missing-model",
                &profile,
                &Config::default(),
                &reqwest::Client::new(),
                &registry,
                None,
            )
            .await
            .unwrap_err();

        assert_eq!(error, "local_tokenizer_unavailable");
    }

    #[tokio::test]
    async fn local_completion_supports_text_and_existing_token_ids() {
        let tokenizer = Arc::new(TestTokenizer::default());
        let registry = TokenizerRegistry::new();
        registry
            .load("id", "m", "m", || {
                let tokenizer = tokenizer.clone();
                async move { Ok(tokenizer as Arc<dyn Tokenizer>) }
            })
            .await
            .unwrap();
        let profile: Profile = serde_json::from_value(json!({
            "model":"m","block_size":2,"token_source":"local_tokenizer"
        }))
        .unwrap();
        let cfg = Config::default();
        let text = KvRequest::new(
            "/v1/completions/render",
            json!({"model":"m","prompt":"hello"}),
        );
        assert_eq!(
            text.tokens(
                "m",
                &profile,
                &cfg,
                &reqwest::Client::new(),
                &registry,
                None,
            )
            .await
            .unwrap()
            .tokens,
            vec![1, 2, 3, 4, 5]
        );
        let ids = KvRequest::new(
            "/v1/completions/render",
            json!({"model":"m","prompt":[7,8,9]}),
        );
        assert_eq!(
            ids.tokens(
                "m",
                &profile,
                &cfg,
                &reqwest::Client::new(),
                &registry,
                None,
            )
            .await
            .unwrap()
            .tokens,
            vec![7, 8, 9]
        );
        assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
    }
}
