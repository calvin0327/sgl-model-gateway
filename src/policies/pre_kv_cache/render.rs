//! Derive the exact request token IDs used to query the worker KV index.

use parking_lot::Mutex;
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::OnceCell;

use super::{
    config::{Config, Profile, TokenSource},
    load::Reservation,
};
use crate::tokenizer::{chat_template::ChatTemplateParams, TokenizerRegistry};

#[derive(Debug)]
pub struct Rendered {
    pub tokens: Vec<u32>,
    pub keys: Vec<super::index::Key>,
}

type RenderResult = Result<Arc<Rendered>, &'static str>;

pub struct KvRequest {
    path: &'static str,
    payload: Value,
    inputs: Mutex<HashMap<String, Arc<OnceCell<RenderResult>>>>,
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
            reservations: Mutex::new(Vec::new()),
        }
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
        build_rendered(tokens, profile.block_size, name, &salt)
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
        let tokenizer = tokenizer_registry
            .get(&profile.model)
            .ok_or("local_tokenizer_unavailable")?;

        let tokens = match self.path {
            "/v1/chat/completions/render" => {
                let messages = self
                    .payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .ok_or("local_chat_messages")?;
                let tools = optional_array(&self.payload, "tools")?;
                let documents = optional_array(&self.payload, "documents")?;
                let template_kwargs = optional_object(&self.payload, "chat_template_kwargs")?;
                let continue_final_message =
                    optional_bool(&self.payload, "continue_final_message")?.unwrap_or(false);
                let rendered = tokenizer
                    .apply_chat_template(
                        messages,
                        ChatTemplateParams {
                            add_generation_prompt: !continue_final_message,
                            tools,
                            documents,
                            template_kwargs: template_kwargs.as_ref(),
                            special_tokens: Some(tokenizer.get_special_tokens()),
                        },
                    )
                    .map_err(|_| "local_chat_template")?;
                tokenizer
                    .encode(&rendered, false)
                    .map_err(|_| "local_tokenize")?
                    .token_ids()
                    .to_vec()
            }
            "/v1/completions/render" => local_completion_tokens(&self.payload, &*tokenizer)?,
            _ => return Err("unsupported_endpoint"),
        };
        if tokens.is_empty() || tokens.len() > cfg.max_render_tokens {
            return Err("render_token_limit");
        }
        build_rendered(tokens, profile.block_size, name, &salt)
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

fn build_rendered(tokens: Vec<u32>, block_size: usize, profile: &str, salt: &str) -> RenderResult {
    let keys = super::index::prefix_keys(&tokens, block_size, profile, salt);
    Ok(Arc::new(Rendered { tokens, keys }))
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

fn optional_bool(body: &Value, key: &str) -> Result<Option<bool>, &'static str> {
    match body.get(key).filter(|value| !value.is_null()) {
        Some(value) => value.as_bool().map(Some).ok_or("local_request_shape"),
        None => Ok(None),
    }
}

fn local_completion_tokens(
    body: &Value,
    tokenizer: &dyn crate::tokenizer::traits::Tokenizer,
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
    use crate::tokenizer::traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    #[tokio::test]
    async fn local_chat_applies_template_once_and_never_requires_render() {
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
