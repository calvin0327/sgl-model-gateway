//! Comprehensive integration tests for OpenAI backend functionality

use axum::{
    body::Body,
    extract::Request,
    http::{Method, StatusCode},
};
use serde_json::json;
use smg::{
    config::{gateway::PolicyConfig, ConfigError, GatewayConfig, HistoryBackend, OracleConfig},
    protocols::{
        chat::ChatCompletionRequest, common::StringOrArray, completion::CompletionRequest,
        generate::GenerateRequest,
    },
    routers::{openai::OpenAIRouter, RouterTrait},
};

use crate::common::mock_openai_server::MockOpenAIServer;

/// Helper function to create a minimal chat completion request for testing
fn create_minimal_chat_request() -> ChatCompletionRequest {
    let val = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100
    });
    serde_json::from_value(val).unwrap()
}

/// Helper function to create a minimal completion request for testing
fn create_minimal_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        prompt: StringOrArray::String("Hello".to_string()),
        suffix: None,
        max_tokens: Some(100),
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        sampling_seed: None,
        other: serde_json::Map::new(),
    }
}

/// Test basic OpenAI router creation and configuration
#[tokio::test]
async fn test_openai_router_creation() {
    let ctx = crate::common::test_app::create_test_app_context().await;
    // Register an external worker before creating the router
    crate::common::test_app::register_external_worker(&ctx, "https://api.openai.com", None);
    let router = OpenAIRouter::new(&ctx);

    assert!(router.is_ok(), "Router creation should succeed");

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
    assert!(!router.is_pd_mode());
}

/// Test server info endpoint
#[tokio::test]
async fn test_openai_router_server_info() {
    let ctx = crate::common::test_app::create_test_app_context().await;
    crate::common::test_app::register_external_worker(&ctx, "https://api.openai.com", None);
    let router = OpenAIRouter::new(&ctx).unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/info")
        .body(Body::empty())
        .unwrap();

    let response = router.get_server_info(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    assert!(body_str.contains("openai"));
}

/// Test models endpoint
#[tokio::test]
async fn test_openai_router_models() {
    // Use mock server for deterministic models response
    let mock_server = MockOpenAIServer::new().await;
    let ctx = crate::common::test_app::create_test_app_context().await;
    crate::common::test_app::register_external_worker(&ctx, &mock_server.base_url(), None);
    let router = OpenAIRouter::new(&ctx).unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let models: serde_json::Value = serde_json::from_str(&body_str).unwrap();

    assert_eq!(models["object"], "list");
    assert!(models["data"].is_array());
}

#[tokio::test]
async fn test_router_factory_openai_mode() {
    let mut router_config = GatewayConfig::new(PolicyConfig::Random);
    router_config.workers.openai_urls = vec!["https://api.openai.com".to_string()];

    let app_context = crate::common::create_test_context(router_config).await;

    let router = smg::routers::RouterFactory::create_openai_router(&app_context);
    assert!(
        router.is_ok(),
        "Router factory should create OpenAI router successfully"
    );

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
}

/// Test that unsupported endpoints return proper error codes
#[tokio::test]
async fn test_unsupported_endpoints() {
    let ctx = crate::common::test_app::create_test_app_context().await;
    crate::common::test_app::register_external_worker(&ctx, "https://api.openai.com", None);
    let router = OpenAIRouter::new(&ctx).unwrap();

    let generate_request = GenerateRequest {
        text: Some("Hello world".to_string()),
        model: None,
        input_ids: None,
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: None,
        return_logprob: Some(false),
        logprob_start_len: None,
        top_logprobs_num: None,
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream: false,
        log_metrics: true,
        return_hidden_states: false,
        modalities: None,
        session_params: None,
        lora_path: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    };

    let response = router.route_generate(None, &generate_request, None).await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

    let completion_request = create_minimal_completion_request();
    let response = router
        .route_completion(None, &completion_request, None)
        .await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_openai_router_circuit_breaker() {
    let ctx = crate::common::test_app::create_test_app_context().await;
    crate::common::test_app::register_external_worker(
        &ctx,
        "http://invalid-url-that-will-fail",
        None,
    );
    let router = OpenAIRouter::new(&ctx).unwrap();

    let chat_request = create_minimal_chat_request();

    // First few requests should fail and record failures
    for _ in 0..3 {
        let response = router.route_chat(None, &chat_request, None).await;
        // Should get either an error or circuit breaker response
        assert!(
            response.status() == StatusCode::INTERNAL_SERVER_ERROR
                || response.status() == StatusCode::SERVICE_UNAVAILABLE
        );
    }
}

/// Test that /v1/models returns models from registered workers' ModelCards
///
/// With the new worker-based design, models are returned from the WorkerRegistry
/// and don't require calling external APIs. Auth headers are used for routing
/// requests to workers, not for the models endpoint.
#[tokio::test]
async fn test_openai_router_models_from_registry() {
    let ctx = crate::common::test_app::create_test_app_context().await;
    // Register a worker with the default model
    crate::common::test_app::register_external_worker(&ctx, "https://api.example.com", None);
    let router = OpenAIRouter::new(&ctx).unwrap();

    // Get models - should return the registered model
    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::OK);
    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let models: serde_json::Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(models["object"], "list");

    // Should have the default model (gpt-3.5-turbo)
    let data = models["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "gpt-3.5-turbo");
}

#[test]
fn oracle_config_validation_requires_config_when_enabled() {
    let config = crate::common::TestGatewayConfigBuilder::new()
        .openai_mode(vec!["https://api.openai.com".to_string()])
        .history_backend(HistoryBackend::Oracle)
        .build_unchecked();

    let err = config
        .validate()
        .expect_err("config should fail without oracle details");

    match err {
        ConfigError::MissingRequired { field } => {
            assert_eq!(field, "storage.oracle");
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn oracle_config_validation_accepts_dsn_only() {
    let config = crate::common::TestGatewayConfigBuilder::new()
        .openai_mode(vec!["https://api.openai.com".to_string()])
        .oracle_history(OracleConfig {
            wallet_path: None,
            connect_descriptor: "tcps://db.example.com:1522/service".to_string(),
            username: "scott".to_string(),
            password: "tiger".to_string(),
            pool_min: 1,
            pool_max: 4,
            pool_timeout_secs: 30,
        })
        .build_unchecked();

    config.validate().expect("dsn-based config should validate");
}

#[test]
fn oracle_config_validation_accepts_wallet_alias() {
    let config = crate::common::TestGatewayConfigBuilder::new()
        .openai_mode(vec!["https://api.openai.com".to_string()])
        .oracle_history(OracleConfig {
            wallet_path: Some("/etc/sglang/oracle-wallet".to_string()),
            connect_descriptor: "db_low".to_string(),
            username: "app_user".to_string(),
            password: "secret".to_string(),
            pool_min: 1,
            pool_max: 8,
            pool_timeout_secs: 45,
        })
        .build_unchecked();

    config
        .validate()
        .expect("wallet-based config should validate");
}
