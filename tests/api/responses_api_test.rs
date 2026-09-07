// Integration test for Responses API

use smg::protocols::{
    common::{Function, GenerationRequest, UsageInfo},
    responses::{
        FunctionTool, ReasoningEffort, ResponseInput, ResponseReasoningParam, ResponseTool,
        ResponsesRequest, ResponsesToolChoice, ServiceTier, ToolChoiceOptions, Truncation,
    },
};

#[test]
fn test_responses_request_creation() {
    let request = ResponsesRequest {
        include: None,
        input: ResponseInput::Text("Hello, world!".to_string()),
        instructions: Some("Be helpful".to_string()),
        max_output_tokens: Some(100),
        max_tool_calls: None,
        metadata: None,
        model: "test-model".to_string(),
        parallel_tool_calls: Some(true),
        previous_response_id: None,
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::Medium),
            summary: None,
        }),
        service_tier: Some(ServiceTier::Auto),
        store: Some(true),
        stream: Some(false),
        temperature: Some(0.7),
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto)),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        })]),
        top_logprobs: Some(5),
        top_p: Some(0.9),
        truncation: Some(Truncation::Disabled),
        text: None,
        user: Some("test-user".to_string()),
        request_id: Some("resp_test123".to_string()),
        priority: 0,
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        stop: None,
        top_k: -1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
        ..Default::default()
    };

    assert!(!request.is_stream());
    assert_eq!(request.get_model(), Some("test-model"));
    let routing_text = request.extract_text_for_routing();
    assert_eq!(routing_text, "Hello, world!");
}

#[test]
fn test_responses_request_sglang_extensions() {
    // Test that SGLang-specific sampling parameters are present and serializable
    let request = ResponsesRequest {
        include: None,
        input: ResponseInput::Text("Test".to_string()),
        instructions: None,
        max_output_tokens: Some(50),
        max_tool_calls: None,
        metadata: None,
        model: "test-model".to_string(),
        parallel_tool_calls: Some(true),
        previous_response_id: None,
        reasoning: None,
        service_tier: Some(ServiceTier::Auto),
        store: Some(true),
        stream: Some(false),
        temperature: Some(0.8),
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto)),
        tools: Some(vec![]),
        top_logprobs: Some(0),
        top_p: Some(0.95),
        truncation: Some(Truncation::Auto),
        text: None,
        user: None,
        request_id: Some("resp_test456".to_string()),
        priority: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.2),
        stop: None,
        // SGLang-specific extensions:
        top_k: 10,
        min_p: 0.05,
        repetition_penalty: 1.1,
        conversation: None,
        ..Default::default()
    };

    // Verify SGLang extensions are present
    assert_eq!(request.top_k, 10);
    assert_eq!(request.min_p, 0.05);
    assert_eq!(request.repetition_penalty, 1.1);

    // Verify serialization works with SGLang extensions
    let json = serde_json::to_string(&request).expect("Serialization should work");
    let parsed: ResponsesRequest =
        serde_json::from_str(&json).expect("Deserialization should work");

    assert_eq!(parsed.top_k, 10);
    assert_eq!(parsed.min_p, 0.05);
    assert_eq!(parsed.repetition_penalty, 1.1);
}

#[test]
fn test_usage_conversion() {
    // Construct UsageInfo directly with cached token details
    let usage_info = UsageInfo {
        prompt_tokens: 15,
        completion_tokens: 25,
        total_tokens: 40,
        reasoning_tokens: Some(8),
        prompt_tokens_details: Some(smg::protocols::common::PromptTokenUsageInfo {
            cached_tokens: 3,
        }),
    };
    let response_usage = usage_info.to_response_usage();

    assert_eq!(response_usage.input_tokens, 15);
    assert_eq!(response_usage.output_tokens, 25);
    assert_eq!(response_usage.total_tokens, 40);

    // Check details are converted correctly
    assert!(response_usage.input_tokens_details.is_some());
    assert_eq!(
        response_usage
            .input_tokens_details
            .as_ref()
            .unwrap()
            .cached_tokens,
        3
    );

    assert!(response_usage.output_tokens_details.is_some());
    assert_eq!(
        response_usage
            .output_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        8
    );

    let back_to_usage = response_usage.to_usage_info();
    assert_eq!(back_to_usage.prompt_tokens, 15);
    assert_eq!(back_to_usage.completion_tokens, 25);
    assert_eq!(back_to_usage.reasoning_tokens, Some(8));
}

#[test]
fn test_reasoning_param_default() {
    let param = ResponseReasoningParam {
        effort: Some(ReasoningEffort::Medium),
        summary: None,
    };

    let json = serde_json::to_string(&param).unwrap();
    let parsed: ResponseReasoningParam = serde_json::from_str(&json).unwrap();

    assert!(matches!(parsed.effort, Some(ReasoningEffort::Medium)));
}

#[test]
fn test_json_serialization() {
    let request = ResponsesRequest {
        include: None,
        input: ResponseInput::Text("Test input".to_string()),
        instructions: Some("Test instructions".to_string()),
        max_output_tokens: Some(200),
        max_tool_calls: Some(5),
        metadata: None,
        model: "gpt-4".to_string(),
        parallel_tool_calls: Some(false),
        previous_response_id: None,
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::High),
            summary: None,
        }),
        service_tier: Some(ServiceTier::Priority),
        store: Some(false),
        stream: Some(true),
        temperature: Some(0.9),
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::Required)),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        })]),
        top_logprobs: Some(10),
        top_p: Some(0.8),
        truncation: Some(Truncation::Auto),
        text: None,
        user: Some("test_user".to_string()),
        request_id: Some("resp_comprehensive_test".to_string()),
        priority: 1,
        frequency_penalty: Some(0.3),
        presence_penalty: Some(0.4),
        stop: None,
        top_k: 50,
        min_p: 0.1,
        repetition_penalty: 1.2,
        conversation: None,
        ..Default::default()
    };

    let json = serde_json::to_string(&request).expect("Serialization should work");
    let parsed: ResponsesRequest =
        serde_json::from_str(&json).expect("Deserialization should work");

    assert_eq!(
        parsed.request_id,
        Some("resp_comprehensive_test".to_string())
    );
    assert_eq!(parsed.model, "gpt-4");
    assert_eq!(parsed.stream, Some(true));
    assert_eq!(parsed.tools.as_ref().map(|t| t.len()), Some(1));
}
