// These modules are used by tests and benchmarks
#![allow(dead_code)]

pub mod mock_worker;
pub mod streaming_helpers;
pub mod test_app;
pub mod test_certs;
pub mod test_config;
pub mod tls_mock_worker;

// Re-export commonly used test builders
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use mock_worker::{MockWorker, MockWorkerConfig};
use serde_json::json;
use smg::{
    app_context::AppContext,
    config::GatewayConfig,
    core::{
        BasicWorkerBuilder, Job, LoadMonitor, ModelCard, RuntimeType, Worker, WorkerRegistry,
        WorkerType,
    },
    middleware::TokenBucket,
    policies::{PolicyFactory, PolicyRegistry},
    protocols::common::{Function, Tool},
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::{router_manager::RouterManager, RouterTrait},
    tokenizer::registry::TokenizerRegistry,
    tool_parser::ParserFactory as ToolParserFactory,
};
#[allow(unused_imports)]
pub use test_config::{TestGatewayConfig, TestGatewayConfigBuilder, TestWorkerConfig};

/// Test context for directly testing mock workers without full router setup.
pub struct WorkerTestContext {
    pub workers: Vec<MockWorker>,
    pub worker_urls: Vec<String>,
}

impl WorkerTestContext {
    pub async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        Self {
            workers,
            worker_urls,
        }
    }

    pub fn first_worker_url(&self) -> Option<&str> {
        self.worker_urls.first().map(|s| s.as_str())
    }

    pub async fn make_request(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let client = reqwest::Client::new();
        let worker_url = self
            .first_worker_url()
            .ok_or_else(|| "No workers available".to_string())?;

        let response = client
            .post(format!("{}{}", worker_url, endpoint))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Request failed with status: {}", response.status()));
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))
    }

    pub async fn make_streaming_request(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<Vec<String>, String> {
        use futures_util::StreamExt;

        let client = reqwest::Client::new();
        let worker_url = self
            .first_worker_url()
            .ok_or_else(|| "No workers available".to_string())?;

        let response = client
            .post(format!("{}{}", worker_url, endpoint))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Request failed with status: {}", response.status()));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.contains("text/event-stream") {
            return Err("Response is not a stream".to_string());
        }

        let mut stream = response.bytes_stream();
        let mut events = Vec::new();

        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if let Some(stripped) = line.strip_prefix("data: ") {
                        events.push(stripped.to_string());
                    }
                }
            }
        }

        Ok(events)
    }

    pub async fn shutdown(mut self) {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        for worker in &mut self.workers {
            worker.stop().await;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

/// Test context for integration tests that go through the full axum app stack.
pub struct AppTestContext {
    pub workers: Vec<MockWorker>,
    pub router: Arc<dyn RouterTrait>,
    pub config: GatewayConfig,
    pub app_context: Arc<AppContext>,
}

impl AppTestContext {
    pub async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        let mut config = TestGatewayConfig::random(3002);
        config.workers.startup_timeout_secs = 1;
        config.workers.startup_check_interval_secs = 1;

        Self::new_with_config(config, worker_configs).await
    }

    pub async fn new_with_config(
        mut config: GatewayConfig,
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        if config.workers.prefill_urls.is_empty()
            && config.workers.decode_urls.is_empty()
            && config.workers.openai_urls.is_empty()
            && config.workers.regular_urls.is_empty()
        {
            config.workers.regular_urls = worker_urls.clone();
        }

        let app_context = create_test_context(config.clone()).await;

        if !worker_urls.is_empty() {
            let job_queue = app_context
                .worker_job_queue
                .get()
                .expect("JobQueue should be initialized");
            let job = Job::InitializeWorkersFromConfig {
                gateway_config: Box::new(config.clone()),
            };
            job_queue
                .submit(job)
                .await
                .expect("Failed to submit worker initialization job");

            let expected_count = worker_urls.len();
            let start = tokio::time::Instant::now();
            let timeout_duration = tokio::time::Duration::from_secs(10);
            loop {
                let healthy_workers = app_context
                    .worker_registry
                    .get_all()
                    .iter()
                    .filter(|w| w.is_healthy())
                    .count();

                if healthy_workers >= expected_count {
                    break;
                }

                if start.elapsed() > timeout_duration {
                    panic!(
                        "Timeout waiting for {} workers to become healthy (only {} ready)",
                        expected_count, healthy_workers
                    );
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        let router = create_router_manager(&app_context);

        Self {
            workers,
            router,
            config,
            app_context,
        }
    }

    pub async fn create_app(&self) -> axum::Router {
        test_app::create_test_app_with_context(
            Arc::clone(&self.router),
            Arc::clone(&self.app_context),
        )
    }

    pub async fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

/// Create the full request-entry router used by gateway integration tests.
pub fn create_router_manager(app_context: &Arc<AppContext>) -> Arc<dyn RouterTrait> {
    RouterManager::from_config(&app_context.config, app_context)
        .expect("Failed to create router manager")
}

/// Helper function to create AppContext for tests
pub async fn create_test_context(config: GatewayConfig) -> Arc<AppContext> {
    let client = reqwest::Client::new();

    // Initialize rate limiter
    let rate_limiter = match config.server.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = config
                .server
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.routing.policy.clone()));
    let prefill_policy = PolicyFactory::create_from_config(
        config
            .routing
            .prefill_policy
            .as_ref()
            .unwrap_or(&config.routing.policy),
    );
    let decode_policy = PolicyFactory::create_from_config(
        config
            .routing
            .decode_policy
            .as_ref()
            .unwrap_or(&config.routing.policy),
    );
    policy_registry.set_prefill_policy(prefill_policy);
    policy_registry.set_decode_policy(decode_policy);

    // Initialize load monitor
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.workers.startup_check_interval_secs,
    )));
    // Create empty OnceLock for worker job queue and workflow engines
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

    let app_context = Arc::new(
        AppContext::builder()
            .gateway_config(config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new())) // tokenizer
            .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
            .tool_parser_factory(Some(ToolParserFactory::new()))
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .load_monitor(load_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .build()
            .unwrap(),
    );

    // Initialize JobQueue after AppContext is created
    let weak_context = Arc::downgrade(&app_context);
    let job_queue = smg::core::JobQueue::new(smg::core::JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize typed workflow engines
    use smg::core::steps::WorkflowEngines;
    let engines = WorkflowEngines::new(&config);
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");

    // Register external workers for OpenAI mode
    for url in &config.workers.openai_urls {
        // Create a worker that supports common test models
        let models = vec![
            ModelCard::new("mock-model"),
            ModelCard::new("gpt-4"),
            ModelCard::new("gpt-3.5-turbo"),
        ];
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .models(models)
                .build(),
        );
        app_context.worker_registry.register(worker);
    }

    app_context
}

/// Helper function to create AppContext for tests with parser factories initialized
pub async fn create_test_context_with_parsers(config: GatewayConfig) -> Arc<AppContext> {
    let client = reqwest::Client::new();

    // Initialize rate limiter
    let rate_limiter = match config.server.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = config
                .server
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    // Initialize registries
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.routing.policy.clone()));

    // Initialize load monitor
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.workers.startup_check_interval_secs,
    )));
    // Create empty OnceLock for worker job queue and workflow engines
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

    // Initialize parser factories
    let reasoning_parser_factory = Some(ReasoningParserFactory::new());
    let tool_parser_factory = Some(ToolParserFactory::new());

    let app_context = Arc::new(
        AppContext::builder()
            .gateway_config(config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer_registry(tokenizer_registry)
            .reasoning_parser_factory(reasoning_parser_factory)
            .tool_parser_factory(tool_parser_factory)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .load_monitor(load_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .build()
            .unwrap(),
    );

    // Initialize JobQueue after AppContext is created
    let weak_context = Arc::downgrade(&app_context);
    let job_queue = smg::core::JobQueue::new(smg::core::JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize typed workflow engines
    use smg::core::steps::WorkflowEngines;
    let engines = WorkflowEngines::new(&config);
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");

    // Register external workers for OpenAI mode
    for url in &config.workers.openai_urls {
        // Create a worker that supports common test models
        let models = vec![
            ModelCard::new("mock-model"),
            ModelCard::new("gpt-4"),
            ModelCard::new("gpt-3.5-turbo"),
        ];
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .models(models)
                .build(),
        );
        app_context.worker_registry.register(worker);
    }

    app_context
}

// Tokenizer download configuration
const TINYLLAMA_TOKENIZER_URL: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json";
const CACHE_DIR: &str = ".tokenizer_cache";
const TINYLLAMA_TOKENIZER_FILENAME: &str = "tinyllama_tokenizer.json";

// Global mutex to prevent concurrent downloads
static DOWNLOAD_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

/// Downloads the TinyLlama tokenizer from HuggingFace if not already cached.
/// Returns the path to the cached tokenizer file.
///
/// This function is thread-safe and will only download the tokenizer once
/// even if called from multiple threads concurrently.
pub fn ensure_tokenizer_cached() -> PathBuf {
    // Get or initialize the mutex
    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));

    // Lock to ensure only one thread downloads at a time
    let _guard = mutex.lock().unwrap();

    let cache_dir = PathBuf::from(CACHE_DIR);
    let tokenizer_path = cache_dir.join(TINYLLAMA_TOKENIZER_FILENAME);

    // Create cache directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    // Download tokenizer if not already cached
    if !tokenizer_path.exists() {
        println!("Downloading TinyLlama tokenizer from HuggingFace...");

        // Use blocking reqwest client since we're in tests/benchmarks
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(TINYLLAMA_TOKENIZER_URL)
            .send()
            .expect("Failed to download tokenizer");

        if !response.status().is_success() {
            panic!("Failed to download tokenizer: HTTP {}", response.status());
        }

        let content = response.bytes().expect("Failed to read tokenizer content");

        if content.len() < 100 {
            panic!("Downloaded content too small: {} bytes", content.len());
        }

        fs::write(&tokenizer_path, content).expect("Failed to write tokenizer to cache");
        println!(
            "Tokenizer downloaded and cached successfully ({} bytes)",
            tokenizer_path.metadata().unwrap().len()
        );
    }

    tokenizer_path
}

/// Common test prompts for consistency across tests
pub const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

/// Pre-computed hashes for verification
pub const EXPECTED_HASHES: [u64; 4] = [
    1209591529327510910,
    4181375434596349981,
    6245658446118930933,
    5097285695902185237,
];

/// Create a comprehensive set of test tools covering all parser test scenarios
#[allow(dead_code)]
pub fn create_test_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "search".to_string(),
                description: Some("Search for information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "location": {"type": "string"},
                        "date": {"type": "string"},
                        "units": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculate".to_string(),
                description: Some("Perform calculations".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
    ]
}
