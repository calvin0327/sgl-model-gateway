//! Compatibility types removed from `openai-protocol` 1.13.

use std::collections::HashMap;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::protocols::worker::JobStatus;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerConfigRequest {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_connector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_engine_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub labels: HashMap<String, String>,
    #[serde(default = "default_health_check_timeout")]
    pub health_check_timeout_secs: u64,
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval_secs: u64,
    #[serde(default = "default_health_success_threshold")]
    pub health_success_threshold: u32,
    #[serde(default = "default_health_failure_threshold")]
    pub health_failure_threshold: u32,
    #[serde(default)]
    pub disable_health_check: bool,
    #[serde(default = "default_max_connection_attempts")]
    pub max_connection_attempts: u32,
    #[serde(default)]
    pub dp_aware: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResponsesGetParams {
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
    #[serde(default)]
    pub starting_after: Option<i64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerLoadsResult {
    pub loads: Vec<WorkerLoadInfo>,
    pub total_workers: usize,
    pub successful: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_id: String,
    pub priority: u32,
    pub cost: f32,
    pub worker_type: String,
    pub is_healthy: bool,
    pub load: usize,
    pub connection_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    pub disable_health_check: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_status: Option<JobStatus>,
}

impl WorkerInfo {
    pub fn pending(worker_id: &str, url: String, job_status: Option<JobStatus>) -> Self {
        Self {
            id: worker_id.to_string(),
            url,
            model_id: crate::protocols::UNKNOWN_MODEL_ID.to_string(),
            priority: 0,
            cost: 1.0,
            worker_type: crate::protocols::UNKNOWN_MODEL_ID.to_string(),
            is_healthy: false,
            load: 0,
            connection_mode: crate::protocols::UNKNOWN_MODEL_ID.to_string(),
            runtime_type: None,
            tokenizer_path: None,
            reasoning_parser: None,
            tool_parser: None,
            chat_template: None,
            bootstrap_port: None,
            metadata: HashMap::new(),
            disable_health_check: false,
            job_status,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_check_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_check_interval_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_success_threshold: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_failure_threshold: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_health_check: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlushCacheResult {
    pub successful: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub total_workers: usize,
    pub http_workers: usize,
    pub message: String,
}

impl IntoResponse for FlushCacheResult {
    fn into_response(self) -> Response {
        let status = if self.failed.is_empty() {
            StatusCode::OK
        } else {
            StatusCode::PARTIAL_CONTENT
        };
        let mut body = json!({
            "status": if self.failed.is_empty() { "success" } else { "partial_success" },
            "message": self.message,
            "workers_flushed": self.successful.len(),
            "total_http_workers": self.http_workers,
            "total_workers": self.total_workers
        });
        if !self.failed.is_empty() {
            body["successful"] = json!(self.successful);
            body["failed"] = json!(self
                .failed
                .into_iter()
                .map(|(url, error)| json!({"worker": url, "error": error}))
                .collect::<Vec<_>>());
        }
        (status, Json(body)).into_response()
    }
}

impl IntoResponse for WorkerLoadsResult {
    fn into_response(self) -> Response {
        let loads: Vec<serde_json::Value> = self
            .loads
            .iter()
            .map(|info| json!({"worker": info.worker, "load": info.load}))
            .collect();
        Json(json!({"workers": loads})).into_response()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerLoadInfo {
    pub worker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,
    pub load: isize,
}

fn default_health_check_timeout() -> u64 {
    30
}

fn default_health_check_interval() -> u64 {
    60
}

fn default_health_success_threshold() -> u32 {
    2
}

fn default_health_failure_threshold() -> u32 {
    3
}

fn default_max_connection_attempts() -> u32 {
    20
}
