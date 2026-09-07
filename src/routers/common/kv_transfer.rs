//! vLLM PD KV-transfer connector handling.

use crate::core::Worker;

const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum KvConnectorMode {
    Mooncake {
        host: String,
        port: u16,
        engine_id: Option<String>,
    },
    Nixl,
    Passthrough,
}

pub(crate) fn connector_mode_for_worker(worker: &dyn Worker) -> KvConnectorMode {
    let metadata = worker.metadata();
    let engine_id = effective_kv_engine_id(
        metadata.kv_engine_id.as_deref(),
        worker.dp_size(),
        worker.dp_rank(),
    );
    match metadata.kv_connector.as_deref() {
        Some("MooncakeConnector") => KvConnectorMode::Mooncake {
            host: metadata.bootstrap_host.clone(),
            port: metadata.bootstrap_port.unwrap_or(DEFAULT_BOOTSTRAP_PORT),
            engine_id,
        },
        Some("NixlConnector") => KvConnectorMode::Nixl,
        _ => KvConnectorMode::Passthrough,
    }
}

pub(crate) fn effective_kv_engine_id(
    base: Option<&str>,
    dp_size: Option<usize>,
    dp_rank: Option<usize>,
) -> Option<String> {
    let base = base.filter(|base| !base.is_empty())?;
    if dp_size.unwrap_or(1) > 1 {
        dp_rank.map(|rank| format!("{}_dp{}", base, rank))
    } else {
        Some(base.to_string())
    }
}

pub(crate) fn mooncake_prefill_params(transfer_id: &str) -> serde_json::Value {
    serde_json::json!({
        "do_remote_decode": true,
        "do_remote_prefill": false,
        "transfer_id": transfer_id,
    })
}

pub(crate) fn mooncake_decode_params(
    transfer_id: &str,
    engine_id: &str,
    host: &str,
    port: u16,
) -> serde_json::Value {
    serde_json::json!({
        "do_remote_decode": false,
        "do_remote_prefill": true,
        "transfer_id": transfer_id,
        "remote_engine_id": engine_id,
        "remote_bootstrap_addr": format!("http://{}:{}", host, port),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DPAwareWorkerBuilder, WorkerType};

    #[test]
    fn connector_mode_uses_worker_metadata_and_dp_rank() {
        let worker = DPAwareWorkerBuilder::new("http://prefill:30000", 1, 2)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .kv_connector("MooncakeConnector")
            .kv_engine_id("engine")
            .build();

        assert_eq!(
            connector_mode_for_worker(&worker),
            KvConnectorMode::Mooncake {
                host: "prefill".to_string(),
                port: 8998,
                engine_id: Some("engine_dp1".to_string()),
            }
        );
    }
}
