//! Worker selection stage: Select appropriate worker(s) based on routing mode

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::PipelineStage;
use crate::routers::{error, grpc::context::RequestContext};

/// Validate the worker selection supplied by the gateway Scheduler.
///
/// Worker selection is deliberately outside the gRPC pipeline. This stage is
/// kept so that the pipeline preserves its ordering and can validate that a
/// request entered with the expected execution mode.
pub(crate) struct WorkerSelectionStage {
    mode: WorkerSelectionMode,
}

pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(mode: WorkerSelectionMode) -> Self {
        Self { mode }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "No Scheduler selection was supplied"
            );
            error::service_unavailable(
                "scheduler_selection_required",
                "Request must be scheduled before entering the gRPC pipeline",
            )
        })?;

        let valid = match self.mode {
            WorkerSelectionMode::Regular => matches!(
                workers,
                crate::routers::grpc::context::WorkerSelection::Single { .. }
            ),
            WorkerSelectionMode::PrefillDecode => matches!(
                workers,
                crate::routers::grpc::context::WorkerSelection::Dual { .. }
            ),
        };
        if !valid {
            return Err(error::bad_request(
                "invalid_scheduler_selection",
                "Scheduler selection does not match the gRPC pipeline mode",
            ));
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}
