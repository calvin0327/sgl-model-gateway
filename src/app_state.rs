use std::sync::Arc;

use crate::{
    app_context::AppContext,
    middleware::QueuedRequest,
    routers::{router_manager::RouterManager, RouterTrait},
};

/// Shared state injected into Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
}
