//! Factory for creating router instances

use std::sync::Arc;

use super::{
    http::{pd_router::PDRouter, router::Router},
    RouterTrait,
};
use crate::app_context::AppContext;

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a regular router
    pub fn create_regular_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = Router::new(ctx)?;

        Ok(Box::new(router))
    }

    /// Create a PD router using policies initialized in the application context.
    pub fn create_pd_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = PDRouter::new(ctx)?;

        Ok(Box::new(router))
    }
}
