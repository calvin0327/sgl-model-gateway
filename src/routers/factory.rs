//! Factory for creating router instances

use std::sync::Arc;

use super::{
    grpc::{pd_router::GrpcPDRouter, router::GrpcRouter},
    http::{pd_router::PDRouter, router::Router},
    openai::OpenAIRouter,
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

    /// Create a gRPC router with injected policy
    pub fn create_grpc_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = GrpcRouter::new(ctx)?;

        Ok(Box::new(router))
    }

    /// Create a gRPC PD router using policies initialized in the application context.
    pub fn create_grpc_pd_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = GrpcPDRouter::new(ctx)?;
        Ok(Box::new(router))
    }

    /// Create an OpenAI router
    ///
    /// Workers should be registered via the external worker registration workflow
    /// before using this router. The workflow discovers models from the provided
    /// endpoints and creates external workers in the registry.
    pub fn create_openai_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = OpenAIRouter::new(ctx)?;
        Ok(Box::new(router))
    }
}
