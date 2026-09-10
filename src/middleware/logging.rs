use std::time::{Duration, Instant};

use axum::{extract::Request, response::Response};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{debug, error, field::Empty, info_span, warn, Span};

use super::{metrics::normalize_path_for_metrics, request_id::RequestId};
use crate::{
    observability::metrics::{method_to_static_str, Metrics},
    routers::error::extract_error_code_from_response,
};

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span
        info_span!(
            target: "smg::otel-trace",
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            request_id = Empty,  // Will be set later
            status_code = Empty,
            latency = Empty,
            latency_us = Empty,
            error = Empty
        )
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        let method = method_to_static_str(request.method().as_str());
        let path = normalize_path_for_metrics(request.uri().path());
        Metrics::record_http_request(method, &path);

        // Log the request start
        debug!(
            target: "smg::request",
            "started processing request"
        );
    }
}

/// Custom on_response handler
#[derive(Clone, Debug)]
pub struct ResponseLogger {
    _start_time: Instant,
}

impl Default for ResponseLogger {
    fn default() -> Self {
        Self {
            _start_time: Instant::now(),
        }
    }
}

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status();
        let status_code = status.as_u16();

        let error_code = extract_error_code_from_response(response);

        // Layer 1: HTTP metrics
        Metrics::record_http_response(status_code, error_code);

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status_code);
        // Use microseconds as integer to avoid format! string allocation
        span.record("latency", latency.as_micros() as u64);
        span.record("latency_us", latency.as_micros() as u64);

        // Log the response completion
        let _enter = span.enter();
        if status.is_server_error() {
            error!(
                target: "smg::response",
                error_code,
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "smg::response",
                error_code,
                "request failed with client error"
            );
        } else {
            debug!(
                target: "smg::response",
                "finished processing request"
            );
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger::default())
}
