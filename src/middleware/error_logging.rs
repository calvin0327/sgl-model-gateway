use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::{body::Body, extract::Request, http::StatusCode, middleware::Next, response::Response};
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame, SizeHint};
use serde_json::Value;
use tracing::{error, warn, Span};

use crate::routers::error::extract_error_code_from_response;

const MAX_ERROR_BODY_BYTES: usize = 8192;
const MAX_ERROR_FIELD_CHARS: usize = 1024;

pub async fn log_error_response(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    let status = response.status();
    if !status.is_client_error() && !status.is_server_error() {
        return response;
    }

    let error_code = extract_error_code_from_response(&response).to_owned();
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(ErrorLoggingBody {
            inner: body,
            status,
            error_code,
            span: Span::current(),
            captured: Vec::new(),
            truncated: false,
            logged: false,
        }),
    )
}

struct ErrorLoggingBody {
    inner: Body,
    status: StatusCode,
    error_code: String,
    span: Span,
    captured: Vec<u8>,
    truncated: bool,
    logged: bool,
}

impl ErrorLoggingBody {
    fn log(&mut self, complete: bool) {
        if self.logged {
            return;
        }
        self.logged = true;
        let (body_code, message) = error_summary(&self.captured, self.truncated || !complete);
        let error_code = if self.error_code.is_empty() {
            body_code
        } else {
            bounded_field(&self.error_code)
        };

        if self.status.is_server_error() {
            error!(
                target: "smg::response",
                parent: &self.span,
                status_code = self.status.as_u16(),
                error_code,
                error_message = message,
                error_body_truncated = self.truncated,
                error_body_complete = complete,
                "HTTP error response details"
            );
        } else {
            warn!(
                target: "smg::response",
                parent: &self.span,
                status_code = self.status.as_u16(),
                error_code,
                error_message = message,
                error_body_truncated = self.truncated,
                error_body_complete = complete,
                "HTTP error response details"
            );
        }
    }
}

impl HttpBody for ErrorLoggingBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let frame = Pin::new(&mut this.inner).poll_frame(context);
        match &frame {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    let remaining = MAX_ERROR_BODY_BYTES - this.captured.len();
                    let captured_len = remaining.min(data.len());
                    this.captured.extend_from_slice(&data[..captured_len]);
                    this.truncated |= captured_len < data.len();
                }
                if this.inner.is_end_stream() {
                    this.log(true);
                }
            }
            Poll::Ready(None) => this.log(true),
            Poll::Ready(Some(Err(_))) => this.log(false),
            Poll::Pending => {}
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for ErrorLoggingBody {
    fn drop(&mut self) {
        self.log(self.inner.is_end_stream());
    }
}

fn bounded_field(value: &str) -> String {
    let mut characters = value.chars();
    let mut result: String = characters
        .by_ref()
        .take(MAX_ERROR_FIELD_CHARS)
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect();
    if characters.next().is_some() {
        result.push_str(" [truncated]");
    }
    result
}

fn error_summary(body: &[u8], incomplete: bool) -> (String, String) {
    if body.is_empty() {
        return (String::new(), "No error response body captured".to_owned());
    }

    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        let detail = value.get("error").unwrap_or(&value);
        let code = detail
            .get("code")
            .and_then(Value::as_str)
            .or_else(|| detail.get("type").and_then(Value::as_str))
            .map(bounded_field)
            .unwrap_or_default();
        let message = detail
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| detail.get("detail").and_then(Value::as_str))
            .or_else(|| detail.as_str());
        if let Some(message) = message {
            return (code, bounded_field(message));
        }
        if let Some(validation_errors) = detail.get("detail").and_then(Value::as_array) {
            let messages: Vec<_> = validation_errors
                .iter()
                .take(8)
                .filter_map(|entry| entry.get("msg").and_then(Value::as_str))
                .map(bounded_field)
                .collect();
            return (code, bounded_field(&messages.join("; ")));
        }
        return (
            code,
            "JSON error body has no recognized message field".to_owned(),
        );
    }

    let text = String::from_utf8_lossy(body);
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let message = if incomplete {
            "Incomplete JSON error body; inspect the client response for details"
        } else {
            "Invalid JSON error body; inspect the client response for details"
        };
        return (String::new(), message.to_owned());
    }
    (String::new(), bounded_field(&text))
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        io::{self, Write},
        sync::{Arc, Mutex},
        time::Duration,
    };

    use axum::{
        http::{HeaderMap, HeaderValue},
        routing::get,
        Router,
    };
    use futures_util::stream;
    use http_body_util::{BodyExt, StreamBody};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use super::*;
    use crate::middleware::{create_logging_layer, RequestIdLayer};

    #[derive(Clone, Default)]
    struct LogWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for LogWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn wrap(body: Body) -> ErrorLoggingBody {
        ErrorLoggingBody {
            inner: body,
            status: StatusCode::BAD_REQUEST,
            error_code: String::new(),
            span: Span::none(),
            captured: Vec::new(),
            truncated: false,
            logged: false,
        }
    }

    #[test]
    fn summaries_extract_diagnostics_without_unrelated_payloads() {
        assert_eq!(
            error_summary(
                br#"{"error":{"code":null,"type":"BadRequestError","message":"Unknown model"},"messages":["private prompt"]}"#,
                false,
            ),
            ("BadRequestError".to_owned(), "Unknown model".to_owned())
        );
        assert_eq!(
            error_summary(
                br#"{"detail":[{"msg":"Field required","input":{"api_key":"secret","messages":["private prompt"]}}]}"#,
                false,
            ),
            (String::new(), "Field required".to_owned())
        );
        assert_eq!(
            error_summary(b"model must be specified\n", false),
            (String::new(), "model must be specified ".to_owned())
        );
        let (_, unknown) = error_summary(br#"{"input":"secret"}"#, false);
        assert!(!unknown.contains("secret"));
        let (_, incomplete) = error_summary(br#"{"input":"secret"#, true);
        assert!(!incomplete.contains("secret"));
        assert!(incomplete.contains("Incomplete JSON"));
    }

    #[test]
    fn summaries_bound_unicode_and_remove_control_characters() {
        let message = format!("\n{}\u{001b}", "\u{4e2d}".repeat(MAX_ERROR_FIELD_CHARS + 1));
        let bounded = bounded_field(&message);
        assert!(bounded.starts_with(' '));
        assert!(bounded.ends_with(" [truncated]"));
        assert!(!bounded.chars().any(char::is_control));
        assert_eq!(bounded.chars().count(), MAX_ERROR_FIELD_CHARS + 12);
    }

    #[tokio::test]
    #[serial_test::serial(error_logging)]
    async fn error_body_preserves_bytes_trailers_and_bounds_capture() {
        let bytes = Bytes::from(vec![b'x'; MAX_ERROR_BODY_BYTES * 2]);
        let mut trailers = HeaderMap::new();
        trailers.insert("x-trailer", HeaderValue::from_static("retained"));
        let frames = stream::iter(vec![
            Ok::<_, Infallible>(Frame::data(bytes.clone())),
            Ok(Frame::trailers(trailers.clone())),
        ]);
        let mut body = wrap(Body::new(StreamBody::new(frames)));
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            bytes
        );
        assert_eq!(body.captured.len(), MAX_ERROR_BODY_BYTES);
        assert!(body.truncated);
        assert_eq!(
            body.frame()
                .await
                .unwrap()
                .unwrap()
                .into_trailers()
                .unwrap(),
            trailers
        );
        assert!(body.frame().await.is_none());
        assert!(body.logged);
    }

    #[tokio::test]
    #[serial_test::serial(error_logging)]
    async fn body_read_errors_are_preserved() {
        let stream = stream::iter(vec![
            Ok(Bytes::from_static(b"partial")),
            Err(io::Error::other("upstream disconnected")),
        ]);
        let mut body = wrap(Body::from_stream(stream));
        assert!(body.frame().await.unwrap().is_ok());
        assert!(!body.logged);
        let error = body.frame().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("upstream disconnected"));
        assert!(body.logged);
    }

    #[tokio::test]
    #[serial_test::serial(error_logging)]
    async fn streamed_errors_log_details_with_request_id_without_buffering_response() {
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_span_list(false)
            .with_writer(move || writer.clone())
            .finish();

        async {
            let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(4);
            let receiver = Arc::new(Mutex::new(Some(receiver)));
            let app = Router::new()
                .route("/ok", get(|| async { "healthy" }))
                .route(
                    "/error",
                    get(move || {
                        let receiver = receiver.lock().unwrap().take().unwrap();
                        async move {
                            let mut response =
                                Response::new(Body::from_stream(ReceiverStream::new(receiver)));
                            *response.status_mut() = StatusCode::BAD_REQUEST;
                            response
                                .headers_mut()
                                .insert("x-original", HeaderValue::from_static("retained"));
                            response.extensions_mut().insert(42_u32);
                            response
                        }
                    }),
                )
                .layer(axum::middleware::from_fn(log_error_response))
                .layer(create_logging_layer())
                .layer(RequestIdLayer::new(vec!["x-request-id".to_owned()]));
            let request = Request::builder()
                .uri("/error")
                .header("x-request-id", "test-error-request")
                .body(Body::empty())
                .unwrap();
            let response =
                tokio::time::timeout(Duration::from_secs(1), app.clone().oneshot(request))
                    .await
                    .expect("error logging must not wait for response body chunks")
                    .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(response.headers()["x-original"], "retained");
            assert_eq!(response.extensions().get::<u32>(), Some(&42));

            let first = br#"{"error":{"code":"invalid_model","message":"#;
            let second = br#""Unknown model"}}"#;
            sender.send(Ok(Bytes::from_static(first))).await.unwrap();
            sender.send(Ok(Bytes::from_static(second))).await.unwrap();
            drop(sender);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(&body[..], [first.as_slice(), second.as_slice()].concat());

            let response = app
                .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(&body[..], b"healthy");
        }
        .with_subscriber(subscriber)
        .await;

        let logs = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
        let details: Vec<Value> = logs
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|entry| entry["fields"]["message"] == "HTTP error response details")
            .collect();
        assert_eq!(details.len(), 1, "{logs}");
        let details = &details[0];
        assert_eq!(details["level"], "WARN");
        assert_eq!(details["fields"]["error_code"], "invalid_model");
        assert_eq!(details["fields"]["error_message"], "Unknown model");
        assert_eq!(details["fields"]["error_body_complete"], true);
        assert_eq!(details["span"]["request_id"], "test-error-request");
        assert!(details["span"]["latency_us"].is_number());
    }

    #[test]
    #[serial_test::serial(error_logging)]
    fn unconsumed_server_error_logs_incomplete_body_once() {
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            let mut body = wrap(Body::from("unconsumed"));
            body.status = StatusCode::SERVICE_UNAVAILABLE;
            body.error_code = "no_available_worker".to_owned();
            drop(body);
        });
        let logs = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
        let entries: Vec<Value> = logs
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(entries.len(), 1, "{logs}");
        assert_eq!(entries[0]["level"], "ERROR");
        assert_eq!(entries[0]["fields"]["error_code"], "no_available_worker");
        assert_eq!(entries[0]["fields"]["error_body_complete"], false);
        assert_eq!(
            entries[0]["fields"]["error_message"],
            "No error response body captured"
        );
    }
}
