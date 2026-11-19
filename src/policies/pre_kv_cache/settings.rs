//! Runtime tuning changes scoring only; subscriptions and cached tokens survive.

use std::sync::Arc;

use axum::{
    extract::{rejection::JsonRejection, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{CostModel, Runtime};
use crate::app_state::AppState;

/// Replace both importance values together so readers never observe half an update.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Weights {
    pub kv_importance: f64,
    pub load_importance: f64,
}

impl From<&CostModel> for Weights {
    fn from(cost: &CostModel) -> Self {
        Self {
            kv_importance: cost.kv_importance,
            load_importance: cost.load_importance,
        }
    }
}

impl Weights {
    fn valid(self) -> bool {
        [self.kv_importance, self.load_importance]
            .iter()
            .all(|value| value.is_finite() && (0.0..=1000.0).contains(value))
            && (self.kv_importance > 0.0 || self.load_importance > 0.0)
    }

    pub(super) fn apply(self, cost: &mut CostModel) {
        cost.kv_importance = self.kv_importance;
        cost.load_importance = self.load_importance;
    }
}

#[derive(Debug)]
pub(crate) enum SettingsError {
    InvalidWeights,
    UnknownProfile,
}

impl IntoResponse for SettingsError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::InvalidWeights => (
                StatusCode::BAD_REQUEST,
                "kv_importance and load_importance must be finite numbers between 0 and 1000, with at least one positive",
            ),
            Self::UnknownProfile => (StatusCode::NOT_FOUND, "unknown pre_kv_cache profile"),
        };
        (status, Json(json!({"error": message}))).into_response()
    }
}

impl Runtime {
    pub(crate) fn weights(&self, profile: &str) -> Option<Weights> {
        self.weights.read().get(profile).copied()
    }

    pub(crate) fn update_weights(
        &self,
        profile: &str,
        update: Weights,
    ) -> Result<Weights, SettingsError> {
        if !update.valid() {
            return Err(SettingsError::InvalidWeights);
        }
        let mut weights = self.weights.write();
        let current = weights
            .get_mut(profile)
            .ok_or(SettingsError::UnknownProfile)?;
        let previous = *current;
        *current = update;
        tracing::info!(
            profile,
            previous_kv_importance = previous.kv_importance,
            previous_load_importance = previous.load_importance,
            kv_importance = update.kv_importance,
            load_importance = update.load_importance,
            "pre_kv_cache runtime weights updated; restart restores startup configuration"
        );
        Ok(update)
    }
}

fn runtime(state: &AppState) -> Result<Arc<Runtime>, Response> {
    state
        .context
        .policy_registry
        .pre_kv_cache_runtime()
        .ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "pre_kv_cache runtime is not installed"})),
            )
                .into_response()
        })
}

fn response(profile: &str, weights: Weights) -> Response {
    Json(json!({
        "profile": profile,
        "kv_importance": weights.kv_importance,
        "load_importance": weights.load_importance,
        "persistence": "process_memory",
        "restart_resets_weights": true,
    }))
    .into_response()
}

pub(crate) async fn get_weights(
    State(state): State<Arc<AppState>>,
    Path(profile): Path<String>,
) -> Response {
    let runtime = match runtime(&state) {
        Ok(runtime) => runtime,
        Err(error) => return error,
    };
    match runtime.weights(&profile) {
        Some(weights) => response(&profile, weights),
        None => SettingsError::UnknownProfile.into_response(),
    }
}

pub(crate) async fn get_status(State(state): State<Arc<AppState>>) -> Response {
    match runtime(&state) {
        Ok(runtime) => Json(runtime.status()).into_response(),
        Err(error) => error,
    }
}

pub(crate) async fn put_weights(
    State(state): State<Arc<AppState>>,
    Path(profile): Path<String>,
    body: Result<Json<Weights>, JsonRejection>,
) -> Response {
    let runtime = match runtime(&state) {
        Ok(runtime) => runtime,
        Err(error) => return error,
    };
    let Json(update) = match body {
        Ok(body) => body,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": error.body_text()})),
            )
                .into_response()
        }
    };
    match runtime.update_weights(&profile, update) {
        Ok(weights) => response(&profile, weights),
        Err(error) => error.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        app_context::AppContext,
        config::{GatewayConfig, PolicyConfig},
        middleware::AuthConfig,
        policies::pre_kv_cache::events,
        routers::{app::build_app, router_manager::RouterManager},
    };
    use axum::{body::Body, http::Request, Router};
    use serde_json::Value;
    use tower::ServiceExt;

    // Discovery enables a valid runtime before any backend has registered.
    fn config() -> super::super::Config {
        let mut config = events::tests::config();
        config.discovery = Some(super::super::Discovery::default());
        config
    }

    // Exercise real route registration and authentication, without opening ports
    // or starting backend subscriptions for a configuration-only request.
    async fn app(runtime: Option<Arc<Runtime>>) -> Router {
        let gateway = GatewayConfig::new(PolicyConfig::RoundRobin);
        let context = Arc::new(AppContext::from_config(gateway.clone()).await.unwrap());
        if let Some(runtime) = runtime {
            context.policy_registry.install_pre_kv_cache(runtime);
        }
        // RouterManager already returns its shared instance.
        let manager = RouterManager::from_config(&gateway, &context).unwrap();
        build_app(
            Arc::new(AppState {
                router: manager,
                context,
                concurrency_queue_tx: None,
                router_manager: None,
            }),
            AuthConfig {
                api_key: Some("test-admin".into()),
            },
            None,
            1024 * 1024,
            vec![],
            vec![],
        )
    }

    async fn request(app: &Router, method: &str, path: &str, body: &str) -> Response {
        app.clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("Authorization", "Bearer test-admin")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn json_body(response: Response) -> Value {
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn admin_weights_updates_round_trip_without_mutating_startup_config() {
        let mut config = config();
        config.profiles.get_mut("m").unwrap().cost.kv_importance = 3.0;
        let runtime = Runtime::new(config, reqwest::Client::new()).unwrap();
        let app = app(Some(runtime.clone())).await;
        let path = "/v1/pre-kv-cache/weights/m";
        let initial = json_body(request(&app, "GET", path, "").await).await;
        assert_eq!(initial["kv_importance"], 3.0);
        assert_eq!(initial["load_importance"], 1.0);
        assert_eq!(initial["restart_resets_weights"], true);

        let changed = request(
            &app,
            "PUT",
            path,
            r#"{"kv_importance":0.5,"load_importance":2}"#,
        )
        .await;
        assert_eq!(changed.status(), StatusCode::OK);
        let changed = json_body(changed).await;
        assert_eq!(changed["kv_importance"], 0.5);
        assert_eq!(changed["load_importance"], 2.0);
        assert_eq!(changed["profile"], "m");
        assert_eq!(changed["persistence"], "process_memory");
        assert_eq!(
            json_body(request(&app, "GET", path, "").await).await,
            changed
        );
        assert_eq!(runtime.config.profiles["m"].cost.kv_importance, 3.0);

        let restarted = Runtime::new((*runtime.config).clone(), reqwest::Client::new()).unwrap();
        assert_eq!(restarted.weights("m").unwrap().kv_importance, 3.0);
    }

    #[tokio::test]
    async fn invalid_updates_leave_both_values_unchanged() {
        let runtime = Runtime::new(config(), reqwest::Client::new()).unwrap();
        let app = app(Some(runtime.clone())).await;
        let original = runtime.weights("m").unwrap();
        for body in [
            r#"{"kv_importance":0,"load_importance":0}"#,
            r#"{"kv_importance":-1,"load_importance":2}"#,
            r#"{"kv_importance":1,"load_importance":1001}"#,
            r#"{"kv_importance":1,"load_importance":1e999}"#,
            r#"{"kv_importance":NaN,"load_importance":1}"#,
            r#"{"kv_importance":2}"#,
            r#"{"kv_importance":2,"load_importance":1,"extra":true}"#,
            r#"{"kv_importance":"2","load_importance":1}"#,
        ] {
            let response = request(&app, "PUT", "/v1/pre-kv-cache/weights/m", body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(runtime.weights("m"), Some(original));
        }
        // Non-finite numbers are rejected at the runtime boundary as well.
        for kv_importance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(runtime
                .update_weights(
                    "m",
                    Weights {
                        kv_importance,
                        load_importance: 1.0,
                    },
                )
                .is_err());
            assert_eq!(runtime.weights("m"), Some(original));
        }
        for body in [
            r#"{"kv_importance":0,"load_importance":1000}"#,
            r#"{"kv_importance":1000,"load_importance":0}"#,
        ] {
            assert_eq!(
                request(&app, "PUT", "/v1/pre-kv-cache/weights/m", body)
                    .await
                    .status(),
                StatusCode::OK,
            );
        }
    }

    #[test]
    fn concurrent_updates_never_expose_mixed_weight_pairs() {
        let runtime = Runtime::new(config(), reqwest::Client::new()).unwrap();
        let first = Weights {
            kv_importance: 2.0,
            load_importance: 0.0,
        };
        let second = Weights {
            kv_importance: 0.0,
            load_importance: 3.0,
        };
        runtime.update_weights("m", first).unwrap();
        // Opposite zero values make a torn pair either invalid or a third policy.
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..1000 {
                    runtime.update_weights("m", first).unwrap();
                }
            });
            scope.spawn(|| {
                for _ in 0..1000 {
                    runtime.update_weights("m", second).unwrap();
                }
            });
            for _ in 0..2000 {
                let snapshot = runtime.weights.read().clone();
                assert!(snapshot["m"] == first || snapshot["m"] == second);
            }
        });
    }

    #[tokio::test]
    async fn admin_endpoints_require_auth_and_report_missing_runtime_or_profile() {
        let runtime = Runtime::new(config(), reqwest::Client::new()).unwrap();
        let configured = app(Some(runtime)).await;
        let unconfigured = app(None).await;
        for (method, path, body) in [
            ("GET", "/v1/pre-kv-cache/weights/m", ""),
            (
                "PUT",
                "/v1/pre-kv-cache/weights/m",
                r#"{"kv_importance":2,"load_importance":1}"#,
            ),
            ("GET", "/v1/pre-kv-cache/status", ""),
        ] {
            let response = configured
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .header("Content-Type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(
                request(&unconfigured, method, path, body).await.status(),
                StatusCode::SERVICE_UNAVAILABLE,
            );
        }
        for (method, body) in [
            ("GET", ""),
            ("PUT", r#"{"kv_importance":2,"load_importance":1}"#),
        ] {
            assert_eq!(
                request(
                    &configured,
                    method,
                    "/v1/pre-kv-cache/weights/missing",
                    body
                )
                .await
                .status(),
                StatusCode::NOT_FOUND,
            );
        }
        assert_eq!(
            request(&configured, "GET", "/v1/pre-kv-cache/status", "")
                .await
                .status(),
            StatusCode::OK,
        );
    }
}
