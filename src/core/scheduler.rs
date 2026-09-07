//! Request scheduling based on worker capabilities and load-balancing policies.
//!
//! The scheduler selects a concrete worker, or a complete Prefill/Decode pair.
//! Routers are intentionally not involved here: they only execute the selection
//! returned by this module.

use std::{collections::HashSet, sync::Arc};

use http::HeaderMap;
use thiserror::Error;
use tracing::info;

use crate::{
    core::{
        model_type::Endpoint, pd_group_id, ConnectionMode, RuntimeType, Worker, WorkerRegistry,
        WorkerType,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    tokenizer::registry::TokenizerRegistry,
};

/// Input needed by the scheduler. Request bodies and clients do not belong here.
#[derive(Debug, Clone)]
pub struct SchedulerInput<'a> {
    pub model_id: Option<&'a str>,
    pub endpoint: Endpoint,
    pub request_text: Option<&'a str>,
    pub token_ids: Option<&'a [u32]>,
    pub headers: Option<&'a HeaderMap>,
}

/// Errors produced while selecting a backend.
#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("no models are available")]
    NoModelsAvailable,

    #[error("model '{model_id}' is not available for endpoint '{endpoint}'")]
    ModelNotFound {
        model_id: String,
        endpoint: Endpoint,
    },

    #[error("model must be specified when multiple models are available: {models}")]
    ModelRequired { models: String },

    #[error("no available {mode} worker for model '{model_id}' and endpoint '{endpoint}'")]
    NoAvailableWorker {
        model_id: String,
        mode: &'static str,
        endpoint: Endpoint,
    },

    #[error("no complete PD worker pair for model '{model_id}' and endpoint '{endpoint}'")]
    NoAvailablePdPair {
        model_id: String,
        endpoint: Endpoint,
    },
}

/// The concrete backend selected for a request.
#[derive(Debug, Clone)]
pub struct SchedulerOutput {
    pub model_id: String,
    pub worker_selection: WorkerSelection,
}

/// Routing mode selected from the available worker categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMode {
    Regular,
    PrefillDecode,
}

/// A single worker or a complete PD worker pair.
#[derive(Debug, Clone)]
pub enum WorkerSelection {
    Regular {
        worker: Arc<dyn Worker>,
    },
    PrefillDecode {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        pd_group_id: String,
    },
}

impl WorkerSelection {
    /// Returns the transport used by a selection.
    pub fn connection_mode(&self) -> ConnectionMode {
        match self {
            Self::Regular { worker } => worker.connection_mode().clone(),
            Self::PrefillDecode { prefill, .. } => prefill.connection_mode().clone(),
        }
    }
}

/// Selects workers and owns all request-routing policies.
pub struct Scheduler {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer_registry: Option<Arc<TokenizerRegistry>>,
}

impl Scheduler {
    pub fn new(worker_registry: Arc<WorkerRegistry>, policy_registry: Arc<PolicyRegistry>) -> Self {
        Self::new_with_tokenizer_registry(worker_registry, policy_registry, None)
    }

    pub fn new_with_tokenizer_registry(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tokenizer_registry: Option<Arc<TokenizerRegistry>>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            tokenizer_registry,
        }
    }

    pub async fn schedule(
        &self,
        req: &SchedulerInput<'_>,
    ) -> Result<SchedulerOutput, SchedulerError> {
        let model_id = self.resolve_model_id(req.model_id)?;
        let available_workers = self.available_workers(&model_id, req.endpoint)?;

        let context_tokens = self.context_tokens(req, &model_id);
        let mut mode_order = Vec::with_capacity(3);

        if let Some(preferred_mode) =
            self.context_preferred_mode(&available_workers, context_tokens)
        {
            mode_order.push(preferred_mode);
        } else if let Some(default_mode) = Self::select_by_mode(&available_workers) {
            mode_order.push(default_mode);
        }

        // Context length determines the first choice only. The remaining
        // modes are fallback choices, so Regular and PD can recover from one
        // another when one side is unhealthy or has no complete PD pair.
        for mode in [WorkerMode::Regular, WorkerMode::PrefillDecode] {
            if !mode_order.contains(&mode) {
                mode_order.push(mode);
            }
        }

        let mut worker_selection = None;
        for mode in mode_order {
            let selection = match mode {
                WorkerMode::Regular => {
                    self.select_regular(&available_workers, &model_id, req)
                        .await
                }
                WorkerMode::PrefillDecode => {
                    self.select_pd(&available_workers, &model_id, context_tokens, req)
                        .await
                }
            };

            if selection.is_some() {
                worker_selection = selection;
                break;
            }
        }

        let worker_selection = worker_selection.ok_or_else(|| {
            let has_pd_worker = available_workers.iter().any(|worker| {
                !matches!(worker.metadata().runtime_type, RuntimeType::External)
                    && matches!(
                        worker.worker_type(),
                        WorkerType::Prefill { .. } | WorkerType::Decode
                    )
            });

            if has_pd_worker {
                SchedulerError::NoAvailablePdPair {
                    model_id: model_id.clone(),
                    endpoint: req.endpoint,
                }
            } else {
                SchedulerError::NoAvailableWorker {
                    model_id: model_id.clone(),
                    mode: "regular or PD",
                    endpoint: req.endpoint,
                }
            }
        })?;

        match &worker_selection {
            WorkerSelection::Regular { worker } => {
                info!(
                    model_id = %model_id,
                    endpoint = ?req.endpoint,
                    context_tokens = ?context_tokens,
                    worker_url = %worker.url(),
                    worker_type = ?worker.worker_type(),
                    runtime_type = %worker.metadata().runtime_type,
                    "Request scheduled to worker"
                );
            }
            WorkerSelection::PrefillDecode {
                prefill,
                decode,
                pd_group_id,
            } => {
                info!(
                    model_id = %model_id,
                    endpoint = ?req.endpoint,
                    context_tokens = ?context_tokens,
                    prefill_url = %prefill.url(),
                    decode_url = %decode.url(),
                    pd_group_id = %pd_group_id,
                    runtime_type = %prefill.metadata().runtime_type,
                    "Request scheduled to PD workers"
                );
            }
        }

        Ok(SchedulerOutput {
            model_id,
            worker_selection,
        })
    }

    fn context_tokens(&self, req: &SchedulerInput<'_>, model_id: &str) -> Option<usize> {
        if let Some(token_ids) = req.token_ids {
            return Some(token_ids.len());
        }

        let request_text = req.request_text?;
        if request_text.is_empty() {
            return Some(0);
        }

        let tokenizer = self
            .tokenizer_registry
            .as_ref()
            .and_then(|registry| registry.get(model_id));

        tokenizer
            .and_then(|tokenizer| {
                tokenizer
                    .encode(request_text, false)
                    .ok()
                    .map(|encoding| encoding.token_ids().len())
            })
            .or_else(|| Some(Self::estimate_context_tokens(request_text)))
    }

    /// Estimate input tokens when the model tokenizer is not available.
    /// ASCII text is approximated as four characters per token; non-ASCII
    /// characters are counted conservatively as one token each.
    fn estimate_context_tokens(text: &str) -> usize {
        let ascii_chars = text
            .chars()
            .filter(|character| character.is_ascii())
            .count();
        let non_ascii_chars = text
            .chars()
            .filter(|character| !character.is_ascii())
            .count();
        ascii_chars.div_ceil(4) + non_ascii_chars
    }

    fn context_preferred_mode(
        &self,
        workers: &[Arc<dyn Worker>],
        context_tokens: Option<usize>,
    ) -> Option<WorkerMode> {
        let context_tokens = context_tokens?;
        let has_context_config = workers
            .iter()
            .any(|worker| worker.metadata().preferred_context_tokens.is_some());

        if !has_context_config {
            return None;
        }

        let regular_limit = workers
            .iter()
            .filter(|worker| {
                matches!(worker.worker_type(), WorkerType::Regular)
                    && !matches!(worker.metadata().runtime_type, RuntimeType::External)
            })
            .filter_map(|worker| Self::matching_context_limit(worker.as_ref(), context_tokens))
            .min();

        // A PD Prefill worker without `preferred_context_tokens` has no
        // preferred upper bound. This makes the usual configuration
        // `regular=10, pd=None` mean: Regular up to 10 tokens, PD above 10.
        let pd_limit = Self::matching_pd_context_limit(workers, context_tokens);

        match (regular_limit, pd_limit) {
            (Some(regular), Some(pd)) if regular <= pd => Some(WorkerMode::Regular),
            (Some(_), Some(_)) => Some(WorkerMode::PrefillDecode),
            (Some(_), None) => Some(WorkerMode::Regular),
            (None, Some(_)) => Some(WorkerMode::PrefillDecode),
            (None, None) => None,
        }
    }

    fn matching_context_limit(worker: &dyn Worker, context_tokens: usize) -> Option<usize> {
        worker
            .metadata()
            .preferred_context_tokens
            .filter(|limit| context_tokens <= *limit)
    }

    /// Check whether a PD Prefill worker is eligible for the request's
    /// context length. `None` means that the worker has no preferred upper
    /// bound and is therefore eligible for any context length.
    fn matches_pd_context_preference(worker: &dyn Worker, context_tokens: usize) -> bool {
        worker
            .metadata()
            .preferred_context_tokens
            .is_none_or(|limit| context_tokens <= limit)
    }

    fn matching_pd_context_limit(
        workers: &[Arc<dyn Worker>],
        context_tokens: usize,
    ) -> Option<usize> {
        let prefill_workers: Vec<_> = workers
            .iter()
            .filter(|worker| {
                !matches!(worker.metadata().runtime_type, RuntimeType::External)
                    && matches!(worker.worker_type(), WorkerType::Prefill { .. })
                    && Self::matches_pd_context_preference(worker.as_ref(), context_tokens)
            })
            .cloned()
            .collect();
        let decode_workers: Vec<_> = workers
            .iter()
            .filter(|worker| {
                !matches!(worker.metadata().runtime_type, RuntimeType::External)
                    && matches!(worker.worker_type(), WorkerType::Decode)
            })
            .cloned()
            .collect();

        Self::select_pd_group(&prefill_workers, &decode_workers)?;

        Some(
            prefill_workers
                .iter()
                .map(|worker| {
                    worker
                        .metadata()
                        .preferred_context_tokens
                        .unwrap_or(usize::MAX)
                })
                .max()
                .unwrap_or(usize::MAX),
        )
    }

    /// Count worker categories and choose the routing mode.
    ///
    /// Priority is intentionally fixed for now:
    ///
    /// 1. Regular, if at least one Regular worker exists.
    /// 2. External, if no Regular worker exists and at least one External
    ///    worker exists.
    /// 3. PrefillDecode, if no Regular/External worker exists and both
    ///    Prefill and Decode workers exist.
    ///
    /// This function only chooses a mode. It does not choose a concrete
    /// worker, validate a PD group, or invoke a load-balancing policy.
    fn select_by_mode(workers: &[Arc<dyn Worker>]) -> Option<WorkerMode> {
        let mut num_regular = 0;
        let mut num_prefill = 0;
        let mut num_decode = 0;

        for worker in workers {
            match worker.worker_type() {
                WorkerType::Regular => num_regular += 1,
                WorkerType::Prefill { .. } => num_prefill += 1,
                WorkerType::Decode => num_decode += 1,
            }
        }

        if num_regular > 0 {
            Some(WorkerMode::Regular)
        } else if num_prefill > 0 && num_decode > 0 {
            Some(WorkerMode::PrefillDecode)
        } else {
            None
        }
    }

    /// Resolve an explicit model or infer the only registered model.
    /// A successful schedule therefore always carries a concrete model ID.
    fn resolve_model_id(&self, model_id: Option<&str>) -> Result<String, SchedulerError> {
        if let Some(model_id) = model_id
            .map(str::trim)
            .filter(|model| !model.is_empty() && *model != crate::core::UNKNOWN_MODEL_ID)
        {
            return Ok(model_id.to_string());
        }

        let models: Vec<_> = self
            .worker_registry
            .get_models()
            .into_iter()
            .filter(|model| model != crate::core::UNKNOWN_MODEL_ID)
            .collect();
        match models.len() {
            0 if self
                .worker_registry
                .get_all()
                .iter()
                .any(|worker| worker.is_healthy()) =>
            {
                Ok(crate::core::UNKNOWN_MODEL_ID.to_string())
            }
            0 => Err(SchedulerError::NoModelsAvailable),
            1 => Ok(models[0].clone()),
            _ => Err(SchedulerError::ModelRequired {
                models: models.join(", "),
            }),
        }
    }

    /// Build one candidate snapshot before choosing a routing mode.
    /// Worker health is owned by HealthCheck. Request failures and retry
    /// attempts must not change the scheduler's candidate set.
    fn available_workers(
        &self,
        model_id: &str,
        endpoint: Endpoint,
    ) -> Result<Vec<Arc<dyn Worker>>, SchedulerError> {
        let model_was_provided = model_id != crate::core::UNKNOWN_MODEL_ID;

        let workers: Vec<_> = self
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|worker| {
                // Workers registered before model discovery (or test/mock
                // workers) carry UNKNOWN_MODEL_ID. They are intentionally
                // wildcard workers until a concrete model card is discovered.
                // Do not reject them merely because the request contains a
                // model ID or because their provisional model type lacks an
                // endpoint capability.
                let unresolved_worker = worker.model_id() == crate::core::UNKNOWN_MODEL_ID;
                (!model_was_provided || unresolved_worker || worker.supports_model(model_id))
                    && (unresolved_worker || worker.supports_endpoint(model_id, endpoint))
                    && worker.is_healthy()
            })
            .collect();

        if !workers.is_empty() || !model_was_provided {
            return Ok(workers);
        }

        let model_is_known = self.worker_registry.get_all().iter().any(|worker| {
            worker.model_id() == crate::core::UNKNOWN_MODEL_ID || worker.supports_model(model_id)
        });
        if model_is_known {
            Ok(workers)
        } else {
            Err(SchedulerError::ModelNotFound {
                model_id: model_id.to_string(),
                endpoint,
            })
        }
    }

    async fn select_regular(
        &self,
        available_workers: &[Arc<dyn Worker>],
        model_id: &str,
        req: &SchedulerInput<'_>,
    ) -> Option<WorkerSelection> {
        let workers: Vec<_> = available_workers
            .iter()
            .filter(|worker| {
                matches!(worker.worker_type(), WorkerType::Regular)
                    && !matches!(worker.metadata().runtime_type, RuntimeType::External)
            })
            .cloned()
            .collect();

        let worker = self
            .select_by_policy(
                &workers,
                self.policy_registry.get_policy_or_default(model_id),
                model_id,
                req,
            )
            .await?;

        Some(WorkerSelection::Regular { worker })
    }

    async fn select_pd(
        &self,
        available_workers: &[Arc<dyn Worker>],
        model_id: &str,
        context_tokens: Option<usize>,
        req: &SchedulerInput<'_>,
    ) -> Option<WorkerSelection> {
        let available: Vec<_> = available_workers
            .iter()
            .filter(|worker| !matches!(worker.metadata().runtime_type, RuntimeType::External))
            .cloned()
            .collect();

        let prefill_workers: Vec<_> = available
            .iter()
            .filter(|worker| matches!(worker.worker_type(), WorkerType::Prefill { .. }))
            .cloned()
            .collect();
        let decode_workers: Vec<_> = available
            .iter()
            .filter(|worker| matches!(worker.worker_type(), WorkerType::Decode))
            .cloned()
            .collect();

        let preferred_prefill_workers: Vec<_> = context_tokens
            .map(|tokens| {
                prefill_workers
                    .iter()
                    .filter(|worker| Self::matches_pd_context_preference(worker.as_ref(), tokens))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let prefill_candidates = if !preferred_prefill_workers.is_empty()
            && Self::select_pd_group(&preferred_prefill_workers, &decode_workers).is_some()
        {
            preferred_prefill_workers
        } else {
            prefill_workers
        };

        let (group_id, transport, runtime_type) =
            Self::select_pd_group(&prefill_candidates, &decode_workers)?;
        let prefill_workers: Vec<_> = prefill_candidates
            .into_iter()
            .filter(|worker| {
                pd_group_id(worker.as_ref()) == group_id
                    && Self::same_transport(worker.connection_mode(), &transport)
                    && worker.metadata().runtime_type == runtime_type
            })
            .collect();
        let decode_workers: Vec<_> = decode_workers
            .into_iter()
            .filter(|worker| {
                pd_group_id(worker.as_ref()) == group_id
                    && Self::same_transport(worker.connection_mode(), &transport)
                    && worker.metadata().runtime_type == runtime_type
            })
            .collect();

        let prefill = self
            .select_by_policy(
                &prefill_workers,
                self.policy_registry.get_prefill_policy(),
                model_id,
                req,
            )
            .await?;
        let decode = self
            .select_by_policy(
                &decode_workers,
                self.policy_registry.get_decode_policy(),
                model_id,
                req,
            )
            .await?;

        if !Self::same_transport(prefill.connection_mode(), decode.connection_mode()) {
            return None;
        }
        debug_assert_eq!(
            prefill.metadata().runtime_type,
            decode.metadata().runtime_type
        );

        Some(WorkerSelection::PrefillDecode {
            prefill,
            decode,
            pd_group_id: group_id,
        })
    }

    fn select_pd_group(
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) -> Option<(String, ConnectionMode, RuntimeType)> {
        let mut seen = HashSet::new();
        let candidates: Vec<(String, ConnectionMode, RuntimeType, usize)> = prefill_workers
            .iter()
            .filter_map(|prefill| {
                let group = pd_group_id(prefill.as_ref()).to_string();
                let transport = prefill.connection_mode().clone();
                let runtime_type = prefill.metadata().runtime_type.clone();
                if !seen.insert((group.clone(), transport.clone(), runtime_type.clone())) {
                    return None;
                }
                let has_matching_decode = decode_workers.iter().any(|decode| {
                    pd_group_id(decode.as_ref()) == group
                        && Self::same_transport(decode.connection_mode(), &transport)
                        && decode.metadata().runtime_type == runtime_type
                });
                if !has_matching_decode {
                    return None;
                }

                let load: usize = prefill_workers
                    .iter()
                    .chain(decode_workers)
                    .filter(|worker| {
                        pd_group_id(worker.as_ref()) == group
                            && Self::same_transport(worker.connection_mode(), &transport)
                            && worker.metadata().runtime_type == runtime_type
                    })
                    .map(|worker| worker.load())
                    .sum();
                Some((group, transport, runtime_type, load))
            })
            .collect();

        candidates
            .into_iter()
            .min_by_key(|(group, _transport, _runtime_type, load)| (*load, group.clone()))
            .map(|(group, transport, runtime_type, _)| (group, transport, runtime_type))
    }

    async fn select_by_policy(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: Arc<dyn LoadBalancingPolicy>,
        model_id: &str,
        req: &SchedulerInput<'_>,
    ) -> Option<Arc<dyn Worker>> {
        if workers.is_empty() {
            return None;
        }

        let hash_ring = self.worker_registry.get_hash_ring(model_id);
        let index = policy
            .select_worker(
                workers,
                &SelectWorkerInfo {
                    request_text: req.request_text,
                    tokens: req.token_ids,
                    headers: req.headers,
                    hash_ring,
                },
            )
            .await?;

        workers.get(index).cloned()
    }

    fn same_transport(left: &ConnectionMode, right: &ConnectionMode) -> bool {
        left == right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::PolicyConfig,
        core::{model_card::ModelCard, BasicWorkerBuilder},
    };

    const MODEL_ID: &str = "deepseek-v4-flash";

    fn scheduler_with_workers(workers: Vec<Arc<dyn Worker>>) -> Scheduler {
        let worker_registry = Arc::new(WorkerRegistry::new());
        for worker in workers {
            worker_registry.register(worker);
        }

        Scheduler::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        )
    }

    fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .build(),
        )
    }

    fn external_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .runtime_type(RuntimeType::External)
                .build(),
        )
    }

    fn pd_worker(url: &str, worker_type: WorkerType, group: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .label("pd_group_id", group)
                .build(),
        )
    }

    fn pd_worker_with_runtime(
        url: &str,
        worker_type: WorkerType,
        group: &str,
        runtime_type: RuntimeType,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .runtime_type(runtime_type)
                .label("pd_group_id", group)
                .build(),
        )
    }

    fn request(endpoint: Endpoint) -> SchedulerInput<'static> {
        SchedulerInput {
            model_id: Some(MODEL_ID),
            endpoint,
            request_text: None,
            token_ids: None,
            headers: None,
        }
    }

    fn request_with_model(endpoint: Endpoint, model_id: &'static str) -> SchedulerInput<'static> {
        SchedulerInput {
            model_id: Some(model_id),
            endpoint,
            request_text: None,
            token_ids: None,
            headers: None,
        }
    }

    #[test]
    fn select_by_mode_uses_regular_before_external_and_pd() {
        let workers = vec![
            worker("http://regular", WorkerType::Regular),
            external_worker("http://external"),
            pd_worker(
                "http://prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "group-a",
            ),
            pd_worker("http://decode", WorkerType::Decode, "group-a"),
        ];

        assert_eq!(
            Scheduler::select_by_mode(&workers),
            Some(WorkerMode::Regular)
        );
    }

    #[tokio::test]
    async fn schedule_returns_regular_worker_when_available() {
        let scheduler = scheduler_with_workers(vec![
            worker("http://regular", WorkerType::Regular),
            pd_worker(
                "http://prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "group-a",
            ),
            pd_worker("http://decode", WorkerType::Decode, "group-a"),
        ]);

        let selection = scheduler.schedule(&request(Endpoint::Chat)).await.unwrap();
        assert!(matches!(
            selection.worker_selection,
            WorkerSelection::Regular { .. }
        ));
    }

    #[tokio::test]
    async fn schedule_rejects_unmatched_model_in_single_model_deployment() {
        let scheduler = scheduler_with_workers(vec![Arc::new(
            BasicWorkerBuilder::new("http://regular")
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new(MODEL_ID))
                .build(),
        ) as Arc<dyn Worker>]);

        let error = scheduler
            .schedule(&request_with_model(Endpoint::Chat, "invalid-model"))
            .await
            .unwrap_err();
        assert!(matches!(error, SchedulerError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn schedule_accepts_configured_model_alias() {
        let model_card = ModelCard::new(MODEL_ID).with_alias("model-alias");
        let scheduler = scheduler_with_workers(vec![Arc::new(
            BasicWorkerBuilder::new("http://regular")
                .worker_type(WorkerType::Regular)
                .model(model_card)
                .build(),
        ) as Arc<dyn Worker>]);

        let selection = scheduler
            .schedule(&request_with_model(Endpoint::Chat, "model-alias"))
            .await
            .unwrap();
        assert!(matches!(
            selection.worker_selection,
            WorkerSelection::Regular { .. }
        ));
    }

    #[tokio::test]
    async fn schedule_returns_pd_pair_when_regular_and_external_are_absent() {
        let scheduler = scheduler_with_workers(vec![
            pd_worker(
                "http://prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "group-a",
            ),
            pd_worker("http://decode", WorkerType::Decode, "group-a"),
        ]);

        let selection = scheduler.schedule(&request(Endpoint::Chat)).await.unwrap();
        match selection.worker_selection {
            WorkerSelection::PrefillDecode {
                prefill,
                decode,
                pd_group_id,
            } => {
                assert_eq!(prefill.url(), "http://prefill");
                assert_eq!(decode.url(), "http://decode");
                assert_eq!(pd_group_id, "group-a");
            }
            other => panic!("expected PD selection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn schedule_rejects_pd_workers_without_a_matching_pair() {
        let scheduler = scheduler_with_workers(vec![
            pd_worker(
                "http://prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "group-a",
            ),
            pd_worker("http://decode", WorkerType::Decode, "group-b"),
        ]);

        let error = scheduler
            .schedule(&request(Endpoint::Chat))
            .await
            .unwrap_err();
        assert!(matches!(error, SchedulerError::NoAvailablePdPair { .. }));
    }

    #[tokio::test]
    async fn schedule_selects_a_pd_pair_with_matching_runtime() {
        let scheduler = scheduler_with_workers(vec![
            pd_worker_with_runtime(
                "http://sglang-prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "group-a",
                RuntimeType::Sglang,
            ),
            pd_worker_with_runtime(
                "http://vllm-prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "group-a",
                RuntimeType::Vllm,
            ),
            pd_worker_with_runtime(
                "http://sglang-decode",
                WorkerType::Decode,
                "group-a",
                RuntimeType::Sglang,
            ),
            pd_worker_with_runtime(
                "http://vllm-decode",
                WorkerType::Decode,
                "group-a",
                RuntimeType::Vllm,
            ),
        ]);

        let selection = scheduler.schedule(&request(Endpoint::Chat)).await.unwrap();
        match selection.worker_selection {
            WorkerSelection::PrefillDecode {
                prefill, decode, ..
            } => {
                assert_eq!(
                    prefill.metadata().runtime_type,
                    decode.metadata().runtime_type
                );
            }
            other => panic!("expected PD selection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn default_sglang_worker_supports_generate() {
        let scheduler = scheduler_with_workers(vec![worker("http://regular", WorkerType::Regular)]);

        let selection = scheduler
            .schedule(&request(Endpoint::Generate))
            .await
            .unwrap();
        assert!(matches!(
            selection.worker_selection,
            WorkerSelection::Regular { .. }
        ));
    }
}
