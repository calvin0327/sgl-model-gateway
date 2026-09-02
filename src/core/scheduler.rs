//! Request scheduling based on worker capabilities and load-balancing policies.
//!
//! The scheduler selects a concrete worker, or a complete Prefill/Decode pair.
//! Routers are intentionally not involved here: they only execute the selection
//! returned by this module.

use std::{collections::BTreeSet, sync::Arc};

use http::HeaderMap;
use thiserror::Error;

use crate::{
    config::RoutingConfig,
    core::{
        model_type::Endpoint, pd_group_id, ConnectionMode, RuntimeType, Worker, WorkerRegistry,
        WorkerType,
    },
    policies::{LoadBalancingPolicy, PolicyFactory, PolicyRegistry, SelectWorkerInfo},
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
    External,
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
    External {
        worker: Arc<dyn Worker>,
    },
}

impl WorkerSelection {
    /// Returns the transport used by a selection.
    pub fn connection_mode(&self) -> ConnectionMode {
        match self {
            Self::Regular { worker } | Self::External { worker } => {
                worker.connection_mode().clone()
            }
            Self::PrefillDecode { prefill, .. } => prefill.connection_mode().clone(),
        }
    }
}

/// Selects workers and owns all request-routing policies.
pub struct Scheduler {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: PolicyRegistry,
}

impl Scheduler {
    pub fn new(worker_registry: Arc<WorkerRegistry>, config: &RoutingConfig) -> Self {
        let policy_registry = PolicyRegistry::new(config.policy.clone());

        let prefill_config = config.prefill_policy.as_ref().unwrap_or(&config.policy);
        let decode_config = config.decode_policy.as_ref().unwrap_or(&config.policy);

        policy_registry.set_prefill_policy(PolicyFactory::create_from_config(prefill_config));
        policy_registry.set_decode_policy(PolicyFactory::create_from_config(decode_config));

        Self {
            worker_registry,
            policy_registry,
        }
    }

    /// Schedule with the first-version priority: Regular -> External -> PD.
    /// Context-length-based mode selection is intentionally not part of this
    /// version.
    pub async fn schedule(
        &self,
        req: &SchedulerInput<'_>,
    ) -> Result<SchedulerOutput, SchedulerError> {
        let model_id = self.resolve_model_id(req.model_id)?;
        let available_workers = self.available_workers(&model_id, req.endpoint);

        let mode = Self::select_by_mode(&available_workers).ok_or_else(|| {
            SchedulerError::NoAvailableWorker {
                model_id: model_id.clone(),
                mode: "regular, external, or PD",
                endpoint: req.endpoint,
            }
        })?;

        let selection = match mode {
            WorkerMode::Regular => {
                self.select_regular(&available_workers, &model_id, req)
                    .await
            }
            WorkerMode::External => self.select_external(&available_workers, &model_id),
            WorkerMode::PrefillDecode => self.select_pd(&available_workers, &model_id, req).await,
        };

        selection
            .map(|worker_selection| SchedulerOutput {
                model_id: model_id.clone(),
                worker_selection,
            })
            .ok_or_else(|| SchedulerError::NoAvailableWorker {
                model_id: model_id.clone(),
                mode: "selected worker mode",
                endpoint: req.endpoint,
            })
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
        let mut num_external = 0;

        for worker in workers {
            if matches!(worker.metadata().runtime_type, RuntimeType::External) {
                num_external += 1;
                continue;
            }

            match worker.worker_type() {
                WorkerType::Regular => num_regular += 1,
                WorkerType::Prefill { .. } => num_prefill += 1,
                WorkerType::Decode => num_decode += 1,
            }
        }

        if num_regular > 0 {
            Some(WorkerMode::Regular)
        } else if num_external > 0 {
            Some(WorkerMode::External)
        } else if num_prefill > 0 && num_decode > 0 {
            Some(WorkerMode::PrefillDecode)
        } else {
            None
        }
    }

    /// Resolve an explicit model or infer the only registered model.
    /// A successful schedule therefore always carries a concrete model ID.
    fn resolve_model_id(&self, model_id: Option<&str>) -> Result<String, SchedulerError> {
        if let Some(model_id) = model_id.map(str::trim).filter(|model| !model.is_empty()) {
            return Ok(model_id.to_string());
        }

        let models = self.worker_registry.get_models();
        match models.len() {
            0 => Err(SchedulerError::NoModelsAvailable),
            1 => Ok(models[0].clone()),
            _ => Err(SchedulerError::ModelRequired {
                models: models.join(", "),
            }),
        }
    }

    /// Build one candidate snapshot before choosing a routing mode.
    /// `is_available()` includes health and circuit-breaker state.
    fn available_workers(&self, model_id: &str, endpoint: Endpoint) -> Vec<Arc<dyn Worker>> {
        let model_was_provided = model_id != crate::core::UNKNOWN_MODEL_ID;

        self.worker_registry
            .get_all()
            .into_iter()
            .filter(|worker| {
                (!model_was_provided || worker.supports_model(model_id))
                    && worker.supports_endpoint(model_id, endpoint)
                    && worker.is_available()
            })
            .collect()
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

    fn select_external(
        &self,
        available_workers: &[Arc<dyn Worker>],
        model_id: &str,
    ) -> Option<WorkerSelection> {
        // External workers can discover models dynamically, so select from the
        // complete External pool rather than relying only on the model index.
        let worker = available_workers
            .iter()
            .filter(|worker| {
                matches!(worker.metadata().runtime_type, RuntimeType::External)
                    && worker.supports_model(model_id)
            })
            .min_by_key(|worker| worker.load())
            .cloned();

        worker.map(|worker| WorkerSelection::External { worker })
    }

    async fn select_pd(
        &self,
        available_workers: &[Arc<dyn Worker>],
        model_id: &str,
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

        let (group_id, transport) = Self::select_pd_group(&prefill_workers, &decode_workers)?;
        let prefill_workers: Vec<_> = prefill_workers
            .into_iter()
            .filter(|worker| {
                pd_group_id(worker.as_ref()) == group_id
                    && Self::same_transport(worker.connection_mode(), &transport)
            })
            .collect();
        let decode_workers: Vec<_> = decode_workers
            .into_iter()
            .filter(|worker| {
                pd_group_id(worker.as_ref()) == group_id
                    && Self::same_transport(worker.connection_mode(), &transport)
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

        Some(WorkerSelection::PrefillDecode {
            prefill,
            decode,
            pd_group_id: group_id,
        })
    }

    fn select_pd_group(
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) -> Option<(String, ConnectionMode)> {
        let prefill_groups: BTreeSet<String> = prefill_workers
            .iter()
            .map(|worker| pd_group_id(worker.as_ref()).to_string())
            .collect();
        let decode_groups: BTreeSet<String> = decode_workers
            .iter()
            .map(|worker| pd_group_id(worker.as_ref()).to_string())
            .collect();

        let candidates: Vec<(String, ConnectionMode, usize)> = prefill_groups
            .intersection(&decode_groups)
            .flat_map(|group| {
                let transports: Vec<ConnectionMode> = prefill_workers
                    .iter()
                    .filter(|worker| pd_group_id(worker.as_ref()) == group.as_str())
                    .map(|worker| worker.connection_mode().clone())
                    .filter(|transport| {
                        decode_workers.iter().any(|worker| {
                            pd_group_id(worker.as_ref()) == group.as_str()
                                && Self::same_transport(worker.connection_mode(), transport)
                        })
                    })
                    .collect();

                transports.into_iter().map(|transport| {
                    let prefill_load: usize = prefill_workers
                        .iter()
                        .filter(|worker| {
                            pd_group_id(worker.as_ref()) == group.as_str()
                                && Self::same_transport(worker.connection_mode(), &transport)
                        })
                        .map(|worker| worker.load())
                        .sum();
                    let decode_load: usize = decode_workers
                        .iter()
                        .filter(|worker| {
                            pd_group_id(worker.as_ref()) == group.as_str()
                                && Self::same_transport(worker.connection_mode(), &transport)
                        })
                        .map(|worker| worker.load())
                        .sum();
                    (group.clone(), transport, prefill_load + decode_load)
                })
            })
            .collect();

        candidates
            .into_iter()
            .min_by_key(|(group, _transport, load)| (*load, group.clone()))
            .map(|(group, transport, _)| (group, transport))
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
        matches!(
            (left, right),
            (ConnectionMode::Http, ConnectionMode::Http)
                | (ConnectionMode::Grpc { .. }, ConnectionMode::Grpc { .. })
        )
    }
}
