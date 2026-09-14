use super::{
    gateway::{
        CircuitBreakerConfig, DiscoveryConfig, GatewayConfig, ObservabilityConfig, PolicyConfig,
        RetryConfig, RoutingConfig, ServerConfig, TokenizerCacheConfig, WorkerConfig,
    },
    ConfigError, ConfigResult,
};

/// Validates resolved configuration before gateway startup.
pub(crate) struct ConfigValidator;

impl ConfigValidator {
    /// Validates each configuration section and their compatibility rules.
    pub(crate) fn validate_gateway(config: &GatewayConfig) -> ConfigResult<()> {
        Self::validate_server(&config.server)?;
        Self::validate_routing(&config.routing)?;
        Self::validate_worker(&config.workers)?;
        Self::validate_tokenizer_cache(&config.model.tokenizer_cache)?;
        Self::validate_observability(&config.observability)?;

        if let Some(discovery) = &config.discovery {
            Self::validate_gateway_discovery(discovery)?;
        }

        Ok(())
    }

    fn validate_server(server: &ServerConfig) -> ConfigResult<()> {
        if server.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.port".to_string(),
                value: server.port.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if server.max_payload_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.max_payload_size".to_string(),
                value: server.max_payload_size.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if server.queue_size > 0 && server.queue_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "routing.queue_timeout_secs".to_string(),
                value: server.queue_timeout_secs.to_string(),
                reason: "Must be > 0 when queue_size > 0".to_string(),
            });
        }
        if let Some(tokens_per_second) = server.rate_limit_tokens_per_second {
            if tokens_per_second < 0 {
                return Err(ConfigError::InvalidValue {
                    field: "routing.rate_limit_tokens_per_second".to_string(),
                    value: tokens_per_second.to_string(),
                    reason: "Must be >= 0 when specified".to_string(),
                });
            }
        }

        if server.client_identity.as_ref().is_some_and(Vec::is_empty) {
            return Err(ConfigError::ValidationFailed {
                reason: "Client identity cannot be empty".to_string(),
            });
        }
        if server.ca_certificates.iter().any(Vec::is_empty) {
            return Err(ConfigError::ValidationFailed {
                reason: "CA certificates cannot be empty".to_string(),
            });
        }
        if let Some(tls) = &server.server_tls {
            if tls.certificate.is_empty() || tls.private_key.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Server TLS certificate and private key cannot be empty".to_string(),
                });
            }
        }
        Ok(())
    }

    fn validate_routing(routing: &RoutingConfig) -> ConfigResult<()> {
        Self::validate_policy(&routing.policy)?;
        if let Some(policy) = &routing.prefill_policy {
            Self::validate_policy(policy)?;
        }
        if let Some(policy) = &routing.decode_policy {
            Self::validate_policy(policy)?;
        }
        Ok(())
    }

    fn validate_worker(workers: &WorkerConfig) -> ConfigResult<()> {
        if workers.backend.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "workers.backend".to_string(),
            });
        }

        if !matches!(workers.backend.as_str(), "sglang" | "vllm") {
            return Err(ConfigError::InvalidValue {
                field: "workers.backend".to_string(),
                value: workers.backend.clone(),
                reason: "Must be 'sglang' or 'vllm'".to_string(),
            });
        }

        if let Some(connector) = &workers.kv_connector {
            if !matches!(connector.as_str(), "mooncake" | "nixl") {
                return Err(ConfigError::InvalidValue {
                    field: "workers.kv_connector".to_string(),
                    value: connector.clone(),
                    reason: "Must be 'mooncake' or 'nixl'".to_string(),
                });
            }
        }

        if workers.kv_engine_id.as_ref().is_some_and(String::is_empty) {
            return Err(ConfigError::InvalidValue {
                field: "workers.kv_engine_id".to_string(),
                value: String::new(),
                reason: "Cannot be empty when specified".to_string(),
            });
        }

        for (field, value) in [
            (
                "workers.regular_preferred_context_tokens",
                workers.regular_preferred_context_tokens,
            ),
            (
                "workers.pd_preferred_context_tokens",
                workers.pd_preferred_context_tokens,
            ),
        ] {
            if value == Some(0) {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: "0".to_string(),
                    reason: "Must be > 0 when specified".to_string(),
                });
            }
        }

        Self::validate_urls(&workers.regular_urls)?;
        Self::validate_urls(&workers.openai_urls)?;
        Self::validate_urls(
            &workers
                .prefill_urls
                .iter()
                .map(|(url, _)| url.clone())
                .collect::<Vec<_>>(),
        )?;
        Self::validate_urls(&workers.decode_urls)?;
        for (_, bootstrap_port) in &workers.prefill_urls {
            if *bootstrap_port == Some(0) {
                return Err(ConfigError::InvalidValue {
                    field: "workers.prefill_urls.bootstrap_port".to_string(),
                    value: "0".to_string(),
                    reason: "Port must be between 1 and 65535".to_string(),
                });
            }
        }

        for (field, value) in [
            ("workers.request_timeout_secs", workers.request_timeout_secs),
            ("workers.startup_timeout_secs", workers.startup_timeout_secs),
            (
                "workers.startup_check_interval_secs",
                workers.startup_check_interval_secs,
            ),
            (
                "workers.pool_idle_timeout_secs",
                workers.pool_idle_timeout_secs,
            ),
            ("workers.connect_timeout_secs", workers.connect_timeout_secs),
            ("workers.tcp_keepalive_secs", workers.tcp_keepalive_secs),
            (
                "workers.health_check.timeout_secs",
                workers.health_check.timeout_secs,
            ),
            (
                "workers.health_check.check_interval_secs",
                workers.health_check.check_interval_secs,
            ),
        ] {
            if value == 0 {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: value.to_string(),
                    reason: "Must be > 0".to_string(),
                });
            }
        }
        if workers.pool_max_idle_per_host == 0 {
            return Err(ConfigError::InvalidValue {
                field: "workers.pool_max_idle_per_host".to_string(),
                value: "0".to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if workers.health_check.failure_threshold == 0
            || workers.health_check.success_threshold == 0
        {
            return Err(ConfigError::ValidationFailed {
                reason: "Health check thresholds must be > 0".to_string(),
            });
        }

        let mut retry = workers.retry.clone();
        if workers.disable_retries {
            retry.max_retries = 1;
        }
        Self::validate_retry(&retry)?;

        let mut circuit_breaker = workers.circuit_breaker.clone();
        if workers.disable_circuit_breaker {
            circuit_breaker.failure_threshold = u32::MAX;
        }
        Self::validate_circuit_breaker(&circuit_breaker)
    }

    fn validate_observability(observability: &ObservabilityConfig) -> ConfigResult<()> {
        if observability.prometheus_port == 0 || observability.prometheus_host.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "Prometheus host must be non-empty and port must be > 0".to_string(),
            });
        }
        if !observability.enable_trace {
            return Ok(());
        }
        Self::validate_otlp_endpoint(
            &observability.otlp_traces_endpoint,
            "observability.otlp_traces_endpoint",
        )
    }

    fn validate_gateway_discovery(discovery: &DiscoveryConfig) -> ConfigResult<()> {
        if discovery.port == 0 || discovery.check_interval_secs == 0 {
            return Err(ConfigError::ValidationFailed {
                reason: "Discovery port and check interval must be > 0".to_string(),
            });
        }

        Ok(())
    }

    fn validate_otlp_endpoint(endpoint: &str, field: &str) -> ConfigResult<()> {
        let Some((host, port)) = endpoint.rsplit_once(':') else {
            return Err(ConfigError::InvalidValue {
                field: field.to_string(),
                value: endpoint.to_string(),
                reason: "expected format <host>:<port>".to_string(),
            });
        };
        if host.is_empty() || !matches!(port.parse::<u16>(), Ok(port) if port > 0) {
            return Err(ConfigError::InvalidValue {
                field: field.to_string(),
                value: endpoint.to_string(),
                reason: "host must be non-empty and port must be between 1 and 65535".to_string(),
            });
        }
        Ok(())
    }

    fn validate_policy(policy: &PolicyConfig) -> ConfigResult<()> {
        match policy {
            PolicyConfig::Random
            | PolicyConfig::RoundRobin
            | PolicyConfig::Manual { .. }
            | PolicyConfig::ConsistentHashing => {}
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold: _,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                if !(0.0..=1.0).contains(cache_threshold) {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_threshold".to_string(),
                        value: cache_threshold.to_string(),
                        reason: "Must be between 0.0 and 1.0".to_string(),
                    });
                }

                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *eviction_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "eviction_interval_secs".to_string(),
                        value: eviction_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *max_tree_size == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "max_tree_size".to_string(),
                        value: max_tree_size.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                if *load_check_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_check_interval_secs".to_string(),
                        value: load_check_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::Bucket {
                balance_abs_threshold: _,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *bucket_adjust_interval_secs < 1 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be >= 1s".to_string(),
                    });
                }
                if *bucket_adjust_interval_secs >= 4294967296 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be < 4294967296s".to_string(),
                    });
                }
            }
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                if *prefix_token_count == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "prefix_token_count".to_string(),
                        value: prefix_token_count.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *load_factor < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_factor".to_string(),
                        value: load_factor.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_retry(retry: &RetryConfig) -> ConfigResult<()> {
        if retry.max_retries < 1 {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_retries".to_string(),
                value: retry.max_retries.to_string(),
                reason: "Must be >= 1 (set to 1 to effectively disable retries)".to_string(),
            });
        }
        if retry.initial_backoff_ms == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.initial_backoff_ms".to_string(),
                value: retry.initial_backoff_ms.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if retry.max_backoff_ms < retry.initial_backoff_ms {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_backoff_ms".to_string(),
                value: retry.max_backoff_ms.to_string(),
                reason: "Must be >= initial_backoff_ms".to_string(),
            });
        }
        if retry.backoff_multiplier < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.backoff_multiplier".to_string(),
                value: retry.backoff_multiplier.to_string(),
                reason: "Must be >= 1.0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&retry.jitter_factor) {
            return Err(ConfigError::InvalidValue {
                field: "retry.jitter_factor".to_string(),
                value: retry.jitter_factor.to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_circuit_breaker(cb: &CircuitBreakerConfig) -> ConfigResult<()> {
        if cb.failure_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.failure_threshold".to_string(),
                value: cb.failure_threshold.to_string(),
                reason: "Must be >= 1 (set to u32::MAX to effectively disable CB)".to_string(),
            });
        }
        if cb.success_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.success_threshold".to_string(),
                value: cb.success_threshold.to_string(),
                reason: "Must be >= 1".to_string(),
            });
        }
        if cb.timeout_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.timeout_duration_secs".to_string(),
                value: cb.timeout_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if cb.window_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.window_duration_secs".to_string(),
                value: cb.window_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_tokenizer_cache(cache: &TokenizerCacheConfig) -> ConfigResult<()> {
        if cache.enable_l0 && cache.l0_max_entries == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l0_max_entries".to_string(),
                value: cache.l0_max_entries.to_string(),
                reason: "Must be > 0 when L0 cache is enabled".to_string(),
            });
        }

        if cache.enable_l1 && cache.l1_max_memory == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l1_max_memory".to_string(),
                value: cache.l1_max_memory.to_string(),
                reason: "Must be > 0 when L1 cache is enabled".to_string(),
            });
        }

        Ok(())
    }

    fn validate_urls(urls: &[String]) -> ConfigResult<()> {
        for url in urls {
            if url.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL cannot be empty".to_string(),
                });
            }

            if !url.starts_with("http://")
                && !url.starts_with("https://")
                && !url.starts_with("grpc://")
            {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL must start with http://, https://, or grpc://".to_string(),
                });
            }

            match ::url::Url::parse(url) {
                Ok(parsed) => {
                    if parsed.host_str().is_none() {
                        return Err(ConfigError::InvalidValue {
                            field: "worker_url".to_string(),
                            value: url.clone(),
                            reason: "URL must have a valid host".to_string(),
                        });
                    }
                }
                Err(e) => {
                    return Err(ConfigError::InvalidValue {
                        field: "worker_url".to_string(),
                        value: url.clone(),
                        reason: format!("Invalid URL format: {}", e),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_backend_is_required() {
        let error = ConfigValidator::validate_worker(&WorkerConfig::default()).unwrap_err();

        assert!(
            matches!(error, ConfigError::MissingRequired { field } if field == "workers.backend")
        );
    }

    #[test]
    fn worker_backend_rejects_external() {
        let mut workers = WorkerConfig::default();
        workers.backend = "external".to_string();

        let error = ConfigValidator::validate_worker(&workers).unwrap_err();

        assert!(
            matches!(error, ConfigError::InvalidValue { field, .. } if field == "workers.backend")
        );
    }

    #[test]
    fn worker_backend_accepts_supported_runtimes() {
        for backend in ["sglang", "vllm"] {
            let mut workers = WorkerConfig::default();
            workers.backend = backend.to_string();

            ConfigValidator::validate_worker(&workers).unwrap();
        }
    }
}
