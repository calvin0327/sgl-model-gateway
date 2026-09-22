//! Latency estimates and poll-gap reservations; no speculative cache insertion.

use super::config::CostModel;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{Arc, Weak},
    time::Instant,
};

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub running: f64,
    pub waiting: f64,
    pub kv_usage: f64,
    pub observed: Instant,
}

#[derive(Debug)]
pub struct Credit {
    pub tokens: usize,
    pub created: Instant,
}

#[derive(Debug, Default)]
pub struct LoadState {
    binding: Option<Arc<dyn crate::core::Worker>>,
    pub snapshot: Option<Snapshot>,
    pub credits: HashMap<u64, Credit>,
}

impl LoadState {
    /// A new registration at the same URL must not inherit the old load sample.
    pub fn bind(&mut self, worker: &Arc<dyn crate::core::Worker>) {
        if self
            .binding
            .as_ref()
            .is_none_or(|w| !Arc::ptr_eq(w, worker))
        {
            self.snapshot = None;
            self.credits.clear();
            self.binding = Some(worker.clone());
        }
    }
}

pub type Loads = Arc<Mutex<HashMap<String, LoadState>>>;

#[derive(Debug)]
pub struct Reservation {
    loads: Weak<Mutex<HashMap<String, LoadState>>>,
    worker: String,
    id: u64,
    committed: bool,
}

impl Reservation {
    pub fn new(loads: &Loads, worker: String, id: u64) -> Self {
        Self {
            loads: Arc::downgrade(loads),
            worker,
            id,
            committed: false,
        }
    }
    pub fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.committed {
            if let Some(loads) = self.loads.upgrade() {
                if let Some(load) = loads.lock().get_mut(&self.worker) {
                    load.credits.remove(&self.id);
                }
            }
        }
    }
}

impl CostModel {
    pub fn prefill_ms(&self, total: usize, cached: usize, speed: f64) -> f64 {
        let remaining = total.saturating_sub(cached) as f64;
        (remaining * self.prefill_ms_per_token
            + remaining * total as f64 / 1_000_000.0 * self.context_ms_per_million_token_pairs)
            / speed
    }

    pub fn cost_ms(
        &self,
        tokens: usize,
        cached: usize,
        speed: f64,
        snapshot: Option<&Snapshot>,
        local_load: usize,
        pending_tokens: usize,
        unknown_baseline: f64,
    ) -> f64 {
        let load = match snapshot {
            Some(s) => {
                let k = s.kv_usage.clamp(0.0, 0.99);
                s.waiting * self.waiting_request_ms
                    + s.running * self.running_request_ms
                    + self.kv_pressure_ms * k / (1.0 - k)
            }
            None => unknown_baseline + local_load as f64 * self.waiting_request_ms,
        };
        load + self.prefill_ms(tokens, cached, speed)
            + pending_tokens as f64 * self.prefill_ms_per_token / speed
    }
}

/// Only plain gauges from this endpoint are consumed. Missing or malformed
/// values remain unknown; GPU utilization is not a prefill-throughput estimate.
pub fn parse_metrics(body: &str) -> Option<Snapshot> {
    fn gauge(body: &str, names: &[&str]) -> Option<f64> {
        for name in names {
            let mut total = 0.0;
            let mut found = false;
            for line in body.lines() {
                let Some(rest) = line.strip_prefix(name) else {
                    continue;
                };
                if !rest.starts_with('{') && !rest.starts_with(char::is_whitespace) {
                    continue;
                }
                // Label values may contain spaces; the sample follows the final brace.
                let sample = if rest.starts_with('{') {
                    rest.rsplit_once('}')?.1
                } else {
                    rest
                };
                let value = sample.split_whitespace().next()?.parse::<f64>().ok()?;
                if !value.is_finite() || value < 0.0 {
                    return None;
                }
                total += value;
                found = true;
            }
            if found {
                return Some(total);
            }
        }
        None
    }
    Some(Snapshot {
        running: gauge(body, &["vllm:num_requests_running"])?,
        waiting: gauge(body, &["vllm:num_requests_waiting"])?,
        kv_usage: gauge(
            body,
            &["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        )
        .unwrap_or(0.0),
        observed: Instant::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_cost_can_outweigh_cache_savings() {
        let c = CostModel::default();
        let hot = Snapshot {
            running: 1.0,
            waiting: 10.0,
            kv_usage: 0.2,
            observed: Instant::now(),
        };
        let idle = Snapshot {
            running: 0.0,
            waiting: 0.0,
            kv_usage: 0.2,
            observed: Instant::now(),
        };
        let a = c.cost_ms(16000, 14000, 1.0, Some(&hot), 0, 0, 0.0);
        let b = c.cost_ms(16000, 8000, 1.0, Some(&idle), 0, 0, 0.0);
        assert!(b < a);
        assert!(c.cost_ms(16000, 14000, 1.0, Some(&idle), 0, 0, 0.0) < b);
        assert!(
            c.cost_ms(16000, 0, 2.0, Some(&idle), 0, 0, 0.0)
                < c.cost_ms(16000, 0, 1.0, Some(&idle), 0, 0, 0.0)
        );
    }

    #[test]
    fn failed_dispatch_releases_only_its_own_credit() {
        let loads = Loads::default();
        loads.lock().entry("w".into()).or_default().credits.insert(
            1,
            Credit {
                tokens: 10,
                created: Instant::now(),
            },
        );
        let guard = Reservation::new(&loads, "w".into(), 1);
        drop(guard);
        assert!(loads.lock()["w"].credits.is_empty());
    }

    #[test]
    fn missing_and_negative_metrics_are_not_idle() {
        assert!(parse_metrics("vllm:num_requests_running 1\n").is_none());
        assert!(
            parse_metrics("vllm:num_requests_running -1\nvllm:num_requests_waiting 0").is_none()
        );
        let s = parse_metrics("vllm:num_requests_running{model_name=\"m\"} 2\nvllm:num_requests_waiting 3\nvllm:kv_cache_usage_perc 0.5").unwrap();
        assert_eq!((s.running, s.waiting, s.kv_usage), (2.0, 3.0, 0.5));
    }
}
