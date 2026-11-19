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
    // Missing occupancy still contributes zero pressure, but is not logged as
    // a measured empty cache. Queue/running gauges remain usable on their own.
    pub kv_usage: Option<f64>,
    pub observed: Instant,
}

/// Keep raw estimates alongside weighted contributions so decisions are explainable.
#[derive(Debug, Clone, Copy)]
pub struct CostBreakdown {
    pub prefill_ms: f64,
    pub load_ms: f64,
    pub reservation_ms: f64,
    pub weighted_kv_ms: f64,
    pub weighted_load_ms: f64,
    pub total_ms: f64,
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

    /// Unknown workers inherit a raw backend baseline; apply weights only later.
    pub fn backend_load_ms(
        &self,
        snapshot: Option<&Snapshot>,
        local_load: usize,
        unknown_baseline: f64,
    ) -> f64 {
        match snapshot {
            Some(s) => {
                let k = s.kv_usage.unwrap_or(0.0).clamp(0.0, 0.99);
                s.waiting * self.waiting_request_ms
                    + s.running * self.running_request_ms
                    + self.kv_pressure_ms * k / (1.0 - k)
            }
            None => unknown_baseline + local_load as f64 * self.waiting_request_ms,
        }
    }

    /// Apply load weight to both observed work and dispatches missing from the poll.
    pub fn cost_breakdown(
        &self,
        tokens: usize,
        cached: usize,
        speed: f64,
        snapshot: Option<&Snapshot>,
        local_load: usize,
        pending_tokens: usize,
        unknown_baseline: f64,
    ) -> CostBreakdown {
        let prefill_ms = self.prefill_ms(tokens, cached, speed);
        let load_ms = self.backend_load_ms(snapshot, local_load, unknown_baseline);
        let reservation_ms = pending_tokens as f64 * self.prefill_ms_per_token / speed;
        let weighted_kv_ms = self.kv_importance * prefill_ms;
        let weighted_load_ms = self.load_importance * (load_ms + reservation_ms);
        CostBreakdown {
            prefill_ms,
            load_ms,
            reservation_ms,
            weighted_kv_ms,
            weighted_load_ms,
            total_ms: weighted_kv_ms + weighted_load_ms,
        }
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
        self.cost_breakdown(
            tokens,
            cached,
            speed,
            snapshot,
            local_load,
            pending_tokens,
            unknown_baseline,
        )
        .total_ms
    }
}

/// Only plain gauges from this endpoint are consumed. Missing or malformed
/// values remain unknown; GPU utilization is not a prefill-throughput estimate.
pub fn parse_metrics(body: &str) -> Option<Snapshot> {
    fn gauge(body: &str, names: &[&str], ratio: bool) -> Option<f64> {
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
                if !value.is_finite() || value < 0.0 || (ratio && value > 1.0) {
                    return None;
                }
                // Counts add across engines; occupancy is a ratio. Match
                // llm-d's maximum so a busy engine is not hidden by averaging.
                total = if ratio {
                    f64::max(total, value)
                } else {
                    total + value
                };
                if !total.is_finite() {
                    return None;
                }
                found = true;
            }
            if found {
                return Some(total);
            }
        }
        None
    }
    Some(Snapshot {
        running: gauge(body, &["vllm:num_requests_running"], false)?,
        waiting: gauge(body, &["vllm:num_requests_waiting"], false)?,
        kv_usage: gauge(
            body,
            &["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
            true,
        ),
        observed: Instant::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_preserve_the_previous_cost_formula() {
        let cost = CostModel::default();
        let snapshot = Snapshot {
            running: 2.0,
            waiting: 3.0,
            kv_usage: Some(0.5),
            observed: Instant::now(),
        };
        let score = cost.cost_breakdown(16000, 8000, 2.0, Some(&snapshot), 7, 2000, 90.0);
        let prefill = (8000.0 * 0.05 + 8000.0 * 16000.0 / 1_000_000.0 * 0.1) / 2.0;
        let backend_load = 3.0 * 100.0 + 2.0 * 10.0 + 10.0 * 0.5 / 0.5;
        let reservation = 2000.0 * 0.05 / 2.0;
        let old_cost = backend_load + prefill + reservation;
        assert!((score.total_ms - old_cost).abs() < 1e-9);
        assert_eq!(score.prefill_ms, prefill);
        assert_eq!(score.load_ms, backend_load);
        assert_eq!(score.reservation_ms, reservation);
        assert_eq!(score.weighted_kv_ms, prefill);
        assert_eq!(score.weighted_load_ms, backend_load + reservation);
    }

    #[test]
    fn weight_changes_can_reverse_cache_and_queue_preference() {
        let busy = Snapshot {
            running: 1.0,
            waiting: 10.0,
            kv_usage: Some(0.2),
            observed: Instant::now(),
        };
        let idle = Snapshot {
            running: 0.0,
            waiting: 0.0,
            ..busy.clone()
        };
        let prefers_cached_worker = |cost: &CostModel| {
            cost.cost_ms(16000, 14000, 1.0, Some(&busy), 0, 0, 0.0)
                < cost.cost_ms(16000, 8000, 1.0, Some(&idle), 0, 0, 0.0)
        };
        assert!(!prefers_cached_worker(&CostModel::default()));
        assert!(prefers_cached_worker(&CostModel {
            kv_importance: 4.0,
            ..Default::default()
        }));
        assert!(prefers_cached_worker(&CostModel {
            load_importance: 0.1,
            ..Default::default()
        }));
    }

    #[test]
    fn zero_weight_removes_only_its_score_component() {
        let cache_only = CostModel {
            load_importance: 0.0,
            ..Default::default()
        };
        let score = cache_only.cost_breakdown(1000, 800, 1.0, None, 10, 10000, 500.0);
        assert!(score.load_ms > 0.0 && score.reservation_ms > 0.0);
        assert_eq!(score.weighted_load_ms, 0.0);
        assert_eq!(score.total_ms, score.prefill_ms);

        let load_only = CostModel {
            kv_importance: 0.0,
            ..Default::default()
        };
        let score = load_only.cost_breakdown(1000, 800, 1.0, None, 10, 10000, 500.0);
        assert!(score.prefill_ms > 0.0);
        assert_eq!(score.weighted_kv_ms, 0.0);
        assert_eq!(score.total_ms, score.load_ms + score.reservation_ms);
    }

    #[test]
    fn missing_load_uses_an_unweighted_baseline_exactly_once() {
        let cost = CostModel {
            kv_importance: 0.0,
            load_importance: 2.0,
            ..Default::default()
        };
        let snapshot = Snapshot {
            running: 0.0,
            waiting: 1.0,
            kv_usage: Some(0.0),
            observed: Instant::now(),
        };
        let baseline = cost.backend_load_ms(Some(&snapshot), 0, 0.0);
        assert_eq!(baseline, 100.0);
        let score = cost.cost_breakdown(1000, 0, 1.0, None, 1, 1000, baseline);
        assert_eq!(score.load_ms, 200.0);
        assert_eq!(score.reservation_ms, 50.0);
        assert_eq!(score.total_ms, 500.0);
    }

    #[test]
    fn queue_cost_can_outweigh_cache_savings() {
        let c = CostModel::default();
        let hot = Snapshot {
            running: 1.0,
            waiting: 10.0,
            kv_usage: Some(0.2),
            observed: Instant::now(),
        };
        let idle = Snapshot {
            running: 0.0,
            waiting: 0.0,
            kv_usage: Some(0.2),
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
    fn engine_counts_sum_but_occupancy_uses_maximum() {
        let snapshot = parse_metrics(concat!(
            "vllm:num_requests_running{engine=\"0\"} 2\n",
            "vllm:num_requests_running{engine=\"1\"} 3\n",
            "vllm:num_requests_waiting{engine=\"0\"} 1\n",
            "vllm:num_requests_waiting{engine=\"1\"} 4\n",
            "vllm:kv_cache_usage_perc{engine=\"0\"} 0.6\n",
            "vllm:kv_cache_usage_perc{engine=\"1\"} 0.7\n",
            "vllm:gpu_cache_usage_perc 0.9\n",
        ))
        .unwrap();
        assert_eq!(
            (snapshot.running, snapshot.waiting, snapshot.kv_usage),
            (5.0, 5.0, Some(0.7))
        );
        for invalid in ["1.1", "-1", "NaN", "inf"] {
            let snapshot = parse_metrics(&format!("vllm:num_requests_running 1\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc {invalid}")).unwrap();
            assert_eq!(snapshot.kv_usage, None);
        }
        assert!(parse_metrics("vllm:num_requests_running{engine=\"0\"} 1e308\nvllm:num_requests_running{engine=\"1\"} 1e308\nvllm:num_requests_waiting 0").is_none());
    }

    #[test]
    fn missing_and_negative_metrics_are_not_idle() {
        assert!(parse_metrics("vllm:num_requests_running 1\n").is_none());
        assert!(
            parse_metrics("vllm:num_requests_running -1\nvllm:num_requests_waiting 0").is_none()
        );
        let s = parse_metrics("vllm:num_requests_running{model_name=\"m\"} 2\nvllm:num_requests_waiting 3\nvllm:kv_cache_usage_perc 0.5").unwrap();
        assert_eq!((s.running, s.waiting, s.kv_usage), (2.0, 3.0, Some(0.5)));
    }

    #[test]
    fn missing_occupancy_is_unknown_but_keeps_existing_queue_cost() {
        let partial =
            parse_metrics("vllm:num_requests_running 2\nvllm:num_requests_waiting 3").unwrap();
        let empty = parse_metrics(
            "vllm:num_requests_running 2\nvllm:num_requests_waiting 3\nvllm:kv_cache_usage_perc 0",
        )
        .unwrap();
        assert_eq!(partial.kv_usage, None);
        assert_eq!(empty.kv_usage, Some(0.0));
        let cost = CostModel::default();
        assert_eq!(cost.backend_load_ms(Some(&partial), 0, 0.0), 320.0);
        assert_eq!(
            cost.backend_load_ms(Some(&partial), 0, 0.0),
            cost.backend_load_ms(Some(&empty), 0, 0.0)
        );
    }
}
