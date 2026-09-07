/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced. Ties
    are randomly broken.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use rand::{seq::IteratorRandom, Rng};
use tracing::{debug, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    CacheAwareConfig, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{Worker, WorkerType};

/// Tag used to isolate prefill/decode/regular worker pools in the cache_aware tree key.
///
/// Trees are keyed by `pool::model` so that an alternating prefill→decode call sequence
/// for the same model cannot evict each other's tenants. Without this isolation, the
/// `tree.insert(text, url)` at the end of every `select_worker` call would overwrite
/// the previous pool's tenant for the same prompt and collapse cache_aware into a
/// flip-flop between pools.
fn pool_tag(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

fn make_tree_key(pool: &str, model: &str) -> String {
    format!("{}::{}", pool, model)
}

fn tree_key_for_worker(worker: &dyn Worker) -> String {
    make_tree_key(
        pool_tag(worker.worker_type()),
        normalize_model_key(worker.model_id()),
    )
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per `(pool, model)` so that prefill, decode, and
/// regular worker pools cannot evict each other's tenants.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    _eviction_task: Option<PeriodicTask>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let tree_key = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "Cache eviction completed for {}, max_size: {}",
                            tree_key, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            trees,
            _eviction_task: eviction_task,
        }
    }

    /// Initialize the tree with worker URLs (used only during initial setup)
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by (pool, model) so each pool gets its own isolated tree.
        let mut grouped: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            grouped
                .entry(tree_key_for_worker(worker.as_ref()))
                .or_default()
                .push(worker);
        }

        for (tree_key, pool_workers) in grouped {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in pool_workers {
                tree.insert("", worker.url());
            }
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        let tree = self
            .trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        if let Some(tree) = self.trees.get(&tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let tree_key = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!("Cache eviction for {}, max_size: {}", tree_key, max_size);
        }
    }

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: &Option<&str>,
        healthy_indices: &[usize],
        tree_key: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        // Use shortest queue when imbalanced. Tie break randomly.
        // Snapshot load() (live atomic count of load). Without snapshot
        // there could be no workers found matching min_load because of
        // load update.
        let loads: Vec<(usize, usize)> = healthy_indices
            .iter()
            .map(|&idx| (idx, workers[idx].load()))
            .collect();
        let min_load = loads.iter().map(|&(_, load)| load).min()?;
        let min_load_idx = loads
            .iter()
            .copied()
            .filter(|&(_, load)| load == min_load)
            .map(|(idx, _)| idx)
            .choose(&mut rand::rng())?;

        // Even in imbalanced mode, update the tree to maintain cache state
        if let Some(text) = request_text {
            // Get the tree reference without locking the entire HashMap
            // DashMap only locks the specific shard containing this key
            let tree = self.trees.get(tree_key).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                // Now we can work with the tree without holding the HashMap lock
                tree.insert(text, worker_url);
            } else {
                warn!(
                    "cache_aware: no tree found for key '{}', skipping cache update — \
                     pool tree was not seeded (init_pd_cache_aware_policies missed or \
                     a race during worker registration)",
                    tree_key
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();

        Some(min_load_idx)
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let request_text = info.request_text;
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the (pool, model) key for this set of workers — the router pre-filters
        // so every healthy worker here belongs to the same pool and same model.
        let pivot = workers[healthy_indices[0]].as_ref();
        let tree_key = tree_key_for_worker(pivot);

        // Get current load statistics - compute min/max in single pass without allocation
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                &request_text,
                &healthy_indices,
                &tree_key,
                max_load,
                min_load,
            );
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(&tree_key).map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Now we work with the tree without holding the HashMap lock
            // Use prefix_match_with_counts to avoid redundant chars().count() calls
            let result = tree.prefix_match_with_counts(text);
            let match_rate = if result.input_char_count == 0 {
                0.0
            } else {
                result.matched_char_count as f32 / result.input_char_count as f32
            };

            // Select worker without String allocation
            let selected_idx = if match_rate > self.config.cache_threshold {
                // Cache hit path: find worker by URL (compare &str directly, no allocation)
                let tenant_url: &str = &result.tenant;
                workers
                    .iter()
                    .position(|w| w.url() == tenant_url)
                    .filter(|&idx| workers[idx].is_healthy())
            } else {
                // Low cache match: use worker with minimum load. Tie break randomly.
                // Snapshot load() (live atomic count of load). Without snapshot
                // there could be no workers found matching min_load because of
                // load update.
                let loads: Vec<(usize, usize)> = healthy_indices
                    .iter()
                    .map(|&idx| (idx, workers[idx].load()))
                    .collect();
                let min_load = loads.iter().map(|&(_, load)| load).min()?;
                loads
                    .iter()
                    .copied()
                    .filter(|&(_, load)| load == min_load)
                    .map(|(idx, _)| idx)
                    .choose(&mut rand::rng())
            };

            if let Some(idx) = selected_idx {
                // Update the tree with this request (use worker URL directly, no allocation)
                tree.insert(text, workers[idx].url());

                // Increment processed counter
                workers[idx].increment_processed();

                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy, remove stale tenant from tree
            if match_rate > self.config.cache_threshold {
                let tenant_url: &str = &result.tenant;
                tree.remove_tenant(tenant_url);
                debug!("Removed stale worker {} from cache tree", tenant_url);
            }

            // Fallback to first healthy worker
            healthy_indices.first().copied()
        } else {
            warn!(
                "cache_aware: no tree found for key '{}', falling back to random \
                 worker selection — pool tree was not seeded \
                 (init_pd_cache_aware_policies missed or a race during worker \
                 registration); cache affinity is effectively disabled until this \
                 clears",
                tree_key
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType, UNKNOWN_MODEL_ID};

    #[tokio::test]
    async fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[tokio::test]
    async fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        let info = SelectWorkerInfo {
            request_text: Some("test"),
            ..Default::default()
        };
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    // In imbalanced mode the overloaded worker must be avoided AND the remaining
    // tied (min-load) workers must be spread across via random tie-breaking.
    #[tokio::test]
    async fn test_cache_aware_imbalanced_random_tie_break() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            max_tree_size: 10000,
        });

        let num_workers = 5;
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for j in 0..num_workers {
            workers.push(Arc::new(
                BasicWorkerBuilder::new(format!("http://w{}:8000", j))
                    .worker_type(WorkerType::Regular)
                    .build(),
            ));
        }

        // Overload worker 0: max=50, min=0 => (50-0) > 5 AND 50 > 0*2.0 => imbalanced.
        for _ in 0..50 {
            workers[0].increment_load();
        }
        policy.init_workers(&workers);

        // Reuse the SAME prompt: the imbalanced branch bypasses cache affinity,
        // so even a guaranteed cache hit must not pin all traffic to one worker.
        let mut selection_counts = vec![0; num_workers];
        for _ in 0..100 {
            let info = SelectWorkerInfo {
                request_text: Some("same_prompt"),
                ..Default::default()
            };
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            selection_counts[idx] += 1;
        }

        // The overloaded worker is never selected in imbalanced mode.
        assert_eq!(
            selection_counts[0], 0,
            "overloaded worker was selected in imbalanced mode: {:?}",
            selection_counts
        );
        // Every tied (load-0) worker is selected at least once.
        for idx in 1..num_workers {
            assert!(
                selection_counts[idx] > 0,
                "tied worker {} was never selected in imbalanced mode: {:?}",
                idx,
                selection_counts
            );
        }
    }

    // Verify random tie breaking for cache misses and low load situations.
    // Important for when concurrency is lower than the number of workers.
    // Without random tie breaking there will be multiple workers with 0 load
    // and all requests will be sent to the same replica which doesn't utilize
    // the available memory on all workers for the KV cache.
    #[tokio::test]
    async fn test_cache_aware_random_tie_break_cold_start() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        // Create 5 workers
        let num_workers = 5;
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for j in 0..num_workers {
            workers.push(Arc::new(
                BasicWorkerBuilder::new(format!("http://w{}:8000", j))
                    .worker_type(WorkerType::Regular)
                    .build(),
            ));
        }
        policy.init_workers(&workers);

        // Send 100 requests with unique prompts to simulate 100 cache misses.
        let mut selection_counts = vec![0; num_workers];
        for i in 0..100 {
            let prompt = format!("{}_unique_request", i);
            let info = SelectWorkerInfo {
                request_text: Some(&prompt),
                ..Default::default()
            };
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            selection_counts[idx] += 1;
        }

        // Verify all workers were selected at least once when prompts are
        // unique.
        for (idx, &count) in selection_counts.iter().enumerate() {
            assert!(
                count > 0,
                "Worker {} was never selected; tie-breaking failed to spread across \
                 all tied workers: {:?}",
                idx,
                selection_counts
            );
        }
    }

    #[tokio::test]
    async fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await;
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test2"),
                    ..Default::default()
                },
            )
            .await;

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[tokio::test]
    async fn test_cache_aware_policy() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 0);
    }

    fn make_prefill(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9000),
                })
                .build(),
        )
    }

    fn make_decode(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Decode)
                .build(),
        )
    }

    /// PD setup with two separate `CacheAwarePolicy` instances — the production
    /// wiring. Each pool's tree is seeded only with its own workers. Across a
    /// 4-turn growing prompt, each pool must stick to one worker.
    #[tokio::test]
    async fn test_pd_pool_isolation_two_policies() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            cache_threshold: 0.0,
            ..Default::default()
        };
        let prefill_policy = CacheAwarePolicy::with_config(config.clone());
        let decode_policy = CacheAwarePolicy::with_config(config);

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            make_prefill("http://prefill0:8000"),
            make_prefill("http://prefill1:8000"),
        ];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![
            make_decode("http://decode0:8000"),
            make_decode("http://decode1:8000"),
        ];
        prefill_policy.init_workers(&prefill_workers);
        decode_policy.init_workers(&decode_workers);

        let turns = [
            "turn1",
            "turn1 turn2",
            "turn1 turn2 turn3",
            "turn1 turn2 turn3 turn4",
        ];

        let mut prefill_idx: Option<usize> = None;
        let mut decode_idx: Option<usize> = None;
        for prompt in turns {
            let info = SelectWorkerInfo {
                request_text: Some(prompt),
                ..Default::default()
            };
            let p = prefill_policy
                .select_worker(&prefill_workers, &info)
                .await
                .expect("prefill pool returns a worker");
            let d = decode_policy
                .select_worker(&decode_workers, &info)
                .await
                .expect("decode pool returns a worker");
            match prefill_idx {
                None => prefill_idx = Some(p),
                Some(pinned) => assert_eq!(
                    p, pinned,
                    "prefill should stay pinned across turns (prompt={prompt:?})"
                ),
            }
            match decode_idx {
                None => decode_idx = Some(d),
                Some(pinned) => assert_eq!(
                    d, pinned,
                    "decode should stay pinned across turns (prompt={prompt:?})"
                ),
            }
        }
    }

    /// Regression: even if a single `CacheAwarePolicy` instance is incorrectly
    /// wired to both pools, pool-aware tree keys must keep their state disjoint.
    /// The pre-fix code shared one trie keyed by `model_id`, so alternating
    /// prefill/decode `tree.insert` calls overwrote each other and the policy
    /// degenerated into worker-flipping random selection.
    #[tokio::test]
    async fn test_pd_pool_isolation_shared_policy_regression() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            cache_threshold: 0.0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            make_prefill("http://prefill0:8000"),
            make_prefill("http://prefill1:8000"),
        ];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![
            make_decode("http://decode0:8000"),
            make_decode("http://decode1:8000"),
        ];

        // One instance, mixed init — pool-aware keys split the trees internally.
        let mut combined: Vec<Arc<dyn Worker>> = Vec::new();
        combined.extend(prefill_workers.iter().cloned());
        combined.extend(decode_workers.iter().cloned());
        policy.init_workers(&combined);

        let turns = [
            "turn1",
            "turn1 turn2",
            "turn1 turn2 turn3",
            "turn1 turn2 turn3 turn4",
        ];

        let mut prefill_idx: Option<usize> = None;
        let mut decode_idx: Option<usize> = None;
        for prompt in turns {
            let info = SelectWorkerInfo {
                request_text: Some(prompt),
                ..Default::default()
            };
            let p = policy
                .select_worker(&prefill_workers, &info)
                .await
                .expect("prefill pool returns a worker");
            let d = policy
                .select_worker(&decode_workers, &info)
                .await
                .expect("decode pool returns a worker");

            assert!(
                prefill_workers[p].url().starts_with("http://prefill"),
                "prefill call must return a prefill index, got {} (prompt={prompt:?})",
                prefill_workers[p].url()
            );
            assert!(
                decode_workers[d].url().starts_with("http://decode"),
                "decode call must return a decode index, got {} (prompt={prompt:?})",
                decode_workers[d].url()
            );

            match prefill_idx {
                None => prefill_idx = Some(p),
                Some(pinned) => assert_eq!(
                    p, pinned,
                    "prefill should stay pinned across turns (prompt={prompt:?})"
                ),
            }
            match decode_idx {
                None => decode_idx = Some(d),
                Some(pinned) => assert_eq!(
                    d, pinned,
                    "decode should stay pinned across turns (prompt={prompt:?})"
                ),
            }
        }
    }

    /// Removing a PD worker via the composite-key `remove_worker(&dyn Worker)` path
    /// must drop it from its own pool's tree without touching the other pool. This
    /// covers `PolicyRegistry::remove_pd_worker_from_cache_aware`, which routes the
    /// removal here based on `worker.worker_type()`.
    #[tokio::test]
    async fn test_pd_pool_isolation_remove_worker() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let prefill0 = make_prefill("http://prefill0:8000");
        let prefill1 = make_prefill("http://prefill1:8000");
        let decode0 = make_decode("http://decode0:8000");
        let decode1 = make_decode("http://decode1:8000");

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];
        let combined: Vec<Arc<dyn Worker>> = vec![
            prefill0.clone(),
            prefill1.clone(),
            decode0.clone(),
            decode1.clone(),
        ];
        policy.init_workers(&combined);

        // Seed both trees with affinity for one prompt.
        let prompt = "shared prefix to seed the cache_aware trees";
        let info = SelectWorkerInfo {
            request_text: Some(prompt),
            ..Default::default()
        };
        policy
            .select_worker(&prefill_workers, &info)
            .await
            .expect("seed prefill");
        policy
            .select_worker(&decode_workers, &info)
            .await
            .expect("seed decode");

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);

        let prefill_before = policy
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree seeded");
        let decode_before = policy
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree seeded");
        assert!(
            prefill_before.starts_with("http://prefill"),
            "prefill tree should hold a prefill tenant before removal, got {prefill_before}"
        );
        assert!(
            decode_before.starts_with("http://decode"),
            "decode tree should hold a decode tenant before removal, got {decode_before}"
        );

        // Drop prefill0 via the composite-key removal path.
        policy.remove_worker(prefill0.as_ref());

        // The prefill tree must no longer point at prefill0 for the seeded prompt.
        let prefill_after = policy
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        assert_ne!(
            &*prefill_after,
            prefill0.url(),
            "prefill0 should be gone from the prefill tree"
        );

        // The decode tree must be byte-for-byte unchanged.
        let decode_after = policy
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_eq!(
            decode_after, decode_before,
            "removing a prefill worker must not touch the decode tree"
        );
    }

    /// Shared setup for `PolicyRegistry::remove_pd_worker_from_cache_aware` tests:
    /// build a registry whose prefill and decode policies are separate
    /// `CacheAwarePolicy` instances seeded with the matching pool's workers, then
    /// return the registry, the per-pool policy handles (for tree inspection), and
    /// representative workers from each pool.
    #[allow(clippy::type_complexity)]
    fn pd_registry_with_cache_aware_pools() -> (
        Arc<crate::policies::PolicyRegistry>,
        Arc<CacheAwarePolicy>,
        Arc<CacheAwarePolicy>,
        Arc<dyn Worker>,
        Arc<dyn Worker>,
        Arc<dyn Worker>,
        Arc<dyn Worker>,
    ) {
        let registry = Arc::new(crate::policies::PolicyRegistry::new(
            crate::config::PolicyConfig::RoundRobin,
        ));
        let no_eviction = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
        let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction));
        registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
        registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

        let prefill0 = make_prefill("http://prefill0:8000");
        let prefill1 = make_prefill("http://prefill1:8000");
        let decode0 = make_decode("http://decode0:8000");
        let decode1 = make_decode("http://decode1:8000");

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];
        registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        (
            registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1,
        )
    }

    /// Seed both pool trees so each has a known tenant for `prompt`, then return
    /// the (prefill_tenant, decode_tenant) snapshot to compare against after a
    /// dispatched removal.
    async fn seed_pd_pools(
        prefill_ca: &CacheAwarePolicy,
        decode_ca: &CacheAwarePolicy,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        prompt: &str,
    ) -> (Arc<str>, Arc<str>) {
        let info = SelectWorkerInfo {
            request_text: Some(prompt),
            ..Default::default()
        };
        prefill_ca
            .select_worker(prefill_workers, &info)
            .await
            .expect("prefill seed");
        decode_ca
            .select_worker(decode_workers, &info)
            .await
            .expect("decode seed");

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let prefill_tenant = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree seeded");
        let decode_tenant = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree seeded");
        (prefill_tenant, decode_tenant)
    }

    /// A `Prefill` worker passed to `remove_pd_worker_from_cache_aware` must hit
    /// the registry's `prefill_policy` and leave `decode_policy` untouched.
    /// Catches a dispatch swap like `Prefill => self.decode_policy.get()`.
    #[tokio::test]
    async fn test_registry_remove_pd_worker_prefill_dispatches_to_prefill_policy() {
        let (registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1) =
            pd_registry_with_cache_aware_pools();
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];

        let prompt = "prefix used to seed both pool trees";
        let (_prefill_before, decode_before) = seed_pd_pools(
            &prefill_ca,
            &decode_ca,
            &prefill_workers,
            &decode_workers,
            prompt,
        )
        .await;

        registry.remove_pd_worker_from_cache_aware(prefill0.as_ref());

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let prefill_after = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        assert_ne!(
            &*prefill_after,
            prefill0.url(),
            "registry dispatch must drop prefill0 from the prefill pool's tree"
        );
        let decode_after = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_eq!(
            decode_after, decode_before,
            "removing a prefill worker must not touch the decode pool's tree"
        );
    }

    /// Mirror of the prefill dispatch test for `Decode`. Catches a dispatch swap
    /// in the other direction (`Decode => self.prefill_policy.get()`).
    #[tokio::test]
    async fn test_registry_remove_pd_worker_decode_dispatches_to_decode_policy() {
        let (registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1) =
            pd_registry_with_cache_aware_pools();
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];

        let prompt = "prefix used to seed both pool trees";
        let (prefill_before, _decode_before) = seed_pd_pools(
            &prefill_ca,
            &decode_ca,
            &prefill_workers,
            &decode_workers,
            prompt,
        )
        .await;

        registry.remove_pd_worker_from_cache_aware(decode0.as_ref());

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let decode_after = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_ne!(
            &*decode_after,
            decode0.url(),
            "registry dispatch must drop decode0 from the decode pool's tree"
        );
        let prefill_after = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        assert_eq!(
            prefill_after, prefill_before,
            "removing a decode worker must not touch the prefill pool's tree"
        );
    }

    /// `remove_pd_worker_from_cache_aware` must short-circuit on `Regular`
    /// workers and silently ignore non-cache_aware policies (`name() != "cache_aware"`).
    /// Both branches are no-ops: neither pool tree changes, and no downcast panic.
    #[tokio::test]
    async fn test_registry_remove_pd_worker_regular_and_non_cache_aware_noop() {
        // (a) Regular worker: should early-return regardless of policy state.
        let (registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1) =
            pd_registry_with_cache_aware_pools();
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];

        let prompt = "regular-noop seed prompt";
        let (prefill_before, decode_before) = seed_pd_pools(
            &prefill_ca,
            &decode_ca,
            &prefill_workers,
            &decode_workers,
            prompt,
        )
        .await;

        let regular: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://regular0:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        registry.remove_pd_worker_from_cache_aware(regular.as_ref());

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let prefill_after = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        let decode_after = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_eq!(
            prefill_after, prefill_before,
            "Regular worker dispatch must not touch the prefill tree"
        );
        assert_eq!(
            decode_after, decode_before,
            "Regular worker dispatch must not touch the decode tree"
        );

        // (b) Non-cache_aware policy: PD pool is round_robin. The downcast must
        // be skipped (no panic) and the call must be a no-op.
        let registry =
            crate::policies::PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin);
        let rr_prefill: Arc<dyn LoadBalancingPolicy> =
            Arc::new(crate::policies::RoundRobinPolicy::new());
        let rr_decode: Arc<dyn LoadBalancingPolicy> =
            Arc::new(crate::policies::RoundRobinPolicy::new());
        registry.set_prefill_policy(rr_prefill);
        registry.set_decode_policy(rr_decode);
        // No panic, no downcast — this would fault if the guard
        // `policy.name() == "cache_aware"` were dropped.
        registry.remove_pd_worker_from_cache_aware(prefill0.as_ref());
        registry.remove_pd_worker_from_cache_aware(decode0.as_ref());
    }

    /// `init_pd_cache_aware_policies` must seed only the pool whose policy is
    /// cache_aware AND whose worker list is non-empty. Covers all four corners:
    /// both seeded, only-prefill-cache_aware, empty-worker short-circuit, and the
    /// non-cache_aware side staying a no-op.
    #[tokio::test]
    async fn test_registry_init_pd_cache_aware_policies_gating() {
        let no_eviction = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);

        let prefill0 = make_prefill("http://prefill0:8000");
        let decode0 = make_decode("http://decode0:8000");
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone()];

        // (a) Both pools are cache_aware with workers → both trees seeded under
        // the correct composite key.
        {
            let registry =
                crate::policies::PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin);
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

            registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

            assert!(
                prefill_ca.trees.contains_key(&prefill_key),
                "prefill cache_aware policy must be seeded under '{prefill_key}'"
            );
            assert!(
                decode_ca.trees.contains_key(&decode_key),
                "decode cache_aware policy must be seeded under '{decode_key}'"
            );
            assert!(
                !prefill_ca.trees.contains_key(&decode_key),
                "prefill_workers must not seed the decode tree key"
            );
            assert!(
                !decode_ca.trees.contains_key(&prefill_key),
                "decode_workers must not seed the prefill tree key"
            );
        }

        // (b) Only prefill is cache_aware (decode is round_robin) → prefill seeded,
        // decode side skipped silently (no downcast, no panic).
        {
            let registry =
                crate::policies::PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin);
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_rr: Arc<dyn LoadBalancingPolicy> =
                Arc::new(crate::policies::RoundRobinPolicy::new());
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_rr);

            registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

            assert!(
                prefill_ca.trees.contains_key(&prefill_key),
                "prefill cache_aware side must seed even when decode side is non-cache_aware"
            );
        }

        // (c) Both cache_aware but prefill worker list is empty → prefill tree
        // NOT seeded (the inner `!is_empty()` guard short-circuits); decode side
        // is still seeded.
        {
            let registry =
                crate::policies::PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin);
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

            registry.init_pd_cache_aware_policies(&[], &decode_workers);

            assert!(
                prefill_ca.trees.is_empty(),
                "empty prefill worker list must not seed the prefill tree"
            );
            assert!(
                decode_ca.trees.contains_key(&decode_key),
                "decode side must still seed when only the prefill list is empty"
            );
        }

        // (d) Both worker lists empty → neither pool seeded (init is a full no-op).
        {
            let registry =
                crate::policies::PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin);
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction));
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

            registry.init_pd_cache_aware_policies(&[], &[]);

            assert!(prefill_ca.trees.is_empty());
            assert!(decode_ca.trees.is_empty());
        }
    }
}
