//! Bounded background I/O. No subscription or metrics request runs on selection.

use parking_lot::Mutex;
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};
use tracing::instrument::WithSubscriber;

// Bound recovery independently of requests; persistent failures back off below.
const REPLAY_COOLDOWN: Duration = Duration::from_secs(30);
const REPLAY_TIMEOUT: Duration = Duration::from_secs(120);
const REPLAY_IDLE_TIMEOUT: Duration = Duration::from_secs(2);
const REPLAY_RETRY_BACKOFF: Duration = Duration::from_millis(100);
const MAX_REPLAY_NO_PROGRESS: usize = 3;
static REPLAY_SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(8);

// A cache miss gives live events time to arrive before checking retained history.
// Quiet workers with no requests never incur periodic replay traffic.
const PROGRESS_GRACE: Duration = Duration::from_secs(2);
const PROGRESS_BUDGET: Duration = Duration::from_secs(8);
const PROGRESS_ACTIVITY: Duration = Duration::from_secs(300);

use super::{
    config::{Config, EventSource},
    diagnostics::Diagnostics,
    events::StreamIndex,
    load::{parse_metrics, Loads},
    render::limited_body,
    WorkerState,
};
use crate::core::{BackendType, WorkerRegistry};

// Receipt and indexing are different stages. These counters remain useful even
// when a duplicate is filtered or a batch fails to update the serving index.
#[derive(Debug, Default)]
pub(super) struct EventProgress {
    pub running: bool,
    pub connection_generation: u64,
    pub last_received: Option<Instant>,
    pub last_received_sequence: Option<u64>,
    pub received_frames: u64,
    pub duplicate_frames: u64,
    pub apply_failures: u64,
    pub replay_in_progress: bool,
    pub connection_event: Option<&'static str>,
    pub last_cache_miss: Option<Instant>,
    pub first_cache_miss: Option<Instant>,
    pub progress_check_in_progress: bool,
    pub progress_checks: u64,
    pub progress_recovered_frames: u64,
    pub last_progress_check: Option<Instant>,
    pub last_progress_error: Option<&'static str>,
}

type ProgressFrames = Vec<(u64, Vec<u8>)>;
type ProgressResult = Result<ProgressFrames, (&'static str, String)>;

// Replay I/O runs separately from live receipt. Commit only if no live event,
// binding or connection changed in the meantime, so a late reply cannot undo
// a live removal, clear, or publisher restart. No serving-index clone is needed.
struct ProgressCheck {
    result: std::sync::mpsc::Receiver<ProgressResult>,
    revision: u64,
    generation: u64,
    received: u64,
    sequence: Option<u64>,
    started: Instant,
}

impl ProgressCheck {
    fn current(&self, state: &WorkerState) -> bool {
        self.revision == state.layout_revision
            && self.generation == state.progress.connection_generation
            && self.received == state.progress.received_frames
            && self.sequence == state.stream.last_seq
    }
}

#[derive(Default)]
struct ProgressRecovery {
    check: Option<ProgressCheck>,
    renewed_at: Option<Instant>,
    next_attempt: Option<Instant>,
    failures: u32,
}

impl ProgressRecovery {
    fn due(&self, state: &WorkerState, now: Instant) -> bool {
        self.check.is_none()
            && self.next_attempt.is_none_or(|at| now >= at)
            && !state.stream.unsupported
            && state
                .progress
                .first_cache_miss
                .is_some_and(|at| now.duration_since(at) >= PROGRESS_GRACE)
            && state
                .progress
                .last_cache_miss
                .is_some_and(|at| now.duration_since(at) <= PROGRESS_ACTIVITY)
            && state
                .progress
                .last_received
                .is_none_or(|at| now.duration_since(at) >= PROGRESS_GRACE)
    }

    fn completed(&mut self, failed: bool) {
        self.renewed_at = None;
        self.failures = if failed {
            self.failures.saturating_add(1)
        } else {
            0
        };
        let seconds = (30u64 << self.failures.saturating_sub(1).min(4)).min(300);
        self.next_attempt = Some(Instant::now() + Duration::from_secs(seconds));
    }
}

// Read only a bounded suffix. A successful empty reply means no retained update,
// not an empty cache. Validate continuity before touching the serving index.
fn progress_frames(
    source: &EventSource,
    cfg: &Config,
    from: u64,
    stop: &AtomicBool,
) -> ProgressResult {
    let _permit = REPLAY_SLOTS.try_acquire().map_err(|_| {
        (
            "replay_capacity",
            "Progress check deferred: replay capacity busy".into(),
        )
    })?;
    let context = zmq::Context::new();
    let error = |err: zmq::Error| ("replay_connect", err.to_string());
    let socket = context.socket(zmq::DEALER).map_err(error)?;
    socket.set_linger(0).map_err(error)?;
    socket.set_rcvtimeo(250).map_err(error)?;
    socket.set_sndtimeo(250).map_err(error)?;
    socket.set_rcvhwm(10_000).map_err(error)?;
    socket
        .set_maxmsgsize(cfg.max_response_bytes as i64)
        .map_err(error)?;
    keepalive(&socket).map_err(error)?;
    socket
        .connect(
            source
                .replay_endpoint
                .as_deref()
                .ok_or(("replay_endpoint", "No replay endpoint configured".into()))?,
        )
        .map_err(error)?;
    socket
        .send_multipart([&[][..], &from.to_be_bytes()[..]], 0)
        .map_err(|err| ("replay_send", err.to_string()))?;
    let started = Instant::now();
    let mut last_received = started;
    let mut batches: ProgressFrames = Vec::new();
    let mut bytes = 0usize;
    while !stop.load(Ordering::Acquire) && started.elapsed() < PROGRESS_BUDGET {
        let frames = match socket.recv_multipart(0) {
            Ok(frames) => frames,
            Err(zmq::Error::EAGAIN) if last_received.elapsed() < REPLAY_IDLE_TIMEOUT => continue,
            Err(err) => return Err(("replay_no_progress", err.to_string())),
        };
        last_received = Instant::now();
        let (seq, body) = replay_frame(&frames, &source.topic)
            .ok_or(("replay_frame", "Invalid progress replay frame".into()))?;
        if body.is_empty() {
            return Ok(batches);
        }
        let expected = batches
            .last()
            .map_or(Some(from), |(last, _)| last.checked_add(1));
        if seq < from || (!batches.is_empty() && expected != Some(seq)) {
            return Err(("replay_sequence", "Non-contiguous progress replay".into()));
        }
        bytes = bytes.saturating_add(body.len());
        if bytes > cfg.max_response_bytes || batches.len() >= 10_000 {
            return Err((
                "replay_capacity",
                "Progress replay exceeded its bounded response budget".into(),
            ));
        }
        batches.push((seq, body.to_vec()));
    }
    Err((
        "replay_timeout",
        "Progress replay cancelled or exceeded eight seconds".into(),
    ))
}

// Apply only complete, contiguous replies. A truncated prefix cannot preserve
// old residency because its missing events may include removals.
fn apply_progress(
    state: &mut WorkerState,
    frames: &ProgressFrames,
    source: &EventSource,
    cfg: &Config,
) -> Result<(), &'static str> {
    for (seq, body) in frames {
        if state
            .stream
            .last_seq
            .is_some_and(|last| last.checked_add(1) != Some(*seq))
        {
            state.stream.invalidate();
            state.stream.history_truncated_at = Some(*seq);
        } else if state.stream.last_seq.is_none() && *seq > 0 {
            state.stream.history_truncated_at = Some(*seq);
        }
        match state.stream.apply(*seq, body, &source.profile, cfg) {
            Ok(()) | Err("missing_parent" | "sparse_hash_unknown") => {}
            Err(reason) => return Err(reason),
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct LiveSequence {
    last: Option<u64>,
    replay_highwater: Option<u64>,
}

#[derive(Debug, PartialEq, Eq)]
enum Delivery {
    Next,
    Duplicate,
    Restart,
}

impl LiveSequence {
    // llm-d checks live regression before replay overlap. A high replay mark
    // must never hide a restarted publisher's low sequence numbers.
    fn observe(&mut self, seq: u64) -> Delivery {
        if self.last.is_some_and(|last| seq < last) {
            self.last = Some(seq);
            self.replay_highwater = None;
            return Delivery::Restart;
        }
        if self.last == Some(seq) {
            return Delivery::Duplicate;
        }
        self.last = Some(seq);
        if self.replay_highwater.is_some_and(|last| seq <= last) {
            return Delivery::Duplicate;
        }
        self.replay_highwater = None;
        Delivery::Next
    }
}

// Probe TCP liveness without expiring a quiet cache or creating KV events.
fn keepalive(socket: &zmq::Socket) -> Result<(), zmq::Error> {
    socket.set_tcp_keepalive(1)?;
    socket.set_tcp_keepalive_idle(60)?;
    socket.set_tcp_keepalive_intvl(15)?;
    socket.set_tcp_keepalive_cnt(3)
}

fn report_isolation(
    diagnostics: &Diagnostics,
    source: &EventSource,
    stream: &StreamIndex,
    origin: &str,
) {
    if let Some(sample) = &stream.last_skip {
        if diagnostics.is_active(&source.worker_url, "unsupported_cache_identity") {
            return;
        }
        diagnostics.warn(&source.worker_url, "unsupported_cache_identity", &format!(
            "Unsupported Store identity isolated; unrelated branches can still be indexed; origin={origin}; sample={}",
            serde_json::to_string(sample).unwrap_or_default()));
    } else if stream.history_complete {
        // An empty serving subset during replay is not evidence that identity
        // coverage recovered. Only a verified clear/full rebuild can prove it.
        diagnostics.recovered(&source.worker_url, "unsupported_cache_identity");
    }
}

// Socket connection is asynchronous: only accepted events prove usable KV state.
// Keep the last accepted time across invalidation so outages are not mislabeled
// as a publisher that has never sent events.
struct EventHealth {
    started: Instant,
    last_event: Option<Instant>,
    announced: bool,
    received: bool,
    problems: BTreeSet<&'static str>,
    replay_problems: BTreeSet<&'static str>,
}

// Keep an interrupted checkpoint private, including across timer retries. Never
// serve it before replay has checked all removals through the latest live event.
#[derive(Default)]
struct ReplayRecovery {
    failures: u32,
    next_attempt: Option<Instant>,
    pending: Option<StreamIndex>,
    live_seq: Option<u64>,
    ancestry_checked: bool,
}

impl ReplayRecovery {
    fn ready(&self) -> bool {
        self.next_attempt.is_none_or(|at| Instant::now() >= at)
    }

    fn failed(&mut self) -> u64 {
        self.failures = self.failures.saturating_add(1);
        let seconds = (REPLAY_COOLDOWN.as_secs() * (1u64 << (self.failures - 1).min(4))).min(300);
        self.next_attempt = Some(Instant::now() + Duration::from_secs(seconds));
        seconds
    }

    fn retry_due(&self) -> bool {
        self.pending.is_some() && self.ready()
    }
}

impl EventHealth {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            last_event: None,
            announced: false,
            received: false,
            problems: BTreeSet::new(),
            replay_problems: BTreeSet::new(),
        }
    }

    fn warn(
        &mut self,
        diagnostics: &Diagnostics,
        source: &EventSource,
        cfg: &Config,
        reason: &'static str,
        detail: &str,
    ) {
        self.problems.insert(reason);
        diagnostics.warn(
            &source.worker_url,
            reason,
            &format!(
                "{detail}; endpoint={} replay_endpoint={:?} topic={} profile={} block_size={}",
                source.endpoint,
                source.replay_endpoint,
                source.topic,
                source.profile,
                cfg.profiles[&source.profile].block_size,
            ),
        );
    }

    fn check(&mut self, diagnostics: &Diagnostics, source: &EventSource, cfg: &Config) {
        // A publisher need not emit anything while its cache is unchanged.
        // Only an unconfirmed subscription warrants a no-event warning.
        if self.last_event.is_some()
            || self.started.elapsed() < Duration::from_millis(cfg.event_stale_ms)
        {
            return;
        }
        if self.received {
            self.warn(diagnostics, source, cfg, "kv_events_unusable", "KV frames received but no usable batch accepted; inspect event errors and last_received_sequence");
        } else {
            self.warn(diagnostics, source, cfg, "no_kv_events",
                "No KV frames received since subscription startup; check worker --kv-events-config, endpoint, topic, and network; an idle worker may have no cache changes");
        }
    }

    fn accepted(
        &mut self,
        diagnostics: &Diagnostics,
        source: &EventSource,
        at: Instant,
        replayed: bool,
    ) {
        if self.last_event.is_some_and(|last| at <= last) {
            return;
        }
        if !self.announced {
            tracing::info!(
                worker_url = %source.worker_url,
                profile = %source.profile,
                endpoint = %source.endpoint,
                topic = %source.topic,
                "pre_kv_cache event stream active"
            );
            self.announced = true;
        }
        self.last_event = Some(at);
        if replayed {
            self.replay_problems.clear();
        }
        for reason in std::mem::take(&mut self.problems) {
            if self.replay_problems.contains(reason)
                || (!replayed && matches!(reason, "missing_parent" | "sparse_hash_unknown"))
            {
                self.problems.insert(reason);
            } else {
                diagnostics.recovered(&source.worker_url, reason);
            }
        }
    }

    // A structurally valid partial batch proves receipt, but cannot repair its
    // missing ancestry. Recover silence warnings without clearing index errors.
    fn partially_indexed(&mut self, diagnostics: &Diagnostics, source: &EventSource, at: Instant) {
        if self.last_event.is_some_and(|last| at <= last) {
            return;
        }
        self.last_event = Some(at);
        self.problems.remove("no_kv_events");
        diagnostics.recovered(&source.worker_url, "no_kv_events");
        self.problems.remove("kv_events_unusable");
        diagnostics.recovered(&source.worker_url, "kv_events_unusable");
    }
}

pub fn subscribe(
    source: EventSource,
    cfg: Arc<Config>,
    registry: Weak<WorkerRegistry>,
    state: Arc<Mutex<WorkerState>>,
    stop: Arc<AtomicBool>,
    diagnostics: Arc<Diagnostics>,
) -> tokio::task::JoinHandle<()> {
    // Supervise the blocking subscriber: a panic must not leave an old index
    // silently usable forever. Reuse the existing subscription and stop flag.
    let dispatcher = tracing::dispatcher::get_default(Clone::clone);
    tokio::spawn(
        async move {
            loop {
                let result = subscribe_once(
                    source.clone(),
                    cfg.clone(),
                    registry.clone(),
                    state.clone(),
                    stop.clone(),
                    diagnostics.clone(),
                )
                .await;
                {
                    let mut state = state.lock();
                    state.progress.running = false;
                    state.progress.replay_in_progress = false;
                    state.progress.progress_check_in_progress = false;
                    state.progress.connection_event = Some("stopped");
                    state.stream.reset_epoch();
                }
                if stop.load(Ordering::Acquire) || registry.upgrade().is_none() {
                    break;
                }
                diagnostics.warn(
                    &source.worker_url,
                    "subscriber_task_stopped",
                    &format!("KV subscriber exited; index invalidated; restarting: {result:?}"),
                );
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
        .with_subscriber(dispatcher),
    )
}

fn subscribe_once(
    source: EventSource,
    cfg: Arc<Config>,
    registry: Weak<WorkerRegistry>,
    state: Arc<Mutex<WorkerState>>,
    stop: Arc<AtomicBool>,
    diagnostics: Arc<Diagnostics>,
) -> tokio::task::JoinHandle<()> {
    // libzmq receives on a dedicated blocking thread with a finite timeout.
    let dispatcher = tracing::dispatcher::get_default(Clone::clone);
    tokio::task::spawn_blocking(move || {
        let _dispatcher = tracing::dispatcher::set_default(&dispatcher);
        let context = zmq::Context::new();
        let mut socket: Option<zmq::Socket> = None;
        let mut monitor: Option<zmq::Socket> = None;
        let mut sequence = LiveSequence::default();
        let mut layout_revision = 0;
        let mut health = EventHealth::new();
        let mut recovery = ReplayRecovery::default();
        let mut progress = ProgressRecovery::default();
        state.lock().progress.running = true;
        while !stop.load(Ordering::Acquire) {
            let Some(registry) = registry.upgrade() else {
                break;
            };
            let worker = registry
                .get_by_url(&source.worker_url)
                .filter(|w| matches!(w.metadata().runtime_type, BackendType::Vllm));
            {
                let mut state = state.lock();
                if state.bind(worker.clone(), &cfg.profiles[&source.profile]) {
                    recovery = ReplayRecovery::default();
                    progress = ProgressRecovery::default();
                    state.progress.progress_check_in_progress = false;
                    state.progress.first_cache_miss = None;
                    state.progress.last_cache_miss = None;
                    socket = None;
                    monitor = None;
                    sequence = LiveSequence::default();
                    health.started = Instant::now();
                    health.last_event = None;
                    health.announced = false;
                    health.received = false;
                }
                if layout_revision != state.layout_revision {
                    recovery = ReplayRecovery::default();
                    progress = ProgressRecovery::default();
                    state.progress.progress_check_in_progress = false;
                    layout_revision = state.layout_revision;
                    socket = None;
                    monitor = None;
                    sequence = LiveSequence::default();
                    // Background layout detection can bind the replacement
                    // before this thread; do not inherit its old silence timer.
                    health.started = Instant::now();
                    health.last_event = None;
                    health.announced = false;
                    health.received = false;
                }
            }
            if worker.is_none() {
                std::thread::sleep(Duration::from_millis(250));
                continue;
            }
            health.check(&diagnostics, &source, &cfg);
            if socket.is_none() {
                if state.lock().stream.last_seq.is_none() {
                    sequence = LiveSequence::default();
                }
                let connected = (|| -> Result<(zmq::Socket, zmq::Socket), zmq::Error> {
                    let s = context.socket(zmq::SUB)?;
                    keepalive(&s)?;
                    let address = format!("inproc://kv-monitor-{}", uuid::Uuid::new_v4());
                    s.monitor(&address, zmq::SocketEvent::DISCONNECTED as i32)?;
                    let m = context.socket(zmq::PAIR)?;
                    m.set_linger(0)?;
                    m.connect(&address)?;
                    s.set_linger(0)?;
                    s.set_rcvtimeo(250)?;
                    s.set_rcvhwm(10_000)?;
                    s.set_maxmsgsize(cfg.max_response_bytes as i64)?;
                    s.set_subscribe(source.topic.as_bytes())?;
                    s.connect(&source.endpoint)?;
                    Ok((s, m))
                })();
                match connected {
                    Ok((s, m)) => {
                        let mut state = state.lock();
                        state.progress.connection_generation =
                            state.progress.connection_generation.saturating_add(1);
                        state.progress.connection_event = Some("connecting");
                        socket = Some(s);
                        monitor = Some(m);
                        // Start with live events. Binding and later automatic
                        // layout detection must not each launch a history scan.
                    }
                    Err(err) => {
                        state.lock().progress.connection_event = Some("connect_failed");
                        error("event_connect");
                        health.warn(
                            &diagnostics,
                            &source,
                            &cfg,
                            "event_connect",
                            &err.to_string(),
                        );
                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }
                }
            }
            // libzmq reconnects silently. Explicitly invalidate on disconnect,
            // including when the publisher restarts with the same URL/sequence.
            if monitor
                .as_ref()
                .is_some_and(|m| m.recv_multipart(zmq::DONTWAIT).is_ok())
            {
                state.lock().stream.reset_epoch();
                recovery = ReplayRecovery::default();
                socket = None;
                monitor = None;
                error("disconnected");
                state.lock().progress.connection_event = Some("disconnected");
                health.warn(
                    &diagnostics,
                    &source,
                    &cfg,
                    "disconnected",
                    "KV publisher disconnected; index invalidated",
                );
                continue;
            }
            // Poll independent history I/O without blocking live receipt. Any
            // concurrent live delivery makes the reply obsolete, not authoritative.
            let completed =
                progress
                    .check
                    .as_ref()
                    .and_then(|check| match check.result.try_recv() {
                        Ok(result) => Some(result),
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(Err((
                            "replay_task_stopped",
                            "Progress replay task stopped".into(),
                        ))),
                        Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    });
            if let Some(result) = completed {
                let check = progress.check.take().unwrap();
                let mut s = state.lock();
                s.progress.progress_check_in_progress = false;
                if check.current(&s) && !stop.load(Ordering::Acquire) && recovery.pending.is_none()
                {
                    let result = result.and_then(|frames| {
                        apply_progress(&mut s, &frames, &source, &cfg)
                            .map_err(|reason| (reason, "Progress replay event rejected".into()))?;
                        Ok(frames.len())
                    });
                    match result {
                        Ok(count) => {
                            s.progress.last_progress_error = None;
                            s.progress.progress_recovered_frames = s
                                .progress
                                .progress_recovered_frames
                                .saturating_add(count as u64);
                            diagnostics.recovered(&source.worker_url, "event_progress_unavailable");
                            if count > 0 {
                                sequence.replay_highwater = s.stream.last_seq;
                                if let Some(at) = s.stream.last_event {
                                    health.accepted(
                                        &diagnostics,
                                        &source,
                                        at,
                                        s.stream.history_complete,
                                    );
                                }
                                if s.progress
                                    .last_cache_miss
                                    .is_some_and(|at| at <= check.started)
                                {
                                    s.progress.first_cache_miss = None;
                                    s.progress.last_cache_miss = None;
                                }
                                report_isolation(
                                    &diagnostics,
                                    &source,
                                    &s.stream,
                                    "progress_replay",
                                );
                                tracing::info!(worker_url = %source.worker_url, recovered_frames = count,
                                    last_sequence = ?s.stream.last_seq, resident_blocks = s.stream.index.len(),
                                    "pre_kv_cache event progress recovered from replay");
                            }
                            progress.completed(false);
                        }
                        Err((reason, detail)) => {
                            s.progress.last_progress_error = Some(reason);
                            diagnostics.warn(&source.worker_url, "event_progress_unavailable", &format!(
                                "Request-driven KV progress check failed: {reason}: {detail}; live receipt continues; checks back off while requests are active"
                            ));
                            progress.completed(true);
                        }
                    }
                    metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone())
                        .set(s.stream.index.len() as f64);
                } else {
                    // Obsolete replies are discarded without changing the index.
                    progress.completed(false);
                }
            }
            if recovery.pending.is_none() && progress.due(&state.lock(), Instant::now()) {
                // Renew BEFORE reading history, keeping this new SUB alive while
                // replay runs and afterwards. Closing it after replay could lose
                // a removal queued just beyond the replay boundary. Local renewal
                // alone proves neither eviction nor restart, so retain the index
                // and sequence watermark. Give the subscription time to propagate.
                if progress.renewed_at.is_none() {
                    socket = None;
                    monitor = None;
                    progress.renewed_at = Some(Instant::now());
                    continue;
                }
                if source.replay_endpoint.is_none() {
                    progress.completed(false);
                } else if progress
                    .renewed_at
                    .is_some_and(|at| at.elapsed() >= PROGRESS_GRACE)
                {
                    let mut s = state.lock();
                    let Some(from) = s
                        .stream
                        .last_seq
                        .map_or(Some(0), |last| last.checked_add(1))
                    else {
                        progress.completed(true);
                        continue;
                    };
                    let (sender, result) = std::sync::mpsc::sync_channel(1);
                    let started = Instant::now();
                    progress.check = Some(ProgressCheck {
                        result,
                        revision: s.layout_revision,
                        generation: s.progress.connection_generation,
                        received: s.progress.received_frames,
                        sequence: s.stream.last_seq,
                        started,
                    });
                    s.progress.progress_check_in_progress = true;
                    s.progress.progress_checks = s.progress.progress_checks.saturating_add(1);
                    s.progress.last_progress_check = Some(started);
                    let source = source.clone();
                    let cfg = cfg.clone();
                    let stop = stop.clone();
                    tokio::task::spawn_blocking(move || {
                        let _ = sender.send(progress_frames(&source, &cfg, from, &stop));
                    });
                }
            }
            let frames = match socket.as_ref().unwrap().recv_multipart(0) {
                Ok(frames) => {
                    let mut state = state.lock();
                    state.progress.received_frames =
                        state.progress.received_frames.saturating_add(1);
                    state.progress.last_received = Some(Instant::now());
                    state.progress.connection_event = Some("received");
                    health.received = true;
                    Some(frames)
                }
                // Retry transient failures even when fully cached traffic
                // produces no new events. Startup has no pending recovery.
                Err(zmq::Error::EAGAIN) if recovery.retry_due() => None,
                Err(zmq::Error::EAGAIN) => continue,
                Err(err) => {
                    socket = None;
                    monitor = None;
                    state.lock().stream.reset_epoch();
                    recovery = ReplayRecovery::default();
                    error("event_receive");
                    state.lock().progress.connection_event = Some("receive_failed");
                    health.warn(
                        &diagnostics,
                        &source,
                        &cfg,
                        "event_receive",
                        &err.to_string(),
                    );
                    continue;
                }
            };
            // Discard queued frames from a connection that closed during recv.
            if monitor
                .as_ref()
                .is_some_and(|m| m.recv_multipart(zmq::DONTWAIT).is_ok())
            {
                state.lock().stream.reset_epoch();
                recovery = ReplayRecovery::default();
                socket = None;
                monitor = None;
                error("disconnected");
                state.lock().progress.connection_event = Some("disconnected");
                health.warn(
                    &diagnostics,
                    &source,
                    &cfg,
                    "disconnected",
                    "KV publisher disconnected while receiving; queued events discarded",
                );
                continue;
            }
            if let Some(frames) = frames.as_deref() {
                let Some((seq, payload)) = frame(frames, &source.topic) else {
                    state.lock().stream.invalidate();
                    recovery = ReplayRecovery::default();
                    error("event_frame");
                    health.warn(
                        &diagnostics,
                        &source,
                        &cfg,
                        "event_frame",
                        "Malformed KV event frames or mismatched topic; index invalidated",
                    );
                    continue;
                };
                // Metadata may change while recv blocks. Never apply an old socket's
                // frames to the new engine incarnation or its replacement layout.
                if state.lock().layout_revision != layout_revision {
                    continue;
                }
                state.lock().progress.last_received_sequence = Some(seq);
                let delivery = sequence.observe(seq);
                if delivery == Delivery::Duplicate {
                    let mut state = state.lock();
                    state.progress.duplicate_frames =
                        state.progress.duplicate_frames.saturating_add(1);
                    continue;
                }
                // Idle residency remains usable. Duplicate delivery changes neither
                // the index nor its observation timestamp; actual gaps invalidate it.
                {
                    let mut s = state.lock();
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    if s.layout_revision != layout_revision {
                        continue;
                    }
                    if delivery == Delivery::Restart {
                        s.stream.reset_epoch();
                        recovery = ReplayRecovery::default();
                        health.warn(
                            &diagnostics,
                            &source,
                            &cfg,
                            "event_sequence",
                            "Live publisher sequence restarted; old residency discarded",
                        );
                    }
                    let last = s.stream.last_seq;
                    if last == Some(seq) {
                        continue;
                    }
                    let gap = last.is_some_and(|last| last.checked_add(1) != Some(seq));
                    if gap {
                        health.replay_problems.insert("event_sequence");
                        health.warn(&diagnostics, &source, &cfg, "event_sequence", &format!(
                        "KV event sequence changed from {} to {seq}; suspending old residency until replay verifies the gap",
                        last.unwrap(),
                    ));
                        // Move, rather than clone, the checkpoint out of serving
                        // state. Missing deletes must not leave stale positive hits
                        // visible while replay is in flight or if it fails.
                        if !s.stream.unsupported {
                            let empty = s.stream.empty_with_layout();
                            let previous = std::mem::replace(&mut s.stream, empty);
                            if seq < last.unwrap() {
                                // A sequence regression belongs to a new epoch.
                                recovery = ReplayRecovery::default();
                            } else if recovery.pending.is_none() {
                                recovery.pending = Some(previous);
                            }
                        }
                        metrics::counter!("pre_kv_cache_event_resets_total", "reason" => "sequence")
                        .increment(1);
                    }
                    let result = s.stream.apply(seq, payload, &source.profile, &cfg);
                    if let Err(reason) = result {
                        s.progress.apply_failures = s.progress.apply_failures.saturating_add(1);
                        error(reason);
                        let observed: Vec<_> = s.stream.observed_groups.values().collect();
                        health.warn(&diagnostics, &source, &cfg, reason, &format!(
                        "KV event processing failed; event_state_usable={}; background replay will attempt recovery when supported; observed_groups={} (not a complete inventory)",
                        s.event_state_usable(),
                        serde_json::to_string(&observed).unwrap_or_default(),
                    ));
                        if let Some(at) = s.stream.last_event {
                            health.partially_indexed(&diagnostics, &source, at);
                        }
                    } else if let Some(at) = s.stream.last_event {
                        progress.renewed_at = None;
                        s.progress.first_cache_miss = None;
                        s.progress.last_cache_miss = None;
                        s.progress.last_progress_error = None;
                        diagnostics.recovered(&source.worker_url, "event_progress_unavailable");
                        diagnostics.recovered(&source.worker_url, "subscriber_task_stopped");
                        health.accepted(&diagnostics, &source, at, s.stream.history_complete);
                        if s.stream.history_complete {
                            recovery = ReplayRecovery::default();
                        }
                    }
                    report_isolation(&diagnostics, &source, &s.stream, "live");
                    metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone())
                    .set(s.stream.index.len() as f64);
                    if s.stream.unsupported || source.replay_endpoint.is_none() {
                        recovery = ReplayRecovery::default();
                    } else if !s.stream.history_complete {
                        // A known truncated suffix cannot regain its missing
                        // ancestors by repeatedly asking for sequence zero.
                        if recovery.pending.is_none()
                            && (gap
                                || (s.stream.incomplete()
                                    && s.stream.history_truncated_at.is_none()
                                    && !recovery.ancestry_checked))
                        {
                            recovery.pending = Some(s.stream.empty_with_layout());
                        }
                        if recovery.pending.is_some() {
                            recovery.live_seq = Some(seq);
                        }
                    }
                }
            }
            if recovery.retry_due() && progress.check.is_none() {
                state.lock().progress.replay_in_progress = true;
                let mut checkpoint = recovery.pending.take().unwrap();
                let live_seq = recovery.live_seq;
                let from_seq = checkpoint
                    .last_seq
                    .and_then(|last| last.checked_add(1))
                    .unwrap_or(0);
                tracing::info!(worker_url = %source.worker_url, from_sequence = from_seq,
                    live_sequence = ?live_seq, "pre_kv_cache replay started");
                let rebuilt = replay_checked(
                    &context,
                    &source,
                    &cfg,
                    live_seq,
                    &stop,
                    &mut checkpoint,
                    monitor.as_ref(),
                );
                state.lock().progress.replay_in_progress = false;
                // A publisher can disconnect/restart while replay blocks. Never
                // publish a response from that old connection into current state.
                if matches!(&rebuilt, Err(("disconnected", _)))
                    || monitor
                        .as_ref()
                        .is_some_and(|m| m.recv_multipart(zmq::DONTWAIT).is_ok())
                {
                    state.lock().stream.reset_epoch();
                    socket = None;
                    monitor = None;
                    recovery = ReplayRecovery::default();
                    error("disconnected");
                    state.lock().progress.connection_event = Some("disconnected");
                    health.warn(
                        &diagnostics,
                        &source,
                        &cfg,
                        "disconnected",
                        "KV publisher disconnected during replay; rebuilt state discarded",
                    );
                    continue;
                }
                let mut s = state.lock();
                if stop.load(Ordering::Acquire) {
                    break;
                }
                if s.layout_revision != layout_revision {
                    continue;
                }
                match rebuilt {
                    Ok(()) => {
                        sequence.replay_highwater = checkpoint.last_seq;
                        for reason in std::mem::take(&mut health.replay_problems) {
                            health.problems.remove(reason);
                            diagnostics.recovered(&source.worker_url, reason);
                        }
                        if let Some(at) = checkpoint.last_event {
                            health.accepted(&diagnostics, &source, at, checkpoint.history_complete);
                        }
                        tracing::info!(worker_url = %source.worker_url, from_sequence = from_seq,
                            last_sequence = ?checkpoint.last_seq, resident_blocks = checkpoint.index.len(),
                            history_complete = checkpoint.history_complete,
                            history_truncated_at = ?checkpoint.history_truncated_at,
                            "pre_kv_cache replay completed");
                        metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone())
                            .set(checkpoint.index.len() as f64);
                        s.stream = checkpoint;
                        report_isolation(&diagnostics, &source, &s.stream, "replay");
                        recovery = ReplayRecovery {
                            // Suppress only an unresolved partial-history scan.
                            // New ancestry loss after a full recovery is repairable.
                            ancestry_checked: !s.stream.history_complete,
                            ..Default::default()
                        };
                    }
                    Err((reason, detail)) => {
                        // Retain validated progress privately for a timer retry.
                        // Protocol/layout errors cannot be healed by retransmit.
                        if matches!(
                            reason,
                            "replay_no_progress"
                                | "replay_timeout"
                                | "replay_incomplete"
                                | "replay_capacity"
                                | "replay_connect"
                                | "replay_send"
                        ) {
                            recovery.pending = Some(checkpoint);
                        } else {
                            // Repeating an identical non-retryable history scan
                            // cannot heal it. A new gap/epoch can request recovery.
                            recovery.ancestry_checked = true;
                        }
                        let retry_after = recovery.failed();
                        error(reason);
                        health.replay_problems.insert(reason);
                        health.warn(&diagnostics, &source, &cfg, reason, &format!(
                            "Replay failed: {detail}; verified progress kept private; timer retry={} after {retry_after}s; live branches remain available",
                            recovery.pending.is_some(),
                        ));
                    }
                }
                s.progress.replay_in_progress = false;
            }
        }
        state.lock().stream.reset_epoch();
    })
}

fn error(reason: &'static str) {
    metrics::counter!("pre_kv_cache_event_errors_total", "reason" => reason).increment(1);
}

fn frame<'a>(frames: &'a [Vec<u8>], topic: &str) -> Option<(u64, &'a [u8])> {
    if frames.len() != 3 || frames[0] != topic.as_bytes() {
        return None;
    }
    let bytes: [u8; 8] = frames[1].as_slice().try_into().ok()?;
    Some((u64::from_be_bytes(bytes), &frames[2]))
}

// Replay's empty-topic/-1 end marker is a control frame, not a KV event. Strip
// only a four-part ROUTER envelope; ordinary events still require the right topic.
fn replay_frame<'a>(frames: &'a [Vec<u8>], topic: &str) -> Option<(u64, &'a [u8])> {
    let frames = if frames.len() == 4 && frames[0].is_empty() {
        &frames[1..]
    } else {
        frames
    };
    if frames.len() == 3
        && frames[0].is_empty()
        && frames[1] == (-1i64).to_be_bytes()
        && frames[2].is_empty()
    {
        return Some((u64::MAX, &frames[2]));
    }
    frame(frames, topic)
}

/// Replay is committed only after a contiguous response including the live
/// boundary. Incomplete history leaves the caller rebuilding an observed subset.
#[cfg(test)]
fn replay(
    context: &zmq::Context,
    source: &EventSource,
    cfg: &Config,
    live_seq: u64,
    stop: &AtomicBool,
) -> Option<StreamIndex> {
    let mut index = StreamIndex::default();
    replay_checked(context, source, cfg, Some(live_seq), stop, &mut index, None)
        .map_err(|(reason, _)| error(reason))
        .ok()
        .map(|()| index)
}

// Stage a contiguous suffix privately. A first-frame jump discards the old
// checkpoint: unseen removals make its hits unsafe. Only rooted branches in the
// suffix can survive. Interior gaps resume at the missing sequence on a fresh
// DEALER; the protocol has no end-sequence or batch-size request parameter.
fn replay_checked(
    context: &zmq::Context,
    source: &EventSource,
    cfg: &Config,
    live_seq: Option<u64>,
    stop: &AtomicBool,
    index: &mut StreamIndex,
    monitor: Option<&zmq::Socket>,
) -> Result<(), (&'static str, String)> {
    // A long replay must not delay invalidation when its live publisher disconnects.
    let check_connection = || {
        if monitor.is_some_and(|m| m.recv_multipart(zmq::DONTWAIT).is_ok()) {
            Err((
                "disconnected",
                "KV publisher disconnected during replay".into(),
            ))
        } else {
            Ok(())
        }
    };
    let endpoint = source
        .replay_endpoint
        .as_deref()
        .ok_or(("replay_endpoint", "No replay endpoint configured".into()))?;
    // Partial checkpoints are valid lower bounds too, provided every subsequent
    // removal is checked before publishing. Missing ancestry remains unknown.
    let mut expected = match index.last_seq {
        Some(last) => last
            .checked_add(1)
            .ok_or(("replay_sequence", "Sequence overflow".into()))?,
        None => 0,
    };
    let deadline = Instant::now() + REPLAY_TIMEOUT;
    let mut no_progress = 0;
    let mut first_event = true;
    while Instant::now() < deadline && !stop.load(Ordering::Acquire) {
        // Bound recovery bursts across workers. Waiting happens on the subscriber
        // thread, never while holding the serving index or a request lock.
        let permit = loop {
            check_connection()?;
            if let Ok(permit) = REPLAY_SLOTS.try_acquire() {
                break permit;
            }
            if stop.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err((
                    "replay_capacity",
                    "Replay cancelled or timed out waiting for capacity".into(),
                ));
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        let connect_error = |err: zmq::Error| ("replay_connect", err.to_string());
        let socket = context.socket(zmq::DEALER).map_err(connect_error)?;
        socket.set_linger(0).map_err(connect_error)?;
        // Accommodate the default worker history burst; correctness still relies
        // on sequence validation and continuation, never on queue capacity.
        socket.set_rcvhwm(10_000).map_err(connect_error)?;
        keepalive(&socket).map_err(connect_error)?;
        socket.set_rcvtimeo(250).map_err(connect_error)?;
        socket.set_sndtimeo(250).map_err(connect_error)?;
        socket
            .set_maxmsgsize(cfg.max_response_bytes as i64)
            .map_err(connect_error)?;
        socket.connect(endpoint).map_err(connect_error)?;
        socket
            .send_multipart([&[][..], &expected.to_be_bytes()[..]], 0)
            .map_err(|err| ("replay_send", err.to_string()))?;
        let from_seq = expected;
        let mut last_received = Instant::now();
        while Instant::now() < deadline && !stop.load(Ordering::Acquire) {
            let received = socket.recv_multipart(0);
            check_connection()?;
            let frames = match received {
                Ok(frames) => frames,
                Err(zmq::Error::EAGAIN) if last_received.elapsed() < REPLAY_IDLE_TIMEOUT => {
                    continue
                }
                Err(_) => break,
            };
            last_received = Instant::now();
            let (seq, payload) = replay_frame(&frames, &source.topic).ok_or_else(|| {
                (
                    "replay_frame",
                    format!(
                        "Malformed replay frames or mismatched topic; frame_lengths={:?}",
                        frames.iter().map(Vec::len).collect::<Vec<_>>()
                    ),
                )
            })?;
            if payload.is_empty() {
                return if live_seq
                    .is_none_or(|live| index.last_seq.is_some_and(|last| last >= live))
                {
                    Ok(())
                } else {
                    Err((
                        "replay_incomplete",
                        "History does not reach the live event boundary".into(),
                    ))
                };
            }
            if first_event && seq > expected {
                // The protocol cannot prove eviction versus transport loss.
                // Conservatively record unavailable history, never completeness.
                index.invalidate();
                index.history_truncated_at = Some(seq);
                tracing::info!(worker_url = %source.worker_url, requested_sequence = expected,
                    history_truncated_at = seq, "pre_kv_cache replay history prefix unavailable; rebuilding a partial index");
                expected = seq;
            }
            first_event = false;
            if seq > expected {
                // Keep earlier progress private and request exactly the missing
                // suffix. A fresh identity prevents late replies interleaving.
                break;
            }
            if seq < expected {
                return Err((
                    "replay_sequence",
                    format!("Expected sequence {expected}, received {seq}"),
                ));
            }
            match index.apply(seq, payload, &source.profile, cfg) {
                Ok(()) | Err("missing_parent" | "sparse_hash_unknown") => {}
                Err(reason) => {
                    return Err((reason, format!("Replay event sequence {seq} rejected")))
                }
            }
            expected = expected
                .checked_add(1)
                .ok_or(("replay_sequence", "Sequence overflow".into()))?;
        }
        drop(socket);
        drop(permit);
        no_progress = if expected == from_seq {
            no_progress + 1
        } else {
            0
        };
        if no_progress >= MAX_REPLAY_NO_PROGRESS {
            return Err((
                "replay_no_progress",
                "Three replay attempts received no new events".into(),
            ));
        }
        tracing::info!(worker_url = %source.worker_url, from_sequence = from_seq,
            next_sequence = expected, "pre_kv_cache replay interrupted; resuming");
        std::thread::sleep(REPLAY_RETRY_BACKOFF);
    }
    Err((
        "replay_timeout",
        "Replay cancelled or exceeded its 120-second total budget".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;
    use crate::policies::pre_kv_cache::events::tests::{config, payload, stored};

    #[test]
    fn progress_checks_require_requests_and_back_off_without_expiring_residency() {
        let now = Instant::now();
        let mut state = WorkerState::default();
        let mut recovery = ProgressRecovery::default();
        assert!(!recovery.due(&state, now));
        state.progress.first_cache_miss = Some(now);
        state.progress.last_cache_miss = Some(now);
        assert!(!recovery.due(&state, now));
        assert!(recovery.due(&state, now + PROGRESS_GRACE));
        state.progress.last_received = Some(now + PROGRESS_GRACE);
        assert!(!recovery.due(&state, now + PROGRESS_GRACE));
        state.progress.last_received = None;
        recovery.completed(true);
        assert!(!recovery.due(&state, now + PROGRESS_GRACE));
        recovery.completed(true);
        assert_eq!(recovery.failures, 2);
        assert!(
            recovery
                .next_attempt
                .unwrap()
                .duration_since(Instant::now())
                > Duration::from_secs(59)
        );
        recovery.next_attempt = None;
        assert!(!recovery.due(&state, now + PROGRESS_ACTIVITY + Duration::from_secs(1)));
    }

    #[test]
    fn progress_reply_cannot_overwrite_live_changes_or_replaced_workers() {
        let mut state = WorkerState::default();
        let (_, result) = std::sync::mpsc::channel();
        let check = ProgressCheck {
            result,
            revision: 0,
            generation: 0,
            received: 0,
            sequence: None,
            started: Instant::now(),
        };
        assert!(check.current(&state));
        state.progress.received_frames += 1;
        assert!(!check.current(&state));
        state.progress.received_frames = 0;
        state.progress.connection_generation += 1;
        assert!(!check.current(&state));
        state.progress.connection_generation = 0;
        state.layout_revision += 1;
        assert!(!check.current(&state));
        state.layout_revision = 0;
        state.stream.last_seq = Some(1);
        assert!(!check.current(&state));
    }

    #[test]
    fn progress_replay_rejects_interior_gaps_and_oversized_history() {
        // A partial reply must never reach the serving index. A retained suffix
        // may start late, but its internal sequence and response budget are strict.
        for (sequences, limit, expected) in [
            (vec![298, 300], 1024, "replay_sequence"),
            (vec![298, 298], 1024, "replay_sequence"),
            (vec![297], 1024, "replay_sequence"),
            (vec![298, 299], 64, "replay_capacity"),
        ] {
            let context = zmq::Context::new();
            let server = context.socket(zmq::ROUTER).unwrap();
            server.set_linger(0).unwrap();
            server.set_rcvtimeo(5000).unwrap();
            server.bind("tcp://127.0.0.1:*").unwrap();
            let source = EventSource {
                worker_url: "http://progress-validation".into(),
                profile: "m".into(),
                endpoint: "tcp://unused:1".into(),
                replay_endpoint: Some(server.get_last_endpoint().unwrap().unwrap()),
                topic: "kv".into(),
                speed: 1.0,
            };
            let thread = std::thread::spawn(move || {
                let request = server.recv_multipart(0).unwrap();
                for seq in sequences {
                    server
                        .send_multipart(
                            [
                                request[0].as_slice(),
                                b"",
                                b"kv",
                                &(seq as u64).to_be_bytes(),
                                b"1234567890123456789012345678901234567890",
                            ],
                            0,
                        )
                        .unwrap();
                }
                std::thread::sleep(Duration::from_millis(100));
            });
            let mut cfg = config();
            cfg.max_response_bytes = limit;
            let result = progress_frames(&source, &cfg, 298, &AtomicBool::new(false));
            assert_eq!(result.unwrap_err().0, expected);
            thread.join().unwrap();
        }
    }

    #[test]
    fn progress_suffix_applies_removals_and_truncation_never_preserves_old_hits() {
        let cfg = config();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            endpoint: "tcp://unused:1".into(),
            topic: "kv".into(),
            replay_endpoint: None,
            speed: 1.0,
        };
        let mut state = WorkerState::default();
        state
            .stream
            .apply(
                297,
                &payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]),
                "m",
                &cfg,
            )
            .unwrap();
        apply_progress(&mut state, &vec![], &source, &cfg).unwrap();
        assert_eq!(state.stream.index.matched_tokens(&[1, 2, 9], 2, "m", ""), 2);
        let removal = rmpv::Value::Array(vec![
            "BlockRemoved".into(),
            rmpv::Value::Array(vec![1.into()]),
        ]);
        apply_progress(
            &mut state,
            &vec![
                (298, payload(vec![removal])),
                (299, payload(vec![stored(2, rmpv::Value::Nil, &[3, 4])])),
            ],
            &source,
            &cfg,
        )
        .unwrap();
        assert_eq!(state.stream.index.matched_tokens(&[1, 2, 9], 2, "m", ""), 0);
        assert_eq!(state.stream.index.matched_tokens(&[3, 4, 9], 2, "m", ""), 2);
        apply_progress(
            &mut state,
            &vec![(400, payload(vec![stored(3, rmpv::Value::Nil, &[5, 6])]))],
            &source,
            &cfg,
        )
        .unwrap();
        assert_eq!(state.stream.index.matched_tokens(&[3, 4, 9], 2, "m", ""), 0);
        assert_eq!(state.stream.history_truncated_at, Some(400));
        assert_eq!(state.stream.index.matched_tokens(&[5, 6, 9], 2, "m", ""), 2);
    }

    #[tokio::test]
    async fn selected_miss_recovers_silent_history_without_a_later_live_frame() {
        // A real SUB receives seq 297, then misses the newly stored prefix.
        // Only the request hint and ROUTER history may make that prefix visible.
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let server = context.socket(zmq::ROUTER).unwrap();
        server.set_linger(0).unwrap();
        server.set_rcvtimeo(5000).unwrap();
        server.bind("tcp://127.0.0.1:*").unwrap();
        let source = EventSource {
            worker_url: "http://progress-worker".into(),
            profile: "m".into(),
            endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
            topic: "kv".into(),
            replay_endpoint: Some(server.get_last_endpoint().unwrap().unwrap()),
            speed: 1.0,
        };
        let registry = Arc::new(WorkerRegistry::new());
        let worker = Arc::new(
            BasicWorkerBuilder::new(&source.worker_url)
                .runtime_type(BackendType::Vllm)
                .build(),
        );
        registry.register(worker);
        let subscription = super::super::Subscription::new(source.clone(), None);
        let state = subscription.state.clone();
        let stop = subscription.stop.clone();
        let cfg = Arc::new(config());
        let task = subscribe(
            source.clone(),
            cfg.clone(),
            Arc::downgrade(&registry),
            state.clone(),
            stop.clone(),
            Arc::new(Diagnostics::default()),
        );
        wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
        publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &297u64.to_be_bytes(),
                    &payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]),
                ],
                0,
            )
            .unwrap();
        wait_until(|| state.lock().stream.last_seq == Some(297)).await;
        let reply = tokio::task::spawn_blocking(move || {
            let request = server.recv_multipart(0).unwrap();
            assert_eq!(request[2], 298u64.to_be_bytes());
            let body = payload(vec![stored(2, rmpv::Value::Nil, &[3, 4])]);
            server
                .send_multipart(
                    [
                        request[0].as_slice(),
                        b"",
                        b"kv",
                        &298u64.to_be_bytes(),
                        &body,
                    ],
                    0,
                )
                .unwrap();
            server
                .send_multipart(
                    [
                        request[0].as_slice(),
                        b"",
                        b"",
                        &u64::MAX.to_be_bytes(),
                        b"",
                    ],
                    0,
                )
                .unwrap();
            std::thread::sleep(Duration::from_millis(300));
        });
        subscription.request_progress_check();
        wait_until(|| state.lock().stream.last_seq == Some(298)).await;
        {
            let state = state.lock();
            assert_eq!(state.progress.received_frames, 1);
            assert_eq!(state.progress.progress_recovered_frames, 1);
            assert_eq!(state.stream.index.matched_tokens(&[3, 4, 9], 2, "m", ""), 2);
            assert_eq!(state.stream.index.matched_tokens(&[1, 2, 9], 2, "m", ""), 2);
        }
        // The subscription used during replay must stay open: a subsequent
        // live eviction must not be lost at a post-replay reconnect boundary.
        let generation = state.lock().progress.connection_generation;
        publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &299u64.to_be_bytes(),
                    &payload(vec![rmpv::Value::Array(vec![
                        "BlockRemoved".into(),
                        rmpv::Value::Array(vec![2.into()]),
                    ])]),
                ],
                0,
            )
            .unwrap();
        wait_until(|| state.lock().stream.last_seq == Some(299)).await;
        assert_eq!(state.lock().progress.connection_generation, generation);
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[3, 4, 9], 2, "m", ""),
            0
        );
        reply.await.unwrap();
        stop.store(true, Ordering::Release);
        task.await.unwrap();
    }

    #[test]
    fn temporary_replay_subset_does_not_recover_identity_warning() {
        let source = EventSource {
            worker_url: "http://identity-fixture".into(),
            profile: "m".into(),
            endpoint: "tcp://127.0.0.1:1".into(),
            topic: "kv".into(),
            replay_endpoint: None,
            speed: 1.0,
        };
        let diagnostics = Diagnostics::default();
        let mut stream = StreamIndex::default();
        let bad = super::super::events::tests::with_extras(
            stored(1, rmpv::Value::Nil, &[1, 2]),
            rmpv::Value::Array(vec![rmpv::Value::Binary(vec![9])]),
        );
        stream
            .apply(0, &payload(vec![bad]), "m", &config())
            .unwrap();
        report_isolation(&diagnostics, &source, &stream, "live");
        report_isolation(&diagnostics, &source, &stream.empty_with_layout(), "live");
        assert!(diagnostics.is_active(&source.worker_url, "unsupported_cache_identity"));
        stream
            .apply(
                1,
                &payload(vec![rmpv::Value::Array(vec!["AllBlocksCleared".into()])]),
                "m",
                &config(),
            )
            .unwrap();
        report_isolation(&diagnostics, &source, &stream, "live");
        assert!(!diagnostics.is_active(&source.worker_url, "unsupported_cache_identity"));
    }

    #[test]
    fn live_restart_is_detected_before_replay_overlap() {
        let mut sequence = LiveSequence::default();
        assert_eq!(sequence.observe(100), Delivery::Next);
        sequence.replay_highwater = Some(200);
        assert_eq!(sequence.observe(101), Delivery::Duplicate);
        assert_eq!(sequence.observe(101), Delivery::Duplicate);
        assert_eq!(sequence.observe(0), Delivery::Restart);
        assert_eq!(sequence.replay_highwater, None);
        assert_eq!(sequence.observe(1), Delivery::Next);
        assert_eq!(sequence.observe(1), Delivery::Duplicate);
    }

    #[test]
    fn tcp_keepalive_does_not_install_an_event_expiration_timer() {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::SUB).unwrap();
        keepalive(&socket).unwrap();
        assert_eq!(socket.get_tcp_keepalive().unwrap(), 1);
        assert_eq!(socket.get_tcp_keepalive_idle().unwrap(), 60);
        assert_eq!(socket.get_tcp_keepalive_intvl().unwrap(), 15);
        assert_eq!(socket.get_tcp_keepalive_cnt().unwrap(), 3);
        assert_eq!(socket.get_rcvtimeo().unwrap(), -1);
    }

    #[derive(Clone, Default)]
    struct LogWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for LogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl LogWriter {
        fn count(&self, level: &str, reason: &str) -> usize {
            String::from_utf8(self.0.lock().clone())
                .unwrap()
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter(|row| row["level"] == level && row["fields"]["reason"] == reason)
                .count()
        }
    }

    async fn wait_until(mut ready: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !ready() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("condition did not become true");
    }

    #[tokio::test]
    async fn subscriber_panic_invalidates_old_residency_and_is_supervised() {
        // Inject a setup panic before opening a socket. The supervisor must not
        // leave the previously valid index serving hits after its task dies.
        let source = EventSource {
            worker_url: "http://supervisor-fixture".into(),
            profile: "missing-profile".into(),
            endpoint: "tcp://127.0.0.1:1".into(),
            topic: "kv".into(),
            replay_endpoint: None,
            speed: 1.0,
        };
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&source.worker_url)
                .runtime_type(BackendType::Vllm)
                .build(),
        ));
        let mut previous = WorkerState::default();
        previous
            .stream
            .apply(
                0,
                &payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]),
                "m",
                &config(),
            )
            .unwrap();
        let state = Arc::new(Mutex::new(previous));
        let stop = Arc::new(AtomicBool::new(false));
        let diagnostics = Arc::new(Diagnostics::default());
        let task = subscribe(
            source.clone(),
            Arc::new(config()),
            Arc::downgrade(&registry),
            state.clone(),
            stop.clone(),
            diagnostics.clone(),
        );
        wait_until(|| diagnostics.is_active(&source.worker_url, "subscriber_task_stopped")).await;
        assert_eq!(state.lock().stream.index.len(), 0);
        assert!(!state.lock().event_state_usable());
        assert!(!state.lock().progress.running);
        stop.store(true, Ordering::Release);
        tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn silent_publisher_rejected_events_and_recovery_are_visible() {
        // Exercise asynchronous ZMQ connection, not merely a successful connect
        // call: silence warns, malformed events explain failure, valid data heals.
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
            topic: "kv".into(),
            replay_endpoint: None,
            speed: 1.0,
        };
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&source.worker_url)
                .runtime_type(BackendType::Vllm)
                .build(),
        ));
        let state = Arc::new(Mutex::new(WorkerState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let mut cfg = config();
        cfg.event_stale_ms = 400;
        let cfg = Arc::new(cfg);
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .finish();
        let task = tracing::subscriber::with_default(subscriber, || {
            subscribe(
                source.clone(),
                cfg.clone(),
                Arc::downgrade(&registry),
                state.clone(),
                stop.clone(),
                Arc::new(Diagnostics::default()),
            )
        });
        wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
        wait_until(|| output.count("WARN", "no_kv_events") == 1).await;
        tokio::time::sleep(Duration::from_millis(550)).await;
        assert_eq!(output.count("WARN", "no_kv_events"), 1);
        publisher
            .send_multipart(
                [b"kv".as_slice(), &0u64.to_be_bytes(), b"invalid-msgpack"],
                0,
            )
            .unwrap();
        wait_until(|| output.count("WARN", "msgpack") + output.count("WARN", "trailing_bytes") > 0)
            .await;
        let valid = payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]);
        publisher
            .send_multipart([b"kv".as_slice(), &1u64.to_be_bytes(), &valid], 0)
            .unwrap();
        wait_until(|| output.count("INFO", "no_kv_events") == 1).await;
        assert_eq!(state.lock().stream.index.len(), 1);
        tokio::time::sleep(Duration::from_millis(650)).await;
        assert_eq!(output.count("WARN", "event_stale"), 0);
        assert!(state.lock().event_state_usable());
        // Idle time preserves the parent's identity and usability. Replayed
        // duplicates must not change the observation timestamp.
        state.lock().stream.last_event = Some(Instant::now() - Duration::from_secs(18 * 3600));
        let idle_at = state.lock().stream.last_event;
        assert!(state.lock().event_state_usable());
        let received_before = state.lock().progress.received_frames;
        let duplicates_before = state.lock().progress.duplicate_frames;
        publisher
            .send_multipart([b"kv".as_slice(), &1u64.to_be_bytes(), &valid], 0)
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(state.lock().stream.last_event, idle_at);
        assert_eq!(state.lock().progress.received_frames, received_before + 1);
        assert_eq!(
            state.lock().progress.duplicate_frames,
            duplicates_before + 1
        );
        let valid = payload(vec![stored(2, 1.into(), &[3, 4])]);
        publisher
            .send_multipart([b"kv".as_slice(), &2u64.to_be_bytes(), &valid], 0)
            .unwrap();
        wait_until(|| state.lock().stream.last_seq == Some(2)).await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""),
            4
        );

        // An eviction after another idle period still removes the old hit.
        state.lock().stream.last_event = Some(Instant::now() - Duration::from_secs(1));
        let removal = payload(vec![rmpv::Value::Array(vec![
            "BlockRemoved".into(),
            rmpv::Value::Array(vec![1.into()]),
        ])]);
        publisher
            .send_multipart([b"kv".as_slice(), &3u64.to_be_bytes(), &removal], 0)
            .unwrap();
        wait_until(|| state.lock().stream.last_seq == Some(3)).await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""),
            0
        );

        // An unsupported format is sticky, even when later heartbeats decode.
        let mut unsupported = stored(2, rmpv::Value::Nil, &[1, 2])
            .as_array()
            .unwrap()
            .clone();
        unsupported.resize(10, rmpv::Value::Nil);
        unsupported[9] = 1.into();
        let body = payload(vec![rmpv::Value::Array(unsupported)]);
        publisher
            .send_multipart([b"kv".as_slice(), &4u64.to_be_bytes(), &body], 0)
            .unwrap();
        wait_until(|| output.count("WARN", "unsupported_group") == 1).await;
        publisher
            .send_multipart([b"kv".as_slice(), &5u64.to_be_bytes(), &valid], 0)
            .unwrap();
        wait_until(|| output.count("WARN", "unsupported_stream") == 1).await;
        assert_eq!(output.count("INFO", "unsupported_group"), 0);
        stop.store(true, Ordering::Release);
        task.await.unwrap();
    }

    #[test]
    fn partial_events_recover_receipt_but_keep_index_failure_visible() {
        // Receipt and complete indexing are separate health signals. A batch
        // with unresolved ancestry cannot be called a silent publisher.
        let cfg = config();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            endpoint: "tcp://worker:5557".into(),
            topic: "kv".into(),
            replay_endpoint: None,
            speed: 1.0,
        };
        let diagnostics = Diagnostics::default();
        let mut health = EventHealth::new();
        health.started -= Duration::from_millis(cfg.event_stale_ms + 1);
        health.check(&diagnostics, &source, &cfg);
        assert!(diagnostics
            .reasons(&source.worker_url)
            .contains(&"no_kv_events"));
        health.warn(
            &diagnostics,
            &source,
            &cfg,
            "missing_parent",
            "missing history",
        );
        let at = Instant::now();
        health.partially_indexed(&diagnostics, &source, at);
        health.partially_indexed(&diagnostics, &source, at);
        health.check(&diagnostics, &source, &cfg);
        assert_eq!(
            diagnostics.reasons(&source.worker_url),
            vec!["missing_parent"]
        );
        health.accepted(&diagnostics, &source, at + Duration::from_millis(1), false);
        assert_eq!(
            diagnostics.reasons(&source.worker_url),
            vec!["missing_parent"]
        );
        health.accepted(&diagnostics, &source, at + Duration::from_millis(2), true);
        assert!(diagnostics.reasons(&source.worker_url).is_empty());
    }

    #[tokio::test]
    async fn failed_load_polls_warn_and_recover_after_valid_metrics() {
        // Preserve the previous snapshot/fallback semantics while making HTTP
        // failure and later recovery observable without per-request polling.
        let ready = Arc::new(AtomicBool::new(false));
        let response_ready = ready.clone();
        let app = axum::Router::new().route(
            "/metrics",
            axum::routing::get(move || {
                let ready = response_ready.clone();
                async move {
                    if ready.load(Ordering::Acquire) {
                        (
                            http::StatusCode::OK,
                            "vllm:num_requests_running 2\nvllm:num_requests_waiting 0\n",
                        )
                    } else {
                        (http::StatusCode::SERVICE_UNAVAILABLE, "")
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&url)
                .runtime_type(BackendType::Vllm)
                .build(),
        ));
        let source = EventSource {
            worker_url: url.clone(),
            profile: "m".into(),
            endpoint: "tcp://unused:1".into(),
            topic: "kv".into(),
            replay_endpoint: None,
            speed: 1.0,
        };
        let loads = Loads::default();
        let stop = Arc::new(AtomicBool::new(false));
        let mut cfg = config();
        cfg.load_poll_ms = 50;
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .finish();
        let task = tracing::subscriber::with_default(subscriber, || {
            poll_load(
                source,
                Arc::new(cfg),
                Arc::downgrade(&registry),
                reqwest::Client::new(),
                loads.clone(),
                stop.clone(),
                Arc::new(Diagnostics::default()),
            )
        });
        wait_until(|| output.count("WARN", "load_metrics_unavailable") == 1).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(output.count("WARN", "load_metrics_unavailable"), 1);
        assert!(loads.lock()[&url].snapshot.is_none());
        ready.store(true, Ordering::Release);
        wait_until(|| output.count("INFO", "load_metrics_unavailable") == 1).await;
        assert_eq!(loads.lock()[&url].snapshot.as_ref().unwrap().running, 2.0);
        stop.store(true, Ordering::Release);
        task.await.unwrap();
        server.abort();
    }

    #[test]
    fn replay_failure_backoff_is_bounded_and_timer_requires_pending_work() {
        let mut recovery = ReplayRecovery::default();
        assert!(recovery.ready());
        assert!(!recovery.retry_due());
        recovery.pending = Some(StreamIndex::default());
        for seconds in [30, 60, 120, 240, 300, 300] {
            assert_eq!(recovery.failed(), seconds);
            assert!(!recovery.ready());
            recovery.next_attempt = Some(Instant::now() - Duration::from_secs(1));
            assert!(recovery.ready());
            assert!(recovery.retry_due());
        }
        recovery = ReplayRecovery::default();
        assert!(recovery.ready());
        assert_eq!(recovery.failures, 0);
    }

    // Stop the blocking subscriber even if an assertion fails, so a regression
    // fails the test instead of hanging Tokio's runtime shutdown.
    struct StopOnDrop(Arc<AtomicBool>);
    impl Drop for StopOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    async fn replay_request(socket: &zmq::Socket, expected: u64) -> Vec<u8> {
        let mut frames = None;
        wait_until(|| {
            frames = socket.recv_multipart(zmq::DONTWAIT).ok();
            frames.is_some()
        })
        .await;
        let frames = frames.unwrap();
        assert_eq!(frames[2], expected.to_be_bytes());
        frames[0].clone()
    }

    fn replay_reply(socket: &zmq::Socket, identity: &[u8], events: &[(u64, Vec<u8>)]) {
        for (seq, body) in events {
            socket
                .send_multipart([identity, &[], b"kv", &seq.to_be_bytes(), body], 0)
                .unwrap();
        }
        // vLLM terminates replay with an empty topic and signed -1 sequence.
        socket
            .send_multipart([identity, &[], &[], &(-1i64).to_be_bytes(), &[]], 0)
            .unwrap();
    }

    #[tokio::test]
    async fn layout_rebinding_and_first_nonzero_root_do_not_start_replay() {
        // Discovery and automatic layout detection may bind twice. Neither
        // transition nor a valid late-joined root needs a scan from sequence zero.
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        // XPUB normally coalesces identical topics while old/new pipes overlap.
        publisher.set_xpub_verbose(true).unwrap();
        let history = context.socket(zmq::ROUTER).unwrap();
        for socket in [&publisher, &history] {
            socket.set_linger(0).unwrap();
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            topic: "kv".into(),
            speed: 1.0,
            endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&source.worker_url)
                .runtime_type(BackendType::Vllm)
                .build(),
        ));
        let state = Arc::new(Mutex::new(WorkerState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let _guard = StopOnDrop(stop.clone());
        let task = subscribe(
            source,
            Arc::new(config()),
            Arc::downgrade(&registry),
            state.clone(),
            stop.clone(),
            Arc::new(Diagnostics::default()),
        );
        wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
        state.lock().layout_revision += 1;
        wait_until(|| {
            publisher
                .recv_multipart(zmq::DONTWAIT)
                .is_ok_and(|frames| frames[0].first() == Some(&1))
        })
        .await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(history.recv_multipart(zmq::DONTWAIT).is_err());
        let root = payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]);
        publisher
            .send_multipart([b"kv".as_slice(), &5620u64.to_be_bytes(), &root], 0)
            .unwrap();
        wait_until(|| state.lock().stream.last_seq == Some(5620)).await;
        assert!(!state.lock().stream.history_complete);
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3], 2, "m", ""),
            2
        );
        assert!(history.recv_multipart(zmq::DONTWAIT).is_err());
        stop.store(true, Ordering::Release);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn lazy_replay_repairs_gaps_and_accepts_truncated_history_without_looping() {
        // Startup is quiet; real gaps recover removals. Truncated history keeps
        // only rooted branches and does not repeatedly scan unavailable ancestors.
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        let history = context.socket(zmq::ROUTER).unwrap();
        for socket in [&publisher, &history] {
            socket.set_linger(0).unwrap();
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            topic: "kv".into(),
            speed: 1.0,
            endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&source.worker_url)
                .runtime_type(BackendType::Vllm)
                .build(),
        ));
        let state = Arc::new(Mutex::new(WorkerState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let _guard = StopOnDrop(stop.clone());
        let diagnostics = Arc::new(Diagnostics::default());
        let task = subscribe(
            source.clone(),
            Arc::new(config()),
            Arc::downgrade(&registry),
            state.clone(),
            stop.clone(),
            diagnostics.clone(),
        );
        wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
        let publish = |seq: u64, body: &[u8]| {
            publisher
                .send_multipart([b"kv".as_slice(), &seq.to_be_bytes(), body], 0)
                .unwrap();
        };
        let root = payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(history.recv_multipart(zmq::DONTWAIT).is_err());
        publish(0, &root);
        wait_until(|| state.lock().stream.last_seq == Some(0)).await;
        assert!(history.recv_multipart(zmq::DONTWAIT).is_err());

        let removal = payload(vec![rmpv::Value::Array(vec![
            "BlockRemoved".into(),
            rmpv::Value::Array(vec![1.into()]),
        ])]);
        let next_root = payload(vec![stored(20, rmpv::Value::Nil, &[5, 6])]);
        publish(2, &next_root); // event 1 was lost; history before 1 is unavailable
        let identity = replay_request(&history, 1).await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3], 2, "m", ""),
            0
        );
        assert!(!state.lock().stream.history_complete);
        replay_reply(
            &history,
            &identity,
            &[(1, removal.clone()), (2, next_root.clone())],
        );
        wait_until(|| state.lock().stream.history_complete).await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3], 2, "m", ""),
            0
        );
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[5, 6, 7], 2, "m", ""),
            2
        );
        assert!(diagnostics.reasons(&source.worker_url).is_empty());

        // Lose ancestry without losing sequence: this previously never replayed.
        state.lock().stream.index.clear();
        let child = payload(vec![stored(21, 20.into(), &[7, 8])]);
        publish(3, &child);
        let identity = replay_request(&history, 0).await;
        assert_eq!(state.lock().stream.last_error, Some("missing_parent"));
        replay_reply(
            &history,
            &identity,
            &[(0, root), (1, removal), (2, next_root), (3, child.clone())],
        );
        wait_until(|| state.lock().stream.history_complete).await;
        assert_eq!(state.lock().stream.last_error, None);
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[5, 6, 7, 8, 9], 2, "m", ""),
            4
        );
        assert!(diagnostics.reasons(&source.worker_url).is_empty());

        // Missing ancestry remains visible, but a valid truncated suffix is not
        // a replay failure and must not repeatedly request the discarded prefix.
        state.lock().stream.index.clear();
        let partial = payload(vec![
            stored(30, rmpv::Value::Nil, &[9, 10]),
            stored(22, 21.into(), &[11, 12]),
        ]);
        publish(4, &partial);
        let identity = replay_request(&history, 0).await;
        replay_reply(&history, &identity, &[(3, child), (4, partial)]);
        wait_until(|| state.lock().stream.history_truncated_at == Some(3)).await;
        assert!(!diagnostics
            .reasons(&source.worker_url)
            .contains(&"replay_gap"));
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[9, 10, 11], 2, "m", ""),
            2
        );
        assert!(!state.lock().stream.history_complete);
        publish(5, &payload(vec![stored(31, 30.into(), &[11, 12])]));
        wait_until(|| state.lock().stream.last_seq == Some(5)).await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[9, 10, 11, 12, 13], 2, "m", ""),
            4
        );
        assert_eq!(state.lock().stream.last_error, None);
        assert!(state.lock().stream.incomplete());
        assert!(history.recv_multipart(zmq::DONTWAIT).is_err());

        // An authoritative cache clear is also evidence of complete recovery.
        publish(
            6,
            &payload(vec![rmpv::Value::Array(vec!["AllBlocksCleared".into()])]),
        );
        wait_until(|| state.lock().stream.last_seq == Some(6)).await;
        assert!(state.lock().stream.history_complete);
        assert!(diagnostics.reasons(&source.worker_url).is_empty());

        // Sequence zero establishes a new history directly; no replay or old
        // branch may survive this reset.
        let restarted = payload(vec![stored(99, rmpv::Value::Nil, &[90, 91])]);
        publish(0, &restarted);
        wait_until(|| state.lock().stream.last_seq == Some(0)).await;
        assert!(history.recv_multipart(zmq::DONTWAIT).is_err());
        assert_eq!(state.lock().stream.history_truncated_at, None);
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[9, 10, 11], 2, "m", ""),
            0
        );
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[90, 91, 92], 2, "m", ""),
            2
        );

        // Disconnect while replay is blocked: invalidate before replay responds,
        // then discard its late reply rather than reviving the old publisher.
        publish(2, &payload(vec![]));
        let identity = replay_request(&history, 1).await;
        drop(publisher);
        wait_until(|| {
            diagnostics
                .reasons(&source.worker_url)
                .contains(&"disconnected")
        })
        .await;
        assert!(state.lock().stream.last_event.is_none());
        assert_eq!(state.lock().stream.index.len(), 0);
        replay_reply(
            &history,
            &identity,
            &[(1, payload(vec![])), (2, payload(vec![]))],
        );
        stop.store(true, Ordering::Release);
        task.await.unwrap();
    }

    #[test]
    fn truncated_replay_discards_old_hits_and_ignores_orphans() {
        // The unavailable interval may contain removals. A surviving old hit
        // would be a false positive; only new, fully rooted branches are usable.
        let context = zmq::Context::new();
        let history = context.socket(zmq::ROUTER).unwrap();
        history.set_linger(0).unwrap();
        history.set_rcvtimeo(3000).unwrap();
        history.bind("tcp://127.0.0.1:*").unwrap();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            topic: "kv".into(),
            endpoint: "tcp://unused:1".into(),
            speed: 1.0,
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let server = std::thread::spawn(move || {
            let request = history.recv_multipart(0).unwrap();
            assert_eq!(request[2], 1u64.to_be_bytes());
            replay_reply(
                &history,
                &request[0],
                &[
                    (5417, payload(vec![stored(2, 1.into(), &[3, 4])])),
                    (5418, payload(vec![stored(3, rmpv::Value::Nil, &[5, 6])])),
                ],
            );
            std::thread::sleep(Duration::from_millis(100));
        });
        let mut index = StreamIndex::default();
        index
            .apply(
                0,
                &payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]),
                "m",
                &config(),
            )
            .unwrap();
        replay_checked(
            &context,
            &source,
            &config(),
            Some(5418),
            &AtomicBool::new(false),
            &mut index,
            None,
        )
        .unwrap();
        assert_eq!(index.history_truncated_at, Some(5417));
        assert!(!index.history_complete);
        assert_eq!(index.last_error, None);
        assert!(index.incomplete());
        assert_eq!(index.index.matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""), 0);
        assert_eq!(index.index.matched_tokens(&[5, 6, 7], 2, "m", ""), 2);
        server.join().unwrap();
    }

    #[test]
    fn interior_replay_gap_resumes_before_applying_later_events() {
        // A dropped deletion must be read on a fresh connection before later
        // stores can be committed. An old response's end marker cannot finish it.
        let context = zmq::Context::new();
        let history = context.socket(zmq::ROUTER).unwrap();
        history.set_linger(0).unwrap();
        history.set_rcvtimeo(3000).unwrap();
        history.bind("tcp://127.0.0.1:*").unwrap();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            topic: "kv".into(),
            endpoint: "tcp://unused:1".into(),
            speed: 1.0,
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let server = std::thread::spawn(move || {
            let first = history.recv_multipart(0).unwrap();
            let later = payload(vec![stored(3, rmpv::Value::Nil, &[5, 6])]);
            replay_reply(
                &history,
                &first[0],
                &[
                    (0, payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])])),
                    (2, later.clone()),
                ],
            );
            let resumed = history.recv_multipart(0).unwrap();
            assert_eq!(resumed[2], 1u64.to_be_bytes());
            assert_ne!(resumed[0], first[0]);
            replay_reply(
                &history,
                &resumed[0],
                &[
                    (
                        1,
                        payload(vec![rmpv::Value::Array(vec![
                            "BlockRemoved".into(),
                            rmpv::Value::Array(vec![1.into()]),
                        ])]),
                    ),
                    (2, later),
                ],
            );
            std::thread::sleep(Duration::from_millis(100));
        });
        let mut index = StreamIndex::default();
        replay_checked(
            &context,
            &source,
            &config(),
            Some(2),
            &AtomicBool::new(false),
            &mut index,
            None,
        )
        .unwrap();
        assert!(index.history_complete);
        assert_eq!(index.history_truncated_at, None);
        assert_eq!(index.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 0);
        assert_eq!(index.index.matched_tokens(&[5, 6, 7], 2, "m", ""), 2);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn timer_retry_preserves_private_progress_without_new_live_events() {
        // After an incomplete response, the publisher goes idle. Recovery must
        // still resume after its cooldown, without exposing a missed deletion.
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        let history = context.socket(zmq::ROUTER).unwrap();
        for socket in [&publisher, &history] {
            socket.set_linger(0).unwrap();
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            topic: "kv".into(),
            speed: 1.0,
            endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&source.worker_url)
                .runtime_type(BackendType::Vllm)
                .build(),
        ));
        let state = Arc::new(Mutex::new(WorkerState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let _guard = StopOnDrop(stop.clone());
        let diagnostics = Arc::new(Diagnostics::default());
        let task = subscribe(
            source.clone(),
            Arc::new(config()),
            Arc::downgrade(&registry),
            state.clone(),
            stop.clone(),
            diagnostics.clone(),
        );
        wait_until(|| publisher.recv_multipart(zmq::DONTWAIT).is_ok()).await;
        let root = payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]);
        publisher
            .send_multipart([b"kv".as_slice(), &0u64.to_be_bytes(), &root], 0)
            .unwrap();
        wait_until(|| state.lock().stream.last_seq == Some(0)).await;
        let later = payload(vec![stored(3, rmpv::Value::Nil, &[5, 6])]);
        publisher
            .send_multipart([b"kv".as_slice(), &2u64.to_be_bytes(), &later], 0)
            .unwrap();
        let identity = replay_request(&history, 1).await;
        replay_reply(
            &history,
            &identity,
            &[(
                1,
                payload(vec![rmpv::Value::Array(vec![
                    "BlockRemoved".into(),
                    rmpv::Value::Array(vec![1.into()]),
                ])]),
            )],
        );
        wait_until(|| {
            diagnostics
                .reasons(&source.worker_url)
                .contains(&"replay_incomplete")
        })
        .await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3], 2, "m", ""),
            0
        );
        let retry = tokio::time::timeout(Duration::from_secs(35), async {
            loop {
                if let Ok(frames) = history.recv_multipart(zmq::DONTWAIT) {
                    break frames;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("timer retry did not run");
        assert_eq!(retry[2], 2u64.to_be_bytes());
        assert_ne!(retry[0], identity);
        replay_reply(&history, &retry[0], &[(2, later)]);
        wait_until(|| state.lock().stream.history_complete).await;
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[1, 2, 3], 2, "m", ""),
            0
        );
        assert_eq!(
            state
                .lock()
                .stream
                .index
                .matched_tokens(&[5, 6, 7], 2, "m", ""),
            2
        );
        assert!(diagnostics.reasons(&source.worker_url).is_empty());
        stop.store(true, Ordering::Release);
        task.await.unwrap();
    }

    #[test]
    fn interrupted_replay_resumes_without_losing_staged_ancestry() {
        // A stalled response must resume at the next sequence, not discard the
        // prefix or hit the old two-second total deadline. Include a deletion.
        let context = zmq::Context::new();
        let history = context.socket(zmq::ROUTER).unwrap();
        history.set_linger(0).unwrap();
        history.set_rcvtimeo(5000).unwrap();
        history.bind("tcp://127.0.0.1:*").unwrap();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            endpoint: "tcp://unused:1".into(),
            topic: "kv".into(),
            speed: 1.0,
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let server = std::thread::spawn(move || {
            let first = history.recv_multipart(0).unwrap();
            assert_eq!(first[2], 0u64.to_be_bytes());
            let root = payload(vec![stored(1, rmpv::Value::Nil, &[1, 2])]);
            history
                .send_multipart(
                    [first[0].as_slice(), &[], b"kv", &0u64.to_be_bytes(), &root],
                    0,
                )
                .unwrap();
            // Omit the terminator to simulate an interrupted replay response.
            let resumed = history.recv_multipart(0).unwrap();
            assert_eq!(resumed[2], 1u64.to_be_bytes());
            replay_reply(
                &history,
                &resumed[0],
                &[
                    (1, payload(vec![stored(2, 1.into(), &[3, 4])])),
                    (
                        2,
                        payload(vec![rmpv::Value::Array(vec![
                            "BlockRemoved".into(),
                            rmpv::Value::Array(vec![1.into()]),
                        ])]),
                    ),
                ],
            );
            std::thread::sleep(Duration::from_millis(100));
        });
        let mut index = StreamIndex::default();
        replay_checked(
            &context,
            &source,
            &config(),
            Some(2),
            &AtomicBool::new(false),
            &mut index,
            None,
        )
        .unwrap();
        assert_eq!(index.last_seq, Some(2));
        assert!(index.history_complete);
        assert_eq!(index.index.len(), 1);
        assert_eq!(index.index.matched_tokens(&[1, 2, 3, 4, 5], 2, "m", ""), 0);
        server.join().unwrap();
    }

    #[test]
    fn replay_control_frames_do_not_relax_live_event_validation() {
        // Both DEALER envelope variants accept the vLLM control marker; the
        // live parser and replay data path must still reject wrong-topic events.
        let end = vec![vec![], (-1i64).to_be_bytes().to_vec(), vec![]];
        for enveloped in [false, true] {
            let mut frames = end.clone();
            if enveloped {
                frames.insert(0, vec![]);
            }
            assert_eq!(replay_frame(&frames, "kv"), Some((u64::MAX, &[][..])));
            assert!(frame(&frames, "kv").is_none());
        }
        let mut not_end = end.clone();
        not_end[2] = payload(vec![]);
        assert!(replay_frame(&not_end, "kv").is_none());
        not_end = end.clone();
        not_end[1] = vec![255];
        assert!(replay_frame(&not_end, "kv").is_none());
        not_end = end;
        not_end[0] = b"other".to_vec();
        assert!(replay_frame(&not_end, "kv").is_none());
        let body = payload(vec![]);
        let event = vec![b"kv".to_vec(), 7u64.to_be_bytes().to_vec(), body.clone()];
        assert_eq!(replay_frame(&event, "kv"), Some((7, body.as_slice())));
        assert!(replay_frame(&event, "other").is_none());
    }

    #[test]
    fn empty_replay_is_allowed_but_never_invents_residency() {
        // Accept vLLM's end marker and the existing topic-tagged variant. Neither
        // an empty history nor a malformed terminator can invent cache residency.
        for (topic, sequence, accepted) in [
            (b"kv".as_slice(), 0u64, true),
            (b"".as_slice(), u64::MAX, true),
            (b"".as_slice(), 0, false),
            (b"other".as_slice(), 0, false),
        ] {
            let context = zmq::Context::new();
            let history = context.socket(zmq::ROUTER).unwrap();
            history.set_linger(0).unwrap();
            history.set_rcvtimeo(3000).unwrap();
            history.bind("tcp://127.0.0.1:*").unwrap();
            let source = EventSource {
                worker_url: "http://worker".into(),
                profile: "m".into(),
                endpoint: "tcp://unused:1".into(),
                topic: "kv".into(),
                speed: 1.0,
                replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
            };
            let server = std::thread::spawn(move || {
                let request = history.recv_multipart(0).unwrap();
                history
                    .send_multipart(
                        [
                            request[0].as_slice(),
                            &[],
                            topic,
                            &sequence.to_be_bytes(),
                            &[],
                        ],
                        0,
                    )
                    .unwrap();
                std::thread::sleep(Duration::from_millis(100));
            });
            let mut index = StreamIndex::default();
            let result = replay_checked(
                &context,
                &source,
                &config(),
                None,
                &AtomicBool::new(false),
                &mut index,
                None,
            );
            assert_eq!(result.is_ok(), accepted);
            assert!(index.last_event.is_none());
            assert!(!index.history_complete);
            assert_eq!(index.index.len(), 0);
            server.join().unwrap();
        }
    }

    #[test]
    fn silent_replay_stops_after_three_attempts() {
        // A missing replay publisher must not consume the full two-minute
        // progress budget, nor retry indefinitely in the background.
        let context = zmq::Context::new();
        let history = context.socket(zmq::ROUTER).unwrap();
        history.set_linger(0).unwrap();
        history.set_rcvtimeo(4000).unwrap();
        history.bind("tcp://127.0.0.1:*").unwrap();
        let source = EventSource {
            worker_url: "http://worker".into(),
            profile: "m".into(),
            endpoint: "tcp://unused:1".into(),
            topic: "kv".into(),
            speed: 1.0,
            replay_endpoint: Some(history.get_last_endpoint().unwrap().unwrap()),
        };
        let server = std::thread::spawn(move || {
            for _ in 0..3 {
                let request = history.recv_multipart(0).unwrap();
                assert_eq!(request[2], 0u64.to_be_bytes());
            }
        });
        let result = replay_checked(
            &context,
            &source,
            &config(),
            None,
            &AtomicBool::new(false),
            &mut StreamIndex::default(),
            None,
        );
        assert_eq!(result.unwrap_err().0, "replay_no_progress");
        server.join().unwrap();
    }

    #[test]
    fn replay_requires_contiguous_history_through_the_live_boundary() {
        // A terminator can finish a partial suffix, but cannot make it complete
        // or finish before the triggering live sequence.
        for (first, live_seq, accepted) in [(0u64, 1, true), (1, 1, true), (0, 2, false)] {
            let context = zmq::Context::new();
            let publisher = context.socket(zmq::ROUTER).unwrap();
            publisher.set_linger(0).unwrap();
            publisher.set_rcvtimeo(3000).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            let server = std::thread::spawn(move || {
                let request = publisher.recv_multipart(0).unwrap();
                assert_eq!(request[2], 0u64.to_be_bytes());
                for seq in first..=1 {
                    let body = payload(vec![stored(seq + 1, rmpv::Value::Nil, &[1, 2])]);
                    publisher
                        .send_multipart(
                            [request[0].as_slice(), &[], b"kv", &seq.to_be_bytes(), &body],
                            0,
                        )
                        .unwrap();
                }
                publisher
                    .send_multipart(
                        [request[0].as_slice(), &[], &[], &(-1i64).to_be_bytes(), &[]],
                        0,
                    )
                    .unwrap();
                // Wait for the caller before dropping the ROUTER's outgoing queue.
                std::thread::sleep(Duration::from_millis(100));
            });
            let source = EventSource {
                worker_url: "http://worker".into(),
                profile: "m".into(),
                endpoint: "tcp://unused:1".into(),
                topic: "kv".into(),
                replay_endpoint: Some(endpoint),
                speed: 1.0,
            };
            let result = replay(
                &context,
                &source,
                &config(),
                live_seq,
                &AtomicBool::new(false),
            );
            assert_eq!(result.is_some(), accepted);
            if let Some(state) = result {
                assert_eq!(state.last_seq, Some(1));
                assert_eq!(state.history_complete, first == 0);
                assert_eq!(state.history_truncated_at, (first > 0).then_some(first));
                assert_eq!(state.index.matched_tokens(&[1, 2, 3], 2, "m", ""), 2);
            }
            server.join().unwrap();
        }
    }
}

pub fn poll_load(
    source: EventSource,
    cfg: Arc<Config>,
    registry: Weak<WorkerRegistry>,
    client: reqwest::Client,
    loads: Loads,
    stop: Arc<AtomicBool>,
    diagnostics: Arc<Diagnostics>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(cfg.load_poll_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        while !stop.load(Ordering::Acquire) {
            interval.tick().await;
            if stop.load(Ordering::Acquire) {
                break;
            }
            let Some(registry) = registry.upgrade() else {
                break;
            };
            let Some(worker) = registry.get_by_url(&source.worker_url) else {
                loads.lock().remove(&source.worker_url);
                continue;
            };
            if !matches!(worker.metadata().runtime_type, BackendType::Vllm) {
                loads.lock().remove(&source.worker_url);
                continue;
            }
            {
                let mut loads = loads.lock();
                if stop.load(Ordering::Acquire) {
                    break;
                }
                loads
                    .entry(source.worker_url.clone())
                    .or_default()
                    .bind(&worker);
            }
            let poll_started = Instant::now();
            let mut request = client.get(format!(
                "{}/metrics",
                source.worker_url.trim_end_matches('/')
            ));
            if let Some(key) = worker.api_key().as_deref() {
                request = request.bearer_auth(key);
            }
            // Keep failure details out of request handling but visible in bounded
            // background WARNs; retain the existing stale-snapshot fallback.
            let sample =
                tokio::time::timeout(Duration::from_millis(cfg.load_poll_ms.min(3000)), async {
                    let response = request.send().await.map_err(|err| err.to_string())?;
                    if !response.status().is_success() {
                        return Err(format!("HTTP {}", response.status()));
                    }
                    let body = limited_body(response, cfg.max_response_bytes)
                        .await
                        .map_err(|err| err.to_string())?;
                    let body = std::str::from_utf8(&body).map_err(|err| err.to_string())?;
                    parse_metrics(body).ok_or_else(|| {
                        "Missing or invalid vLLM running/waiting request gauges".to_owned()
                    })
                })
                .await
                .unwrap_or_else(|err| Err(format!("Metrics request timed out: {err}")));
            let current = registry
                .get_by_url(&source.worker_url)
                .is_some_and(|w| Arc::ptr_eq(&w, &worker));
            if !current {
                let mut loads = loads.lock();
                if stop.load(Ordering::Acquire) {
                    break;
                }
                loads.remove(&source.worker_url);
                continue;
            }
            if let Ok(sample) = sample {
                if sample.kv_usage.is_none() {
                    diagnostics.warn(&source.worker_url, "kv_usage_unavailable", "KV occupancy gauge missing or invalid; pressure contributes zero while running/waiting request gauges remain usable");
                } else {
                    diagnostics.recovered(&source.worker_url, "kv_usage_unavailable");
                }
                let mut loads = loads.lock();
                // A retired subscription must not recreate its load after cleanup.
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let load = loads.entry(source.worker_url.clone()).or_default();
                load.bind(&worker);
                load.snapshot = Some(sample);
                // Keep dispatches racing this poll; the engine may not have seen them.
                load.credits
                    .retain(|_, credit| credit.created >= poll_started);
                diagnostics.recovered(&source.worker_url, "load_metrics_unavailable");
            } else if let Err(detail) = sample {
                metrics::counter!("pre_kv_cache_load_errors_total").increment(1);
                diagnostics.warn(
                    &source.worker_url,
                    "load_metrics_unavailable",
                    &format!("{detail}; endpoint={}/metrics profile={}; load scoring will use its existing fallback when the last sample expires", source.worker_url.trim_end_matches('/'), source.profile),
                );
            }
        }
    }.with_current_subscriber())
}
