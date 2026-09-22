//! Bounded background I/O. No subscription or metrics request runs on selection.

use parking_lot::Mutex;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use super::{
    config::{Config, EventSource},
    events::StreamIndex,
    load::{parse_metrics, Loads},
    render::limited_body,
    WorkerState,
};
use crate::core::{BackendType, WorkerRegistry};

pub fn subscribe(
    source: EventSource,
    cfg: Arc<Config>,
    registry: Weak<WorkerRegistry>,
    state: Arc<Mutex<WorkerState>>,
    stop: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    // libzmq receives on a dedicated blocking thread with a finite timeout.
    tokio::task::spawn_blocking(move || {
        let context = zmq::Context::new();
        let mut socket: Option<zmq::Socket> = None;
        let mut monitor: Option<zmq::Socket> = None;
        let mut replay_highwater: Option<u64> = None;
        while !stop.load(Ordering::Acquire) {
            let Some(registry) = registry.upgrade() else {
                break;
            };
            let worker = registry
                .get_by_url(&source.worker_url)
                .filter(|w| matches!(w.metadata().runtime_type, BackendType::Vllm));
            {
                let mut state = state.lock();
                let same = match (&state.binding, &worker) {
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    (None, None) => true,
                    _ => false,
                };
                if !same {
                    socket = None;
                    monitor = None;
                    state.stream = StreamIndex::default();
                    state.binding = worker.clone();
                    replay_highwater = None;
                }
            }
            if worker.is_none() {
                std::thread::sleep(Duration::from_millis(250));
                continue;
            }
            if socket.is_none() {
                replay_highwater = None;
                let connected = (|| -> Result<(zmq::Socket, zmq::Socket), zmq::Error> {
                    let s = context.socket(zmq::SUB)?;
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
                        socket = Some(s);
                        monitor = Some(m);
                    }
                    Err(_) => {
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
                state.lock().stream.invalidate();
                socket = None;
                monitor = None;
                error("disconnected");
                continue;
            }
            let frames = match socket.as_ref().unwrap().recv_multipart(0) {
                Ok(frames) => frames,
                Err(zmq::Error::EAGAIN) => continue,
                Err(_) => {
                    socket = None;
                    monitor = None;
                    state.lock().stream.invalidate();
                    continue;
                }
            };
            // Discard queued frames from a connection that closed during recv.
            if monitor
                .as_ref()
                .is_some_and(|m| m.recv_multipart(zmq::DONTWAIT).is_ok())
            {
                state.lock().stream.invalidate();
                socket = None;
                monitor = None;
                continue;
            }
            let Some((seq, payload)) = frame(&frames, &source.topic) else {
                state.lock().stream.invalidate();
                error("event_frame");
                continue;
            };
            // Replay can pass queued live frames. Ignore that overlap only on
            // this connection; a disconnect clears the high-water mark.
            if replay_highwater.is_some_and(|last| seq <= last) {
                continue;
            }
            replay_highwater = None;
            let need_replay = {
                let s = state.lock();
                !s.stream.unsupported
                    && s.stream
                        .last_seq
                        .is_none_or(|last| seq != last && last.checked_add(1) != Some(seq))
            };
            if need_replay && source.replay_endpoint.is_some() {
                state.lock().stream.invalidate();
                if let Some(rebuilt) = replay(&context, &source, &cfg, seq, &stop) {
                    let mut state = state.lock();
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    replay_highwater = rebuilt.last_seq;
                    state.stream = rebuilt;
                    continue;
                }
            }
            let mut s = state.lock();
            if stop.load(Ordering::Acquire) {
                break;
            }
            if s.stream
                .last_event
                .is_some_and(|t| t.elapsed() > Duration::from_millis(cfg.event_stale_ms))
            {
                s.stream.invalidate();
            }
            if let Err(reason) = s.stream.apply(seq, payload, &source.profile, &cfg) {
                error(reason);
            }
            metrics::gauge!("pre_kv_cache_index_blocks", "worker" => source.worker_url.clone())
                .set(s.stream.index.len() as f64);
        }
        state.lock().stream.invalidate();
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

/// Replay is committed only after a contiguous response including the live
/// boundary. Incomplete history leaves the caller rebuilding an observed subset.
fn replay(
    context: &zmq::Context,
    source: &EventSource,
    cfg: &Config,
    live_seq: u64,
    stop: &AtomicBool,
) -> Option<StreamIndex> {
    let socket = context.socket(zmq::DEALER).ok()?;
    socket.set_linger(0).ok()?;
    socket.set_rcvtimeo(250).ok()?;
    socket.set_sndtimeo(250).ok()?;
    socket.set_maxmsgsize(cfg.max_response_bytes as i64).ok()?;
    socket.connect(source.replay_endpoint.as_deref()?).ok()?;
    socket
        .send_multipart([&[][..], &0u64.to_be_bytes()[..]], 0)
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut index = StreamIndex::default();
    let mut expected = 0u64;
    while Instant::now() < deadline && !stop.load(Ordering::Acquire) {
        let mut frames = match socket.recv_multipart(0) {
            Ok(f) => f,
            Err(zmq::Error::EAGAIN) => continue,
            Err(_) => return None,
        };
        if frames.first().is_some_and(Vec::is_empty) {
            frames.remove(0);
        }
        if frames.len() == 3 && frames[2].is_empty() {
            return index
                .last_seq
                .filter(|last| *last >= live_seq)
                .map(|_| index);
        }
        let (seq, payload) = frame(&frames, &source.topic)?;
        if seq != expected {
            error("replay_gap");
            return None;
        }
        index.apply(seq, payload, &source.profile, cfg).ok()?;
        expected = expected.checked_add(1)?;
    }
    error("replay_timeout");
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::pre_kv_cache::events::tests::{config, payload, stored};

    #[test]
    fn replay_requires_contiguous_history_through_the_live_boundary() {
        for first in [0u64, 1] {
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
                        [request[0].as_slice(), &[], b"kv", &2u64.to_be_bytes(), &[]],
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
            let result = replay(&context, &source, &config(), 1, &AtomicBool::new(false));
            assert_eq!(result.is_some(), first == 0);
            if let Some(state) = result {
                assert_eq!(state.last_seq, Some(1));
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
            let sample =
                tokio::time::timeout(Duration::from_millis(cfg.load_poll_ms.min(3000)), async {
                    let response = request.send().await.ok()?;
                    if !response.status().is_success() {
                        return None;
                    }
                    let body = limited_body(response, cfg.max_response_bytes).await.ok()?;
                    parse_metrics(std::str::from_utf8(&body).ok()?)
                })
                .await
                .ok()
                .flatten();
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
            if let Some(sample) = sample {
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
            } else {
                metrics::counter!("pre_kv_cache_load_errors_total").increment(1);
            }
        }
    })
}
