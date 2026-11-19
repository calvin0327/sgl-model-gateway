//! Bounded warnings for degraded cache routing, with an explicit recovery event.

use parking_lot::Mutex;
use std::{collections::HashMap, time::Instant};

#[derive(Debug, Default)]
pub(super) struct Diagnostics {
    warnings: Mutex<HashMap<(String, &'static str), Instant>>,
}

impl Diagnostics {
    pub(super) fn is_active(&self, scope: &str, reason: &'static str) -> bool {
        self.warnings
            .lock()
            .contains_key(&(scope.to_owned(), reason))
    }

    // Include unbound workers too: a profile mismatch prevents a subscription
    // from being created, but must still be visible through the status endpoint.
    pub(super) fn by_scope(&self) -> std::collections::BTreeMap<String, Vec<&'static str>> {
        let mut scopes: std::collections::BTreeMap<String, Vec<&'static str>> = Default::default();
        for (scope, reason) in self.warnings.lock().keys() {
            scopes.entry(scope.clone()).or_default().push(*reason);
        }
        for reasons in scopes.values_mut() {
            reasons.sort_unstable();
        }
        scopes
    }

    /// Expose outstanding reasons through the authenticated business-port API.
    /// Keep request bodies, token IDs and connection credentials out of diagnostics.
    pub(super) fn reasons(&self, scope: &str) -> Vec<&'static str> {
        let mut reasons: Vec<_> = self
            .warnings
            .lock()
            .keys()
            .filter_map(|(key, reason)| (key == scope).then_some(*reason))
            .collect();
        reasons.sort_unstable();
        reasons
    }

    // Warn once per unresolved fault. Polls and requests keep the status visible
    // without repeating the same warning; recovery allows the next outage to warn.
    pub(super) fn warn(&self, scope: &str, reason: &'static str, detail: &str) {
        let mut warnings = self.warnings.lock();
        let key = (scope.to_owned(), reason);
        if warnings.contains_key(&key) {
            return;
        }
        // Bound retained scopes when workers are repeatedly replaced.
        if warnings.len() >= 1024 && !warnings.contains_key(&key) {
            if let Some(oldest) = warnings
                .iter()
                .min_by_key(|(_, at)| *at)
                .map(|(key, _)| key.clone())
            {
                warnings.remove(&oldest);
            }
        }
        warnings.insert(key, Instant::now());
        tracing::warn!(scope, reason, detail, "pre_kv_cache degraded");
    }

    // Recovery is emitted only after evidence from the affected background path.
    pub(super) fn recovered(&self, scope: &str, reason: &'static str) {
        if self
            .warnings
            .lock()
            .remove(&(scope.to_owned(), reason))
            .is_some()
        {
            tracing::info!(scope, reason, "pre_kv_cache recovered");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, sync::Arc};

    #[derive(Clone, Default)]
    struct LogWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for LogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn warnings_are_bounded_per_scope_and_reason_and_recover_once() {
        let output = LogWriter::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            let diagnostics = Diagnostics::default();
            diagnostics.warn("worker", "no_kv_events", "check publisher");
            diagnostics.warn("worker", "no_kv_events", "check publisher");
            diagnostics.warn("worker", "event_frame", "invalid frame");
            diagnostics.warn("other", "no_kv_events", "check publisher");
            diagnostics.recovered("worker", "no_kv_events");
            diagnostics.recovered("worker", "no_kv_events");
            diagnostics.warn("worker", "no_kv_events", "new outage");
            *diagnostics
                .warnings
                .lock()
                .get_mut(&("worker".into(), "no_kv_events"))
                .unwrap() = Instant::now() - std::time::Duration::from_secs(31);
            diagnostics.warn("worker", "no_kv_events", "ongoing outage");
        });
        let logs = String::from_utf8(output.0.lock().clone()).unwrap();
        let records: Vec<serde_json::Value> = logs
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            records.iter().filter(|row| row["level"] == "WARN").count(),
            4
        );
        assert_eq!(
            records.iter().filter(|row| row["level"] == "INFO").count(),
            1
        );
    }
}
