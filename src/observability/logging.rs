//! Logging infrastructure with non-blocking file I/O.

use std::{
    fmt::{self, Write as FmtWrite},
    path::PathBuf,
};

use anyhow::{Context, Result};
use tracing::{
    field::{Field, Visit},
    Level,
};
use tracing_appender::{
    non_blocking::WorkerGuard,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_log::LogTracer;
use tracing_subscriber::{
    field::{MakeVisitor, VisitFmt, VisitOutput},
    fmt::{format::Writer as FormatWriter, time::ChronoLocal},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

use super::otel_trace::get_otel_layer;
use crate::config::ObservabilityConfig;

const TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
const TIME_FORMAT_MS: &str = "%Y-%m-%d %H:%M:%S%.3f";
const DEFAULT_LOG_TARGET: &str = "smg";
const ANSI_CYAN: &str = "\x1b[36m";
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_RESET: &str = "\x1b[0m";

fn get_time_format() -> &'static str {
    match std::env::var("SGLANG_LOG_MS") {
        Ok(v) if matches!(v.trim().to_lowercase().as_str(), "true" | "1") => TIME_FORMAT_MS,
        _ => TIME_FORMAT,
    }
}

#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub level: Level,
    pub json_format: bool,
    pub log_dir: Option<String>,
    pub colorize: bool,
    pub log_file_name: String,
    pub log_targets: Option<Vec<String>>,
}

impl LoggingConfig {
    // Init LoggingConfig by ObservabilityConfig.
    pub fn from_config(config: &ObservabilityConfig) -> Self {
        let level = config
            .log_level
            .as_deref()
            .and_then(|value| value.parse::<Level>().ok())
            .unwrap_or(Level::INFO);

        Self {
            level,
            json_format: config.json_log,
            log_dir: config.log_dir.clone(),
            colorize: true,
            log_file_name: "smg".to_string(),
            log_targets: None,
        }
    }
}

impl Default for LoggingConfig {
    #[inline]
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: true,
            log_file_name: "smg".to_string(),
            log_targets: Some(vec![DEFAULT_LOG_TARGET.to_string()]),
        }
    }
}

/// Guard that keeps the file appender thread alive.
#[allow(dead_code)]
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

#[derive(Debug, Default)]
struct NonItalicFields;

impl<'a> MakeVisitor<FormatWriter<'a>> for NonItalicFields {
    type Visitor = NonItalicVisitor<'a>;

    fn make_visitor(&self, target: FormatWriter<'a>) -> Self::Visitor {
        NonItalicVisitor {
            writer: target,
            is_empty: true,
            result: Ok(()),
        }
    }
}

struct NonItalicVisitor<'a> {
    writer: FormatWriter<'a>,
    is_empty: bool,
    result: fmt::Result,
}

impl NonItalicVisitor<'_> {
    fn maybe_pad(&mut self) {
        if self.is_empty {
            self.is_empty = false;
        } else {
            self.result = self.writer.write_char(' ');
        }
    }
}

impl Visit for NonItalicVisitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        if self.result.is_err() {
            return;
        }

        if field.name() == "message" {
            self.record_debug(field, &format_args!("{}", value));
        } else {
            self.record_debug(field, &value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if self.result.is_err() {
            return;
        }

        let name = field.name();
        if name.starts_with("log.") {
            return;
        }

        self.maybe_pad();
        let sanitized = SanitizedDebug {
            value,
            sanitize: self.writer.sanitizes_ansi_escapes(),
        };

        self.result = if name == "message" {
            write!(self.writer, "{:?}", sanitized)
        } else {
            let name = name.strip_prefix("r#").unwrap_or(name);
            if self.writer.has_ansi_escapes() {
                write!(
                    self.writer,
                    "{ANSI_CYAN}{name}{ANSI_RESET}{ANSI_DIM}={ANSI_RESET}{:?}",
                    sanitized
                )
            } else {
                write!(self.writer, "{name}={:?}", sanitized)
            }
        };
    }
}

impl VisitOutput<fmt::Result> for NonItalicVisitor<'_> {
    fn finish(self) -> fmt::Result {
        self.result
    }
}

impl VisitFmt for NonItalicVisitor<'_> {
    fn writer(&mut self) -> &mut dyn fmt::Write {
        &mut self.writer
    }
}

struct SanitizedDebug<'a> {
    value: &'a dyn fmt::Debug,
    sanitize: bool,
}

impl fmt::Debug for SanitizedDebug<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.sanitize {
            return write!(formatter, "{:?}", self.value);
        }

        let mut escaping_formatter = EscapingFormatter { inner: formatter };
        write!(escaping_formatter, "{:?}", self.value)
    }
}

struct EscapingFormatter<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
}

impl fmt::Write for EscapingFormatter<'_, '_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for character in text.chars() {
            match character {
                '\x1b' => self.inner.write_str("\\x1b")?,
                '\x07' => self.inner.write_str("\\x07")?,
                '\x08' => self.inner.write_str("\\x08")?,
                '\x0c' => self.inner.write_str("\\x0c")?,
                '\x7f' => self.inner.write_str("\\x7f")?,
                character if ('\u{80}'..='\u{9f}').contains(&character) => {
                    write!(self.inner, "\\u{{{:x}}}", character as u32)?
                }
                character => self.inner.write_char(character)?,
            }
        }
        Ok(())
    }
}

#[inline]
const fn level_to_str(level: Level) -> &'static str {
    match level {
        Level::TRACE => "trace",
        Level::DEBUG => "debug",
        Level::INFO => "info",
        Level::WARN => "warn",
        Level::ERROR => "error",
    }
}

#[inline]
fn build_filter_string(targets: &[String], level_filter: &str) -> String {
    // Exact capacity: sum of target lengths + "=" and level per target + commas between
    let capacity = targets.iter().map(String::len).sum::<usize>()
        + targets.len() * (1 + level_filter.len())
        + targets.len().saturating_sub(1);
    let mut filter_string = String::with_capacity(capacity);

    for (i, target) in targets.iter().enumerate() {
        if i > 0 {
            filter_string.push(',');
        }
        filter_string.push_str(target);
        filter_string.push('=');
        filter_string.push_str(level_filter);
    }

    filter_string
}

pub fn init_logging(
    config: LoggingConfig,
    otel_layer_config: Option<&ObservabilityConfig>,
) -> Result<LogGuard> {
    // The host process may already have installed a logger or subscriber.
    // Preserve the legacy embedded-Python behavior: use the existing global
    // subscriber instead of rejecting a subsequent Router.start() call.
    let _ = LogTracer::init();

    let level_filter = level_to_str(config.level);

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let filter_string = match &config.log_targets {
            Some(targets) if !targets.is_empty() => build_filter_string(targets, level_filter),
            _ => {
                let mut s =
                    String::with_capacity(DEFAULT_LOG_TARGET.len() + 1 + level_filter.len());
                s.push_str(DEFAULT_LOG_TARGET);
                s.push('=');
                s.push_str(level_filter);
                s
            }
        };

        EnvFilter::new(filter_string)
    });

    let mut layers = Vec::with_capacity(3);

    let time_fmt = get_time_format();

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_timer(ChronoLocal::new(time_fmt.to_string()));

    let stdout_layer = if config.json_format {
        stdout_layer
            .json()
            .flatten_event(true)
            .with_filter(env_filter.clone())
            .boxed()
    } else {
        stdout_layer
            .compact()
            .with_target(false)
            .with_file(false)
            .with_line_number(false)
            .fmt_fields(NonItalicFields)
            .with_filter(env_filter.clone())
            .boxed()
    };

    layers.push(stdout_layer);

    let mut file_guard = None;

    if let Some(log_dir) = &config.log_dir {
        let log_dir = PathBuf::from(log_dir);

        if !log_dir.exists() {
            std::fs::create_dir_all(&log_dir)
                .with_context(|| format!("create log directory {}", log_dir.display()))?;
        }

        let file_appender =
            RollingFileAppender::new(Rotation::DAILY, log_dir, &config.log_file_name);

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        file_guard = Some(guard);

        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_timer(ChronoLocal::new(time_fmt.to_string()))
            .with_writer(non_blocking);

        let file_layer = if config.json_format {
            file_layer
                .json()
                .flatten_event(true)
                .with_filter(env_filter.clone())
                .boxed()
        } else {
            file_layer
                .compact()
                .with_target(false)
                .with_file(false)
                .with_line_number(false)
                .fmt_fields(NonItalicFields)
                .with_filter(env_filter.clone())
                .boxed()
        };

        layers.push(file_layer);
    }

    let mut otel_enabled = false;
    if let Some(otel_layer_config) = otel_layer_config {
        if otel_layer_config.enable_trace {
            let otel_layer = get_otel_layer().context("create OpenTelemetry tracing layer")?;
            layers.push(otel_layer);
            otel_enabled = true;
        }
    }

    let subscriber = tracing_subscriber::registry().with(layers);

    if otel_enabled {
        subscriber
            .try_init()
            .context("initialize tracing subscriber with OpenTelemetry")?;
    } else {
        // Embedded callers, such as Python, may start the router more than
        // once in one process after a global subscriber is already installed.
        let _ = subscriber.try_init();
    }

    Ok(LogGuard {
        _file_guard: file_guard,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{Arc, Mutex},
    };

    use tracing_subscriber::{fmt::MakeWriter, Registry};

    use super::*;

    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl io::Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("lock poisoned").write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.0.lock().expect("lock poisoned").flush()
        }
    }

    impl<'a> MakeWriter<'a> for SharedBuffer {
        type Writer = SharedBuffer;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn compact_fields_keep_color_without_italic() {
        let buffer = SharedBuffer::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(true)
            .compact()
            .with_target(false)
            .with_file(false)
            .with_line_number(false)
            .fmt_fields(NonItalicFields)
            .with_writer(buffer.clone());
        let subscriber = Registry::default().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "smg::logging_test", example = "value", "formatted");
            tracing::info!(target: "smg::logging_test", count = %42, "display value");
        });

        let output = String::from_utf8(
            buffer
                .0
                .lock()
                .expect("lock poisoned")
                .drain(..)
                .collect::<Vec<_>>(),
        )
        .expect("logs are UTF-8");

        assert!(output.contains(ANSI_CYAN));
        let expected_field =
            format!("{ANSI_CYAN}example{ANSI_RESET}{ANSI_DIM}={ANSI_RESET}\"value\"");
        assert!(
            output.contains(&expected_field),
            "unexpected log output: {output:?}"
        );
        assert!(!output.contains("\x1b[3m"));
        let expected_count = format!("{ANSI_CYAN}count{ANSI_RESET}{ANSI_DIM}={ANSI_RESET}42");
        assert!(
            output.contains(&expected_count),
            "unexpected log output: {output:?}"
        );
    }
}
