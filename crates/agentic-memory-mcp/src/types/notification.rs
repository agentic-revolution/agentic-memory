//! MCP notification types for server-to-client and client-to-server notifications.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Progress token — either string or number.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProgressToken {
    /// String token.
    String(String),
    /// Numeric token.
    Number(i64),
}

/// Progress notification params (server → client).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressParams {
    /// The progress token from the original request.
    pub progress_token: ProgressToken,
    /// Current progress value.
    pub progress: f64,
    /// Optional total progress value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

/// Log message notification params (server → client).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMessageParams {
    /// Log level.
    pub level: LogLevel,
    /// Optional logger name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logger: Option<String>,
    /// Log message data.
    pub data: Value,
}

/// Log levels for MCP logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Debug level.
    Debug,
    /// Info level.
    Info,
    /// Warning level.
    Warning,
    /// Error level.
    Error,
}

/// Resource updated notification (server → client).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdatedParams {
    /// URI of the updated resource.
    pub uri: String,
}
