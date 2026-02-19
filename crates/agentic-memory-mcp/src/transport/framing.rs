//! Message framing for newline-delimited JSON.

use crate::types::{JsonRpcMessage, McpError, McpResult};

/// Parse a single line of text as a JSON-RPC message.
pub fn parse_message(line: &str) -> McpResult<JsonRpcMessage> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(McpError::ParseError("Empty message".to_string()));
    }

    serde_json::from_str(trimmed).map_err(|e| McpError::ParseError(e.to_string()))
}

/// Serialize a value to a JSON line (with trailing newline).
pub fn frame_message(value: &serde_json::Value) -> McpResult<String> {
    let mut json = serde_json::to_string(value).map_err(McpError::Json)?;
    json.push('\n');
    Ok(json)
}
