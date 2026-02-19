//! Tool: memory_resolve — Follow supersedes chain to get the latest version.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct ResolveParams {
    node_id: u64,
}

/// Return the tool definition for memory_resolve.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_resolve".to_string(),
        description: Some(
            "Follow the supersedes chain to get the latest version of a belief".to_string(),
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "node_id": { "type": "integer", "description": "Node ID to resolve" }
            },
            "required": ["node_id"]
        }),
    }
}

/// Execute the memory_resolve tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: ResolveParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let session = session.lock().await;

    let resolved = session
        .query_engine()
        .resolve(session.graph(), params.node_id)
        .map_err(|e| McpError::AgenticMemory(format!("Resolve failed: {e}")))?;

    let latest = json!({
        "id": resolved.id,
        "event_type": resolved.event_type.name(),
        "content": resolved.content,
        "confidence": resolved.confidence,
        "session_id": resolved.session_id,
        "created_at": resolved.created_at,
    });

    let is_same = resolved.id == params.node_id;

    Ok(ToolCallResult::json(&json!({
        "original_id": params.node_id,
        "resolved_id": resolved.id,
        "is_latest": is_same,
        "latest": latest,
    })))
}
