//! Tool: session_start — Begin a new interaction session.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StartParams {
    session_id: Option<u32>,
    metadata: Option<Value>,
}

/// Return the tool definition for session_start.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "session_start".to_string(),
        description: Some("Start a new interaction session".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "integer", "description": "Optional explicit session ID" },
                "metadata": { "type": "object", "description": "Optional session metadata" }
            }
        }),
    }
}

/// Execute the session_start tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: StartParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let mut session = session.lock().await;
    let session_id = session.start_session(params.session_id)?;

    Ok(ToolCallResult::json(&json!({
        "session_id": session_id,
        "message": format!("Session {session_id} started"),
    })))
}
