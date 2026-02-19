//! Tool: session_end — End a session and optionally create an episode summary.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct EndParams {
    session_id: Option<u32>,
    #[serde(default = "default_create_episode")]
    create_episode: bool,
    summary: Option<String>,
}

fn default_create_episode() -> bool {
    true
}

/// Return the tool definition for session_end.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "session_end".to_string(),
        description: Some(
            "End a session and optionally create an episode summary node".to_string(),
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "integer" },
                "create_episode": { "type": "boolean", "default": true },
                "summary": { "type": "string", "description": "Episode summary content" }
            }
        }),
    }
}

/// Execute the session_end tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: EndParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let mut session = session.lock().await;
    let session_id = params
        .session_id
        .unwrap_or_else(|| session.current_session_id());

    if params.create_episode {
        let summary = params
            .summary
            .unwrap_or_else(|| format!("Session {session_id} completed"));

        let episode_id = session.end_session_with_episode(session_id, &summary)?;

        Ok(ToolCallResult::json(&json!({
            "session_id": session_id,
            "episode_node_id": episode_id,
            "summary": summary,
        })))
    } else {
        session.save()?;
        Ok(ToolCallResult::json(&json!({
            "session_id": session_id,
            "episode_node_id": null,
            "message": "Session ended without episode",
        })))
    }
}
