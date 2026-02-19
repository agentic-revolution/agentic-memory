//! Tool: memory_stats — Get statistics about the memory graph.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::{json, Value};

use agentic_memory::EventType;

use crate::session::SessionManager;
use crate::types::{McpResult, ToolCallResult, ToolDefinition};

/// Return the tool definition for memory_stats.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_stats".to_string(),
        description: Some("Get statistics about the memory graph".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {}
        }),
    }
}

/// Execute the memory_stats tool.
pub async fn execute(
    _args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let session = session.lock().await;
    let graph = session.graph();

    let type_index = graph.type_index();
    let session_index = graph.session_index();

    let type_counts = json!({
        "fact": type_index.count(EventType::Fact),
        "decision": type_index.count(EventType::Decision),
        "inference": type_index.count(EventType::Inference),
        "correction": type_index.count(EventType::Correction),
        "skill": type_index.count(EventType::Skill),
        "episode": type_index.count(EventType::Episode),
    });

    let file_size = std::fs::metadata(session.file_path())
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(ToolCallResult::json(&json!({
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "dimension": graph.dimension(),
        "session_count": session_index.session_count(),
        "current_session": session.current_session_id(),
        "type_counts": type_counts,
        "file_size_bytes": file_size,
        "file_path": session.file_path().display().to_string(),
    })))
}
