//! Resource handler for `amem://session/{id}` — all nodes from a session.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::json;

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ReadResourceResult, ResourceContent};

/// Read all nodes belonging to a session.
pub async fn read_session(
    id: u32,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ReadResourceResult> {
    let session = session.lock().await;
    let graph = session.graph();

    let node_ids = graph.session_index().get_session(id);
    if node_ids.is_empty() {
        return Err(McpError::SessionNotFound(id));
    }

    let nodes: Vec<serde_json::Value> = node_ids
        .iter()
        .filter_map(|nid| {
            graph.get_node(*nid).map(|node| {
                json!({
                    "id": node.id,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                    "created_at": node.created_at,
                })
            })
        })
        .collect();

    let content = json!({
        "session_id": id,
        "node_count": nodes.len(),
        "nodes": nodes,
    });

    Ok(ReadResourceResult {
        contents: vec![ResourceContent {
            uri: format!("amem://session/{id}"),
            mime_type: Some("application/json".to_string()),
            text: Some(serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string())),
            blob: None,
        }],
    })
}
