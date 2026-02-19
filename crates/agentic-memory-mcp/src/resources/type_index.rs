//! Resource handler for `amem://types/{type}` — all nodes of a given event type.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::json;

use agentic_memory::EventType;

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ReadResourceResult, ResourceContent};

/// Read all nodes of a specific event type.
pub async fn read_type(
    type_name: &str,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ReadResourceResult> {
    let event_type = EventType::from_name(type_name)
        .ok_or_else(|| McpError::InvalidParams(format!("Unknown event type: {type_name}")))?;

    let session = session.lock().await;
    let graph = session.graph();

    let node_ids = graph.type_index().get(event_type);
    let nodes: Vec<serde_json::Value> = node_ids
        .iter()
        .filter_map(|nid| {
            graph.get_node(*nid).map(|node| {
                json!({
                    "id": node.id,
                    "content": node.content,
                    "confidence": node.confidence,
                    "session_id": node.session_id,
                    "created_at": node.created_at,
                })
            })
        })
        .collect();

    let content = json!({
        "event_type": type_name,
        "count": nodes.len(),
        "nodes": nodes,
    });

    Ok(ReadResourceResult {
        contents: vec![ResourceContent {
            uri: format!("amem://types/{type_name}"),
            mime_type: Some("application/json".to_string()),
            text: Some(serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string())),
            blob: None,
        }],
    })
}
