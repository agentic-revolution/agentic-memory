//! Resource handler for `amem://node/{id}` — single node with edges.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::json;

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ReadResourceResult, ResourceContent};

/// Read a single node resource by ID.
pub async fn read_node(
    id: u64,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ReadResourceResult> {
    let session = session.lock().await;
    let graph = session.graph();

    let node = graph.get_node(id).ok_or(McpError::NodeNotFound(id))?;

    let outgoing: Vec<serde_json::Value> = graph
        .edges_from(id)
        .iter()
        .map(|e| {
            json!({
                "target_id": e.target_id,
                "edge_type": e.edge_type.name(),
                "weight": e.weight,
            })
        })
        .collect();

    let incoming: Vec<serde_json::Value> = graph
        .edges_to(id)
        .iter()
        .map(|e| {
            json!({
                "source_id": e.source_id,
                "edge_type": e.edge_type.name(),
                "weight": e.weight,
            })
        })
        .collect();

    let content = json!({
        "id": node.id,
        "event_type": node.event_type.name(),
        "content": node.content,
        "confidence": node.confidence,
        "session_id": node.session_id,
        "created_at": node.created_at,
        "access_count": node.access_count,
        "last_accessed": node.last_accessed,
        "decay_score": node.decay_score,
        "outgoing_edges": outgoing,
        "incoming_edges": incoming,
    });

    Ok(ReadResourceResult {
        contents: vec![ResourceContent {
            uri: format!("amem://node/{id}"),
            mime_type: Some("application/json".to_string()),
            text: Some(serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string())),
            blob: None,
        }],
    })
}
