//! Tool: memory_context — Get full context (subgraph) around a node.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct ContextParams {
    node_id: u64,
    #[serde(default = "default_depth")]
    depth: u32,
}

fn default_depth() -> u32 {
    2
}

/// Return the tool definition for memory_context.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_context".to_string(),
        description: Some("Get the full context (subgraph) around a node".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "node_id": { "type": "integer" },
                "depth": { "type": "integer", "default": 2, "minimum": 1, "maximum": 5 }
            },
            "required": ["node_id"]
        }),
    }
}

/// Execute the memory_context tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: ContextParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let session = session.lock().await;

    let subgraph = session
        .query_engine()
        .context(session.graph(), params.node_id, params.depth)
        .map_err(|e| McpError::AgenticMemory(format!("Context query failed: {e}")))?;

    let nodes: Vec<Value> = subgraph
        .nodes
        .iter()
        .map(|event| {
            json!({
                "id": event.id,
                "event_type": event.event_type.name(),
                "content": event.content,
                "confidence": event.confidence,
                "session_id": event.session_id,
            })
        })
        .collect();

    let edges: Vec<Value> = subgraph
        .edges
        .iter()
        .map(|e| {
            json!({
                "source_id": e.source_id,
                "target_id": e.target_id,
                "edge_type": e.edge_type.name(),
                "weight": e.weight,
            })
        })
        .collect();

    Ok(ToolCallResult::json(&json!({
        "center_id": subgraph.center_id,
        "depth": params.depth,
        "node_count": nodes.len(),
        "edge_count": edges.len(),
        "nodes": nodes,
        "edges": edges,
    })))
}
