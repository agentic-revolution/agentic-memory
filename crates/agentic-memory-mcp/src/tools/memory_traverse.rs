//! Tool: memory_traverse — Walk the graph from a starting node.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{EdgeType, TraversalDirection, TraversalParams};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct TraverseParams {
    start_id: u64,
    #[serde(default)]
    edge_types: Vec<String>,
    #[serde(default = "default_direction")]
    direction: String,
    #[serde(default = "default_max_depth")]
    max_depth: u32,
    #[serde(default = "default_max_results")]
    max_results: usize,
    min_confidence: Option<f32>,
}

fn default_direction() -> String {
    "forward".to_string()
}

fn default_max_depth() -> u32 {
    5
}

fn default_max_results() -> usize {
    20
}

/// Return the tool definition for memory_traverse.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_traverse".to_string(),
        description: Some(
            "Walk the graph from a starting node, following edges of specified types".to_string(),
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "start_id": { "type": "integer", "description": "Starting node ID" },
                "edge_types": { "type": "array", "items": { "type": "string" } },
                "direction": { "type": "string", "enum": ["forward", "backward", "both"], "default": "forward" },
                "max_depth": { "type": "integer", "default": 5 },
                "max_results": { "type": "integer", "default": 20 },
                "min_confidence": { "type": "number" }
            },
            "required": ["start_id"]
        }),
    }
}

/// Execute the memory_traverse tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: TraverseParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let edge_types: Vec<EdgeType> = if params.edge_types.is_empty() {
        vec![
            EdgeType::CausedBy,
            EdgeType::Supports,
            EdgeType::Contradicts,
            EdgeType::Supersedes,
            EdgeType::RelatedTo,
            EdgeType::PartOf,
            EdgeType::TemporalNext,
        ]
    } else {
        params
            .edge_types
            .iter()
            .filter_map(|name| EdgeType::from_name(name))
            .collect()
    };

    let direction = match params.direction.as_str() {
        "backward" => TraversalDirection::Backward,
        "both" => TraversalDirection::Both,
        _ => TraversalDirection::Forward,
    };

    let traversal = TraversalParams {
        start_id: params.start_id,
        edge_types,
        direction,
        max_depth: params.max_depth,
        max_results: params.max_results,
        min_confidence: params.min_confidence.unwrap_or(0.0),
    };

    let session = session.lock().await;
    let result = session
        .query_engine()
        .traverse(session.graph(), traversal)
        .map_err(|e| McpError::AgenticMemory(format!("Traversal failed: {e}")))?;

    let visited: Vec<Value> = result
        .visited
        .iter()
        .filter_map(|id| {
            session.graph().get_node(*id).map(|node| {
                json!({
                    "id": node.id,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                    "depth": result.depths.get(id).copied().unwrap_or(0),
                })
            })
        })
        .collect();

    let edges: Vec<Value> = result
        .edges_traversed
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
        "start_id": params.start_id,
        "visited_count": visited.len(),
        "visited": visited,
        "edges_traversed": edges,
    })))
}
