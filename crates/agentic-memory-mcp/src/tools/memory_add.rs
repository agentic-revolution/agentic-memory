//! Tool: memory_add — Add a cognitive event to the memory graph.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{EdgeType, EventType};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

/// Input parameters for memory_add.
#[derive(Debug, Deserialize)]
struct AddParams {
    event_type: String,
    content: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
    #[serde(default)]
    edges: Vec<EdgeParam>,
}

#[derive(Debug, Deserialize)]
struct EdgeParam {
    target_id: u64,
    edge_type: String,
    #[serde(default = "default_weight")]
    weight: f32,
}

fn default_confidence() -> f32 {
    0.9
}

fn default_weight() -> f32 {
    1.0
}

/// Return the tool definition for memory_add.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_add".to_string(),
        description: Some("Add a new cognitive event to the memory graph".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "enum": ["fact", "decision", "inference", "correction", "skill", "episode"],
                    "description": "Type of cognitive event"
                },
                "content": {
                    "type": "string",
                    "description": "The content of the memory"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.9,
                    "description": "Confidence level (0.0 to 1.0)"
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_id": { "type": "integer" },
                            "edge_type": {
                                "type": "string",
                                "enum": ["caused_by", "supports", "contradicts", "supersedes", "related_to", "part_of", "temporal_next"]
                            },
                            "weight": { "type": "number", "default": 1.0 }
                        },
                        "required": ["target_id", "edge_type"]
                    }
                }
            },
            "required": ["event_type", "content"]
        }),
    }
}

/// Execute the memory_add tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: AddParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    // Validate confidence is within [0.0, 1.0]
    if !(0.0..=1.0).contains(&params.confidence) {
        return Err(McpError::InvalidParams(format!(
            "confidence must be between 0.0 and 1.0, got {}",
            params.confidence
        )));
    }

    let event_type = EventType::from_name(&params.event_type).ok_or_else(|| {
        McpError::InvalidParams(format!("Unknown event type: {}", params.event_type))
    })?;

    let edges: Vec<(u64, EdgeType, f32)> = params
        .edges
        .iter()
        .map(|e| {
            let edge_type = EdgeType::from_name(&e.edge_type).ok_or_else(|| {
                McpError::InvalidParams(format!("Unknown edge type: {}", e.edge_type))
            })?;
            Ok((e.target_id, edge_type, e.weight))
        })
        .collect::<McpResult<Vec<_>>>()?;

    let mut session = session.lock().await;
    let (node_id, edges_created) =
        session.add_event(event_type, &params.content, params.confidence, edges)?;

    Ok(ToolCallResult::json(&json!({
        "node_id": node_id,
        "event_type": params.event_type,
        "edges_created": edges_created
    })))
}
