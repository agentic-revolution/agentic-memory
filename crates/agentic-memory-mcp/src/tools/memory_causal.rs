//! Tool: memory_causal — Impact analysis: what depends on this node?

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{CausalParams, EdgeType};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct CausalInputParams {
    node_id: u64,
    #[serde(default = "default_max_depth")]
    max_depth: u32,
}

fn default_max_depth() -> u32 {
    5
}

/// Return the tool definition for memory_causal.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_causal".to_string(),
        description: Some(
            "Impact analysis — find everything that depends on a given node".to_string(),
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "node_id": { "type": "integer" },
                "max_depth": { "type": "integer", "default": 5 }
            },
            "required": ["node_id"]
        }),
    }
}

/// Execute the memory_causal tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: CausalInputParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let causal_params = CausalParams {
        node_id: params.node_id,
        max_depth: params.max_depth,
        dependency_types: vec![EdgeType::CausedBy, EdgeType::Supports],
    };

    let session = session.lock().await;

    let result = session
        .query_engine()
        .causal(session.graph(), causal_params)
        .map_err(|e| McpError::AgenticMemory(format!("Causal analysis failed: {e}")))?;

    let dependents: Vec<Value> = result
        .dependents
        .iter()
        .filter_map(|id| {
            session.graph().get_node(*id).map(|node| {
                json!({
                    "id": node.id,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                })
            })
        })
        .collect();

    Ok(ToolCallResult::json(&json!({
        "root_id": result.root_id,
        "dependent_count": result.dependents.len(),
        "affected_decisions": result.affected_decisions,
        "affected_inferences": result.affected_inferences,
        "dependents": dependents,
    })))
}
