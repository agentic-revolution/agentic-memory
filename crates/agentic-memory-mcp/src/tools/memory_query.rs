//! Tool: memory_query — Pattern query for matching nodes.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{EventType, PatternParams, PatternSort};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct QueryParams {
    #[serde(default)]
    event_types: Vec<String>,
    min_confidence: Option<f32>,
    max_confidence: Option<f32>,
    #[serde(default)]
    session_ids: Vec<u32>,
    created_after: Option<u64>,
    created_before: Option<u64>,
    #[serde(default = "default_max_results")]
    max_results: usize,
    #[serde(default = "default_sort")]
    sort_by: String,
}

fn default_max_results() -> usize {
    20
}

fn default_sort() -> String {
    "most_recent".to_string()
}

/// Return the tool definition for memory_query.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_query".to_string(),
        description: Some("Find memories matching conditions (pattern query)".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "event_types": { "type": "array", "items": { "type": "string" } },
                "min_confidence": { "type": "number" },
                "max_confidence": { "type": "number" },
                "session_ids": { "type": "array", "items": { "type": "integer" } },
                "created_after": { "type": "integer" },
                "created_before": { "type": "integer" },
                "max_results": { "type": "integer", "default": 20 },
                "sort_by": {
                    "type": "string",
                    "enum": ["most_recent", "highest_confidence", "most_accessed", "most_important"],
                    "default": "most_recent"
                }
            }
        }),
    }
}

/// Execute the memory_query tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: QueryParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let event_types: Vec<EventType> = params
        .event_types
        .iter()
        .filter_map(|name| EventType::from_name(name))
        .collect();

    let sort_by = match params.sort_by.as_str() {
        "highest_confidence" => PatternSort::HighestConfidence,
        "most_accessed" => PatternSort::MostAccessed,
        "most_important" => PatternSort::MostImportant,
        _ => PatternSort::MostRecent,
    };

    let pattern = PatternParams {
        event_types,
        min_confidence: params.min_confidence,
        max_confidence: params.max_confidence,
        session_ids: params.session_ids,
        created_after: params.created_after,
        created_before: params.created_before,
        min_decay_score: None,
        max_results: params.max_results,
        sort_by,
    };

    let session = session.lock().await;
    let results = session
        .query_engine()
        .pattern(session.graph(), pattern)
        .map_err(|e| McpError::AgenticMemory(format!("Pattern query failed: {e}")))?;

    let nodes: Vec<Value> = results
        .iter()
        .map(|event| {
            json!({
                "id": event.id,
                "event_type": event.event_type.name(),
                "content": event.content,
                "confidence": event.confidence,
                "session_id": event.session_id,
                "created_at": event.created_at,
                "decay_score": event.decay_score,
                "access_count": event.access_count,
            })
        })
        .collect();

    Ok(ToolCallResult::json(&json!({
        "count": nodes.len(),
        "nodes": nodes
    })))
}
