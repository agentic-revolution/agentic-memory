//! Tool: memory_temporal — Compare knowledge across time periods.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{TemporalParams, TimeRange};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct TemporalInputParams {
    range_a: RangeSpec,
    range_b: RangeSpec,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RangeSpec {
    #[serde(rename = "time_window")]
    TimeWindow { start: u64, end: u64 },
    #[serde(rename = "session")]
    Session { session_id: u32 },
    #[serde(rename = "sessions")]
    Sessions { session_ids: Vec<u32> },
}

impl RangeSpec {
    fn to_time_range(&self) -> TimeRange {
        match self {
            RangeSpec::TimeWindow { start, end } => TimeRange::TimeWindow {
                start: *start,
                end: *end,
            },
            RangeSpec::Session { session_id } => TimeRange::Session(*session_id),
            RangeSpec::Sessions { session_ids } => TimeRange::Sessions(session_ids.clone()),
        }
    }
}

/// Return the tool definition for memory_temporal.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_temporal".to_string(),
        description: Some("Compare knowledge across two time periods".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "range_a": {
                    "oneOf": [
                        { "type": "object", "properties": { "type": { "const": "time_window" }, "start": { "type": "integer" }, "end": { "type": "integer" } } },
                        { "type": "object", "properties": { "type": { "const": "session" }, "session_id": { "type": "integer" } } },
                        { "type": "object", "properties": { "type": { "const": "sessions" }, "session_ids": { "type": "array" } } }
                    ]
                },
                "range_b": { "description": "Same structure as range_a" }
            },
            "required": ["range_a", "range_b"]
        }),
    }
}

/// Execute the memory_temporal tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: TemporalInputParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    let temporal_params = TemporalParams {
        range_a: params.range_a.to_time_range(),
        range_b: params.range_b.to_time_range(),
    };

    let session = session.lock().await;

    let result = session
        .query_engine()
        .temporal(session.graph(), temporal_params)
        .map_err(|e| McpError::AgenticMemory(format!("Temporal comparison failed: {e}")))?;

    Ok(ToolCallResult::json(&json!({
        "added": result.added,
        "corrected": result.corrected,
        "unchanged": result.unchanged,
        "potentially_stale": result.potentially_stale,
        "summary": {
            "added_count": result.added.len(),
            "corrected_count": result.corrected.len(),
            "unchanged_count": result.unchanged.len(),
            "stale_count": result.potentially_stale.len(),
        }
    })))
}
