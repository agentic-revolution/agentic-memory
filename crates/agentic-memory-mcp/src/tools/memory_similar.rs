//! Tool: memory_similar — Find semantically similar memories.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde::Deserialize;
use serde_json::{json, Value};

use agentic_memory::{EventType, SimilarityParams};

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, ToolCallResult, ToolDefinition};

#[derive(Debug, Deserialize)]
struct SimilarParams {
    query_text: Option<String>,
    query_vec: Option<Vec<f32>>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_min_similarity")]
    min_similarity: f32,
    #[serde(default)]
    event_types: Vec<String>,
}

fn default_top_k() -> usize {
    10
}

fn default_min_similarity() -> f32 {
    0.5
}

/// Return the tool definition for memory_similar.
pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "memory_similar".to_string(),
        description: Some("Find semantically similar memories using vector similarity".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query_text": { "type": "string" },
                "query_vec": { "type": "array", "items": { "type": "number" } },
                "top_k": { "type": "integer", "default": 10 },
                "min_similarity": { "type": "number", "default": 0.5 },
                "event_types": { "type": "array", "items": { "type": "string" } }
            }
        }),
    }
}

/// Execute the memory_similar tool.
pub async fn execute(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<ToolCallResult> {
    let params: SimilarParams =
        serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

    // Need either query_vec or query_text with embeddings
    let query_vec = if let Some(vec) = params.query_vec {
        vec
    } else if params.query_text.is_some() {
        // Without an embedding model, we can't convert text to vectors.
        // Return a helpful error.
        return Ok(ToolCallResult::error(
            "query_text requires an embedding model. Provide query_vec directly or use memory_query for text-based search.".to_string(),
        ));
    } else {
        return Err(McpError::InvalidParams(
            "Either query_vec or query_text is required".to_string(),
        ));
    };

    let event_types: Vec<EventType> = params
        .event_types
        .iter()
        .filter_map(|name| EventType::from_name(name))
        .collect();

    let similarity_params = SimilarityParams {
        query_vec,
        top_k: params.top_k,
        min_similarity: params.min_similarity,
        event_types,
        skip_zero_vectors: true,
    };

    let session = session.lock().await;
    let results = session
        .query_engine()
        .similarity(session.graph(), similarity_params)
        .map_err(|e| McpError::AgenticMemory(format!("Similarity search failed: {e}")))?;

    let matches: Vec<Value> = results
        .iter()
        .filter_map(|m| {
            session.graph().get_node(m.node_id).map(|node| {
                json!({
                    "node_id": m.node_id,
                    "similarity": m.similarity,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                })
            })
        })
        .collect();

    Ok(ToolCallResult::json(&json!({
        "count": matches.len(),
        "matches": matches,
    })))
}
