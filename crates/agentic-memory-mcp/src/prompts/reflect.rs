//! Prompt template: "Why did I decide X?"

use serde_json::Value;

use crate::types::{McpError, McpResult, PromptGetResult, PromptMessage, ToolContent};

/// Expand the `reflect` prompt with the given arguments.
pub fn expand(args: Value) -> McpResult<PromptGetResult> {
    let topic = args
        .get("topic")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidParams("'topic' argument is required".to_string()))?;

    let node_hint = args
        .get("node_id")
        .and_then(|v| v.as_u64())
        .map(|id| format!("\nStart from node #{id}.\n"))
        .unwrap_or_default();

    let text = format!(
        "I want to understand my reasoning about: {topic}\n\
         {node_hint}\n\
         Please help me reflect by:\n\
         1. Use memory_query to find relevant decisions or beliefs\n\
         2. Use memory_traverse with direction=\"backward\" to find the reasoning chain\n\
         3. Use memory_causal to understand dependencies\n\
         4. Summarize the reasoning chain clearly"
    );

    Ok(PromptGetResult {
        description: Some("Guide for understanding past decisions".to_string()),
        messages: vec![PromptMessage {
            role: "user".to_string(),
            content: ToolContent::Text { text },
        }],
    })
}
