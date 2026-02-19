//! Prompt template: "Update my understanding of X."

use serde_json::Value;

use crate::types::{McpError, McpResult, PromptGetResult, PromptMessage, ToolContent};

/// Expand the `correct` prompt with the given arguments.
pub fn expand(args: Value) -> McpResult<PromptGetResult> {
    let old_belief = args
        .get("old_belief")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidParams("'old_belief' argument is required".to_string()))?;

    let new_information = args
        .get("new_information")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            McpError::InvalidParams("'new_information' argument is required".to_string())
        })?;

    let reason = args
        .get("reason")
        .and_then(|v| v.as_str())
        .map(|r| format!("\nReason: {r}\n"))
        .unwrap_or_default();

    let text = format!(
        "I need to correct my understanding:\n\n\
         Previous belief: {old_belief}\n\
         Correct information: {new_information}\n\
         {reason}\n\
         Please:\n\
         1. Use memory_query to find the node containing the old belief\n\
         2. Use memory_causal to see what depends on this belief\n\
         3. Use memory_correct to create the correction\n\
         4. Consider if dependent decisions should also be corrected"
    );

    Ok(PromptGetResult {
        description: Some("Guide for updating beliefs".to_string()),
        messages: vec![PromptMessage {
            role: "user".to_string(),
            content: ToolContent::Text { text },
        }],
    })
}
