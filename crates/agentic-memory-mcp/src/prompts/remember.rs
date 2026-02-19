//! Prompt template: "Remember this about the user."

use serde_json::Value;

use crate::types::{McpError, McpResult, PromptGetResult, PromptMessage, ToolContent};

/// Expand the `remember` prompt with the given arguments.
pub fn expand(args: Value) -> McpResult<PromptGetResult> {
    let information = args
        .get("information")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidParams("'information' argument is required".to_string()))?;

    let context = args.get("context").and_then(|v| v.as_str()).unwrap_or("");

    let context_line = if context.is_empty() {
        String::new()
    } else {
        format!("\nContext: {context}\n")
    };

    let text = format!(
        "I need to remember the following information:\n\n\
         {information}\n\
         {context_line}\n\
         Please analyze this information and:\n\
         1. Determine the appropriate event type (fact, decision, inference, skill)\n\
         2. Identify any existing memories this might relate to or contradict\n\
         3. Use the memory_add tool to store this information with appropriate edges"
    );

    Ok(PromptGetResult {
        description: Some("Guide for storing new information".to_string()),
        messages: vec![PromptMessage {
            role: "user".to_string(),
            content: ToolContent::Text { text },
        }],
    })
}
