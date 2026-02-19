//! Prompt template: "Summarize session N."

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::Value;

use crate::session::SessionManager;
use crate::types::{McpResult, PromptGetResult, PromptMessage, ToolContent};

/// Expand the `summarize` prompt with the given arguments.
pub async fn expand(
    args: Value,
    session: &Arc<Mutex<SessionManager>>,
) -> McpResult<PromptGetResult> {
    let session = session.lock().await;
    let session_id = args
        .get("session_id")
        .and_then(|v| v.as_u64())
        .map(|id| id as u32)
        .unwrap_or_else(|| session.current_session_id());

    let node_ids = session.graph().session_index().get_session(session_id);
    let node_count = node_ids.len();

    let node_list: Vec<String> = node_ids
        .iter()
        .filter_map(|id| {
            session.graph().get_node(*id).map(|node| {
                format!(
                    "- [#{} {}] {}",
                    node.id,
                    node.event_type.name(),
                    if node.content.len() > 80 {
                        format!("{}...", &node.content[..80])
                    } else {
                        node.content.clone()
                    }
                )
            })
        })
        .collect();

    let text = format!(
        "Please summarize session {session_id} which contains {node_count} memories:\n\n\
         {}\n\n\
         Create a concise episode summary capturing:\n\
         1. The main topic or goal\n\
         2. Key facts learned\n\
         3. Important decisions made\n\
         4. Any corrections\n\
         5. The outcome\n\n\
         Then use session_end with create_episode=true and your summary.",
        node_list.join("\n")
    );

    Ok(PromptGetResult {
        description: Some(format!("Guide for summarizing session {session_id}")),
        messages: vec![PromptMessage {
            role: "user".to_string(),
            content: ToolContent::Text { text },
        }],
    })
}
