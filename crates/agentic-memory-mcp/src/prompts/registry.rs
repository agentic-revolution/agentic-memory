//! Prompt registration and dispatch for MCP prompt templates.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::Value;

use crate::session::SessionManager;
use crate::types::{McpError, McpResult, PromptArgument, PromptDefinition, PromptGetResult};

use super::{correct, reflect, remember, summarize};

/// Registry of all available MCP prompts.
pub struct PromptRegistry;

impl PromptRegistry {
    /// List all available prompt definitions.
    pub fn list_prompts() -> Vec<PromptDefinition> {
        vec![
            PromptDefinition {
                name: "remember".to_string(),
                description: Some("Guide for storing new information in memory".to_string()),
                arguments: Some(vec![
                    PromptArgument {
                        name: "information".to_string(),
                        description: Some("What to remember".to_string()),
                        required: true,
                    },
                    PromptArgument {
                        name: "context".to_string(),
                        description: Some("Why this is important".to_string()),
                        required: false,
                    },
                ]),
            },
            PromptDefinition {
                name: "reflect".to_string(),
                description: Some(
                    "Guide for understanding past decisions and reasoning".to_string(),
                ),
                arguments: Some(vec![
                    PromptArgument {
                        name: "topic".to_string(),
                        description: Some("What decision or belief to reflect on".to_string()),
                        required: true,
                    },
                    PromptArgument {
                        name: "node_id".to_string(),
                        description: Some("Specific node ID to start from".to_string()),
                        required: false,
                    },
                ]),
            },
            PromptDefinition {
                name: "correct".to_string(),
                description: Some(
                    "Guide for updating beliefs and correcting past info".to_string(),
                ),
                arguments: Some(vec![
                    PromptArgument {
                        name: "old_belief".to_string(),
                        description: Some("What was previously believed".to_string()),
                        required: true,
                    },
                    PromptArgument {
                        name: "new_information".to_string(),
                        description: Some("The correct information".to_string()),
                        required: true,
                    },
                    PromptArgument {
                        name: "reason".to_string(),
                        description: Some("Why this is being corrected".to_string()),
                        required: false,
                    },
                ]),
            },
            PromptDefinition {
                name: "summarize".to_string(),
                description: Some("Guide for creating a session summary".to_string()),
                arguments: Some(vec![PromptArgument {
                    name: "session_id".to_string(),
                    description: Some("Session ID to summarize (defaults to current)".to_string()),
                    required: false,
                }]),
            },
        ]
    }

    /// Expand a prompt with the given arguments, dispatching to the appropriate handler.
    pub async fn get(
        name: &str,
        arguments: Option<Value>,
        session: &Arc<Mutex<SessionManager>>,
    ) -> McpResult<PromptGetResult> {
        let args = arguments.unwrap_or(Value::Object(serde_json::Map::new()));

        match name {
            "remember" => remember::expand(args),
            "reflect" => reflect::expand(args),
            "correct" => correct::expand(args),
            "summarize" => summarize::expand(args, session).await,
            _ => Err(McpError::PromptNotFound(name.to_string())),
        }
    }
}
