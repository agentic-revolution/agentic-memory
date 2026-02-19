//! MCP request parameter types for tools, resources, and prompts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parameters for tools/call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    /// Tool name.
    pub name: String,
    /// Tool arguments.
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// Parameters for resources/read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReadParams {
    /// Resource URI.
    pub uri: String,
}

/// Parameters for resources/subscribe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSubscribeParams {
    /// Resource URI to subscribe to.
    pub uri: String,
}

/// Parameters for resources/unsubscribe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUnsubscribeParams {
    /// Resource URI to unsubscribe from.
    pub uri: String,
}

/// Parameters for prompts/get.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptGetParams {
    /// Prompt name.
    pub name: String,
    /// Prompt arguments.
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// Parameters for $/cancelRequest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequestParams {
    /// The request ID to cancel.
    #[serde(rename = "requestId")]
    pub request_id: Value,
    /// Optional reason for cancellation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Cursor-based pagination for list operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListParams {
    /// Cursor for the next page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}
