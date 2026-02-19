//! MCP capability and initialization types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MCP protocol version this server implements.
pub const MCP_VERSION: &str = "2024-11-05";

/// Server name constant.
pub const SERVER_NAME: &str = "agentic-memory-mcp";

/// Server version constant.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Implementation info for server or client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    /// Name of the implementation.
    pub name: String,
    /// Version string.
    pub version: String,
}

/// Client capabilities sent during initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Experimental capabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimental: Option<HashMap<String, serde_json::Value>>,
    /// Sampling capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
    /// Roots capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
}

/// Server capabilities advertised during initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Experimental capabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimental: Option<HashMap<String, serde_json::Value>>,
    /// Logging capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
    /// Prompts capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Resources capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Tools capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
}

/// Sampling capability marker.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Roots capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RootsCapability {
    /// Whether the client supports roots/list_changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Logging capability marker.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingCapability {}

/// Prompts capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptsCapability {
    /// Whether the server supports prompts/list_changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Resources capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourcesCapability {
    /// Whether the server supports resource subscriptions.
    #[serde(default)]
    pub subscribe: bool,
    /// Whether the server supports resources/list_changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Tools capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Whether the server supports tools/list_changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Initialize request parameters from client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Requested protocol version.
    pub protocol_version: String,
    /// Client capabilities.
    pub capabilities: ClientCapabilities,
    /// Client implementation info.
    pub client_info: Implementation,
}

/// Initialize response result from server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Negotiated protocol version.
    pub protocol_version: String,
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Server implementation info.
    pub server_info: Implementation,
    /// Optional instructions for the client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl ServerCapabilities {
    /// Build the default capabilities for this server.
    pub fn default_capabilities() -> Self {
        Self {
            experimental: None,
            logging: Some(LoggingCapability {}),
            prompts: Some(PromptsCapability {
                list_changed: false,
            }),
            resources: Some(ResourcesCapability {
                subscribe: true,
                list_changed: false,
            }),
            tools: Some(ToolsCapability {
                list_changed: false,
            }),
        }
    }
}

impl InitializeResult {
    /// Build the default initialization result.
    pub fn default_result() -> Self {
        Self {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ServerCapabilities::default_capabilities(),
            server_info: Implementation {
                name: SERVER_NAME.to_string(),
                version: SERVER_VERSION.to_string(),
            },
            instructions: Some(
                "AgenticMemory MCP server provides persistent cognitive graph memory. \
                 Use tools to add, query, traverse, and correct memories. \
                 Use resources to browse the memory graph. \
                 Use prompts for guided memory operations."
                    .to_string(),
            ),
        }
    }
}
