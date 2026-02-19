//! Resource registration and dispatch for MCP resources.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::session::SessionManager;
use crate::types::{
    McpError, McpResult, ReadResourceResult, ResourceDefinition, ResourceTemplateDefinition,
};

use super::{graph, node, session, templates, type_index};

/// Registry of all available MCP resources.
pub struct ResourceRegistry;

impl ResourceRegistry {
    /// List all resource URI templates.
    pub fn list_templates() -> Vec<ResourceTemplateDefinition> {
        templates::list_templates()
    }

    /// List all concrete (non-templated) resources.
    pub fn list_resources() -> Vec<ResourceDefinition> {
        templates::list_resources()
    }

    /// Read a resource by URI, dispatching to the appropriate handler.
    pub async fn read(
        uri: &str,
        session: &Arc<Mutex<SessionManager>>,
    ) -> McpResult<ReadResourceResult> {
        if let Some(id_str) = uri.strip_prefix("amem://node/") {
            let id: u64 = id_str
                .parse()
                .map_err(|_| McpError::InvalidParams(format!("Invalid node ID: {id_str}")))?;
            node::read_node(id, session).await
        } else if let Some(id_str) = uri.strip_prefix("amem://session/") {
            let id: u32 = id_str
                .parse()
                .map_err(|_| McpError::InvalidParams(format!("Invalid session ID: {id_str}")))?;
            session::read_session(id, session).await
        } else if let Some(type_name) = uri.strip_prefix("amem://types/") {
            type_index::read_type(type_name, session).await
        } else if uri == "amem://graph/stats" {
            graph::read_stats(session).await
        } else if uri == "amem://graph/recent" {
            graph::read_recent(session).await
        } else if uri == "amem://graph/important" {
            graph::read_important(session).await
        } else {
            Err(McpError::ResourceNotFound(uri.to_string()))
        }
    }
}
