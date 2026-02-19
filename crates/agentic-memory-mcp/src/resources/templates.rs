//! Resource URI template definitions.

use crate::types::{ResourceDefinition, ResourceTemplateDefinition};

/// Return all resource URI templates (parameterized).
pub fn list_templates() -> Vec<ResourceTemplateDefinition> {
    vec![
        ResourceTemplateDefinition {
            uri_template: "amem://node/{id}".to_string(),
            name: "Memory Node".to_string(),
            description: Some("A single cognitive event node with its edges".to_string()),
            mime_type: Some("application/json".to_string()),
        },
        ResourceTemplateDefinition {
            uri_template: "amem://session/{id}".to_string(),
            name: "Session Nodes".to_string(),
            description: Some("All nodes from a specific session".to_string()),
            mime_type: Some("application/json".to_string()),
        },
        ResourceTemplateDefinition {
            uri_template: "amem://types/{type}".to_string(),
            name: "Nodes by Type".to_string(),
            description: Some("All nodes of a specific event type".to_string()),
            mime_type: Some("application/json".to_string()),
        },
    ]
}

/// Return all concrete (non-templated) resource definitions.
pub fn list_resources() -> Vec<ResourceDefinition> {
    vec![
        ResourceDefinition {
            uri: "amem://graph/stats".to_string(),
            name: "Graph Statistics".to_string(),
            description: Some("Overall memory graph statistics".to_string()),
            mime_type: Some("application/json".to_string()),
        },
        ResourceDefinition {
            uri: "amem://graph/recent".to_string(),
            name: "Recent Nodes".to_string(),
            description: Some("Most recently created nodes (top 20)".to_string()),
            mime_type: Some("application/json".to_string()),
        },
        ResourceDefinition {
            uri: "amem://graph/important".to_string(),
            name: "Important Nodes".to_string(),
            description: Some("Nodes with highest decay scores (top 20)".to_string()),
            mime_type: Some("application/json".to_string()),
        },
    ]
}
