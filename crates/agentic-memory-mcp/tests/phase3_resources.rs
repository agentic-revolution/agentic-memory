//! Phase 3: Resource functionality tests.

mod common;

use serde_json::json;

use agentic_memory_mcp::resources::ResourceRegistry;
use agentic_memory_mcp::tools::ToolRegistry;

use common::fixtures::create_test_session;

#[tokio::test]
async fn test_resource_templates_list() {
    let templates = ResourceRegistry::list_templates();
    assert!(templates.len() >= 3);

    let uris: Vec<&str> = templates.iter().map(|t| t.uri_template.as_str()).collect();
    assert!(uris.contains(&"amem://node/{id}"));
    assert!(uris.contains(&"amem://session/{id}"));
    assert!(uris.contains(&"amem://types/{type}"));
}

#[tokio::test]
async fn test_resource_stats() {
    let session = create_test_session();

    // Add a node first
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Test fact"})),
        &session,
    )
    .await
    .unwrap();

    let result = ResourceRegistry::read("amem://graph/stats", &session)
        .await
        .unwrap();

    assert_eq!(result.contents.len(), 1);
    let text = result.contents[0].text.as_ref().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["node_count"], 1);
}

#[tokio::test]
async fn test_resource_node() {
    let session = create_test_session();

    // Add a node
    let add_result = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Node resource test"})),
        &session,
    )
    .await
    .unwrap();

    let text = match &add_result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let node_id = parsed["node_id"].as_u64().unwrap();

    // Read node resource
    let result = ResourceRegistry::read(&format!("amem://node/{node_id}"), &session)
        .await
        .unwrap();

    assert_eq!(result.contents.len(), 1);
    let text = result.contents[0].text.as_ref().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["id"], node_id);
    assert_eq!(parsed["event_type"], "fact");
    assert_eq!(parsed["content"], "Node resource test");
}

#[tokio::test]
async fn test_resource_recent() {
    let session = create_test_session();

    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Recent test"})),
        &session,
    )
    .await
    .unwrap();

    let result = ResourceRegistry::read("amem://graph/recent", &session)
        .await
        .unwrap();

    let text = result.contents[0].text.as_ref().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["count"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_resource_not_found() {
    let session = create_test_session();
    let result = ResourceRegistry::read("amem://nonexistent/resource", &session).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_resource_node_not_found() {
    let session = create_test_session();
    let result = ResourceRegistry::read("amem://node/99999", &session).await;
    assert!(result.is_err());
}
