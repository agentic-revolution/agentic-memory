//! Phase 3: Tool functionality tests.

mod common;

use serde_json::json;

use agentic_memory_mcp::tools::ToolRegistry;

use common::fixtures::create_test_session;

#[tokio::test]
async fn test_memory_add_fact() {
    let session = create_test_session();
    let result = ToolRegistry::call(
        "memory_add",
        Some(json!({
            "event_type": "fact",
            "content": "User prefers Rust",
            "confidence": 0.95
        })),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text content"),
    };

    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["node_id"].as_u64().is_some());
    assert_eq!(parsed["event_type"], "fact");
}

#[tokio::test]
async fn test_memory_add_with_edges() {
    let session = create_test_session();

    // Add a fact first
    let result1 = ToolRegistry::call(
        "memory_add",
        Some(json!({
            "event_type": "fact",
            "content": "Team knows Rust"
        })),
        &session,
    )
    .await
    .unwrap();

    let text1 = match &result1.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let node_id_1: serde_json::Value = serde_json::from_str(text1).unwrap();
    let id1 = node_id_1["node_id"].as_u64().unwrap();

    // Add a decision caused by the fact
    let result2 = ToolRegistry::call(
        "memory_add",
        Some(json!({
            "event_type": "decision",
            "content": "Use Rust for the backend",
            "edges": [{"target_id": id1, "edge_type": "caused_by", "weight": 1.0}]
        })),
        &session,
    )
    .await
    .unwrap();

    let text2 = match &result2.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed2: serde_json::Value = serde_json::from_str(text2).unwrap();
    assert_eq!(parsed2["edges_created"], 1);
}

#[tokio::test]
async fn test_memory_query_empty() {
    let session = create_test_session();
    let result = ToolRegistry::call("memory_query", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["count"], 0);
}

#[tokio::test]
async fn test_memory_query_with_results() {
    let session = create_test_session();

    // Add some facts
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Fact A"})),
        &session,
    )
    .await
    .unwrap();

    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "decision", "content": "Decision B"})),
        &session,
    )
    .await
    .unwrap();

    // Query all
    let result = ToolRegistry::call("memory_query", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["count"], 2);

    // Query by type
    let result = ToolRegistry::call(
        "memory_query",
        Some(json!({"event_types": ["fact"]})),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["count"], 1);
}

#[tokio::test]
async fn test_memory_correct() {
    let session = create_test_session();

    // Add initial fact
    let result = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Earth is flat"})),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let old_id = parsed["node_id"].as_u64().unwrap();

    // Correct it
    let result = ToolRegistry::call(
        "memory_correct",
        Some(json!({
            "old_node_id": old_id,
            "new_content": "Earth is round",
            "reason": "Scientific consensus"
        })),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["old_node_id"], old_id);
    assert!(parsed["new_node_id"].as_u64().is_some());
    assert_eq!(parsed["supersedes"], true);
}

#[tokio::test]
async fn test_memory_resolve() {
    let session = create_test_session();

    // Add fact and correct it
    let r1 = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Version 1"})),
        &session,
    )
    .await
    .unwrap();
    let t1 = match &r1.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let p1: serde_json::Value = serde_json::from_str(t1).unwrap();
    let id1 = p1["node_id"].as_u64().unwrap();

    ToolRegistry::call(
        "memory_correct",
        Some(json!({"old_node_id": id1, "new_content": "Version 2"})),
        &session,
    )
    .await
    .unwrap();

    // Resolve from original
    let result = ToolRegistry::call("memory_resolve", Some(json!({"node_id": id1})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["resolved_id"].as_u64().is_some());
    // The resolved_id should be different from the original since we corrected it
    assert_ne!(parsed["resolved_id"], parsed["original_id"]);
}

#[tokio::test]
async fn test_memory_stats() {
    let session = create_test_session();

    // Add a node
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Test fact"})),
        &session,
    )
    .await
    .unwrap();

    let result = ToolRegistry::call("memory_stats", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["node_count"], 1);
    assert!(parsed["type_counts"]["fact"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_session_lifecycle() {
    let session = create_test_session();

    // Start session
    let result = ToolRegistry::call("session_start", Some(json!({})), &session)
        .await
        .unwrap();
    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["session_id"].as_u64().is_some());

    // Add something
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Session test fact"})),
        &session,
    )
    .await
    .unwrap();

    // End session with episode
    let result = ToolRegistry::call(
        "session_end",
        Some(json!({
            "create_episode": true,
            "summary": "Test session summary"
        })),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["episode_node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_tool_not_found() {
    let session = create_test_session();
    let result = ToolRegistry::call("nonexistent_tool", Some(json!({})), &session).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_memory_context() {
    let session = create_test_session();

    // Add a node
    let result = ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Central fact"})),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let id = parsed["node_id"].as_u64().unwrap();

    // Get context
    let result = ToolRegistry::call(
        "memory_context",
        Some(json!({"node_id": id, "depth": 1})),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["center_id"], id);
    assert!(parsed["node_count"].as_u64().unwrap() >= 1);
}
