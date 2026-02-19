//! Concurrent access: Multiple agents operating simultaneously.
//!
//! Tests verify that multiple concurrent readers can query the same .amem file,
//! and that rapid sequential handoffs between agents maintain data integrity.

use std::sync::Arc;
use tokio::sync::{Barrier, Mutex};

use agentic_memory_mcp::types::{JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, RequestId};
use agentic_memory_mcp::{ProtocolHandler, SessionManager};
use serde_json::json;
use tempfile::tempdir;

// ─── Helpers ───────────────────────────────────────────────────────────────

fn create_handler(path_str: &str) -> ProtocolHandler {
    let session = SessionManager::open(path_str).expect("Failed to open session");
    let session_arc = Arc::new(Mutex::new(session));
    ProtocolHandler::new(session_arc)
}

async fn init_handler(handler: &ProtocolHandler) {
    let init_req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: RequestId::Number(0),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        })),
    };
    handler
        .handle_message(JsonRpcMessage::Request(init_req))
        .await;

    let init_notif = JsonRpcNotification {
        jsonrpc: "2.0".to_string(),
        method: "initialized".to_string(),
        params: None,
    };
    handler
        .handle_message(JsonRpcMessage::Notification(init_notif))
        .await;
}

async fn call_tool(
    handler: &ProtocolHandler,
    name: &str,
    args: serde_json::Value,
) -> serde_json::Value {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: RequestId::Number(1),
        method: "tools/call".to_string(),
        params: Some(json!({"name": name, "arguments": args})),
    };
    handler
        .handle_message(JsonRpcMessage::Request(req))
        .await
        .unwrap()
}

fn parse_query_results(response: &serde_json::Value) -> Vec<serde_json::Value> {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .expect("Expected text in tool response");
    let parsed: serde_json::Value = serde_json::from_str(text).expect("Expected JSON in text");
    parsed["nodes"]
        .as_array()
        .expect("Expected nodes array")
        .clone()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

/// Multiple readers, no writers (should always work)
#[tokio::test]
async fn test_concurrent_readers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shared.amem");
    let path_str = path.to_str().unwrap().to_string();

    // Seed with data
    {
        let handler = create_handler(&path_str);
        init_handler(&handler).await;

        for i in 0..100 {
            call_tool(
                &handler,
                "memory_add",
                json!({
                    "event_type": "fact",
                    "content": format!("Fact number {}", i)
                }),
            )
            .await;
        }
        call_tool(&handler, "session_end", json!({"create_episode": false})).await;
    }

    // Spawn 5 concurrent readers
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    for agent_num in 0..5u32 {
        let path_clone = path_str.clone();
        let barrier_clone = barrier.clone();

        handles.push(tokio::spawn(async move {
            barrier_clone.wait().await; // Synchronize start

            let handler = create_handler(&path_clone);
            init_handler(&handler).await;

            let response = call_tool(
                &handler,
                "memory_query",
                json!({
                    "event_types": ["fact"],
                    "max_results": 100
                }),
            )
            .await;

            let facts = parse_query_results(&response);
            assert_eq!(
                facts.len(),
                100,
                "Agent {} should see all 100 facts",
                agent_num
            );

            agent_num
        }));
    }

    for handle in handles {
        let agent_num = handle.await.unwrap();
        println!("Agent {} completed successfully", agent_num);
    }
}

/// Sequential write-then-read (simulates real agent handoff)
#[tokio::test]
async fn test_rapid_handoff() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shared.amem");
    let path_str = path.to_str().unwrap();

    // 10 agents, each adds one fact then hands off
    for i in 0..10u64 {
        let handler = create_handler(path_str);
        init_handler(&handler).await;

        // Read previous count
        let stats = call_tool(&handler, "memory_stats", json!({})).await;
        let text = stats["result"]["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        let prev_count = parsed["node_count"].as_u64().unwrap();

        assert_eq!(prev_count, i, "Should see {} nodes from previous agents", i);

        // Add one fact
        call_tool(
            &handler,
            "memory_add",
            json!({
                "event_type": "fact",
                "content": format!("Fact from agent {}", i)
            }),
        )
        .await;

        call_tool(&handler, "session_end", json!({"create_episode": false})).await;
        drop(handler);
    }

    // Final verification
    let final_handler = create_handler(path_str);
    init_handler(&final_handler).await;

    let stats = call_tool(&final_handler, "memory_stats", json!({})).await;
    let text = stats["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["node_count"].as_u64().unwrap(), 10);
}
