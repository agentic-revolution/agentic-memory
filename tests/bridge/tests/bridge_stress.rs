//! Stress tests: Large graphs, many operations, performance validation.
//!
//! Tests verify that the MCP server handles 10K nodes, deep traversals,
//! and large file sizes within acceptable bounds.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

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

fn extract_node_id(response: &serde_json::Value) -> u64 {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .expect("Expected text in tool response");
    let parsed: serde_json::Value = serde_json::from_str(text).expect("Expected JSON in text");
    parsed["node_id"]
        .as_u64()
        .or_else(|| parsed["new_node_id"].as_u64())
        .expect("Expected node_id or new_node_id in response")
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

fn parse_traverse_results(response: &serde_json::Value) -> Vec<u64> {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .expect("Expected text in tool response");
    let parsed: serde_json::Value = serde_json::from_str(text).expect("Expected JSON in text");
    parsed["visited"]
        .as_array()
        .expect("Expected visited array")
        .iter()
        .map(|n| n["id"].as_u64().expect("Expected id in visited node"))
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

/// 10K nodes via MCP
#[tokio::test]
async fn test_10k_nodes_via_mcp() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.amem");

    let handler = create_handler(path.to_str().unwrap());
    init_handler(&handler).await;

    let start = Instant::now();

    for i in 0..10_000u64 {
        call_tool(
            &handler,
            "memory_add",
            json!({
                "event_type": "fact",
                "content": format!("Fact {}: Lorem ipsum data point", i)
            }),
        )
        .await;

        if i % 1000 == 0 && i > 0 {
            println!("Added {} nodes...", i);
        }
    }

    let write_time = start.elapsed();
    println!("Write 10K nodes: {:?}", write_time);

    // Query performance
    let query_start = Instant::now();
    let response = call_tool(
        &handler,
        "memory_query",
        json!({
            "event_types": ["fact"],
            "max_results": 100
        }),
    )
    .await;
    let query_time = query_start.elapsed();
    println!("Query 100 from 10K: {:?}", query_time);

    let facts = parse_query_results(&response);
    assert_eq!(facts.len(), 100);

    // Stats check
    let stats = call_tool(&handler, "memory_stats", json!({})).await;
    let text = stats["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["node_count"].as_u64().unwrap(), 10_000);

    call_tool(&handler, "session_end", json!({"create_episode": false})).await;

    // File size check
    let file_size = std::fs::metadata(&path).unwrap().len();
    println!(
        "File size for 10K nodes: {} bytes ({:.2} MB)",
        file_size,
        file_size as f64 / 1_000_000.0
    );
    assert!(file_size < 50_000_000, "Should be under 50MB");
}

/// Deep traversal performance
#[tokio::test]
async fn test_deep_traversal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chain.amem");

    let handler = create_handler(path.to_str().unwrap());
    init_handler(&handler).await;

    // Build chain of 100 nodes
    let mut last_id: Option<u64> = None;

    for i in 0..100u64 {
        let args = if let Some(prev) = last_id {
            json!({
                "event_type": "inference",
                "content": format!("Chain node {}", i),
                "edges": [{"target_id": prev, "edge_type": "caused_by"}]
            })
        } else {
            json!({
                "event_type": "fact",
                "content": "Chain root"
            })
        };

        let r = call_tool(&handler, "memory_add", args).await;
        last_id = Some(extract_node_id(&r));
    }

    // Traverse full chain
    let start = Instant::now();
    let traverse_response = call_tool(
        &handler,
        "memory_traverse",
        json!({
            "start_id": last_id.unwrap(),
            "edge_types": ["caused_by"],
            "direction": "forward",
            "max_depth": 100,
            "max_results": 200
        }),
    )
    .await;
    let traverse_time = start.elapsed();

    println!("Traverse 100-node chain: {:?}", traverse_time);
    assert!(
        traverse_time.as_millis() < 100,
        "Should complete in under 100ms"
    );

    let visited = parse_traverse_results(&traverse_response);
    assert_eq!(visited.len(), 100);
}
