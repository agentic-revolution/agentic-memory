//! Resource handlers for `amem://graph/*` — graph-level statistics and views.

use std::sync::Arc;
use tokio::sync::Mutex;

use serde_json::json;

use agentic_memory::EventType;

use crate::session::SessionManager;
use crate::types::{McpResult, ReadResourceResult, ResourceContent};

/// Read overall graph statistics.
pub async fn read_stats(session: &Arc<Mutex<SessionManager>>) -> McpResult<ReadResourceResult> {
    let session = session.lock().await;
    let graph = session.graph();
    let type_index = graph.type_index();
    let session_index = graph.session_index();

    let content = json!({
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "dimension": graph.dimension(),
        "session_count": session_index.session_count(),
        "type_counts": {
            "fact": type_index.count(EventType::Fact),
            "decision": type_index.count(EventType::Decision),
            "inference": type_index.count(EventType::Inference),
            "correction": type_index.count(EventType::Correction),
            "skill": type_index.count(EventType::Skill),
            "episode": type_index.count(EventType::Episode),
        }
    });

    Ok(ReadResourceResult {
        contents: vec![ResourceContent {
            uri: "amem://graph/stats".to_string(),
            mime_type: Some("application/json".to_string()),
            text: Some(serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string())),
            blob: None,
        }],
    })
}

/// Read the most recently created nodes (top 20).
pub async fn read_recent(session: &Arc<Mutex<SessionManager>>) -> McpResult<ReadResourceResult> {
    let session = session.lock().await;
    let graph = session.graph();

    let recent_ids = graph.temporal_index().most_recent(20);
    let nodes: Vec<serde_json::Value> = recent_ids
        .iter()
        .filter_map(|id| {
            graph.get_node(*id).map(|node| {
                json!({
                    "id": node.id,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                    "session_id": node.session_id,
                    "created_at": node.created_at,
                })
            })
        })
        .collect();

    let content = json!({
        "count": nodes.len(),
        "nodes": nodes,
    });

    Ok(ReadResourceResult {
        contents: vec![ResourceContent {
            uri: "amem://graph/recent".to_string(),
            mime_type: Some("application/json".to_string()),
            text: Some(serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string())),
            blob: None,
        }],
    })
}

/// Read the most important nodes by decay score (top 20).
pub async fn read_important(session: &Arc<Mutex<SessionManager>>) -> McpResult<ReadResourceResult> {
    let session = session.lock().await;
    let graph = session.graph();

    // Get all nodes sorted by decay_score descending
    let mut nodes_with_scores: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| (n.id, n.decay_score))
        .collect();
    nodes_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_nodes: Vec<serde_json::Value> = nodes_with_scores
        .iter()
        .take(20)
        .filter_map(|(id, _)| {
            graph.get_node(*id).map(|node| {
                json!({
                    "id": node.id,
                    "event_type": node.event_type.name(),
                    "content": node.content,
                    "confidence": node.confidence,
                    "decay_score": node.decay_score,
                    "session_id": node.session_id,
                })
            })
        })
        .collect();

    let content = json!({
        "count": top_nodes.len(),
        "nodes": top_nodes,
    });

    Ok(ReadResourceResult {
        contents: vec![ResourceContent {
            uri: "amem://graph/important".to_string(),
            mime_type: Some("application/json".to_string()),
            text: Some(serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string())),
            blob: None,
        }],
    })
}
