//! Phase 4: Streaming and progress tracking tests.

use tokio::sync::mpsc;

use agentic_memory_mcp::streaming::chunked::chunk_results;
use agentic_memory_mcp::streaming::ProgressTracker;
use agentic_memory_mcp::types::JsonRpcNotification;

#[tokio::test]
async fn test_progress_tracking_lifecycle() {
    let (tx, mut rx) = mpsc::channel::<JsonRpcNotification>(16);
    let tracker = ProgressTracker::new(tx);

    // Start tracking
    let token = tracker.start(Some(100.0)).await;
    assert!(!token.is_empty());

    // Update progress
    tracker.update(&token, 25.0).await.unwrap();

    // Should have received a progress notification
    let notification = rx.recv().await.unwrap();
    assert_eq!(notification.method, "notifications/progress");

    // Update again
    tracker.update(&token, 75.0).await.unwrap();
    let _notif2 = rx.recv().await.unwrap();

    // Complete
    tracker.complete(&token).await;

    // After completion, is_cancelled should return true (token no longer tracked)
    assert!(tracker.is_cancelled(&token).await);
}

#[tokio::test]
async fn test_progress_cancellation() {
    let (tx, _rx) = mpsc::channel::<JsonRpcNotification>(16);
    let tracker = ProgressTracker::new(tx);

    let token = tracker.start(Some(50.0)).await;

    // Not cancelled initially
    assert!(!tracker.is_cancelled(&token).await);

    // Cancel
    tracker.cancel(&token).await;

    // Now it should be cancelled
    assert!(tracker.is_cancelled(&token).await);
}

#[tokio::test]
async fn test_progress_unknown_token() {
    let (tx, _rx) = mpsc::channel::<JsonRpcNotification>(16);
    let tracker = ProgressTracker::new(tx);

    // Unknown token should be treated as cancelled
    assert!(tracker.is_cancelled("nonexistent-token").await);

    // Updating unknown token should be a no-op (not error)
    tracker.update("nonexistent-token", 10.0).await.unwrap();
}

#[tokio::test]
async fn test_chunked_results() {
    let results: Vec<serde_json::Value> = (0..10).map(|i| serde_json::json!({"id": i})).collect();

    let chunks = chunk_results(results, 3);

    assert_eq!(chunks.len(), 4); // 10 / 3 = 3 full + 1 partial
    assert_eq!(chunks[0].len(), 3);
    assert_eq!(chunks[1].len(), 3);
    assert_eq!(chunks[2].len(), 3);
    assert_eq!(chunks[3].len(), 1);
}

#[tokio::test]
async fn test_chunked_results_empty() {
    let results: Vec<serde_json::Value> = vec![];
    let chunks = chunk_results(results, 5);
    assert!(chunks.is_empty());
}

#[tokio::test]
async fn test_chunked_results_single_chunk() {
    let results: Vec<serde_json::Value> = (0..3).map(|i| serde_json::json!({"id": i})).collect();

    let chunks = chunk_results(results, 10);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), 3);
}

#[tokio::test]
async fn test_multiple_concurrent_progress_tokens() {
    let (tx, mut rx) = mpsc::channel::<JsonRpcNotification>(32);
    let tracker = ProgressTracker::new(tx);

    // Start two operations
    let token1 = tracker.start(Some(100.0)).await;
    let token2 = tracker.start(Some(50.0)).await;

    // Update both
    tracker.update(&token1, 10.0).await.unwrap();
    tracker.update(&token2, 25.0).await.unwrap();

    // Both should generate notifications
    let _n1 = rx.recv().await.unwrap();
    let _n2 = rx.recv().await.unwrap();

    // Cancel one, complete the other
    tracker.cancel(&token1).await;
    tracker.complete(&token2).await;

    assert!(tracker.is_cancelled(&token1).await);
    assert!(tracker.is_cancelled(&token2).await); // completed = removed = treated as cancelled
}
