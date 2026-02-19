//! Phase 4: Session management tests.

mod common;

use serde_json::json;

use agentic_memory_mcp::session::Transaction;
use agentic_memory_mcp::tools::ToolRegistry;

use common::fixtures::create_test_session;

#[tokio::test]
async fn test_session_start_and_end() {
    let session = create_test_session();

    // Start a new session
    let result = ToolRegistry::call("session_start", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let session_id = parsed["session_id"].as_u64().unwrap();
    assert!(session_id > 0);

    // Add a fact in this session
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Session test fact"})),
        &session,
    )
    .await
    .unwrap();

    // End the session with episode creation
    let result = ToolRegistry::call(
        "session_end",
        Some(json!({
            "create_episode": true,
            "summary": "Test session completed"
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
    assert!(parsed["episode_node_id"].as_u64().is_some());
}

#[tokio::test]
async fn test_auto_save_triggers() {
    let session = create_test_session();

    // Add data to mark dirty
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Auto-save test"})),
        &session,
    )
    .await
    .unwrap();

    // Manually trigger maybe_auto_save
    {
        let mut sess = session.lock().await;
        sess.mark_dirty();
        // The interval won't have elapsed yet, so this should be a no-op
        sess.maybe_auto_save().unwrap();
    }

    // Verify the session still works after auto-save check
    let result = ToolRegistry::call("memory_stats", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["node_count"], 1);
}

#[tokio::test]
async fn test_transaction_batching() {
    let session = create_test_session();

    {
        let mut sess = session.lock().await;

        let event1 = agentic_memory::CognitiveEventBuilder::new(
            agentic_memory::EventType::Fact,
            "Batch fact 1".to_string(),
        )
        .session_id(1)
        .build();

        let event2 = agentic_memory::CognitiveEventBuilder::new(
            agentic_memory::EventType::Decision,
            "Batch decision 1".to_string(),
        )
        .session_id(1)
        .build();

        let mut tx = Transaction::new(&mut sess);
        tx.add_node(event1);
        tx.add_node(event2);
        let node_ids = tx.commit().unwrap();

        assert_eq!(node_ids.len(), 2);
    }

    // Verify both nodes were added
    let result = ToolRegistry::call("memory_stats", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.content[0] {
        agentic_memory_mcp::types::ToolContent::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["node_count"], 2);
}

#[tokio::test]
async fn test_explicit_save() {
    let session = create_test_session();

    // Add data
    ToolRegistry::call(
        "memory_add",
        Some(json!({"event_type": "fact", "content": "Save test"})),
        &session,
    )
    .await
    .unwrap();

    // Explicitly save
    {
        let mut sess = session.lock().await;
        sess.save().unwrap();
    }

    // Verify file path is accessible
    {
        let sess = session.lock().await;
        assert!(sess.file_path().exists());
    }
}

#[tokio::test]
async fn test_session_id_continuity() {
    let session = create_test_session();

    // Get initial session ID
    let initial_id = {
        let sess = session.lock().await;
        sess.current_session_id()
    };

    // Start a new session
    {
        let mut sess = session.lock().await;
        let new_id = sess.start_session(None).unwrap();
        assert!(new_id >= initial_id);
    }
}

#[tokio::test]
async fn test_drop_saves_dirty() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("drop_test.amem");
    let path_str = path.display().to_string();

    // Create session, add data, and let it drop
    {
        let mut session = agentic_memory_mcp::session::SessionManager::open(&path_str).unwrap();
        session
            .add_event(
                agentic_memory::EventType::Fact,
                "Drop save test",
                0.9,
                vec![],
            )
            .unwrap();
        // Session drops here, triggering save
    }

    // Reopen and verify data persisted
    let session = agentic_memory_mcp::session::SessionManager::open(&path_str).unwrap();
    assert_eq!(session.graph().node_count(), 1);
}
