//! Test data fixtures for MCP server tests.

use std::sync::Arc;
use tokio::sync::Mutex;

use agentic_memory_mcp::session::SessionManager;

/// Create a temporary session manager for testing.
pub fn create_test_session() -> Arc<Mutex<SessionManager>> {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("test.amem");
    // Keep the tempdir alive by leaking it (tests are short-lived).
    let path_str = path.display().to_string();
    std::mem::forget(dir);

    let session = SessionManager::open(&path_str).expect("Failed to create test session");
    Arc::new(Mutex::new(session))
}
