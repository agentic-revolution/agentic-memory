//! Periodic auto-save background task.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use super::manager::SessionManager;

/// Spawn a background task that periodically auto-saves the session.
pub fn spawn_autosave(
    session: Arc<Mutex<SessionManager>>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let mut session = session.lock().await;
            if let Err(e) = session.maybe_auto_save() {
                tracing::error!("Auto-save failed: {e}");
            }
        }
    })
}
