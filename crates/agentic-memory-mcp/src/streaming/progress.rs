//! Progress token handling for long-running operations.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};

use crate::types::{JsonRpcNotification, McpResult, ProgressParams, ProgressToken};

/// State of a tracked progress operation.
#[derive(Debug)]
struct ProgressState {
    total: Option<f64>,
    current: f64,
    cancelled: bool,
}

/// Tracks progress for long-running operations and sends notifications.
pub struct ProgressTracker {
    active: Arc<RwLock<HashMap<String, ProgressState>>>,
    notification_tx: mpsc::Sender<JsonRpcNotification>,
}

impl ProgressTracker {
    /// Create a new progress tracker with a notification channel.
    pub fn new(notification_tx: mpsc::Sender<JsonRpcNotification>) -> Self {
        Self {
            active: Arc::new(RwLock::new(HashMap::new())),
            notification_tx,
        }
    }

    /// Start tracking a new operation. Returns a unique token.
    pub async fn start(&self, total: Option<f64>) -> String {
        let token = uuid::Uuid::new_v4().to_string();
        let state = ProgressState {
            total,
            current: 0.0,
            cancelled: false,
        };
        self.active.write().await.insert(token.clone(), state);
        token
    }

    /// Update the progress of an operation.
    pub async fn update(&self, token: &str, current: f64) -> McpResult<()> {
        let total = {
            let mut active = self.active.write().await;
            if let Some(state) = active.get_mut(token) {
                state.current = current;
                state.total
            } else {
                return Ok(());
            }
        };

        let params = ProgressParams {
            progress_token: ProgressToken::String(token.to_string()),
            progress: current,
            total,
        };

        let notification = JsonRpcNotification::new(
            "notifications/progress".to_string(),
            Some(serde_json::to_value(params).unwrap_or_default()),
        );

        let _ = self.notification_tx.send(notification).await;
        Ok(())
    }

    /// Mark an operation as cancelled.
    pub async fn cancel(&self, token: &str) {
        let mut active = self.active.write().await;
        if let Some(state) = active.get_mut(token) {
            state.cancelled = true;
        }
    }

    /// Complete and remove an operation.
    pub async fn complete(&self, token: &str) {
        self.active.write().await.remove(token);
    }

    /// Check if an operation has been cancelled.
    pub async fn is_cancelled(&self, token: &str) -> bool {
        self.active
            .read()
            .await
            .get(token)
            .map(|s| s.cancelled)
            .unwrap_or(true)
    }
}
