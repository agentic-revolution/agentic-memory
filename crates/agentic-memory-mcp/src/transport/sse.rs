//! SSE transport — Server-Sent Events over HTTP for web-based MCP clients.

#[cfg(feature = "sse")]
use std::sync::Arc;

#[cfg(feature = "sse")]
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};

#[cfg(feature = "sse")]
use tokio::sync::Mutex;

#[cfg(feature = "sse")]
use crate::protocol::ProtocolHandler;
#[cfg(feature = "sse")]
use crate::types::McpResult;

/// SSE transport for web-based MCP clients.
#[cfg(feature = "sse")]
pub struct SseTransport {
    handler: Arc<ProtocolHandler>,
}

#[cfg(feature = "sse")]
impl SseTransport {
    /// Create a new SSE transport.
    pub fn new(handler: ProtocolHandler) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }

    /// Run the SSE server on the given address.
    pub async fn run(&self, addr: &str) -> McpResult<()> {
        let handler = self.handler.clone();

        let app = Router::new()
            .route("/mcp", post(Self::handle_request))
            .route("/health", get(|| async { "ok" }))
            .with_state(handler);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(crate::types::McpError::Io)?;

        tracing::info!("SSE transport listening on {addr}");

        axum::serve(listener, app)
            .await
            .map_err(|e| crate::types::McpError::Transport(e.to_string()))?;

        Ok(())
    }

    async fn handle_request(
        State(handler): State<Arc<ProtocolHandler>>,
        Json(body): Json<serde_json::Value>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let msg: crate::types::JsonRpcMessage =
            serde_json::from_value(body).map_err(|_| StatusCode::BAD_REQUEST)?;

        match handler.handle_message(msg).await {
            Some(response) => Ok(Json(response)),
            None => Ok(Json(serde_json::Value::Null)),
        }
    }
}
