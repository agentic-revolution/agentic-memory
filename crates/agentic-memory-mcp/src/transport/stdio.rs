//! Stdio transport — reads JSON-RPC from stdin, writes to stdout.

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::protocol::ProtocolHandler;
use crate::types::{JsonRpcError, McpError, McpResult, RequestId, JSONRPC_VERSION};

use super::framing;

/// Stdio transport for desktop MCP clients.
pub struct StdioTransport {
    handler: ProtocolHandler,
}

impl StdioTransport {
    /// Create a new stdio transport with the given handler.
    pub fn new(handler: ProtocolHandler) -> Self {
        Self { handler }
    }

    /// Run the transport loop — reads from stdin, writes to stdout.
    pub async fn run(&self) -> McpResult<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        tracing::info!("Stdio transport started");

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await.map_err(McpError::Io)?;

            if bytes_read == 0 {
                tracing::info!("EOF on stdin, shutting down");
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match framing::parse_message(trimmed) {
                Ok(msg) => {
                    if let Some(response) = self.handler.handle_message(msg).await {
                        let framed = framing::frame_message(&response)?;
                        stdout
                            .write_all(framed.as_bytes())
                            .await
                            .map_err(McpError::Io)?;
                        stdout.flush().await.map_err(McpError::Io)?;
                    }
                }
                Err(e) => {
                    tracing::warn!("Parse error: {e}");
                    let error_response = JsonRpcError {
                        jsonrpc: JSONRPC_VERSION.to_string(),
                        id: RequestId::Null,
                        error: crate::types::JsonRpcErrorObject {
                            code: e.code(),
                            message: e.to_string(),
                            data: None,
                        },
                    };
                    let value = serde_json::to_value(error_response)
                        .map_err(|e| McpError::InternalError(e.to_string()))?;
                    let framed = framing::frame_message(&value)?;
                    stdout
                        .write_all(framed.as_bytes())
                        .await
                        .map_err(McpError::Io)?;
                    stdout.flush().await.map_err(McpError::Io)?;
                }
            }
        }

        Ok(())
    }
}
