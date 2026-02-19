//! MCP capability negotiation during initialization.

use crate::types::{
    ClientCapabilities, InitializeParams, InitializeResult, McpError, McpResult, MCP_VERSION,
};

/// Stored client capabilities after negotiation.
#[derive(Debug, Clone, Default)]
pub struct NegotiatedCapabilities {
    /// The client's declared capabilities.
    pub client: ClientCapabilities,
    /// Whether the handshake is complete.
    pub initialized: bool,
}

impl NegotiatedCapabilities {
    /// Process an initialize request and return the result.
    pub fn negotiate(&mut self, params: InitializeParams) -> McpResult<InitializeResult> {
        // Verify protocol version compatibility
        if params.protocol_version != MCP_VERSION {
            tracing::warn!(
                "Client requested protocol version {}, server supports {}. Proceeding with server version.",
                params.protocol_version,
                MCP_VERSION
            );
        }

        self.client = params.capabilities;

        tracing::info!(
            "Initialized with client: {} v{}",
            params.client_info.name,
            params.client_info.version
        );

        Ok(InitializeResult::default_result())
    }

    /// Mark the handshake as complete (after receiving `initialized` notification).
    pub fn mark_initialized(&mut self) -> McpResult<()> {
        self.initialized = true;
        tracing::info!("MCP handshake complete");
        Ok(())
    }

    /// Check that the handshake is complete before processing requests.
    pub fn ensure_initialized(&self) -> McpResult<()> {
        if !self.initialized {
            return Err(McpError::InvalidRequest(
                "Server not yet initialized. Send 'initialize' first.".to_string(),
            ));
        }
        Ok(())
    }
}
