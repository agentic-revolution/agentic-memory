//! JSON-RPC message validation per MCP spec.

use crate::types::{JsonRpcRequest, McpError, McpResult, JSONRPC_VERSION};

/// Validate that a JSON-RPC request is well-formed.
pub fn validate_request(request: &JsonRpcRequest) -> McpResult<()> {
    if request.jsonrpc != JSONRPC_VERSION {
        return Err(McpError::InvalidRequest(format!(
            "Expected jsonrpc version \"{JSONRPC_VERSION}\", got \"{}\"",
            request.jsonrpc
        )));
    }

    if request.method.is_empty() {
        return Err(McpError::InvalidRequest(
            "Method name must not be empty".to_string(),
        ));
    }

    Ok(())
}
