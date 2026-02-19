//! Phase 1: Type tests — JSON-RPC message types, error codes, capabilities.

use serde_json::json;

use agentic_memory_mcp::types::*;

#[test]
fn test_request_id_string() {
    let id = RequestId::String("test-123".to_string());
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "\"test-123\"");

    let parsed: RequestId = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, id);
}

#[test]
fn test_request_id_number() {
    let id = RequestId::Number(42);
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "42");

    let parsed: RequestId = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, id);
}

#[test]
fn test_request_id_null() {
    let id = RequestId::Null;
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "null");
}

#[test]
fn test_jsonrpc_request_parsing() {
    let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.jsonrpc, "2.0");
    assert_eq!(req.id, RequestId::Number(1));
    assert_eq!(req.method, "initialize");
}

#[test]
fn test_jsonrpc_response_creation() {
    let resp = JsonRpcResponse::new(RequestId::Number(1), json!({"status": "ok"}));
    assert_eq!(resp.jsonrpc, "2.0");
    assert_eq!(resp.id, RequestId::Number(1));
    assert_eq!(resp.result["status"], "ok");
}

#[test]
fn test_jsonrpc_error_creation() {
    let err = JsonRpcError::new(RequestId::Number(1), -32600, "Invalid request".to_string());
    assert_eq!(err.error.code, -32600);
    assert_eq!(err.error.message, "Invalid request");
}

#[test]
fn test_jsonrpc_notification_no_id() {
    let notif = JsonRpcNotification::new("initialized".to_string(), None);
    let json = serde_json::to_string(&notif).unwrap();
    assert!(!json.contains("\"id\""));
    assert!(json.contains("\"initialized\""));
}

#[test]
fn test_mcp_error_codes() {
    let err = McpError::ParseError("bad json".to_string());
    assert_eq!(err.code(), error_codes::PARSE_ERROR);

    let err = McpError::MethodNotFound("foo".to_string());
    assert_eq!(err.code(), error_codes::METHOD_NOT_FOUND);

    let err = McpError::ToolNotFound("bar".to_string());
    assert_eq!(err.code(), mcp_error_codes::TOOL_NOT_FOUND);

    let err = McpError::NodeNotFound(42);
    assert_eq!(err.code(), mcp_error_codes::NODE_NOT_FOUND);

    let err = McpError::SessionNotFound(1);
    assert_eq!(err.code(), mcp_error_codes::SESSION_NOT_FOUND);
}

#[test]
fn test_mcp_error_to_json_rpc() {
    let err = McpError::ToolNotFound("unknown_tool".to_string());
    let rpc_err = err.to_json_rpc_error(RequestId::Number(5));
    assert_eq!(rpc_err.id, RequestId::Number(5));
    assert_eq!(rpc_err.error.code, mcp_error_codes::TOOL_NOT_FOUND);
    assert!(rpc_err.error.message.contains("unknown_tool"));
}

#[test]
fn test_server_capabilities() {
    let caps = ServerCapabilities::default_capabilities();
    assert!(caps.tools.is_some());
    assert!(caps.resources.is_some());
    assert!(caps.prompts.is_some());
    assert!(caps.logging.is_some());
}

#[test]
fn test_initialize_result() {
    let result = InitializeResult::default_result();
    assert_eq!(result.protocol_version, MCP_VERSION);
    assert_eq!(result.server_info.name, SERVER_NAME);
    assert!(result.instructions.is_some());
}

#[test]
fn test_tool_call_result_text() {
    let result = ToolCallResult::text("hello".to_string());
    assert_eq!(result.content.len(), 1);
    assert!(result.is_error.is_none());
}

#[test]
fn test_tool_call_result_error() {
    let result = ToolCallResult::error("something failed".to_string());
    assert_eq!(result.is_error, Some(true));
}

#[test]
fn test_tool_definition_serialization() {
    let def = ToolDefinition {
        name: "test".to_string(),
        description: Some("A test tool".to_string()),
        input_schema: json!({"type": "object"}),
    };
    let json = serde_json::to_value(&def).unwrap();
    assert_eq!(json["name"], "test");
    assert_eq!(json["inputSchema"]["type"], "object");
}

#[test]
fn test_resource_definition_serialization() {
    let def = ResourceDefinition {
        uri: "amem://graph/stats".to_string(),
        name: "Stats".to_string(),
        description: Some("Graph stats".to_string()),
        mime_type: Some("application/json".to_string()),
    };
    let json = serde_json::to_value(&def).unwrap();
    assert_eq!(json["uri"], "amem://graph/stats");
    assert_eq!(json["mimeType"], "application/json");
}

#[test]
fn test_prompt_definition_serialization() {
    let def = PromptDefinition {
        name: "remember".to_string(),
        description: Some("Remember info".to_string()),
        arguments: Some(vec![PromptArgument {
            name: "information".to_string(),
            description: Some("What to remember".to_string()),
            required: true,
        }]),
    };
    let json = serde_json::to_value(&def).unwrap();
    assert_eq!(json["name"], "remember");
    assert!(json["arguments"].is_array());
}
