//! AgenticMemory MCP Server — universal LLM access to persistent graph memory.
//!
//! This library implements an MCP (Model Context Protocol) server that exposes
//! AgenticMemory functionality to any MCP-compatible LLM client.

pub mod config;
pub mod prompts;
pub mod protocol;
pub mod resources;
pub mod session;
pub mod streaming;
pub mod tools;
pub mod transport;
pub mod types;

pub use config::ServerConfig;
pub use protocol::ProtocolHandler;
pub use session::SessionManager;
pub use transport::StdioTransport;
