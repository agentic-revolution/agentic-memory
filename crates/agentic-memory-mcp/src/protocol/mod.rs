//! MCP protocol layer — message handling, validation, and capability negotiation.

pub mod handler;
pub mod negotiation;
pub mod validator;

pub use handler::ProtocolHandler;
