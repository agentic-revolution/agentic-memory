//! All MCP data types used by the server.

pub mod capabilities;
pub mod error;
pub mod message;
pub mod notification;
pub mod request;
pub mod response;

// Re-export commonly used types for convenience.
pub use capabilities::*;
pub use error::*;
pub use message::*;
pub use notification::*;
pub use request::*;
pub use response::*;
