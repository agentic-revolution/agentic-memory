//! Transport layer — I/O for stdio and SSE.

pub mod framing;
pub mod stdio;

#[cfg(feature = "sse")]
pub mod sse;

pub use stdio::StdioTransport;

#[cfg(feature = "sse")]
pub use sse::SseTransport;
