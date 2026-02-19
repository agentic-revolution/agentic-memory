//! MCP resource implementations — read-only views of the memory graph.

pub mod graph;
pub mod node;
pub mod registry;
pub mod session;
pub mod templates;
pub mod type_index;

pub use registry::ResourceRegistry;
