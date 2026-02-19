//! Configuration loading and resolution.

pub mod loader;

pub use loader::{load_config, resolve_memory_path, ServerConfig};
