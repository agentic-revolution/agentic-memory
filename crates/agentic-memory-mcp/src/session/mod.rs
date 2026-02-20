//! Session management: graph lifecycle, transactions, and auto-save.

pub mod autosave;
pub mod manager;
#[cfg(feature = "sse")]
pub mod tenant;
pub mod transaction;

pub use manager::SessionManager;
pub use transaction::Transaction;
