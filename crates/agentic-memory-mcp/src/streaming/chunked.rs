//! Chunked result streaming for large traversals.

use serde_json::Value;

/// Split a large result set into chunks for streaming.
pub fn chunk_results(results: Vec<Value>, chunk_size: usize) -> Vec<Vec<Value>> {
    results.chunks(chunk_size).map(|c| c.to_vec()).collect()
}
