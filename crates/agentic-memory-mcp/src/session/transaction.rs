//! Batched write transactions for high-throughput scenarios.

use super::manager::SessionManager;
use crate::types::McpResult;
use agentic_memory::{CognitiveEvent, Edge, WriteEngine};

/// A batched transaction that collects operations and commits them in one save.
pub struct Transaction<'a> {
    session: &'a mut SessionManager,
    events: Vec<CognitiveEvent>,
    edges: Vec<Edge>,
}

impl<'a> Transaction<'a> {
    /// Create a new transaction on the given session manager.
    pub fn new(session: &'a mut SessionManager) -> Self {
        Self {
            session,
            events: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Queue a node to be added.
    pub fn add_node(&mut self, event: CognitiveEvent) -> &mut Self {
        self.events.push(event);
        self
    }

    /// Queue an edge to be added.
    pub fn add_edge(&mut self, edge: Edge) -> &mut Self {
        self.edges.push(edge);
        self
    }

    /// Commit all queued operations in a single batch, then save.
    pub fn commit(self) -> McpResult<Vec<u64>> {
        let dimension = self.session.graph().dimension();
        let write_engine = WriteEngine::new(dimension);

        let result = write_engine
            .ingest(self.session.graph_mut(), self.events, self.edges)
            .map_err(|e| {
                crate::types::McpError::AgenticMemory(format!("Transaction commit failed: {e}"))
            })?;

        self.session.mark_dirty();
        self.session.save()?;

        Ok(result.new_node_ids)
    }
}
