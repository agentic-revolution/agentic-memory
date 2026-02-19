//! Document length table for BM25 normalization.

use crate::engine::tokenizer::Tokenizer;
use crate::graph::MemoryGraph;
use crate::types::CognitiveEvent;

/// Document length table for BM25 normalization.
/// Stores the token count for each node's content indexed by node ID (dense array).
pub struct DocLengths {
    /// node_id → token count (document length). Indexed by node ID.
    lengths: Vec<u32>,
}

impl DocLengths {
    /// Create an empty doc lengths table.
    pub fn new() -> Self {
        Self {
            lengths: Vec::new(),
        }
    }

    /// Build from graph — tokenize all content and count tokens.
    pub fn build(graph: &MemoryGraph, tokenizer: &Tokenizer) -> Self {
        let mut lengths = Vec::new();

        for node in graph.nodes() {
            let id = node.id as usize;
            if id >= lengths.len() {
                lengths.resize(id + 1, 0);
            }
            lengths[id] = tokenizer.tokenize(&node.content).len() as u32;
        }

        Self { lengths }
    }

    /// Get token count for a node.
    pub fn get(&self, node_id: u64) -> u32 {
        let idx = node_id as usize;
        if idx < self.lengths.len() {
            self.lengths[idx]
        } else {
            0
        }
    }

    /// Average document length.
    pub fn average(&self) -> f32 {
        let non_zero: Vec<u32> = self.lengths.iter().filter(|&&l| l > 0).copied().collect();
        if non_zero.is_empty() {
            0.0
        } else {
            non_zero.iter().sum::<u32>() as f32 / non_zero.len() as f32
        }
    }

    /// Number of documents with non-zero length.
    pub fn len(&self) -> usize {
        self.lengths.iter().filter(|&&l| l > 0).count()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a single node's document length.
    pub fn add_node(&mut self, event: &CognitiveEvent) {
        let count = Tokenizer::new().tokenize(&event.content).len() as u32;
        let id = event.id as usize;
        if id >= self.lengths.len() {
            self.lengths.resize(id + 1, 0);
        }
        self.lengths[id] = count;
    }

    /// Remove a node's document length.
    pub fn remove_node(&mut self, id: u64) {
        let idx = id as usize;
        if idx < self.lengths.len() {
            self.lengths[idx] = 0;
        }
    }

    /// Clear all lengths.
    pub fn clear(&mut self) {
        self.lengths.clear();
    }

    /// Rebuild from a graph.
    pub fn rebuild(&mut self, graph: &MemoryGraph) {
        *self = Self::build(graph, &Tokenizer::new());
    }

    /// Serialize to bytes for file writing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(self.lengths.len() as u64).to_le_bytes());
        for &len in &self.lengths {
            buf.extend_from_slice(&len.to_le_bytes());
        }
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        let count = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
        let expected_size = 8 + count * 4;
        if data.len() < expected_size {
            return None;
        }

        let mut lengths = Vec::with_capacity(count);
        for i in 0..count {
            let offset = 8 + i * 4;
            let len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            lengths.push(len);
        }

        Some(Self { lengths })
    }
}

impl Default for DocLengths {
    fn default() -> Self {
        Self::new()
    }
}
