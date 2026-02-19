//! BM25 inverted index mapping terms to posting lists.

use std::collections::HashMap;

use crate::engine::tokenizer::Tokenizer;
use crate::graph::MemoryGraph;
use crate::types::CognitiveEvent;

/// An inverted index mapping tokenized terms to posting lists (sorted node ID arrays).
pub struct TermIndex {
    /// term → sorted Vec of (node_id, term_frequency_in_node)
    postings: HashMap<String, Vec<(u64, u32)>>,
    /// Total number of documents (nodes) indexed.
    doc_count: u64,
    /// Average document length in tokens.
    avg_doc_length: f32,
}

impl TermIndex {
    /// Create an empty term index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_count: 0,
            avg_doc_length: 0.0,
        }
    }

    /// Build the index from all node contents in the graph.
    pub fn build(graph: &MemoryGraph, tokenizer: &Tokenizer) -> Self {
        let mut index = Self::new();
        let mut total_tokens: u64 = 0;

        for node in graph.nodes() {
            let freqs = tokenizer.term_frequencies(&node.content);
            let doc_len: u32 = freqs.values().sum();
            total_tokens += doc_len as u64;

            for (term, freq) in freqs {
                let posting = index.postings.entry(term).or_default();
                // Maintain sort order by node_id
                let pos = posting
                    .binary_search_by_key(&node.id, |(id, _)| *id)
                    .unwrap_or_else(|p| p);
                posting.insert(pos, (node.id, freq));
            }

            index.doc_count += 1;
        }

        if index.doc_count > 0 {
            index.avg_doc_length = total_tokens as f32 / index.doc_count as f32;
        }

        index
    }

    /// Look up a term. Returns (node_id, term_frequency) pairs.
    pub fn get(&self, term: &str) -> &[(u64, u32)] {
        self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of nodes containing a term (document frequency).
    pub fn doc_frequency(&self, term: &str) -> usize {
        self.postings.get(term).map(|v| v.len()).unwrap_or(0)
    }

    /// Total number of indexed documents.
    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    /// Average document length.
    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }

    /// Number of unique terms.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Add a single node to the index incrementally.
    pub fn add_node(&mut self, event: &CognitiveEvent) {
        let tokenizer = Tokenizer::new();
        let freqs = tokenizer.term_frequencies(&event.content);
        for (term, freq) in freqs {
            let posting = self.postings.entry(term).or_default();
            let pos = posting
                .binary_search_by_key(&event.id, |(id, _)| *id)
                .unwrap_or_else(|p| p);
            posting.insert(pos, (event.id, freq));
        }
        self.doc_count += 1;
        // avg_doc_length becomes approximate after incremental adds
    }

    /// Remove a node from the index.
    pub fn remove_node(&mut self, id: u64) {
        for posting in self.postings.values_mut() {
            if let Ok(pos) = posting.binary_search_by_key(&id, |(nid, _)| *nid) {
                posting.remove(pos);
            }
        }
        self.doc_count = self.doc_count.saturating_sub(1);
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.postings.clear();
        self.doc_count = 0;
        self.avg_doc_length = 0.0;
    }

    /// Rebuild the index from a graph.
    pub fn rebuild(&mut self, graph: &MemoryGraph) {
        *self = Self::build(graph, &Tokenizer::new());
    }

    /// Serialize the term index to bytes for file writing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        buf.extend_from_slice(&self.doc_count.to_le_bytes());
        buf.extend_from_slice(&self.avg_doc_length.to_le_bytes());
        buf.extend_from_slice(&(self.postings.len() as u32).to_le_bytes());

        // Sort terms for deterministic output
        let mut terms: Vec<&String> = self.postings.keys().collect();
        terms.sort();

        for term in terms {
            let postings = &self.postings[term];
            let term_bytes = term.as_bytes();
            buf.extend_from_slice(&(term_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(term_bytes);
            buf.extend_from_slice(&(postings.len() as u32).to_le_bytes());
            for &(node_id, term_freq) in postings {
                buf.extend_from_slice(&node_id.to_le_bytes());
                buf.extend_from_slice(&term_freq.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize a term index from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let mut pos = 0;

        if data.len() < 16 {
            return None;
        }

        let doc_count = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let avg_doc_length = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let term_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut postings = HashMap::with_capacity(term_count);

        for _ in 0..term_count {
            if pos + 2 > data.len() {
                return None;
            }
            let term_len = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2;

            if pos + term_len > data.len() {
                return None;
            }
            let term = std::str::from_utf8(&data[pos..pos + term_len])
                .ok()?
                .to_string();
            pos += term_len;

            if pos + 4 > data.len() {
                return None;
            }
            let posting_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;

            let mut posting_list = Vec::with_capacity(posting_count);
            for _ in 0..posting_count {
                if pos + 12 > data.len() {
                    return None;
                }
                let node_id = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
                let term_freq = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
                pos += 4;
                posting_list.push((node_id, term_freq));
            }

            postings.insert(term, posting_list);
        }

        Some(Self {
            postings,
            doc_count,
            avg_doc_length,
        })
    }
}

impl Default for TermIndex {
    fn default() -> Self {
        Self::new()
    }
}
