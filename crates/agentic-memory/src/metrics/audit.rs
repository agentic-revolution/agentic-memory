use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use super::tokens::Layer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub tool: String,
    pub layer: Layer,
    pub tokens_used: u64,
    pub tokens_saved: u64,
    pub cache_hit: bool,
}

impl AuditEntry {
    pub fn new(tool: String, layer: Layer, tokens_used: u64, tokens_saved: u64) -> Self {
        Self { timestamp: Utc::now(), tool, layer, tokens_used, tokens_saved, cache_hit: matches!(layer, Layer::Cache) }
    }
}

pub struct AuditLog { entries: Vec<AuditEntry>, max_entries: usize }

impl AuditLog {
    pub fn new(max_entries: usize) -> Self { Self { entries: Vec::new(), max_entries } }
    pub fn record(&mut self, entry: AuditEntry) {
        self.entries.push(entry);
        if self.entries.len() > self.max_entries { self.entries.drain(..self.entries.len() - self.max_entries); }
    }
    pub fn entries(&self) -> &[AuditEntry] { &self.entries }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn total_tokens_used(&self) -> u64 { self.entries.iter().map(|e| e.tokens_used).sum() }
    pub fn total_tokens_saved(&self) -> u64 { self.entries.iter().map(|e| e.tokens_saved).sum() }
}

impl Default for AuditLog { fn default() -> Self { Self::new(10_000) } }
