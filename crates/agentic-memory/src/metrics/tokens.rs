use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layer { Cache, Index, Scoped, Delta, Full }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetrics {
    pub layer: Layer,
    pub tokens_used: u64,
    pub tokens_saved: u64,
    pub cache_hit: bool,
}

impl ResponseMetrics {
    pub fn from_cache(full_cost: u64) -> Self { Self { layer: Layer::Cache, tokens_used: 0, tokens_saved: full_cost, cache_hit: true } }
    pub fn from_query(layer: Layer, tokens: u64, full_cost: u64) -> Self { Self { layer, tokens_used: tokens, tokens_saved: full_cost.saturating_sub(tokens), cache_hit: false } }
}

pub struct TokenMetrics {
    pub total: AtomicU64,
    pub cache_savings: AtomicU64,
    pub scope_savings: AtomicU64,
    pub delta_savings: AtomicU64,
}

impl TokenMetrics {
    pub fn new() -> Self { Self { total: AtomicU64::new(0), cache_savings: AtomicU64::new(0), scope_savings: AtomicU64::new(0), delta_savings: AtomicU64::new(0) } }
    pub fn record(&self, layer: Layer, tokens: u64, potential: u64) {
        self.total.fetch_add(tokens, Ordering::Relaxed);
        let saved = potential.saturating_sub(tokens);
        match layer {
            Layer::Cache => { self.cache_savings.fetch_add(saved, Ordering::Relaxed); }
            Layer::Scoped => { self.scope_savings.fetch_add(saved, Ordering::Relaxed); }
            Layer::Delta => { self.delta_savings.fetch_add(saved, Ordering::Relaxed); }
            _ => {}
        }
    }
    pub fn total_tokens(&self) -> u64 { self.total.load(Ordering::Relaxed) }
    pub fn total_savings(&self) -> u64 { self.cache_savings.load(Ordering::Relaxed) + self.scope_savings.load(Ordering::Relaxed) + self.delta_savings.load(Ordering::Relaxed) }
    pub fn conservation_score(&self) -> f64 {
        let total = self.total_tokens();
        let saved = self.total_savings();
        let potential = total + saved;
        if potential == 0 { 1.0 } else { saved as f64 / potential as f64 }
    }
}

impl Default for TokenMetrics { fn default() -> Self { Self::new() } }
