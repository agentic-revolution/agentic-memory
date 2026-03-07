use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConservationScore {
    pub total_used: u64,
    pub cache_savings: u64,
    pub scope_savings: u64,
    pub delta_savings: u64,
    pub total_requests: u64,
    pub cache_hits: u64,
}

impl ConservationScore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn score(&self) -> f64 {
        let savings = self.cache_savings + self.scope_savings + self.delta_savings;
        let potential = self.total_used + savings;
        if potential == 0 {
            1.0
        } else {
            savings as f64 / potential as f64
        }
    }
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }
    pub fn record_request(&mut self, tokens_used: u64, tokens_saved: u64, cache_hit: bool) {
        self.total_used += tokens_used;
        self.total_requests += 1;
        if cache_hit {
            self.cache_hits += 1;
            self.cache_savings += tokens_saved;
        }
    }
}
