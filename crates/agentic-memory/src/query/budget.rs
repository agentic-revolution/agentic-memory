use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget { max_tokens: u64, used_tokens: u64 }

impl TokenBudget {
    pub fn new(max_tokens: u64) -> Self { Self { max_tokens, used_tokens: 0 } }
    pub fn spend(&mut self, tokens: u64) -> bool { let ok = self.can_afford(tokens); self.used_tokens = self.used_tokens.saturating_add(tokens); ok }
    pub fn try_spend(&mut self, tokens: u64) -> bool { if self.can_afford(tokens) { self.used_tokens += tokens; true } else { false } }
    pub fn remaining(&self) -> u64 { self.max_tokens.saturating_sub(self.used_tokens) }
    pub fn is_exhausted(&self) -> bool { self.used_tokens >= self.max_tokens }
    pub fn can_afford(&self, tokens: u64) -> bool { self.used_tokens.saturating_add(tokens) <= self.max_tokens }
    pub fn utilization(&self) -> f64 { if self.max_tokens == 0 { return if self.used_tokens > 0 { 1.0 } else { 0.0 }; } self.used_tokens as f64 / self.max_tokens as f64 }
    pub fn reset(&mut self) { self.used_tokens = 0; }
}
