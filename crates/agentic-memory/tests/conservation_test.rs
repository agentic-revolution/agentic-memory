//! Conservation tests for AgenticMemory.
//!
//! Exercises cache, query, and metrics modules to verify token conservation
//! properties hold across the foundation layer.

use std::time::Duration;

use agentic_memory::cache::LruCache;
use agentic_memory::metrics::tokens::{Layer, TokenMetrics};
use agentic_memory::query::budget::TokenBudget;
use agentic_memory::query::intent::ExtractionIntent;

// ---------------------------------------------------------------------------
// Test 1: Cache hit is cheaper than miss
// ---------------------------------------------------------------------------

#[test]
fn test_cache_hit_cheaper() {
    let mut cache: LruCache<String, String> = LruCache::new(100, Duration::from_secs(300));

    // First access: miss
    assert!(cache.get(&"key1".to_string()).is_none());

    // Insert
    cache.insert("key1".to_string(), "value1".to_string());

    // Second access: hit (0 token cost)
    assert!(cache.get(&"key1".to_string()).is_some());

    // Verify metrics
    assert!(cache.metrics().hits() >= 1);
    assert!(cache.metrics().misses() >= 1);
    assert!(cache.metrics().hit_rate() > 0.0);
}

// ---------------------------------------------------------------------------
// Test 2: Scoped query cheaper than full
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_cheaper_than_full() {
    let minimal = ExtractionIntent::IdsOnly;
    let full = ExtractionIntent::Full;

    assert!(minimal.estimated_tokens() < full.estimated_tokens());
    assert!(
        minimal.estimated_tokens() < full.estimated_tokens() / 10,
        "IdsOnly should be at least 10x cheaper than Full"
    );
    assert!(minimal.is_minimal());
    assert!(!full.is_minimal());
    assert!(full.is_full());
}

// ---------------------------------------------------------------------------
// Test 3: Budget enforcement
// ---------------------------------------------------------------------------

#[test]
fn test_budget_enforcement() {
    let mut budget = TokenBudget::new(100);
    assert!(!budget.is_exhausted());
    assert_eq!(budget.remaining(), 100);

    // Spend within budget
    assert!(budget.spend(30));
    assert_eq!(budget.remaining(), 70);
    assert!(!budget.is_exhausted());

    // Spend more than remaining — spend still occurs but returns false
    assert!(!budget.spend(80));
    assert!(budget.is_exhausted());
}

// ---------------------------------------------------------------------------
// Test 4: Conservation score
// ---------------------------------------------------------------------------

#[test]
fn test_conservation_score() {
    let metrics = TokenMetrics::new();

    // Full retrieval: 100 tokens used, potential was 100
    metrics.record(Layer::Full, 100, 100);

    // Cache hit: 0 tokens used, potential was 400
    metrics.record(Layer::Cache, 0, 400);

    // total_tokens = 100, cache_savings = 400, total_savings = 400
    // conservation = 400 / (100 + 400) = 0.8
    let score = metrics.conservation_score();
    assert!(
        score > 0.7 && score < 0.9,
        "Conservation score should be ~0.8, got {}",
        score
    );
}

// ---------------------------------------------------------------------------
// Test 5: Cache invalidation
// ---------------------------------------------------------------------------

#[test]
fn test_cache_invalidation() {
    let mut cache: LruCache<String, i32> = LruCache::new(100, Duration::from_secs(300));

    cache.insert("key".to_string(), 42);
    assert!(cache.contains(&"key".to_string()));

    // Invalidate the entry
    assert!(cache.invalidate(&"key".to_string()));
    assert!(!cache.contains(&"key".to_string()));

    // Double invalidation returns false
    assert!(!cache.invalidate(&"key".to_string()));
}
