use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    current_size: AtomicUsize,
}

impl CacheMetrics {
    pub fn new() -> Self {
        Self { hits: AtomicU64::new(0), misses: AtomicU64::new(0), evictions: AtomicU64::new(0), current_size: AtomicUsize::new(0) }
    }
    pub fn record_hit(&self) { self.hits.fetch_add(1, Ordering::Relaxed); }
    pub fn record_miss(&self) { self.misses.fetch_add(1, Ordering::Relaxed); }
    pub fn record_eviction(&self) { self.evictions.fetch_add(1, Ordering::Relaxed); }
    pub fn set_size(&self, size: usize) { self.current_size.store(size, Ordering::Relaxed); }
    pub fn hits(&self) -> u64 { self.hits.load(Ordering::Relaxed) }
    pub fn misses(&self) -> u64 { self.misses.load(Ordering::Relaxed) }
    pub fn evictions(&self) -> u64 { self.evictions.load(Ordering::Relaxed) }
    pub fn current_size(&self) -> usize { self.current_size.load(Ordering::Relaxed) }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits() + self.misses();
        if total == 0 { 0.0 } else { self.hits() as f64 / total as f64 }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self { Self::new() }
}
