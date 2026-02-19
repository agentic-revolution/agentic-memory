//! Shared tokenizer for BM25 text search and index building.

use std::collections::{HashMap, HashSet};

/// Stop words to exclude from tokenization.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "need", "must", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "about", "but", "not", "or", "and", "if", "it", "its", "this", "that", "which", "who", "what",
    "when", "where", "how", "all", "each", "both", "few", "more", "most", "other", "some", "such",
    "no", "than", "too", "very", "just", "also",
];

/// Deterministic tokenizer for BM25 text search.
pub struct Tokenizer {
    stop_words: HashSet<&'static str>,
}

impl Tokenizer {
    /// Create a new tokenizer with the default stop word list.
    pub fn new() -> Self {
        Self {
            stop_words: STOP_WORDS.iter().copied().collect(),
        }
    }

    /// Tokenize text into lowercase terms, excluding stop words and short tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|token| token.len() >= 2)
            .filter(|token| !self.stop_words.contains(token))
            .map(|s| s.to_string())
            .collect()
    }

    /// Tokenize and return term frequencies.
    pub fn term_frequencies(&self, text: &str) -> HashMap<String, u32> {
        let mut freqs = HashMap::new();
        for token in self.tokenize(text) {
            *freqs.entry(token).or_insert(0) += 1;
        }
        freqs
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}
