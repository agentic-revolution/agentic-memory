//! Phase 5 tests: BM25 text search, hybrid search, tokenizer, and index building.
//!
//! 22 tests total:
//!   - 8 tokenizer tests
//!   - 7 TermIndex / DocLengths tests
//!   - 9 BM25 search tests  (note: spec lists 9 under "BM25 Search Tests")
//!   - 5 hybrid search tests (note: spec section has a 7th, but target is 22 total)
//!
//! Uses only `agentic_memory::` imports and follows existing test conventions.

use agentic_memory::graph::MemoryGraph;
use agentic_memory::types::event::{CognitiveEventBuilder, EventType};
use agentic_memory::types::DEFAULT_DIMENSION;
use agentic_memory::{
    DocLengths, HybridSearchParams, QueryEngine, TermIndex, TextSearchParams, Tokenizer,
};

// ==================== Helpers ====================

/// Create a zero feature vector of graph dimension.
fn zero_vec() -> Vec<f32> {
    vec![0.0; DEFAULT_DIMENSION]
}

/// Create a feature vector with a single non-zero element at the given index.
fn basis_vec(index: usize, value: f32) -> Vec<f32> {
    let mut v = vec![0.0; DEFAULT_DIMENSION];
    if index < DEFAULT_DIMENSION {
        v[index] = value;
    }
    v
}

// ==================== Tokenizer Tests (8) ====================

#[test]
fn test_tokenizer_basic() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize("Hello World");
    assert_eq!(tokens, vec!["hello", "world"]);
}

#[test]
fn test_tokenizer_stop_words_removed() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize("the quick brown fox");
    assert_eq!(tokens, vec!["quick", "brown", "fox"]);
}

#[test]
fn test_tokenizer_short_tokens_removed() {
    let tokenizer = Tokenizer::new();
    // Tokens shorter than 2 characters are removed.
    // "I" (1 char) -> removed
    // "a" (1 char + stop word) -> removed
    // "am" (2 chars, not a stop word) -> kept
    // "Rust" -> "rust" (kept)
    // "developer" -> kept
    let tokens = tokenizer.tokenize("I am a Rust developer");
    assert_eq!(tokens, vec!["am", "rust", "developer"]);

    // Demonstrate that single-character tokens are always removed
    let tokens2 = tokenizer.tokenize("I x y Rust");
    // "I", "x", "y" are all single chars -> removed
    assert_eq!(tokens2, vec!["rust"]);
}

#[test]
fn test_tokenizer_punctuation_stripped() {
    let tokenizer = Tokenizer::new();
    // Punctuation chars (', !) are not alphanumeric, so they act as split boundaries.
    // "user's" -> split on ' -> ["user", "s"] -> "s" is < 2 chars -> removed
    // "name" -> kept (not stop word, >= 2 chars)
    // "is" -> stop word -> removed
    // "Marcus!" -> split on ! -> ["marcus"] -> kept
    let tokens = tokenizer.tokenize("user's name is Marcus!");
    assert_eq!(tokens, vec!["user", "name", "marcus"]);
}

#[test]
fn test_tokenizer_deterministic() {
    let tokenizer = Tokenizer::new();
    let input = "The Rust programming language is blazingly fast and memory-safe";
    let expected = tokenizer.tokenize(input);
    for _ in 0..100 {
        let result = tokenizer.tokenize(input);
        assert_eq!(result, expected, "Tokenizer output must be deterministic");
    }
}

#[test]
fn test_tokenizer_unicode() {
    let tokenizer = Tokenizer::new();
    // Unicode letters are alphanumeric, so they should be preserved.
    // The accented characters are kept as part of tokens.
    let tokens = tokenizer.tokenize("cafe resume naive");
    // "cafe", "resume", "naive" are all >= 2 chars, not stop words
    assert_eq!(tokens, vec!["cafe", "resume", "naive"]);

    // With actual diacritics: e with accent is still alphanumeric
    let tokens2 = tokenizer.tokenize("\u{00e9}l\u{00e8}ve r\u{00e9}sum\u{00e9} na\u{00ef}ve");
    // These should tokenize to lowercased forms preserving unicode
    assert!(!tokens2.is_empty(), "Unicode tokens should not be empty");
    // Each token should be >= 2 characters
    for tok in &tokens2 {
        assert!(tok.len() >= 2, "Token '{}' should be >= 2 chars", tok);
    }
}

#[test]
fn test_tokenizer_empty_string() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize("");
    assert!(tokens.is_empty());
}

#[test]
fn test_tokenizer_only_stop_words() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize("the is a");
    assert!(tokens.is_empty());
}

// ==================== TermIndex Tests (5) ====================

#[test]
fn test_term_index_build() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    // 5 nodes with known content
    let contents = [
        "Rust programming language",        // tokens: rust, programming, language
        "Python programming language",      // tokens: python, programming, language
        "Rust memory safety guarantees",    // tokens: rust, memory, safety, guarantees
        "JavaScript web development",       // tokens: javascript, web, development
        "Rust and Python interoperability", // tokens: rust, python, interoperability
    ];

    for content in &contents {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id(1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }

    let index = TermIndex::build(&graph, &tokenizer);

    // "rust" appears in nodes 0, 2, 4 -> doc_frequency = 3
    assert_eq!(index.doc_frequency("rust"), 3);

    // "programming" appears in nodes 0, 1 -> doc_frequency = 2
    assert_eq!(index.doc_frequency("programming"), 2);

    // "language" appears in nodes 0, 1 -> doc_frequency = 2
    assert_eq!(index.doc_frequency("language"), 2);

    // "python" appears in nodes 1, 4 -> doc_frequency = 2
    assert_eq!(index.doc_frequency("python"), 2);

    // "javascript" appears only in node 3 -> doc_frequency = 1
    assert_eq!(index.doc_frequency("javascript"), 1);

    // Total documents
    assert_eq!(index.doc_count(), 5);

    // Verify posting lists contain the right node IDs
    let rust_postings = index.get("rust");
    let rust_ids: Vec<u64> = rust_postings.iter().map(|(id, _)| *id).collect();
    assert!(rust_ids.contains(&0));
    assert!(rust_ids.contains(&2));
    assert!(rust_ids.contains(&4));
    assert_eq!(rust_ids.len(), 3);
}

#[test]
fn test_term_index_doc_frequency() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    // "rust" in 3 of 5 nodes
    let contents = [
        "Rust programming",
        "Python scripting",
        "Rust systems design",
        "JavaScript frontend",
        "Rust embedded devices",
    ];

    for content in &contents {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id(1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }

    let index = TermIndex::build(&graph, &tokenizer);
    assert_eq!(index.doc_frequency("rust"), 3);
    assert_eq!(index.doc_frequency("python"), 1);
    assert_eq!(index.doc_frequency("nonexistent"), 0);
}

#[test]
fn test_term_index_add_node_incremental() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    let event0 = CognitiveEventBuilder::new(EventType::Fact, "Rust programming")
        .session_id(1)
        .confidence(0.9)
        .build();
    graph.add_node(event0).unwrap();

    let mut index = TermIndex::build(&graph, &tokenizer);
    assert_eq!(index.doc_frequency("rust"), 1);
    assert_eq!(index.doc_frequency("quantum"), 0);
    assert_eq!(index.doc_count(), 1);

    // Incrementally add a new node
    let mut new_event = CognitiveEventBuilder::new(EventType::Fact, "quantum computing research")
        .session_id(1)
        .confidence(0.9)
        .build();
    new_event.id = 1;
    index.add_node(&new_event);

    // "quantum" should now appear
    assert_eq!(index.doc_frequency("quantum"), 1);
    assert_eq!(index.doc_count(), 2);

    // "rust" unchanged
    assert_eq!(index.doc_frequency("rust"), 1);
}

#[test]
fn test_term_index_remove_node() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    let event0 = CognitiveEventBuilder::new(EventType::Fact, "Rust programming language")
        .session_id(1)
        .confidence(0.9)
        .build();
    let event1 = CognitiveEventBuilder::new(EventType::Fact, "Python scripting language")
        .session_id(1)
        .confidence(0.9)
        .build();
    let id0 = graph.add_node(event0).unwrap();
    graph.add_node(event1).unwrap();

    let mut index = TermIndex::build(&graph, &tokenizer);
    assert_eq!(index.doc_frequency("rust"), 1);
    assert_eq!(index.doc_frequency("language"), 2);
    assert_eq!(index.doc_count(), 2);

    // Remove node 0
    index.remove_node(id0);
    assert_eq!(index.doc_frequency("rust"), 0);
    // "language" still in node 1
    assert_eq!(index.doc_frequency("language"), 1);
    assert_eq!(index.doc_count(), 1);
}

#[test]
fn test_term_index_write_read_roundtrip() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    let contents = [
        "Rust programming language",
        "Python data science",
        "Machine learning algorithms",
    ];
    for content in &contents {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id(1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }

    let original = TermIndex::build(&graph, &tokenizer);
    let bytes = original.to_bytes();
    let restored = TermIndex::from_bytes(&bytes).expect("Should deserialize successfully");

    // Verify doc count and avg doc length match
    assert_eq!(restored.doc_count(), original.doc_count());
    assert!(
        (restored.avg_doc_length() - original.avg_doc_length()).abs() < f32::EPSILON,
        "avg_doc_length mismatch: {} vs {}",
        restored.avg_doc_length(),
        original.avg_doc_length()
    );

    // Verify term count
    assert_eq!(restored.term_count(), original.term_count());

    // Verify specific posting lists
    for term in &[
        "rust",
        "programming",
        "language",
        "python",
        "data",
        "science",
        "machine",
        "learning",
        "algorithms",
    ] {
        let orig_postings = original.get(term);
        let rest_postings = restored.get(term);
        assert_eq!(
            orig_postings.len(),
            rest_postings.len(),
            "Posting list length mismatch for term '{}'",
            term
        );
        for (orig, rest) in orig_postings.iter().zip(rest_postings.iter()) {
            assert_eq!(orig.0, rest.0, "Node ID mismatch for term '{}'", term);
            assert_eq!(orig.1, rest.1, "Term freq mismatch for term '{}'", term);
        }
    }
}

// ==================== DocLengths Tests (2) ====================

#[test]
fn test_doc_lengths_build() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    // Node 0: "Rust programming" -> 2 tokens
    let event0 = CognitiveEventBuilder::new(EventType::Fact, "Rust programming")
        .session_id(1)
        .confidence(0.9)
        .build();
    // Node 1: "the quick brown fox jumps" -> stop words removed: "quick", "brown", "fox", "jumps" -> 4 tokens
    let event1 = CognitiveEventBuilder::new(EventType::Fact, "the quick brown fox jumps")
        .session_id(1)
        .confidence(0.9)
        .build();
    // Node 2: "a" -> all tokens removed (stop word + short) -> 0 tokens
    let event2 = CognitiveEventBuilder::new(EventType::Fact, "a")
        .session_id(1)
        .confidence(0.9)
        .build();

    let id0 = graph.add_node(event0).unwrap();
    let id1 = graph.add_node(event1).unwrap();
    let id2 = graph.add_node(event2).unwrap();

    let doc_lengths = DocLengths::build(&graph, &tokenizer);

    // Verify manually tokenized lengths
    let manual_tokens_0 = tokenizer.tokenize("Rust programming");
    let manual_tokens_1 = tokenizer.tokenize("the quick brown fox jumps");
    let manual_tokens_2 = tokenizer.tokenize("a");

    assert_eq!(
        doc_lengths.get(id0),
        manual_tokens_0.len() as u32,
        "Node 0 length mismatch"
    );
    assert_eq!(
        doc_lengths.get(id1),
        manual_tokens_1.len() as u32,
        "Node 1 length mismatch"
    );
    assert_eq!(
        doc_lengths.get(id2),
        manual_tokens_2.len() as u32,
        "Node 2 length mismatch"
    );
}

#[test]
fn test_doc_lengths_write_read_roundtrip() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let tokenizer = Tokenizer::new();

    let contents = [
        "Rust programming language",
        "Python data science toolkit",
        "Machine learning algorithms research",
    ];
    for content in &contents {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id(1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }

    let original = DocLengths::build(&graph, &tokenizer);
    let bytes = original.to_bytes();
    let restored = DocLengths::from_bytes(&bytes).expect("Should deserialize successfully");

    // Verify each node length matches
    for id in 0..3u64 {
        assert_eq!(
            restored.get(id),
            original.get(id),
            "DocLength mismatch for node {}",
            id
        );
    }

    // Verify aggregate stats
    assert_eq!(restored.len(), original.len());
    assert!(
        (restored.average() - original.average()).abs() < f32::EPSILON,
        "Average doc length mismatch: {} vs {}",
        restored.average(),
        original.average()
    );
}

// ==================== BM25 Search Tests (9) ====================

/// Build a graph with 10 nodes about different topics for BM25 testing.
/// Returns the graph and a vec of (node_id, topic_description).
fn build_topic_graph() -> MemoryGraph {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let contents = [
        "Rust programming language systems development",
        "Python machine learning data science",
        "JavaScript web frontend development",
        "Rust memory safety ownership borrowing",
        "Database SQL query optimization techniques",
        "Cloud computing infrastructure deployment",
        "Rust concurrency async await tokio runtime",
        "Python natural language processing NLP",
        "DevOps continuous integration testing pipeline",
        "Rust WebAssembly WASM browser performance",
    ];

    for (i, content) in contents.iter().enumerate() {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id((i / 3) as u32 + 1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }

    graph
}

#[test]
fn test_bm25_basic_search() {
    let graph = build_topic_graph();
    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "Rust programming".to_string(),
        max_results: 10,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    // Should have results
    assert!(!results.is_empty(), "Should find nodes about Rust");

    // The top result should be one of the Rust-related nodes (IDs 0, 3, 6, 9)
    let rust_node_ids: Vec<u64> = vec![0, 3, 6, 9];
    assert!(
        rust_node_ids.contains(&results[0].node_id),
        "Top result (node {}) should be a Rust-related node",
        results[0].node_id
    );

    // All Rust nodes should appear in results
    let result_ids: Vec<u64> = results.iter().map(|m| m.node_id).collect();
    for &rust_id in &rust_node_ids {
        assert!(
            result_ids.contains(&rust_id),
            "Rust node {} should appear in results",
            rust_id
        );
    }
}

#[test]
fn test_bm25_idf_weighting() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Add "project" (common term) to all 5 nodes
    // Add "quantum" (rare term) to only 1 node
    let contents = [
        "project alpha development team",
        "project beta development team",
        "project gamma development team",
        "project delta development team",
        "quantum project epsilon research", // only node with "quantum"
    ];

    for content in &contents {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id(1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "quantum project".to_string(),
        max_results: 10,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    assert!(!results.is_empty());

    // Node 4 (the one with "quantum") should rank highest because "quantum"
    // has higher IDF weight than "project" (which appears in all nodes).
    assert_eq!(
        results[0].node_id, 4,
        "Node with rare term 'quantum' should rank first, got node {}",
        results[0].node_id
    );
}

#[test]
fn test_bm25_length_normalization() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Short node: "Rust developer" (2 tokens after tokenization)
    let short_event = CognitiveEventBuilder::new(EventType::Fact, "Rust developer")
        .session_id(1)
        .confidence(0.9)
        .build();

    // Long node: "Rust" mentioned once among many tokens
    let long_content = format!(
        "Rust {}",
        (0..99)
            .map(|i| format!("filler{}", i))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let long_event = CognitiveEventBuilder::new(EventType::Fact, &long_content)
        .session_id(1)
        .confidence(0.9)
        .build();

    let short_id = graph.add_node(short_event).unwrap();
    let long_id = graph.add_node(long_event).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "Rust".to_string(),
        max_results: 10,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    assert!(results.len() >= 2, "Should find both nodes");

    // Find scores for short and long nodes
    let short_score = results
        .iter()
        .find(|m| m.node_id == short_id)
        .map(|m| m.score)
        .expect("Short node should be in results");
    let long_score = results
        .iter()
        .find(|m| m.node_id == long_id)
        .map(|m| m.score)
        .expect("Long node should be in results");

    // Short node should rank higher (BM25 length normalization penalizes longer docs)
    assert!(
        short_score > long_score,
        "Short node (score {}) should score higher than long node (score {})",
        short_score,
        long_score
    );
}

#[test]
fn test_bm25_no_match_returns_empty() {
    let graph = build_topic_graph();
    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "xyzzyplugh nonexistent".to_string(),
        max_results: 10,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    assert!(
        results.is_empty(),
        "Search for nonexistent term should return empty"
    );
}

#[test]
fn test_bm25_filter_by_type() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Mix of facts and decisions, all mentioning "Rust"
    let event0 = CognitiveEventBuilder::new(EventType::Fact, "Rust programming language")
        .session_id(1)
        .confidence(0.9)
        .build();
    let event1 = CognitiveEventBuilder::new(EventType::Decision, "Decided to use Rust")
        .session_id(1)
        .confidence(0.9)
        .build();
    let event2 = CognitiveEventBuilder::new(EventType::Fact, "Rust is memory safe")
        .session_id(1)
        .confidence(0.9)
        .build();
    let event3 = CognitiveEventBuilder::new(EventType::Decision, "Rust for backend services")
        .session_id(1)
        .confidence(0.9)
        .build();

    graph.add_node(event0).unwrap();
    let id1 = graph.add_node(event1).unwrap();
    graph.add_node(event2).unwrap();
    let id3 = graph.add_node(event3).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "Rust".to_string(),
        max_results: 10,
        event_types: vec![EventType::Decision],
        session_ids: vec![],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    // Only Decision nodes should appear
    assert!(!results.is_empty());
    let result_ids: Vec<u64> = results.iter().map(|m| m.node_id).collect();
    for &rid in &result_ids {
        let node = graph.get_node(rid).unwrap();
        assert_eq!(
            node.event_type,
            EventType::Decision,
            "Node {} should be a Decision, got {:?}",
            rid,
            node.event_type
        );
    }
    assert!(result_ids.contains(&id1));
    assert!(result_ids.contains(&id3));
}

#[test]
fn test_bm25_filter_by_session() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Nodes in sessions 1, 2, 3 all mentioning "Rust"
    for session in 1..=3u32 {
        let event = CognitiveEventBuilder::new(
            EventType::Fact,
            format!("Rust development session {}", session),
        )
        .session_id(session)
        .confidence(0.9)
        .build();
        graph.add_node(event).unwrap();
    }

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "Rust".to_string(),
        max_results: 10,
        event_types: vec![],
        session_ids: vec![1, 2],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    // Only sessions 1 and 2 should appear
    assert_eq!(results.len(), 2);
    for m in &results {
        let node = graph.get_node(m.node_id).unwrap();
        assert!(
            node.session_id == 1 || node.session_id == 2,
            "Node {} has session {}, expected 1 or 2",
            m.node_id,
            node.session_id
        );
    }
}

#[test]
fn test_bm25_slow_path_matches_fast_path() {
    // CRITICAL: Verify the slow path (no index) and fast path (with index)
    // produce the same results.
    let graph = build_topic_graph();
    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let query = "Rust programming language systems".to_string();

    // Fast path: with indexes
    let fast_params = TextSearchParams {
        query: query.clone(),
        max_results: 100,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };
    let results_fast = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), fast_params)
        .unwrap();

    // Slow path: without indexes (pass None)
    let slow_params = TextSearchParams {
        query: query.clone(),
        max_results: 100,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };
    let results_slow = engine.text_search(&graph, None, None, slow_params).unwrap();

    // Same number of results
    assert_eq!(
        results_fast.len(),
        results_slow.len(),
        "Fast path ({} results) and slow path ({} results) should return same count",
        results_fast.len(),
        results_slow.len()
    );

    // Same set of nodes returned
    let mut fast_ids: Vec<u64> = results_fast.iter().map(|m| m.node_id).collect();
    let mut slow_ids: Vec<u64> = results_slow.iter().map(|m| m.node_id).collect();
    fast_ids.sort();
    slow_ids.sort();
    assert_eq!(
        fast_ids, slow_ids,
        "Fast and slow paths should return the same set of node IDs"
    );

    // For each node, the BM25 score should be very close in both paths
    for fast in &results_fast {
        let slow_match = results_slow
            .iter()
            .find(|s| s.node_id == fast.node_id)
            .expect("Node should appear in both result sets");
        assert!(
            (fast.score - slow_match.score).abs() < 1e-4,
            "Score mismatch for node {}: fast={}, slow={}",
            fast.node_id,
            fast.score,
            slow_match.score
        );
    }

    // Verify ordering is by descending score (both should be sorted)
    for results in [&results_fast, &results_slow] {
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score - 1e-6,
                "Results should be sorted by descending score"
            );
        }
    }
}

#[test]
fn test_bm25_max_results_limit() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Add 100 nodes, all containing "development"
    for i in 0..100 {
        let event = CognitiveEventBuilder::new(
            EventType::Fact,
            format!("development project number {} details", i),
        )
        .session_id(1)
        .confidence(0.9)
        .build();
        graph.add_node(event).unwrap();
    }

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = TextSearchParams {
        query: "development".to_string(),
        max_results: 5,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };

    let results = engine
        .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    assert_eq!(results.len(), 5, "Should return exactly 5 results");
}

#[test]
fn test_bm25_scores_are_positive() {
    let graph = build_topic_graph();
    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    // Try several different queries
    let queries = [
        "Rust",
        "Python machine learning",
        "web development",
        "cloud computing deployment",
    ];

    for query_str in &queries {
        let params = TextSearchParams {
            query: query_str.to_string(),
            max_results: 100,
            event_types: vec![],
            session_ids: vec![],
            min_score: 0.0,
        };

        let results = engine
            .text_search(&graph, Some(&term_index), Some(&doc_lengths), params)
            .unwrap();

        for m in &results {
            assert!(
                m.score >= 0.0,
                "BM25 score for node {} with query '{}' is {}, expected >= 0.0",
                m.node_id,
                query_str,
                m.score
            );
        }
    }
}

// ==================== Hybrid Search Tests (5) ====================

#[test]
fn test_hybrid_basic() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Nodes with both content and non-zero feature vectors
    let event0 = CognitiveEventBuilder::new(EventType::Fact, "Rust programming language")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    let event1 = CognitiveEventBuilder::new(EventType::Fact, "Python data science")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(1, 1.0))
        .build();
    let event2 = CognitiveEventBuilder::new(EventType::Decision, "Use Rust for performance")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 0.8))
        .build();

    graph.add_node(event0).unwrap();
    graph.add_node(event1).unwrap();
    graph.add_node(event2).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = HybridSearchParams {
        query_text: "Rust programming".to_string(),
        query_vec: Some(basis_vec(0, 1.0)),
        max_results: 10,
        event_types: vec![],
        text_weight: 0.5,
        vector_weight: 0.5,
        rrf_k: 60,
    };

    let results = engine
        .hybrid_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    assert!(!results.is_empty(), "Hybrid search should return results");

    // Node 0 should rank well (matches both text and vector)
    let top_ids: Vec<u64> = results.iter().map(|m| m.node_id).collect();
    assert!(
        top_ids.contains(&0),
        "Node 0 should appear in hybrid results"
    );

    // Verify all scores are positive
    for m in &results {
        assert!(
            m.combined_score > 0.0,
            "Combined score for node {} should be positive",
            m.node_id
        );
    }
}

#[test]
fn test_hybrid_rrf_fusion() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // RRF combines ranks from BM25 and vector search.
    // To demonstrate RRF properly, we need enough nodes that rank-spread
    // makes a difference. We create 10 filler nodes plus 3 key nodes.
    //
    // The key insight: a node ranking #1 in BM25 but very low in vector
    // should score lower overall than a node ranking well in BOTH signals.

    // Add 8 filler nodes (moderate BM25, moderate vector) to push rankings apart
    for i in 0..8 {
        let mut fv = vec![0.0f32; DEFAULT_DIMENSION];
        fv[i % DEFAULT_DIMENSION] = 0.2;
        let event = CognitiveEventBuilder::new(
            EventType::Fact,
            format!("general topic filler content number {}", i),
        )
        .session_id(1)
        .confidence(0.9)
        .feature_vec(fv)
        .build();
        graph.add_node(event).unwrap();
    }

    // Node A (id=8): Strong BM25 (#1 for "quantum"), very poor vector match
    let event_a = CognitiveEventBuilder::new(
        EventType::Fact,
        "quantum quantum quantum research breakthrough",
    )
    .session_id(1)
    .confidence(0.9)
    .feature_vec(basis_vec(7, 1.0)) // orthogonal to query vec at dim 0
    .build();
    let id_a = graph.add_node(event_a).unwrap();

    // Node B (id=9): No BM25 match at all, #1 in vector
    let event_b = CognitiveEventBuilder::new(EventType::Fact, "advanced systems engineering")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0)) // perfectly aligned with query
        .build();
    let id_b = graph.add_node(event_b).unwrap();

    // Node C (id=10): #2 in BM25 AND #2 in vector
    let event_c = CognitiveEventBuilder::new(EventType::Fact, "quantum computing overview")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 0.95)) // very close to query vec
        .build();
    let id_c = graph.add_node(event_c).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    let params = HybridSearchParams {
        query_text: "quantum computing".to_string(),
        query_vec: Some(basis_vec(0, 1.0)),
        max_results: 15,
        event_types: vec![],
        text_weight: 0.5,
        vector_weight: 0.5,
        rrf_k: 60,
    };

    let results = engine
        .hybrid_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    assert!(results.len() >= 3, "Should have at least 3 results");

    let result_ids: Vec<u64> = results.iter().map(|m| m.node_id).collect();

    // All three key nodes should appear
    assert!(
        result_ids.contains(&id_a),
        "Node A should appear (BM25 match)"
    );
    assert!(
        result_ids.contains(&id_b),
        "Node B should appear (vector match)"
    );
    assert!(
        result_ids.contains(&id_c),
        "Node C should appear (both signals)"
    );

    // Node C, which ranks well in BOTH BM25 and vector, should rank
    // higher than Node B (vector-only, no BM25) because C gets RRF
    // contributions from both signals.
    let pos_b = result_ids.iter().position(|&id| id == id_b).unwrap();
    let pos_c = result_ids.iter().position(|&id| id == id_c).unwrap();
    assert!(
        pos_c < pos_b,
        "Node C (pos {}) should rank higher than Node B (pos {}), \
         because C scores well in both BM25 and vector",
        pos_c,
        pos_b
    );

    // Verify all combined scores are positive
    for m in &results {
        assert!(m.combined_score > 0.0, "All RRF scores should be positive");
    }
}

#[test]
fn test_hybrid_text_only_fallback() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // All nodes have zero feature vectors
    let event0 = CognitiveEventBuilder::new(EventType::Fact, "Rust programming language")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    let event1 = CognitiveEventBuilder::new(EventType::Fact, "Python scripting language")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();

    graph.add_node(event0).unwrap();
    graph.add_node(event1).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    // Provide a query vector, but all nodes have zero vectors -> should fall back
    let params = HybridSearchParams {
        query_text: "Rust programming".to_string(),
        query_vec: Some(basis_vec(0, 1.0)),
        max_results: 10,
        event_types: vec![],
        text_weight: 0.5,
        vector_weight: 0.5,
        rrf_k: 60,
    };

    let results = engine
        .hybrid_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    // Should still return BM25 results even though vectors are all zeros
    assert!(
        !results.is_empty(),
        "Hybrid search should fall back to BM25 when all vectors are zero"
    );

    // Node 0 ("Rust programming language") should match
    let result_ids: Vec<u64> = results.iter().map(|m| m.node_id).collect();
    assert!(
        result_ids.contains(&0),
        "Should find Rust node via BM25 fallback"
    );
}

#[test]
fn test_hybrid_weights() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Create nodes with distinct text and vector characteristics
    // Node 0: Strong text match for "quantum", poor vector match
    let event0 =
        CognitiveEventBuilder::new(EventType::Fact, "quantum computing research breakthrough")
            .session_id(1)
            .confidence(0.9)
            .feature_vec(basis_vec(5, 1.0))
            .build();

    // Node 1: Poor text match, strong vector match
    let event1 = CognitiveEventBuilder::new(EventType::Fact, "advanced engineering systems")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();

    // Node 2: Moderate in both
    let event2 = CognitiveEventBuilder::new(EventType::Fact, "quantum systems analysis")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 0.5))
        .build();

    graph.add_node(event0).unwrap();
    graph.add_node(event1).unwrap();
    graph.add_node(event2).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    // Test 1: text_weight=1.0, vector_weight=0.0 -> ordering should match text_search
    // for the nodes that appear in BM25 results.
    let text_only_hybrid = HybridSearchParams {
        query_text: "quantum".to_string(),
        query_vec: Some(basis_vec(0, 1.0)),
        max_results: 10,
        event_types: vec![],
        text_weight: 1.0,
        vector_weight: 0.0,
        rrf_k: 60,
    };
    let text_search_params = TextSearchParams {
        query: "quantum".to_string(),
        max_results: 10,
        event_types: vec![],
        session_ids: vec![],
        min_score: 0.0,
    };

    let hybrid_results = engine
        .hybrid_search(
            &graph,
            Some(&term_index),
            Some(&doc_lengths),
            text_only_hybrid,
        )
        .unwrap();
    let text_results = engine
        .text_search(
            &graph,
            Some(&term_index),
            Some(&doc_lengths),
            text_search_params,
        )
        .unwrap();

    // The text-matched nodes should appear in the same relative order in both
    let text_ids: Vec<u64> = text_results.iter().map(|m| m.node_id).collect();
    let hybrid_ids: Vec<u64> = hybrid_results.iter().map(|m| m.node_id).collect();

    // Every text result should appear in hybrid results
    for &tid in &text_ids {
        assert!(
            hybrid_ids.contains(&tid),
            "Text result node {} should appear in hybrid results",
            tid
        );
    }

    // The relative ordering of text-matched nodes should be preserved
    let hybrid_text_order: Vec<u64> = hybrid_ids
        .iter()
        .filter(|id| text_ids.contains(id))
        .copied()
        .collect();
    assert_eq!(
        hybrid_text_order, text_ids,
        "With text_weight=1.0, vector_weight=0.0, the relative ordering of \
         text-matched nodes should match pure text_search"
    );
}

#[test]
fn test_hybrid_empty_query() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let event = CognitiveEventBuilder::new(EventType::Fact, "Rust programming")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    graph.add_node(event).unwrap();

    let tokenizer = Tokenizer::new();
    let term_index = TermIndex::build(&graph, &tokenizer);
    let doc_lengths = DocLengths::build(&graph, &tokenizer);
    let engine = QueryEngine::new();

    // Empty query string
    let params = HybridSearchParams {
        query_text: "".to_string(),
        query_vec: Some(basis_vec(0, 1.0)),
        max_results: 10,
        event_types: vec![],
        text_weight: 0.5,
        vector_weight: 0.5,
        rrf_k: 60,
    };

    // Should not panic
    let results = engine
        .hybrid_search(&graph, Some(&term_index), Some(&doc_lengths), params)
        .unwrap();

    // Empty text query tokenizes to empty -> BM25 returns nothing.
    // Vector component may return results, but either way no panic.
    // The key assertion is that we get here without panicking.
    let _ = results;
}
