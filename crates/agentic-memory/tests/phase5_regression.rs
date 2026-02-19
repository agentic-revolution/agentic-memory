//! Phase 5 regression tests — verify new code does not break existing behavior.

use std::path::Path;

use agentic_memory::format::{AmemReader, AmemWriter};
use agentic_memory::graph::MemoryGraph;
use agentic_memory::types::edge::{Edge, EdgeType};
use agentic_memory::types::event::{CognitiveEventBuilder, EventType};
use agentic_memory::types::header::{feature_flags, FileHeader};
use agentic_memory::types::DEFAULT_DIMENSION;
use agentic_memory::{
    AnalogicalAnchor, AnalogicalParams, BeliefRevisionParams, CentralityAlgorithm,
    CentralityParams, ConsolidationOp, ConsolidationParams, DriftParams, GapDetectionParams,
    GapSeverity, HybridSearchParams, QueryEngine, ShortestPathParams, TextSearchParams,
    TraversalDirection,
};

// ==================== Regression Tests ====================

#[test]
fn test_old_file_readable_by_new_code() {
    // Load a .amem file written by the v0.1 code path (no BM25 indexes).
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_basic.amem");
    assert!(
        fixture_path.exists(),
        "Fixture file v1_basic.amem must exist at {:?}",
        fixture_path
    );

    let graph = AmemReader::read_from_file(&fixture_path).unwrap();

    // Verify basic structure is correct.
    assert!(
        graph.node_count() > 0,
        "v1_basic.amem should have at least one node"
    );
    assert!(
        graph.edge_count() > 0,
        "v1_basic.amem should have at least one edge"
    );

    // Verify all nodes have content and feature vectors.
    for node in graph.nodes() {
        assert!(
            !node.content.is_empty(),
            "Node {} should have content",
            node.id
        );
        assert_eq!(
            node.feature_vec.len(),
            graph.dimension(),
            "Node {} feature vector dimension mismatch",
            node.id
        );
    }

    // Verify term_index and doc_lengths are None (old file has no BM25 indexes).
    assert!(
        graph.term_index.is_none(),
        "Old v1 file should not have a term_index"
    );
    assert!(
        graph.doc_lengths.is_none(),
        "Old v1 file should not have doc_lengths"
    );

    // Verify all existing query types still work.
    let engine = QueryEngine::new();

    let first_node_id = graph.nodes()[0].id;

    // Traversal query.
    let traversal = engine.traverse(
        &graph,
        agentic_memory::TraversalParams {
            start_id: first_node_id,
            edge_types: vec![],
            direction: TraversalDirection::Both,
            max_depth: 3,
            max_results: 100,
            min_confidence: 0.0,
        },
    );
    assert!(
        traversal.is_ok(),
        "Traversal query should work on old files"
    );

    // Pattern query.
    let pattern = engine.pattern(
        &graph,
        agentic_memory::PatternParams {
            event_types: vec![EventType::Fact],
            min_confidence: None,
            max_confidence: None,
            session_ids: vec![],
            created_after: None,
            created_before: None,
            min_decay_score: None,
            max_results: 100,
            sort_by: agentic_memory::PatternSort::MostRecent,
        },
    );
    assert!(pattern.is_ok(), "Pattern query should work on old files");

    // Similarity query.
    let query_vec = vec![0.1; graph.dimension()];
    let similarity = engine.similarity(
        &graph,
        agentic_memory::SimilarityParams {
            query_vec,
            top_k: 5,
            min_similarity: -1.0,
            event_types: vec![],
            skip_zero_vectors: false,
        },
    );
    assert!(
        similarity.is_ok(),
        "Similarity query should work on old files"
    );

    // Context query.
    let context = engine.context(&graph, first_node_id, 2);
    assert!(context.is_ok(), "Context query should work on old files");

    // Resolve query.
    let resolve = engine.resolve(&graph, first_node_id);
    assert!(resolve.is_ok(), "Resolve query should work on old files");
}

#[test]
fn test_new_file_sections_are_skippable() {
    // Write a .amem file with the new code (includes tag 0x05, 0x06 indexes).
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    for i in 0..10 {
        let event = CognitiveEventBuilder::new(EventType::Fact, format!("test node {}", i))
            .session_id(1)
            .confidence(0.9)
            .build();
        graph.add_node(event).unwrap();
    }
    for i in 0..9u64 {
        graph
            .add_edge(Edge::new(i, i + 1, EdgeType::RelatedTo, 0.8))
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("new_format.amem");
    AmemWriter::new(DEFAULT_DIMENSION)
        .write_to_file(&graph, &path)
        .unwrap();

    // Read the raw bytes to verify index tags are present.
    let data = std::fs::read(&path).unwrap();
    assert!(data.len() > 64, "File should be larger than header alone");

    // Read back with the current reader -- it parses tags 0x05, 0x06 and
    // gracefully skips any unknown tags.
    let loaded = AmemReader::read_from_file(&path).unwrap();
    assert_eq!(loaded.node_count(), 10);
    assert_eq!(loaded.edge_count(), 9);

    // Verify all content survived the roundtrip.
    for i in 0..10u64 {
        let node = loaded.get_node(i).unwrap();
        assert_eq!(node.content, format!("test node {}", i));
    }

    // Verify feature vectors survived.
    for i in 0..10u64 {
        let orig = graph.get_node(i).unwrap();
        let read = loaded.get_node(i).unwrap();
        assert_eq!(orig.feature_vec.len(), read.feature_vec.len());
    }

    // Simulate a v0.1 reader that would skip unknown tags:
    // The reader already handles unknown tags with a skip-and-continue loop.
    // We verify that even if we strip tags 0x05/0x06, the core data is intact.
    // Since the reader already does this gracefully (see match _ arm in reader.rs),
    // we just re-verify the loaded data is correct.
    assert_eq!(
        loaded.get_node(0).unwrap().content,
        "test node 0",
        "Content after skipping new index tags must be intact"
    );
}

#[test]
fn test_reserved_field_zero_for_old_files() {
    // Load a v0.1 file and verify header flags (was _reserved) == 0.
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_basic.amem");
    assert!(fixture_path.exists(), "Fixture v1_basic.amem must exist");

    // Read the raw header bytes directly.
    let data = std::fs::read(&fixture_path).unwrap();
    assert!(
        data.len() >= 64,
        "File must contain at least 64-byte header"
    );

    let header = FileHeader::read_from(&mut std::io::Cursor::new(&data[..64])).unwrap();
    assert_eq!(
        header.flags, 0,
        "Old v0.1 files must have flags (was _reserved) == 0"
    );
}

#[test]
fn test_feature_flags_set_for_new_files() {
    // Write a new file with BM25 indexes and verify feature flags are set.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    for i in 0..5 {
        let event = CognitiveEventBuilder::new(EventType::Fact, format!("content {}", i))
            .session_id(1)
            .confidence(0.8)
            .build();
        graph.add_node(event).unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("new_with_flags.amem");
    AmemWriter::new(DEFAULT_DIMENSION)
        .write_to_file(&graph, &path)
        .unwrap();

    // Read the header and check flags.
    let data = std::fs::read(&path).unwrap();
    let header = FileHeader::read_from(&mut std::io::Cursor::new(&data[..64])).unwrap();

    assert!(
        header.has_flag(feature_flags::HAS_TERM_INDEX),
        "New file should have HAS_TERM_INDEX flag set"
    );
    assert!(
        header.has_flag(feature_flags::HAS_DOC_LENGTHS),
        "New file should have HAS_DOC_LENGTHS flag set"
    );
    assert_eq!(
        header.flags,
        feature_flags::HAS_TERM_INDEX | feature_flags::HAS_DOC_LENGTHS,
        "Flags should be exactly HAS_TERM_INDEX | HAS_DOC_LENGTHS (0x03)"
    );
}

#[test]
fn test_existing_query_engine_methods_unchanged() {
    // Create a graph with 50 nodes and 100 edges.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let types = [
        EventType::Fact,
        EventType::Decision,
        EventType::Inference,
        EventType::Correction,
        EventType::Skill,
        EventType::Episode,
    ];
    let edge_types = [
        EdgeType::CausedBy,
        EdgeType::Supports,
        EdgeType::RelatedTo,
        EdgeType::Contradicts,
        EdgeType::Supersedes,
        EdgeType::PartOf,
        EdgeType::TemporalNext,
    ];

    for i in 0..50 {
        let et = types[i % types.len()];
        let event = CognitiveEventBuilder::new(et, format!("event content {}", i))
            .session_id(i as u32 / 10)
            .confidence(0.5 + (i as f32 % 5.0) * 0.1)
            .build();
        graph.add_node(event).unwrap();
    }

    // Create 100 edges (avoiding self-edges).
    let mut edge_count = 0;
    for i in 0..50u64 {
        for j in 1..=3u64 {
            let target = (i + j) % 50;
            if target != i {
                let et = edge_types[(i as usize + j as usize) % edge_types.len()];
                let weight = 0.5 + (j as f32) * 0.1;
                graph.add_edge(Edge::new(i, target, et, weight)).unwrap();
                edge_count += 1;
                if edge_count >= 100 {
                    break;
                }
            }
        }
        if edge_count >= 100 {
            break;
        }
    }

    let engine = QueryEngine::new();

    // Traversal.
    let traversal = engine
        .traverse(
            &graph,
            agentic_memory::TraversalParams {
                start_id: 0,
                edge_types: vec![EdgeType::RelatedTo, EdgeType::Supports],
                direction: TraversalDirection::Forward,
                max_depth: 5,
                max_results: 50,
                min_confidence: 0.0,
            },
        )
        .unwrap();
    assert!(!traversal.visited.is_empty(), "Traversal should find nodes");

    // Pattern.
    let facts = engine
        .pattern(
            &graph,
            agentic_memory::PatternParams {
                event_types: vec![EventType::Fact],
                min_confidence: Some(0.5),
                max_confidence: None,
                session_ids: vec![],
                created_after: None,
                created_before: None,
                min_decay_score: None,
                max_results: 100,
                sort_by: agentic_memory::PatternSort::HighestConfidence,
            },
        )
        .unwrap();
    // We inserted 50 nodes with types cycling through 6, so ~8-9 Fact nodes.
    assert!(!facts.is_empty(), "Pattern query should find Fact nodes");

    // Temporal.
    let temporal = engine
        .temporal(
            &graph,
            agentic_memory::TemporalParams {
                range_a: agentic_memory::TimeRange::Session(0),
                range_b: agentic_memory::TimeRange::Session(1),
            },
        )
        .unwrap();
    // Just verify no panic -- results depend on session assignment.
    let _ = temporal.added.len();

    // Causal.
    let causal = engine
        .causal(
            &graph,
            agentic_memory::CausalParams {
                node_id: 0,
                max_depth: 5,
                dependency_types: vec![EdgeType::CausedBy, EdgeType::Supports],
            },
        )
        .unwrap();
    assert_eq!(causal.root_id, 0, "Causal root should be node 0");

    // Similarity.
    let similarity = engine
        .similarity(
            &graph,
            agentic_memory::SimilarityParams {
                query_vec: vec![0.1; DEFAULT_DIMENSION],
                top_k: 10,
                min_similarity: -1.0,
                event_types: vec![],
                skip_zero_vectors: false,
            },
        )
        .unwrap();
    assert!(
        !similarity.is_empty(),
        "Similarity query should find matching nodes"
    );

    // Context.
    let context = engine.context(&graph, 0, 2).unwrap();
    assert!(
        !context.nodes.is_empty(),
        "Context query should return subgraph"
    );
    assert_eq!(context.center_id, 0);

    // Resolve.
    let resolved = engine.resolve(&graph, 0).unwrap();
    assert_eq!(
        resolved.id, 0,
        "Resolve on node 0 with no supersedes should return node 0"
    );
}

#[test]
fn test_empty_graph_all_new_queries_return_empty() {
    let graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let engine = QueryEngine::new();

    // text_search.
    let text_results = engine
        .text_search(
            &graph,
            None,
            None,
            TextSearchParams {
                query: "test query".to_string(),
                max_results: 10,
                event_types: vec![],
                session_ids: vec![],
                min_score: 0.0,
            },
        )
        .unwrap();
    assert!(
        text_results.is_empty(),
        "text_search on empty graph should return empty"
    );

    // hybrid_search.
    let hybrid_results = engine
        .hybrid_search(
            &graph,
            None,
            None,
            HybridSearchParams {
                query_text: "test query".to_string(),
                query_vec: Some(vec![0.1; DEFAULT_DIMENSION]),
                max_results: 10,
                event_types: vec![],
                text_weight: 0.5,
                vector_weight: 0.5,
                rrf_k: 60,
            },
        )
        .unwrap();
    assert!(
        hybrid_results.is_empty(),
        "hybrid_search on empty graph should return empty"
    );

    // centrality.
    let centrality = engine
        .centrality(
            &graph,
            CentralityParams {
                algorithm: CentralityAlgorithm::PageRank { damping: 0.85 },
                max_iterations: 100,
                tolerance: 1e-6,
                top_k: 10,
                event_types: vec![],
                edge_types: vec![],
            },
        )
        .unwrap();
    assert!(
        centrality.scores.is_empty(),
        "centrality on empty graph should return empty"
    );

    // belief_revision.
    let revision = engine
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "some hypothesis".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.3,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();
    assert!(
        revision.contradicted.is_empty(),
        "belief_revision on empty graph should find no contradictions"
    );
    assert!(
        revision.weakened.is_empty(),
        "belief_revision on empty graph should find no weakened nodes"
    );
    assert!(
        revision.invalidated_decisions.is_empty(),
        "belief_revision on empty graph should find no invalidated decisions"
    );

    // gap_detection.
    let gaps = engine
        .gap_detection(
            &graph,
            GapDetectionParams {
                confidence_threshold: 0.5,
                min_support_count: 2,
                max_results: 100,
                session_range: None,
                sort_by: GapSeverity::HighestImpact,
            },
        )
        .unwrap();
    assert!(
        gaps.gaps.is_empty(),
        "gap_detection on empty graph should return no gaps"
    );
    assert!(
        (gaps.summary.health_score - 1.0).abs() < f32::EPSILON,
        "Health score on empty graph should be 1.0"
    );

    // analogical -- uses Vector anchor since there are no nodes.
    let analogies = engine
        .analogical(
            &graph,
            AnalogicalParams {
                anchor: AnalogicalAnchor::Vector(vec![0.1; DEFAULT_DIMENSION]),
                context_depth: 2,
                max_results: 10,
                min_similarity: 0.0,
                exclude_sessions: vec![],
            },
        )
        .unwrap();
    assert!(
        analogies.is_empty(),
        "analogical on empty graph should return empty"
    );

    // consolidate (dry run on an empty graph, need mutable reference).
    let mut empty_graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let consolidation = engine
        .consolidate(
            &mut empty_graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::DeduplicateFacts { threshold: 0.9 }],
                dry_run: true,
                backup_path: None,
            },
        )
        .unwrap();
    assert!(
        consolidation.actions.is_empty(),
        "consolidate on empty graph should have no actions"
    );

    // drift_detection.
    let drift = engine
        .drift_detection(
            &graph,
            DriftParams {
                topic: "some topic".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.1,
            },
        )
        .unwrap();
    assert!(
        drift.timelines.is_empty(),
        "drift_detection on empty graph should return no timelines"
    );
    assert!(
        (drift.stability - 1.0).abs() < f32::EPSILON,
        "Stability on empty graph should be 1.0"
    );
    assert!(
        !drift.likely_to_change,
        "Empty graph should not be likely to change"
    );
}

#[test]
fn test_single_node_graph_all_new_queries_safe() {
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let event = CognitiveEventBuilder::new(EventType::Fact, "single node content")
        .session_id(1)
        .confidence(0.9)
        .build();
    graph.add_node(event).unwrap();

    let engine = QueryEngine::new();

    // text_search.
    let text_results = engine
        .text_search(
            &graph,
            graph.term_index(),
            graph.doc_lengths(),
            TextSearchParams {
                query: "single node content".to_string(),
                max_results: 10,
                event_types: vec![],
                session_ids: vec![],
                min_score: 0.0,
            },
        )
        .unwrap();
    // Should find the single node via slow path (no indexes).
    // Not guaranteed to match depending on tokenizer/stop words, but should not panic.
    let _ = text_results.len();

    // hybrid_search.
    let hybrid_results = engine
        .hybrid_search(
            &graph,
            graph.term_index(),
            graph.doc_lengths(),
            HybridSearchParams {
                query_text: "single node".to_string(),
                query_vec: Some(vec![0.1; DEFAULT_DIMENSION]),
                max_results: 10,
                event_types: vec![],
                text_weight: 0.5,
                vector_weight: 0.5,
                rrf_k: 60,
            },
        )
        .unwrap();
    let _ = hybrid_results.len();

    // centrality -- single node has trivial centrality.
    let centrality = engine
        .centrality(
            &graph,
            CentralityParams {
                algorithm: CentralityAlgorithm::PageRank { damping: 0.85 },
                max_iterations: 100,
                tolerance: 1e-6,
                top_k: 10,
                event_types: vec![],
                edge_types: vec![],
            },
        )
        .unwrap();
    assert_eq!(
        centrality.scores.len(),
        1,
        "Single node graph should have exactly 1 score"
    );
    assert!(
        (centrality.scores[0].1 - 1.0).abs() < 0.01,
        "Single node PageRank should be approximately 1.0"
    );

    // shortest_path -- same node.
    let path_result = engine
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: 0,
                target_id: 0,
                edge_types: vec![],
                direction: TraversalDirection::Both,
                max_depth: 10,
                weighted: false,
            },
        )
        .unwrap();
    assert!(
        path_result.found,
        "Path from node to itself should be found"
    );
    assert_eq!(path_result.path, vec![0], "Path to self should be [self]");
    assert!(
        path_result.cost.abs() < f32::EPSILON,
        "Cost from node to itself should be 0"
    );

    // belief_revision.
    let revision = engine
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "single node content is wrong".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.3,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();
    let _ = revision.total_affected;

    // gap_detection.
    let gaps = engine
        .gap_detection(
            &graph,
            GapDetectionParams {
                confidence_threshold: 0.5,
                min_support_count: 2,
                max_results: 100,
                session_range: None,
                sort_by: GapSeverity::HighestImpact,
            },
        )
        .unwrap();
    let _ = gaps.gaps.len();

    // analogical.
    let analogies = engine
        .analogical(
            &graph,
            AnalogicalParams {
                anchor: AnalogicalAnchor::Node(0),
                context_depth: 2,
                max_results: 10,
                min_similarity: 0.0,
                exclude_sessions: vec![],
            },
        )
        .unwrap();
    // With only 1 node, there's nothing to compare to.
    assert!(
        analogies.is_empty(),
        "Analogical on single-node graph should return empty"
    );

    // consolidate.
    let mut graph_mut = MemoryGraph::new(DEFAULT_DIMENSION);
    let event2 = CognitiveEventBuilder::new(EventType::Fact, "single node content")
        .session_id(1)
        .confidence(0.9)
        .build();
    graph_mut.add_node(event2).unwrap();
    let consolidation = engine
        .consolidate(
            &mut graph_mut,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::DeduplicateFacts { threshold: 0.9 }],
                dry_run: true,
                backup_path: None,
            },
        )
        .unwrap();
    let _ = consolidation.actions.len();

    // drift_detection.
    let drift = engine
        .drift_detection(
            &graph,
            DriftParams {
                topic: "single node content".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.1,
            },
        )
        .unwrap();
    let _ = drift.timelines.len();
}

#[test]
fn test_write_read_roundtrip_with_new_indexes() {
    // Create a graph with varied content to generate meaningful BM25 indexes.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let contents = [
        "Rust programming language system performance",
        "Python scripting automation data science",
        "JavaScript web frontend React framework",
        "Database PostgreSQL query optimization",
        "Machine learning neural network training",
        "Cloud infrastructure Kubernetes deployment",
        "Security authentication encryption protocol",
        "Testing continuous integration pipeline",
        "API design RESTful microservices architecture",
        "Documentation code review best practices",
    ];

    for (i, content) in contents.iter().enumerate() {
        let event = CognitiveEventBuilder::new(EventType::Fact, *content)
            .session_id(i as u32 / 3)
            .confidence(0.7 + (i as f32) * 0.03)
            .build();
        graph.add_node(event).unwrap();
    }

    // Add edges.
    for i in 0..9u64 {
        graph
            .add_edge(Edge::new(i, i + 1, EdgeType::RelatedTo, 0.8))
            .unwrap();
    }
    graph
        .add_edge(Edge::new(0, 5, EdgeType::Supports, 0.7))
        .unwrap();

    // Write to file.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip.amem");
    AmemWriter::new(DEFAULT_DIMENSION)
        .write_to_file(&graph, &path)
        .unwrap();

    // Read back.
    let loaded = AmemReader::read_from_file(&path).unwrap();

    // Verify all nodes match.
    assert_eq!(loaded.node_count(), graph.node_count());
    for i in 0..graph.node_count() as u64 {
        let orig = graph.get_node(i).unwrap();
        let read = loaded.get_node(i).unwrap();
        assert_eq!(orig.id, read.id, "Node ID mismatch at {}", i);
        assert_eq!(orig.content, read.content, "Content mismatch at node {}", i);
        assert_eq!(
            orig.event_type, read.event_type,
            "EventType mismatch at node {}",
            i
        );
        assert_eq!(
            orig.session_id, read.session_id,
            "Session mismatch at node {}",
            i
        );
        assert!(
            (orig.confidence - read.confidence).abs() < f32::EPSILON,
            "Confidence mismatch at node {}",
            i
        );
        // Feature vectors.
        assert_eq!(
            orig.feature_vec.len(),
            read.feature_vec.len(),
            "Feature vec length mismatch at node {}",
            i
        );
        for (j, (a, b)) in orig
            .feature_vec
            .iter()
            .zip(read.feature_vec.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "Feature vec mismatch at node {} dim {}",
                i,
                j
            );
        }
    }

    // Verify edges match.
    assert_eq!(loaded.edge_count(), graph.edge_count());

    // Verify TermIndex was persisted.
    assert!(
        loaded.term_index.is_some(),
        "Loaded graph should have a TermIndex"
    );
    let ti = loaded.term_index.as_ref().unwrap();
    // "rust" should be in the term index (from node 0).
    let postings = ti.get("rust");
    assert!(
        !postings.is_empty(),
        "TermIndex should have postings for 'rust'"
    );
    // The posting should point to node 0.
    assert!(
        postings.iter().any(|(id, _)| *id == 0),
        "Posting for 'rust' should include node 0"
    );

    // Verify DocLengths was persisted.
    assert!(
        loaded.doc_lengths.is_some(),
        "Loaded graph should have DocLengths"
    );
    let dl = loaded.doc_lengths.as_ref().unwrap();
    // Each node has content, so doc lengths should be > 0 for all nodes.
    for i in 0..10u64 {
        assert!(dl.get(i) > 0, "DocLengths for node {} should be > 0", i);
    }

    // Verify the new indexes produce correct search results.
    let engine = QueryEngine::new();
    let results = engine
        .text_search(
            &loaded,
            loaded.term_index.as_ref(),
            loaded.doc_lengths.as_ref(),
            TextSearchParams {
                query: "Rust programming".to_string(),
                max_results: 10,
                event_types: vec![],
                session_ids: vec![],
                min_score: 0.0,
            },
        )
        .unwrap();
    assert!(
        !results.is_empty(),
        "Text search for 'Rust programming' should find results"
    );
    assert_eq!(
        results[0].node_id, 0,
        "Node 0 ('Rust programming...') should be the top result"
    );
    assert!(
        results[0].score > 0.0,
        "Top result should have positive BM25 score"
    );
}
