//! Phase 5 tests: Centrality (PageRank, Degree, Betweenness) + Shortest Path.

use agentic_memory::graph::MemoryGraph;
use agentic_memory::types::edge::{Edge, EdgeType};
use agentic_memory::types::event::{CognitiveEventBuilder, EventType};
use agentic_memory::types::DEFAULT_DIMENSION;

use agentic_memory::engine::graph_algo::{
    CentralityAlgorithm, CentralityParams, ShortestPathParams,
};
use agentic_memory::engine::query::QueryEngine;
use agentic_memory::graph::traversal::TraversalDirection;

// ==================== Helpers ====================

/// Create a zero feature vector of graph dimension.
fn zero_vec() -> Vec<f32> {
    vec![0.0; DEFAULT_DIMENSION]
}

/// Shorthand: add a Fact node with the given content and return its ID.
fn add_fact(graph: &mut MemoryGraph, content: &str) -> u64 {
    let event = CognitiveEventBuilder::new(EventType::Fact, content)
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(event).unwrap()
}

// ==================== PageRank Tests ====================

#[test]
fn test_pagerank_simple_chain() {
    // A -> B -> C  (3 nodes, 2 edges)
    // C should have the highest rank (most pointed to indirectly).
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
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

    // The first entry (highest score) should be node C.
    assert!(!result.scores.is_empty(), "Scores should not be empty");
    assert_eq!(
        result.scores[0].0, c,
        "C should have the highest PageRank in A->B->C chain"
    );
}

#[test]
fn test_pagerank_star_topology() {
    // A -> C, B -> C, D -> C  (3 nodes pointing to 1)
    // C should have the highest rank.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");
    let d = add_fact(&mut graph, "D");

    graph
        .add_edge(Edge::new(a, c, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(d, c, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
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

    assert!(!result.scores.is_empty());
    assert_eq!(
        result.scores[0].0, c,
        "C should have the highest PageRank in star topology"
    );

    // C's score should be significantly higher than any other node.
    let c_score = result.scores[0].1;
    for &(id, score) in &result.scores[1..] {
        assert!(
            c_score > score,
            "C score ({}) should be > node {} score ({})",
            c_score,
            id,
            score
        );
    }
}

#[test]
fn test_pagerank_convergence() {
    // 100-node random graph — verify convergence in < 100 iterations.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let mut ids = Vec::new();
    for i in 0..100 {
        ids.push(add_fact(&mut graph, &format!("node_{}", i)));
    }

    // Create a pseudo-random set of edges (deterministic pattern).
    for i in 0..100u64 {
        let target1 = (i * 7 + 13) % 100;
        let target2 = (i * 11 + 37) % 100;
        if i != target1 {
            graph
                .add_edge(Edge::new(
                    ids[i as usize],
                    ids[target1 as usize],
                    EdgeType::RelatedTo,
                    0.8,
                ))
                .unwrap();
        }
        if i != target2 && target1 != target2 {
            graph
                .add_edge(Edge::new(
                    ids[i as usize],
                    ids[target2 as usize],
                    EdgeType::Supports,
                    0.6,
                ))
                .unwrap();
        }
    }

    let qe = QueryEngine::new();
    let result = qe
        .centrality(
            &graph,
            CentralityParams {
                algorithm: CentralityAlgorithm::PageRank { damping: 0.85 },
                max_iterations: 100,
                tolerance: 1e-6,
                top_k: 100,
                event_types: vec![],
                edge_types: vec![],
            },
        )
        .unwrap();

    assert!(
        result.converged,
        "PageRank should converge within 100 iterations"
    );
    assert!(
        result.iterations < 100,
        "PageRank converged in {} iterations, expected < 100",
        result.iterations
    );
}

#[test]
fn test_pagerank_scores_sum_to_one() {
    // Sum of all PageRank scores should be approximately 1.0.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");
    let d = add_fact(&mut graph, "D");
    let e = add_fact(&mut graph, "E");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(c, d, EdgeType::Supports, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(d, e, EdgeType::RelatedTo, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(e, a, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .centrality(
            &graph,
            CentralityParams {
                algorithm: CentralityAlgorithm::PageRank { damping: 0.85 },
                max_iterations: 200,
                tolerance: 1e-8,
                top_k: 100,
                event_types: vec![],
                edge_types: vec![],
            },
        )
        .unwrap();

    let sum: f32 = result.scores.iter().map(|(_, s)| s).sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "PageRank scores should sum to ~1.0, got {}",
        sum
    );
}

#[test]
fn test_pagerank_isolated_node() {
    // Node with no edges gets the base score (1-d)/N.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C-isolated");

    // Only A -> B; C has no edges.
    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
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

    // Find C's score.
    let c_score = result
        .scores
        .iter()
        .find(|(id, _)| *id == c)
        .map(|(_, s)| *s)
        .expect("C should appear in scores");

    // C is isolated but still receives the base teleportation score plus dangling
    // contributions.  It should be positive.
    assert!(
        c_score > 0.0,
        "Isolated node should still have a positive PageRank, got {}",
        c_score
    );
}

// ==================== Degree Centrality Test ====================

#[test]
fn test_degree_centrality() {
    // Known graph:
    //   A -- B -- C   (A has 1 connection, B has 2, C has 1)
    //        |
    //        D         (B has 3 total, D has 1)
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");
    let d = add_fact(&mut graph, "D");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, d, EdgeType::Supports, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .centrality(
            &graph,
            CentralityParams {
                algorithm: CentralityAlgorithm::Degree,
                max_iterations: 0,
                tolerance: 0.0,
                top_k: 10,
                event_types: vec![],
                edge_types: vec![],
            },
        )
        .unwrap();

    // B participates in all 3 edges (source or target), giving it the highest degree.
    assert!(!result.scores.is_empty());
    assert_eq!(
        result.scores[0].0, b,
        "B should have the highest degree centrality"
    );

    // Verify B has a higher score than A, C, and D.
    let b_score = result.scores[0].1;
    for &(id, score) in &result.scores[1..] {
        assert!(
            b_score >= score,
            "B score ({}) should be >= node {} score ({})",
            b_score,
            id,
            score
        );
    }
}

// ==================== Betweenness Centrality Test ====================

#[test]
fn test_betweenness_simple() {
    // A -- B -- C  (B is the only bridge)
    // B should have the highest betweenness.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .centrality(
            &graph,
            CentralityParams {
                algorithm: CentralityAlgorithm::Betweenness,
                max_iterations: 0,
                tolerance: 0.0,
                top_k: 10,
                event_types: vec![],
                edge_types: vec![],
            },
        )
        .unwrap();

    assert!(!result.scores.is_empty());
    assert_eq!(
        result.scores[0].0, b,
        "B should have the highest betweenness centrality as the only bridge"
    );

    // A and C are endpoints; their betweenness should be 0 (they never sit between others).
    let a_score = result
        .scores
        .iter()
        .find(|(id, _)| *id == a)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let c_score = result
        .scores
        .iter()
        .find(|(id, _)| *id == c)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    assert!(
        a_score < f32::EPSILON,
        "A should have betweenness ~0, got {}",
        a_score
    );
    assert!(
        c_score < f32::EPSILON,
        "C should have betweenness ~0, got {}",
        c_score
    );
}

// ==================== Shortest Path Tests ====================

#[test]
fn test_shortest_path_direct_edge() {
    // A -> B with a direct edge. Path = [A, B], cost = 1.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: b,
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 10,
                weighted: false,
            },
        )
        .unwrap();

    assert!(result.found, "Path should be found");
    assert_eq!(result.path, vec![a, b]);
    assert!(
        (result.cost - 1.0).abs() < f32::EPSILON,
        "Cost should be 1.0, got {}",
        result.cost
    );
}

#[test]
fn test_shortest_path_two_hops() {
    // A -> B -> C, no direct A->C edge. Path = [A, B, C], cost = 2.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: c,
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 10,
                weighted: false,
            },
        )
        .unwrap();

    assert!(result.found, "Path should be found");
    assert_eq!(result.path, vec![a, b, c]);
    assert!(
        (result.cost - 2.0).abs() < f32::EPSILON,
        "Cost should be 2.0, got {}",
        result.cost
    );
}

#[test]
fn test_shortest_path_no_path() {
    // A and B in disconnected components. found = false.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    // No edges at all.

    let qe = QueryEngine::new();
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: b,
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 100,
                weighted: false,
            },
        )
        .unwrap();

    assert!(
        !result.found,
        "No path should exist between disconnected nodes"
    );
    assert!(result.path.is_empty(), "Path should be empty");
}

#[test]
fn test_shortest_path_bidirectional_faster() {
    // Long chain: A -> B -> C -> D -> E -> F
    // Bidirectional BFS should find the path.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");
    let d = add_fact(&mut graph, "D");
    let e = add_fact(&mut graph, "E");
    let f = add_fact(&mut graph, "F");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(c, d, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(d, e, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(e, f, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: f,
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 20,
                weighted: false,
            },
        )
        .unwrap();

    assert!(result.found, "Path should be found in long chain");
    // Path should contain all nodes in order.
    assert_eq!(result.path.len(), 6, "Path length should be 6");
    assert_eq!(result.path[0], a);
    assert_eq!(*result.path.last().unwrap(), f);
    assert!(
        (result.cost - 5.0).abs() < f32::EPSILON,
        "Cost should be 5.0, got {}",
        result.cost
    );
}

#[test]
fn test_shortest_path_respects_edge_types() {
    // A -[CausedBy]-> B -[Supports]-> C -[CausedBy]-> D
    // Search with edge_types=[CausedBy] from A to D.
    // Should NOT reach D because the Supports edge between B and C blocks the path.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");
    let d = add_fact(&mut graph, "D");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(b, c, EdgeType::Supports, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(c, d, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();

    // Only follow CausedBy edges: A -> B is reachable, but B -> C is Supports so path stops.
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: d,
                edge_types: vec![EdgeType::CausedBy],
                direction: TraversalDirection::Forward,
                max_depth: 10,
                weighted: false,
            },
        )
        .unwrap();

    assert!(
        !result.found,
        "Path should not be found when Supports edge blocks the CausedBy-only search"
    );

    // But A -> B should be findable with CausedBy filter.
    let result_ab = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: b,
                edge_types: vec![EdgeType::CausedBy],
                direction: TraversalDirection::Forward,
                max_depth: 10,
                weighted: false,
            },
        )
        .unwrap();

    assert!(
        result_ab.found,
        "A -> B should be found with CausedBy filter"
    );
    assert_eq!(result_ab.path, vec![a, b]);
}

#[test]
fn test_shortest_path_weighted() {
    // A -> B  (weight 0.9, cost = 1.0 - 0.9 = 0.1)
    // A -> C  (weight 0.5, cost = 0.5)
    // C -> B  (weight 0.5, cost = 0.5)
    // Direct A->B cost = 0.1.  A->C->B cost = 0.5 + 0.5 = 1.0.
    // Weighted shortest path should prefer the direct A->B route.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");
    let b = add_fact(&mut graph, "B");
    let c = add_fact(&mut graph, "C");

    graph
        .add_edge(Edge::new(a, b, EdgeType::CausedBy, 0.9))
        .unwrap();
    graph
        .add_edge(Edge::new(a, c, EdgeType::CausedBy, 0.5))
        .unwrap();
    graph
        .add_edge(Edge::new(c, b, EdgeType::CausedBy, 0.5))
        .unwrap();

    let qe = QueryEngine::new();
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: b,
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 10,
                weighted: true,
            },
        )
        .unwrap();

    assert!(result.found, "Weighted path should be found");
    // Direct path A->B should be preferred (cost 0.1 vs 1.0).
    assert_eq!(
        result.path,
        vec![a, b],
        "Weighted path should prefer direct A->B (cost 0.1) over A->C->B (cost 1.0)"
    );
    assert!(
        (result.cost - 0.1).abs() < 0.01,
        "Cost should be ~0.1 (1.0 - 0.9), got {}",
        result.cost
    );
}

#[test]
fn test_shortest_path_max_depth() {
    // Build a chain of 11 nodes (depth 10 from first to last).
    // max_depth=5 should NOT find the full path.
    // max_depth=15 should find it.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let mut ids = Vec::new();
    for i in 0..11 {
        ids.push(add_fact(&mut graph, &format!("node_{}", i)));
    }
    for i in 0..10 {
        graph
            .add_edge(Edge::new(ids[i], ids[i + 1], EdgeType::CausedBy, 1.0))
            .unwrap();
    }

    let qe = QueryEngine::new();

    // max_depth=5: path exists at depth 10, should not be found.
    let result_shallow = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: ids[0],
                target_id: ids[10],
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 5,
                weighted: false,
            },
        )
        .unwrap();

    assert!(
        !result_shallow.found,
        "Path at depth 10 should NOT be found with max_depth=5"
    );

    // max_depth=15: should find it.
    let result_deep = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: ids[0],
                target_id: ids[10],
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 15,
                weighted: false,
            },
        )
        .unwrap();

    assert!(
        result_deep.found,
        "Path at depth 10 should be found with max_depth=15"
    );
    assert_eq!(
        result_deep.path.len(),
        11,
        "Path should contain all 11 nodes"
    );
}

#[test]
fn test_shortest_path_same_node() {
    // source_id == target_id. Path = [source_id], cost = 0.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);
    let a = add_fact(&mut graph, "A");

    let qe = QueryEngine::new();
    let result = qe
        .shortest_path(
            &graph,
            ShortestPathParams {
                source_id: a,
                target_id: a,
                edge_types: vec![],
                direction: TraversalDirection::Forward,
                max_depth: 10,
                weighted: false,
            },
        )
        .unwrap();

    assert!(result.found, "Same-node path should always be found");
    assert_eq!(
        result.path,
        vec![a],
        "Path should contain only the source node"
    );
    assert!(
        result.cost.abs() < f32::EPSILON,
        "Cost should be 0.0, got {}",
        result.cost
    );
}
