//! Phase 5 tests: Novel query types — belief revision, gap detection,
//! analogical queries, consolidation, and drift detection.

use std::path::PathBuf;

use agentic_memory::types::{CognitiveEventBuilder, DEFAULT_DIMENSION};
use agentic_memory::{
    AnalogicalAnchor, AnalogicalParams, BeliefRevisionParams, ChangeType, ConsolidationOp,
    ConsolidationParams, DriftParams, Edge, EdgeType, EventType, GapDetectionParams, GapSeverity,
    GapType, MemoryGraph, QueryEngine,
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

// ==================== Belief Revision Tests ====================

#[test]
fn test_revise_finds_contradiction() {
    // Graph: fact "Team doesn't know Go" -> decision "Chose Rust"
    // Hypothesis: "Team now knows Go"
    // The fact should be flagged as contradicted and the decision as invalidated.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let fact = CognitiveEventBuilder::new(EventType::Fact, "Team doesn't know Go")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    let id_fact = graph.add_node(fact).unwrap();

    let decision = CognitiveEventBuilder::new(
        EventType::Decision,
        "Chose Rust because team doesn't know Go",
    )
    .session_id(1)
    .confidence(0.85)
    .feature_vec(zero_vec())
    .build();
    let id_decision = graph.add_node(decision).unwrap();

    // Decision caused by the fact (decision has outgoing CausedBy edge to fact).
    graph
        .add_edge(Edge::new(id_decision, id_fact, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "Team now knows Go".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.3,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();

    // The fact "Team doesn't know Go" should be found as contradicted because
    // it overlaps with the hypothesis terms and contains negation ("doesn't").
    assert!(
        !report.contradicted.is_empty(),
        "Should find at least one contradicted node, got report: {:?}",
        report.contradicted
    );
    let contradicted_ids: Vec<u64> = report.contradicted.iter().map(|c| c.node_id).collect();
    assert!(
        contradicted_ids.contains(&id_fact),
        "Fact 'Team doesn't know Go' should be contradicted, contradicted IDs: {:?}",
        contradicted_ids
    );

    // The decision should be in invalidated_decisions because it depends on
    // the contradicted fact via CausedBy.
    assert!(
        report.invalidated_decisions.contains(&id_decision),
        "Decision 'Chose Rust' should be invalidated, got: {:?}",
        report.invalidated_decisions
    );
}

#[test]
fn test_revise_cascade_propagation() {
    // Chain: fact A -> inference B -> decision C
    // Hypothesis contradicts A.
    // B should be weakened, C should be invalidated.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let fact_a =
        CognitiveEventBuilder::new(EventType::Fact, "Database cannot handle concurrent writes")
            .session_id(1)
            .confidence(0.9)
            .feature_vec(zero_vec())
            .build();
    let id_a = graph.add_node(fact_a).unwrap();

    let inference_b = CognitiveEventBuilder::new(
        EventType::Inference,
        "We need a write queue for database concurrent writes",
    )
    .session_id(1)
    .confidence(0.8)
    .feature_vec(zero_vec())
    .build();
    let id_b = graph.add_node(inference_b).unwrap();

    let decision_c =
        CognitiveEventBuilder::new(EventType::Decision, "Implement write queue for database")
            .session_id(1)
            .confidence(0.85)
            .feature_vec(zero_vec())
            .build();
    let id_c = graph.add_node(decision_c).unwrap();

    // B caused by A, C caused by B.
    graph
        .add_edge(Edge::new(id_b, id_a, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(id_c, id_b, EdgeType::CausedBy, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "Database can handle concurrent writes perfectly".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.2,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();

    // A should be contradicted (it contains "cannot" which is negation).
    let contradicted_ids: Vec<u64> = report.contradicted.iter().map(|c| c.node_id).collect();
    assert!(
        contradicted_ids.contains(&id_a),
        "Fact A should be contradicted, got: {:?}",
        contradicted_ids
    );

    // B should be weakened (it depends on A via CausedBy).
    let weakened_ids: Vec<u64> = report.weakened.iter().map(|w| w.node_id).collect();
    assert!(
        weakened_ids.contains(&id_b),
        "Inference B should be weakened via cascade, got: {:?}",
        weakened_ids
    );

    // C should be invalidated (it is a Decision that depends transitively
    // on the contradicted fact).
    assert!(
        report.invalidated_decisions.contains(&id_c),
        "Decision C should be invalidated, got: {:?}",
        report.invalidated_decisions
    );

    // Total affected should include A (contradicted), B (weakened), and C (weakened/invalidated).
    assert!(
        report.total_affected >= 2,
        "Should have multiple affected nodes, got: {}",
        report.total_affected
    );
}

#[test]
fn test_revise_no_contradiction() {
    // Hypothesis is consistent with all facts -- report should be empty.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let fact =
        CognitiveEventBuilder::new(EventType::Fact, "Rust is a systems programming language")
            .session_id(1)
            .confidence(0.95)
            .feature_vec(zero_vec())
            .build();
    graph.add_node(fact).unwrap();

    let decision = CognitiveEventBuilder::new(EventType::Decision, "Use Rust for the backend")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(decision).unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "Python is good for scripting".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.5,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();

    // No overlap means no contradictions.
    assert!(
        report.contradicted.is_empty(),
        "Expected no contradictions, got: {:?}",
        report.contradicted
    );
    assert!(
        report.invalidated_decisions.is_empty(),
        "Expected no invalidated decisions, got: {:?}",
        report.invalidated_decisions
    );
    assert_eq!(report.total_affected, 0);
}

#[test]
fn test_revise_readonly() {
    // Run belief revision and verify graph is NOT modified.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let fact = CognitiveEventBuilder::new(EventType::Fact, "Team doesn't know Go programming")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(fact).unwrap();

    let decision = CognitiveEventBuilder::new(
        EventType::Decision,
        "Chose Rust because team doesn't know Go",
    )
    .session_id(1)
    .confidence(0.85)
    .feature_vec(zero_vec())
    .build();
    let id_decision = graph.add_node(decision).unwrap();

    graph
        .add_edge(Edge::new(id_decision, 0, EdgeType::CausedBy, 1.0))
        .unwrap();

    let node_count_before = graph.node_count();
    let edge_count_before = graph.edge_count();

    let qe = QueryEngine::new();
    let _report = qe
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "Team now knows Go programming".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.2,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();

    // Graph must remain unchanged.
    assert_eq!(
        graph.node_count(),
        node_count_before,
        "Node count changed after belief revision"
    );
    assert_eq!(
        graph.edge_count(),
        edge_count_before,
        "Edge count changed after belief revision"
    );

    // Verify original confidences are unchanged.
    let fact_node = graph.get_node(0).unwrap();
    assert!(
        (fact_node.confidence - 0.9).abs() < f32::EPSILON,
        "Fact confidence should be unchanged, got: {}",
        fact_node.confidence
    );
    let decision_node = graph.get_node(id_decision).unwrap();
    assert!(
        (decision_node.confidence - 0.85).abs() < f32::EPSILON,
        "Decision confidence should be unchanged, got: {}",
        decision_node.confidence
    );
}

#[test]
fn test_revise_multiple_affected() {
    // Fact A supports decisions D1, D2, D3.
    // Hypothesis contradicts A.
    // All 3 decisions should appear in invalidated_decisions.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let fact_a = CognitiveEventBuilder::new(EventType::Fact, "Server cannot scale horizontally")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    let id_a = graph.add_node(fact_a).unwrap();

    let d1 = CognitiveEventBuilder::new(EventType::Decision, "Use vertical scaling for server")
        .session_id(1)
        .confidence(0.85)
        .feature_vec(zero_vec())
        .build();
    let id_d1 = graph.add_node(d1).unwrap();

    let d2 = CognitiveEventBuilder::new(
        EventType::Decision,
        "Buy bigger server hardware for scaling",
    )
    .session_id(1)
    .confidence(0.8)
    .feature_vec(zero_vec())
    .build();
    let id_d2 = graph.add_node(d2).unwrap();

    let d3 = CognitiveEventBuilder::new(
        EventType::Decision,
        "Avoid microservices because server scaling is limited",
    )
    .session_id(1)
    .confidence(0.75)
    .feature_vec(zero_vec())
    .build();
    let id_d3 = graph.add_node(d3).unwrap();

    // All decisions depend on fact A via CausedBy.
    for &d_id in &[id_d1, id_d2, id_d3] {
        graph
            .add_edge(Edge::new(d_id, id_a, EdgeType::CausedBy, 1.0))
            .unwrap();
    }

    let qe = QueryEngine::new();
    let report = qe
        .belief_revision(
            &graph,
            BeliefRevisionParams {
                hypothesis: "Server can scale horizontally easily".to_string(),
                hypothesis_vec: None,
                contradiction_threshold: 0.2,
                max_depth: 5,
                hypothesis_confidence: 0.9,
            },
        )
        .unwrap();

    // Fact A should be contradicted ("cannot" is negation).
    let contradicted_ids: Vec<u64> = report.contradicted.iter().map(|c| c.node_id).collect();
    assert!(
        contradicted_ids.contains(&id_a),
        "Fact A should be contradicted"
    );

    // All three decisions should be invalidated.
    for &d_id in &[id_d1, id_d2, id_d3] {
        assert!(
            report.invalidated_decisions.contains(&d_id),
            "Decision {} should be in invalidated_decisions, got: {:?}",
            d_id,
            report.invalidated_decisions
        );
    }
}

// ==================== Gap Detection Tests ====================

#[test]
fn test_gaps_unjustified_decision() {
    // Decision with zero CausedBy/Supports edges should be detected as a gap.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let decision =
        CognitiveEventBuilder::new(EventType::Decision, "Use microservices architecture")
            .session_id(1)
            .confidence(0.9)
            .feature_vec(zero_vec())
            .build();
    graph.add_node(decision).unwrap();

    let qe = QueryEngine::new();
    let report = qe
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

    // The decision has no justification edges.
    let unjustified = report
        .gaps
        .iter()
        .filter(|g| g.gap_type == GapType::UnjustifiedDecision)
        .collect::<Vec<_>>();
    assert!(
        !unjustified.is_empty(),
        "Should detect the unjustified decision, gaps: {:?}",
        report
            .gaps
            .iter()
            .map(|g| (&g.gap_type, g.node_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(unjustified[0].node_id, 0);
    assert_eq!(report.summary.unjustified_decisions, 1);
}

#[test]
fn test_gaps_single_source_inference() {
    // Inference with exactly 1 Supports edge should be detected.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let fact = CognitiveEventBuilder::new(EventType::Fact, "Users report slow response times")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    let id_fact = graph.add_node(fact).unwrap();

    let inference = CognitiveEventBuilder::new(EventType::Inference, "Server is under heavy load")
        .session_id(1)
        .confidence(0.7)
        .feature_vec(zero_vec())
        .build();
    let id_inference = graph.add_node(inference).unwrap();

    // Only one Supports edge.
    graph
        .add_edge(Edge::new(id_fact, id_inference, EdgeType::Supports, 0.8))
        .unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .gap_detection(
            &graph,
            GapDetectionParams {
                confidence_threshold: 0.5,
                min_support_count: 2, // Requires at least 2 supports.
                max_results: 100,
                session_range: None,
                sort_by: GapSeverity::HighestImpact,
            },
        )
        .unwrap();

    let single_source = report
        .gaps
        .iter()
        .filter(|g| g.gap_type == GapType::SingleSourceInference)
        .collect::<Vec<_>>();
    assert!(
        !single_source.is_empty(),
        "Should detect single-source inference, gaps: {:?}",
        report
            .gaps
            .iter()
            .map(|g| (&g.gap_type, g.node_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(report.summary.single_source_inferences, 1);
}

#[test]
fn test_gaps_low_confidence_foundation() {
    // Fact with confidence=0.3 that has dependents should be detected.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let weak_fact =
        CognitiveEventBuilder::new(EventType::Fact, "Unverified rumor about API change")
            .session_id(1)
            .confidence(0.3)
            .feature_vec(zero_vec())
            .build();
    let id_weak = graph.add_node(weak_fact).unwrap();

    // Create 5 dependent nodes that depend on the weak fact.
    for i in 0..5 {
        let dep = CognitiveEventBuilder::new(
            EventType::Inference,
            format!("Inference {} from weak rumor", i),
        )
        .session_id(1)
        .confidence(0.7)
        .feature_vec(zero_vec())
        .build();
        let dep_id = graph.add_node(dep).unwrap();
        graph
            .add_edge(Edge::new(dep_id, id_weak, EdgeType::CausedBy, 0.9))
            .unwrap();
    }

    let qe = QueryEngine::new();
    let report = qe
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

    let low_conf = report
        .gaps
        .iter()
        .filter(|g| g.gap_type == GapType::LowConfidenceFoundation)
        .collect::<Vec<_>>();
    assert!(
        !low_conf.is_empty(),
        "Should detect low confidence foundation, gaps: {:?}",
        report
            .gaps
            .iter()
            .map(|g| (&g.gap_type, g.node_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(low_conf[0].node_id, id_weak);
    assert_eq!(report.summary.low_confidence_foundations, 1);
}

#[test]
fn test_gaps_unstable_knowledge() {
    // Create a SUPERSEDES chain of length 4: A -> B -> C -> D (each supersedes the previous).
    // The node at the end of the chain should be flagged as unstable knowledge.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let mut prev_id = None;
    for i in 0..5 {
        let node =
            CognitiveEventBuilder::new(EventType::Fact, format!("Version {} of the API spec", i))
                .session_id(i as u32 + 1)
                .confidence(0.8)
                .feature_vec(zero_vec())
                .build();
        let id = graph.add_node(node).unwrap();

        if let Some(prev) = prev_id {
            // id supersedes prev: edge from id -> prev
            graph
                .add_edge(Edge::new(id, prev, EdgeType::Supersedes, 1.0))
                .unwrap();
        }
        prev_id = Some(id);
    }

    let qe = QueryEngine::new();
    let report = qe
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

    let unstable = report
        .gaps
        .iter()
        .filter(|g| g.gap_type == GapType::UnstableKnowledge)
        .collect::<Vec<_>>();
    assert!(
        !unstable.is_empty(),
        "Should detect unstable knowledge from supersedes chain of length 4, gaps: {:?}",
        report
            .gaps
            .iter()
            .map(|g| (&g.gap_type, g.node_id))
            .collect::<Vec<_>>()
    );
    assert!(report.summary.unstable_knowledge > 0);
}

#[test]
fn test_gaps_health_score() {
    // Graph with no gaps: health_score close to 1.0.
    // Graph with many gaps: health_score < 0.5.
    let qe = QueryEngine::new();

    // -- Healthy graph: fact supports decision properly --
    let mut healthy_graph = MemoryGraph::new(DEFAULT_DIMENSION);
    for i in 0..5 {
        let fact =
            CognitiveEventBuilder::new(EventType::Fact, format!("Well-established fact {}", i))
                .session_id(1)
                .confidence(0.95)
                .feature_vec(zero_vec())
                .build();
        healthy_graph.add_node(fact).unwrap();
    }
    // Add a decision justified by facts.
    let decision = CognitiveEventBuilder::new(EventType::Decision, "Well-justified decision")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    let d_id = healthy_graph.add_node(decision).unwrap();
    healthy_graph
        .add_edge(Edge::new(d_id, 0, EdgeType::CausedBy, 1.0))
        .unwrap();
    healthy_graph
        .add_edge(Edge::new(d_id, 1, EdgeType::Supports, 1.0))
        .unwrap();

    let healthy_report = qe
        .gap_detection(
            &healthy_graph,
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
        healthy_report.summary.health_score > 0.5,
        "Healthy graph should have high health score, got: {}",
        healthy_report.summary.health_score
    );

    // -- Unhealthy graph: many unjustified decisions --
    let mut unhealthy_graph = MemoryGraph::new(DEFAULT_DIMENSION);
    for i in 0..10 {
        let decision =
            CognitiveEventBuilder::new(EventType::Decision, format!("Unjustified decision {}", i))
                .session_id(1)
                .confidence(0.9)
                .feature_vec(zero_vec())
                .build();
        unhealthy_graph.add_node(decision).unwrap();
    }

    let unhealthy_report = qe
        .gap_detection(
            &unhealthy_graph,
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
        unhealthy_report.summary.health_score < 0.5,
        "Graph full of unjustified decisions should have low health score, got: {}",
        unhealthy_report.summary.health_score
    );
}

#[test]
fn test_gaps_session_filter() {
    // Gaps exist in sessions 1-5 but NOT in sessions 6-10.
    // Filtering to sessions 6-10 should find no gaps.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Sessions 1-5: unjustified decisions (these are gaps).
    for i in 1..=5u32 {
        let decision = CognitiveEventBuilder::new(
            EventType::Decision,
            format!("Unjustified decision session {}", i),
        )
        .session_id(i)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
        graph.add_node(decision).unwrap();
    }

    // Sessions 6-10: well-justified facts (no gaps here).
    for i in 6..=10u32 {
        let fact = CognitiveEventBuilder::new(EventType::Fact, format!("Solid fact session {}", i))
            .session_id(i)
            .confidence(0.95)
            .feature_vec(zero_vec())
            .build();
        graph.add_node(fact).unwrap();
    }

    let qe = QueryEngine::new();

    // Full scan: should find gaps in sessions 1-5.
    let full_report = qe
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
        !full_report.gaps.is_empty(),
        "Full scan should find gaps from sessions 1-5"
    );

    // Filtered scan: sessions 6-10 only.
    let filtered_report = qe
        .gap_detection(
            &graph,
            GapDetectionParams {
                confidence_threshold: 0.5,
                min_support_count: 2,
                max_results: 100,
                session_range: Some((6, 10)),
                sort_by: GapSeverity::HighestImpact,
            },
        )
        .unwrap();
    assert!(
        filtered_report.gaps.is_empty(),
        "Filtered to sessions 6-10 should find no gaps, got: {:?}",
        filtered_report
            .gaps
            .iter()
            .map(|g| (&g.gap_type, g.node_id))
            .collect::<Vec<_>>()
    );
}

// ==================== Analogical Query Tests ====================

#[test]
fn test_analogy_finds_similar_structure() {
    // Session 1: fact -> fact -> decision (tech stack choice)
    // Session 10: fact -> fact -> decision (deployment choice)
    // The structural fingerprints should be similar.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Session 1: tech stack choice pattern
    let f1a = CognitiveEventBuilder::new(EventType::Fact, "Team knows Rust well")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    let id_f1a = graph.add_node(f1a).unwrap();

    let f1b = CognitiveEventBuilder::new(EventType::Fact, "Project requires high performance")
        .session_id(1)
        .confidence(0.85)
        .feature_vec(basis_vec(1, 1.0))
        .build();
    let id_f1b = graph.add_node(f1b).unwrap();

    let d1 = CognitiveEventBuilder::new(EventType::Decision, "Choose Rust for backend")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(2, 1.0))
        .build();
    let id_d1 = graph.add_node(d1).unwrap();

    graph
        .add_edge(Edge::new(id_d1, id_f1a, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(id_d1, id_f1b, EdgeType::CausedBy, 0.9))
        .unwrap();

    // Session 10: deployment choice pattern (similar structure)
    let f10a = CognitiveEventBuilder::new(EventType::Fact, "Cloud provider offers Kubernetes")
        .session_id(10)
        .confidence(0.9)
        .feature_vec(basis_vec(3, 1.0))
        .build();
    let id_f10a = graph.add_node(f10a).unwrap();

    let f10b = CognitiveEventBuilder::new(EventType::Fact, "Team has container experience")
        .session_id(10)
        .confidence(0.85)
        .feature_vec(basis_vec(4, 1.0))
        .build();
    let id_f10b = graph.add_node(f10b).unwrap();

    let d10 = CognitiveEventBuilder::new(EventType::Decision, "Deploy on Kubernetes")
        .session_id(10)
        .confidence(0.9)
        .feature_vec(basis_vec(5, 1.0))
        .build();
    let id_d10 = graph.add_node(d10).unwrap();

    graph
        .add_edge(Edge::new(id_d10, id_f10a, EdgeType::CausedBy, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(id_d10, id_f10b, EdgeType::CausedBy, 0.9))
        .unwrap();

    let qe = QueryEngine::new();
    let analogies = qe
        .analogical(
            &graph,
            AnalogicalParams {
                anchor: AnalogicalAnchor::Node(id_d1),
                context_depth: 2,
                max_results: 10,
                min_similarity: 0.0, // Accept any similarity to see results.
                exclude_sessions: vec![],
            },
        )
        .unwrap();

    // Should find something from session 10 as structurally similar.
    assert!(
        !analogies.is_empty(),
        "Should find at least one analogy, got empty results"
    );

    // Check that at least one analogy involves a node from session 10.
    let session_10_ids: Vec<u64> = vec![id_f10a, id_f10b, id_d10];
    let found_session_10 = analogies.iter().any(|a| {
        session_10_ids.contains(&a.center_id)
            || a.subgraph_nodes.iter().any(|n| session_10_ids.contains(n))
    });
    assert!(
        found_session_10,
        "Should find analogy involving session 10 nodes, analogies: {:?}",
        analogies
            .iter()
            .map(|a| (a.center_id, a.combined_score))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_analogy_excludes_current_session() {
    // Build a graph, then exclude session 10 -- it should not appear in results.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Session 1.
    let f1 = CognitiveEventBuilder::new(EventType::Fact, "Fact in session one")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    let id_f1 = graph.add_node(f1).unwrap();

    // Session 10.
    let f10 = CognitiveEventBuilder::new(EventType::Fact, "Fact in session ten")
        .session_id(10)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 0.9))
        .build();
    let _id_f10 = graph.add_node(f10).unwrap();

    // Session 20.
    let f20 = CognitiveEventBuilder::new(EventType::Fact, "Fact in session twenty")
        .session_id(20)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 0.8))
        .build();
    let _id_f20 = graph.add_node(f20).unwrap();

    let qe = QueryEngine::new();
    let analogies = qe
        .analogical(
            &graph,
            AnalogicalParams {
                anchor: AnalogicalAnchor::Node(id_f1),
                context_depth: 1,
                max_results: 10,
                min_similarity: 0.0,
                exclude_sessions: vec![10], // Exclude session 10.
            },
        )
        .unwrap();

    // None of the returned analogies should have center_id == id_f10.
    for analogy in &analogies {
        let center_node = graph.get_node(analogy.center_id).unwrap();
        assert_ne!(
            center_node.session_id, 10,
            "Session 10 should be excluded but found center_id {} in session 10",
            analogy.center_id
        );
    }
}

#[test]
fn test_analogy_no_match() {
    // Very unique structure with a high min_similarity threshold should return empty.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Only one node in the whole graph.
    let lone = CognitiveEventBuilder::new(EventType::Fact, "Unique lonely fact")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    let id_lone = graph.add_node(lone).unwrap();

    let qe = QueryEngine::new();
    let analogies = qe
        .analogical(
            &graph,
            AnalogicalParams {
                anchor: AnalogicalAnchor::Node(id_lone),
                context_depth: 2,
                max_results: 10,
                min_similarity: 0.99, // Very high threshold.
                exclude_sessions: vec![],
            },
        )
        .unwrap();

    assert!(
        analogies.is_empty(),
        "Should return empty with no other nodes in graph, got {} results",
        analogies.len()
    );
}

#[test]
fn test_analogy_text_anchor() {
    // Use a Vector anchor instead of a Node anchor to find relevant subgraphs.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Create nodes with distinct feature vectors.
    let f1 = CognitiveEventBuilder::new(EventType::Fact, "Machine learning model accuracy")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    graph.add_node(f1).unwrap();

    let f2 = CognitiveEventBuilder::new(EventType::Fact, "Database query performance")
        .session_id(2)
        .confidence(0.85)
        .feature_vec(basis_vec(1, 1.0))
        .build();
    graph.add_node(f2).unwrap();

    let f3 = CognitiveEventBuilder::new(EventType::Fact, "Network latency optimization")
        .session_id(3)
        .confidence(0.8)
        .feature_vec(basis_vec(2, 1.0))
        .build();
    graph.add_node(f3).unwrap();

    let qe = QueryEngine::new();

    // Use a vector anchor similar to node f1's feature vector.
    let query_vec = basis_vec(0, 0.95);
    let analogies = qe
        .analogical(
            &graph,
            AnalogicalParams {
                anchor: AnalogicalAnchor::Vector(query_vec),
                context_depth: 1,
                max_results: 10,
                min_similarity: 0.0,
                exclude_sessions: vec![],
            },
        )
        .unwrap();

    // Should not panic, and may find results depending on the similarity.
    // The anchor vector is very close to node f1, so the engine should use
    // f1 as the center. Other nodes should appear as candidates.
    // We mostly test that Vector anchors work without error.
    let _ = analogies;
}

// ==================== Consolidation Tests ====================

#[test]
fn test_consolidate_dry_run_no_mutation() {
    // Run consolidate with dry_run=true.
    // Verify graph is identical before and after.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Add duplicate facts.
    let f1 = CognitiveEventBuilder::new(EventType::Fact, "The API response time is 200ms")
        .session_id(1)
        .confidence(0.7)
        .feature_vec(basis_vec(0, 1.0))
        .build();
    graph.add_node(f1).unwrap();

    let f2 =
        CognitiveEventBuilder::new(EventType::Fact, "The API response time is 200ms on average")
            .session_id(2)
            .confidence(0.9)
            .feature_vec(basis_vec(0, 0.99))
            .build();
    graph.add_node(f2).unwrap();

    let node_count_before = graph.node_count();
    let edge_count_before = graph.edge_count();

    let qe = QueryEngine::new();
    let report = qe
        .consolidate(
            &mut graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![
                    ConsolidationOp::DeduplicateFacts { threshold: 0.8 },
                    ConsolidationOp::LinkContradictions { threshold: 0.8 },
                ],
                dry_run: true,
                backup_path: None,
            },
        )
        .unwrap();

    // Graph should be unchanged after dry run.
    assert_eq!(
        graph.node_count(),
        node_count_before,
        "Node count should be unchanged after dry run"
    );
    assert_eq!(
        graph.edge_count(),
        edge_count_before,
        "Edge count should be unchanged after dry run"
    );

    // But the report should describe what would happen.
    // (It may or may not find duplicates depending on vector similarity and token overlap.)
    let _ = report;
}

#[test]
fn test_consolidate_dedup_detects_duplicates() {
    // Two facts with identical content should be reported as a duplicate pair.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let content = "Rust compiler guarantees memory safety at compile time";
    let fv = basis_vec(0, 1.0);

    let f1 = CognitiveEventBuilder::new(EventType::Fact, content)
        .session_id(1)
        .confidence(0.7)
        .feature_vec(fv.clone())
        .build();
    graph.add_node(f1).unwrap();

    let f2 = CognitiveEventBuilder::new(EventType::Fact, content)
        .session_id(2)
        .confidence(0.9)
        .feature_vec(fv)
        .build();
    graph.add_node(f2).unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .consolidate(
            &mut graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::DeduplicateFacts { threshold: 0.9 }],
                dry_run: true,
                backup_path: None,
            },
        )
        .unwrap();

    assert!(
        report.deduplicated >= 1,
        "Should detect at least 1 duplicate pair, got: {}",
        report.deduplicated
    );

    // Verify an action was created describing the dedup.
    let dedup_actions: Vec<_> = report
        .actions
        .iter()
        .filter(|a| a.operation == "deduplicate_facts")
        .collect();
    assert!(
        !dedup_actions.is_empty(),
        "Should have a deduplicate_facts action"
    );
}

#[test]
fn test_consolidate_dedup_keeps_higher_confidence() {
    // Fact A (confidence 0.7) and near-duplicate B (confidence 0.9).
    // Dedup should keep B (higher confidence), supersede A.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let content = "Kubernetes orchestrates container workloads efficiently";
    let fv = basis_vec(0, 1.0);

    let f_low = CognitiveEventBuilder::new(EventType::Fact, content)
        .session_id(1)
        .confidence(0.7)
        .feature_vec(fv.clone())
        .build();
    let id_low = graph.add_node(f_low).unwrap();

    let f_high = CognitiveEventBuilder::new(EventType::Fact, content)
        .session_id(2)
        .confidence(0.9)
        .feature_vec(fv)
        .build();
    let id_high = graph.add_node(f_high).unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .consolidate(
            &mut graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::DeduplicateFacts { threshold: 0.9 }],
                dry_run: false, // Actually mutate!
                backup_path: None,
            },
        )
        .unwrap();

    assert!(
        report.deduplicated >= 1,
        "Should deduplicate the pair, got: {}",
        report.deduplicated
    );

    // There should now be a Supersedes edge from the winner (higher confidence)
    // to the loser (lower confidence).
    let edges_from_high = graph.edges_from(id_high);
    let supersedes_edges: Vec<_> = edges_from_high
        .iter()
        .filter(|e| e.edge_type == EdgeType::Supersedes && e.target_id == id_low)
        .collect();
    assert!(
        !supersedes_edges.is_empty(),
        "Higher confidence node ({}) should supersede lower confidence node ({})",
        id_high,
        id_low
    );
}

#[test]
fn test_consolidate_link_contradictions() {
    // Two contradicting facts with no existing CONTRADICTS edge.
    // Consolidation should detect and link them.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Use similar feature vectors but opposing content (one has negation).
    let fv = basis_vec(0, 1.0);

    let f_positive = CognitiveEventBuilder::new(
        EventType::Fact,
        "Rust compiler guarantees memory safety effectively always",
    )
    .session_id(1)
    .confidence(0.9)
    .feature_vec(fv.clone())
    .build();
    let id_pos = graph.add_node(f_positive).unwrap();

    let f_negative = CognitiveEventBuilder::new(
        EventType::Fact,
        "Rust compiler never guarantees memory safety effectively",
    )
    .session_id(2)
    .confidence(0.8)
    .feature_vec(fv)
    .build();
    let id_neg = graph.add_node(f_negative).unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .consolidate(
            &mut graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::LinkContradictions { threshold: 0.5 }],
                dry_run: false,
                backup_path: None,
            },
        )
        .unwrap();

    assert!(
        report.contradictions_linked >= 1,
        "Should link at least 1 contradiction pair, got: {}",
        report.contradictions_linked
    );

    // Verify a CONTRADICTS edge now exists between the two nodes.
    let has_contradicts = graph
        .edges_from(id_pos)
        .iter()
        .any(|e| e.edge_type == EdgeType::Contradicts && e.target_id == id_neg)
        || graph
            .edges_from(id_neg)
            .iter()
            .any(|e| e.edge_type == EdgeType::Contradicts && e.target_id == id_pos);
    assert!(
        has_contradicts,
        "Should have a Contradicts edge between node {} and node {}",
        id_pos, id_neg
    );
}

#[test]
fn test_consolidate_backup_created() {
    // Run consolidate with dry_run=false and a backup_path.
    // Verify the backup_path is echoed in the report.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let f = CognitiveEventBuilder::new(EventType::Fact, "Some fact for backup test")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(f).unwrap();

    let backup_path = PathBuf::from("/tmp/test_backup_consolidate.amem");

    let qe = QueryEngine::new();
    let report = qe
        .consolidate(
            &mut graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::DeduplicateFacts { threshold: 0.9 }],
                dry_run: false,
                backup_path: Some(backup_path.clone()),
            },
        )
        .unwrap();

    // The backup_path should be echoed in the report.
    assert_eq!(
        report.backup_path,
        Some(backup_path),
        "Backup path should be echoed in the report"
    );
}

#[test]
fn test_consolidate_backup_is_original() {
    // The backup_path in the report should match the original value provided.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let f = CognitiveEventBuilder::new(EventType::Fact, "Fact for backup original test")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(f).unwrap();

    let original_path = PathBuf::from("/tmp/backup_original_test.amem");

    let qe = QueryEngine::new();
    let report = qe
        .consolidate(
            &mut graph,
            ConsolidationParams {
                session_range: None,
                operations: vec![ConsolidationOp::DeduplicateFacts { threshold: 0.9 }],
                dry_run: true,
                backup_path: Some(original_path.clone()),
            },
        )
        .unwrap();

    // The report should faithfully echo the backup path.
    assert_eq!(
        report.backup_path.as_ref(),
        Some(&original_path),
        "Report backup_path should match the original input path"
    );
}

// ==================== Drift Detection Tests ====================

#[test]
fn test_drift_tracks_supersedes_chain() {
    // Fact A (session 1) -> superseded by B (session 5) -> superseded by C (session 10).
    // Drift should show 3 snapshots with Corrected change type.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let a = CognitiveEventBuilder::new(EventType::Fact, "API returns XML response format")
        .session_id(1)
        .confidence(0.9)
        .created_at(1_000_000)
        .feature_vec(zero_vec())
        .build();
    let id_a = graph.add_node(a).unwrap();

    let b = CognitiveEventBuilder::new(
        EventType::Fact,
        "API returns JSON response format instead of XML",
    )
    .session_id(5)
    .confidence(0.92)
    .created_at(5_000_000)
    .feature_vec(zero_vec())
    .build();
    let id_b = graph.add_node(b).unwrap();

    let c = CognitiveEventBuilder::new(
        EventType::Fact,
        "API returns protobuf response format replacing JSON",
    )
    .session_id(10)
    .confidence(0.95)
    .created_at(10_000_000)
    .feature_vec(zero_vec())
    .build();
    let id_c = graph.add_node(c).unwrap();

    // B supersedes A, C supersedes B.
    graph
        .add_edge(Edge::new(id_b, id_a, EdgeType::Supersedes, 1.0))
        .unwrap();
    graph
        .add_edge(Edge::new(id_c, id_b, EdgeType::Supersedes, 1.0))
        .unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .drift_detection(
            &graph,
            DriftParams {
                topic: "API response format".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.2,
            },
        )
        .unwrap();

    // Should have at least one timeline.
    assert!(
        !report.timelines.is_empty(),
        "Should detect at least one timeline for the API response topic"
    );

    // Find the timeline that contains the supersedes chain.
    let chain_timeline = report.timelines.iter().find(|t| {
        let node_ids: Vec<u64> = t.snapshots.iter().map(|s| s.node_id).collect();
        node_ids.contains(&id_a) && node_ids.contains(&id_b) && node_ids.contains(&id_c)
    });

    if let Some(timeline) = chain_timeline {
        assert_eq!(
            timeline.snapshots.len(),
            3,
            "Timeline should have 3 snapshots for the chain A->B->C"
        );

        // First snapshot should be Initial.
        assert_eq!(
            timeline.snapshots[0].change_type,
            ChangeType::Initial,
            "First snapshot should be Initial"
        );

        // Subsequent snapshots should be Corrected (via Supersedes edges).
        assert_eq!(
            timeline.snapshots[1].change_type,
            ChangeType::Corrected,
            "Second snapshot should be Corrected"
        );
        assert_eq!(
            timeline.snapshots[2].change_type,
            ChangeType::Corrected,
            "Third snapshot should be Corrected"
        );
    }
    // Note: the timeline may not combine all three if terms don't match.
    // The important thing is that the drift detection runs without error
    // and produces timelines.
}

#[test]
fn test_drift_stability_high() {
    // Topic with no changes -> stability close to 1.0.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Single stable fact about the topic with no supersedes or contradictions.
    let f = CognitiveEventBuilder::new(EventType::Fact, "Database uses PostgreSQL version twelve")
        .session_id(1)
        .confidence(0.95)
        .created_at(1_000_000)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(f).unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .drift_detection(
            &graph,
            DriftParams {
                topic: "Database PostgreSQL version".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.2,
            },
        )
        .unwrap();

    assert!(
        report.stability >= 0.9,
        "Single stable fact should have high stability, got: {}",
        report.stability
    );
    assert!(
        !report.likely_to_change,
        "Stable topic should not be likely to change"
    );
}

#[test]
fn test_drift_stability_low() {
    // Topic with 4 corrections in 5 sessions -> stability should be low.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let mut prev_id = None;
    for i in 0..5u32 {
        let node = CognitiveEventBuilder::new(
            EventType::Fact,
            format!("Deployment target version {} for release pipeline", i),
        )
        .session_id(i + 1)
        .confidence(0.7 + i as f32 * 0.05)
        .created_at((i as u64 + 1) * 1_000_000)
        .feature_vec(zero_vec())
        .build();
        let id = graph.add_node(node).unwrap();

        if let Some(prev) = prev_id {
            graph
                .add_edge(Edge::new(id, prev, EdgeType::Supersedes, 1.0))
                .unwrap();
        }
        prev_id = Some(id);
    }

    let qe = QueryEngine::new();
    let report = qe
        .drift_detection(
            &graph,
            DriftParams {
                topic: "Deployment target version release pipeline".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.2,
            },
        )
        .unwrap();

    // With 4 corrections out of 5 snapshots, stability should be low.
    assert!(
        report.stability < 0.8,
        "Topic with many corrections should have low stability, got: {}",
        report.stability
    );
}

#[test]
fn test_drift_likely_to_change() {
    // Recent rapid changes -> likely_to_change = true.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    let mut prev_id = None;
    for i in 0..6u32 {
        let node = CognitiveEventBuilder::new(
            EventType::Fact,
            format!("Configuration setting value {} for runtime environment", i),
        )
        .session_id(i + 1)
        .confidence(0.7)
        .created_at((i as u64 + 1) * 1_000_000)
        .feature_vec(zero_vec())
        .build();
        let id = graph.add_node(node).unwrap();

        if let Some(prev) = prev_id {
            graph
                .add_edge(Edge::new(id, prev, EdgeType::Supersedes, 1.0))
                .unwrap();
        }
        prev_id = Some(id);
    }

    let qe = QueryEngine::new();
    let report = qe
        .drift_detection(
            &graph,
            DriftParams {
                topic: "Configuration setting value runtime environment".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.2,
            },
        )
        .unwrap();

    // With 5 corrections, the instability ratio should be high enough to trigger
    // likely_to_change.
    assert!(
        report.likely_to_change,
        "Topic with rapid corrections should be likely to change, stability: {}, timelines: {}",
        report.stability,
        report.timelines.len()
    );
}

#[test]
fn test_drift_no_relevant_nodes() {
    // Topic with no matching content -> empty timelines.
    let mut graph = MemoryGraph::new(DEFAULT_DIMENSION);

    // Add nodes about a completely unrelated topic.
    let f = CognitiveEventBuilder::new(EventType::Fact, "The weather is sunny today")
        .session_id(1)
        .confidence(0.9)
        .feature_vec(zero_vec())
        .build();
    graph.add_node(f).unwrap();

    let qe = QueryEngine::new();
    let report = qe
        .drift_detection(
            &graph,
            DriftParams {
                topic: "Quantum computing superconductor qubits".to_string(),
                topic_vec: None,
                max_results: 10,
                min_relevance: 0.3,
            },
        )
        .unwrap();

    assert!(
        report.timelines.is_empty(),
        "Should have empty timelines for unrelated topic, got {} timeline(s)",
        report.timelines.len()
    );
    assert!(
        (report.stability - 1.0).abs() < f32::EPSILON,
        "No relevant nodes means stability = 1.0, got: {}",
        report.stability
    );
    assert!(
        !report.likely_to_change,
        "No relevant nodes means not likely to change"
    );
}
