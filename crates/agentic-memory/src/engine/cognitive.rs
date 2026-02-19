//! Cognitive queries: belief revision, gap detection, analogical query, drift detection (queries 12-16).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::engine::tokenizer::Tokenizer;
use crate::graph::MemoryGraph;
use crate::index::cosine_similarity;
use crate::types::{AmemResult, CognitiveEvent, Edge, EdgeType, EventType};

// ---------------------------------------------------------------------------
// Query 12: Belief Revision
// ---------------------------------------------------------------------------

/// Parameters for a belief revision query.
#[derive(Debug, Clone)]
pub struct BeliefRevisionParams {
    /// The hypothesis to test against the knowledge graph.
    pub hypothesis: String,
    /// Optional feature vector for the hypothesis (for similarity search).
    pub hypothesis_vec: Option<Vec<f32>>,
    /// Minimum similarity threshold to consider something a contradiction.
    pub contradiction_threshold: f32,
    /// Maximum depth for cascade propagation.
    pub max_depth: u32,
    /// Confidence of the hypothesis itself.
    pub hypothesis_confidence: f32,
}

/// A node that contradicts the hypothesis.
#[derive(Debug, Clone)]
pub struct ContradictedNode {
    /// The node ID of the contradicting node.
    pub node_id: u64,
    /// How strongly it contradicts (higher = stronger).
    pub contradiction_strength: f32,
    /// Why it was flagged (e.g., negation detected, explicit Contradicts edge).
    pub reason: String,
}

/// A node whose confidence is weakened by the hypothesis.
#[derive(Debug, Clone)]
pub struct WeakenedNode {
    /// The node ID that was weakened.
    pub node_id: u64,
    /// Original confidence of the node.
    pub original_confidence: f32,
    /// New (reduced) confidence after applying the hypothesis.
    pub revised_confidence: f32,
    /// How many hops from a contradiction this node is.
    pub depth: u32,
}

/// A single step in a cascade of weakened beliefs.
#[derive(Debug, Clone)]
pub struct CascadeStep {
    /// The node ID affected.
    pub node_id: u64,
    /// The edge type that propagated the weakening.
    pub via_edge: EdgeType,
    /// The source node that caused this weakening.
    pub from_node: u64,
    /// Depth in the cascade.
    pub depth: u32,
}

/// The effect of a cascade step (used for export compatibility).
#[derive(Debug, Clone)]
pub struct CascadeEffect {
    /// The node ID affected.
    pub node_id: u64,
    /// The weakening factor applied.
    pub weakening: f32,
}

/// Full report from a belief revision query.
#[derive(Debug, Clone)]
pub struct RevisionReport {
    /// Nodes that directly contradict the hypothesis.
    pub contradicted: Vec<ContradictedNode>,
    /// Nodes whose confidence is weakened by cascade.
    pub weakened: Vec<WeakenedNode>,
    /// Decision node IDs that are invalidated (transitively depend on contradicted nodes).
    pub invalidated_decisions: Vec<u64>,
    /// Total number of affected nodes.
    pub total_affected: usize,
    /// The cascade path showing how weakening propagated.
    pub cascade: Vec<CascadeStep>,
}

// ---------------------------------------------------------------------------
// Query 13: Gap Detection
// ---------------------------------------------------------------------------

/// How to sort/rank detected gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapSeverity {
    /// Gaps with the most downstream impact first.
    HighestImpact,
    /// Gaps with the lowest confidence first.
    LowestConfidence,
    /// Most recent gaps first.
    MostRecent,
}

/// The type of knowledge gap detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapType {
    /// A Decision node with no incoming CausedBy or Supports edges.
    UnjustifiedDecision,
    /// An Inference node with only one incoming Supports edge.
    SingleSourceInference,
    /// A Fact or Inference node with confidence below threshold.
    LowConfidenceFoundation,
    /// A node that has been superseded 3+ times (unstable knowledge).
    UnstableKnowledge,
    /// A node with very low decay score (stale evidence).
    StaleEvidence,
}

/// Parameters for gap detection.
#[derive(Debug, Clone)]
pub struct GapDetectionParams {
    /// Nodes with confidence below this are flagged as low-confidence foundations.
    pub confidence_threshold: f32,
    /// Minimum number of Supports edges an Inference should have.
    pub min_support_count: u32,
    /// Maximum number of gaps to return.
    pub max_results: usize,
    /// Optional session range filter (inclusive start, inclusive end).
    pub session_range: Option<(u32, u32)>,
    /// How to sort the detected gaps.
    pub sort_by: GapSeverity,
}

/// A single detected gap in the knowledge graph.
#[derive(Debug, Clone)]
pub struct Gap {
    /// The node ID that has a gap.
    pub node_id: u64,
    /// The type of gap.
    pub gap_type: GapType,
    /// Severity score (0.0 = minor, 1.0 = critical).
    pub severity: f32,
    /// Human-readable description of the gap.
    pub description: String,
    /// Number of downstream nodes affected by this gap.
    pub downstream_count: usize,
}

/// Summary statistics of all detected gaps.
#[derive(Debug, Clone)]
pub struct GapSummary {
    /// Total number of gaps detected.
    pub total_gaps: usize,
    /// Number of unjustified decisions.
    pub unjustified_decisions: usize,
    /// Number of single-source inferences.
    pub single_source_inferences: usize,
    /// Number of low-confidence foundations.
    pub low_confidence_foundations: usize,
    /// Number of unstable knowledge nodes.
    pub unstable_knowledge: usize,
    /// Number of stale evidence nodes.
    pub stale_evidence: usize,
    /// Overall health score: 1.0 - (total_gaps / total_nodes).clamp(0.0, 1.0).
    pub health_score: f32,
}

/// Full report from gap detection.
#[derive(Debug, Clone)]
pub struct GapReport {
    /// The detected gaps, sorted according to the params.
    pub gaps: Vec<Gap>,
    /// Summary statistics.
    pub summary: GapSummary,
}

// ---------------------------------------------------------------------------
// Query 14: Analogical Query
// ---------------------------------------------------------------------------

/// What to use as the anchor for an analogical query.
#[derive(Debug, Clone)]
pub enum AnalogicalAnchor {
    /// Use a specific node ID as the anchor.
    Node(u64),
    /// Use a feature vector as the anchor.
    Vector(Vec<f32>),
}

/// A structural fingerprint of a subgraph for comparison.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Count of each EventType in the subgraph.
    pub event_type_counts: HashMap<u8, usize>,
    /// Count of each EdgeType in the subgraph.
    pub edge_type_counts: HashMap<u8, usize>,
    /// Maximum depth of causal chains (CausedBy edges).
    pub causal_chain_depth: u32,
    /// Average branching factor (outgoing edges per node).
    pub branching_factor: f32,
}

/// A single analogy found.
#[derive(Debug, Clone)]
pub struct Analogy {
    /// The center node ID of the analogous subgraph.
    pub center_id: u64,
    /// Structural similarity score (0.0 to 1.0).
    pub structural_similarity: f32,
    /// Content similarity score (cosine similarity, 0.0 to 1.0).
    pub content_similarity: f32,
    /// Combined score: 0.6 * structural + 0.4 * content.
    pub combined_score: f32,
    /// The structural fingerprint of this analogy.
    pub pattern: PatternMatch,
    /// All node IDs in the analogous subgraph.
    pub subgraph_nodes: Vec<u64>,
}

/// Parameters for an analogical query.
#[derive(Debug, Clone)]
pub struct AnalogicalParams {
    /// The anchor to find analogies for.
    pub anchor: AnalogicalAnchor,
    /// Depth of context to extract around the anchor and candidates.
    pub context_depth: u32,
    /// Maximum number of analogies to return.
    pub max_results: usize,
    /// Minimum combined similarity to include.
    pub min_similarity: f32,
    /// Sessions to exclude from results.
    pub exclude_sessions: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Query 16: Drift Detection
// ---------------------------------------------------------------------------

/// The type of change between consecutive belief snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// The initial version of a belief.
    Initial,
    /// A refinement (similar content, higher confidence).
    Refined,
    /// A correction (explicit Correction event or Supersedes edge).
    Corrected,
    /// A contradiction (Contradicts edge).
    Contradicted,
    /// A reinforcement (new Supports edge or access).
    Reinforced,
}

/// A single snapshot of a belief at a point in time.
#[derive(Debug, Clone)]
pub struct BeliefSnapshot {
    /// The node ID.
    pub node_id: u64,
    /// Session in which this snapshot was created.
    pub session_id: u32,
    /// Timestamp of creation.
    pub created_at: u64,
    /// Confidence at this point.
    pub confidence: f32,
    /// Content summary (first 120 chars).
    pub content_preview: String,
    /// How this snapshot changed from the previous one.
    pub change_type: ChangeType,
}

/// A timeline tracking how a belief evolved over time.
#[derive(Debug, Clone)]
pub struct BeliefTimeline {
    /// Ordered snapshots from earliest to latest.
    pub snapshots: Vec<BeliefSnapshot>,
    /// Total number of changes.
    pub change_count: usize,
    /// Number of corrections in the timeline.
    pub correction_count: usize,
    /// Number of contradictions in the timeline.
    pub contradiction_count: usize,
}

/// Parameters for drift detection.
#[derive(Debug, Clone)]
pub struct DriftParams {
    /// Topic to track drift for (will be tokenized for term matching).
    pub topic: String,
    /// Optional feature vector for the topic (for similarity search).
    pub topic_vec: Option<Vec<f32>>,
    /// Maximum number of timelines to return.
    pub max_results: usize,
    /// Minimum relevance score to consider a node part of the topic.
    pub min_relevance: f32,
}

/// Full drift detection report.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Timelines showing belief evolution.
    pub timelines: Vec<BeliefTimeline>,
    /// Overall stability score (0.0 = highly unstable, 1.0 = perfectly stable).
    pub stability: f32,
    /// Whether the topic is likely to change again based on historical patterns.
    pub likely_to_change: bool,
}

// ---------------------------------------------------------------------------
// Negation words used by belief revision to detect contradictions in text.
// ---------------------------------------------------------------------------

const NEGATION_WORDS: &[&str] = &[
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nothing",
    "nowhere",
    "nobody",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
    "false",
    "incorrect",
    "wrong",
    "invalid",
    "untrue",
    "deny",
    "denied",
    "disagree",
    "unlike",
    "opposite",
    "contrary",
    "instead",
    "rather",
];

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl super::query::QueryEngine {
    // -----------------------------------------------------------------------
    // Query 12: Belief Revision
    // -----------------------------------------------------------------------

    /// Analyze how a hypothesis would affect existing beliefs.
    ///
    /// This is a **read-only** operation: it does not modify the graph.
    /// It finds contradictions via text overlap + vector similarity, checks
    /// for negation words, and propagates weakening through CausedBy / Supports
    /// edges.
    pub fn belief_revision(
        &self,
        graph: &MemoryGraph,
        params: BeliefRevisionParams,
    ) -> AmemResult<RevisionReport> {
        let tokenizer = Tokenizer::new();
        let hypothesis_terms: HashSet<String> =
            tokenizer.tokenize(&params.hypothesis).into_iter().collect();

        if hypothesis_terms.is_empty() && params.hypothesis_vec.is_none() {
            return Ok(RevisionReport {
                contradicted: Vec::new(),
                weakened: Vec::new(),
                invalidated_decisions: Vec::new(),
                total_affected: 0,
                cascade: Vec::new(),
            });
        }

        // Build a set of negation words for quick lookup.
        let negation_set: HashSet<&str> = NEGATION_WORDS.iter().copied().collect();

        // Phase 1: Find contradictions.
        // A node is a candidate contradiction if:
        //   (a) it has a Contradicts edge to/from a node that is textually/vectorially
        //       similar to the hypothesis, OR
        //   (b) the node itself is textually similar to the hypothesis AND contains
        //       negation words relative to the hypothesis.
        let mut contradicted: Vec<ContradictedNode> = Vec::new();
        let mut contradicted_ids: HashSet<u64> = HashSet::new();

        for node in graph.nodes() {
            let node_terms: HashSet<String> =
                tokenizer.tokenize(&node.content).into_iter().collect();

            // --- Text overlap (BM25-like term overlap) ---
            let overlap_count = hypothesis_terms.intersection(&node_terms).count();
            let text_sim = if hypothesis_terms.is_empty() {
                0.0
            } else {
                overlap_count as f32 / hypothesis_terms.len() as f32
            };

            // --- Vector similarity ---
            let vec_sim = if let Some(ref hvec) = params.hypothesis_vec {
                if !node.feature_vec.iter().all(|&x| x == 0.0) {
                    cosine_similarity(hvec, &node.feature_vec)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Combined relevance
            let relevance = if params.hypothesis_vec.is_some() {
                0.5 * text_sim + 0.5 * vec_sim
            } else {
                text_sim
            };

            if relevance < params.contradiction_threshold {
                continue;
            }

            // Check for negation: does the node content contain negation words
            // relative to the overlapping terms?
            let node_content_lower = node.content.to_lowercase();
            let has_negation = negation_set
                .iter()
                .any(|neg| node_content_lower.contains(neg));

            // Check for explicit Contradicts edges involving this node.
            let has_contradicts_edge = graph
                .edges_from(node.id)
                .iter()
                .any(|e| e.edge_type == EdgeType::Contradicts)
                || graph
                    .edges_to(node.id)
                    .iter()
                    .any(|e| e.edge_type == EdgeType::Contradicts);

            // Also check if this node is a Correction event type.
            let is_correction = node.event_type == EventType::Correction;

            if has_negation || has_contradicts_edge || is_correction {
                let strength = relevance
                    * if has_contradicts_edge { 1.0 } else { 0.8 }
                    * if has_negation { 1.0 } else { 0.7 };

                let reason = if has_contradicts_edge {
                    "explicit Contradicts edge in graph".to_string()
                } else if has_negation {
                    "negation detected in content".to_string()
                } else {
                    "correction event with high similarity".to_string()
                };

                contradicted_ids.insert(node.id);
                contradicted.push(ContradictedNode {
                    node_id: node.id,
                    contradiction_strength: strength.clamp(0.0, 1.0),
                    reason,
                });
            }
        }

        // Sort contradictions by strength descending.
        contradicted.sort_by(|a, b| {
            b.contradiction_strength
                .partial_cmp(&a.contradiction_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Phase 2: Cascade — propagate weakening through CausedBy and Supports edges.
        let mut weakened: Vec<WeakenedNode> = Vec::new();
        let mut cascade: Vec<CascadeStep> = Vec::new();
        let mut visited: HashSet<u64> = contradicted_ids.clone();
        let mut queue: VecDeque<(u64, u32, f32)> = VecDeque::new();

        // Seed the queue with contradicted nodes.
        for cn in &contradicted {
            queue.push_back((cn.node_id, 0, cn.contradiction_strength));
        }

        while let Some((current_id, depth, weakening_factor)) = queue.pop_front() {
            if depth >= params.max_depth {
                continue;
            }

            // Find nodes that depend on current_id via CausedBy or Supports edges.
            // A node N depends on current_id if N has an outgoing CausedBy/Supports
            // edge pointing to current_id, meaning "N was caused by current_id" or
            // "N is supported by current_id".
            for edge in graph.edges_to(current_id) {
                if edge.edge_type != EdgeType::CausedBy && edge.edge_type != EdgeType::Supports {
                    continue;
                }
                let dependent_id = edge.source_id;
                if visited.contains(&dependent_id) {
                    continue;
                }
                visited.insert(dependent_id);

                if let Some(dep_node) = graph.get_node(dependent_id) {
                    // Weakening decays with depth: factor * edge_weight * 0.7^depth
                    let decay = 0.7f32.powi(depth as i32 + 1);
                    let effective_weakening = weakening_factor * edge.weight * decay;
                    let revised = (dep_node.confidence - effective_weakening).clamp(0.0, 1.0);

                    weakened.push(WeakenedNode {
                        node_id: dependent_id,
                        original_confidence: dep_node.confidence,
                        revised_confidence: revised,
                        depth: depth + 1,
                    });

                    cascade.push(CascadeStep {
                        node_id: dependent_id,
                        via_edge: edge.edge_type,
                        from_node: current_id,
                        depth: depth + 1,
                    });

                    queue.push_back((dependent_id, depth + 1, effective_weakening));
                }
            }
        }

        // Phase 3: Identify invalidated decisions.
        let mut invalidated_decisions: Vec<u64> = Vec::new();
        let affected_ids: HashSet<u64> = contradicted_ids
            .iter()
            .chain(weakened.iter().map(|w| &w.node_id))
            .copied()
            .collect();

        for &node_id in &affected_ids {
            if let Some(node) = graph.get_node(node_id) {
                if node.event_type == EventType::Decision {
                    invalidated_decisions.push(node_id);
                }
            }
        }
        invalidated_decisions.sort_unstable();
        invalidated_decisions.dedup();

        let total_affected = affected_ids.len();

        Ok(RevisionReport {
            contradicted,
            weakened,
            invalidated_decisions,
            total_affected,
            cascade,
        })
    }

    // -----------------------------------------------------------------------
    // Query 13: Gap Detection
    // -----------------------------------------------------------------------

    /// Detect gaps in the knowledge graph: unjustified decisions, single-source
    /// inferences, low-confidence foundations, unstable knowledge, and stale evidence.
    ///
    /// This is a **read-only** operation.
    pub fn gap_detection(
        &self,
        graph: &MemoryGraph,
        params: GapDetectionParams,
    ) -> AmemResult<GapReport> {
        let session_filter: Option<(u32, u32)> = params.session_range;
        let mut gaps: Vec<Gap> = Vec::new();

        for node in graph.nodes() {
            // Apply session range filter.
            if let Some((start, end)) = session_filter {
                if node.session_id < start || node.session_id > end {
                    continue;
                }
            }

            // --- Unjustified decisions ---
            if node.event_type == EventType::Decision {
                let incoming = graph.edges_to(node.id);
                let has_justification = incoming.iter().any(|e| {
                    e.edge_type == EdgeType::CausedBy || e.edge_type == EdgeType::Supports
                });
                if !has_justification {
                    let downstream = self.count_downstream(graph, node.id);
                    gaps.push(Gap {
                        node_id: node.id,
                        gap_type: GapType::UnjustifiedDecision,
                        severity: 0.9, // High severity: decisions need justification
                        description: format!(
                            "Decision node {} has no CausedBy or Supports edges",
                            node.id
                        ),
                        downstream_count: downstream,
                    });
                }
            }

            // --- Single-source inferences ---
            if node.event_type == EventType::Inference {
                let incoming = graph.edges_to(node.id);
                let support_count = incoming
                    .iter()
                    .filter(|e| e.edge_type == EdgeType::Supports)
                    .count();
                if (support_count as u32) < params.min_support_count {
                    let downstream = self.count_downstream(graph, node.id);
                    gaps.push(Gap {
                        node_id: node.id,
                        gap_type: GapType::SingleSourceInference,
                        severity: 0.7,
                        description: format!(
                            "Inference node {} has only {} Supports edge(s), needs at least {}",
                            node.id, support_count, params.min_support_count
                        ),
                        downstream_count: downstream,
                    });
                }
            }

            // --- Low-confidence foundations ---
            if (node.event_type == EventType::Fact || node.event_type == EventType::Inference)
                && node.confidence < params.confidence_threshold
            {
                // Only flag if other nodes depend on this one.
                let dependents = graph.edges_to(node.id);
                let has_dependents = dependents.iter().any(|e| {
                    e.edge_type == EdgeType::CausedBy || e.edge_type == EdgeType::Supports
                });
                if has_dependents {
                    let downstream = self.count_downstream(graph, node.id);
                    gaps.push(Gap {
                        node_id: node.id,
                        gap_type: GapType::LowConfidenceFoundation,
                        severity: 1.0 - node.confidence, // Lower confidence = higher severity
                        description: format!(
                            "Node {} has confidence {:.2} (below {:.2}) and is depended upon",
                            node.id, node.confidence, params.confidence_threshold
                        ),
                        downstream_count: downstream,
                    });
                }
            }

            // --- Unstable knowledge (SUPERSEDES chains >= 3) ---
            {
                let supersedes_count = self.count_supersedes_chain(graph, node.id);
                if supersedes_count >= 3 {
                    let downstream = self.count_downstream(graph, node.id);
                    gaps.push(Gap {
                        node_id: node.id,
                        gap_type: GapType::UnstableKnowledge,
                        severity: (supersedes_count as f32 / 5.0).clamp(0.0, 1.0),
                        description: format!(
                            "Node {} has been superseded {} times (unstable)",
                            node.id, supersedes_count
                        ),
                        downstream_count: downstream,
                    });
                }
            }

            // --- Stale evidence ---
            if node.decay_score < 0.2 && node.event_type == EventType::Fact {
                let has_dependents = graph.edges_to(node.id).iter().any(|e| {
                    e.edge_type == EdgeType::CausedBy || e.edge_type == EdgeType::Supports
                });
                if has_dependents {
                    let downstream = self.count_downstream(graph, node.id);
                    gaps.push(Gap {
                        node_id: node.id,
                        gap_type: GapType::StaleEvidence,
                        severity: 1.0 - node.decay_score,
                        description: format!(
                            "Fact node {} has decay score {:.2} and is depended upon",
                            node.id, node.decay_score
                        ),
                        downstream_count: downstream,
                    });
                }
            }
        }

        // Sort gaps.
        match params.sort_by {
            GapSeverity::HighestImpact => {
                gaps.sort_by(|a, b| b.downstream_count.cmp(&a.downstream_count));
            }
            GapSeverity::LowestConfidence => {
                gaps.sort_by(|a, b| {
                    b.severity
                        .partial_cmp(&a.severity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            GapSeverity::MostRecent => {
                gaps.sort_by(|a, b| {
                    let ts_a = graph.get_node(a.node_id).map(|n| n.created_at).unwrap_or(0);
                    let ts_b = graph.get_node(b.node_id).map(|n| n.created_at).unwrap_or(0);
                    ts_b.cmp(&ts_a)
                });
            }
        }

        // Build summary before truncating.
        let total_gaps = gaps.len();
        let unjustified_decisions = gaps
            .iter()
            .filter(|g| g.gap_type == GapType::UnjustifiedDecision)
            .count();
        let single_source_inferences = gaps
            .iter()
            .filter(|g| g.gap_type == GapType::SingleSourceInference)
            .count();
        let low_confidence_foundations = gaps
            .iter()
            .filter(|g| g.gap_type == GapType::LowConfidenceFoundation)
            .count();
        let unstable_knowledge = gaps
            .iter()
            .filter(|g| g.gap_type == GapType::UnstableKnowledge)
            .count();
        let stale_evidence = gaps
            .iter()
            .filter(|g| g.gap_type == GapType::StaleEvidence)
            .count();

        let total_nodes = graph.node_count();
        let health_score = if total_nodes > 0 {
            1.0 - (total_gaps as f32 / total_nodes as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let summary = GapSummary {
            total_gaps,
            unjustified_decisions,
            single_source_inferences,
            low_confidence_foundations,
            unstable_knowledge,
            stale_evidence,
            health_score,
        };

        gaps.truncate(params.max_results);

        Ok(GapReport { gaps, summary })
    }

    /// Count how many nodes transitively depend on a given node via CausedBy/Supports.
    fn count_downstream(&self, graph: &MemoryGraph, node_id: u64) -> usize {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<u64> = VecDeque::new();
        visited.insert(node_id);
        queue.push_back(node_id);

        while let Some(current) = queue.pop_front() {
            for edge in graph.edges_to(current) {
                if (edge.edge_type == EdgeType::CausedBy || edge.edge_type == EdgeType::Supports)
                    && !visited.contains(&edge.source_id)
                {
                    visited.insert(edge.source_id);
                    queue.push_back(edge.source_id);
                }
            }
        }

        // Subtract 1 because we don't count the node itself.
        visited.len().saturating_sub(1)
    }

    /// Count the length of a SUPERSEDES chain ending at (or passing through) a node.
    /// Follows Supersedes edges backwards: find nodes that supersede this one, then
    /// follow the chain.
    fn count_supersedes_chain(&self, graph: &MemoryGraph, node_id: u64) -> usize {
        let mut count = 0usize;
        let mut current = node_id;
        let mut visited: HashSet<u64> = HashSet::new();
        visited.insert(current);

        // Walk backwards: find who supersedes this node.
        loop {
            let mut found = false;
            for edge in graph.edges_to(current) {
                if edge.edge_type == EdgeType::Supersedes && !visited.contains(&edge.source_id) {
                    visited.insert(edge.source_id);
                    current = edge.source_id;
                    count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        // Also walk forwards: find what this node supersedes.
        current = node_id;
        loop {
            let mut found = false;
            for edge in graph.edges_from(current) {
                if edge.edge_type == EdgeType::Supersedes && !visited.contains(&edge.target_id) {
                    visited.insert(edge.target_id);
                    current = edge.target_id;
                    count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        count
    }

    // -----------------------------------------------------------------------
    // Query 14: Analogical Query
    // -----------------------------------------------------------------------

    /// Find subgraphs that are structurally and semantically analogous to the
    /// anchor.
    ///
    /// Uses **structural fingerprinting** (node type counts, edge type counts,
    /// causal chain depth, branching factor) rather than full subgraph
    /// isomorphism.
    ///
    /// `combined_score = 0.6 * structural + 0.4 * content`
    ///
    /// This is a **read-only** operation.
    pub fn analogical(
        &self,
        graph: &MemoryGraph,
        params: AnalogicalParams,
    ) -> AmemResult<Vec<Analogy>> {
        let exclude_sessions: HashSet<u32> = params.exclude_sessions.iter().copied().collect();

        // Step 1: Determine the anchor subgraph and its fingerprint.
        let (anchor_center, anchor_vec) = match &params.anchor {
            AnalogicalAnchor::Node(id) => {
                let node = graph
                    .get_node(*id)
                    .ok_or(crate::types::AmemError::NodeNotFound(*id))?;
                (*id, node.feature_vec.clone())
            }
            AnalogicalAnchor::Vector(v) => {
                // Find the most similar node to use as anchor center.
                let mut best_id = 0u64;
                let mut best_sim = -1.0f32;
                for node in graph.nodes() {
                    if node.feature_vec.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let sim = cosine_similarity(v, &node.feature_vec);
                    if sim > best_sim {
                        best_sim = sim;
                        best_id = node.id;
                    }
                }
                if best_sim < 0.0 {
                    return Ok(Vec::new());
                }
                (best_id, v.clone())
            }
        };

        let anchor_subgraph = self.context(graph, anchor_center, params.context_depth)?;
        let anchor_fp =
            self.structural_fingerprint(graph, &anchor_subgraph.nodes, &anchor_subgraph.edges);
        let anchor_session = graph
            .get_node(anchor_center)
            .map(|n| n.session_id)
            .unwrap_or(0);

        // Step 2: For every other node, extract context and compare fingerprints.
        let mut analogies: Vec<Analogy> = Vec::new();
        let anchor_node_set: HashSet<u64> = anchor_subgraph.nodes.iter().map(|n| n.id).collect();

        for node in graph.nodes() {
            // Skip nodes in the anchor subgraph itself.
            if anchor_node_set.contains(&node.id) {
                continue;
            }
            // Skip excluded sessions.
            if exclude_sessions.contains(&node.session_id) {
                continue;
            }
            // Skip nodes in the same session as the anchor (unless there's only one session).
            if node.session_id == anchor_session
                && graph.nodes().len() > anchor_subgraph.nodes.len()
            {
                continue;
            }

            // Extract candidate subgraph.
            let candidate_subgraph = match self.context(graph, node.id, params.context_depth) {
                Ok(sg) => sg,
                Err(_) => continue,
            };

            // Structural fingerprint comparison.
            let candidate_fp = self.structural_fingerprint(
                graph,
                &candidate_subgraph.nodes,
                &candidate_subgraph.edges,
            );
            let structural_sim = self.compare_fingerprints(&anchor_fp, &candidate_fp);

            // Content similarity: average cosine similarity of center nodes.
            let content_sim = if !anchor_vec.iter().all(|&x| x == 0.0)
                && !node.feature_vec.iter().all(|&x| x == 0.0)
            {
                cosine_similarity(&anchor_vec, &node.feature_vec).max(0.0)
            } else {
                0.0
            };

            let combined = 0.6 * structural_sim + 0.4 * content_sim;

            if combined >= params.min_similarity {
                analogies.push(Analogy {
                    center_id: node.id,
                    structural_similarity: structural_sim,
                    content_similarity: content_sim,
                    combined_score: combined,
                    pattern: candidate_fp,
                    subgraph_nodes: candidate_subgraph.nodes.iter().map(|n| n.id).collect(),
                });
            }
        }

        // Sort by combined score descending.
        analogies.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        analogies.truncate(params.max_results);

        Ok(analogies)
    }

    /// Build a structural fingerprint for a set of nodes and edges.
    fn structural_fingerprint(
        &self,
        _graph: &MemoryGraph,
        nodes: &[CognitiveEvent],
        edges: &[Edge],
    ) -> PatternMatch {
        // Event type counts.
        let mut event_type_counts: HashMap<u8, usize> = HashMap::new();
        for node in nodes {
            *event_type_counts.entry(node.event_type as u8).or_insert(0) += 1;
        }

        // Edge type counts.
        let mut edge_type_counts: HashMap<u8, usize> = HashMap::new();
        for edge in edges {
            *edge_type_counts.entry(edge.edge_type as u8).or_insert(0) += 1;
        }

        // Causal chain depth: longest chain of CausedBy edges.
        let node_set: HashSet<u64> = nodes.iter().map(|n| n.id).collect();
        let causal_edges: Vec<&Edge> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::CausedBy)
            .collect();

        let causal_chain_depth = if causal_edges.is_empty() {
            0
        } else {
            // Build adjacency for causal edges within this subgraph.
            let mut causal_adj: HashMap<u64, Vec<u64>> = HashMap::new();
            for e in &causal_edges {
                if node_set.contains(&e.source_id) && node_set.contains(&e.target_id) {
                    causal_adj.entry(e.source_id).or_default().push(e.target_id);
                }
            }
            // BFS from each node to find max depth.
            let mut max_depth = 0u32;
            for &start_id in node_set.iter() {
                let mut visited_local: HashSet<u64> = HashSet::new();
                let mut q: VecDeque<(u64, u32)> = VecDeque::new();
                visited_local.insert(start_id);
                q.push_back((start_id, 0));
                while let Some((cur, d)) = q.pop_front() {
                    max_depth = max_depth.max(d);
                    if let Some(neighbors) = causal_adj.get(&cur) {
                        for &nb in neighbors {
                            if !visited_local.contains(&nb) {
                                visited_local.insert(nb);
                                q.push_back((nb, d + 1));
                            }
                        }
                    }
                }
            }
            max_depth
        };

        // Branching factor: average outgoing edges per node (within the subgraph).
        let branching_factor = if nodes.is_empty() {
            0.0
        } else {
            let mut out_counts: HashMap<u64, usize> = HashMap::new();
            for n in nodes {
                out_counts.insert(n.id, 0);
            }
            for e in edges {
                if let Some(c) = out_counts.get_mut(&e.source_id) {
                    *c += 1;
                }
            }
            let total: usize = out_counts.values().sum();
            total as f32 / nodes.len() as f32
        };

        PatternMatch {
            event_type_counts,
            edge_type_counts,
            causal_chain_depth,
            branching_factor,
        }
    }

    /// Compare two structural fingerprints and return a similarity in [0.0, 1.0].
    fn compare_fingerprints(&self, a: &PatternMatch, b: &PatternMatch) -> f32 {
        // 1. Event type distribution similarity (cosine on count vectors).
        let et_sim = self.map_cosine_similarity(&a.event_type_counts, &b.event_type_counts);

        // 2. Edge type distribution similarity (cosine on count vectors).
        let edge_sim = self.map_cosine_similarity(&a.edge_type_counts, &b.edge_type_counts);

        // 3. Causal chain depth similarity.
        let max_chain = a.causal_chain_depth.max(b.causal_chain_depth).max(1) as f32;
        let chain_sim =
            1.0 - (a.causal_chain_depth as f32 - b.causal_chain_depth as f32).abs() / max_chain;

        // 4. Branching factor similarity.
        let max_bf = a.branching_factor.max(b.branching_factor).max(0.01);
        let bf_sim = 1.0 - (a.branching_factor - b.branching_factor).abs() / max_bf;

        // Weighted combination.
        0.3 * et_sim + 0.3 * edge_sim + 0.2 * chain_sim + 0.2 * bf_sim
    }

    /// Cosine similarity on two sparse count maps.
    fn map_cosine_similarity(&self, a: &HashMap<u8, usize>, b: &HashMap<u8, usize>) -> f32 {
        let all_keys: HashSet<u8> = a.keys().chain(b.keys()).copied().collect();
        if all_keys.is_empty() {
            return 1.0; // Both empty = identical.
        }

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for &key in &all_keys {
            let va = *a.get(&key).unwrap_or(&0) as f64;
            let vb = *b.get(&key).unwrap_or(&0) as f64;
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (dot / denom) as f32
        }
    }

    // -----------------------------------------------------------------------
    // Query 16: Drift Detection
    // -----------------------------------------------------------------------

    /// Track how beliefs on a given topic have changed over time.
    ///
    /// Finds relevant nodes, groups them into SUPERSEDES chains and clusters
    /// of similar content across sessions, and reports stability.
    ///
    /// This is a **read-only** operation.
    pub fn drift_detection(
        &self,
        graph: &MemoryGraph,
        params: DriftParams,
    ) -> AmemResult<DriftReport> {
        let tokenizer = Tokenizer::new();
        let topic_terms: HashSet<String> = tokenizer.tokenize(&params.topic).into_iter().collect();

        if topic_terms.is_empty() && params.topic_vec.is_none() {
            return Ok(DriftReport {
                timelines: Vec::new(),
                stability: 1.0,
                likely_to_change: false,
            });
        }

        // Phase 1: Find nodes relevant to the topic.
        let mut relevant: Vec<(u64, f32)> = Vec::new(); // (node_id, relevance_score)

        for node in graph.nodes() {
            let node_terms: HashSet<String> =
                tokenizer.tokenize(&node.content).into_iter().collect();

            // Text overlap.
            let overlap = topic_terms.intersection(&node_terms).count();
            let text_sim = if topic_terms.is_empty() {
                0.0
            } else {
                overlap as f32 / topic_terms.len() as f32
            };

            // Vector similarity.
            let vec_sim = if let Some(ref tvec) = params.topic_vec {
                if !node.feature_vec.iter().all(|&x| x == 0.0) {
                    cosine_similarity(tvec, &node.feature_vec).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let relevance = if params.topic_vec.is_some() {
                0.5 * text_sim + 0.5 * vec_sim
            } else {
                text_sim
            };

            if relevance >= params.min_relevance {
                relevant.push((node.id, relevance));
            }
        }

        if relevant.is_empty() {
            return Ok(DriftReport {
                timelines: Vec::new(),
                stability: 1.0,
                likely_to_change: false,
            });
        }

        // Sort by relevance descending, then take top candidates.
        relevant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let relevant_ids: HashSet<u64> = relevant.iter().map(|(id, _)| *id).collect();

        // Phase 2: Build SUPERSEDES chains among relevant nodes.
        // Group into chains: each chain is a timeline.
        let mut chain_roots: Vec<u64> = Vec::new();
        let mut assigned: HashSet<u64> = HashSet::new();

        // Find chain roots: relevant nodes that are NOT superseded by another relevant node.
        for &(node_id, _) in &relevant {
            // Check if this node supersedes someone (it has outgoing Supersedes edges).
            let _supersedes_another = graph.edges_from(node_id).iter().any(|e| {
                e.edge_type == EdgeType::Supersedes && relevant_ids.contains(&e.target_id)
            });

            // Check if someone supersedes this node.
            let is_superseded = graph.edges_to(node_id).iter().any(|e| {
                e.edge_type == EdgeType::Supersedes && relevant_ids.contains(&e.source_id)
            });

            // A root is a node that is NOT superseded by another relevant node,
            // OR a node that supersedes others (the latest in a chain).
            if !is_superseded {
                chain_roots.push(node_id);
            }
        }

        // If no chain roots found (e.g., circular supersedes), use the most recent nodes.
        if chain_roots.is_empty() {
            // Fall back to top relevant nodes.
            for &(node_id, _) in relevant.iter().take(params.max_results) {
                chain_roots.push(node_id);
            }
        }

        let mut timelines: Vec<BeliefTimeline> = Vec::new();

        for &root_id in &chain_roots {
            if assigned.contains(&root_id) {
                continue;
            }

            let mut chain: Vec<u64> = Vec::new();
            let mut current = root_id;
            let mut chain_visited: HashSet<u64> = HashSet::new();

            // Walk backwards through the Supersedes chain: root -> what it supersedes -> ...
            // The root is the latest version; follow Supersedes edges forward (source supersedes target).
            chain_visited.insert(current);
            chain.push(current);
            assigned.insert(current);

            loop {
                let mut next = None;
                for edge in graph.edges_from(current) {
                    if edge.edge_type == EdgeType::Supersedes
                        && !chain_visited.contains(&edge.target_id)
                    {
                        next = Some(edge.target_id);
                        break;
                    }
                }
                match next {
                    Some(next_id) => {
                        chain_visited.insert(next_id);
                        chain.push(next_id);
                        assigned.insert(next_id);
                        current = next_id;
                    }
                    None => break,
                }
            }

            // Also check if anyone supersedes the root (walk backwards).
            current = root_id;
            loop {
                let mut prev = None;
                for edge in graph.edges_to(current) {
                    if edge.edge_type == EdgeType::Supersedes
                        && !chain_visited.contains(&edge.source_id)
                    {
                        prev = Some(edge.source_id);
                        break;
                    }
                }
                match prev {
                    Some(prev_id) => {
                        chain_visited.insert(prev_id);
                        chain.insert(0, prev_id);
                        assigned.insert(prev_id);
                        current = prev_id;
                    }
                    None => break,
                }
            }

            // Reverse: we want chronological order (oldest first).
            // The chain currently has latest first (root) then older.
            // Actually let's sort by created_at.
            chain.sort_by_key(|&id| graph.get_node(id).map(|n| n.created_at).unwrap_or(0));

            // Build snapshots.
            let mut snapshots: Vec<BeliefSnapshot> = Vec::new();
            let mut correction_count = 0usize;
            let mut contradiction_count = 0usize;

            for (i, &nid) in chain.iter().enumerate() {
                if let Some(node) = graph.get_node(nid) {
                    let change_type =
                        if i == 0 {
                            ChangeType::Initial
                        } else {
                            let prev_id = chain[i - 1];
                            // Check edge types between this node and the previous.
                            let has_supersedes = graph.edges_from(nid).iter().any(|e| {
                                e.edge_type == EdgeType::Supersedes && e.target_id == prev_id
                            });
                            let has_contradicts = graph.edges_from(nid).iter().any(|e| {
                                e.edge_type == EdgeType::Contradicts && e.target_id == prev_id
                            }) || graph.edges_to(nid).iter().any(|e| {
                                e.edge_type == EdgeType::Contradicts && e.source_id == prev_id
                            });
                            let has_supports = graph.edges_from(nid).iter().any(|e| {
                                e.edge_type == EdgeType::Supports && e.target_id == prev_id
                            }) || graph.edges_to(nid).iter().any(|e| {
                                e.edge_type == EdgeType::Supports && e.source_id == prev_id
                            });

                            if has_contradicts {
                                ChangeType::Contradicted
                            } else if node.event_type == EventType::Correction || has_supersedes {
                                ChangeType::Corrected
                            } else if has_supports {
                                ChangeType::Reinforced
                            } else {
                                // Default: if confidence is higher, refined; else corrected.
                                let prev_conf =
                                    graph.get_node(prev_id).map(|n| n.confidence).unwrap_or(0.0);
                                if node.confidence >= prev_conf {
                                    ChangeType::Refined
                                } else {
                                    ChangeType::Corrected
                                }
                            }
                        };

                    match change_type {
                        ChangeType::Corrected => correction_count += 1,
                        ChangeType::Contradicted => contradiction_count += 1,
                        _ => {}
                    }

                    let content_preview = if node.content.len() > 120 {
                        format!("{}...", &node.content[..120])
                    } else {
                        node.content.clone()
                    };

                    snapshots.push(BeliefSnapshot {
                        node_id: nid,
                        session_id: node.session_id,
                        created_at: node.created_at,
                        confidence: node.confidence,
                        content_preview,
                        change_type,
                    });
                }
            }

            if !snapshots.is_empty() {
                let change_count = snapshots.len().saturating_sub(1);
                timelines.push(BeliefTimeline {
                    snapshots,
                    change_count,
                    correction_count,
                    contradiction_count,
                });
            }
        }

        // Also add isolated relevant nodes that were not part of any chain.
        for &(node_id, _) in &relevant {
            if assigned.contains(&node_id) {
                continue;
            }
            assigned.insert(node_id);

            if let Some(node) = graph.get_node(node_id) {
                let content_preview = if node.content.len() > 120 {
                    format!("{}...", &node.content[..120])
                } else {
                    node.content.clone()
                };

                timelines.push(BeliefTimeline {
                    snapshots: vec![BeliefSnapshot {
                        node_id,
                        session_id: node.session_id,
                        created_at: node.created_at,
                        confidence: node.confidence,
                        content_preview,
                        change_type: ChangeType::Initial,
                    }],
                    change_count: 0,
                    correction_count: 0,
                    contradiction_count: 0,
                });
            }
        }

        // Sort timelines by number of changes descending (most volatile first).
        timelines.sort_by(|a, b| b.change_count.cmp(&a.change_count));
        timelines.truncate(params.max_results);

        // Compute stability: based on how many corrections/contradictions occurred.
        let total_changes: usize = timelines.iter().map(|t| t.change_count).sum();
        let total_corrections: usize = timelines.iter().map(|t| t.correction_count).sum();
        let total_contradictions: usize = timelines.iter().map(|t| t.contradiction_count).sum();
        let total_snapshots: usize = timelines.iter().map(|t| t.snapshots.len()).sum();

        let stability = if total_snapshots <= 1 {
            1.0
        } else {
            let volatility =
                (total_corrections + total_contradictions) as f32 / total_snapshots as f32;
            (1.0 - volatility).clamp(0.0, 1.0)
        };

        // Likely to change: if there have been recent corrections/contradictions
        // (more than 30% of changes are corrections/contradictions).
        let likely_to_change = if total_changes == 0 {
            false
        } else {
            let instability_ratio =
                (total_corrections + total_contradictions) as f32 / total_changes as f32;
            instability_ratio > 0.3
        };

        Ok(DriftReport {
            timelines,
            stability,
            likely_to_change,
        })
    }
}
