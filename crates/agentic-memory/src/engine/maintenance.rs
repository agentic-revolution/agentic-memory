//! Memory consolidation — query 15.
//!
//! Provides deduplication, orphan pruning, contradiction linking,
//! episode compression, and inference promotion operations on a
//! [`MemoryGraph`].  This is the only query type that mutates the
//! graph.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::graph::MemoryGraph;
use crate::index::cosine_similarity;
use crate::types::{AmemResult, Edge, EdgeType, EventType};

use super::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single consolidation operation to run.
pub enum ConsolidationOp {
    /// Merge near-duplicate Fact nodes.
    /// `threshold` is the minimum cosine similarity to consider two facts
    /// duplicates (typically 0.90 -- 0.98).
    DeduplicateFacts { threshold: f32 },

    /// Report orphaned nodes that could be pruned.
    /// `max_decay` is the ceiling on `decay_score` for a node to be
    /// considered orphaned (e.g. 0.1).
    ///
    /// **V1: dry-run only** -- the consolidation method will never remove
    /// nodes, only report them.
    PruneOrphans { max_decay: f32 },

    /// Discover contradictory pairs and link them with `Contradicts` edges.
    /// `threshold` is the minimum cosine similarity between two Fact/Inference
    /// nodes for them to be candidates (the method additionally checks for
    /// negation words).
    LinkContradictions { threshold: f32 },

    /// Report groups of Episode nodes that could be compressed.
    /// `group_size` is the minimum number of contiguous episodes to consider
    /// compressible.
    ///
    /// **V1: dry-run only** -- the consolidation method will never compress
    /// episodes, only report them.
    CompressEpisodes { group_size: u32 },

    /// Promote well-established Inference nodes to Fact.
    /// Requires `access_count >= min_access` **and** `confidence >=
    /// min_confidence`.
    PromoteInferences {
        min_access: u32,
        min_confidence: f32,
    },
}

/// Parameters for a consolidation run.
pub struct ConsolidationParams {
    /// If set, only consider nodes whose `session_id` falls in
    /// `[start, end]` (inclusive).
    pub session_range: Option<(u32, u32)>,

    /// The operations to execute, in order.
    pub operations: Vec<ConsolidationOp>,

    /// When `true`, no mutations are applied -- the report describes what
    /// *would* happen.
    pub dry_run: bool,

    /// Optional path for the caller to store a pre-consolidation backup.
    /// The consolidation method itself does **not** write files; it simply
    /// copies this value into the report for the caller to act on.
    pub backup_path: Option<PathBuf>,
}

/// A single action taken (or proposed) during consolidation.
pub struct ConsolidationAction {
    /// Human-readable operation name (e.g. "deduplicate_facts").
    pub operation: String,
    /// Human-readable description of the action.
    pub description: String,
    /// Node IDs affected by this action.
    pub affected_nodes: Vec<u64>,
}

/// Summary report returned after consolidation.
pub struct ConsolidationReport {
    /// Detailed list of every action taken (or proposed).
    pub actions: Vec<ConsolidationAction>,
    /// Number of duplicate pairs resolved.
    pub deduplicated: usize,
    /// Number of orphaned nodes reported (never actually removed in V1).
    pub pruned: usize,
    /// Number of new `Contradicts` edges added (or proposed).
    pub contradictions_linked: usize,
    /// Number of episode groups reported (never actually compressed in V1).
    pub episodes_compressed: usize,
    /// Number of Inference nodes promoted to Fact.
    pub inferences_promoted: usize,
    /// Echoed back from [`ConsolidationParams::backup_path`].
    pub backup_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Negation words used by the contradiction detector.
// ---------------------------------------------------------------------------

const NEGATION_WORDS: &[&str] = &[
    "not",
    "never",
    "no",
    "neither",
    "nor",
    "cannot",
    "can't",
    "won't",
    "doesn't",
    "don't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "shouldn't",
    "wouldn't",
    "couldn't",
    "hardly",
    "barely",
    "false",
    "incorrect",
    "wrong",
    "untrue",
    "impossible",
    "deny",
    "denied",
    "disagree",
    "unlike",
    "opposite",
];

// ---------------------------------------------------------------------------
// Implementation on QueryEngine
// ---------------------------------------------------------------------------

impl super::query::QueryEngine {
    /// Run a set of consolidation operations against `graph`.
    ///
    /// If `params.dry_run` is `true`, the graph is not mutated; the returned
    /// report describes what *would* happen.
    ///
    /// When `dry_run` is `false`:
    /// * `DeduplicateFacts` adds `Supersedes` edges from the surviving node to
    ///   each duplicate.
    /// * `LinkContradictions` adds `Contradicts` edges.
    /// * `PromoteInferences` changes `event_type` from `Inference` to `Fact`.
    /// * `PruneOrphans` and `CompressEpisodes` are always dry-run-only in V1.
    pub fn consolidate(
        &self,
        graph: &mut MemoryGraph,
        params: ConsolidationParams,
    ) -> AmemResult<ConsolidationReport> {
        let mut report = ConsolidationReport {
            actions: Vec::new(),
            deduplicated: 0,
            pruned: 0,
            contradictions_linked: 0,
            episodes_compressed: 0,
            inferences_promoted: 0,
            backup_path: params.backup_path.clone(),
        };

        // Pre-compute the set of in-scope node IDs when a session range is
        // specified.
        let session_filter: Option<(u32, u32)> = params.session_range;

        for op in &params.operations {
            match op {
                ConsolidationOp::DeduplicateFacts { threshold } => {
                    self.op_deduplicate_facts(
                        graph,
                        *threshold,
                        session_filter,
                        params.dry_run,
                        &mut report,
                    );
                }
                ConsolidationOp::PruneOrphans { max_decay } => {
                    // Always dry-run in V1.
                    self.op_prune_orphans(graph, *max_decay, session_filter, &mut report);
                }
                ConsolidationOp::LinkContradictions { threshold } => {
                    self.op_link_contradictions(
                        graph,
                        *threshold,
                        session_filter,
                        params.dry_run,
                        &mut report,
                    );
                }
                ConsolidationOp::CompressEpisodes { group_size } => {
                    // Always dry-run in V1.
                    self.op_compress_episodes(graph, *group_size, session_filter, &mut report);
                }
                ConsolidationOp::PromoteInferences {
                    min_access,
                    min_confidence,
                } => {
                    self.op_promote_inferences(
                        graph,
                        *min_access,
                        *min_confidence,
                        session_filter,
                        params.dry_run,
                        &mut report,
                    );
                }
            }
        }

        Ok(report)
    }

    // -----------------------------------------------------------------------
    // DeduplicateFacts
    // -----------------------------------------------------------------------

    fn op_deduplicate_facts(
        &self,
        graph: &mut MemoryGraph,
        threshold: f32,
        session_filter: Option<(u32, u32)>,
        dry_run: bool,
        report: &mut ConsolidationReport,
    ) {
        let tokenizer = Tokenizer::new();

        // Collect Fact node IDs, respecting the session filter.
        let fact_ids: Vec<u64> = graph
            .nodes()
            .iter()
            .filter(|n| {
                n.event_type == EventType::Fact && in_session_range(n.session_id, session_filter)
            })
            .map(|n| n.id)
            .collect();

        // Group facts by cluster so we only compare within-cluster pairs.
        let cluster_count = graph.cluster_map().cluster_count();
        let fact_set: HashSet<u64> = fact_ids.iter().copied().collect();

        // Build cluster -> [fact ids in that cluster].
        let mut cluster_groups: Vec<Vec<u64>> = Vec::new();
        if cluster_count > 0 {
            for ci in 0..cluster_count {
                let members: Vec<u64> = graph
                    .cluster_map()
                    .get_cluster(ci)
                    .iter()
                    .copied()
                    .filter(|id| fact_set.contains(id))
                    .collect();
                if members.len() >= 2 {
                    cluster_groups.push(members);
                }
            }
        }

        // Fallback: if no clusters, treat all facts as one group.
        if cluster_groups.is_empty() && fact_ids.len() >= 2 {
            cluster_groups.push(fact_ids.clone());
        }

        // Track which nodes have already been marked as duplicates so we
        // don't supersede the same node twice.
        let mut superseded: HashSet<u64> = HashSet::new();

        for group in &cluster_groups {
            for i in 0..group.len() {
                if superseded.contains(&group[i]) {
                    continue;
                }
                for j in (i + 1)..group.len() {
                    if superseded.contains(&group[j]) {
                        continue;
                    }

                    // Borrow two separate snapshots so we don't alias &graph.
                    let (vec_a, conf_a, content_a) = match graph.get_node(group[i]) {
                        Some(n) => (n.feature_vec.clone(), n.confidence, n.content.clone()),
                        None => continue,
                    };
                    let (vec_b, conf_b, content_b) = match graph.get_node(group[j]) {
                        Some(n) => (n.feature_vec.clone(), n.confidence, n.content.clone()),
                        None => continue,
                    };

                    let sim = cosine_similarity(&vec_a, &vec_b);
                    if sim < threshold {
                        continue;
                    }

                    // Also require high token-level overlap.
                    let tokens_a: HashSet<String> =
                        tokenizer.tokenize(&content_a).into_iter().collect();
                    let tokens_b: HashSet<String> =
                        tokenizer.tokenize(&content_b).into_iter().collect();

                    if tokens_a.is_empty() && tokens_b.is_empty() {
                        continue;
                    }

                    let intersection = tokens_a.intersection(&tokens_b).count();
                    let union = tokens_a.union(&tokens_b).count();
                    let jaccard = if union > 0 {
                        intersection as f32 / union as f32
                    } else {
                        0.0
                    };

                    if jaccard < 0.5 {
                        continue;
                    }

                    // Determine winner (higher confidence survives).
                    let (winner, loser) = if conf_a >= conf_b {
                        (group[i], group[j])
                    } else {
                        (group[j], group[i])
                    };

                    superseded.insert(loser);

                    report.actions.push(ConsolidationAction {
                        operation: "deduplicate_facts".to_string(),
                        description: format!(
                            "Node {} supersedes duplicate node {} (cosine={:.3}, jaccard={:.3})",
                            winner, loser, sim, jaccard,
                        ),
                        affected_nodes: vec![winner, loser],
                    });
                    report.deduplicated += 1;

                    if !dry_run {
                        let edge = Edge {
                            source_id: winner,
                            target_id: loser,
                            edge_type: EdgeType::Supersedes,
                            weight: sim,
                            created_at: crate::types::now_micros(),
                        };
                        // Ignore error if the edge cannot be added (e.g.
                        // duplicate or limit reached).
                        let _ = graph.add_edge(edge);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // PruneOrphans (dry-run only in V1)
    // -----------------------------------------------------------------------

    fn op_prune_orphans(
        &self,
        graph: &MemoryGraph,
        max_decay: f32,
        session_filter: Option<(u32, u32)>,
        report: &mut ConsolidationReport,
    ) {
        let orphan_ids: Vec<u64> = graph
            .nodes()
            .iter()
            .filter(|n| {
                n.access_count == 0
                    && n.decay_score < max_decay
                    && in_session_range(n.session_id, session_filter)
                    && graph.edges_to(n.id).is_empty()
            })
            .map(|n| n.id)
            .collect();

        if !orphan_ids.is_empty() {
            report.actions.push(ConsolidationAction {
                operation: "prune_orphans".to_string(),
                description: format!(
                    "Would prune {} orphaned node(s) with decay_score < {:.2} and no incoming edges",
                    orphan_ids.len(),
                    max_decay,
                ),
                affected_nodes: orphan_ids.clone(),
            });
            report.pruned += orphan_ids.len();
        }
    }

    // -----------------------------------------------------------------------
    // LinkContradictions
    // -----------------------------------------------------------------------

    fn op_link_contradictions(
        &self,
        graph: &mut MemoryGraph,
        threshold: f32,
        session_filter: Option<(u32, u32)>,
        dry_run: bool,
        report: &mut ConsolidationReport,
    ) {
        let tokenizer = Tokenizer::new();

        // Collect candidate nodes: Facts and Inferences.
        let candidates: Vec<u64> = graph
            .nodes()
            .iter()
            .filter(|n| {
                (n.event_type == EventType::Fact || n.event_type == EventType::Inference)
                    && in_session_range(n.session_id, session_filter)
            })
            .map(|n| n.id)
            .collect();

        // Build a set of existing Contradicts pairs for dedup.
        let mut existing_contradictions: HashSet<(u64, u64)> = HashSet::new();
        for edge in graph.edges() {
            if edge.edge_type == EdgeType::Contradicts {
                let pair = ordered_pair(edge.source_id, edge.target_id);
                existing_contradictions.insert(pair);
            }
        }

        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let id_a = candidates[i];
                let id_b = candidates[j];

                // Skip if already linked.
                if existing_contradictions.contains(&ordered_pair(id_a, id_b)) {
                    continue;
                }

                let (vec_a, content_a) = match graph.get_node(id_a) {
                    Some(n) => (n.feature_vec.clone(), n.content.clone()),
                    None => continue,
                };
                let (vec_b, content_b) = match graph.get_node(id_b) {
                    Some(n) => (n.feature_vec.clone(), n.content.clone()),
                    None => continue,
                };

                let sim = cosine_similarity(&vec_a, &vec_b);
                if sim < threshold {
                    continue;
                }

                // Check for negation: at least one of the two contents must
                // contain a negation word that does NOT appear in the other.
                let tokens_a: HashSet<String> =
                    tokenizer.tokenize(&content_a).into_iter().collect();
                let tokens_b: HashSet<String> =
                    tokenizer.tokenize(&content_b).into_iter().collect();

                let neg_set: HashSet<&str> = NEGATION_WORDS.iter().copied().collect();

                let neg_in_a = tokens_a.iter().any(|t| neg_set.contains(t.as_str()));
                let neg_in_b = tokens_b.iter().any(|t| neg_set.contains(t.as_str()));

                // Contradiction signal: high similarity but exactly one side
                // uses negation, OR both use negation words that differ.
                if !(neg_in_a ^ neg_in_b) {
                    continue;
                }

                existing_contradictions.insert(ordered_pair(id_a, id_b));

                report.actions.push(ConsolidationAction {
                    operation: "link_contradictions".to_string(),
                    description: format!(
                        "Nodes {} and {} appear contradictory (cosine={:.3})",
                        id_a, id_b, sim,
                    ),
                    affected_nodes: vec![id_a, id_b],
                });
                report.contradictions_linked += 1;

                if !dry_run {
                    let edge = Edge {
                        source_id: id_a,
                        target_id: id_b,
                        edge_type: EdgeType::Contradicts,
                        weight: sim,
                        created_at: crate::types::now_micros(),
                    };
                    let _ = graph.add_edge(edge);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // CompressEpisodes (dry-run only in V1)
    // -----------------------------------------------------------------------

    fn op_compress_episodes(
        &self,
        graph: &MemoryGraph,
        group_size: u32,
        session_filter: Option<(u32, u32)>,
        report: &mut ConsolidationReport,
    ) {
        // Collect Episode nodes sorted by creation time.
        let mut episodes: Vec<(u64, u64, u32)> = graph
            .nodes()
            .iter()
            .filter(|n| {
                n.event_type == EventType::Episode && in_session_range(n.session_id, session_filter)
            })
            .map(|n| (n.id, n.created_at, n.session_id))
            .collect();

        episodes.sort_by_key(|&(_, ts, _)| ts);

        if episodes.len() < group_size as usize {
            return;
        }

        // Group contiguous episodes from the same session.
        let mut groups: Vec<Vec<u64>> = Vec::new();
        let mut current_group: Vec<u64> = vec![episodes[0].0];
        let mut current_session = episodes[0].2;

        for &(id, _, session) in &episodes[1..] {
            if session == current_session {
                current_group.push(id);
            } else {
                if current_group.len() >= group_size as usize {
                    groups.push(std::mem::take(&mut current_group));
                } else {
                    current_group.clear();
                }
                current_group.push(id);
                current_session = session;
            }
        }
        if current_group.len() >= group_size as usize {
            groups.push(current_group);
        }

        for group in &groups {
            report.actions.push(ConsolidationAction {
                operation: "compress_episodes".to_string(),
                description: format!(
                    "Would compress {} contiguous episode(s) into a summary",
                    group.len(),
                ),
                affected_nodes: group.clone(),
            });
            report.episodes_compressed += group.len();
        }
    }

    // -----------------------------------------------------------------------
    // PromoteInferences
    // -----------------------------------------------------------------------

    fn op_promote_inferences(
        &self,
        graph: &mut MemoryGraph,
        min_access: u32,
        min_confidence: f32,
        session_filter: Option<(u32, u32)>,
        dry_run: bool,
        report: &mut ConsolidationReport,
    ) {
        // First pass: collect IDs of eligible Inference nodes.
        let eligible: Vec<u64> = graph
            .nodes()
            .iter()
            .filter(|n| {
                n.event_type == EventType::Inference
                    && n.access_count >= min_access
                    && n.confidence >= min_confidence
                    && in_session_range(n.session_id, session_filter)
            })
            .map(|n| n.id)
            .collect();

        for &id in &eligible {
            report.actions.push(ConsolidationAction {
                operation: "promote_inferences".to_string(),
                description: format!("Promote inference node {} to fact", id),
                affected_nodes: vec![id],
            });
            report.inferences_promoted += 1;

            if !dry_run {
                if let Some(node) = graph.get_node_mut(id) {
                    node.event_type = EventType::Fact;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether `session_id` falls within an optional inclusive range.
fn in_session_range(session_id: u32, range: Option<(u32, u32)>) -> bool {
    match range {
        Some((lo, hi)) => session_id >= lo && session_id <= hi,
        None => true,
    }
}

/// Return the pair `(min, max)` so we can use it as a canonical key.
fn ordered_pair(a: u64, b: u64) -> (u64, u64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}
