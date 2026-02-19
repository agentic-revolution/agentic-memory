//! BM25 text search and hybrid search (queries 8-9).

use std::collections::HashMap;

use crate::engine::tokenizer::Tokenizer;
use crate::graph::MemoryGraph;
use crate::index::cosine_similarity;
use crate::index::{DocLengths, TermIndex};
use crate::types::{AmemResult, EventType};

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Parameters for BM25 text search.
pub struct TextSearchParams {
    /// The search query string (will be tokenized).
    pub query: String,
    /// Maximum number of results.
    pub max_results: usize,
    /// Filter by event type(s). Empty = all types.
    pub event_types: Vec<EventType>,
    /// Filter by session ID(s). Empty = all sessions.
    pub session_ids: Vec<u32>,
    /// Minimum BM25 score to include (default: 0.0).
    pub min_score: f32,
}

/// A single BM25 text search match.
pub struct TextMatch {
    pub node_id: u64,
    pub score: f32,
    /// Which query terms matched in this node's content.
    pub matched_terms: Vec<String>,
}

/// Parameters for hybrid BM25 + vector search.
pub struct HybridSearchParams {
    /// Text query for BM25 component.
    pub query_text: String,
    /// Feature vector for similarity component. If None, runs BM25-only.
    pub query_vec: Option<Vec<f32>>,
    /// Maximum number of final results.
    pub max_results: usize,
    /// Filter by event type(s). Empty = all types.
    pub event_types: Vec<EventType>,
    /// Weight for BM25 component (0.0 to 1.0, default: 0.5).
    pub text_weight: f32,
    /// Weight for vector component (0.0 to 1.0, default: 0.5).
    pub vector_weight: f32,
    /// RRF constant k (default: 60).
    pub rrf_k: u32,
}

/// A single hybrid search match.
pub struct HybridMatch {
    pub node_id: u64,
    /// Combined RRF score.
    pub combined_score: f32,
    /// BM25 rank (1-based, 0 if not in BM25 results).
    pub text_rank: u32,
    /// Similarity rank (1-based, 0 if not in similarity results).
    pub vector_rank: u32,
    /// Raw BM25 score.
    pub text_score: f32,
    /// Raw cosine similarity.
    pub vector_similarity: f32,
}

impl super::query::QueryEngine {
    /// BM25 text search over node contents.
    /// Uses TermIndex if available, falls back to full scan.
    pub fn text_search(
        &self,
        graph: &MemoryGraph,
        term_index: Option<&TermIndex>,
        doc_lengths: Option<&DocLengths>,
        params: TextSearchParams,
    ) -> AmemResult<Vec<TextMatch>> {
        let tokenizer = Tokenizer::new();
        let query_terms = tokenizer.tokenize(&params.query);

        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        // Build type/session filter sets
        let type_filter: std::collections::HashSet<EventType> =
            params.event_types.iter().copied().collect();
        let session_filter: std::collections::HashSet<u32> =
            params.session_ids.iter().copied().collect();

        let matches = if let (Some(ti), Some(dl)) = (term_index, doc_lengths) {
            // Fast path: use pre-built indexes
            self.bm25_fast_path(graph, ti, dl, &query_terms, &type_filter, &session_filter)
        } else {
            // Slow path: full scan
            self.bm25_slow_path(
                graph,
                &tokenizer,
                &query_terms,
                &type_filter,
                &session_filter,
            )
        };

        let mut results: Vec<TextMatch> = matches
            .into_iter()
            .filter(|m| m.score >= params.min_score)
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(params.max_results);

        Ok(results)
    }

    fn bm25_fast_path(
        &self,
        graph: &MemoryGraph,
        term_index: &TermIndex,
        doc_lengths: &DocLengths,
        query_terms: &[String],
        type_filter: &std::collections::HashSet<EventType>,
        session_filter: &std::collections::HashSet<u32>,
    ) -> Vec<TextMatch> {
        let n = term_index.doc_count() as f32;
        let avgdl = term_index.avg_doc_length();

        // Collect all candidate node IDs from posting lists
        let mut scores: HashMap<u64, (f32, Vec<String>)> = HashMap::new();

        for term in query_terms {
            let postings = term_index.get(term);
            let df = postings.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for &(node_id, tf) in postings {
                // Apply filters
                if let Some(node) = graph.get_node(node_id) {
                    if !type_filter.is_empty() && !type_filter.contains(&node.event_type) {
                        continue;
                    }
                    if !session_filter.is_empty() && !session_filter.contains(&node.session_id) {
                        continue;
                    }
                }

                let dl = doc_lengths.get(node_id) as f32;
                let tf_f = tf as f32;
                let bm25_term = idf * (tf_f * (BM25_K1 + 1.0))
                    / (tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl.max(1.0)));

                let entry = scores.entry(node_id).or_insert((0.0, Vec::new()));
                entry.0 += bm25_term;
                if !entry.1.contains(term) {
                    entry.1.push(term.clone());
                }
            }
        }

        scores
            .into_iter()
            .map(|(node_id, (score, matched_terms))| TextMatch {
                node_id,
                score,
                matched_terms,
            })
            .collect()
    }

    fn bm25_slow_path(
        &self,
        graph: &MemoryGraph,
        tokenizer: &Tokenizer,
        query_terms: &[String],
        type_filter: &std::collections::HashSet<EventType>,
        session_filter: &std::collections::HashSet<u32>,
    ) -> Vec<TextMatch> {
        let nodes = graph.nodes();
        if nodes.is_empty() {
            return Vec::new();
        }

        // Build temporary term data
        let n = nodes.len() as f32;
        let mut doc_freqs: HashMap<String, usize> = HashMap::new();
        let mut node_data: Vec<(u64, HashMap<String, u32>, u32)> = Vec::new();
        let mut total_tokens: u64 = 0;

        for node in nodes {
            if !type_filter.is_empty() && !type_filter.contains(&node.event_type) {
                continue;
            }
            if !session_filter.is_empty() && !session_filter.contains(&node.session_id) {
                continue;
            }

            let freqs = tokenizer.term_frequencies(&node.content);
            let doc_len: u32 = freqs.values().sum();
            total_tokens += doc_len as u64;

            for term in freqs.keys() {
                *doc_freqs.entry(term.clone()).or_insert(0) += 1;
            }

            node_data.push((node.id, freqs, doc_len));
        }

        let avgdl = if node_data.is_empty() {
            0.0
        } else {
            total_tokens as f32 / node_data.len() as f32
        };

        let mut results = Vec::new();

        for (node_id, freqs, doc_len) in &node_data {
            let mut score = 0.0f32;
            let mut matched = Vec::new();

            for term in query_terms {
                if let Some(&tf) = freqs.get(term) {
                    let df = *doc_freqs.get(term).unwrap_or(&0) as f32;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let tf_f = tf as f32;
                    let dl = *doc_len as f32;
                    let bm25_term = idf * (tf_f * (BM25_K1 + 1.0))
                        / (tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl.max(1.0)));
                    score += bm25_term;
                    if !matched.contains(term) {
                        matched.push(term.clone());
                    }
                }
            }

            if score > 0.0 {
                results.push(TextMatch {
                    node_id: *node_id,
                    score,
                    matched_terms: matched,
                });
            }
        }

        results
    }

    /// Hybrid BM25 + vector search with Reciprocal Rank Fusion.
    pub fn hybrid_search(
        &self,
        graph: &MemoryGraph,
        term_index: Option<&TermIndex>,
        doc_lengths: Option<&DocLengths>,
        params: HybridSearchParams,
    ) -> AmemResult<Vec<HybridMatch>> {
        let overfetch = params.max_results * 3;

        // Normalize weights
        let total_weight = params.text_weight + params.vector_weight;
        let (tw, vw) = if total_weight > 0.0 {
            (
                params.text_weight / total_weight,
                params.vector_weight / total_weight,
            )
        } else {
            (0.5, 0.5)
        };

        // Run BM25 search
        let bm25_results = self.text_search(
            graph,
            term_index,
            doc_lengths,
            TextSearchParams {
                query: params.query_text.clone(),
                max_results: overfetch,
                event_types: params.event_types.clone(),
                session_ids: Vec::new(),
                min_score: 0.0,
            },
        )?;

        // Build BM25 rank map
        let mut bm25_map: HashMap<u64, (u32, f32)> = HashMap::new();
        for (rank, m) in bm25_results.iter().enumerate() {
            bm25_map.insert(m.node_id, ((rank + 1) as u32, m.score));
        }

        // Run vector search if available
        let mut vec_map: HashMap<u64, (u32, f32)> = HashMap::new();
        let has_vectors = params.query_vec.is_some()
            && graph
                .nodes()
                .iter()
                .any(|n| n.feature_vec.iter().any(|&x| x != 0.0));

        if has_vectors {
            if let Some(ref qvec) = params.query_vec {
                let type_filter: std::collections::HashSet<EventType> =
                    params.event_types.iter().copied().collect();
                let mut sim_results: Vec<(u64, f32)> = Vec::new();

                for node in graph.nodes() {
                    if !type_filter.is_empty() && !type_filter.contains(&node.event_type) {
                        continue;
                    }
                    if node.feature_vec.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let sim = cosine_similarity(qvec, &node.feature_vec);
                    if sim > 0.0 {
                        sim_results.push((node.id, sim));
                    }
                }

                sim_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                sim_results.truncate(overfetch);

                for (rank, (node_id, sim)) in sim_results.iter().enumerate() {
                    vec_map.insert(*node_id, ((rank + 1) as u32, *sim));
                }
            }
        }

        // Combine all candidate node IDs
        let mut all_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        all_ids.extend(bm25_map.keys());
        all_ids.extend(vec_map.keys());

        let max_bm25_rank = (bm25_results.len() + 1) as u32;
        let max_vec_rank = (vec_map.len() + 1) as u32;
        let rrf_k = params.rrf_k as f32;

        let mut hybrid_results: Vec<HybridMatch> = all_ids
            .into_iter()
            .map(|node_id| {
                let (text_rank, text_score) = bm25_map
                    .get(&node_id)
                    .copied()
                    .unwrap_or((max_bm25_rank, 0.0));
                let (vector_rank, vector_similarity) = vec_map
                    .get(&node_id)
                    .copied()
                    .unwrap_or((max_vec_rank, 0.0));

                let rrf_text = tw / (rrf_k + text_rank as f32);
                let rrf_vec = if has_vectors {
                    vw / (rrf_k + vector_rank as f32)
                } else {
                    0.0
                };
                let combined_score = rrf_text + rrf_vec;

                HybridMatch {
                    node_id,
                    combined_score,
                    text_rank,
                    vector_rank,
                    text_score,
                    vector_similarity,
                }
            })
            .collect();

        hybrid_results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hybrid_results.truncate(params.max_results);

        Ok(hybrid_results)
    }
}
