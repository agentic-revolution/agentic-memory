//! Graph algorithms: centrality (PageRank, degree, betweenness) and shortest path (queries 10-11).

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::graph::traversal::TraversalDirection;
use crate::graph::MemoryGraph;
use crate::types::{AmemResult, Edge, EdgeType, EventType};

/// Which centrality algorithm to use.
#[derive(Debug, Clone)]
pub enum CentralityAlgorithm {
    /// Standard PageRank — importance flows through edges.
    PageRank { damping: f32 },
    /// Degree centrality — simple count of connections.
    Degree,
    /// Betweenness centrality — how often a node appears on shortest paths.
    Betweenness,
}

/// Parameters for a centrality query.
pub struct CentralityParams {
    pub algorithm: CentralityAlgorithm,
    pub max_iterations: u32,
    pub tolerance: f32,
    pub top_k: usize,
    pub event_types: Vec<EventType>,
    pub edge_types: Vec<EdgeType>,
}

/// Result of a centrality computation.
pub struct CentralityResult {
    /// Node ID → centrality score, sorted by score descending.
    pub scores: Vec<(u64, f32)>,
    pub algorithm: CentralityAlgorithm,
    pub iterations: u32,
    pub converged: bool,
}

/// Parameters for shortest path query.
pub struct ShortestPathParams {
    pub source_id: u64,
    pub target_id: u64,
    pub edge_types: Vec<EdgeType>,
    pub direction: TraversalDirection,
    pub max_depth: u32,
    pub weighted: bool,
}

/// Result of a shortest path query.
pub struct PathResult {
    /// Ordered list of node IDs from source to target (inclusive). Empty if no path.
    pub path: Vec<u64>,
    /// Edges traversed along the path.
    pub edges: Vec<Edge>,
    /// Total path length.
    pub cost: f32,
    pub found: bool,
}

impl super::query::QueryEngine {
    /// Compute centrality scores for nodes in the graph.
    pub fn centrality(
        &self,
        graph: &MemoryGraph,
        params: CentralityParams,
    ) -> AmemResult<CentralityResult> {
        let type_filter: HashSet<EventType> = params.event_types.iter().copied().collect();
        let edge_filter: HashSet<EdgeType> = params.edge_types.iter().copied().collect();

        // Collect candidate nodes
        let node_ids: Vec<u64> = graph
            .nodes()
            .iter()
            .filter(|n| type_filter.is_empty() || type_filter.contains(&n.event_type))
            .map(|n| n.id)
            .collect();

        let node_set: HashSet<u64> = node_ids.iter().copied().collect();

        // Collect relevant edges
        let edges: Vec<&Edge> = graph
            .edges()
            .iter()
            .filter(|e| {
                node_set.contains(&e.source_id)
                    && node_set.contains(&e.target_id)
                    && (edge_filter.is_empty() || edge_filter.contains(&e.edge_type))
            })
            .collect();

        match params.algorithm {
            CentralityAlgorithm::PageRank { damping } => self.pagerank(
                &node_ids,
                &edges,
                damping,
                params.max_iterations,
                params.tolerance,
                params.top_k,
            ),
            CentralityAlgorithm::Degree => self.degree_centrality(&node_ids, &edges, params.top_k),
            CentralityAlgorithm::Betweenness => {
                self.betweenness_centrality(&node_ids, &edges, params.top_k)
            }
        }
    }

    fn pagerank(
        &self,
        node_ids: &[u64],
        edges: &[&Edge],
        damping: f32,
        max_iterations: u32,
        tolerance: f32,
        top_k: usize,
    ) -> AmemResult<CentralityResult> {
        let n = node_ids.len();
        if n == 0 {
            return Ok(CentralityResult {
                scores: Vec::new(),
                algorithm: CentralityAlgorithm::PageRank { damping },
                iterations: 0,
                converged: true,
            });
        }

        let id_to_idx: HashMap<u64, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Build outgoing edges and incoming edges
        let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];

        for edge in edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = (
                id_to_idx.get(&edge.source_id),
                id_to_idx.get(&edge.target_id),
            ) {
                outgoing[src_idx].push(tgt_idx);
                incoming[tgt_idx].push(src_idx);
            }
        }

        let mut pr = vec![1.0 / n as f32; n];
        let mut iterations = 0;
        let mut converged = false;

        for _ in 0..max_iterations {
            iterations += 1;
            let mut new_pr = vec![(1.0 - damping) / n as f32; n];

            // Dangling node rank
            let dangling_sum: f32 = (0..n)
                .filter(|&i| outgoing[i].is_empty())
                .map(|i| pr[i])
                .sum();

            for i in 0..n {
                new_pr[i] += damping * dangling_sum / n as f32;
                for &j in &incoming[i] {
                    let out_degree = outgoing[j].len() as f32;
                    if out_degree > 0.0 {
                        new_pr[i] += damping * pr[j] / out_degree;
                    }
                }
            }

            // Check convergence
            let max_diff = (0..n)
                .map(|i| (new_pr[i] - pr[i]).abs())
                .fold(0.0f32, f32::max);

            pr = new_pr;

            if max_diff < tolerance {
                converged = true;
                break;
            }
        }

        let mut scores: Vec<(u64, f32)> = node_ids
            .iter()
            .zip(pr.iter())
            .map(|(&id, &s)| (id, s))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(CentralityResult {
            scores,
            algorithm: CentralityAlgorithm::PageRank { damping },
            iterations,
            converged,
        })
    }

    fn degree_centrality(
        &self,
        node_ids: &[u64],
        edges: &[&Edge],
        top_k: usize,
    ) -> AmemResult<CentralityResult> {
        let n = node_ids.len();
        let mut degrees: HashMap<u64, u32> = HashMap::new();
        for &id in node_ids {
            degrees.insert(id, 0);
        }

        for edge in edges {
            *degrees.entry(edge.source_id).or_insert(0) += 1;
            *degrees.entry(edge.target_id).or_insert(0) += 1;
        }

        let max_possible = if n > 1 { 2 * (n - 1) } else { 1 };

        let mut scores: Vec<(u64, f32)> = degrees
            .into_iter()
            .map(|(id, deg)| (id, deg as f32 / max_possible as f32))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(CentralityResult {
            scores,
            algorithm: CentralityAlgorithm::Degree,
            iterations: 0,
            converged: true,
        })
    }

    fn betweenness_centrality(
        &self,
        node_ids: &[u64],
        edges: &[&Edge],
        top_k: usize,
    ) -> AmemResult<CentralityResult> {
        let n = node_ids.len();
        if n == 0 {
            return Ok(CentralityResult {
                scores: Vec::new(),
                algorithm: CentralityAlgorithm::Betweenness,
                iterations: 0,
                converged: true,
            });
        }

        let id_to_idx: HashMap<u64, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Build adjacency list (both directions for undirected betweenness)
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in edges {
            if let (Some(&src), Some(&tgt)) = (
                id_to_idx.get(&edge.source_id),
                id_to_idx.get(&edge.target_id),
            ) {
                adj[src].push(tgt);
                adj[tgt].push(src);
            }
        }

        let mut betweenness = vec![0.0f32; n];

        // Sample source nodes if graph is large
        let sources: Vec<usize> = if n > 10_000 {
            (0..1000.min(n)).collect()
        } else {
            (0..n).collect()
        };

        // Brandes' algorithm
        for &s in &sources {
            let mut stack: Vec<usize> = Vec::new();
            let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n];
            sigma[s] = 1.0;
            let mut dist: Vec<i64> = vec![-1; n];
            dist[s] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for &w in &adj[v] {
                    if dist[w] < 0 {
                        queue.push_back(w);
                        dist[w] = dist[v] + 1;
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        pred[w].push(v);
                    }
                }
            }

            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    betweenness[w] += delta[w] as f32;
                }
            }
        }

        // Normalize
        let norm = if n > 2 {
            ((n - 1) * (n - 2)) as f32
        } else {
            1.0
        };

        let mut scores: Vec<(u64, f32)> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, betweenness[i] / norm))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(CentralityResult {
            scores,
            algorithm: CentralityAlgorithm::Betweenness,
            iterations: 0,
            converged: true,
        })
    }

    /// Find the shortest path between two nodes.
    pub fn shortest_path(
        &self,
        graph: &MemoryGraph,
        params: ShortestPathParams,
    ) -> AmemResult<PathResult> {
        // Same node
        if params.source_id == params.target_id {
            return Ok(PathResult {
                path: vec![params.source_id],
                edges: Vec::new(),
                cost: 0.0,
                found: true,
            });
        }

        // Check nodes exist
        if graph.get_node(params.source_id).is_none() {
            return Err(crate::types::AmemError::NodeNotFound(params.source_id));
        }
        if graph.get_node(params.target_id).is_none() {
            return Err(crate::types::AmemError::NodeNotFound(params.target_id));
        }

        let edge_filter: HashSet<EdgeType> = params.edge_types.iter().copied().collect();

        if params.weighted {
            self.dijkstra_path(graph, &params, &edge_filter)
        } else {
            self.bidirectional_bfs(graph, &params, &edge_filter)
        }
    }

    fn bidirectional_bfs(
        &self,
        graph: &MemoryGraph,
        params: &ShortestPathParams,
        edge_filter: &HashSet<EdgeType>,
    ) -> AmemResult<PathResult> {
        let mut forward_visited: HashMap<u64, u64> = HashMap::new(); // node -> parent
        let mut backward_visited: HashMap<u64, u64> = HashMap::new();
        let mut forward_queue: VecDeque<(u64, u32)> = VecDeque::new();
        let mut backward_queue: VecDeque<(u64, u32)> = VecDeque::new();

        forward_visited.insert(params.source_id, params.source_id);
        backward_visited.insert(params.target_id, params.target_id);
        forward_queue.push_back((params.source_id, 0));
        backward_queue.push_back((params.target_id, 0));

        let half_depth = params.max_depth / 2 + 1;
        let mut meeting_node: Option<u64> = None;

        // Helper to get neighbors
        let get_neighbors = |node_id: u64, forward: bool| -> Vec<u64> {
            let mut neighbors = Vec::new();
            match params.direction {
                TraversalDirection::Forward | TraversalDirection::Both => {
                    if forward {
                        for edge in graph.edges_from(node_id) {
                            if edge_filter.is_empty() || edge_filter.contains(&edge.edge_type) {
                                neighbors.push(edge.target_id);
                            }
                        }
                    }
                }
                TraversalDirection::Backward => {}
            }
            match params.direction {
                TraversalDirection::Backward | TraversalDirection::Both => {
                    if forward {
                        for edge in graph.edges_to(node_id) {
                            if edge_filter.is_empty() || edge_filter.contains(&edge.edge_type) {
                                neighbors.push(edge.source_id);
                            }
                        }
                    }
                }
                TraversalDirection::Forward => {}
            }
            // For backward search, reverse the directions
            if !forward {
                let mut rev_neighbors = Vec::new();
                match params.direction {
                    TraversalDirection::Forward | TraversalDirection::Both => {
                        for edge in graph.edges_to(node_id) {
                            if edge_filter.is_empty() || edge_filter.contains(&edge.edge_type) {
                                rev_neighbors.push(edge.source_id);
                            }
                        }
                    }
                    TraversalDirection::Backward => {}
                }
                match params.direction {
                    TraversalDirection::Backward | TraversalDirection::Both => {
                        for edge in graph.edges_from(node_id) {
                            if edge_filter.is_empty() || edge_filter.contains(&edge.edge_type) {
                                rev_neighbors.push(edge.target_id);
                            }
                        }
                    }
                    TraversalDirection::Forward => {}
                }
                return rev_neighbors;
            }
            neighbors
        };

        'outer: while !forward_queue.is_empty() || !backward_queue.is_empty() {
            // Expand forward
            if let Some((node, depth)) = forward_queue.pop_front() {
                if depth < half_depth {
                    for neighbor in get_neighbors(node, true) {
                        forward_visited.entry(neighbor).or_insert_with(|| {
                            forward_queue.push_back((neighbor, depth + 1));
                            node
                        });
                        if backward_visited.contains_key(&neighbor) {
                            forward_visited.entry(neighbor).or_insert(node);
                            meeting_node = Some(neighbor);
                            break 'outer;
                        }
                    }
                }
            }

            // Expand backward
            if let Some((node, depth)) = backward_queue.pop_front() {
                if depth < half_depth {
                    for neighbor in get_neighbors(node, false) {
                        backward_visited.entry(neighbor).or_insert_with(|| {
                            backward_queue.push_back((neighbor, depth + 1));
                            node
                        });
                        if forward_visited.contains_key(&neighbor) {
                            backward_visited.entry(neighbor).or_insert(node);
                            meeting_node = Some(neighbor);
                            break 'outer;
                        }
                    }
                }
            }
        }

        match meeting_node {
            Some(mid) => {
                // Reconstruct path
                let mut forward_path = Vec::new();
                let mut current = mid;
                while current != params.source_id {
                    forward_path.push(current);
                    current = forward_visited[&current];
                }
                forward_path.push(params.source_id);
                forward_path.reverse();

                let mut backward_path = Vec::new();
                current = mid;
                while current != params.target_id {
                    current = backward_visited[&current];
                    backward_path.push(current);
                }

                let mut path = forward_path;
                path.extend(backward_path);

                let cost = (path.len() - 1) as f32;

                // Collect edges along the path
                let mut edges = Vec::new();
                for i in 0..path.len() - 1 {
                    for edge in graph.edges_from(path[i]) {
                        if edge.target_id == path[i + 1] {
                            edges.push(*edge);
                            break;
                        }
                    }
                    if edges.len() < i + 1 {
                        // Try reverse direction
                        for edge in graph.edges_from(path[i + 1]) {
                            if edge.target_id == path[i] {
                                edges.push(*edge);
                                break;
                            }
                        }
                    }
                }

                Ok(PathResult {
                    path,
                    edges,
                    cost,
                    found: true,
                })
            }
            None => Ok(PathResult {
                path: Vec::new(),
                edges: Vec::new(),
                cost: 0.0,
                found: false,
            }),
        }
    }

    fn dijkstra_path(
        &self,
        graph: &MemoryGraph,
        params: &ShortestPathParams,
        edge_filter: &HashSet<EdgeType>,
    ) -> AmemResult<PathResult> {
        #[derive(PartialEq)]
        struct State {
            cost: f32,
            node: u64,
        }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let mut dist: HashMap<u64, f32> = HashMap::new();
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(params.source_id, 0.0);
        heap.push(State {
            cost: 0.0,
            node: params.source_id,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if node == params.target_id {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = params.target_id;
                while current != params.source_id {
                    path.push(current);
                    current = prev[&current];
                }
                path.push(params.source_id);
                path.reverse();

                // Collect edges
                let mut edges = Vec::new();
                for i in 0..path.len() - 1 {
                    for edge in graph.edges_from(path[i]) {
                        if edge.target_id == path[i + 1] {
                            edges.push(*edge);
                            break;
                        }
                    }
                }

                return Ok(PathResult {
                    path,
                    edges,
                    cost,
                    found: true,
                });
            }

            if cost > *dist.get(&node).unwrap_or(&f32::INFINITY) {
                continue;
            }

            // Explore neighbors
            for edge in graph.edges_from(node) {
                if !edge_filter.is_empty() && !edge_filter.contains(&edge.edge_type) {
                    continue;
                }
                let edge_cost = 1.0 - edge.weight; // Higher weight = lower cost
                let next_cost = cost + edge_cost;

                if next_cost < *dist.get(&edge.target_id).unwrap_or(&f32::INFINITY) {
                    dist.insert(edge.target_id, next_cost);
                    prev.insert(edge.target_id, node);
                    heap.push(State {
                        cost: next_cost,
                        node: edge.target_id,
                    });
                }
            }

            // If direction allows backward/both, also check incoming edges
            if matches!(
                params.direction,
                TraversalDirection::Backward | TraversalDirection::Both
            ) {
                for edge in graph.edges_to(node) {
                    if !edge_filter.is_empty() && !edge_filter.contains(&edge.edge_type) {
                        continue;
                    }
                    let edge_cost = 1.0 - edge.weight;
                    let next_cost = cost + edge_cost;

                    if next_cost < *dist.get(&edge.source_id).unwrap_or(&f32::INFINITY) {
                        dist.insert(edge.source_id, next_cost);
                        prev.insert(edge.source_id, node);
                        heap.push(State {
                            cost: next_cost,
                            node: edge.source_id,
                        });
                    }
                }
            }
        }

        Ok(PathResult {
            path: Vec::new(),
            edges: Vec::new(),
            cost: 0.0,
            found: false,
        })
    }
}
