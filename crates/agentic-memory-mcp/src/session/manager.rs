//! Graph lifecycle management, file I/O, and session tracking.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use agentic_memory::{
    AmemReader, AmemWriter, CognitiveEventBuilder, Edge, EdgeType, EventType, MemoryGraph,
    QueryEngine, WriteEngine,
};

use crate::types::{McpError, McpResult};

/// Default auto-save interval.
const DEFAULT_AUTO_SAVE_SECS: u64 = 30;

/// Manages the memory graph lifecycle, file I/O, and session state.
pub struct SessionManager {
    graph: MemoryGraph,
    query_engine: QueryEngine,
    write_engine: WriteEngine,
    file_path: PathBuf,
    current_session: u32,
    dirty: bool,
    last_save: Instant,
    auto_save_interval: Duration,
}

impl SessionManager {
    /// Open or create a memory file at the given path.
    pub fn open(path: &str) -> McpResult<Self> {
        let file_path = PathBuf::from(path);
        let dimension = agentic_memory::DEFAULT_DIMENSION;

        let graph = if file_path.exists() {
            tracing::info!("Opening existing memory file: {}", file_path.display());
            AmemReader::read_from_file(&file_path)
                .map_err(|e| McpError::AgenticMemory(format!("Failed to read memory file: {e}")))?
        } else {
            tracing::info!("Creating new memory file: {}", file_path.display());
            // Ensure parent directory exists
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    McpError::Io(std::io::Error::other(format!(
                        "Failed to create directory {}: {e}",
                        parent.display()
                    )))
                })?;
            }
            MemoryGraph::new(dimension)
        };

        // Determine the next session ID from existing sessions
        let session_ids = graph.session_index().session_ids();
        let current_session = session_ids.iter().copied().max().unwrap_or(0) + 1;

        tracing::info!(
            "Session {} started. Graph has {} nodes, {} edges.",
            current_session,
            graph.node_count(),
            graph.edge_count()
        );

        Ok(Self {
            graph,
            query_engine: QueryEngine::new(),
            write_engine: WriteEngine::new(dimension),
            file_path,
            current_session,
            dirty: false,
            last_save: Instant::now(),
            auto_save_interval: Duration::from_secs(DEFAULT_AUTO_SAVE_SECS),
        })
    }

    /// Get an immutable reference to the graph.
    pub fn graph(&self) -> &MemoryGraph {
        &self.graph
    }

    /// Get a mutable reference to the graph and mark as dirty.
    pub fn graph_mut(&mut self) -> &mut MemoryGraph {
        self.dirty = true;
        &mut self.graph
    }

    /// Get the query engine.
    pub fn query_engine(&self) -> &QueryEngine {
        &self.query_engine
    }

    /// Get the write engine.
    pub fn write_engine(&self) -> &WriteEngine {
        &self.write_engine
    }

    /// Current session ID.
    pub fn current_session_id(&self) -> u32 {
        self.current_session
    }

    /// Start a new session, optionally with an explicit ID.
    pub fn start_session(&mut self, explicit_id: Option<u32>) -> McpResult<u32> {
        let session_id = explicit_id.unwrap_or_else(|| {
            let ids = self.graph.session_index().session_ids();
            ids.iter().copied().max().unwrap_or(0) + 1
        });

        self.current_session = session_id;
        tracing::info!("Started session {session_id}");
        Ok(session_id)
    }

    /// End a session and optionally create an episode summary.
    pub fn end_session_with_episode(&mut self, session_id: u32, summary: &str) -> McpResult<u64> {
        let episode_id = self
            .write_engine
            .compress_session(&mut self.graph, session_id, summary)
            .map_err(|e| McpError::AgenticMemory(format!("Failed to compress session: {e}")))?;

        self.dirty = true;
        self.save()?;

        tracing::info!("Ended session {session_id}, created episode node {episode_id}");

        Ok(episode_id)
    }

    /// Save the graph to file.
    pub fn save(&mut self) -> McpResult<()> {
        if !self.dirty {
            return Ok(());
        }

        let writer = AmemWriter::new(self.graph.dimension());
        writer
            .write_to_file(&self.graph, &self.file_path)
            .map_err(|e| McpError::AgenticMemory(format!("Failed to write memory file: {e}")))?;

        self.dirty = false;
        self.last_save = Instant::now();
        tracing::debug!("Saved memory file: {}", self.file_path.display());
        Ok(())
    }

    /// Check if auto-save is needed and save if so.
    pub fn maybe_auto_save(&mut self) -> McpResult<()> {
        if self.dirty && self.last_save.elapsed() >= self.auto_save_interval {
            self.save()?;
        }
        Ok(())
    }

    /// Mark the graph as dirty (needs saving).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Get the file path.
    pub fn file_path(&self) -> &PathBuf {
        &self.file_path
    }

    /// Add a cognitive event to the graph.
    pub fn add_event(
        &mut self,
        event_type: EventType,
        content: &str,
        confidence: f32,
        edges: Vec<(u64, EdgeType, f32)>,
    ) -> McpResult<(u64, usize)> {
        let event = CognitiveEventBuilder::new(event_type, content.to_string())
            .session_id(self.current_session)
            .confidence(confidence)
            .build();

        // First, add the node to get its assigned ID
        let result = self
            .write_engine
            .ingest(&mut self.graph, vec![event], vec![])
            .map_err(|e| McpError::AgenticMemory(format!("Failed to add event: {e}")))?;

        let node_id = result.new_node_ids.first().copied().ok_or_else(|| {
            McpError::InternalError("No node ID returned from ingest".to_string())
        })?;

        // Then add edges with the correct source_id
        let mut edge_count = 0;
        for (target_id, edge_type, weight) in &edges {
            let edge = Edge::new(node_id, *target_id, *edge_type, *weight);
            self.graph
                .add_edge(edge)
                .map_err(|e| McpError::AgenticMemory(format!("Failed to add edge: {e}")))?;
            edge_count += 1;
        }

        self.dirty = true;
        self.maybe_auto_save()?;

        Ok((node_id, edge_count))
    }

    /// Correct a previous belief.
    pub fn correct_node(&mut self, old_node_id: u64, new_content: &str) -> McpResult<u64> {
        let new_id = self
            .write_engine
            .correct(
                &mut self.graph,
                old_node_id,
                new_content,
                self.current_session,
            )
            .map_err(|e| McpError::AgenticMemory(format!("Failed to correct node: {e}")))?;

        self.dirty = true;
        self.maybe_auto_save()?;

        Ok(new_id)
    }
}

impl Drop for SessionManager {
    fn drop(&mut self) {
        if self.dirty {
            if let Err(e) = self.save() {
                tracing::error!("Failed to save on drop: {e}");
            }
        }
    }
}
