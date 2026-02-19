//! Index structures for fast lookup. Each index is independent and incrementally updateable.

pub mod cluster_map;
pub mod doc_lengths;
pub mod session_index;
pub mod temporal_index;
pub mod term_index;
pub mod type_index;

pub use cluster_map::{cosine_similarity, ClusterMap};
pub use doc_lengths::DocLengths;
pub use session_index::SessionIndex;
pub use temporal_index::TemporalIndex;
pub use term_index::TermIndex;
pub use type_index::TypeIndex;
