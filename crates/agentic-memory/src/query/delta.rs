use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType { Created, Updated, Deleted }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedState<T> { version: u64, last_modified: DateTime<Utc>, state: T, changes: Vec<(u64, DateTime<Utc>, ChangeType, T)>, max_history: usize }

impl<T: Clone> VersionedState<T> {
    pub fn new(initial: T) -> Self { Self { version: 0, last_modified: Utc::now(), state: initial, changes: Vec::new(), max_history: 100 } }
    pub fn version(&self) -> u64 { self.version }
    pub fn state(&self) -> &T { &self.state }
    pub fn record_change(&mut self, change_type: ChangeType, new_state: T) {
        self.version += 1;
        self.last_modified = Utc::now();
        self.state = new_state.clone();
        self.changes.push((self.version, self.last_modified, change_type, new_state));
        if self.changes.len() > self.max_history { self.changes.drain(..self.changes.len() - self.max_history); }
    }
    pub fn changes_since(&self, since_version: u64) -> DeltaResult<T> {
        if since_version >= self.version { return DeltaResult::Unchanged { version: self.version, _marker: PhantomData }; }
        let changes: Vec<_> = self.changes.iter().filter(|(v, _, _, _)| *v > since_version).cloned().collect();
        if changes.is_empty() { DeltaResult::Unchanged { version: self.version, _marker: PhantomData } }
        else { DeltaResult::Changed { version: self.version, count: changes.len(), _marker: PhantomData } }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaResult<T> {
    Unchanged { version: u64, #[serde(skip)] _marker: PhantomData<T> },
    Changed { version: u64, count: usize, #[serde(skip)] _marker: PhantomData<T> },
}

impl<T> DeltaResult<T> {
    pub fn is_unchanged(&self) -> bool { matches!(self, Self::Unchanged { .. }) }
    pub fn version(&self) -> u64 { match self { Self::Unchanged { version, .. } | Self::Changed { version, .. } => *version } }
}
