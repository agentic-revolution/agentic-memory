use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub struct CacheInvalidator<K> {
    dependents: HashMap<K, HashSet<K>>,
}

impl<K: Eq + Hash + Clone> CacheInvalidator<K> {
    pub fn new() -> Self {
        Self {
            dependents: HashMap::new(),
        }
    }
    pub fn add_dependency(&mut self, dep: K, dependent: K) {
        self.dependents.entry(dep).or_default().insert(dependent);
    }
    pub fn cascade(&self, key: &K) -> Vec<K> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![key.clone()];
        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            result.push(current.clone());
            if let Some(deps) = self.dependents.get(&current) {
                for dep in deps {
                    if !visited.contains(dep) {
                        stack.push(dep.clone());
                    }
                }
            }
        }
        result
    }
    pub fn clear(&mut self) {
        self.dependents.clear();
    }
}

impl<K: Eq + Hash + Clone> Default for CacheInvalidator<K> {
    fn default() -> Self {
        Self::new()
    }
}
