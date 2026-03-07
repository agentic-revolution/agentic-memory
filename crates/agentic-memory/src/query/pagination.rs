use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPage<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
    pub has_more: bool,
    pub total: Option<usize>,
}

impl<T: Clone> CursorPage<T> {
    pub fn from_slice(data: &[T], cursor: Option<&str>, limit: usize) -> Self {
        let offset = cursor.and_then(|c| c.parse::<usize>().ok()).unwrap_or(0);
        if offset >= data.len() {
            return Self {
                items: vec![],
                next_cursor: None,
                has_more: false,
                total: Some(data.len()),
            };
        }
        let end = (offset + limit).min(data.len());
        let has_more = end < data.len();
        Self {
            items: data[offset..end].to_vec(),
            next_cursor: if has_more {
                Some(end.to_string())
            } else {
                None
            },
            has_more,
            total: Some(data.len()),
        }
    }
    pub fn empty() -> Self {
        Self {
            items: vec![],
            next_cursor: None,
            has_more: false,
            total: Some(0),
        }
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
