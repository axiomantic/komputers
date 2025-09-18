//! Idiomatic Rust thread-safe handle tracking.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Thread-safe handle tracker using idiomatic Rust patterns.
///
/// This provides a Rust-idiomatic way to track GPU resource handles
/// with proper RAII cleanup and thread safety.
#[derive(Debug, Clone)]
pub struct HandleTracker {
    inner: Arc<HandleTrackerInner>,
}

#[derive(Debug)]
struct HandleTrackerInner {
    active_handles: Mutex<HashMap<String, Vec<u64>>>,
    next_id: AtomicU64,
}

impl HandleTracker {
    /// Create a new handle tracker.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HandleTrackerInner {
                active_handles: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Register a handle for tracking.
    pub fn register_handle(&self, category: &str, handle: u64) {
        let mut handles = self.inner.active_handles.lock().unwrap();
        handles
            .entry(category.to_string())
            .or_default()
            .push(handle);
    }

    /// Unregister a handle.
    pub fn unregister_handle(&self, category: &str, handle: u64) -> bool {
        let mut handles = self.inner.active_handles.lock().unwrap();
        if let Some(category_handles) = handles.get_mut(category) {
            if let Some(pos) = category_handles.iter().position(|&h| h == handle) {
                category_handles.swap_remove(pos);
                return true;
            }
        }
        false
    }

    /// Get all handles for a category.
    pub fn get_handles(&self, category: &str) -> Vec<u64> {
        let handles = self.inner.active_handles.lock().unwrap();
        handles.get(category).cloned().unwrap_or_default()
    }

    /// Get total handle count across all categories.
    pub fn total_handle_count(&self) -> usize {
        let handles = self.inner.active_handles.lock().unwrap();
        handles.values().map(|v| v.len()).sum()
    }

    /// Get handle count for a specific category.
    pub fn category_handle_count(&self, category: &str) -> usize {
        let handles = self.inner.active_handles.lock().unwrap();
        handles.get(category).map(|v| v.len()).unwrap_or(0)
    }

    /// Get statistics about handle usage.
    pub fn statistics(&self) -> HandleStatistics {
        let handles = self.inner.active_handles.lock().unwrap();
        let mut stats = HandleStatistics {
            categories: HashMap::new(),
            total_handles: 0,
        };

        for (category, category_handles) in handles.iter() {
            let count = category_handles.len();
            stats.categories.insert(category.clone(), count);
            stats.total_handles += count;
        }

        stats
    }

    /// Generate a unique ID (can be used for tracking purposes).
    pub fn generate_id(&self) -> u64 {
        self.inner.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Clear all handles for a category.
    pub fn clear_category(&self, category: &str) -> usize {
        let mut handles = self.inner.active_handles.lock().unwrap();
        handles.remove(category).map(|v| v.len()).unwrap_or(0)
    }

    /// Clear all handles.
    pub fn clear_all(&self) -> usize {
        let mut handles = self.inner.active_handles.lock().unwrap();
        let total = handles.values().map(|v| v.len()).sum();
        handles.clear();
        total
    }
}

impl Default for HandleTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about handle usage.
#[derive(Debug, Clone)]
pub struct HandleStatistics {
    /// Number of handles per category.
    pub categories: HashMap<String, usize>,
    /// Total number of handles across all categories.
    pub total_handles: usize,
}

/// RAII handle wrapper that automatically unregisters on drop.
#[derive(Debug)]
pub struct TrackedHandle {
    handle: u64,
    category: String,
    tracker: HandleTracker,
    unregistered: bool,
}

impl TrackedHandle {
    /// Create a new tracked handle.
    pub fn new(tracker: HandleTracker, category: &str, handle: u64) -> Self {
        tracker.register_handle(category, handle);
        Self {
            handle,
            category: category.to_string(),
            tracker,
            unregistered: false,
        }
    }

    /// Get the handle value.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Get the category.
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Manually unregister (useful for early cleanup).
    pub fn unregister(&mut self) -> bool {
        if !self.unregistered {
            self.unregistered = true;
            self.tracker.unregister_handle(&self.category, self.handle)
        } else {
            false
        }
    }
}

impl Drop for TrackedHandle {
    fn drop(&mut self) {
        self.unregister();
    }
}

/// Global handle tracker instance for convenience.
static GLOBAL_TRACKER: std::sync::OnceLock<HandleTracker> = std::sync::OnceLock::new();

/// Get the global handle tracker.
pub fn global_tracker() -> &'static HandleTracker {
    GLOBAL_TRACKER.get_or_init(HandleTracker::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_handle_tracker_basic_operations() {
        let tracker = HandleTracker::new();

        // Register some handles
        tracker.register_handle("tensor", 1);
        tracker.register_handle("tensor", 2);
        tracker.register_handle("sequence", 3);

        assert_eq!(tracker.category_handle_count("tensor"), 2);
        assert_eq!(tracker.category_handle_count("sequence"), 1);
        assert_eq!(tracker.total_handle_count(), 3);

        // Unregister a handle
        assert!(tracker.unregister_handle("tensor", 1));
        assert_eq!(tracker.category_handle_count("tensor"), 1);
        assert_eq!(tracker.total_handle_count(), 2);

        // Get handles
        let tensor_handles = tracker.get_handles("tensor");
        assert_eq!(tensor_handles, vec![2]);
    }

    #[test]
    fn test_tracked_handle_raii() {
        let tracker = HandleTracker::new();

        {
            let _tracked = TrackedHandle::new(tracker.clone(), "test", 42);
            assert_eq!(tracker.category_handle_count("test"), 1);
        }

        // Handle should be automatically unregistered
        assert_eq!(tracker.category_handle_count("test"), 0);
    }

    #[test]
    fn test_thread_safety() {
        let tracker = Arc::new(HandleTracker::new());
        let mut handles = vec![];

        // Spawn multiple threads that register handles
        for i in 0..10 {
            let tracker_clone = Arc::clone(&tracker);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let handle_id = (i * 10 + j) as u64;
                    tracker_clone.register_handle("concurrent", handle_id);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all handles were registered
        assert_eq!(tracker.category_handle_count("concurrent"), 100);
    }

    #[test]
    fn test_statistics() {
        let tracker = HandleTracker::new();

        tracker.register_handle("tensor", 1);
        tracker.register_handle("tensor", 2);
        tracker.register_handle("sequence", 3);
        tracker.register_handle("algorithm", 4);

        let stats = tracker.statistics();
        assert_eq!(stats.total_handles, 4);
        assert_eq!(stats.categories.get("tensor"), Some(&2));
        assert_eq!(stats.categories.get("sequence"), Some(&1));
        assert_eq!(stats.categories.get("algorithm"), Some(&1));
    }

    #[test]
    fn test_global_tracker() {
        let tracker1 = global_tracker();
        let tracker2 = global_tracker();

        // Should be the same instance
        assert!(Arc::ptr_eq(&tracker1.inner, &tracker2.inner));

        tracker1.register_handle("global", 999);
        assert_eq!(tracker2.category_handle_count("global"), 1);
    }
}
