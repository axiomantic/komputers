//! HandleTracker module statistics verification tests.
//!
//! These tests verify the accuracy and reliability of handle lifecycle
//! management, statistics calculation, concurrent access patterns,
//! and edge case handling in the HandleTracker system.

use kompute::handle_tracker::{global_tracker, HandleTracker, TrackedHandle};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

/// Test handle registration and unregistration statistics accuracy.
#[test]
fn test_handle_statistics_accuracy() {
    println!("Testing handle statistics accuracy...");

    let tracker = HandleTracker::new();

    // Initial state - should be empty
    let initial_stats = tracker.statistics();
    assert_eq!(initial_stats.total_handles, 0);
    assert!(initial_stats.categories.is_empty());

    // Register handles across multiple categories
    tracker.register_handle("tensor", 100);
    tracker.register_handle("tensor", 101);
    tracker.register_handle("tensor", 102);
    tracker.register_handle("sequence", 200);
    tracker.register_handle("sequence", 201);
    tracker.register_handle("algorithm", 300);
    tracker.register_handle("manager", 400);
    tracker.register_handle("manager", 401);
    tracker.register_handle("manager", 402);
    tracker.register_handle("manager", 403);

    // Verify statistics accuracy
    let stats = tracker.statistics();
    assert_eq!(stats.total_handles, 10);
    assert_eq!(stats.categories.len(), 4);
    assert_eq!(stats.categories.get("tensor"), Some(&3));
    assert_eq!(stats.categories.get("sequence"), Some(&2));
    assert_eq!(stats.categories.get("algorithm"), Some(&1));
    assert_eq!(stats.categories.get("manager"), Some(&4));

    // Verify individual category counts match statistics
    assert_eq!(tracker.category_handle_count("tensor"), 3);
    assert_eq!(tracker.category_handle_count("sequence"), 2);
    assert_eq!(tracker.category_handle_count("algorithm"), 1);
    assert_eq!(tracker.category_handle_count("manager"), 4);
    assert_eq!(tracker.total_handle_count(), 10);

    // Unregister some handles and verify statistics update
    assert!(tracker.unregister_handle("tensor", 101));
    assert!(tracker.unregister_handle("manager", 400));
    assert!(tracker.unregister_handle("manager", 401));

    let updated_stats = tracker.statistics();
    assert_eq!(updated_stats.total_handles, 7);
    assert_eq!(updated_stats.categories.get("tensor"), Some(&2));
    assert_eq!(updated_stats.categories.get("manager"), Some(&2));

    println!("  ✓ Handle statistics accuracy verified");
}

/// Test concurrent handle operations and statistics consistency.
#[test]
fn test_concurrent_statistics_consistency() {
    println!("Testing concurrent statistics consistency...");

    const THREAD_COUNT: usize = 16;
    const HANDLES_PER_THREAD: usize = 100;

    let tracker = Arc::new(HandleTracker::new());
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));
    let mut handles = Vec::new();

    // Phase 1: Concurrent registration
    for thread_id in 0..THREAD_COUNT {
        let tracker_clone = Arc::clone(&tracker);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let category = format!("thread_{}", thread_id % 4); // Use 4 categories
            for i in 0..HANDLES_PER_THREAD {
                let handle_id = (thread_id * HANDLES_PER_THREAD + i) as u64;
                tracker_clone.register_handle(&category, handle_id);
            }

            // Return some handles for later unregistration
            let start_handle = (thread_id * HANDLES_PER_THREAD) as u64;
            (start_handle..start_handle + 10).collect::<Vec<_>>()
        });
        handles.push(handle);
    }

    // Collect handles to unregister from each thread
    let mut handles_to_unregister = Vec::new();
    for handle in handles {
        let thread_handles = handle.join().unwrap();
        handles_to_unregister.extend(thread_handles);
    }

    // Verify total registration
    let stats_after_registration = tracker.statistics();
    assert_eq!(
        stats_after_registration.total_handles,
        THREAD_COUNT * HANDLES_PER_THREAD
    );
    assert_eq!(stats_after_registration.categories.len(), 4);

    // Each category should have handles from 4 threads (16 threads / 4 categories)
    let expected_per_category = (THREAD_COUNT / 4) * HANDLES_PER_THREAD;
    for i in 0..4 {
        let category = format!("thread_{}", i);
        assert_eq!(
            stats_after_registration.categories.get(&category),
            Some(&expected_per_category)
        );
    }

    // Phase 2: Concurrent unregistration
    let mut unregister_handles = Vec::new();
    for chunk in handles_to_unregister.chunks(THREAD_COUNT) {
        for (thread_id, &handle_id) in chunk.iter().enumerate() {
            let tracker_clone = Arc::clone(&tracker);
            let category = format!("thread_{}", thread_id % 4);

            let handle =
                thread::spawn(move || tracker_clone.unregister_handle(&category, handle_id));
            unregister_handles.push(handle);
        }
    }

    // Wait for unregistration to complete
    let mut successful_unregistrations = 0;
    for handle in unregister_handles {
        if handle.join().unwrap() {
            successful_unregistrations += 1;
        }
    }

    // Verify statistics consistency after unregistration
    let final_stats = tracker.statistics();
    let expected_remaining = (THREAD_COUNT * HANDLES_PER_THREAD) - successful_unregistrations;
    assert_eq!(final_stats.total_handles, expected_remaining);

    println!("  ✓ Concurrent statistics consistency verified");
    println!(
        "  • Registered: {} handles",
        THREAD_COUNT * HANDLES_PER_THREAD
    );
    println!("  • Unregistered: {} handles", successful_unregistrations);
    println!("  • Remaining: {} handles", expected_remaining);
}

/// Test TrackedHandle RAII behavior and statistics impact.
#[test]
fn test_tracked_handle_raii_statistics() {
    println!("Testing TrackedHandle RAII statistics...");

    let tracker = HandleTracker::new();

    // Create multiple tracked handles
    let mut tracked_handles = Vec::new();
    for i in 0..10 {
        let category = if i % 2 == 0 { "even" } else { "odd" };
        let tracked = TrackedHandle::new(tracker.clone(), category, i as u64);
        tracked_handles.push(tracked);
    }

    // Verify registration statistics
    let stats = tracker.statistics();
    assert_eq!(stats.total_handles, 10);
    assert_eq!(stats.categories.get("even"), Some(&5));
    assert_eq!(stats.categories.get("odd"), Some(&5));

    // Drop half the handles
    tracked_handles.truncate(5);

    // Verify automatic unregistration
    let stats_after_drop = tracker.statistics();
    assert_eq!(stats_after_drop.total_handles, 5);

    // Manual unregistration of remaining handles
    for mut tracked in tracked_handles {
        assert!(tracked.unregister());
        // Double unregistration should return false
        assert!(!tracked.unregister());
    }

    // Verify all handles are gone
    let final_stats = tracker.statistics();
    assert_eq!(final_stats.total_handles, 0);

    println!("  ✓ TrackedHandle RAII behavior verified");
}

/// Test edge cases and error conditions in statistics.
#[test]
fn test_statistics_edge_cases() {
    println!("Testing statistics edge cases...");

    let tracker = HandleTracker::new();

    // Test with empty tracker
    assert_eq!(tracker.total_handle_count(), 0);
    assert_eq!(tracker.category_handle_count("nonexistent"), 0);
    assert!(tracker.get_handles("nonexistent").is_empty());

    // Test unregistering nonexistent handles
    assert!(!tracker.unregister_handle("nonexistent", 999));
    assert!(!tracker.unregister_handle("tensor", 999));

    // Test clearing empty categories
    assert_eq!(tracker.clear_category("nonexistent"), 0);
    assert_eq!(tracker.clear_all(), 0);

    // Register some handles
    tracker.register_handle("test", 1);
    tracker.register_handle("test", 2);
    tracker.register_handle("other", 3);

    // Test clearing specific category
    let cleared_count = tracker.clear_category("test");
    assert_eq!(cleared_count, 2);
    assert_eq!(tracker.category_handle_count("test"), 0);
    assert_eq!(tracker.category_handle_count("other"), 1);

    // Test clearing all
    tracker.register_handle("test", 4);
    tracker.register_handle("test", 5);
    let total_cleared = tracker.clear_all();
    assert_eq!(total_cleared, 3); // 2 in test + 1 in other
    assert_eq!(tracker.total_handle_count(), 0);

    println!("  ✓ Statistics edge cases handled correctly");
}

/// Test duplicate handle registration behavior.
#[test]
fn test_duplicate_handle_statistics() {
    println!("Testing duplicate handle registration...");

    let tracker = HandleTracker::new();

    // Register the same handle multiple times in the same category
    tracker.register_handle("test", 42);
    tracker.register_handle("test", 42);
    tracker.register_handle("test", 42);

    // Should have 3 entries (duplicates allowed)
    assert_eq!(tracker.category_handle_count("test"), 3);
    let handles = tracker.get_handles("test");
    assert_eq!(handles, vec![42, 42, 42]);

    // Unregistering should remove only one instance
    assert!(tracker.unregister_handle("test", 42));
    assert_eq!(tracker.category_handle_count("test"), 2);

    // Register across different categories
    tracker.register_handle("other", 42);
    assert_eq!(tracker.total_handle_count(), 3); // 2 in test + 1 in other

    // Unregistering from wrong category should fail
    assert!(!tracker.unregister_handle("wrong", 42));
    assert_eq!(tracker.total_handle_count(), 3);

    println!("  ✓ Duplicate handle behavior verified");
}

/// Test global tracker statistics isolation and consistency.
#[test]
fn test_global_tracker_statistics() {
    println!("Testing global tracker statistics...");

    let global1 = global_tracker();
    let global2 = global_tracker();

    // Verify they're the same instance
    assert!(std::ptr::eq(global1, global2));

    // Clear any existing state
    global1.clear_all();

    // Register handles via different references
    global1.register_handle("global_test", 1000);
    global2.register_handle("global_test", 1001);

    // Statistics should be consistent across references
    let stats1 = global1.statistics();
    let stats2 = global2.statistics();
    assert_eq!(stats1.total_handles, stats2.total_handles);
    assert_eq!(stats1.categories, stats2.categories);
    assert_eq!(stats1.total_handles, 2);

    // Test concurrent access to global tracker
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let barrier_clone = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            let global = global_tracker();
            for i in 0..25 {
                global.register_handle("global_concurrent", (thread_id * 25 + i) as u64);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify global statistics after concurrent operations
    let final_stats = global_tracker().statistics();
    assert_eq!(final_stats.categories.get("global_concurrent"), Some(&100));
    assert!(final_stats.total_handles >= 102); // 100 + at least 2 from earlier

    println!("  ✓ Global tracker statistics verified");
}

/// Test handle ID generation and uniqueness statistics.
#[test]
fn test_id_generation_statistics() {
    println!("Testing ID generation statistics...");

    let tracker = HandleTracker::new();

    // Generate multiple IDs and verify uniqueness
    let mut generated_ids = Vec::new();
    for _ in 0..1000 {
        let id = tracker.generate_id();
        generated_ids.push(id);
    }

    // Verify all IDs are unique and sequential
    generated_ids.sort();
    for i in 1..generated_ids.len() {
        assert_ne!(
            generated_ids[i],
            generated_ids[i - 1],
            "IDs should be unique"
        );
    }

    // Verify they start from 1 and are sequential
    assert_eq!(generated_ids[0], 1);
    assert_eq!(generated_ids[999], 1000);

    // Test concurrent ID generation
    let barrier = Arc::new(Barrier::new(8));
    let tracker = Arc::new(HandleTracker::new());
    let mut handles = Vec::new();

    for _ in 0..8 {
        let tracker_clone = Arc::clone(&tracker);
        let barrier_clone = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            let mut thread_ids = Vec::new();
            for _ in 0..125 {
                thread_ids.push(tracker_clone.generate_id());
            }
            thread_ids
        });
        handles.push(handle);
    }

    let mut all_concurrent_ids = Vec::new();
    for handle in handles {
        let thread_ids = handle.join().unwrap();
        all_concurrent_ids.extend(thread_ids);
    }

    // Verify concurrent IDs are unique
    all_concurrent_ids.sort();
    for i in 1..all_concurrent_ids.len() {
        assert_ne!(
            all_concurrent_ids[i],
            all_concurrent_ids[i - 1],
            "Concurrent IDs should be unique"
        );
    }

    assert_eq!(all_concurrent_ids.len(), 1000);

    println!("  ✓ ID generation statistics verified");
    println!(
        "  • Generated {} unique IDs sequentially",
        generated_ids.len()
    );
    println!(
        "  • Generated {} unique IDs concurrently",
        all_concurrent_ids.len()
    );
}

/// Comprehensive performance and statistics stress test.
#[test]
fn test_handle_tracker_performance_statistics() {
    println!("Testing handle tracker performance statistics...");

    let tracker = HandleTracker::new();
    let start_time = Instant::now();

    // Large-scale operations
    const OPERATIONS: usize = 10_000;
    const CATEGORIES: usize = 50;

    // Mass registration
    for i in 0..OPERATIONS {
        let category = format!("perf_test_{}", i % CATEGORIES);
        tracker.register_handle(&category, i as u64);
    }

    let registration_time = start_time.elapsed();

    // Verify statistics after mass registration
    let stats = tracker.statistics();
    assert_eq!(stats.total_handles, OPERATIONS);
    assert_eq!(stats.categories.len(), CATEGORIES);

    // Each category should have approximately OPERATIONS/CATEGORIES handles
    let expected_per_category = OPERATIONS / CATEGORIES;
    for i in 0..CATEGORIES {
        let category = format!("perf_test_{}", i);
        let count = stats.categories.get(&category).unwrap_or(&0);
        assert_eq!(*count, expected_per_category);
    }

    // Mass unregistration
    let unregister_start = Instant::now();
    for i in (0..OPERATIONS).step_by(2) {
        // Remove every other handle
        let category = format!("perf_test_{}", i % CATEGORIES);
        tracker.unregister_handle(&category, i as u64);
    }
    let unregistration_time = unregister_start.elapsed();

    // Verify statistics after mass unregistration
    let final_stats = tracker.statistics();
    assert_eq!(final_stats.total_handles, OPERATIONS / 2);

    println!("  ✓ Performance statistics verified");
    println!(
        "  • Registration time: {:?} for {} operations",
        registration_time, OPERATIONS
    );
    println!(
        "  • Unregistration time: {:?} for {} operations",
        unregistration_time,
        OPERATIONS / 2
    );
    println!(
        "  • Operations per second: {:.0}",
        OPERATIONS as f64 / registration_time.as_secs_f64()
    );
}
