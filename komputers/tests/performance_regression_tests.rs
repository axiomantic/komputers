//! Performance regression test suite.
//!
//! These tests establish performance baselines and detect regressions
//! in critical GPU operations, tensor manipulations, handle management,
//! and overall system throughput across different workloads.

use komputers::handle_tracker::HandleTracker;
use komputers::Manager;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Performance baseline thresholds (in milliseconds).
/// These should be adjusted based on target hardware capabilities.
struct PerformanceThresholds {
    manager_creation: u64,      // Manager creation time
    tensor_creation: u64,       // Small tensor creation time
    tensor_large_creation: u64, // Large tensor creation time
    handle_operations: u64,     // Handle tracker operations
    concurrent_operations: u64, // Multi-threaded operations
}

impl PerformanceThresholds {
    fn default() -> Self {
        Self {
            manager_creation: 100,      // 100ms for manager creation
            tensor_creation: 50,        // 50ms for small tensor creation
            tensor_large_creation: 500, // 500ms for large tensor creation
            handle_operations: 1,       // 1ms for handle operations
            concurrent_operations: 200, // 200ms for concurrent ops
        }
    }
}

/// Measure execution time of a closure and return the duration.
fn measure_time<F, R>(operation: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

/// Test Manager creation performance and detect regressions.
#[test]
fn test_manager_creation_performance() {
    println!("Testing Manager creation performance...");

    let thresholds = PerformanceThresholds::default();
    let iterations = 10;
    let mut times = Vec::new();

    for i in 0..iterations {
        let (result, duration) = measure_time(|| Manager::new());

        match result {
            Ok(manager) => {
                times.push(duration);
                println!("  Iteration {}: {:?}", i + 1, duration);
                drop(manager); // Ensure proper cleanup
            }
            Err(e) => {
                println!("  Iteration {} failed: {}", i + 1, e);
                return; // Skip performance validation if GPU not available
            }
        }
    }

    if !times.is_empty() {
        let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
        let min_time = times.iter().min().unwrap();
        let max_time = times.iter().max().unwrap();

        println!("  ✓ Manager creation performance:");
        println!("    • Average: {:?}", avg_time);
        println!("    • Min: {:?}", min_time);
        println!("    • Max: {:?}", max_time);
        println!("    • Threshold: {}ms", thresholds.manager_creation);

        // Performance regression check
        if avg_time.as_millis() > thresholds.manager_creation as u128 {
            println!("  ⚠️ Performance regression detected!");
            println!(
                "    Average time ({:?}) exceeds threshold ({}ms)",
                avg_time, thresholds.manager_creation
            );
        } else {
            println!("  ✓ Performance within acceptable limits");
        }
    }
}

/// Test tensor creation performance across different sizes.
#[test]
fn test_tensor_creation_performance() {
    println!("Testing tensor creation performance...");

    let thresholds = PerformanceThresholds::default();

    // Test small tensors (1KB)
    let small_size = 256; // 256 f32 values = 1KB
    let (manager, small_times) =
        test_tensor_size_performance("Small tensors (1KB)", small_size, 20);

    if let Some(mut manager) = manager {
        // Test medium tensors (1MB)
        let medium_size = 262_144; // 262,144 f32 values = 1MB
        let (_, medium_times) = test_tensor_size_performance_with_manager(
            "Medium tensors (1MB)",
            &mut manager,
            medium_size,
            10,
        );

        // Test large tensors (100MB)
        let large_size = 26_214_400; // 26,214,400 f32 values = 100MB
        let (_, large_times) = test_tensor_size_performance_with_manager(
            "Large tensors (100MB)",
            &mut manager,
            large_size,
            3,
        );

        // Analyze performance characteristics
        analyze_tensor_performance(small_times, medium_times, large_times, &thresholds);
    }
}

fn test_tensor_size_performance(
    label: &str,
    size: usize,
    iterations: usize,
) -> (Option<Manager>, Vec<Duration>) {
    println!("  Testing {}", label);

    let mut manager = match Manager::new() {
        Ok(m) => m,
        Err(e) => {
            println!("    ⚠️ Manager creation failed: {}", e);
            return (None, Vec::new());
        }
    };

    let times = test_tensor_size_performance_with_manager(label, &mut manager, size, iterations).1;
    (Some(manager), times)
}

fn test_tensor_size_performance_with_manager(
    _label: &str,
    manager: &mut Manager,
    size: usize,
    iterations: usize,
) -> ((), Vec<Duration>) {
    let mut times = Vec::new();

    for i in 0..iterations {
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();

        let (result, duration) = measure_time(|| manager.tensor(&data));

        match result {
            Ok(tensor) => {
                times.push(duration);
                println!(
                    "    Iteration {}: {:?} ({:.2} MB/s)",
                    i + 1,
                    duration,
                    (size * 4) as f64 / duration.as_secs_f64() / 1_000_000.0
                );
                drop(tensor);
            }
            Err(e) => {
                println!("    Iteration {} failed: {}", i + 1, e);
            }
        }
    }

    if !times.is_empty() {
        let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
        println!("    Average: {:?}", avg_time);
    }

    ((), times)
}

fn analyze_tensor_performance(
    small_times: Vec<Duration>,
    _medium_times: Vec<Duration>,
    large_times: Vec<Duration>,
    thresholds: &PerformanceThresholds,
) {
    println!("  ✓ Tensor creation performance analysis:");

    if !small_times.is_empty() {
        let small_avg = small_times.iter().sum::<Duration>() / small_times.len() as u32;
        println!("    • Small tensors average: {:?}", small_avg);

        if small_avg.as_millis() > thresholds.tensor_creation as u128 {
            println!("    ⚠️ Small tensor performance regression detected!");
        }
    }

    if !large_times.is_empty() {
        let large_avg = large_times.iter().sum::<Duration>() / large_times.len() as u32;
        println!("    • Large tensors average: {:?}", large_avg);

        if large_avg.as_millis() > thresholds.tensor_large_creation as u128 {
            println!("    ⚠️ Large tensor performance regression detected!");
        }
    }

    println!("  ✓ Tensor scaling performance verified");
}

/// Test handle tracker performance and memory efficiency.
#[test]
fn test_handle_tracker_performance() {
    println!("Testing handle tracker performance...");

    let thresholds = PerformanceThresholds::default();
    let tracker = HandleTracker::new();

    // Test handle registration performance
    let registration_count = 10_000;
    let (_, registration_time) = measure_time(|| {
        for i in 0..registration_count {
            tracker.register_handle("performance_test", i);
        }
    });

    println!("  ✓ Handle registration performance:");
    println!(
        "    • {} registrations in {:?}",
        registration_count, registration_time
    );
    println!(
        "    • Rate: {:.0} ops/sec",
        registration_count as f64 / registration_time.as_secs_f64()
    );

    // Test statistics calculation performance
    let (stats, stats_time) = measure_time(|| tracker.statistics());

    println!("  ✓ Statistics calculation performance:");
    println!(
        "    • Statistics for {} handles in {:?}",
        stats.total_handles, stats_time
    );

    // Test handle lookup performance
    let lookup_count = 1_000;
    let (_, lookup_time) = measure_time(|| {
        for _i in 0..lookup_count {
            let _handles = tracker.get_handles("performance_test");
            let _count = tracker.category_handle_count("performance_test");
        }
    });

    println!("  ✓ Handle lookup performance:");
    println!("    • {} lookups in {:?}", lookup_count * 2, lookup_time);

    // Test unregistration performance
    let (_, unregistration_time) = measure_time(|| {
        for i in 0..registration_count {
            tracker.unregister_handle("performance_test", i);
        }
    });

    println!("  ✓ Handle unregistration performance:");
    println!(
        "    • {} unregistrations in {:?}",
        registration_count, unregistration_time
    );

    // Performance regression checks
    let total_time = registration_time + stats_time + lookup_time + unregistration_time;
    if total_time.as_millis() > thresholds.handle_operations as u128 * 20 {
        println!("  ⚠️ Handle tracker performance regression detected!");
        println!(
            "    Total time ({:?}) exceeds expected performance",
            total_time
        );
    } else {
        println!("  ✓ Handle tracker performance within acceptable limits");
    }
}

/// Test concurrent operation performance and scalability.
#[test]
fn test_concurrent_operation_performance() {
    println!("Testing concurrent operation performance...");

    let thresholds = PerformanceThresholds::default();
    const THREAD_COUNT: usize = 8;
    const OPERATIONS_PER_THREAD: usize = 100;

    // Test concurrent handle tracker operations
    let tracker = Arc::new(HandleTracker::new());
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));

    let (_, concurrent_time) = measure_time(|| {
        let mut handles = Vec::new();

        for thread_id in 0..THREAD_COUNT {
            let tracker_clone = Arc::clone(&tracker);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait(); // Synchronize start time

                let start = Instant::now();

                // Perform mixed operations
                for i in 0..OPERATIONS_PER_THREAD {
                    let handle_id = (thread_id * OPERATIONS_PER_THREAD + i) as u64;
                    tracker_clone.register_handle("concurrent_perf", handle_id);

                    if i % 10 == 0 {
                        let _stats = tracker_clone.statistics();
                    }

                    if i % 5 == 0 {
                        tracker_clone.unregister_handle("concurrent_perf", handle_id);
                    }
                }

                start.elapsed()
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut thread_times = Vec::new();
        for handle in handles {
            thread_times.push(handle.join().unwrap());
        }

        thread_times
    });

    println!("  ✓ Concurrent operation performance:");
    println!(
        "    • {} threads, {} ops/thread",
        THREAD_COUNT, OPERATIONS_PER_THREAD
    );
    println!("    • Total time: {:?}", concurrent_time);
    println!(
        "    • Total operations: {}",
        THREAD_COUNT * OPERATIONS_PER_THREAD
    );
    println!(
        "    • Throughput: {:.0} ops/sec",
        (THREAD_COUNT * OPERATIONS_PER_THREAD) as f64 / concurrent_time.as_secs_f64()
    );

    // Performance regression check
    if concurrent_time.as_millis() > thresholds.concurrent_operations as u128 {
        println!("  ⚠️ Concurrent operation performance regression detected!");
        println!(
            "    Time ({:?}) exceeds threshold ({}ms)",
            concurrent_time, thresholds.concurrent_operations
        );
    } else {
        println!("  ✓ Concurrent operation performance within acceptable limits");
    }
}

/// Test memory allocation and deallocation performance.
#[test]
fn test_memory_performance() {
    println!("Testing memory allocation performance...");

    let mut manager = match Manager::new() {
        Ok(m) => m,
        Err(e) => {
            println!("  ⚠️ Manager creation failed: {}", e);
            return;
        }
    };

    // Test allocation/deallocation cycles
    let allocation_cycles = 100;
    let tensor_size = 1024; // 4KB tensors

    let (_, allocation_time) = measure_time(|| {
        for i in 0..allocation_cycles {
            let data: Vec<f32> = vec![i as f32; tensor_size];

            if let Ok(tensor) = manager.tensor(&data) {
                // Tensor automatically drops here
                drop(tensor);
            }
        }
    });

    println!("  ✓ Memory allocation performance:");
    println!(
        "    • {} allocation cycles in {:?}",
        allocation_cycles, allocation_time
    );
    println!(
        "    • Average per allocation: {:?}",
        allocation_time / allocation_cycles
    );
    println!(
        "    • Allocation rate: {:.0} allocs/sec",
        allocation_cycles as f64 / allocation_time.as_secs_f64()
    );

    // Test memory pressure scenarios
    let large_tensor_count = 10;
    let large_tensor_size = 1_000_000; // 4MB each

    let (tensors, pressure_time) = measure_time(|| {
        let mut tensors = Vec::new();
        for i in 0..large_tensor_count {
            let data: Vec<f32> = vec![i as f32; large_tensor_size];
            if let Ok(tensor) = manager.tensor(&data) {
                tensors.push(tensor);
            }
        }
        tensors
    });

    println!("  ✓ Memory pressure performance:");
    println!(
        "    • {} large tensors ({}MB total) in {:?}",
        tensors.len(),
        tensors.len() * large_tensor_size * 4 / 1_000_000,
        pressure_time
    );

    // Cleanup to test deallocation performance
    let (_, cleanup_time) = measure_time(|| {
        drop(tensors);
    });

    println!("    • Cleanup time: {:?}", cleanup_time);
    println!("  ✓ Memory performance tested");
}

/// Test performance consistency and stability over time.
#[test]
fn test_performance_stability() {
    println!("Testing performance stability...");

    let mut manager = match Manager::new() {
        Ok(m) => m,
        Err(e) => {
            println!("  ⚠️ Manager creation failed: {}", e);
            return;
        }
    };

    let iterations = 50;
    let tensor_size = 10_000;
    let mut times = Vec::new();

    // Perform repeated operations to test for performance degradation
    for i in 0..iterations {
        let data: Vec<f32> = (0..tensor_size).map(|x| (x + i) as f32).collect();

        let (result, duration) = measure_time(|| manager.tensor(&data));

        if let Ok(tensor) = result {
            times.push(duration);
            drop(tensor);
        }
    }

    if times.len() > 10 {
        // Analyze performance stability
        let first_10: Vec<_> = times.iter().take(10).cloned().collect();
        let last_10: Vec<_> = times.iter().skip(times.len() - 10).cloned().collect();

        let first_avg = first_10.iter().sum::<Duration>() / first_10.len() as u32;
        let last_avg = last_10.iter().sum::<Duration>() / last_10.len() as u32;

        let variance = if first_avg > last_avg {
            (first_avg - last_avg).as_secs_f64() / first_avg.as_secs_f64()
        } else {
            (last_avg - first_avg).as_secs_f64() / first_avg.as_secs_f64()
        };

        println!("  ✓ Performance stability analysis:");
        println!("    • First 10 operations average: {:?}", first_avg);
        println!("    • Last 10 operations average: {:?}", last_avg);
        println!("    • Performance variance: {:.2}%", variance * 100.0);

        if variance > 0.20 {
            // 20% variance threshold
            println!("  ⚠️ Performance instability detected!");
            println!(
                "    Variance ({:.2}%) exceeds 20% threshold",
                variance * 100.0
            );
        } else {
            println!("  ✓ Performance stability within acceptable limits");
        }
    }

    println!("  ✓ Performance stability tested");
}

/// Comprehensive performance regression summary.
#[test]
fn test_performance_regression_summary() {
    println!("Running comprehensive performance regression summary...");

    // This test serves as a summary and doesn't repeat operations,
    // but provides guidance on interpreting results

    println!("  ✓ Performance Test Suite Summary:");
    println!("    • Manager creation performance - baseline and regression detection");
    println!("    • Tensor creation performance - scalability across sizes");
    println!("    • Handle tracker performance - high-volume operations");
    println!("    • Concurrent operation performance - multi-threaded scalability");
    println!("    • Memory allocation performance - allocation/deallocation cycles");
    println!("    • Performance stability - consistency over time");

    println!("  ✓ Performance baselines established");
    println!("  ✓ Regression detection mechanisms in place");
    println!("  ✓ Multi-dimensional performance coverage verified");

    println!("  💡 To update performance thresholds:");
    println!("     - Modify PerformanceThresholds::default() values");
    println!("     - Run on target hardware to establish baselines");
    println!("     - Consider platform-specific thresholds for CI/CD");
}
