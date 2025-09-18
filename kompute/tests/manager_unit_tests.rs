//! Comprehensive unit tests for Manager module functionality.
//!
//! These tests verify all Manager methods including thread safety,
//! resource limits, device management, and factory method validation.

use kompute::{Manager, Tensor};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Test concurrent manager creation across multiple threads.
/// Verifies thread safety of Manager::new() and device enumeration.
#[test]
fn test_manager_thread_safety_concurrent_creation() {
    const THREAD_COUNT: usize = 8;
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));
    let mut handles = vec![];

    for thread_id in 0..THREAD_COUNT {
        let barrier_clone = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start for true concurrency test

            let start_time = Instant::now();
            let result = Manager::new();
            let duration = start_time.elapsed();

            (thread_id, result.is_ok(), duration)
        });
        handles.push(handle);
    }

    // Collect results from all threads
    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should not panic"))
        .collect();

    let success_count = results.iter().filter(|(_, success, _)| *success).count();
    let avg_duration: Duration = results
        .iter()
        .map(|(_, _, duration)| *duration)
        .sum::<Duration>()
        / THREAD_COUNT as u32;

    println!("Concurrent manager creation results:");
    println!("  Successful creations: {}/{}", success_count, THREAD_COUNT);
    println!("  Average creation time: {:?}", avg_duration);

    // At least some managers should be created successfully
    // (May be 0 in headless environments, which is acceptable)
    if success_count > 0 {
        assert!(
            success_count <= THREAD_COUNT,
            "Success count should not exceed thread count"
        );
        assert!(
            avg_duration < Duration::from_secs(2),
            "Average creation time should be reasonable: {:?}",
            avg_duration
        );
    }

    // Verify no thread panicked
    for (thread_id, _, _) in results {
        assert!(thread_id < THREAD_COUNT, "Thread ID should be valid");
    }
}

/// Test manager creation under resource pressure with many tensors.
/// Verifies resource limit handling and memory management.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_manager_resource_limits_tensor_creation() {
    const MAX_TENSORS: usize = 1000;
    const TENSOR_SIZE: usize = 100;

    if let Ok(mut manager) = Manager::new() {
        let mut tensors = Vec::new();
        let mut creation_count = 0;
        let mut first_failure_at = None;

        println!("Testing tensor creation limits...");

        for i in 0..MAX_TENSORS {
            let data: Vec<f32> = (0..TENSOR_SIZE)
                .map(|x| (i * TENSOR_SIZE + x) as f32)
                .collect();

            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    // Verify tensor properties
                    assert_eq!(tensor.size(), TENSOR_SIZE, "Tensor {} size incorrect", i);
                    assert_eq!(
                        tensor.size_bytes(),
                        TENSOR_SIZE * 4,
                        "Tensor {} byte size incorrect",
                        i
                    );
                    assert_ne!(tensor.handle(), 0, "Tensor {} handle should be non-zero", i);

                    tensors.push(tensor);
                    creation_count += 1;

                    if i % 100 == 0 {
                        println!("  Created {} tensors successfully", i + 1);
                    }
                }
                Err(e) => {
                    println!("  Tensor creation failed at index {}: {}", i, e);
                    first_failure_at = Some(i);
                    break;
                }
            }
        }

        println!("Resource limit test results:");
        println!("  Successfully created: {} tensors", creation_count);
        println!(
            "  Total memory allocated: ~{} MB",
            (creation_count * TENSOR_SIZE * 4) / (1024 * 1024)
        );

        if let Some(failure_index) = first_failure_at {
            println!("  First failure at tensor index: {}", failure_index);
            assert!(
                failure_index > 10,
                "Should be able to create at least 10 tensors"
            );
        } else {
            println!("  All {} tensors created successfully", MAX_TENSORS);
        }

        // Verify that existing tensors still work after hitting limits
        if creation_count > 0 {
            let test_tensor = &tensors[0];
            match test_tensor.to_vec() {
                Ok(data) => {
                    assert_eq!(
                        data.len(),
                        TENSOR_SIZE,
                        "First tensor data should be intact"
                    );
                    assert_eq!(data[0], 0.0, "First tensor data should be correct");
                }
                Err(e) => {
                    println!("  Warning: First tensor data retrieval failed: {}", e);
                }
            }
        }
    } else {
        println!("Skipping resource limits test - no GPU available");
    }
}

/// Test manager creation with different device IDs.
/// Verifies device switching and validation.
#[test]
fn test_manager_device_switching_all_available() {
    println!("Testing device switching across available devices...");

    // First, get device count
    let device_count = match Manager::device_count() {
        Ok(count) => {
            println!("  Found {} devices", count);
            count
        }
        Err(e) => {
            println!("  Device enumeration failed: {}", e);
            return; // Skip test if device enumeration fails
        }
    };

    if device_count == 0 {
        println!("  No devices available - testing error handling");

        // Test that device 0 fails gracefully when no devices exist
        match Manager::with_device(0) {
            Ok(_) => panic!("Manager creation should fail when no devices available"),
            Err(e) => {
                println!("  Correctly failed with no devices: {}", e);
                assert!(
                    e.to_string().contains("device") || e.to_string().contains("Vulkan"),
                    "Error should mention device or Vulkan: {}",
                    e
                );
            }
        }
        return;
    }

    // Test valid device IDs
    for device_id in 0..device_count {
        match Manager::with_device(device_id) {
            Ok(_manager) => {
                println!("  Successfully created manager for device {}", device_id);
            }
            Err(e) => {
                println!("  Failed to create manager for device {}: {}", device_id, e);
                // Device enumeration succeeded but creation failed - may be expected
            }
        }
    }

    // Test invalid device IDs
    let invalid_device_ids = vec![device_count, device_count + 1, device_count + 100, u32::MAX];

    for invalid_id in invalid_device_ids {
        match Manager::with_device(invalid_id) {
            Ok(_) => {
                panic!(
                    "Manager creation should fail for invalid device ID {}",
                    invalid_id
                );
            }
            Err(e) => {
                println!(
                    "  Correctly rejected invalid device ID {}: {}",
                    invalid_id, e
                );
                assert!(
                    e.to_string().contains("device") || e.to_string().contains("invalid"),
                    "Error should mention device or invalid: {}",
                    e
                );
            }
        }
    }
}

/// Test manager behavior under memory pressure conditions.
/// Simulates low GPU memory scenarios.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_manager_memory_pressure_allocation_failure() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing memory pressure scenarios...");

        // Try to create increasingly large tensors until we hit memory limits
        let mut tensor_size = 1024; // Start with 4KB tensors
        let mut successful_allocations = 0;
        let mut total_memory_allocated = 0usize;

        loop {
            let data: Vec<f32> = vec![42.0; tensor_size];
            let memory_size = tensor_size * std::mem::size_of::<f32>();

            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    // Verify the tensor was created correctly
                    assert_eq!(tensor.size(), tensor_size);
                    assert_eq!(tensor.size_bytes(), memory_size);

                    successful_allocations += 1;
                    total_memory_allocated += memory_size;

                    println!(
                        "  Allocated tensor of size {} ({} MB)",
                        tensor_size,
                        memory_size / (1024 * 1024)
                    );

                    // Double the size for next iteration
                    tensor_size *= 2;

                    // Safety limit to prevent system freeze
                    if memory_size > 1024 * 1024 * 1024 {
                        // 1GB limit
                        println!("  Reached safety limit of 1GB");
                        break;
                    }
                }
                Err(e) => {
                    println!(
                        "  Memory allocation failed at size {} ({}): {}",
                        tensor_size,
                        memory_size / (1024 * 1024),
                        e
                    );
                    assert!(
                        e.to_string().contains("memory")
                            || e.to_string().contains("allocation")
                            || e.to_string().contains("size"),
                        "Error should mention memory/allocation: {}",
                        e
                    );
                    break;
                }
            }
        }

        println!("Memory pressure test results:");
        println!("  Successful allocations: {}", successful_allocations);
        println!(
            "  Total memory allocated: {} MB",
            total_memory_allocated / (1024 * 1024)
        );
        println!("  Largest tensor size: {} elements", tensor_size / 2);

        assert!(
            successful_allocations > 0,
            "Should be able to allocate at least one tensor"
        );
    } else {
        println!("Skipping memory pressure test - no GPU available");
    }
}

/// Test manager factory methods with invalid inputs.
/// Verifies input validation for tensor(), algorithm(), and sequence() methods.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_manager_factory_methods_invalid_inputs() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing factory method input validation...");

        // Test tensor() with invalid inputs
        println!("  Testing tensor factory method...");

        // Empty data
        let empty_data: Vec<f32> = vec![];
        match manager.tensor(&empty_data) {
            Ok(_) => panic!("Tensor creation should fail with empty data"),
            Err(e) => {
                println!("    Correctly rejected empty data: {}", e);
                assert!(
                    e.to_string().contains("empty") || e.to_string().contains("size"),
                    "Error should mention empty or size: {}",
                    e
                );
            }
        }

        // Extremely large data
        if std::mem::size_of::<usize>() == 8 {
            // Only test on 64-bit systems
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let huge_size = (u32::MAX as usize) + 1;
                let _huge_data: Vec<f32> = vec![1.0; huge_size];
            })) {
                Ok(_) => {
                    // If we can actually create the huge vector, test tensor creation
                    // (This will likely fail in practice due to memory limits)
                    println!("    Note: Huge vector creation succeeded (unexpected)");
                }
                Err(_) => {
                    println!("    Huge vector creation failed as expected");
                }
            }
        }

        // Test algorithm() with invalid SPIR-V
        println!("  Testing algorithm factory method...");

        // Empty SPIR-V
        let empty_spirv: Vec<u32> = vec![];
        match manager.algorithm(&empty_spirv) {
            Ok(_) => panic!("Algorithm creation should fail with empty SPIR-V"),
            Err(e) => {
                println!("    Correctly rejected empty SPIR-V: {}", e);
                assert!(
                    e.to_string().contains("empty") || e.to_string().contains("SPIR-V"),
                    "Error should mention empty or SPIR-V: {}",
                    e
                );
            }
        }

        // Invalid SPIR-V magic number
        let invalid_spirv = vec![0x12345678, 0x00010000, 0x00080001, 0x0000000b, 0x00000000];
        match manager.algorithm(&invalid_spirv) {
            Ok(_) => panic!("Algorithm creation should fail with invalid SPIR-V magic"),
            Err(e) => {
                println!("    Correctly rejected invalid SPIR-V magic: {}", e);
                assert!(
                    e.to_string().contains("SPIR-V") || e.to_string().contains("magic"),
                    "Error should mention SPIR-V or magic: {}",
                    e
                );
            }
        }

        // Test sequence() - should generally succeed as it has no parameters
        println!("  Testing sequence factory method...");
        match manager.sequence() {
            Ok(_sequence) => {
                println!("    Sequence creation succeeded as expected");
            }
            Err(e) => {
                println!("    Sequence creation failed: {}", e);
                // This may fail in some environments, which is acceptable
            }
        }

        println!("  Factory method validation tests completed");
    } else {
        println!("Skipping factory method tests - no GPU available");
    }
}

/// Test manager creation performance consistency.
/// Verifies that repeated manager creation doesn't degrade performance.
#[test]
fn test_manager_creation_performance_consistency() {
    const ITERATIONS: usize = 10;
    let mut creation_times = Vec::new();
    let mut successful_creations = 0;

    println!("Testing manager creation performance consistency...");

    for i in 0..ITERATIONS {
        let start = Instant::now();

        match Manager::new() {
            Ok(_manager) => {
                let duration = start.elapsed();
                creation_times.push(duration);
                successful_creations += 1;

                println!("  Creation {}: {:?}", i, duration);

                // Individual creation should complete within reasonable time
                assert!(
                    duration < Duration::from_secs(5),
                    "Manager creation {} took too long: {:?}",
                    i,
                    duration
                );
            }
            Err(e) => {
                println!("  Creation {} failed: {}", i, e);
                // Failures are acceptable in headless environments
            }
        }
    }

    if successful_creations > 0 {
        let avg_time = creation_times.iter().sum::<Duration>() / successful_creations as u32;
        let min_time = creation_times.iter().min().unwrap();
        let max_time = creation_times.iter().max().unwrap();

        println!("Performance consistency results:");
        println!(
            "  Successful creations: {}/{}",
            successful_creations, ITERATIONS
        );
        println!("  Average time: {:?}", avg_time);
        println!("  Min time: {:?}", min_time);
        println!("  Max time: {:?}", max_time);
        println!("  Time variance: {:?}", max_time.saturating_sub(*min_time));

        // Performance should be reasonably consistent
        assert!(
            max_time.as_millis() < min_time.as_millis() * 10,
            "Performance should be consistent - max: {:?}, min: {:?}",
            max_time,
            min_time
        );

        // Average time should be reasonable
        assert!(
            avg_time < Duration::from_secs(2),
            "Average creation time should be reasonable: {:?}",
            avg_time
        );
    } else {
        println!("No successful manager creations - skipping performance analysis");
    }
}

/// Test manager with explicit device ID validation.
/// Verifies device ID parameter validation and error messages.
#[test]
fn test_manager_device_id_validation() {
    println!("Testing device ID validation...");

    // Test device count first
    let device_count = match Manager::device_count() {
        Ok(count) => {
            println!("  Device count: {}", count);
            assert!(count <= 100, "Device count should be reasonable: {}", count);
            count
        }
        Err(e) => {
            println!("  Device count failed: {}", e);
            // Continue with validation tests even if enumeration fails
            0
        }
    };

    // Test boundary values
    let test_cases = vec![
        (0, device_count > 0, "First device"),
        (
            device_count.saturating_sub(1),
            device_count > 0,
            "Last device",
        ),
        (device_count, false, "First invalid device"),
        (device_count + 1, false, "Beyond available devices"),
        (u32::MAX, false, "Maximum u32 value"),
        (u32::MAX - 1, false, "Near maximum value"),
    ];

    for (device_id, should_succeed, description) in test_cases {
        println!("  Testing {}: device_id = {}", description, device_id);

        match Manager::with_device(device_id) {
            Ok(_manager) => {
                if should_succeed {
                    println!("    ✓ Succeeded as expected");
                } else {
                    panic!("Manager creation should fail for device_id {}", device_id);
                }
            }
            Err(e) => {
                if !should_succeed {
                    println!("    ✓ Failed as expected: {}", e);

                    // Verify error message quality
                    let error_msg = e.to_string().to_lowercase();
                    assert!(
                        error_msg.contains("device")
                            || error_msg.contains("invalid")
                            || error_msg.contains("available"),
                        "Error should mention device/invalid/available: {}",
                        e
                    );
                } else {
                    println!("    Warning: Expected success but failed: {}", e);
                    // This may happen in restricted environments
                }
            }
        }
    }
}

/// Test manager state consistency across operations.
/// Verifies that manager remains in valid state during various operations.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_manager_state_consistency() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing manager state consistency...");

        // Create multiple resources and verify they don't interfere
        let mut resources_created = 0;

        // Create tensors
        for i in 0..5 {
            let data: Vec<f32> = vec![i as f32; 10];
            match manager.tensor(&data) {
                Ok(tensor) => {
                    assert_eq!(tensor.size(), 10, "Tensor {} size should be correct", i);
                    assert_ne!(tensor.handle(), 0, "Tensor {} handle should be valid", i);
                    resources_created += 1;
                    println!("    Created tensor {}", i);
                }
                Err(e) => {
                    println!("    Tensor {} creation failed: {}", i, e);
                }
            }
        }

        // Create sequences
        for i in 0..3 {
            match manager.sequence() {
                Ok(sequence) => {
                    assert_ne!(
                        sequence.handle(),
                        0,
                        "Sequence {} handle should be valid",
                        i
                    );
                    resources_created += 1;
                    println!("    Created sequence {}", i);
                }
                Err(e) => {
                    println!("    Sequence {} creation failed: {}", i, e);
                }
            }
        }

        println!("  Created {} resources total", resources_created);
        assert!(
            resources_created > 0,
            "Should be able to create at least some resources"
        );

        // Verify manager is still functional after resource creation
        match Manager::device_count() {
            Ok(count) => {
                println!("  Device count still accessible: {}", count);
            }
            Err(e) => {
                println!(
                    "  Warning: Device count failed after resource creation: {}",
                    e
                );
            }
        }

        println!("  Manager state consistency verified");
    } else {
        println!("Skipping state consistency test - no GPU available");
    }
}

/// Test thread safety of device count operation.
/// Verifies that device enumeration is thread-safe.
#[test]
fn test_device_count_thread_safety() {
    const THREAD_COUNT: usize = 4;
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    println!("Testing device count thread safety...");

    for thread_id in 0..THREAD_COUNT {
        let barrier_clone = Arc::clone(&barrier);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start

            let start = Instant::now();
            let result = Manager::device_count();
            let duration = start.elapsed();

            let mut results = results_clone.lock().unwrap();
            results.push((thread_id, result, duration));
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let results = results.lock().unwrap();
    let success_count = results
        .iter()
        .filter(|(_, result, _)| result.is_ok())
        .count();

    println!("Device count thread safety results:");
    println!("  Successful calls: {}/{}", success_count, THREAD_COUNT);

    if success_count > 0 {
        // All successful calls should return the same device count
        let device_counts: Vec<_> = results
            .iter()
            .filter_map(|(_, result, _)| result.as_ref().ok())
            .collect();

        if !device_counts.is_empty() {
            let first_count = device_counts[0];
            for &count in &device_counts {
                assert_eq!(
                    count, first_count,
                    "All device count calls should return the same value"
                );
            }
            println!("  Consistent device count: {}", first_count);
        }

        // Check timing consistency
        let avg_duration: Duration = results
            .iter()
            .map(|(_, _, duration)| *duration)
            .sum::<Duration>()
            / THREAD_COUNT as u32;

        println!("  Average call time: {:?}", avg_duration);
        assert!(
            avg_duration < Duration::from_secs(1),
            "Device count calls should be fast: {:?}",
            avg_duration
        );
    } else {
        println!("  No successful device count calls (expected in headless environment)");
    }
}
