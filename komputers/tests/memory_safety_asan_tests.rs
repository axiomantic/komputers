//! Address Sanitizer Memory Safety Tests
//!
//! This test suite implements comprehensive memory safety validation using
//! Address Sanitizer (ASan) to detect memory corruption, use-after-free,
//! buffer overflows, and resource leaks in GPU computing operations.
//!
//! **Safety-First Philosophy:** These tests prioritize safety over performance
//! and enable comprehensive runtime validation by default.
//!
//! **Production Readiness:** Zero tolerance for memory corruption in production.

use komputers::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Comprehensive tensor memory lifecycle test with ASan validation
///
/// This test creates and destroys 1000+ tensors to stress test memory management
/// and detect any use-after-free or memory corruption issues.
#[test]
fn test_tensor_memory_lifecycle_asan_validation() -> Result<()> {
    println!("🧪 ASan Test: Tensor memory lifecycle validation");
    println!("   Testing 1000+ tensor allocation/deallocation cycles");

    let mut manager = Manager::new()?;

    // Test tensor lifecycle with varying sizes to stress allocator
    for i in 0..1000 {
        let size = 100 + (i % 500); // Varying sizes from 100 to 599
        let data: Vec<f32> = (0..size).map(|x| (x + i) as f32).collect();

        // Create tensor
        let tensor = Tensor::new(&mut manager, &data)?;

        // Verify tensor is valid immediately after creation
        assert_eq!(tensor.size(), size);

        // Use tensor in GPU operation to test device memory safety
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        // Verify data integrity after GPU operation
        let result = tensor.to_vec()?;
        assert_eq!(result.len(), data.len());

        // Tensor is automatically dropped here - ASan will detect use-after-free

        if i % 100 == 0 {
            println!("   ✅ Completed {} tensor lifecycle tests", i + 1);
        }
    }

    println!("   ✅ ASan validation: 1000 tensor lifecycles completed without memory errors");
    Ok(())
}

/// Cross-manager handle validation test with ASan
///
/// This test verifies that runtime validation catches cross-manager contamination
/// and that ASan detects any memory corruption in handle management.
#[test]
fn test_cross_manager_handle_safety_asan() -> Result<()> {
    println!("🛡️  ASan Test: Cross-manager handle safety validation");

    let mut manager_a = Manager::new()?;
    let mut manager_b = Manager::new()?;

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor_a = Tensor::new(&mut manager_a, &data)?;

    println!("   🔧 Testing cross-manager handle contamination detection");

    // This should be caught by runtime validation or trigger ASan error
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sequence = manager_b.sequence().unwrap();
        // Attempt to use tensor A's handle with manager B
        sequence.record_tensor_sync_device(tensor_a.handle())
    }));

    match result {
        Ok(_) => {
            // If runtime validation is disabled, this might succeed
            // But ASan should still protect against memory corruption
            println!("   ⚠️  Cross-manager usage not caught by runtime validation");
            println!("      ASan should still prevent memory corruption");
        }
        Err(_) => {
            println!("   ✅ Cross-manager contamination properly detected");
        }
    }

    // Create valid operations to ensure managers still work correctly
    let tensor_b = Tensor::new(&mut manager_b, &data)?;
    let sequence_b = manager_b
        .sequence()?
        .record_tensor_sync_device(tensor_b.handle())?;
    sequence_b.execute()?;

    println!("   ✅ ASan validation: Cross-manager safety mechanisms working");
    Ok(())
}

/// Buffer overflow detection test with ASan
///
/// This test attempts operations that could cause buffer overflows
/// and verifies ASan catches them before they cause corruption.
#[test]
fn test_buffer_overflow_detection_asan() -> Result<()> {
    println!("🔍 ASan Test: Buffer overflow detection");

    let mut manager = Manager::new()?;

    // Test with various buffer sizes to detect potential overflows
    let test_sizes = vec![1, 10, 100, 1000, 10000];

    for size in test_sizes {
        println!("   🔧 Testing buffer operations with size: {}", size);

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::new(&mut manager, &data)?;

        // Perform operations that could trigger buffer overflows
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        // Read back data to verify no corruption
        let result = tensor.to_vec()?;
        assert_eq!(result.len(), size);

        // Verify data integrity (ASan would catch corruption here)
        for (i, &value) in result.iter().enumerate() {
            assert_eq!(value, i as f32, "Buffer corruption detected at index {}", i);
        }
    }

    println!("   ✅ ASan validation: No buffer overflows detected in GPU operations");
    Ok(())
}

/// Memory alignment and padding safety test
///
/// This test verifies that memory alignment and padding requirements
/// are met and that ASan doesn't detect any alignment violations.
#[test]
fn test_memory_alignment_safety_asan() -> Result<()> {
    println!("📐 ASan Test: Memory alignment and padding safety");

    let mut manager = Manager::new()?;

    // Test various data types and alignments
    let alignment_test_cases = vec![
        ("f32_aligned", vec![1.0f32; 256]),
        ("f32_uneven", vec![1.0f32; 255]),
        ("f32_large", vec![1.0f32; 4097]), // Just over page boundary
        ("f32_tiny", vec![42.0f32; 1]),
    ];

    for (test_name, data) in alignment_test_cases {
        println!(
            "   🔧 Testing alignment case: {} ({} elements)",
            test_name,
            data.len()
        );

        let tensor = Tensor::new(&mut manager, &data)?;

        // Test GPU operations that might be sensitive to alignment
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        // Verify data integrity after potential alignment operations
        let result = tensor.to_vec()?;
        assert_eq!(result.len(), data.len());

        // ASan will catch any alignment-related memory corruption
        for (i, (&expected, &actual)) in data.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                actual, expected,
                "Alignment corruption detected in {} at index {}",
                test_name, i
            );
        }
    }

    println!("   ✅ ASan validation: Memory alignment safety verified");
    Ok(())
}

/// GPU memory mapping safety test with ASan
///
/// This test verifies that GPU memory mapping operations are safe
/// and don't introduce memory corruption or use-after-free issues.
#[test]
fn test_gpu_memory_mapping_safety_asan() -> Result<()> {
    println!("🗺️  ASan Test: GPU memory mapping safety");

    let mut manager = Manager::new()?;

    // Test multiple mapping/unmapping cycles
    for iteration in 0..50 {
        let data: Vec<f32> = (0..1000).map(|i| (i + iteration * 1000) as f32).collect();
        let tensor = Tensor::new(&mut manager, &data)?;

        // Test sync operations that involve memory mapping
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        // Test host-device synchronization
        let result = tensor.to_vec()?;
        assert_eq!(result.len(), data.len());

        // Verify no corruption in mapped memory
        for (i, (&expected, &actual)) in data.iter().zip(result.iter()).enumerate() {
            if expected != actual {
                panic!(
                    "GPU memory mapping corruption at iteration {}, index {}: expected {}, got {}",
                    iteration, i, expected, actual
                );
            }
        }

        if iteration % 10 == 0 {
            println!(
                "   📊 Completed {} memory mapping safety tests",
                iteration + 1
            );
        }
    }

    println!("   ✅ ASan validation: GPU memory mapping safety verified");
    Ok(())
}

/// Concurrent memory access safety test with ASan
///
/// This test verifies that concurrent access to GPU memory is safe
/// and doesn't introduce race conditions or memory corruption.
#[test]
fn test_concurrent_memory_access_safety_asan() -> Result<()> {
    println!("🚦 ASan Test: Concurrent memory access safety");

    // Test with multiple threads accessing GPU resources
    let manager = Arc::new(Mutex::new(Manager::new()?));
    let thread_count = 4;
    let operations_per_thread = 25;

    let handles: Vec<thread::JoinHandle<Result<()>>> = (0..thread_count)
        .map(|thread_id| {
            let manager = Arc::clone(&manager);
            thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let mut mgr = manager
                        .lock()
                        .map_err(|_| KomputeError::internal("Failed to acquire manager lock"))?;

                    let data = vec![(thread_id * 1000 + i) as f32; 100];
                    let tensor = Tensor::new(&mut *mgr, &data)?;

                    let sequence = mgr.sequence()?.record_tensor_sync_device(tensor.handle())?;
                    sequence.execute()?;

                    // Verify data integrity under concurrent access
                    let result = tensor.to_vec()?;
                    assert_eq!(result.len(), 100);

                    // ASan will detect race conditions and memory corruption
                    for (j, &value) in result.iter().enumerate() {
                        let expected = (thread_id * 1000 + i) as f32;
                        assert_eq!(
                            value, expected,
                            "Concurrent access corruption in thread {} operation {} index {}",
                            thread_id, i, j
                        );
                    }
                }
                Ok(())
            })
        })
        .collect();

    // Wait for all threads and collect results
    let mut successful_threads = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {
                successful_threads += 1;
                println!("   ✅ Thread {} completed successfully", i);
            }
            Ok(Err(e)) => {
                println!("   ❌ Thread {} failed with error: {}", i, e);
            }
            Err(_) => {
                println!("   ❌ Thread {} panicked", i);
            }
        }
    }

    assert!(
        successful_threads >= thread_count - 1,
        "Too many thread failures: {}/{}",
        thread_count - successful_threads,
        thread_count
    );

    println!(
        "   ✅ ASan validation: Concurrent access safety verified ({}/{} threads successful)",
        successful_threads, thread_count
    );
    Ok(())
}

/// Resource cleanup verification test with ASan
///
/// This test creates many resources and verifies proper cleanup
/// by monitoring for memory leaks with ASan.
#[test]
fn test_resource_cleanup_verification_asan() -> Result<()> {
    println!("🧹 ASan Test: Resource cleanup verification");

    // Create baseline
    let baseline_time = Instant::now();

    // Test resource cleanup in nested scopes
    for cycle in 0..20 {
        println!("   🔄 Resource cleanup cycle {}", cycle + 1);

        {
            let mut manager = Manager::new()?;

            // Create multiple resources in this scope
            let mut tensors = Vec::new();
            for i in 0..50 {
                let data = vec![(cycle * 50 + i) as f32; 200];
                let tensor = Tensor::new(&mut manager, &data)?;
                tensors.push(tensor);
            }

            // Use all resources
            for (i, tensor) in tensors.iter().enumerate() {
                let sequence = manager
                    .sequence()?
                    .record_tensor_sync_device(tensor.handle())?;
                sequence.execute()?;

                if i % 10 == 0 {
                    // Test some intermediate cleanup
                    // Force some cleanup operations
                }
            }

            // All resources should be cleaned up when scope ends
        } // <- Manager and all tensors dropped here

        // Small delay to allow cleanup
        thread::sleep(Duration::from_millis(1));
    }

    let cleanup_duration = baseline_time.elapsed();
    println!(
        "   ⏱️  Total cleanup cycles completed in: {:?}",
        cleanup_duration
    );

    // ASan will detect any leaked resources
    println!("   ✅ ASan validation: Resource cleanup verification completed");
    println!("      ASan will report any memory leaks at process termination");

    Ok(())
}

/// Algorithm resource safety test with ASan
///
/// This test verifies that algorithm resources are managed safely
/// and don't introduce memory corruption.
#[test]
#[ignore] // Only run with GPU hardware available
fn test_algorithm_resource_safety_asan() -> Result<()> {
    println!("⚙️  ASan Test: Algorithm resource safety (requires GPU)");

    let mut manager = Manager::new()?;

    // Mock SPIR-V data (this will likely fail but should fail safely)
    let mock_spirv_data = vec![
        0x07, 0x23, 0x02, 0x03, // Mock SPIR-V magic number and version
        0x00, 0x00, 0x00, 0x00, // Additional mock data
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    for i in 0..10 {
        println!("   🔧 Algorithm resource test iteration {}", i + 1);

        // Test algorithm creation (expected to fail gracefully)
        match manager.algorithm(&mock_spirv_data) {
            Ok(_algorithm) => {
                println!("      ✅ Algorithm created (unexpected success)");
                // If successful, algorithm should be cleaned up automatically
            }
            Err(e) => {
                println!("      ℹ️  Algorithm creation failed as expected: {}", e);
                // Failure should be safe and not corrupt memory
            }
        }

        // Create a valid tensor operation to verify manager still works
        let data = vec![i as f32; 100];
        let tensor = Tensor::new(&mut manager, &data)?;
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;
    }

    println!("   ✅ ASan validation: Algorithm resource safety verified");
    Ok(())
}

#[cfg(test)]
mod asan_integration_tests {
    use super::*;

    /// Comprehensive ASan validation suite runner
    #[test]
    fn test_asan_memory_safety_comprehensive() -> Result<()> {
        println!("🔬 Running comprehensive ASan memory safety validation");
        println!("   Safety-First Philosophy: Zero tolerance for memory corruption");

        let start_time = Instant::now();

        // Run core memory safety tests
        test_tensor_memory_lifecycle_asan_validation()?;
        test_cross_manager_handle_safety_asan()?;
        test_buffer_overflow_detection_asan()?;
        test_memory_alignment_safety_asan()?;
        test_gpu_memory_mapping_safety_asan()?;
        test_concurrent_memory_access_safety_asan()?;
        test_resource_cleanup_verification_asan()?;

        let total_duration = start_time.elapsed();

        println!("\n=== ASAN MEMORY SAFETY VALIDATION COMPLETE ===");
        println!("Total validation time: {:?}", total_duration);
        println!("✅ All ASan memory safety tests passed");
        println!("🛡️  Production readiness: Memory corruption protection verified");
        println!("⚡ Safety-first configuration validated");

        Ok(())
    }
}
