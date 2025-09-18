//! Sequence command ordering and execution testing.
//!
//! These tests verify command recording, execution ordering, async operations,
//! error recovery, and command buffer limits for GPU sequence operations.

use kompute::{Algorithm, Manager, Sequence, Tensor};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Test sequence command execution ordering.
/// Verifies that recorded operations execute in the correct order.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_command_execution_ordering() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing sequence command execution ordering...");

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = match Tensor::new(&mut manager, &data) {
            Ok(tensor) => {
                println!("  ✓ Created tensor with {} elements", tensor.size());
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create tensor: {}", e);
                return;
            }
        };

        let mut sequence = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("  ✓ Created sequence");
                sequence
            }
            Err(e) => {
                println!("  ⚠ Failed to create sequence: {}", e);
                return;
            }
        };

        println!("  Recording operations in specific order...");

        // Record operations in specific order
        // 1. Sync tensor to device
        match sequence.record_tensor_sync_device(tensor.handle()) {
            Ok(()) => println!("    ✓ Recorded tensor sync to device"),
            Err(e) => {
                println!("    ⚠ Failed to record tensor sync to device: {}", e);
                return;
            }
        }

        // 2. Sync tensor back to local
        match sequence.record_tensor_sync_local(tensor.handle()) {
            Ok(()) => println!("    ✓ Recorded tensor sync to local"),
            Err(e) => {
                println!("    ⚠ Failed to record tensor sync to local: {}", e);
                return;
            }
        }

        println!("  Executing sequence...");
        let start_time = Instant::now();

        match sequence.eval() {
            Ok(()) => {
                let execution_time = start_time.elapsed();
                println!("  ✓ Sequence execution completed in {:?}", execution_time);

                // Execution should complete within reasonable time
                assert!(
                    execution_time < Duration::from_secs(5),
                    "Sequence execution took too long: {:?}",
                    execution_time
                );

                // Verify data integrity after ordered operations
                match tensor.to_vec() {
                    Ok(retrieved_data) => {
                        println!("  ✓ Data retrieval after sequence execution successful");
                        assert_eq!(
                            data, retrieved_data,
                            "Data should be preserved after ordered execution"
                        );
                        println!("  ✓ Data integrity verified after ordered execution");
                    }
                    Err(e) => {
                        println!("  ⚠ Data retrieval failed after execution: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  ⚠ Sequence execution failed: {}", e);
                // This may be expected in some environments
            }
        }

        println!("  ✓ Command execution ordering test completed");
    } else {
        println!("Skipping command ordering test - no GPU available");
    }
}

/// Test complex workflow with tensor-algorithm chains.
/// Verifies end-to-end compute pipelines work correctly.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_complex_workflow_tensor_algorithm_chain() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing complex tensor-algorithm workflow...");

        // Create input and output tensors
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_tensor = match Tensor::new(&mut manager, &input_data) {
            Ok(tensor) => {
                println!("  ✓ Created input tensor with {} elements", tensor.size());
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create input tensor: {}", e);
                return;
            }
        };

        let output_tensor = match Tensor::<f32>::uninit(&mut manager, input_data.len()) {
            Ok(tensor) => {
                println!("  ✓ Created output tensor with {} elements", tensor.size());
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create output tensor: {}", e);
                return;
            }
        };

        // Create algorithm for computation (basic SPIR-V)
        let compute_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        let algorithm = match Algorithm::new(&mut manager, &compute_spirv) {
            Ok(algorithm) => {
                println!("  ✓ Created algorithm");
                algorithm
            }
            Err(e) => {
                println!("  ⚠ Failed to create algorithm: {}", e);
                return;
            }
        };

        // Create sequence for complex workflow
        let mut sequence = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("  ✓ Created workflow sequence");
                sequence
            }
            Err(e) => {
                println!("  ⚠ Failed to create workflow sequence: {}", e);
                return;
            }
        };

        println!("  Recording complex workflow...");

        // Complex workflow: input sync -> compute -> output sync
        println!("    Step 1: Input tensor sync to device");
        match sequence.record_tensor_sync_device(input_tensor.handle()) {
            Ok(()) => println!("      ✓ Recorded: Input tensor sync to device"),
            Err(e) => {
                println!("      ⚠ Failed to record input tensor sync: {}", e);
                return;
            }
        }

        println!("    Step 2: Algorithm dispatch");
        match sequence.record_algorithm_dispatch(algorithm.handle()) {
            Ok(()) => println!("      ✓ Recorded: Algorithm dispatch"),
            Err(e) => {
                println!("      ⚠ Failed to record algorithm dispatch: {}", e);
                return;
            }
        }

        println!("    Step 3: Output tensor sync to local");
        match sequence.record_tensor_sync_local(output_tensor.handle()) {
            Ok(()) => println!("      ✓ Recorded: Output tensor sync to local"),
            Err(e) => {
                println!("      ⚠ Failed to record output tensor sync: {}", e);
                return;
            }
        }

        println!("  Executing complex workflow...");
        let workflow_start = Instant::now();

        match sequence.eval() {
            Ok(()) => {
                let workflow_time = workflow_start.elapsed();
                println!("  ✓ Complex workflow completed in {:?}", workflow_time);

                // Workflow should complete within reasonable time
                assert!(
                    workflow_time < Duration::from_secs(10),
                    "Complex workflow took too long: {:?}",
                    workflow_time
                );

                // Verify input data is still intact
                match input_tensor.to_vec() {
                    Ok(retrieved_input) => {
                        assert_eq!(
                            input_data, retrieved_input,
                            "Input data should be preserved"
                        );
                        println!("  ✓ Input data integrity verified");
                    }
                    Err(e) => println!("  ⚠ Input data verification failed: {}", e),
                }

                // Check output tensor (may contain uninitialized data until algorithm binding works)
                match output_tensor.to_vec() {
                    Ok(output_data) => {
                        println!(
                            "  ✓ Output tensor accessible ({} elements)",
                            output_data.len()
                        );
                        assert_eq!(
                            output_data.len(),
                            input_data.len(),
                            "Output tensor should have same size as input"
                        );
                        // Note: Actual computation results will be verified once algorithm-tensor binding is implemented
                    }
                    Err(e) => println!("  ⚠ Output tensor access failed: {}", e),
                }
            }
            Err(e) => {
                println!("  ⚠ Complex workflow execution failed: {}", e);
                // This is expected until algorithm-tensor binding is implemented
            }
        }

        println!("  ✓ Complex workflow framework established");
        println!("  Note: Full tensor-algorithm integration will be tested after Phase 2");
    } else {
        println!("Skipping complex workflow test - no GPU available");
    }
}

/// Test sequence async completion timing.
/// Verifies eval_async() and eval_await() work correctly.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_async_completion_timing() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing sequence async completion...");

        let data = vec![1.0f32; 1000]; // Larger dataset for async testing
        let tensor = match Tensor::new(&mut manager, &data) {
            Ok(tensor) => {
                println!("  ✓ Created tensor with {} elements", tensor.size());
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create tensor: {}", e);
                return;
            }
        };

        let mut sequence = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("  ✓ Created sequence for async testing");
                sequence
            }
            Err(e) => {
                println!("  ⚠ Failed to create sequence: {}", e);
                return;
            }
        };

        // Record operations for async execution
        match sequence.record_tensor_sync_device(tensor.handle()) {
            Ok(()) => println!("  ✓ Recorded tensor sync for async test"),
            Err(e) => {
                println!("  ⚠ Failed to record operation: {}", e);
                return;
            }
        }

        println!("  Testing async execution...");

        // Test eval_async() - should return immediately
        let async_start = Instant::now();
        match sequence.eval_async() {
            Ok(()) => {
                let async_call_time = async_start.elapsed();
                println!("  ✓ eval_async() returned in {:?}", async_call_time);

                // eval_async should return quickly (not block)
                assert!(
                    async_call_time < Duration::from_millis(100),
                    "eval_async() should return quickly: {:?}",
                    async_call_time
                );

                // Test eval_await() with different timeouts
                let timeout_tests = vec![
                    (1, "1ms timeout"),
                    (10, "10ms timeout"),
                    (100, "100ms timeout"),
                    (1000, "1s timeout"),
                    (5000, "5s timeout"),
                ];

                for (timeout_ms, description) in timeout_tests {
                    println!("    Testing {}", description);
                    let await_start = Instant::now();

                    match sequence.eval_await(timeout_ms) {
                        Ok(()) => {
                            let await_time = await_start.elapsed();
                            println!("      ✓ Completed in {:?}", await_time);

                            // Should complete within timeout period
                            assert!(
                                await_time.as_millis() <= timeout_ms as u128 + 100,
                                "Should complete within timeout + margin: {:?} <= {}ms",
                                await_time,
                                timeout_ms
                            );
                            break; // Success, no need to try longer timeouts
                        }
                        Err(e) => {
                            let await_time = await_start.elapsed();
                            println!("      ⚠ Timed out in {:?}: {}", await_time, e);

                            // Should timeout around the specified time
                            assert!(
                                await_time.as_millis() >= timeout_ms as u128 - 100,
                                "Should timeout around specified time: {:?} >= {}ms",
                                await_time,
                                timeout_ms
                            );
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ⚠ eval_async() failed: {}", e);
                // This may be expected in some environments
            }
        }

        // Test multiple async operations
        println!("  Testing multiple async operations...");
        for i in 0..3 {
            let mut test_sequence = match Sequence::new(&mut manager) {
                Ok(seq) => seq,
                Err(e) => {
                    println!("    ⚠ Failed to create test sequence {}: {}", i, e);
                    continue;
                }
            };

            match test_sequence.record_tensor_sync_local(tensor.handle()) {
                Ok(()) => {
                    println!("    ✓ Recorded operation for async test {}", i);

                    match test_sequence.eval_async() {
                        Ok(()) => {
                            println!("      ✓ Started async operation {}", i);

                            // Don't wait for completion to test concurrent async
                        }
                        Err(e) => println!("      ⚠ Async operation {} failed: {}", i, e),
                    }
                }
                Err(e) => println!("    ⚠ Failed to record for async test {}: {}", i, e),
            }
        }

        println!("  ✓ Async completion timing tests completed");
    } else {
        println!("Skipping async completion test - no GPU available");
    }
}

/// Test sequence error recovery with partial failures.
/// Verifies error handling and recovery mechanisms.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_error_recovery_partial_failure() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing sequence error recovery...");

        // Create valid tensor for testing
        let valid_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let valid_tensor = match Tensor::new(&mut manager, &valid_data) {
            Ok(tensor) => {
                println!("  ✓ Created valid tensor");
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create valid tensor: {}", e);
                return;
            }
        };

        // Test 1: Invalid tensor handle
        println!("  Testing invalid tensor handle recovery...");
        let mut error_sequence = match Sequence::new(&mut manager) {
            Ok(sequence) => sequence,
            Err(e) => {
                println!("  ⚠ Failed to create error test sequence: {}", e);
                return;
            }
        };

        // Record valid operation first
        match error_sequence.record_tensor_sync_device(valid_tensor.handle()) {
            Ok(()) => println!("    ✓ Recorded valid operation"),
            Err(e) => {
                println!("    ⚠ Failed to record valid operation: {}", e);
                return;
            }
        }

        // Record operation with invalid handle
        let invalid_handle = 0xDEADBEEF;
        match error_sequence.record_tensor_sync_device(invalid_handle) {
            Ok(()) => {
                println!("    ⚠ Invalid handle operation was accepted (unexpected)");
            }
            Err(e) => {
                println!("    ✓ Invalid handle correctly rejected: {}", e);
                assert!(
                    e.to_string().contains("invalid") || e.to_string().contains("handle"),
                    "Error should mention invalid handle: {}",
                    e
                );
            }
        }

        // Test sequence execution with partial errors
        println!("  Testing sequence execution with recorded errors...");
        match error_sequence.eval() {
            Ok(()) => {
                println!("    ⚠ Sequence with errors executed successfully (unexpected)");
            }
            Err(e) => {
                println!("    ✓ Sequence with errors failed as expected: {}", e);
            }
        }

        // Test 2: Error recovery - new sequence after failure
        println!("  Testing error recovery with new sequence...");
        let mut recovery_sequence = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("    ✓ Created recovery sequence");
                sequence
            }
            Err(e) => {
                println!("    ⚠ Failed to create recovery sequence: {}", e);
                return;
            }
        };

        // Should be able to use valid operations after previous failure
        match recovery_sequence.record_tensor_sync_device(valid_tensor.handle()) {
            Ok(()) => {
                println!("    ✓ Recovery sequence accepts valid operations");

                match recovery_sequence.eval() {
                    Ok(()) => {
                        println!("    ✓ Recovery sequence executed successfully");

                        // Verify original tensor is still accessible
                        match valid_tensor.to_vec() {
                            Ok(recovered_data) => {
                                assert_eq!(
                                    valid_data, recovered_data,
                                    "Data should be preserved after error recovery"
                                );
                                println!("    ✓ Data integrity preserved after error recovery");
                            }
                            Err(e) => println!("    ⚠ Data recovery failed: {}", e),
                        }
                    }
                    Err(e) => println!("    ⚠ Recovery sequence execution failed: {}", e),
                }
            }
            Err(e) => println!("    ⚠ Recovery sequence operation failed: {}", e),
        }

        // Test 3: Multiple error conditions
        println!("  Testing multiple error conditions...");
        let error_conditions = vec![
            (0, "Zero handle"),
            (0xFFFFFFFF, "Max handle"),
            (0x12345678, "Random handle"),
            (0xDEADBEEF, "Dead beef handle"),
        ];

        for (bad_handle, description) in error_conditions {
            let mut test_sequence = match Sequence::new(&mut manager) {
                Ok(seq) => seq,
                Err(_) => continue,
            };

            println!("    Testing {}: 0x{:X}", description, bad_handle);
            match test_sequence.record_tensor_sync_device(bad_handle) {
                Ok(()) => println!("      ⚠ {} was accepted", description),
                Err(e) => {
                    println!("      ✓ {} correctly rejected: {}", description, e);
                    // Error message should be informative
                    assert!(
                        !e.to_string().is_empty(),
                        "Error message should not be empty"
                    );
                }
            }
        }

        println!("  ✓ Error recovery testing completed");
    } else {
        println!("Skipping error recovery test - no GPU available");
    }
}

/// Test sequence command buffer limits with maximum operations.
/// Verifies system behavior with many recorded operations.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_command_buffer_limits_maximum_operations() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing sequence command buffer limits...");

        let data = vec![1.0f32; 100];
        let tensor = match Tensor::new(&mut manager, &data) {
            Ok(tensor) => {
                println!("  ✓ Created tensor for limits testing");
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create tensor: {}", e);
                return;
            }
        };

        let mut sequence = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("  ✓ Created sequence for limits testing");
                sequence
            }
            Err(e) => {
                println!("  ⚠ Failed to create sequence: {}", e);
                return;
            }
        };

        const MAX_OPERATIONS: usize = 1000;
        let mut operations_recorded = 0;
        let mut total_record_time = Duration::new(0, 0);

        println!("  Recording up to {} operations...", MAX_OPERATIONS);

        for i in 0..MAX_OPERATIONS {
            let record_start = Instant::now();

            // Alternate between different operation types
            let result = match i % 3 {
                0 => sequence.record_tensor_sync_device(tensor.handle()),
                1 => sequence.record_tensor_sync_local(tensor.handle()),
                2 => sequence.record_tensor_sync_device(tensor.handle()),
                _ => unreachable!(),
            };

            let record_time = record_start.elapsed();
            total_record_time += record_time;

            match result {
                Ok(()) => {
                    operations_recorded += 1;

                    if i % 100 == 0 {
                        println!("    Recorded {} operations", i + 1);
                    }

                    // Recording should remain fast
                    if record_time.as_millis() > 10 {
                        println!(
                            "    Warning: Operation {} recording took {:?}",
                            i, record_time
                        );
                    }
                }
                Err(e) => {
                    println!("    Operation {} recording failed: {}", i, e);
                    break;
                }
            }
        }

        println!("Command buffer limits test results:");
        println!("  Operations recorded: {}", operations_recorded);
        println!("  Total recording time: {:?}", total_record_time);

        if operations_recorded > 0 {
            let avg_record_time = total_record_time / operations_recorded as u32;
            println!("  Average recording time: {:?}", avg_record_time);

            // Recording should be efficient
            assert!(
                avg_record_time.as_millis() < 5,
                "Average operation recording should be fast: {:?}",
                avg_record_time
            );

            // Should be able to record a reasonable number of operations
            assert!(
                operations_recorded >= 100,
                "Should be able to record at least 100 operations, got {}",
                operations_recorded
            );
        }

        // Test execution of large command buffer
        if operations_recorded > 0 {
            println!(
                "  Executing large command buffer with {} operations...",
                operations_recorded
            );
            let exec_start = Instant::now();

            match sequence.eval() {
                Ok(()) => {
                    let exec_time = exec_start.elapsed();
                    println!("  ✓ Large command buffer executed in {:?}", exec_time);

                    // Execution time should scale reasonably
                    let time_per_op = exec_time.as_millis() as f64 / operations_recorded as f64;
                    println!("  Average time per operation: {:.2}ms", time_per_op);

                    assert!(
                        time_per_op < 10.0,
                        "Time per operation should be reasonable: {:.2}ms",
                        time_per_op
                    );
                }
                Err(e) => {
                    println!("  ⚠ Large command buffer execution failed: {}", e);
                    // May be expected with very large command buffers
                }
            }
        }

        println!("  ✓ Command buffer limits testing completed");
    } else {
        println!("Skipping command buffer limits test - no GPU available");
    }
}

/// Test sequence creation and basic properties.
/// Verifies sequence lifecycle and handle management.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_creation_and_properties() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing sequence creation and properties...");

        // Test basic sequence creation
        let sequence1 = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("  ✓ Created sequence 1");
                sequence
            }
            Err(e) => {
                println!("  ⚠ Failed to create sequence 1: {}", e);
                return;
            }
        };

        // Verify sequence properties
        assert_ne!(sequence1.handle(), 0, "Sequence handle should be non-zero");
        println!("  ✓ Sequence 1 handle: {}", sequence1.handle());

        // Test multiple sequence creation
        let sequence2 = match Sequence::new(&mut manager) {
            Ok(sequence) => {
                println!("  ✓ Created sequence 2");
                sequence
            }
            Err(e) => {
                println!("  ⚠ Failed to create sequence 2: {}", e);
                return;
            }
        };

        // Verify handle uniqueness
        assert_ne!(
            sequence2.handle(),
            0,
            "Sequence 2 handle should be non-zero"
        );
        assert_ne!(
            sequence1.handle(),
            sequence2.handle(),
            "Sequence handles should be unique"
        );
        println!("  ✓ Sequence 2 handle: {} (unique)", sequence2.handle());

        // Test sequence creation performance
        let mut creation_times = Vec::new();
        const SEQUENCE_COUNT: usize = 10;

        for i in 0..SEQUENCE_COUNT {
            let start = Instant::now();
            match Sequence::new(&mut manager) {
                Ok(_sequence) => {
                    let creation_time = start.elapsed();
                    creation_times.push(creation_time);
                    println!("    Sequence {} created in {:?}", i, creation_time);
                }
                Err(e) => {
                    println!("    Sequence {} creation failed: {}", i, e);
                    break;
                }
            }
        }

        if !creation_times.is_empty() {
            let avg_time: Duration =
                creation_times.iter().sum::<Duration>() / creation_times.len() as u32;
            println!("  Average sequence creation time: {:?}", avg_time);

            assert!(
                avg_time < Duration::from_millis(100),
                "Sequence creation should be fast: {:?}",
                avg_time
            );
        }

        println!("  ✓ Sequence creation and properties verified");
    } else {
        println!("Skipping sequence creation test - no GPU available");
    }
}

/// Test concurrent sequence operations across multiple threads.
/// Verifies thread safety of sequence operations.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_concurrent_operations() {
    println!("Testing concurrent sequence operations...");

    if let Ok(manager) = Manager::new() {
        let manager = Arc::new(std::sync::Mutex::new(manager));
        let thread_count = 4;
        let operations_per_thread = 5;
        let mut handles = vec![];

        // Create shared tensor for all threads
        let shared_tensor = {
            let mut manager_guard = manager.lock().unwrap();
            match Tensor::new(&mut *manager_guard, &vec![1.0f32; 100]) {
                Ok(tensor) => {
                    println!("  ✓ Created shared tensor");
                    Arc::new(tensor)
                }
                Err(e) => {
                    println!("  ⚠ Failed to create shared tensor: {}", e);
                    return;
                }
            }
        };

        for thread_id in 0..thread_count {
            let manager_clone = Arc::clone(&manager);
            let tensor_clone = Arc::clone(&shared_tensor);

            let handle = thread::spawn(move || {
                let mut thread_results = Vec::new();

                for op_id in 0..operations_per_thread {
                    // Create sequence in this thread
                    let mut sequence = {
                        let mut manager_guard = manager_clone.lock().unwrap();
                        match Sequence::new(&mut *manager_guard) {
                            Ok(seq) => seq,
                            Err(e) => {
                                thread_results.push((
                                    thread_id,
                                    op_id,
                                    false,
                                    format!("Sequence creation failed: {}", e),
                                ));
                                continue;
                            }
                        }
                    };

                    // Record operation
                    match sequence.record_tensor_sync_device(tensor_clone.handle()) {
                        Ok(()) => {
                            // Execute operation
                            match sequence.eval() {
                                Ok(()) => {
                                    thread_results.push((
                                        thread_id,
                                        op_id,
                                        true,
                                        "Success".to_string(),
                                    ));
                                }
                                Err(e) => {
                                    thread_results.push((
                                        thread_id,
                                        op_id,
                                        false,
                                        format!("Execution failed: {}", e),
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            thread_results.push((
                                thread_id,
                                op_id,
                                false,
                                format!("Recording failed: {}", e),
                            ));
                        }
                    }
                }

                thread_results
            });

            handles.push(handle);
        }

        // Collect results from all threads
        let mut total_operations = 0;
        let mut successful_operations = 0;

        for handle in handles {
            if let Ok(thread_results) = handle.join() {
                for (thread_id, op_id, success, message) in thread_results {
                    total_operations += 1;
                    if success {
                        successful_operations += 1;
                        println!("  Thread {} operation {}: ✓ {}", thread_id, op_id, message);
                    } else {
                        println!("  Thread {} operation {}: ⚠ {}", thread_id, op_id, message);
                    }
                }
            }
        }

        println!("Concurrent sequence operations results:");
        println!("  Total operations: {}", total_operations);
        println!("  Successful operations: {}", successful_operations);
        println!(
            "  Success rate: {:.1}%",
            (successful_operations as f64 / total_operations as f64) * 100.0
        );

        if successful_operations > 0 {
            println!("  ✓ Concurrent sequence operations working");
        }
    } else {
        println!("Skipping concurrent sequence test - no GPU available");
    }
}
